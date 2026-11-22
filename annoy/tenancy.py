# Copyright (c) 2026 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-tenant Annoy index manager with a shared, configurable memory budget.

One process serves many tenants, each with its own on-disk Annoy index.
Instead of keeping every index resident forever, this manager mounts an
index on first use, keeps the hot set in memory under a byte budget, and
unloads least-recently-used tenants back to disk. A later query remounts
transparently.

Design notes / guarantees:

* Same source for memory accounting: ``estimate_resident_bytes(path)`` is
  the *only* function that turns an index file into a byte count. Budget
  accounting, eviction decisions and the ``report()`` output all use it,
  so the reported footprint and the accounted footprint can never drift
  apart. Annoy loads an index by mapping the file contents, so the file
  size (plus a fixed object overhead) is the honest resident estimate.
* Query safety: every query holds a pin (reference count) on the mounted
  index. Eviction only considers entries with refcount == 0, so a query
  never reads freed memory. Loading happens outside every lock, so a slow
  mount of one tenant never blocks queries on other tenants.
* Concurrency: there is no global lock on the query path. Each tenant
  entry has its own lock, so queries for different tenants (and, to a
  large extent, the same tenant) do not serialise on manager bookkeeping.
  ``_used`` read-modify-writes are guarded by ``_usage_lock``, which is
  only ever taken *after* an entry lock or with no lock held, never the
  other way around — so the two cannot deadlock.
* No writes: the manager never creates, modifies or deletes index files.
  Unloading is a pure in-memory operation (munmap), so a process exit at
  any point cannot leave a corrupt temporary file behind.
* The capability is opt-in: nothing here changes ``AnnoyIndex`` itself.
  Callers that do not use this module see byte-for-byte identical
  behaviour from the existing build/save/load/query interfaces.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

from .annoylib import Annoy as _AnnoyIndex

# Fixed per-mounted-index overhead: the AnnoyIndex Python/C++ object,
# roots vector and bookkeeping, on top of the mapped file bytes.
INDEX_OBJECT_OVERHEAD_BYTES = 64 * 1024


def estimate_resident_bytes(path):
    """Single source of truth for how much memory a mounted index costs.

    Used for budget accounting, eviction decisions and reporting alike.
    """
    return os.path.getsize(path) + INDEX_OBJECT_OVERHEAD_BYTES


def _file_sig(path):
    """Signature used to detect that an index file was replaced on disk."""
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


class TenantError(RuntimeError):
    """Base class for all tenant manager errors."""


class UnknownTenantError(TenantError):
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        super().__init__("unknown tenant: %r" % (tenant_id,))


class TenantBudgetError(TenantError):
    """A single tenant's index can never fit in the configured budget."""

    def __init__(self, tenant_id, required_bytes, available_bytes):
        self.tenant_id = tenant_id
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        super().__init__(
            "tenant %r needs %d bytes resident but the total budget is only "
            "%d bytes; raise the budget or shrink the index"
            % (tenant_id, required_bytes, available_bytes))


class TenantMountError(TenantError):
    """The index file for one tenant could not be mounted.

    Other tenants are unaffected. Never swallowed: a failed (re)mount
    raises instead of returning an empty result that would look like
    "no similar items".
    """

    def __init__(self, tenant_id, reason):
        self.tenant_id = tenant_id
        self.reason = reason
        super().__init__("cannot mount index for tenant %r: %s"
                         % (tenant_id, reason))


class TenantBusyError(TenantError):
    """Operation requires the tenant to be unpinned, but queries hold it."""


_UNLOADED = "unloaded"
_LOADING = "loading"
_MOUNTED = "mounted"


class _Entry(object):
    __slots__ = (
        "tenant_id", "path", "dim", "metric",
        "lock", "cond", "waiters",
        "state", "index", "refcount", "file_sig", "est_bytes",
        "n_items", "file_size", "retired", "last_stat_at",
        "hits", "misses", "loads", "failed_loads", "evictions",
        "queries", "total_query_s",
        "registered_at", "last_access", "last_unload_at", "last_error",
    )

    def __init__(self, tenant_id, path, dim, metric):
        self.tenant_id = tenant_id
        self.path = path
        self.dim = dim
        self.metric = metric
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.waiters = 0
        self.state = _UNLOADED
        self.index = None
        self.refcount = 0
        self.file_sig = None
        self.est_bytes = 0
        self.n_items = None
        self.file_size = None
        self.retired = False
        self.last_stat_at = None
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.failed_loads = 0
        self.evictions = 0
        self.queries = 0
        self.total_query_s = 0.0
        self.registered_at = time.time()
        self.last_access = None
        self.last_unload_at = None
        self.last_error = None


class TenantIndexManager(object):
    """Mounts per-tenant Annoy indexes under a shared memory budget.

    Parameters
    ----------
    metric:
        Default distance metric for tenants that do not override it.
    budget_bytes:
        Total resident budget across all mounted tenants. ``None`` means
        unlimited (nothing is ever evicted for budget reasons). Can be
        changed at any time via :meth:`set_budget` — no restart needed.
    idle_timeout_s:
        If set, mounted tenants with no access for this many seconds are
        unloaded automatically (their last access time is recorded first).
        A daemon reaper thread enforces this; :meth:`reap_idle` can also
        be called directly.
    reaper_interval_s:
        How often the reaper thread wakes up. Defaults to half the idle
        timeout.
    staleness_check_interval_s:
        ``None`` (default) means every query stats the index file, so a
        replaced file is always remounted before the very next query.
        Setting e.g. 0.1 lets the manager trust the mounted copy for up
        to that many seconds between stats, trading the strictness of
        that guarantee for query throughput.
    """

    def __init__(self, metric="angular", budget_bytes=None,
                 idle_timeout_s=None, reaper_interval_s=None,
                 staleness_check_interval_s=None):
        self._metric = metric
        self._budget = budget_bytes
        self._idle_timeout = idle_timeout_s
        self._staleness_interval = staleness_check_interval_s
        self._tenants = {}
        self._tenants_lock = threading.Lock()  # guards register/unregister
        self._usage_lock = threading.Lock()    # guards _used read-modify-write
        self._used = 0
        self._closed = False
        self._stop = threading.Event()
        self._reaper = None
        if idle_timeout_s is not None:
            interval = reaper_interval_s
            if interval is None:
                interval = max(idle_timeout_s / 2.0, 0.05)
            self._reaper = threading.Thread(
                target=self._reap_loop, args=(interval,),
                name="tenant-index-reaper", daemon=True)
            self._reaper.start()

    # ------------------------------------------------------------------
    # tenant administration
    # ------------------------------------------------------------------

    def register(self, tenant_id, path, dim, metric=None):
        """Register a tenant's index file. Does not mount anything."""
        with self._tenants_lock:
            self._check_open()
            if tenant_id in self._tenants:
                raise TenantError("tenant %r is already registered"
                                  % (tenant_id,))
            self._tenants[tenant_id] = _Entry(
                tenant_id, path, dim, metric or self._metric)

    def unregister(self, tenant_id):
        """Remove a tenant, unloading its index first if it is mounted."""
        with self._tenants_lock:
            e = self._tenants.get(tenant_id)
            if e is None:
                raise UnknownTenantError(tenant_id)
            with e.lock:
                if e.refcount > 0 or e.state == _LOADING:
                    raise TenantBusyError(
                        "tenant %r has queries in flight" % (tenant_id,))
                e.retired = True
                if e.state == _MOUNTED:
                    self._unload_entry_locked(e)
            del self._tenants[tenant_id]

    def tenant_ids(self):
        with self._tenants_lock:
            return sorted(self._tenants)

    # ------------------------------------------------------------------
    # query path
    # ------------------------------------------------------------------

    @contextmanager
    def lease(self, tenant_id):
        """Pin the tenant's mounted index and yield the raw AnnoyIndex.

        While the lease is held the index cannot be unloaded, so it is
        safe to run any number of queries against it. This is the
        reference-counting mechanism that keeps queries and unloads from
        racing.
        """
        idx, e = self._acquire(tenant_id)
        try:
            yield idx
        finally:
            self._release(e, None)

    def query(self, tenant_id, vector, n, search_k=-1,
              include_distances=False):
        """Mount (if needed) and run get_nns_by_vector for one tenant."""
        t0 = time.perf_counter()
        idx, e = self._acquire(tenant_id)
        try:
            out = idx.get_nns_by_vector(
                vector, n, search_k=search_k,
                include_distances=include_distances)
        finally:
            self._release(e, time.perf_counter() - t0)
        return out

    def query_by_item(self, tenant_id, item, n, search_k=-1,
                      include_distances=False):
        """Mount (if needed) and run get_nns_by_item for one tenant."""
        t0 = time.perf_counter()
        idx, e = self._acquire(tenant_id)
        try:
            out = idx.get_nns_by_item(
                item, n, search_k=search_k,
                include_distances=include_distances)
        finally:
            self._release(e, time.perf_counter() - t0)
        return out

    # ------------------------------------------------------------------
    # budget / lifecycle management
    # ------------------------------------------------------------------

    def set_budget(self, budget_bytes):
        """Change the total memory budget online, without a restart.

        Shrinking the budget immediately evicts least-recently-used
        tenants (never ones with queries in flight) until the mounted
        set fits again.
        """
        if budget_bytes is not None and budget_bytes < 0:
            raise ValueError("budget must be non-negative or None")
        self._budget = budget_bytes
        self._enforce_budget()

    def get_budget(self):
        return self._budget

    def used_bytes(self):
        with self._usage_lock:
            return self._used

    def evict(self, tenant_id):
        """Unload one tenant now. Returns True if it was mounted."""
        e = self._tenants.get(tenant_id)
        if e is None:
            raise UnknownTenantError(tenant_id)
        with e.lock:
            if e.state != _MOUNTED:
                return False
            if e.refcount > 0:
                raise TenantBusyError(
                    "tenant %r has queries in flight" % (tenant_id,))
            self._unload_entry_locked(e)
            return True

    def reap_idle(self, now=None):
        """Unload mounted tenants idle longer than the configured timeout.

        Records each tenant's last access time before releasing it.
        Returns the list of evicted tenant ids.
        """
        if now is None:
            now = time.time()
        evicted = []
        timeout = self._idle_timeout
        if timeout is None:
            return evicted
        for e in self._tenant_snapshot():
            with e.lock:
                if (e.state == _MOUNTED and e.refcount == 0
                        and e.last_access is not None
                        and now - e.last_access >= timeout):
                    self._unload_entry_locked(e)
                    evicted.append(e.tenant_id)
        return evicted

    def close(self):
        """Stop the reaper thread and unload every tenant."""
        self._stop.set()
        reaper = self._reaper
        if reaper is not None and reaper.is_alive():
            reaper.join(timeout=5.0)
        self._closed = True
        for e in self._tenant_snapshot():
            with e.lock:
                if e.state == _MOUNTED and e.refcount == 0:
                    self._unload_entry_locked(e)
                e.cond.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def report(self):
        """Snapshot of budget usage and per-tenant statistics."""
        tenants = {}
        mounted = 0
        for e in self._tenant_snapshot():
            with e.lock:
                accesses = e.hits + e.misses
                if e.state == _MOUNTED:
                    mounted += 1
                tenants[e.tenant_id] = {
                    "path": e.path,
                    "metric": e.metric,
                    "dim": e.dim,
                    "state": e.state,
                    "mounted": e.state == _MOUNTED,
                    "n_items": e.n_items,
                    "file_size": e.file_size,
                    # Current accounted usage; 0 when not mounted. Comes
                    # from the same estimate_resident_bytes() source as
                    # the budget accounting.
                    "resident_bytes": (e.est_bytes
                                       if e.state == _MOUNTED else 0),
                    "refcount": e.refcount,
                    "hits": e.hits,
                    "misses": e.misses,
                    "hit_rate": (float(e.hits) / accesses
                                 ) if accesses else None,
                    "loads": e.loads,
                    "remounts": max(0, e.loads - 1),
                    "failed_loads": e.failed_loads,
                    "evictions": e.evictions,
                    "queries": e.queries,
                    "avg_query_ms": (1000.0 * e.total_query_s / e.queries
                                     if e.queries else None),
                    "registered_at": e.registered_at,
                    "last_access": e.last_access,
                    "last_unload_at": e.last_unload_at,
                    "last_error": e.last_error,
                }
        return {
            "budget_bytes": self._budget,
            "used_bytes": self.used_bytes(),
            "mounted_tenants": mounted,
            "total_tenants": len(tenants),
            "tenants": tenants,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _check_open(self):
        if self._closed:
            raise TenantError("manager is closed")

    def _tenant_snapshot(self):
        with self._tenants_lock:
            return list(self._tenants.values())

    def _reap_loop(self, interval):
        while not self._stop.wait(interval):
            try:
                self.reap_idle()
            except Exception:
                # The reaper must never take the process down.
                pass

    def _add_used(self, delta):
        with self._usage_lock:
            self._used += delta

    def _staleness_probe(self, e):
        """Signature of the on-disk file, or None if it cannot be stat'ed.

        With the default strict configuration this stats the file on
        every call; with staleness_check_interval_s set it returns the
        cached signature inside the freshness window.
        """
        interval = self._staleness_interval
        if interval is not None and e.last_stat_at is not None:
            if time.monotonic() - e.last_stat_at < interval:
                return e.file_sig
        try:
            sig = _file_sig(e.path)
        except OSError:
            sig = None
        e.last_stat_at = time.monotonic()
        return sig

    def _acquire(self, tenant_id):
        """Pin the tenant's index, mounting/remounting as needed.

        Returns (index, entry). The caller must later call _release.
        """
        e = self._tenants.get(tenant_id)  # dict read is atomic under the GIL
        if e is None:
            raise UnknownTenantError(tenant_id)
        while True:
            # Staleness probe outside every lock: a replaced (or deleted)
            # file is noticed before the query runs, never after.
            sig = self._staleness_probe(e)
            with e.lock:
                if e.retired:
                    raise UnknownTenantError(tenant_id)
                if e.state == _MOUNTED and sig is not None \
                        and e.file_sig == sig:
                    # Fast path: mounted and the file is unchanged.
                    e.refcount += 1
                    e.hits += 1
                    return e.index, e
                if e.state == _LOADING:
                    # Another thread is already mounting this tenant;
                    # wait so the file is loaded exactly once.
                    e.waiters += 1
                    try:
                        e.cond.wait()
                    finally:
                        e.waiters -= 1
                    continue
                if e.state == _MOUNTED:
                    # The file was replaced (or vanished): the next query
                    # must see the new mapping, never the old one.
                    if e.refcount > 0:
                        # In-flight queries finish on the old mapping
                        # (they hold their own pin, so this is safe);
                        # wait for them to drain before swapping.
                        e.waiters += 1
                        try:
                            e.cond.wait()
                        finally:
                            e.waiters -= 1
                        continue
                    self._unload_entry_locked(e)
                    continue
                # _UNLOADED: become the loader.
                e.state = _LOADING
                e.misses += 1
                budget = self._budget
            # Load outside every lock.
            try:
                idx, new_sig, est = self._load(e, budget)
            except TenantError as exc:
                with e.lock:
                    e.state = _UNLOADED
                    e.failed_loads += 1
                    e.last_error = str(exc)
                    e.cond.notify_all()
                raise
            with e.lock:
                e.index = idx
                e.file_sig = new_sig
                e.file_size = new_sig[1]
                e.est_bytes = est
                e.n_items = idx.get_n_items()
                e.state = _MOUNTED
                e.loads += 1
                e.refcount = 1
                e.last_access = time.time()
                e.last_stat_at = time.monotonic()
                e.last_error = None
                e.cond.notify_all()
            self._add_used(est)
            self._enforce_budget(protect=e)
            return idx, e

    def _release(self, e, query_elapsed_s):
        with e.lock:
            e.refcount -= 1
            e.last_access = time.time()
            if query_elapsed_s is not None:
                e.queries += 1
                e.total_query_s += query_elapsed_s
            if e.waiters:
                e.cond.notify_all()

    def _load(self, e, budget):
        """Load the index file. Runs WITHOUT any manager lock held."""
        try:
            sig = _file_sig(e.path)
        except OSError as exc:
            raise TenantMountError(
                e.tenant_id, "index file is missing: %s (%s)" % (e.path, exc))
        est = estimate_resident_bytes(e.path)
        if budget is not None and est > budget:
            raise TenantBudgetError(e.tenant_id, est, budget)
        idx = _AnnoyIndex(e.dim, e.metric)
        try:
            idx.load(e.path)
        except (IOError, OSError) as exc:
            raise TenantMountError(
                e.tenant_id,
                "load failed (corrupt file, wrong dimension or wrong "
                "metric): %s" % (exc,))
        return idx, sig, est

    def _unload_entry_locked(self, e):
        # Only ever called with e.refcount == 0 and e.lock held, so no
        # query can be reading the mapping we drop here. Pure in-memory
        # operation: nothing is written to disk, so an exit mid-unload
        # cannot leave a corrupt temporary file behind.
        est = e.est_bytes
        e.index = None
        e.state = _UNLOADED
        e.est_bytes = 0
        e.evictions += 1
        e.last_unload_at = time.time()
        # e.last_access is intentionally preserved for reporting.
        # Lock ordering: entry lock -> _usage_lock, never the reverse.
        self._add_used(-est)

    def _enforce_budget(self, protect=None):
        while True:
            budget = self._budget
            if budget is None or self._used <= budget:
                return
            # Find the least recently used evictable tenant.
            victim = None
            victim_ts = None
            for e in self._tenant_snapshot():
                if e is protect:
                    continue
                with e.lock:
                    if e.state == _MOUNTED and e.refcount == 0:
                        ts = e.last_access or 0.0
                        if victim is None or ts < victim_ts:
                            victim = e
                            victim_ts = ts
            if victim is None:
                # Everything mounted is pinned by in-flight queries;
                # running over budget transiently is allowed and safe.
                return
            with victim.lock:
                # Re-check: a query may have pinned it since the scan.
                if victim.state == _MOUNTED and victim.refcount == 0:
                    self._unload_entry_locked(victim)
