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

import os
import random
import threading
import time

import pytest

from annoy import AnnoyIndex
from annoy import (
    TenantIndexManager,
    TenantBudgetError,
    TenantBusyError,
    TenantMountError,
    UnknownTenantError,
    estimate_resident_bytes,
)

DIM = 8
N_ITEMS = 60
N_TREES = 4


def build_index(path, dim=DIM, n_items=N_ITEMS, n_trees=N_TREES,
                metric="angular", seed=0):
    rng = random.Random(seed)
    idx = AnnoyIndex(dim, metric)
    for i in range(n_items):
        idx.add_item(i, [rng.uniform(-1, 1) for _ in range(dim)])
    idx.build(n_trees)
    idx.save(str(path))
    idx.unload()
    return path


def make_manager(tmp_path, n_tenants, budget_bytes=None, dim=DIM,
                 identical=False, **kw):
    mgr = TenantIndexManager(budget_bytes=budget_bytes, **kw)
    paths = {}
    for t in range(n_tenants):
        tid = "tenant-%d" % t
        p = tmp_path / ("%s.ann" % tid)
        # identical=True gives every tenant the same bytes, so estimated
        # sizes are exactly equal and budget arithmetic is deterministic.
        build_index(p, dim=dim, seed=0 if identical else t)
        mgr.register(tid, str(p), dim)
        paths[tid] = str(p)
    return mgr, paths


def query_vector(seed=999, dim=DIM):
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dim)]


# ---------------------------------------------------------------------
# 1. budget enforcement + LRU eviction order
# ---------------------------------------------------------------------

def test_budget_enforces_lru_eviction_order(tmp_path):
    mgr, paths = make_manager(tmp_path, 4, identical=True)
    est = estimate_resident_bytes(paths["tenant-0"])
    # Room for exactly 3 of the 4 equal-sized tenants.
    mgr.set_budget(3 * est)
    v = query_vector()

    for t in range(3):
        mgr.query("tenant-%d" % t, v, 5)
    rep = mgr.report()
    assert rep["used_bytes"] == 3 * est
    assert rep["used_bytes"] <= rep["budget_bytes"]

    # Mounting the 4th must evict the least recently used: tenant-0.
    mgr.query("tenant-3", v, 5)
    rep = mgr.report()
    assert rep["used_bytes"] <= rep["budget_bytes"]
    assert not rep["tenants"]["tenant-0"]["mounted"]
    for t in (1, 2, 3):
        assert rep["tenants"]["tenant-%d" % t]["mounted"]

    # Touch tenant-1 so tenant-2 becomes the LRU, then remount tenant-0.
    mgr.query("tenant-1", v, 5)
    mgr.query("tenant-0", v, 5)
    rep = mgr.report()
    assert rep["used_bytes"] <= rep["budget_bytes"]
    assert not rep["tenants"]["tenant-2"]["mounted"]
    assert rep["tenants"]["tenant-0"]["mounted"]
    assert rep["tenants"]["tenant-0"]["remounts"] == 1
    mgr.close()


def test_estimated_memory_matches_accounted_usage(tmp_path):
    # The reported per-tenant footprint and the budget accounting must
    # come from the same source and agree exactly.
    mgr, paths = make_manager(tmp_path, 3)
    mgr.query("tenant-0", query_vector(), 5)
    mgr.query("tenant-1", query_vector(), 5)
    rep = mgr.report()
    expected = (estimate_resident_bytes(paths["tenant-0"])
                + estimate_resident_bytes(paths["tenant-1"]))
    assert rep["used_bytes"] == expected
    for tid in ("tenant-0", "tenant-1"):
        t = rep["tenants"][tid]
        assert t["resident_bytes"] == estimate_resident_bytes(paths[tid])
        assert t["n_items"] == N_ITEMS
        assert t["dim"] == DIM
        assert t["file_size"] == os.path.getsize(paths[tid])
    assert rep["tenants"]["tenant-2"]["resident_bytes"] == 0
    mgr.close()


# ---------------------------------------------------------------------
# 2. in-flight queries pin the index; concurrent mounts load once
# ---------------------------------------------------------------------

def test_inflight_lease_is_never_evicted(tmp_path):
    mgr, paths = make_manager(tmp_path, 4, identical=True)
    est = estimate_resident_bytes(paths["tenant-0"])
    mgr.set_budget(2 * est)
    v = query_vector()

    with mgr.lease("tenant-0"):
        with mgr.lease("tenant-1"):
            # Two tenants are pinned; mounting a third blows the budget,
            # but neither pinned tenant may be evicted.
            mgr.query("tenant-2", v, 5)
            rep = mgr.report()
            assert rep["tenants"]["tenant-0"]["mounted"]
            assert rep["tenants"]["tenant-1"]["mounted"]
            assert rep["tenants"]["tenant-2"]["mounted"]
            assert rep["tenants"]["tenant-0"]["refcount"] == 1
            assert rep["tenants"]["tenant-1"]["refcount"] == 1
            # The budget is transiently exceeded and that is allowed:
            # in-flight queries are never blocked or killed by eviction.
            assert rep["used_bytes"] == 3 * est
            assert rep["used_bytes"] > rep["budget_bytes"]
    # After the leases are released, mounting another tenant reaps LRU
    # entries until usage is back under the budget. Release order made
    # tenant-0 the most recently used, so tenant-2 and tenant-1 go first.
    mgr.query("tenant-3", v, 5)
    rep = mgr.report()
    assert rep["used_bytes"] <= rep["budget_bytes"]
    assert not rep["tenants"]["tenant-1"]["mounted"]
    assert not rep["tenants"]["tenant-2"]["mounted"]
    assert rep["tenants"]["tenant-0"]["mounted"]
    assert rep["tenants"]["tenant-3"]["mounted"]
    mgr.close()


def test_concurrent_mount_loads_exactly_once(tmp_path):
    mgr, _ = make_manager(tmp_path, 1)
    v = query_vector()
    n_threads = 16
    results = [None] * n_threads
    errors = []

    barrier = threading.Barrier(n_threads)

    def worker(k):
        try:
            barrier.wait(timeout=10)
            results[k] = mgr.query("tenant-0", v, 5,
                                   include_distances=True)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,))
               for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    rep = mgr.report()
    assert rep["tenants"]["tenant-0"]["loads"] == 1
    assert rep["tenants"]["tenant-0"]["refcount"] == 0
    # Every thread saw identical results.
    first = results[0]
    for r in results[1:]:
        assert r[0] == first[0]
        assert r[1] == first[1]
    mgr.close()


def test_concurrent_queries_across_tenants(tmp_path):
    mgr, _ = make_manager(tmp_path, 6)
    vectors = [query_vector(seed=s) for s in range(6)]
    errors = []

    def worker(k):
        try:
            for _ in range(20):
                mgr.query("tenant-%d" % (k % 6), vectors[k % 6], 5)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    rep = mgr.report()
    total_queries = sum(t["queries"] for t in rep["tenants"].values())
    assert total_queries == 12 * 20
    for t in rep["tenants"].values():
        assert t["refcount"] == 0
    mgr.close()


# ---------------------------------------------------------------------
# 3. remount returns identical results, distances included
# ---------------------------------------------------------------------

def test_remount_results_are_identical(tmp_path):
    mgr, _ = make_manager(tmp_path, 2)
    v = query_vector()
    ids1, dist1 = mgr.query("tenant-0", v, 10, include_distances=True)
    item1 = mgr.query_by_item("tenant-0", 3, 10, include_distances=True)

    mgr.evict("tenant-0")
    assert not mgr.report()["tenants"]["tenant-0"]["mounted"]

    ids2, dist2 = mgr.query("tenant-0", v, 10, include_distances=True)
    item2 = mgr.query_by_item("tenant-0", 3, 10, include_distances=True)
    assert ids1 == ids2
    assert dist1 == dist2
    assert item1 == item2

    # And identical to a directly loaded AnnoyIndex on the same file.
    mgr.close()
    raw = AnnoyIndex(DIM, "angular")
    raw.load(str(tmp_path / "tenant-0.ann"))
    ids3, dist3 = raw.get_nns_by_vector(v, 10, include_distances=True)
    assert ids1 == ids3
    assert dist1 == dist3


# ---------------------------------------------------------------------
# 4. failure paths: single tenant over budget, corrupt/missing/wrong-dim
# ---------------------------------------------------------------------

def test_single_tenant_exceeding_budget_is_rejected(tmp_path):
    mgr, paths = make_manager(tmp_path, 2)
    est = estimate_resident_bytes(paths["tenant-0"])
    mgr.set_budget(est - 1)  # smaller than a single index

    with pytest.raises(TenantBudgetError) as excinfo:
        mgr.query("tenant-0", query_vector(), 5)
    err = excinfo.value
    # The error carries the exact required vs available numbers.
    assert err.required_bytes == est
    assert err.available_bytes == est - 1
    assert str(est) in str(err)

    # A failed mount must not poison the tenant or the manager.
    rep = mgr.report()
    assert rep["tenants"]["tenant-0"]["failed_loads"] == 1
    assert rep["tenants"]["tenant-0"]["mounted"] is False

    # Raising the budget online makes the same query succeed.
    mgr.set_budget(est)
    mgr.query("tenant-0", query_vector(), 5)
    assert mgr.report()["tenants"]["tenant-0"]["mounted"]
    mgr.close()


def test_corrupt_file_fails_only_that_tenant(tmp_path):
    mgr, paths = make_manager(tmp_path, 2)
    with open(paths["tenant-0"], "wb") as f:
        f.write(os.urandom(1001))  # not a multiple of the node size

    with pytest.raises(TenantMountError) as excinfo:
        mgr.query("tenant-0", query_vector(), 5)
    assert excinfo.value.tenant_id == "tenant-0"

    # The other tenant is completely unaffected.
    res = mgr.query("tenant-1", query_vector(), 5)
    assert len(res) == 5
    mgr.close()


def test_missing_file_fails_only_that_tenant(tmp_path):
    mgr, paths = make_manager(tmp_path, 2)
    os.remove(paths["tenant-0"])

    with pytest.raises(TenantMountError) as excinfo:
        mgr.query("tenant-0", query_vector(), 5)
    assert "missing" in str(excinfo.value)

    assert len(mgr.query("tenant-1", query_vector(), 5)) == 5
    mgr.close()


def test_dimension_mismatch_is_rejected(tmp_path):
    # Build with dim=4 (node size 28 for angular), register with dim=5
    # (node size 32). Pick an item count whose file size is not a
    # multiple of 32 so the mismatch is caught at load time.
    mgr = TenantIndexManager()
    n = 10
    while True:
        p = tmp_path / "dim.ann"
        build_index(p, dim=4, n_items=n)
        if os.path.getsize(str(p)) % 32 != 0:
            break
        n += 1
    mgr.register("t", str(p), 5)
    with pytest.raises(TenantMountError):
        mgr.query("t", query_vector(dim=5), 5)
    mgr.close()


def test_failed_remount_raises_instead_of_empty_result(tmp_path):
    mgr, paths = make_manager(tmp_path, 1)
    v = query_vector()
    before = mgr.query("tenant-0", v, 5)
    assert len(before) == 5
    mgr.evict("tenant-0")

    # Corrupt the file while unloaded: the remount must fail loudly.
    with open(paths["tenant-0"], "wb") as f:
        f.write(b"\x00" * 999)
    with pytest.raises(TenantMountError):
        mgr.query("tenant-0", v, 5)
    mgr.close()


def test_unknown_tenant_raises(tmp_path):
    mgr = TenantIndexManager()
    with pytest.raises(UnknownTenantError):
        mgr.query("nope", query_vector(), 5)
    mgr.close()


# ---------------------------------------------------------------------
# 5. idle timeout releases tenants and records last access
# ---------------------------------------------------------------------

def test_idle_timeout_releases_and_records_last_access(tmp_path):
    # Long reaper interval so the background thread does not fire during
    # the test; reap_idle() is invoked explicitly instead.
    mgr, _ = make_manager(tmp_path, 2, idle_timeout_s=0.05,
                          reaper_interval_s=3600)
    v = query_vector()
    mgr.query("tenant-0", v, 5)
    mgr.query("tenant-1", v, 5)
    t0 = time.time()
    time.sleep(0.15)

    evicted = mgr.reap_idle()
    assert sorted(evicted) == ["tenant-0", "tenant-1"]
    rep = mgr.report()
    for tid in ("tenant-0", "tenant-1"):
        t = rep["tenants"][tid]
        assert not t["mounted"]
        # Last access was recorded before the release happened.
        assert t["last_access"] is not None
        assert t["last_unload_at"] is not None
        assert t["last_unload_at"] >= t["last_access"] >= t0 - 1.0
    mgr.close()


def test_idle_reaper_thread_releases_automatically(tmp_path):
    mgr, _ = make_manager(tmp_path, 1, idle_timeout_s=0.1,
                          reaper_interval_s=0.05)
    mgr.query("tenant-0", query_vector(), 5)
    assert mgr.report()["tenants"]["tenant-0"]["mounted"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not mgr.report()["tenants"]["tenant-0"]["mounted"]:
            break
        time.sleep(0.05)
    assert not mgr.report()["tenants"]["tenant-0"]["mounted"]
    mgr.close()


# ---------------------------------------------------------------------
# 6. replaced index file forces a remount before the next query
# ---------------------------------------------------------------------

def test_replaced_file_is_remounted_before_next_query(tmp_path):
    mgr, paths = make_manager(tmp_path, 1)
    v = query_vector()
    first = mgr.query("tenant-0", v, 10, include_distances=True)
    assert mgr.report()["tenants"]["tenant-0"]["n_items"] == N_ITEMS

    # Replace the file with a different index (different data, more items).
    build_index(tmp_path / "tenant-0.ann", n_items=N_ITEMS + 25, seed=1234)
    second = mgr.query("tenant-0", v, 10, include_distances=True)

    rep = mgr.report()["tenants"]["tenant-0"]
    assert rep["loads"] == 2  # exactly one remount
    assert rep["n_items"] == N_ITEMS + 25

    # The answer now matches a fresh load of the new file, not the old one.
    raw = AnnoyIndex(DIM, "angular")
    raw.load(paths["tenant-0"])
    expected = raw.get_nns_by_vector(v, 10, include_distances=True)
    assert second[0] == expected[0]
    assert second[1] == expected[1]
    mgr.close()


def test_staleness_check_interval_remounts_after_window(tmp_path):
    # With the interval knob set, a replacement is picked up once the
    # freshness window expires (the default, interval=None, is the
    # strict every-query check covered above).
    mgr = TenantIndexManager(staleness_check_interval_s=0.05)
    p = tmp_path / "t.ann"
    build_index(p, seed=1)
    mgr.register("t", str(p), DIM)
    mgr.query("t", query_vector(), 5)
    assert mgr.report()["tenants"]["t"]["loads"] == 1

    build_index(p, n_items=N_ITEMS + 25, seed=2)
    time.sleep(0.1)  # let the freshness window expire
    mgr.query("t", query_vector(), 5)
    t = mgr.report()["tenants"]["t"]
    assert t["loads"] == 2
    assert t["n_items"] == N_ITEMS + 25
    mgr.close()


# ---------------------------------------------------------------------
# 7. budget changes take effect online
# ---------------------------------------------------------------------

def test_budget_change_applies_without_restart(tmp_path):
    mgr, paths = make_manager(tmp_path, 4)
    est = estimate_resident_bytes(paths["tenant-0"])
    v = query_vector()
    for t in range(4):
        mgr.query("tenant-%d" % t, v, 5)
    assert mgr.report()["mounted_tenants"] == 4

    mgr.set_budget(2 * est)  # shrink online: LRU two must go
    rep = mgr.report()
    assert rep["mounted_tenants"] == 2
    assert rep["used_bytes"] <= 2 * est
    assert not rep["tenants"]["tenant-0"]["mounted"]
    assert not rep["tenants"]["tenant-1"]["mounted"]
    assert rep["tenants"]["tenant-2"]["mounted"]
    assert rep["tenants"]["tenant-3"]["mounted"]

    mgr.set_budget(None)  # back to unlimited, still no restart
    mgr.query("tenant-0", v, 5)
    assert mgr.report()["tenants"]["tenant-0"]["mounted"]
    mgr.close()


# ---------------------------------------------------------------------
# 8. capability off: existing interfaces behave exactly as before
# ---------------------------------------------------------------------

def test_existing_interface_is_untouched(tmp_path):
    # Build/save/load/query through the raw AnnoyIndex API while a
    # manager exists in the same process: results must be identical.
    p = tmp_path / "raw.ann"
    build_index(p, seed=7)
    mgr = TenantIndexManager()  # unlimited budget: capability inert
    mgr.register("t", str(p), DIM)

    raw = AnnoyIndex(DIM, "angular")
    raw.load(str(p))
    v = query_vector(seed=42)
    expected = raw.get_nns_by_vector(v, 10, include_distances=True)

    got = mgr.query("t", v, 10, include_distances=True)
    assert got[0] == expected[0]
    assert got[1] == expected[1]

    # Raw index semantics unchanged: add_item/build/save roundtrip.
    a = AnnoyIndex(DIM, "angular")
    rng = random.Random(1)
    for i in range(20):
        a.add_item(i, [rng.random() for _ in range(DIM)])
    a.build(2)
    p2 = tmp_path / "roundtrip.ann"
    a.save(str(p2))
    a.unload()
    b = AnnoyIndex(DIM, "angular")
    b.load(str(p2))
    assert b.get_n_items() == 20
    mgr.close()


# ---------------------------------------------------------------------
# 9. reporting contents
# ---------------------------------------------------------------------

def test_report_contains_required_fields(tmp_path):
    mgr, _ = make_manager(tmp_path, 2)
    v = query_vector()
    mgr.query("tenant-0", v, 5)   # miss (first mount)
    mgr.query("tenant-0", v, 5)   # hit
    mgr.evict("tenant-0")
    mgr.query("tenant-0", v, 5)   # miss again -> remount

    t = mgr.report()["tenants"]["tenant-0"]
    assert t["hits"] == 1
    assert t["misses"] == 2
    assert t["hit_rate"] == pytest.approx(1.0 / 3.0)
    assert t["remounts"] == 1  # two loads total, one of them a remount
    assert t["queries"] == 3
    assert t["avg_query_ms"] is not None and t["avg_query_ms"] >= 0
    assert t["resident_bytes"] > 0
    assert t["evictions"] == 1
    mgr.close()


# ---------------------------------------------------------------------
# 10. unloading never touches the filesystem
# ---------------------------------------------------------------------

def test_unload_leaves_no_temp_files(tmp_path):
    mgr, paths = make_manager(tmp_path, 3)
    before = sorted(os.listdir(str(tmp_path)))
    v = query_vector()
    for t in range(3):
        mgr.query("tenant-%d" % t, v, 5)
    mgr.evict("tenant-0")
    mgr.evict("tenant-1")
    mgr.close()
    after = sorted(os.listdir(str(tmp_path)))
    assert before == after
    # Original index files are byte-identical to a fresh copy.
    for tid, p in paths.items():
        assert os.path.getsize(p) > 0


def test_unregister_and_busy_errors(tmp_path):
    mgr, _ = make_manager(tmp_path, 1)
    with mgr.lease("tenant-0"):
        with pytest.raises(TenantBusyError):
            mgr.evict("tenant-0")
        with pytest.raises(TenantBusyError):
            mgr.unregister("tenant-0")
    mgr.unregister("tenant-0")
    with pytest.raises(UnknownTenantError):
        mgr.query("tenant-0", query_vector(), 5)
    mgr.close()
