# Copyright (c) 2013 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This module is a dummy wrapper around the underlying C++ module.
import time

from .annoylib import Annoy as _CAnnoy

# The C++ extension remains the default. AnnoyIndex below is a thin
# observability wrapper that adds query timing without touching the hot
# read-only path in C++.
AnnoyIndex = _CAnnoy


class MonitoredAnnoyIndex(_CAnnoy):
    """AnnoyIndex plus lightweight query-latency instrumentation.

    Everything delegates to the underlying C++ object. The only additions are
    rolling counters for query count/total time and a stats() snapshot used to
    decide when tombstone density warrants a compact().
    """

    def __init__(self, f, metric):
        super(MonitoredAnnoyIndex, self).__init__(f, metric)
        self._query_count = 0
        self._query_ns_total = 0
        self._query_ns_max = 0

    def _record(self, elapsed):
        ns = int(elapsed * 1e9)
        self._query_count += 1
        self._query_ns_total += ns
        if ns > self._query_ns_max:
            self._query_ns_max = ns

    def get_nns_by_item(self, i, n, search_k=-1, include_distances=False):
        t0 = time.perf_counter()
        try:
            return super(MonitoredAnnoyIndex, self).get_nns_by_item(
                i, n, search_k, include_distances)
        finally:
            self._record(time.perf_counter() - t0)

    def get_nns_by_vector(self, vector, n, search_k=-1, include_distances=False):
        t0 = time.perf_counter()
        try:
            return super(MonitoredAnnoyIndex, self).get_nns_by_vector(
                vector, n, search_k, include_distances)
        finally:
            self._record(time.perf_counter() - t0)

    def stats(self):
        """Snapshot of health/observability counters."""
        active = self.get_n_active_items()
        frac = self.get_deleted_fraction()
        # dead/(active+dead) = frac  =>  dead = frac*active/(1-frac)
        dead = int(round(frac * active / (1.0 - frac))) if frac < 1.0 else active
        avg_ms = (self._query_ns_total / self._query_count / 1e6
                  if self._query_count else 0.0)
        return {
            "n_items_span": self.get_n_items(),
            "n_active": active,
            "n_deleted": dead,
            "n_appended": self.get_n_appended_items(),
            "deleted_fraction": frac,
            "query_count": self._query_count,
            "query_avg_ms": avg_ms,
            "query_max_ms": self._query_ns_max / 1e6,
            # Default operating threshold: compact once a third of known ids
            # are tombstoned. Tunable by the caller.
            "compact_recommended": frac >= 1.0 / 3.0,
        }

    def reset_timing(self):
        self._query_count = 0
        self._query_ns_total = 0
        self._query_ns_max = 0
