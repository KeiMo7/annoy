# Copyright (c) 2013 Spotify AB
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

"""Incremental updates for Annoy indexes.

``IncrementalIndex`` wraps a regular ``AnnoyIndex`` and adds:

* incremental batches: new items are inserted into the existing trees
  without a full rebuild and are searchable immediately;
* a separate id space for incrementally added items (all new ids are
  greater than every original item id);
* atomic, crash-safe persistence with a backup of the previous file;
* a controlled full rebuild that compacts an incrementally grown index
  back into the plain full-build format.

The underlying file format is unchanged, so a saved index can always be
opened with a plain ``AnnoyIndex`` regardless of whether it was built in
one shot or grown incrementally.
"""

import json
import os
import warnings

from .annoylib import Annoy as _Annoy


class IncrementalIndex(object):
    def __init__(self, f, metric="angular", n_trees=10):
        self.f = f
        self.metric = metric
        self.n_trees = n_trees
        self._index = _Annoy(f, metric)
        self._base_n_items = 0
        self._new_ids = []
        self._new_ids_set = set()
        self._built = False
        self._mutable = True

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def build(self, vectors, n_trees=None):
        """Initial full build from ``vectors`` (ids 0..n-1)."""
        vectors = [self._check_vector(v) for v in vectors]
        if self._built:
            raise ValueError("index has already been built")
        for i, v in enumerate(vectors):
            self._index.add_item(i, v)
        if n_trees is not None:
            self.n_trees = n_trees
        self._index.build(self.n_trees)
        self._base_n_items = self._index.get_n_items()
        self._built = True

    @classmethod
    def load(cls, path, prefault=False):
        """Loads an index saved with :meth:`save`.

        If the main index file is missing or corrupt, falls back to the
        previous file (``<path>.bak``) automatically.
        """
        meta = cls._read_meta(path)
        index = _Annoy(meta["f"], meta["metric"])
        try:
            cls._check_file(path, meta)
            index.load(path, prefault)
        except Exception:
            bak = path + ".bak"
            bak_meta = cls._read_meta_file(path + ".meta.bak")
            index = _Annoy(bak_meta["f"], bak_meta["metric"])
            cls._check_file(bak, bak_meta)
            index.load(bak, prefault)
            meta = bak_meta
            warnings.warn(
                "index file %r is corrupt or missing; fell back to %r" % (path, bak)
            )
        self = cls(meta["f"], meta["metric"], meta.get("n_trees", 10))
        self._index = index
        self._base_n_items = meta.get("base_n_items", index.get_n_items())
        self._new_ids = [i for i in meta.get("new_ids", [])]
        self._new_ids_set = set(self._new_ids)
        self._built = True
        self._mutable = False
        return self

    # ------------------------------------------------------------------
    # incremental updates
    # ------------------------------------------------------------------
    def add_batch(self, vectors, ids=None):
        """Adds a batch of new items and returns their ids.

        The batch is applied atomically: concurrent queries either see
        all of it or none of it. With ``ids=None`` fresh ids are
        allocated above the original id space. Explicit ``ids`` that
        already exist are overwritten (last write wins, with a warning).
        An empty batch is a successful no-op.
        """
        vectors = [self._check_vector(v) for v in vectors]
        if not self._built:
            raise ValueError("build() the index before adding batches")
        if not vectors:
            return []
        if ids is not None and len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        self._ensure_mutable()
        if ids is None:
            new_ids = list(self._index.append_items(vectors))
        else:
            pairs = self._dedup(ids, vectors)
            self._index.add_items(pairs)
            new_ids = [i for i, _ in pairs]
        for i in new_ids:
            if i >= self._base_n_items and i not in self._new_ids_set:
                self._new_ids_set.add(i)
                self._new_ids.append(i)
        return new_ids

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """Atomically saves the index, keeping the previous file as backup."""
        tmp = path + ".tmp"
        bak = path + ".bak"
        self._index.save(tmp)
        size = os.path.getsize(tmp)
        if os.path.exists(path):
            os.replace(path, bak)
        os.replace(tmp, path)
        self._write_meta(path, size)
        # Re-open through the final path so later incremental batches thaw
        # the on-disk file that callers actually see.
        self._index.unload()
        self._index.load(path)
        self._mutable = False

    def rebuild(self, n_trees=None):
        """Controlled full rebuild: compacts the incrementally grown index
        back into the plain full-build format. Item ids are preserved."""
        if n_trees is not None:
            self.n_trees = n_trees
        fresh = _Annoy(self.f, self.metric)
        for i in self._all_ids():
            fresh.add_item(i, self._index.get_item_vector(i))
        fresh.build(self.n_trees)
        self._index = fresh
        self._base_n_items = fresh.get_n_items()
        self._new_ids = []
        self._new_ids_set = set()
        self._mutable = True

    # ------------------------------------------------------------------
    # queries (identical semantics to AnnoyIndex)
    # ------------------------------------------------------------------
    def get_nns_by_vector(self, vector, n, search_k=-1, include_distances=False):
        return self._index.get_nns_by_vector(vector, n, search_k, include_distances)

    def get_nns_by_item(self, item, n, search_k=-1, include_distances=False):
        return self._index.get_nns_by_item(item, n, search_k, include_distances)

    def get_item_vector(self, item):
        return self._index.get_item_vector(item)

    def get_distance(self, i, j):
        return self._index.get_distance(i, j)

    def get_n_items(self):
        return self._index.get_n_items()

    def get_n_trees(self):
        return self._index.get_n_trees()

    @property
    def base_n_items(self):
        """Number of original items; ids >= this belong to new items."""
        return self._base_n_items

    @property
    def new_ids(self):
        """Ids added incrementally since the last full build/rebuild."""
        return list(self._new_ids)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _all_ids(self):
        return list(range(self._base_n_items)) + list(self._new_ids)

    def _check_vector(self, v):
        v = list(v)
        if len(v) != self.f:
            raise ValueError(
                "vector has dimension %d, index expects %d" % (len(v), self.f)
            )
        return v

    def _dedup(self, ids, vectors):
        seen = {}
        for i, v in zip(ids, vectors):
            if i in seen or self._index.has_item(i):
                warnings.warn(
                    "duplicate item id %d; the last write wins" % i, stacklevel=3
                )
            seen[i] = v
        return list(seen.items())

    def _ensure_mutable(self):
        if not self._mutable:
            self._index.thaw()
            self._mutable = True

    def _meta(self, size):
        return {
            "f": self.f,
            "metric": self.metric,
            "n_trees": self.n_trees,
            "base_n_items": self._base_n_items,
            "new_ids": list(self._new_ids),
            "size": size,
        }

    def _write_meta(self, path, size):
        tmp = path + ".meta.tmp"
        bak = path + ".meta.bak"
        final = path + ".meta"
        with open(tmp, "w") as f:
            json.dump(self._meta(size), f)
        if os.path.exists(final):
            os.replace(final, bak)
        os.replace(tmp, final)

    @staticmethod
    def _read_meta(path):
        for candidate in (path + ".meta", path + ".meta.bak"):
            try:
                return IncrementalIndex._read_meta_file(candidate)
            except (IOError, ValueError):
                continue
        raise IOError("no readable meta file for %r" % path)

    @staticmethod
    def _read_meta_file(candidate):
        with open(candidate) as f:
            return json.load(f)

    @staticmethod
    def _check_file(path, meta):
        if not os.path.exists(path):
            raise IOError("index file %r does not exist" % path)
        expected = meta.get("size")
        if expected is not None and os.path.getsize(path) != expected:
            raise IOError(
                "index file %r has size %d, expected %d"
                % (path, os.path.getsize(path), expected)
            )
