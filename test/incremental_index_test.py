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

import os
import random
import threading
import warnings

import numpy
import pytest

from annoy import AnnoyIndex, IncrementalIndex


def random_vectors(n, f, seed=0):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(f)] for _ in range(n)]


def exact_nns(id_vectors, query, n):
    ids = [i for i, _ in id_vectors]
    q = numpy.asarray(query, dtype=numpy.float64)
    q /= numpy.linalg.norm(q)
    v = numpy.asarray([vec for _, vec in id_vectors], dtype=numpy.float64)
    v /= numpy.linalg.norm(v, axis=1, keepdims=True)
    sims = v @ q
    return set(ids[i] for i in numpy.argsort(-sims)[:n])


def recall_at_n(index, id_vectors, n, n_queries=50, seed=123):
    queries = random_vectors(n_queries, len(id_vectors[0][1]), seed=seed)
    hits = 0
    for q in queries:
        exact = exact_nns(id_vectors, q, n)
        approx = set(index.get_nns_by_vector(q, n))
        hits += len(exact & approx) / float(n)
    return hits / n_queries


def make_index(f=10, n_base=1000, n_trees=8, seed=0):
    base = random_vectors(n_base, f, seed=seed)
    idx = IncrementalIndex(f, "angular", n_trees=n_trees)
    idx.build(base)
    return idx, base


def test_single_batch_incremental():
    idx, base = make_index()
    batch = random_vectors(100, 10, seed=1)
    ids = idx.add_batch(batch)
    assert len(ids) == 100
    # new items land in a separate id space above the original items
    assert all(i >= idx.base_n_items for i in ids)
    # and are searchable immediately
    for i, v in zip(ids, batch):
        assert idx.get_nns_by_vector(v, 1)[0] == i
    # no new trees were created: items went into the existing forest
    assert idx.get_n_trees() == 8


def test_multiple_batches_accumulate():
    idx, base = make_index()
    all_ids = []
    for b in range(5):
        batch = random_vectors(50, 10, seed=100 + b)
        ids = idx.add_batch(batch)
        all_ids += ids
        for i, v in zip(ids, batch):
            assert idx.get_nns_by_vector(v, 1)[0] == i
    assert len(set(all_ids)) == 250
    assert idx.get_n_trees() == 8
    # old items are still found
    for i in range(0, 1000, 100):
        assert idx.get_nns_by_item(i, 1)[0] == i


def test_new_items_reachable_from_old_items():
    # a new item almost identical to an old one must show up as its neighbor
    idx, base = make_index()
    v = list(base[7])
    v[0] += 1e-6
    ids = idx.add_batch([v])
    assert ids[0] in idx.get_nns_by_item(7, 5)


def test_empty_batch():
    idx, base = make_index()
    n_items = idx.get_n_items()
    assert idx.add_batch([]) == []
    assert idx.get_n_items() == n_items


def test_dimension_mismatch_rejected():
    idx, base = make_index()
    n_items = idx.get_n_items()
    with pytest.raises(ValueError):
        idx.add_batch([[1.0] * 11])
    with pytest.raises(ValueError):
        idx.add_batch([[1.0] * 9])
    # nothing was inserted
    assert idx.get_n_items() == n_items
    # bad vector in the middle of a batch rejects the whole batch
    good = random_vectors(3, 10, seed=5)
    with pytest.raises(ValueError):
        idx.add_batch([good[0], [1.0] * 4, good[1]])
    assert idx.get_n_items() == n_items


def test_duplicate_ids_last_write_wins():
    idx, base = make_index()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        idx.add_batch([[1.0] * 10, [2.0] * 10], ids=[3, 3])
        assert len(w) >= 1
        assert "last write wins" in str(w[0].message)
    assert abs(idx.get_item_vector(3)[0] - 2.0) < 1e-6
    # overwriting an existing id does not grow the item count
    n_items = idx.get_n_items()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx.add_batch([[5.0] * 10], ids=[3])
    assert idx.get_n_items() == n_items
    assert abs(idx.get_item_vector(3)[0] - 5.0) < 1e-6


def test_persistence_roundtrip(tmp_path):
    idx, base = make_index()
    batch = random_vectors(80, 10, seed=2)
    ids = idx.add_batch(batch)
    path = str(tmp_path / "index.ann")
    idx.save(path)

    reloaded = IncrementalIndex.load(path)
    assert reloaded.base_n_items == idx.base_n_items
    assert reloaded.new_ids == idx.new_ids
    for i, v in zip(ids, batch):
        assert reloaded.get_nns_by_vector(v, 1)[0] == i
    # behavior continues consistently: more increments after reload
    more = random_vectors(40, 10, seed=3)
    more_ids = reloaded.add_batch(more)
    for i, v in zip(more_ids, more):
        assert reloaded.get_nns_by_vector(v, 1)[0] == i
    reloaded.save(path)
    again = IncrementalIndex.load(path)
    for i, v in zip(more_ids, more):
        assert again.get_nns_by_vector(v, 1)[0] == i


def test_saved_index_loads_with_plain_annoy(tmp_path):
    # callers cannot tell whether the file was built fully or incrementally
    idx, base = make_index()
    batch = random_vectors(60, 10, seed=4)
    ids = idx.add_batch(batch)
    path = str(tmp_path / "index.ann")
    idx.save(path)
    raw = AnnoyIndex(10, "angular")
    raw.load(path)
    for i, v in zip(ids, batch):
        assert raw.get_nns_by_vector(v, 1)[0] == i
    assert raw.get_n_items() == idx.get_n_items()


def test_corrupt_file_falls_back_to_backup(tmp_path):
    idx, base = make_index()
    idx.add_batch(random_vectors(30, 10, seed=6))
    path = str(tmp_path / "index.ann")
    idx.save(path)
    n_items_first_save = idx.get_n_items()
    idx.add_batch(random_vectors(30, 10, seed=7))
    idx.save(path)
    # corrupt the main file; the previous version must take over
    with open(path, "r+b") as f:
        f.truncate(64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reloaded = IncrementalIndex.load(path)
        assert any("fell back" in str(x.message) for x in w)
    assert reloaded.get_n_items() == n_items_first_save


def test_partial_write_leaves_consistent_state(tmp_path):
    # a crash during save leaves at most a stale .tmp file; the main
    # file (or its backup) stays loadable
    idx, base = make_index()
    path = str(tmp_path / "index.ann")
    idx.save(path)
    with open(path + ".tmp", "wb") as f:
        f.write(os.urandom(100))
    reloaded = IncrementalIndex.load(path)
    assert reloaded.get_n_items() == idx.get_n_items()


def test_rebuild_compacts_to_full_format(tmp_path):
    idx, base = make_index()
    ids = []
    batches = []
    for b in range(3):
        batch = random_vectors(40, 10, seed=20 + b)
        batches += batch
        ids += idx.add_batch(batch)
    idx.rebuild()
    assert idx.new_ids == []
    # ids and vectors are preserved
    for i, v in zip(ids, batches):
        got = idx.get_item_vector(i)
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, v))
        assert idx.get_nns_by_vector(v, 1)[0] == i
    # and the compacted index still round-trips through disk
    path = str(tmp_path / "index.ann")
    idx.save(path)
    reloaded = IncrementalIndex.load(path)
    for i, v in zip(ids, batches):
        assert reloaded.get_nns_by_vector(v, 1)[0] == i


def test_unbuild_after_incremental_insert_is_rejected():
    idx, base = make_index()
    idx.add_batch(random_vectors(10, 10, seed=30))
    with pytest.raises(Exception):
        idx._index.unbuild()


def test_recall_close_to_full_rebuild():
    f, n_base, n_new, n_trees = 20, 4000, 800, 10
    base = random_vectors(n_base, f, seed=42)
    extra = random_vectors(n_new, f, seed=43)

    inc = IncrementalIndex(f, "angular", n_trees=n_trees)
    inc.build(base)
    inc_ids = []
    for b in range(4):
        inc_ids += inc.add_batch(extra[b * 200:(b + 1) * 200])

    full = AnnoyIndex(f, "angular")
    for i, v in enumerate(base):
        full.add_item(i, v)
    for i, v in zip(inc_ids, extra):
        full.add_item(i, v)
    full.build(n_trees)

    id_vectors = list(enumerate(base)) + list(zip(inc_ids, extra))
    r_inc = recall_at_n(inc, id_vectors, 10)
    r_full = recall_at_n(full, id_vectors, 10)
    assert r_inc > 0.4
    assert abs(r_inc - r_full) < 0.1


def test_concurrent_append_and_query():
    idx, base = make_index(f=10, n_base=2000)
    errors = []
    observed_n_items = []
    stop = threading.Event()

    def reader():
        rng = random.Random(0)
        try:
            while not stop.is_set():
                q = [rng.gauss(0, 1) for _ in range(10)]
                res = idx.get_nns_by_vector(q, 10)
                n_items = idx.get_n_items()
                observed_n_items.append(n_items)
                for r in res:
                    assert 0 <= r < n_items, "out of bounds item id %d" % r
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def writer():
        for b in range(20):
            idx.add_batch(random_vectors(25, 10, seed=1000 + b))

    readers = [threading.Thread(target=reader) for _ in range(4)]
    writer_thread = threading.Thread(target=writer)
    for t in readers:
        t.start()
    writer_thread.start()
    writer_thread.join()
    stop.set()
    for t in readers:
        t.join()
    assert not errors
    # n_items only grows and every observed value is a consistent snapshot
    assert observed_n_items == sorted(observed_n_items)
    # everything the writer added is visible after the dust settles
    assert idx.get_n_items() >= 2000 + 20 * 25
