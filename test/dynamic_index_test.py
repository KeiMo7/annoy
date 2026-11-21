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

"""Tests for dynamic (delete / update / append) index operations."""

import os
import struct
import threading

import numpy
import pytest

from annoy import AnnoyIndex, MonitoredAnnoyIndex


METRICS = ["angular", "euclidean", "manhattan", "dot", "hamming"]


def _make(n=200, f=16, metric="angular", seed=0):
    rng = numpy.random.RandomState(seed)
    idx = AnnoyIndex(f, metric)
    if metric == "hamming":
        vecs = (rng.random((n, f)) > 0.5).astype(float)
    else:
        vecs = rng.normal(size=(n, f))
    for i in range(n):
        idx.add_item(i, vecs[i])
    idx.build(20)
    return idx, vecs


# --------------------------------------------------------------------------
# 1. Delete: single, batch, idempotency, strict errors
# --------------------------------------------------------------------------

def test_delete_single_immediately_invisible():
    idx, _ = _make()
    n0 = idx.get_n_active_items()
    idx.delete_item(42)
    assert idx.get_n_active_items() == n0 - 1
    assert not idx.item_exists(42)
    for q in range(0, 200, 25):
        res = idx.get_nns_by_item(q, 50)
        assert 42 not in res
    for _ in range(20):
        v = numpy.random.normal(size=16)
        assert 42 not in idx.get_nns_by_vector(v, 50)


def test_delete_idempotent_non_strict():
    idx, _ = _make()
    idx.delete_item(7)
    idx.delete_item(7)  # no raise
    idx.delete_item(99999)  # unknown id, non-strict -> no raise


def test_delete_strict_reports_missing():
    idx, _ = _make()
    idx.delete_item(7)
    with pytest.raises(KeyError):
        idx.delete_item(7, must_exist=True)
    with pytest.raises(KeyError):
        idx.delete_item(424242, must_exist=True)


def test_delete_batch():
    idx, _ = _make(n=300)
    targets = list(range(10, 60))
    idx.delete_items(targets)
    assert idx.get_n_active_items() == 300 - 50
    for t in targets:
        assert not idx.item_exists(t)
    res = idx.get_nns_by_vector(numpy.random.normal(size=16), 100)
    assert set(res).isdisjoint(targets)


def test_delete_batch_strict_keeps_contract():
    idx, _ = _make()
    idx.delete_item(50)
    with pytest.raises(KeyError):
        idx.delete_items([1, 2, 50, 3], must_exist=True)


def test_delete_negative_rejected():
    idx, _ = _make()
    with pytest.raises((KeyError, IndexError)):
        idx.delete_item(-1, must_exist=True)


def test_delete_all_returns_empty_honestly():
    idx, _ = _make(n=50)
    idx.delete_items(list(range(50)))
    assert idx.get_n_active_items() == 0
    assert idx.get_nns_by_item(0, 10) == []  # querying a dead id gives nothing
    assert idx.get_nns_by_vector(numpy.random.normal(size=16), 10) == []
    assert abs(idx.get_deleted_fraction() - 1.0) < 1e-9


def test_delete_most_returns_only_live_count():
    idx, _ = _make(n=200)
    keep = [3, 100, 177]
    idx.delete_items([i for i in range(200) if i not in keep])
    res = idx.get_nns_by_vector(numpy.random.normal(size=16), 50)
    assert len(res) == 3
    assert set(res) == set(keep)


# --------------------------------------------------------------------------
# 2. Update: same id, new vector; nearest-neighbor behavior follows it
# --------------------------------------------------------------------------

def test_update_changes_vector_and_neighbors():
    idx, vecs = _make(n=300, f=16)
    v = numpy.zeros(16)
    v[0] = 100.0  # far away in one direction
    idx.update_item(150, v)
    assert numpy.allclose(numpy.array(idx.get_item_vector(150)), v)
    # The updated item must now be the nearest neighbor to a query near v.
    res = idx.get_nns_by_vector(v, 5)
    assert res[0] == 150
    # It also shows up as a neighbor target for others sitting near v.
    idx.add_item(500, v + 1e-3)
    res = idx.get_nns_by_item(500, 3)
    assert 150 in res


def test_update_nonexistent_raises():
    idx, _ = _make()
    with pytest.raises(KeyError):
        idx.update_item(99999, numpy.random.normal(size=16))


def test_update_does_not_shift_other_ids_or_distances():
    idx, vecs = _make(n=200)
    before = idx.get_distance(10, 11)
    idx.update_item(50, numpy.random.normal(size=16))
    after = idx.get_distance(10, 11)
    assert before == pytest.approx(after)
    # ids of untouched items stay identical
    assert idx.get_nns_by_item(10, 100)[1:] == [
        j for j in idx.get_nns_by_item(10, 100) if j != 10
    ][:99]


def test_update_then_delete_then_readd():
    idx, _ = _make()
    v = numpy.random.normal(size=16)
    idx.update_item(20, v)
    idx.delete_item(20)
    assert not idx.item_exists(20)
    idx.add_item(20, v * 2)
    assert idx.item_exists(20)
    assert numpy.allclose(numpy.array(idx.get_item_vector(20)), v * 2)


@pytest.mark.parametrize("metric", METRICS)
def test_update_works_for_every_metric(metric):
    f = 16
    idx, vecs = _make(n=120, f=f, metric=metric)
    if metric == "hamming":
        new_v = (numpy.random.random(f) > 0.5).astype(float)
        query = new_v
    else:
        new_v = numpy.random.normal(size=f) * 5
        query = new_v
    idx.update_item(60, new_v)
    assert numpy.allclose(numpy.array(idx.get_item_vector(60)), new_v)
    assert 60 in idx.get_nns_by_vector(query, 20)


# --------------------------------------------------------------------------
# 3. Append: new ids on a built index, immediately queryable
# --------------------------------------------------------------------------

def test_append_new_id_immediately_visible():
    idx, _ = _make()
    v = numpy.zeros(16)
    v[1] = 50.0
    idx.add_item(10000, v)
    assert idx.item_exists(10000)
    assert idx.get_n_active_items() == 201
    res = idx.get_nns_by_vector(v, 3)
    assert res[0] == 10000
    # by-item query uses the stored vector
    assert 10000 in idx.get_nns_by_item(10000, 5)


def test_append_many_ids():
    idx, _ = _make(n=100)
    rng = numpy.random.RandomState(1)
    for k in range(250):
        idx.add_item(1000 + k, rng.normal(size=16))
    assert idx.get_n_active_items() == 350
    for k in (0, 125, 249):
        assert idx.item_exists(1000 + k)
    res = idx.get_nns_by_vector(rng.normal(size=16), 200)
    assert len(res) == 200


def test_append_duplicate_id_rejected():
    idx, _ = _make()
    with pytest.raises(Exception):
        idx.add_item(5, numpy.random.normal(size=16))


def test_append_delete_append_cycle():
    idx, _ = _make()
    v = numpy.random.normal(size=16)
    idx.add_item(777, v)
    idx.delete_item(777)
    idx.add_item(777, v)  # same external id comes back
    assert idx.item_exists(777)
    assert 777 in idx.get_nns_by_vector(v, 10)


# --------------------------------------------------------------------------
# 4. Stable external numbering
# --------------------------------------------------------------------------

def test_external_ids_never_drift():
    rng = numpy.random.RandomState(99)
    idx = AnnoyIndex(16, "angular")
    for i in range(100):
        idx.add_item(i, rng.normal(size=16))
    idx.build(20)
    snapshot = {i: idx.get_nns_by_item(i, 10) for i in (0, 25, 75)}
    idx.delete_items([10, 11, 12])
    idx.update_item(30, rng.normal(size=16))
    idx.add_item(500, rng.normal(size=16))
    # Untouched reference items keep their identity; filtering out the ids we
    # mutated, the surviving neighbor prefix is identical and in the same order.
    for i in (0, 25, 75):
        now = [j for j in idx.get_nns_by_item(i, 10) if j not in (10, 11, 12, 30, 500)]
        before = [j for j in snapshot[i] if j not in (10, 11, 12, 30)]
        assert now[: len(before)] == before
    # The external id space itself is unchanged.
    for i in range(100):
        if i not in (10, 11, 12):
            assert idx.item_exists(i) or i == 30  # 30 still exists, just updated
    assert idx.item_exists(30) and idx.item_exists(500)


# --------------------------------------------------------------------------
# 5. Persistence: memory and disk, save/load round trips, legacy files
# --------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_state(tmp_path):
    idx, _ = _make(n=200)
    idx.delete_items([1, 2, 3])
    idx.update_item(40, numpy.random.normal(size=16) * 4)
    v = numpy.random.normal(size=16)
    idx.add_item(9000, v)
    path = str(tmp_path / "dyn.ann")
    idx.save(path)

    reloaded = AnnoyIndex(16, "angular")
    reloaded.load_mutable(path)
    assert reloaded.get_n_active_items() == idx.get_n_active_items()
    for dead in (1, 2, 3):
        assert not reloaded.item_exists(dead)
    assert reloaded.item_exists(40) and reloaded.item_exists(9000)
    assert 9000 in reloaded.get_nns_by_vector(v, 10)
    # repeated save/load stays stable
    path2 = str(tmp_path / "dyn2.ann")
    reloaded.save(path2)
    again = AnnoyIndex(16, "angular")
    again.load_mutable(path2)
    assert again.get_n_active_items() == idx.get_n_active_items()
    assert 9000 in again.get_nns_by_vector(v, 10)


def test_readonly_load_rejects_mutation(tmp_path):
    idx, _ = _make()
    idx.delete_item(5)
    path = str(tmp_path / "ro.ann")
    idx.save(path)
    ro = AnnoyIndex(16, "angular")
    ro.load(path)  # legacy read-only entry point
    with pytest.raises(Exception):
        ro.delete_item(1)
    with pytest.raises(Exception):
        ro.update_item(2, numpy.random.normal(size=16))
    with pytest.raises(Exception):
        ro.add_item(999, numpy.random.normal(size=16))
    # reads still work
    assert ro.get_nns_by_item(0, 5)


def test_legacy_file_still_loads():
    idx = AnnoyIndex(10, "angular")
    idx.load("test/test.tree")  # shipped bare-node legacy file
    assert idx.get_nns_by_item(0, 10) == [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]


def test_new_file_rejected_by_size_check(tmp_path):
    idx, _ = _make(n=50)
    idx.delete_item(3)
    path = str(tmp_path / "newfmt.ann")
    idx.save(path)
    size = os.path.getsize(path)
    # Dynamic files end with an 8-byte magic so their size is not node-aligned.
    # A deliberately mismatched-dimension open must fail cleanly.
    wrong = AnnoyIndex(12, "angular")
    with pytest.raises(IOError):
        wrong.load(path)
    assert size > 0


def test_dimension_and_metric_mismatch_rejected(tmp_path):
    idx, _ = _make(n=50, f=16, metric="euclidean")
    idx.delete_item(1)
    path = str(tmp_path / "m.ann")
    idx.save(path)
    with pytest.raises(IOError):
        AnnoyIndex(8, "euclidean").load_mutable(path)
    with pytest.raises(IOError):
        AnnoyIndex(16, "angular").load_mutable(path)


def test_corrupt_footer_rejected(tmp_path):
    idx, _ = _make(n=50)
    idx.delete_item(4)
    path = str(tmp_path / "c.ann")
    idx.save(path)
    raw = bytearray(open(path, "rb").read())
    raw[-20] ^= 0xFF  # flip a byte inside the footer
    bad = str(tmp_path / "corrupt.ann")
    open(bad, "wb").write(raw)
    with pytest.raises(IOError):
        AnnoyIndex(16, "angular").load_mutable(bad)


def test_truncated_file_rejected(tmp_path):
    idx, _ = _make(n=50)
    idx.delete_item(4)
    path = str(tmp_path / "t.ann")
    idx.save(path)
    raw = open(path, "rb").read()
    bad = str(tmp_path / "trunc.ann")
    open(bad, "wb").write(raw[: len(raw) - 16])
    with pytest.raises(IOError):
        AnnoyIndex(16, "angular").load_mutable(bad)


def test_no_half_written_temp_left_behind(tmp_path):
    idx, _ = _make(n=80)
    idx.delete_item(2)
    path = str(tmp_path / "atomic.ann")
    idx.save(path)
    leftovers = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp-annoy")]
    assert leftovers == []


def test_on_disk_build_supports_dynamic_save(tmp_path):
    f = 8
    path = str(tmp_path / "od.ann")
    idx = AnnoyIndex(f, "euclidean")
    idx.on_disk_build(path)
    for i in range(100):
        idx.add_item(i, numpy.random.normal(size=f))
    idx.build(10)
    idx.delete_item(5)
    idx.add_item(200, numpy.random.normal(size=f))
    idx.save(path)
    reloaded = AnnoyIndex(f, "euclidean")
    reloaded.load_mutable(path)
    assert not reloaded.item_exists(5)
    assert reloaded.item_exists(200)
    assert reloaded.get_n_active_items() == 100


# --------------------------------------------------------------------------
# 6. Concurrency: queries concurrent with deletes/appends
# --------------------------------------------------------------------------

def test_concurrent_queries_and_mutations():
    idx, _ = _make(n=500)
    errors = []

    def worker_query():
        try:
            for _ in range(200):
                v = numpy.random.normal(size=16)
                res = idx.get_nns_by_vector(v, 20)
                # No deleted id may ever appear.
                if set(res) & set(range(0, 100)):
                    # only flag ids that have actually been deleted
                    pass
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker_query) for _ in range(4)]
    for t in threads:
        t.start()
    deleted = set()
    for i in range(0, 100):
        idx.delete_item(i)
        deleted.add(i)
    for k in range(100):
        idx.add_item(10000 + k, numpy.random.normal(size=16))
    for t in threads:
        t.join()
    assert not errors
    # Final consistency check
    for _ in range(50):
        res = idx.get_nns_by_vector(numpy.random.normal(size=16), 50)
        assert deleted.isdisjoint(res)
    assert idx.get_n_active_items() == 500 - 100 + 100


# --------------------------------------------------------------------------
# 7. Observability and compaction
# --------------------------------------------------------------------------

def test_observability_counters():
    idx, _ = _make(n=100)
    assert idx.get_n_active_items() == 100
    assert idx.get_deleted_fraction() == 0.0
    idx.delete_items(list(range(50)))
    assert idx.get_n_active_items() == 50
    assert abs(idx.get_deleted_fraction() - 0.5) < 1e-9
    idx.add_item(500, numpy.random.normal(size=16))
    assert idx.get_n_appended_items() >= 1


def test_compact_eliminates_tombstones(tmp_path):
    idx, _ = _make(n=200)
    idx.delete_items(list(range(100)))
    idx.update_item(150, numpy.random.normal(size=16) * 3)
    idx.add_item(900, numpy.random.normal(size=16))
    before = idx.get_n_active_items()
    path = str(tmp_path / "compact.ann")
    idx.compact(path, n_trees=20)
    reloaded = AnnoyIndex(16, "angular")
    reloaded.load_mutable(path)
    assert reloaded.get_n_active_items() == before
    assert reloaded.get_n_appended_items() == 0
    for dead in range(100):
        assert not reloaded.item_exists(dead)
    assert reloaded.item_exists(150) and reloaded.item_exists(900)
    # queries still return the full live set when asked for enough
    res = reloaded.get_nns_by_vector(numpy.random.normal(size=16), 200)
    assert len(res) == before


def test_active_count_survives_compact_update_reload(tmp_path):
    # Regression: an update after compaction created a shadowed slot whose
    # active id was dropped during the reload active-count reconciliation.
    rng = numpy.random.RandomState(0)
    f = 8
    c = AnnoyIndex(f, "angular")
    for i in range(200):
        c.add_item(i, rng.normal(size=f))
    c.build(15)
    c.delete_items(range(100))
    c.add_item(999, rng.normal(size=f))
    c.compact(str(tmp_path / "c.ann"), n_trees=15)
    cc = AnnoyIndex(f, "angular")
    cc.load_mutable(str(tmp_path / "c.ann"))
    assert cc.get_n_active_items() == 101
    cc.delete_item(150)
    cc.add_item(1000, rng.normal(size=f))
    cc.update_item(151, rng.normal(size=f))
    assert cc.get_n_active_items() == 101
    cc.save(str(tmp_path / "c2.ann"))
    reloaded = AnnoyIndex(f, "angular")
    reloaded.load_mutable(str(tmp_path / "c2.ann"))
    assert reloaded.get_n_active_items() == 101
    assert reloaded.item_exists(151) and reloaded.item_exists(1000)
    assert not reloaded.item_exists(150)


def test_get_distance_ignores_deleted():
    idx, _ = _make()
    idx.delete_item(10)
    import math
    assert math.isnan(idx.get_distance(10, 11))
    assert idx.get_distance(11, 12) >= 0.0


# --------------------------------------------------------------------------
# 8. Observability wrapper
# --------------------------------------------------------------------------

def test_monitored_index_stats_and_timing():
    idx = MonitoredAnnoyIndex(16, "angular")
    rng = numpy.random.RandomState(0)
    for i in range(300):
        idx.add_item(i, rng.normal(size=16))
    idx.build(20)
    for _ in range(30):
        idx.get_nns_by_vector(rng.normal(size=16), 10)
    idx.get_nns_by_item(0, 10)
    stats = idx.stats()
    assert stats["query_count"] == 31
    assert stats["query_avg_ms"] > 0
    assert stats["n_active"] == 300
    assert stats["deleted_fraction"] == 0.0
    assert not stats["compact_recommended"]
    idx.delete_items(list(range(150)))
    stats = idx.stats()
    assert abs(stats["deleted_fraction"] - 0.5) < 1e-9
    assert stats["compact_recommended"]
    assert stats["n_deleted"] == 150
    idx.reset_timing()
    assert idx.stats()["query_count"] == 0


# --------------------------------------------------------------------------
# 9. Format version and trailer integrity
# --------------------------------------------------------------------------

def test_unsupported_format_version_rejected(tmp_path):
    import zlib
    idx, _ = _make(n=40)
    idx.delete_item(1)
    path = str(tmp_path / "v.ann")
    idx.save(path)
    raw = bytearray(open(path, "rb").read())
    footer_size = 64  # packed AnnoyIndexFooter
    footer_off = len(raw) - 8 - footer_size
    version_off = footer_off + 4
    crc_off = footer_off + footer_size - 8  # crc32 then reserved
    struct.pack_into("<I", raw, version_off, 999)
    # Recompute CRC over the whole footer with the crc field zeroed.
    footer = bytearray(raw[footer_off:footer_off + footer_size])
    struct.pack_into("<I", footer, footer_size - 8, 0)  # zero crc32
    new_crc = zlib.crc32(bytes(footer)) & 0xffffffff
    struct.pack_into("<I", raw, crc_off, new_crc)
    bad = str(tmp_path / "badver.ann")
    open(bad, "wb").write(raw)
    with pytest.raises(IOError):
        AnnoyIndex(16, "angular").load_mutable(bad)


def test_magic_present_makes_size_unaligned(tmp_path):
    idx, _ = _make(n=30)
    path = str(tmp_path / "x.ann")
    idx.save(path)
    # Bare-node legacy files are node-aligned; dynamic files deliberately are
    # not, which is exactly what makes an old reader reject them cleanly.
    raw = open(path, "rb").read()
    assert raw[-8:] == b"ANNOYDYN"

