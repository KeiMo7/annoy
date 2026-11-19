import csv
import os

import pytest

from annoy import (
    AnnoyIndex,
    check_indexes,
    export_report_csv,
    format_report,
    read_index_metadata,
)


def build_index(path, f=8, metric="angular", n_items=100, legacy=False):
    index = AnnoyIndex(f, metric)
    for item in range(n_items):
        vector = [((item + offset) % 17) / 17.0 for offset in range(f)]
        index.add_item(item, vector)
    index.build(5)
    index.save(str(path), legacy=legacy)
    return index


def test_parameter_mismatch_rejected_with_path_and_values(tmp_path):
    path = tmp_path / "params.ann"
    build_index(path, f=8, metric="angular")

    with pytest.raises(IOError) as excinfo:
        AnnoyIndex(9, "angular").load(str(path))
    message = str(excinfo.value)
    assert str(path) in message
    assert "caller dimension=9" in message
    assert "file dimension=8" in message

    with pytest.raises(IOError) as excinfo:
        AnnoyIndex(8, "euclidean").load(str(path))
    message = str(excinfo.value)
    assert str(path) in message
    assert "metric='euclidean'" in message
    assert "metric='angular'" in message

    inspected = read_index_metadata(path, f=9, metric="angular")
    assert inspected["status"] == "incompatible"
    assert inspected["dimension"] == 8
    assert inspected["metric"] == "angular"
    assert str(path) in inspected["error"]


def test_legacy_index_loads_with_warning():
    index = AnnoyIndex(10, "angular")
    with pytest.warns(RuntimeWarning, match="Legacy index .* compatibility mode"):
        assert index.load("test/test.tree")
    assert index.get_load_status() == "legacy"
    assert index.get_metadata()["metadata_present"] is False
    assert index.get_nns_by_item(0, 10) == [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]


def test_corrupted_header_is_not_parameter_mismatch(tmp_path):
    path = tmp_path / "corrupt.ann"
    build_index(path)
    data = bytearray(path.read_bytes())
    data[28] ^= 0xFF
    path.write_bytes(data)

    inspected = read_index_metadata(path, f=8, metric="angular")
    assert inspected["status"] == "corrupted"
    assert "checksum mismatch" in inspected["error"]
    assert str(path) in inspected["error"]

    with pytest.raises(IOError) as excinfo:
        AnnoyIndex(8, "angular").load(str(path))
    message = str(excinfo.value)
    assert "Corrupted index" in message
    assert "parameter mismatch" not in message
    assert str(path) in message


def test_damaged_header_magic_is_detected_by_trailer(tmp_path):
    path = tmp_path / "damaged-magic.ann"
    build_index(path)
    data = bytearray(path.read_bytes())
    data[:8] = b"BROKEN!!"
    path.write_bytes(data)

    inspected = read_index_metadata(path)
    assert inspected["status"] == "corrupted"
    assert "header magic is damaged" in inspected["error"]
    assert str(path) in inspected["error"]

    with pytest.raises(IOError) as excinfo:
        AnnoyIndex(8, "angular").load(str(path))
    assert "metadata header magic is damaged" in str(excinfo.value)


def test_truncated_body_is_corrupted(tmp_path):
    path = tmp_path / "truncated.ann"
    build_index(path)
    path.write_bytes(path.read_bytes()[:-17])

    inspected = read_index_metadata(path)
    assert inspected["status"] == "corrupted"
    assert "header declares" in inspected["error"]
    assert str(path) in inspected["error"]


def test_metadata_read_does_not_require_body(tmp_path):
    path = tmp_path / "metadata-only.ann"
    build_index(path, f=12, metric="manhattan")
    original = read_index_metadata(path)

    data = bytearray(path.read_bytes())
    body_start = 4096
    body_end = body_start + original["body_size"]
    data[body_start:body_end] = b"\0" * original["body_size"]
    path.write_bytes(data)

    inspected = read_index_metadata(path, f=12, metric="manhattan")
    assert inspected["status"] == "ok"
    assert inspected["format"] == "annoy-v2"
    assert inspected["dimension"] == 12
    assert inspected["metric"] == "manhattan"
    assert inspected["node_size"] == original["node_size"]


def test_legacy_save_downgrade(tmp_path):
    path = tmp_path / "legacy-export.ann"
    index = build_index(tmp_path / "source.ann", f=8, metric="angular")
    index.save(str(path), legacy=True)

    raw = path.read_bytes()
    assert raw[:8] != b"ANNOYIDX"
    inspected = read_index_metadata(path, f=8, metric="angular")
    assert inspected["status"] == "legacy"
    assert inspected["dimension"] == 8
    assert inspected["metric"] == "angular"

    restored = AnnoyIndex(8, "angular")
    with pytest.warns(RuntimeWarning, match="Legacy index"):
        restored.load(str(path))
    assert restored.get_n_items() == index.get_n_items()


def test_unsupported_future_version(tmp_path):
    path = tmp_path / "future.ann"
    build_index(path)
    data = bytearray(path.read_bytes())
    data[8:12] = (2).to_bytes(4, "little")
    path.write_bytes(data)

    inspected = read_index_metadata(path)
    assert inspected["status"] == "unsupported"
    assert "format version 2" in inspected["error"]

    with pytest.raises(IOError) as excinfo:
        AnnoyIndex(8, "angular").load(str(path))
    assert "Unsupported index" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_batch_audit_continues_and_exports_csv(tmp_path):
    good = tmp_path / "good.ann"
    legacy = tmp_path / "legacy.ann"
    bad = tmp_path / "bad.ann"
    corrupt = tmp_path / "corrupt.ann"
    missing = tmp_path / "missing.ann"

    index = build_index(good, f=8, metric="angular")
    index.save(str(legacy), legacy=True)
    build_index(bad, f=4, metric="euclidean")

    data = bytearray(good.read_bytes())
    data[16] ^= 0xFF
    corrupt.write_bytes(data)

    report = check_indexes(
        [good, legacy, bad, corrupt, missing], f=8, metric="angular", n_jobs=4
    )
    assert report["summary"] == {"total": 5, "loadable": 2, "rebuild": 3}
    assert {entry["path"] for entry in report["loadable"]} == {str(good), str(legacy)}
    assert {entry["path"] for entry in report["rebuild"]} == {
        str(bad),
        str(corrupt),
        str(missing),
    }
    assert all(entry["dimension"] is not None for entry in report["loadable"])
    assert all(entry["metric"] for entry in report["loadable"])

    text = format_report(report)
    assert "Loadable (2)" in text
    assert "Rebuild (3)" in text
    assert "dimension=8 metric=angular" in text

    csv_path = tmp_path / "audit.csv"
    export_report_csv(report, csv_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert {row["classification"] for row in rows} == {"loadable", "rebuild"}
    assert all("dimension" in row and "metric" in row for row in rows)


def test_on_disk_build_writes_metadata(tmp_path):
    path = tmp_path / "on-disk.ann"
    index = AnnoyIndex(2, "euclidean")
    index.on_disk_build(str(path))
    index.add_item(0, [0, 0])
    index.add_item(1, [1, 0])
    index.add_item(2, [2, 0])
    index.build(3)

    inspected = read_index_metadata(path)
    assert inspected["status"] == "ok"
    assert inspected["dimension"] == 2
    assert inspected["metric"] == "euclidean"
    assert inspected["item_count"] == 3


def test_new_format_fails_old_reader_node_size_check(tmp_path):
    path = tmp_path / "new-layout.ann"
    build_index(path, f=10, metric="angular")
    inspected = read_index_metadata(path)
    assert os.path.getsize(path) % inspected["node_size"] != 0
