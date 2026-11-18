import concurrent.futures
import random
import threading
import time

import pytest

from annoy import AnnoyIndex


def _make_index(n_items=12000, f=24, n_trees=8):
    index = AnnoyIndex(f, "euclidean")
    rng = random.Random(137)
    vectors = [[rng.gauss(0, 1) for _ in range(f)] for _ in range(n_items)]
    for item, vector in enumerate(vectors):
        index.add_item(item, vector)
    index.build(n_trees, n_jobs=1)
    return index, vectors


def test_parallel_queries_match_single_thread_parameter_matrix():
    index, vectors = _make_index()
    cases = []
    for item in (0, 17, 400, 11999):
        for n, search_k in ((1, 10), (10, 100), (50, 500)):
            cases.append(("item", item, n, search_k))
    for item in (3, 99, 5000):
        cases.append(("vector", vectors[item], 25, 250))

    def run(case):
        kind, value, n, search_k = case
        if kind == "item":
            return index.get_nns_by_item(value, n, search_k, include_distances=True)
        return index.get_nns_by_vector(value, n, search_k, include_distances=True)

    expected = [run(case) for case in cases]
    index.reset_query_stats()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(executor.map(run, cases * 8))

    for i, result in enumerate(actual):
        assert result == expected[i % len(cases)]

    stats = index.get_query_stats()
    assert stats["queries_started"] == len(cases) * 8
    assert stats["queries_completed"] == len(cases) * 8
    assert stats["queries_rejected"] == 0


def _wait_for_parallelism(index, minimum=2, timeout=2.0):
    deadline = time.monotonic() + timeout
    observed = 0
    while time.monotonic() < deadline:
        observed = max(observed, index.get_query_stats()["active_queries"])
        if observed >= minimum:
            return observed
        time.sleep(0.001)
    return observed


def test_queries_run_in_parallel_and_limit_queues():
    index, _ = _make_index(n_items=50000, f=48, n_trees=10)
    barrier = threading.Barrier(8)

    def one_query(_):
        barrier.wait()
        return index.get_nns_by_item(1, 100, search_k=100000)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(one_query, i) for i in range(8)]
        observed_active = _wait_for_parallelism(index)
        results = [future.result() for future in futures]

    assert observed_active >= 2
    assert len(results[0]) == 100

    index.set_max_query_concurrency(2)
    index.reset_query_stats()
    barrier = threading.Barrier(8)

    def repeated_queries(_):
        barrier.wait()
        for _ in range(4):
            index.get_nns_by_item(2, 100, search_k=100000)

    max_active = 0
    saw_queue = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(repeated_queries, i) for i in range(8)]
        while not all(future.done() for future in futures):
            stats = index.get_query_stats()
            max_active = max(max_active, stats["active_queries"])
            saw_queue = saw_queue or stats["queued_queries"] > 0
            time.sleep(0.001)
        for future in futures:
            future.result()

    stats = index.get_query_stats()
    assert max_active <= 2
    assert saw_queue or stats["queued_events"] > 0
    assert stats["queued_events"] > 0
    assert stats["max_wait_ms"] > 0


def test_write_read_parallelism_never_returns_partial_results():
    index, _ = _make_index(n_items=12000, f=20, n_trees=8)
    expected = index.get_nns_by_item(7, 40, search_k=400)
    stop = threading.Event()
    failures = []
    not_ready = 0
    lock = threading.Lock()

    def reader():
        nonlocal not_ready
        while not stop.is_set():
            try:
                result = index.get_nns_by_item(7, 40, search_k=400)
            except RuntimeError as exc:
                assert "not built" in str(exc) or "unloaded" in str(exc)
                with lock:
                    not_ready += 1
                continue
            except Exception as exc:
                failures.append(exc)
                return
            if result != expected:
                failures.append(AssertionError("query observed a partially updated index"))
                return

    readers = [threading.Thread(target=reader) for _ in range(8)]
    for thread in readers:
        thread.start()

    for _ in range(3):
        index.unbuild()
        assert index.build(8, n_jobs=1)
    stop.set()
    for thread in readers:
        thread.join(timeout=5)

    assert not failures
    assert all(not thread.is_alive() for thread in readers)
    assert index.get_nns_by_item(7, 40, search_k=400) == expected
    assert not_ready >= 0


def test_unload_then_query_returns_explicit_error():
    index, _ = _make_index(n_items=1000, f=8, n_trees=4)
    index.unload()

    with pytest.raises(RuntimeError, match="not built|unloaded"):
        index.get_nns_by_item(0, 10)
    with pytest.raises(RuntimeError, match="not built|unloaded"):
        index.get_nns_by_vector([0.0] * 8, 10)

    stats = index.get_query_stats()
    assert stats["queries_rejected"] == 2


def test_multiple_index_instances_do_not_share_query_gate():
    writer_index, _ = _make_index(n_items=20000, f=32, n_trees=4)
    reader_index, vectors = _make_index(n_items=3000, f=16, n_trees=6)
    expected = reader_index.get_nns_by_vector(vectors[10], 20, search_k=200)

    writer_index.unbuild()
    started = threading.Event()

    def rebuild_other_index():
        started.set()
        assert writer_index.build(100, n_jobs=1)

    writer = threading.Thread(target=rebuild_other_index)
    writer.start()
    assert started.wait(1)
    time.sleep(0.02)
    assert writer.is_alive()

    reader_index.reset_query_stats()
    for _ in range(100):
        assert reader_index.get_nns_by_vector(vectors[10], 20, search_k=200) == expected

    stats = reader_index.get_query_stats()
    assert stats["queued_events"] == 0
    assert stats["max_wait_ms"] < 50
    writer.join(timeout=10)
    assert not writer.is_alive()


def test_out_of_range_and_released_queries_are_counted_as_rejections():
    index, _ = _make_index(n_items=1000, f=8, n_trees=4)
    index.reset_query_stats()

    with pytest.raises(IndexError):
        index.get_nns_by_item(1001, 5)
    assert index.get_query_stats()["queries_rejected"] == 1

    index.unload()
    with pytest.raises(RuntimeError):
        index.get_nns_by_vector([0.0] * 8, 5)
    assert index.get_query_stats()["queries_rejected"] == 2
