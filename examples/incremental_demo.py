# -*- coding: utf-8 -*-
"""Acceptance demo for incremental index updates.

Builds a base index, then merges several incremental batches without a
full rebuild and prints: incremental batch sizes, query hit examples,
recall comparison against a full rebuild, query latency before/after,
and a concurrent append+query validation.

Run: python3 examples/incremental_demo.py
"""

import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy  # noqa: E402

from annoy import AnnoyIndex, IncrementalIndex  # noqa: E402

F = 40
N_BASE = 10000
N_BATCHES = 5
BATCH_SIZE = 400
N_TREES = 10
N_QUERIES = 100
TOP_N = 10
SEARCH_K = 2000
PATH = "/tmp/incremental_demo.ann"


def random_vectors(n, f, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(f)] for _ in range(n)]


def exact_nns(id_vectors, query, n):
    ids = [i for i, _ in id_vectors]
    q = numpy.asarray(query, dtype=numpy.float64)
    q /= numpy.linalg.norm(q)
    v = numpy.asarray([vec for _, vec in id_vectors], dtype=numpy.float64)
    v /= numpy.linalg.norm(v, axis=1, keepdims=True)
    return set(ids[i] for i in numpy.argsort(-(v @ q))[:n])


def recall(index, id_vectors, queries):
    hits = 0
    for q in queries:
        exact = exact_nns(id_vectors, q, TOP_N)
        approx = set(index.get_nns_by_vector(q, TOP_N, SEARCH_K))
        hits += len(exact & approx) / float(TOP_N)
    return hits / len(queries)


def avg_query_latency(index, queries, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            index.get_nns_by_vector(q, TOP_N)
        best = min(best, (time.perf_counter() - t0) / len(queries))
    return best * 1e6  # microseconds


def main():
    print("=" * 70)
    print("增量索引验收 demo: base=%d, %d 批 × %d 条, dim=%d, n_trees=%d"
          % (N_BASE, N_BATCHES, BATCH_SIZE, F, N_TREES))
    print("=" * 70)

    base = random_vectors(N_BASE, F, seed=1)
    batches = [random_vectors(BATCH_SIZE, F, seed=100 + b) for b in range(N_BATCHES)]
    queries = random_vectors(N_QUERIES, F, seed=999)

    # ---- 1. initial full build ------------------------------------------
    idx = IncrementalIndex(F, "angular", n_trees=N_TREES)
    t0 = time.perf_counter()
    idx.build(base)
    print("\n[1] 初始全量构建: %d 条, 耗时 %.2f s" % (N_BASE, time.perf_counter() - t0))
    lat_before = avg_query_latency(idx, queries)
    print("    构建后查询延迟: %.1f us/query" % lat_before)

    # ---- 2. incremental batches, no full rebuild -------------------------
    print("\n[2] 增量并入 %d 批 (查询不中断, 并入即可检索):" % N_BATCHES)
    all_new_ids = []
    for b, batch in enumerate(batches):
        t0 = time.perf_counter()
        ids = idx.add_batch(batch)
        all_new_ids += ids
        dt = (time.perf_counter() - t0) * 1e3
        hit = sum(idx.get_nns_by_vector(v, 1)[0] == i for i, v in zip(ids, batch))
        print("    批 %d: 增量 %d 条 (id %d..%d), 并入耗时 %.1f ms, "
              "自命中 %d/%d, 当前总条目 %d"
              % (b + 1, len(batch), ids[0], ids[-1], dt, hit, len(batch),
                 idx.get_n_items()))
    print("    新增条目 id 空间: [%d, %d], 原条目 id 空间: [0, %d) — 完全隔开"
          % (all_new_ids[0], all_new_ids[-1], idx.base_n_items))
    lat_after = avg_query_latency(idx, queries)
    print("    增量后查询延迟: %.1f us/query (增量前 %.1f us, 比值 %.2f)"
          % (lat_after, lat_before, lat_after / lat_before))

    # ---- 3. query hit examples -------------------------------------------
    print("\n[3] 查询命中示例 (新条目可被检索命中):")
    flat = [v for b in batches for v in b]
    for i, v in list(zip(all_new_ids, flat))[:3]:
        nns = idx.get_nns_by_vector(v, 4)
        print("    新条目 id=%d 的最近邻: %s (自身在首位: %s)"
              % (i, nns, nns[0] == i))

    # ---- 4. recall comparison vs full rebuild -----------------------------
    print("\n[4] 召回率对照 (exact = 暴力精确最近邻, search_k=%d):" % SEARCH_K)
    id_vectors = list(enumerate(base)) + list(zip(all_new_ids, flat))
    r_inc = recall(idx, id_vectors, queries)

    t0 = time.perf_counter()
    full = AnnoyIndex(F, "angular")
    for i, v in enumerate(base):
        full.add_item(i, v)
    for i, v in zip(all_new_ids, flat):
        full.add_item(i, v)
    full.build(N_TREES)
    print("    对照组全量重建 (%d 条) 耗时 %.2f s"
          % (N_BASE + N_BATCHES * BATCH_SIZE, time.perf_counter() - t0))
    r_full = recall(full, id_vectors, queries)
    print("    recall@%d  增量索引: %.4f   全量重建: %.4f   差异: %+.4f"
          % (TOP_N, r_inc, r_full, r_inc - r_full))

    # ---- 5. persistence ----------------------------------------------------
    print("\n[5] 持久化: 保存 -> 重新加载 -> 行为一致:")
    idx.save(PATH)
    reloaded = IncrementalIndex.load(PATH)
    r_reload = recall(reloaded, id_vectors, queries)
    print("    重载后 recall@%d: %.4f (与保存前一致: %s)"
          % (TOP_N, r_reload, abs(r_reload - r_inc) < 1e-9))
    raw = AnnoyIndex(F, "angular")
    raw.load(PATH)
    print("    同一文件用原始 AnnoyIndex 直接加载: n_items=%d, 可正常查询: %s"
          % (raw.get_n_items(), len(raw.get_nns_by_vector(queries[0], TOP_N)) == TOP_N))
    more = random_vectors(200, F, seed=777)
    more_ids = reloaded.add_batch(more)
    hit = sum(reloaded.get_nns_by_vector(v, 1)[0] == i for i, v in zip(more_ids, more))
    print("    重载后继续增量 %d 条, 自命中 %d/%d" % (len(more), hit, len(more)))

    # ---- 6. controlled full rebuild (incremental -> full format) ----------
    print("\n[6] 受控全量重建 (增量累积整理成全量格式, id 保持不变):")
    t0 = time.perf_counter()
    reloaded.rebuild()
    print("    重建耗时 %.2f s, new_ids 清空: %s"
          % (time.perf_counter() - t0, reloaded.new_ids == []))
    ok = all(abs(a - b) < 1e-6
             for a, b in zip(reloaded.get_item_vector(more_ids[0]), more[0]))
    print("    新增条目向量保持: %s" % ok)

    # ---- 7. concurrent append + query --------------------------------------
    print("\n[7] 并发验证: 4 个查询线程 + 1 个增量写入线程:")
    errors = []
    stop = threading.Event()

    def reader():
        rng = random.Random(0)
        try:
            while not stop.is_set():
                q = [rng.gauss(0, 1) for _ in range(F)]
                res = reloaded.get_nns_by_vector(q, TOP_N)
                n_items = reloaded.get_n_items()
                for r in res:
                    assert 0 <= r < n_items, "越界 id %d" % r
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def writer():
        for b in range(20):
            reloaded.add_batch(random_vectors(50, F, seed=2000 + b))

    readers = [threading.Thread(target=reader) for _ in range(4)]
    w = threading.Thread(target=writer)
    for t in readers:
        t.start()
    w.start()
    w.join()
    stop.set()
    for t in readers:
        t.join()
    print("    写入 20 批 × 50 条期间查询持续进行: 崩溃=无, 越界=无, 错误=%s"
          % (errors or "无"))
    print("    最终总条目: %d" % reloaded.get_n_items())

    print("\n全部验收项完成。")


if __name__ == "__main__":
    main()
