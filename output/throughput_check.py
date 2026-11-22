# Throughput check: the manager's refcount/lock bookkeeping must not
# measurably slow down multi-threaded queries compared to a raw,
# permanently-resident AnnoyIndex.
#
# Three modes are compared:
#   raw      - AnnoyIndex used directly (no manager)
#   strict   - manager, default: stat the index file before every query
#              (guarantees a replaced file is remounted immediately)
#   interval - manager with staleness_check_interval_s=0.1: the stat is
#              amortised, isolating the pure refcount/lock cost
#
# Run from the repository root:
#   PYTHONPATH=. python output/throughput_check.py | tee output/throughput_check.txt
import random
import threading
import time

from annoy import AnnoyIndex, TenantIndexManager

DIM = 32
N_ITEMS = 100000
N_QUERIES = 4000
N_THREADS = 4
PATH = "/tmp/throughput_check.ann"


def build():
    rng = random.Random(0)
    idx = AnnoyIndex(DIM, "angular")
    for i in range(N_ITEMS):
        idx.add_item(i, [rng.uniform(-1, 1) for _ in range(DIM)])
    idx.build(10)
    idx.save(PATH)
    idx.unload()


def bench(query_fn, n_threads):
    def worker(k):
        rng = random.Random(k)
        for _ in range(N_QUERIES // n_threads):
            query_fn([rng.uniform(-1, 1) for _ in range(DIM)])

    threads = [threading.Thread(target=worker, args=(k,))
               for k in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    return N_QUERIES / dt


def main():
    build()
    raw = AnnoyIndex(DIM, "angular")
    raw.load(PATH)

    strict = TenantIndexManager()
    strict.register("t", PATH, DIM)
    strict.query("t", [0.0] * DIM, 5)  # warm the mount

    interval = TenantIndexManager(staleness_check_interval_s=0.1)
    interval.register("t", PATH, DIM)
    interval.query("t", [0.0] * DIM, 5)

    print("index: %d items, dim=%d, 10 trees; %d queries per configuration"
          % (N_ITEMS, DIM, N_QUERIES))
    for n_threads in (1, N_THREADS):
        raw_qps = bench(lambda v: raw.get_nns_by_vector(v, 10), n_threads)
        strict_qps = bench(lambda v: strict.query("t", v, 10), n_threads)
        interval_qps = bench(lambda v: interval.query("t", v, 10), n_threads)
        print("threads=%d" % n_threads)
        print("  raw                %9.0f qps" % raw_qps)
        print("  managed (strict)   %9.0f qps  overhead=%+.1f%%"
              % (strict_qps, (raw_qps - strict_qps) / raw_qps * 100.0))
        print("  managed (interval) %9.0f qps  overhead=%+.1f%%"
              % (interval_qps, (raw_qps - interval_qps) / raw_qps * 100.0))
    strict.close()
    interval.close()


if __name__ == "__main__":
    main()
