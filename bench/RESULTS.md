# Concurrency validation results

## Environment

- Host: WSL2, Ubuntu 26.04.1 LTS, x86_64
- CPU: 13th Gen Intel Core i9-13900HX, 8 online logical CPUs (4 cores x 2 SMT), 36 MiB L3
- RAM: 5.8 GiB visible to the VM
- Compiler: g++ 15.2.0
- Python used for functional tests: CPython 3.14.4
- Release flags: `-O3 -ffast-math -fno-associative-math -march=native -DANNOYLIB_MULTITHREADED_BUILD`
- Race detector build: `g++ -O1 -g -fsanitize=thread -fno-omit-frame-pointer`

## Commands

```bash
g++ -std=c++17 -O3 -ffast-math -fno-associative-math -march=native \
  -DANNOYLIB_MULTITHREADED_BUILD -pthread -Isrc \
  bench/concurrency_benchmark.cc -o /tmp/annoy_concurrency_benchmark

/tmp/annoy_concurrency_benchmark --threads 8 --seconds 3

g++ -std=c++17 -O1 -g -fsanitize=thread -fno-omit-frame-pointer \
  -DANNOYLIB_MULTITHREADED_BUILD -pthread -Isrc \
  test/concurrency_stress.cc -o /tmp/annoy_concurrency_tsan

TSAN_OPTIONS='halt_on_error=1 second_deadlock_stack=1' \
  /tmp/annoy_concurrency_tsan

.venv/bin/python -m pytest -q test \
  --ignore=test/accuracy_test.py --ignore=test/memory_leak_test.py
```

The benchmark index has 50,000 items, 48 dimensions, and 10 trees. Every
worker cycles through item queries with `n=10..40` and `search_k=1000..3000`.
`serialized` holds one caller-level `std::mutex` around the complete query and
models the previous external big-lock deployment. `concurrent` calls the same
index without that caller-level lock.

## Throughput and tail latency

```text
environment cpus=8 items=50000 dimensions=48 trees=10
serialized   threads=1  qps=9334.2 queries=28005 p50_us=103.4 p95_us=164.1 p99_us=202.3 p999_us=308.5 max_us=1665.5 queued_events=0 max_wait_ms=0.012
concurrent   threads=1  qps=9361.2 queries=28085 p50_us=102.0 p95_us=163.6 p99_us=221.9 p999_us=377.0 max_us=1917.5 queued_events=0 max_wait_ms=0.001
serialized   threads=2  qps=8417.7 queries=25256 p50_us=116.4 p95_us=177.9 p99_us=3815.1 p999_us=16873.0 max_us=58476.6 queued_events=0 max_wait_ms=0.011
concurrent   threads=2  qps=16698.0 queries=50100 p50_us=111.6 p95_us=191.0 p99_us=245.3 p999_us=462.8 max_us=10658.6 queued_events=0 max_wait_ms=0.016
serialized   threads=4  qps=5219.5 queries=15663 p50_us=175.0 p95_us=355.1 p99_us=15159.7 p999_us=83076.9 max_us=179496.9 queued_events=0 max_wait_ms=0.002
concurrent   threads=4  qps=26642.9 queries=79934 p50_us=143.0 p95_us=233.6 p99_us=295.9 p999_us=470.6 max_us=4222.9 queued_events=0 max_wait_ms=0.078
serialized   threads=8  qps=7636.5 queries=22918 p50_us=127.4 p95_us=211.8 p99_us=6856.8 p999_us=197515.3 max_us=897758.8 queued_events=0 max_wait_ms=0.016
concurrent   threads=8  qps=46828.7 queries=140634 p50_us=160.3 p95_us=278.6 p99_us=372.1 p999_us=760.7 max_us=5683.9 queued_events=0 max_wait_ms=0.165
```

At 8 worker threads, the concurrent path reached 46,828.7 QPS versus 7,636.5
QPS for the external-lock path (6.1x). p99 was 0.372 ms versus 6.857 ms, and
p999 was 0.761 ms versus 197.515 ms.

For the single-thread compatibility check, the unmodified baseline header and
the concurrent implementation were run on one pinned CPU (`taskset -c 0`) for
three 5-second runs. Median QPS was about 8.30k for the baseline and 8.80k for
the new path; the variance between runs was larger than the difference. The
1-thread row above also measured 9,334.2 QPS (serialized wrapper) versus
9,361.2 QPS (new path). There is no repeatable single-thread regression.

## ThreadSanitizer

The stress test runs four readers against structural `unbuild`/`build`, four
readers against `unload`/`load`, and both writer threads concurrently:

```text
tsan_stress seconds=5.37 queries=42040 not_ready=30736 errors=0
```

There were no ThreadSanitizer race reports and no process exit error.

## 180-second soak

Eight concurrent query threads, 5,676,756 queries total:

```text
soak samples: 0s/27724KB 10s/27724KB 20s/27724KB 30s/27724KB 40s/27724KB 50s/27724KB 60s/27724KB 70s/27724KB 80s/27724KB 90s/27724KB 100s/27724KB 110s/27724KB 120s/27724KB 130s/27724KB 140s/27724KB 150s/27724KB 160s/27724KB 170s/27724KB final=27692KB
soak_summary threads=8 seconds=180.0 queries=5676756 baseline_rss_kb=27712 min_rss_kb=27692 max_rss_kb=27724 final_rss_kb=27692 final_minus_baseline_kb=-20
```

RSS stayed within a 32 KiB band and ended 20 KiB below the baseline sample.

## Functional tests

```text
105 passed, 1 skipped in 51.66s
```

The suite covers concurrent equality across parameter combinations, verified
parallelism (`active_queries > 1`), a configured limit with queued waiters,
write/read rebuild races, unload-after-query errors, independent instances,
and rejection counters. The skipped test is the existing Hamming test that is
already marked `@pytest.mark.skip`. `accuracy_test.py` was excluded because its external
`ann-benchmarks.com` downloads returned HTTP 403 in this environment;
`memory_leak_test.py` is an unbounded legacy loop and was replaced by the
180-second RSS soak above.
