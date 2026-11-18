#include "annoylib.h"
#include "kissrandom.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Annoy::AnnoyIndex;
using Annoy::AnnoyIndexMultiThreadedBuildPolicy;
using Annoy::Euclidean;
using Annoy::Kiss64Random;

typedef AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexMultiThreadedBuildPolicy> Index;

static long resident_set_kb() {
  std::ifstream statm("/proc/self/statm");
  long total_pages = 0;
  long resident_pages = 0;
  statm >> total_pages >> resident_pages;
  const long page_kb = 4;
  return resident_pages * page_kb;
}

static double percentile_us(const std::vector<uint64_t>& values, double percentile) {
  if (values.empty()) {
    return 0;
  }
  size_t rank = (size_t)std::ceil(values.size() * percentile);
  if (rank == 0) {
    rank = 1;
  }
  if (rank > values.size()) {
    rank = values.size();
  }
  return values[rank - 1] / 1000.0;
}

struct RunResult {
  double seconds;
  double qps;
  uint64_t queries;
  double p50_us;
  double p95_us;
  double p99_us;
  double p999_us;
  double max_us;
  Annoy::AnnoyQueryStats stats;
};

static RunResult run_queries(Index* index, int threads, double seconds, bool serialized, int n_items) {
  std::atomic<bool> start(false);
  std::atomic<bool> stop(false);
  std::mutex serialized_mutex;
  std::vector<std::vector<uint64_t> > latencies(threads);
  std::vector<std::thread> workers;

  for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
    workers.push_back(std::thread([&, thread_idx]() {
      std::vector<uint64_t>& local = latencies[thread_idx];
      uint64_t iteration = 0;
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!stop.load(std::memory_order_relaxed)) {
        const int32_t item = (int32_t)((thread_idx * 7919 + iteration * 104729) % n_items);
        const size_t n = 10 + (size_t)((thread_idx + iteration) % 4) * 10;
        const int search_k = 1000 + (int)((thread_idx * 3 + iteration) % 5) * 500;
        std::vector<int32_t> result;
        const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (serialized) {
          std::lock_guard<std::mutex> lock(serialized_mutex);
          index->get_nns_by_item(item, n, search_k, &result, NULL);
        } else {
          index->get_nns_by_item(item, n, search_k, &result, NULL);
        }
        const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        local.push_back((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        iteration++;
      }
    }));
  }

  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  stop.store(true, std::memory_order_relaxed);
  for (size_t i = 0; i < workers.size(); i++) {
    workers[i].join();
  }
  const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::vector<uint64_t> all;
  for (size_t i = 0; i < latencies.size(); i++) {
    all.insert(all.end(), latencies[i].begin(), latencies[i].end());
  }
  std::sort(all.begin(), all.end());

  RunResult result;
  result.seconds = std::chrono::duration<double>(end - begin).count();
  result.queries = all.size();
  result.qps = result.queries / result.seconds;
  result.p50_us = percentile_us(all, 0.50);
  result.p95_us = percentile_us(all, 0.95);
  result.p99_us = percentile_us(all, 0.99);
  result.p999_us = percentile_us(all, 0.999);
  result.max_us = all.empty() ? 0 : all.back() / 1000.0;
  result.stats = index->get_query_stats();
  return result;
}

static void print_result(const char* mode, int threads, const RunResult& result) {
  std::cout << std::left << std::setw(12) << mode
            << " threads=" << std::setw(2) << threads
            << " qps=" << std::fixed << std::setprecision(1) << result.qps
            << " queries=" << result.queries
            << " p50_us=" << std::setprecision(1) << result.p50_us
            << " p95_us=" << result.p95_us
            << " p99_us=" << result.p99_us
            << " p999_us=" << result.p999_us
            << " max_us=" << result.max_us
            << " queued_events=" << result.stats.queued_events
            << " max_wait_ms=" << std::setprecision(3)
            << (result.stats.max_wait_ns / 1000000.0)
            << std::endl;
}

static void run_soak(Index* index, int threads, double seconds, int n_items) {
  std::atomic<bool> start(false);
  std::atomic<bool> stop(false);
  std::atomic<uint64_t> queries(0);
  std::vector<std::thread> workers;

  for (int thread_idx = 0; thread_idx < threads; thread_idx++) {
    workers.push_back(std::thread([&, thread_idx]() {
      uint64_t iteration = 0;
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!stop.load(std::memory_order_relaxed)) {
        const int32_t item = (int32_t)((thread_idx * 7919 + iteration * 104729) % n_items);
        std::vector<int32_t> result;
        index->get_nns_by_item(item, 50, 2000, &result, NULL);
        queries.fetch_add(1, std::memory_order_relaxed);
        iteration++;
      }
    }));
  }

  const long baseline_rss = resident_set_kb();
  long min_rss = baseline_rss;
  long max_rss = baseline_rss;
  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);

  std::cout << "soak samples:";
  while (true) {
    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
    if (elapsed >= seconds) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    const long rss = resident_set_kb();
    min_rss = std::min(min_rss, rss);
    max_rss = std::max(max_rss, rss);
    std::cout << " " << std::fixed << std::setprecision(0) << elapsed << "s/" << rss << "KB";
    std::cout.flush();
  }

  stop.store(true, std::memory_order_relaxed);
  for (size_t i = 0; i < workers.size(); i++) {
    workers[i].join();
  }
  const long final_rss = resident_set_kb();
  min_rss = std::min(min_rss, final_rss);
  max_rss = std::max(max_rss, final_rss);
  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - begin).count();

  std::cout << " final=" << final_rss << "KB";
  std::cout << "\nsoak_summary threads=" << threads
            << " seconds=" << std::fixed << std::setprecision(1) << elapsed
            << " queries=" << queries.load()
            << " baseline_rss_kb=" << baseline_rss
            << " min_rss_kb=" << min_rss
            << " max_rss_kb=" << max_rss
            << " final_rss_kb=" << final_rss
            << " final_minus_baseline_kb=" << (final_rss - baseline_rss)
            << std::endl;
}

int main(int argc, char** argv) {
  int max_threads = 8;
  double seconds = 3.0;
  double soak_seconds = 0.0;
  int n_items = 50000;

  for (int i = 1; i < argc; i++) {
    const std::string arg(argv[i]);
    if (arg == "--threads" && i + 1 < argc) {
      max_threads = std::atoi(argv[++i]);
    } else if (arg == "--seconds" && i + 1 < argc) {
      seconds = std::atof(argv[++i]);
    } else if (arg == "--soak-seconds" && i + 1 < argc) {
      soak_seconds = std::atof(argv[++i]);
    } else if (arg == "--items" && i + 1 < argc) {
      n_items = std::atoi(argv[++i]);
    } else {
      std::cerr << "unknown or incomplete argument: " << arg << std::endl;
      return 2;
    }
  }

  std::cout << "environment cpus=" << std::thread::hardware_concurrency()
            << " items=" << n_items
            << " dimensions=48 trees=10" << std::endl;

  Index index(48);
  std::mt19937 rng(137);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> vector(48);
  for (int item = 0; item < n_items; item++) {
    for (int j = 0; j < 48; j++) {
      vector[j] = normal(rng);
    }
    index.add_item(item, &vector[0]);
  }
  index.build(10, -1);
  index.set_max_query_concurrency(0);

  for (int threads = 1; threads <= max_threads; threads *= 2) {
    index.reset_query_stats();
    RunResult serialized = run_queries(&index, threads, seconds, true, n_items);
    print_result("serialized", threads, serialized);

    index.reset_query_stats();
    RunResult concurrent = run_queries(&index, threads, seconds, false, n_items);
    print_result("concurrent", threads, concurrent);
  }

  if (soak_seconds > 0) {
    index.reset_query_stats();
    run_soak(&index, max_threads, soak_seconds, n_items);
  }
  return 0;
}
