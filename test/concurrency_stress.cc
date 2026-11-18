#include "annoylib.h"
#include "kissrandom.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

using Annoy::AnnoyIndex;
using Annoy::AnnoyIndexMultiThreadedBuildPolicy;
using Annoy::Euclidean;
using Annoy::Kiss64Random;

typedef AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexMultiThreadedBuildPolicy> Index;

static void build_index(Index* index, int n_items, int dimensions) {
  std::mt19937 rng(137);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> vector(dimensions);
  for (int item = 0; item < n_items; item++) {
    for (int j = 0; j < dimensions; j++) {
      vector[j] = normal(rng);
    }
    index->add_item(item, &vector[0]);
  }
  index->build(8, 1);
}

int main() {
  const int dimensions = 24;
  const int n_items = 8000;
  Index structural(dimensions);
  Index lifecycle(dimensions);
  build_index(&structural, n_items, dimensions);
  build_index(&lifecycle, n_items, dimensions);

  std::vector<int32_t> expected;
  structural.get_nns_by_item(17, 30, 200, &expected, NULL);
  std::vector<float> query_vector(dimensions);
  structural.get_item(17, &query_vector[0]);
  char path[256];
  std::snprintf(path, sizeof(path), "/tmp/annoy_tsan_%d.ann", (int)getpid());
  if (!lifecycle.save(path)) {
    std::fprintf(stderr, "failed to save lifecycle index\n");
    return 1;
  }

  std::atomic<bool> start(false);
  std::atomic<bool> stop(false);
  std::atomic<uint64_t> queries(0);
  std::atomic<uint64_t> not_ready(0);
  std::atomic<uint64_t> errors(0);

  std::vector<std::thread> threads;
  for (int thread_idx = 0; thread_idx < 4; thread_idx++) {
    threads.push_back(std::thread([&, thread_idx]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      uint64_t iteration = thread_idx;
      while (!stop.load(std::memory_order_relaxed)) {
        std::vector<int32_t> result;
        Annoy::AnnoyQueryStatus status = structural.get_nns_by_item_with_status(
          17, 30, 200, &result, NULL);
        if (status == Annoy::ANNOY_QUERY_OK) {
          if (result != expected) {
            errors.fetch_add(1, std::memory_order_relaxed);
          }
        } else if (status == Annoy::ANNOY_QUERY_NOT_READY) {
          not_ready.fetch_add(1, std::memory_order_relaxed);
        } else {
          errors.fetch_add(1, std::memory_order_relaxed);
        }
        if ((iteration++ & 15) == 0) {
          std::vector<int32_t> vector_result;
          status = structural.get_nns_by_vector_with_status(
            &query_vector[0], 10, 100, &vector_result, NULL);
          if (status != Annoy::ANNOY_QUERY_OK && status != Annoy::ANNOY_QUERY_NOT_READY) {
            errors.fetch_add(1, std::memory_order_relaxed);
          }
        }
        queries.fetch_add(1, std::memory_order_relaxed);
      }
    }));
  }

  for (int thread_idx = 0; thread_idx < 4; thread_idx++) {
    threads.push_back(std::thread([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!stop.load(std::memory_order_relaxed)) {
        std::vector<int32_t> result;
        Annoy::AnnoyQueryStatus status = lifecycle.get_nns_by_vector_with_status(
          &query_vector[0], 10, 100, &result, NULL);
        if (status == Annoy::ANNOY_QUERY_NOT_READY) {
          not_ready.fetch_add(1, std::memory_order_relaxed);
        } else if (status != Annoy::ANNOY_QUERY_OK) {
          errors.fetch_add(1, std::memory_order_relaxed);
        }
        queries.fetch_add(1, std::memory_order_relaxed);
      }
    }));
  }

  std::thread structural_writer([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!stop.load(std::memory_order_relaxed)) {
      structural.unbuild();
      structural.build(8, 1);
    }
  });

  std::thread lifecycle_writer([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!stop.load(std::memory_order_relaxed)) {
      lifecycle.unload();
      if (!lifecycle.load(path)) {
        errors.fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
  });

  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::seconds(5));
  stop.store(true, std::memory_order_relaxed);
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  structural_writer.join();
  lifecycle_writer.join();
  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - begin).count();
  unlink(path);

  std::printf("tsan_stress seconds=%.2f queries=%llu not_ready=%llu errors=%llu\n",
    elapsed,
    (unsigned long long)queries.load(),
    (unsigned long long)not_ready.load(),
    (unsigned long long)errors.load());
  return errors.load() == 0 ? 0 : 2;
}
