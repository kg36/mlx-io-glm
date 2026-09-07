// Copyright © 2026 LivMLX contributors.
// Bounded, runtime-gated native events for the LivMLX Perfetto capture.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mlx/api.h"

#ifdef __APPLE__
#include <pthread.h>
#include <time.h>
#endif

namespace mlx::core::perfetto_trace {

enum class Kind : uint8_t { metal, ssd_read };

struct Record {
  Kind kind;
  uint64_t t0_ns;
  uint64_t t1_ns;
  uint64_t t2_ns;
  uint64_t tid;
  uint64_t id;
  uint64_t ticket;
  uint64_t bytes;
  uint32_t expert;
  uint16_t replica;
  uint16_t spec_begin;
};

struct FinishResult {
  size_t records{0};
  uint64_t dropped{0};
  bool complete{false};
};

inline uint64_t clock_ns() {
#ifdef __APPLE__
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
#endif
}

inline uint64_t thread_id() {
#ifdef __APPLE__
  uint64_t value = 0;
  pthread_threadid_np(nullptr, &value);
  return value;
#else
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
}

class MLX_API State {
 public:
  static State& instance();

  uint64_t start(std::string output_path, size_t max_records) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (
        active_session_.load(std::memory_order_relaxed) != 0 ||
        pending_.load(std::memory_order_relaxed) != 0) {
      throw std::runtime_error("native Perfetto trace is already active");
    }
    output_path_ = std::move(output_path);
    max_records_ = max_records;
    records_.clear();
    records_.reserve(std::min<size_t>(max_records_, 262144));
    dropped_ = 0;
    const uint64_t session = ++session_;
    active_session_.store(session, std::memory_order_release);
    return clock_ns();
  }

  uint64_t begin() {
    uint64_t session = active_session_.load(std::memory_order_relaxed);
    if (session == 0) {
      return 0;
    }
    pending_.fetch_add(1, std::memory_order_relaxed);
    if (active_session_.load(std::memory_order_acquire) != session) {
      complete_one();
      return 0;
    }
    return session;
  }

  void append(uint64_t session, Record record) {
    if (session == 0) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (session == session_) {
        if (records_.size() < max_records_) {
          records_.push_back(record);
        } else {
          ++dropped_;
        }
      }
    }
    complete_one();
  }

  void stop() {
    active_session_.store(0, std::memory_order_release);
  }

  FinishResult finish(std::chrono::milliseconds timeout) {
    stop();
    std::vector<Record> records;
    std::string path;
    uint64_t dropped = 0;
    bool complete = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      complete = done_.wait_for(lock, timeout, [this] {
        return pending_.load(std::memory_order_acquire) == 0;
      });
      records = records_;
      path = output_path_;
      dropped = dropped_;
    }
    if (path.empty()) {
      return {records.size(), dropped, complete};
    }
    FILE* out = std::fopen(path.c_str(), "w");
    if (out == nullptr) {
      throw std::runtime_error("cannot create native Perfetto trace output");
    }
    for (const auto& record : records) {
      if (record.kind == Kind::metal) {
        std::fprintf(
            out,
            "{\"type\":\"metal\",\"t0\":%llu,\"t1\":%llu,\"t2\":%llu,"
            "\"tid\":%llu,\"id\":%llu}\n",
            static_cast<unsigned long long>(record.t0_ns),
            static_cast<unsigned long long>(record.t1_ns),
            static_cast<unsigned long long>(record.t2_ns),
            static_cast<unsigned long long>(record.tid),
            static_cast<unsigned long long>(record.id));
      } else {
        std::fprintf(
            out,
            "{\"type\":\"ssd_read\",\"t0\":%llu,\"t1\":%llu,"
            "\"tid\":%llu,\"ticket\":%llu,\"bytes\":%llu,"
            "\"expert\":%u,\"replica\":%u,\"spec_begin\":%u}\n",
            static_cast<unsigned long long>(record.t0_ns),
            static_cast<unsigned long long>(record.t1_ns),
            static_cast<unsigned long long>(record.tid),
            static_cast<unsigned long long>(record.ticket),
            static_cast<unsigned long long>(record.bytes),
            record.expert,
            record.replica,
            record.spec_begin);
      }
    }
    std::fclose(out);
    return {records.size(), dropped, complete};
  }

 private:
  void complete_one() {
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      done_.notify_all();
    }
  }

  std::atomic<uint64_t> active_session_{0};
  std::atomic<size_t> pending_{0};
  uint64_t session_{0};
  size_t max_records_{0};
  uint64_t dropped_{0};
  std::string output_path_;
  std::vector<Record> records_;
  std::mutex mutex_;
  std::condition_variable done_;
};

inline uint64_t start(const std::string& path, size_t max_records) {
  return State::instance().start(path, max_records);
}

inline uint64_t begin() {
  return State::instance().begin();
}

inline void stop() {
  State::instance().stop();
}

inline FinishResult finish(size_t timeout_ms) {
  return State::instance().finish(std::chrono::milliseconds(timeout_ms));
}

inline void record_ssd_read(
    uint64_t session,
    uint64_t started_ns,
    uint64_t ticket,
    size_t replica,
    size_t expert,
    size_t bytes,
    size_t spec_begin) {
  if (session == 0) return;
  State::instance().append(
      session,
      {Kind::ssd_read,
       started_ns,
       clock_ns(),
       0,
       thread_id(),
       0,
       ticket,
       bytes,
       static_cast<uint32_t>(expert),
       static_cast<uint16_t>(replica),
       static_cast<uint16_t>(spec_begin)});
}

// Scope precisely covers the authoritative read and any short-read retries.
// No clock sampling, allocation, or locking occurs while tracing is disabled.
class ReadScope {
 public:
  ReadScope(size_t replica, size_t expert, size_t bytes)
      : session_(begin()), started_(session_ ? clock_ns() : 0),
        replica_(replica), expert_(expert), bytes_(bytes) {}
  ~ReadScope() {
    finish();
  }
  void finish() {
    const auto session = session_;
    session_ = 0;
    record_ssd_read(session, started_, 0, replica_, expert_, bytes_, 0);
  }
  ReadScope(const ReadScope&) = delete;
  ReadScope& operator=(const ReadScope&) = delete;
 private:
  uint64_t session_, started_;
  size_t replica_, expert_, bytes_;
};

} // namespace mlx::core::perfetto_trace
