// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <sys/mman.h>
#include <mach/mach_time.h>
#include <pthread.h>

#include <dispatch/dispatch.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/einsum.h"
#include "mlx/expert_ssd_io_event.h"
#include "mlx/io.h"
#include "mlx/ops.h"
#include "mlx/perfetto_trace.h"
#include "mlx/utils.h"
#include "python/src/convert.h"
#include "python/src/load.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

using Scalar = std::variant<bool, int64_t, double>;

class ExpertSSDMarkovState {
 public:
  size_t history_limit() const { return history_limit_; }
  ExpertSSDMarkovState(size_t expert_count, size_t history_limit)
      : expert_count_(expert_count),
        history_limit_(history_limit),
        transitions_(expert_count * expert_count, 0.0f),
        totals_(expert_count, 0.0f) {
    if (expert_count_ == 0 || history_limit_ == 0) {
      throw std::invalid_argument(
          "ExpertSSD Markov dimensions must be positive");
    }
  }

  std::vector<float> update_and_score(
      const std::vector<int64_t>& requested,
      const std::vector<int64_t>& candidates) {
    validate(requested);
    validate(candidates);
    size_t distance = 1;
    for (auto iterator = history_.rbegin(); iterator != history_.rend();
         ++iterator, ++distance) {
      const auto weight = 1.0f / static_cast<float>(distance);
      for (const auto previous : *iterator) {
        auto* row = transitions_.data() + previous * expert_count_;
        for (const auto expert : requested) {
          row[expert] += weight;
        }
        totals_[previous] += weight;
      }
    }
    history_.push_back(requested);
    if (history_.size() > history_limit_) {
      history_.pop_front();
    }

    std::vector<float> scores(candidates.size(), 0.0f);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const auto candidate = candidates[index];
      float score = 0.0f;
      for (const auto expert : requested) {
        const auto total = totals_[expert];
        if (total > 0.0f) {
          score += transitions_[expert * expert_count_ + candidate] / total;
        }
      }
      scores[index] = score;
    }
    return scores;
  }

  std::string serialize() const {
    std::string output("DSMK1", 5);
    append(output, static_cast<uint32_t>(expert_count_));
    append(output, static_cast<uint32_t>(history_limit_));
    append(output, static_cast<uint32_t>(history_.size()));
    append_bytes(output, transitions_.data(), transitions_.size());
    append_bytes(output, totals_.data(), totals_.size());
    for (const auto& route : history_) {
      append(output, static_cast<uint32_t>(route.size()));
      append_bytes(output, route.data(), route.size());
    }
    return output;
  }

  void restore(const std::string& payload) {
    if (payload.size() < 5 || payload.compare(0, 5, "DSMK1") != 0) {
      throw std::invalid_argument("ExpertSSD Markov snapshot has wrong format");
    }
    size_t offset = 5;
    const auto expert_count = read<uint32_t>(payload, offset);
    const auto history_limit = read<uint32_t>(payload, offset);
    const auto history_size = read<uint32_t>(payload, offset);
    if (expert_count != expert_count_ || history_limit != history_limit_ ||
        history_size > history_limit_) {
      throw std::invalid_argument(
          "ExpertSSD Markov snapshot dimensions are incompatible");
    }
    std::vector<float> transitions(transitions_.size());
    std::vector<float> totals(totals_.size());
    read_bytes(payload, offset, transitions.data(), transitions.size());
    read_bytes(payload, offset, totals.data(), totals.size());
    const auto invalid_score = [](float value) {
      return !std::isfinite(value) || value < 0.0f;
    };
    if (std::any_of(
            transitions.begin(), transitions.end(), invalid_score) ||
        std::any_of(totals.begin(), totals.end(), invalid_score)) {
      throw std::invalid_argument(
          "ExpertSSD Markov snapshot contains invalid scores");
    }
    std::deque<std::vector<int64_t>> history;
    for (size_t index = 0; index < history_size; ++index) {
      const auto route_size = read<uint32_t>(payload, offset);
      if (route_size > expert_count_) {
        throw std::invalid_argument(
            "ExpertSSD Markov snapshot route is too large");
      }
      std::vector<int64_t> route(route_size);
      read_bytes(payload, offset, route.data(), route.size());
      validate(route);
      history.push_back(std::move(route));
    }
    if (offset != payload.size()) {
      throw std::invalid_argument(
          "ExpertSSD Markov snapshot has trailing bytes");
    }
    transitions_ = std::move(transitions);
    totals_ = std::move(totals);
    history_ = std::move(history);
  }

 private:
  template <typename T>
  static void append(std::string& output, const T& value) {
    output.append(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  template <typename T>
  static void append_bytes(
      std::string& output,
      const T* values,
      size_t count) {
    output.append(
        reinterpret_cast<const char*>(values), count * sizeof(T));
  }

  template <typename T>
  static T read(const std::string& payload, size_t& offset) {
    T value;
    read_bytes(payload, offset, &value, 1);
    return value;
  }

  template <typename T>
  static void read_bytes(
      const std::string& payload,
      size_t& offset,
      T* values,
      size_t count) {
    const size_t bytes = count * sizeof(T);
    if (offset > payload.size() || bytes > payload.size() - offset) {
      throw std::invalid_argument("ExpertSSD Markov snapshot is truncated");
    }
    if (bytes != 0) {
      std::memcpy(values, payload.data() + offset, bytes);
    }
    offset += bytes;
  }

  void validate(const std::vector<int64_t>& experts) const {
    for (const auto expert : experts) {
      if (expert < 0 || static_cast<size_t>(expert) >= expert_count_) {
        throw std::invalid_argument(
            "ExpertSSD Markov expert id is out of range");
      }
    }
  }

  size_t expert_count_;
  size_t history_limit_;
  std::vector<float> transitions_;
  std::vector<float> totals_;
  std::deque<std::vector<int64_t>> history_;
};

struct ExpertSSDRouteCachePlan {
  uint64_t reservation{0};
  std::vector<int64_t> routed;
  std::vector<int64_t> unique;
  std::vector<int32_t> counts;
  std::vector<int32_t> compact;
  std::vector<int32_t> gate_up_rows;
  std::vector<int32_t> down_rows;
  std::vector<uint8_t> resident_before;
  std::vector<int64_t> missing;
  std::vector<int32_t> missing_gate_up_rows;
  std::vector<int32_t> missing_down_rows;
  std::vector<int64_t> evicted;
  size_t hits{0};
  size_t misses{0};
};


struct ExpertSSDGlobalPoolPlan {
  uint64_t reservation{0};
  std::vector<int64_t> routed;
  std::vector<int64_t> unique;
  std::vector<int32_t> counts;
  std::vector<int32_t> compact;
  std::vector<uint8_t> shared_bank;
  std::vector<int32_t> rows;
  std::vector<uint8_t> resident_before;
  std::vector<int64_t> missing;
  std::vector<uint8_t> missing_shared_bank;
  std::vector<int32_t> missing_rows;
  std::vector<int64_t> evicted_layers;
  std::vector<int64_t> evicted_experts;
  size_t hits{0};
  size_t misses{0};
};

class ExpertSSDCacheState {
 public:
  ExpertSSDCacheState(
      size_t expert_count,
      size_t capacity,
      double retention_decay,
      double interval_alpha,
      double initial_interval,
      double age_weight,
      double markov_boost,
      std::shared_ptr<ExpertSSDMarkovState> markov)
      : ExpertSSDCacheState(
            expert_count, {0}, {static_cast<int64_t>(capacity)}, 0,
            retention_decay, interval_alpha, initial_interval, age_weight,
            0.0, markov ? markov_boost : 0.0,
            markov ? markov->history_limit() : 1, 0.05, 0.0) {
    legacy_private_ = true;
    auto& state = layers_.front();
    if (markov) state.markov = std::move(markov);
    state.down_expert_to_row.assign(expert_count_, -1);
    state.legacy_retention.assign(expert_count_, 0.0);
    state.free_down_rows = state.free_rows;
  }

  ExpertSSDCacheState(
      size_t expert_count,
      const std::vector<int64_t>& layers,
      const std::vector<int64_t>& private_capacities,
      size_t shared_capacity,
      double retention_decay,
      double interval_alpha,
      double initial_interval,
      double age_weight,
      double deadline_weight,
      double markov_boost,
      size_t markov_history,
      double miss_pressure_alpha,
      double miss_pressure_weight)
      : expert_count_(expert_count),
        shared_capacity_(shared_capacity),
        retention_decay_(retention_decay),
        interval_alpha_(interval_alpha),
        initial_interval_(initial_interval),
        age_weight_(age_weight),
        deadline_weight_(deadline_weight),
        markov_boost_(markov_boost),
        markov_history_(markov_history),
        miss_pressure_alpha_(miss_pressure_alpha),
        miss_pressure_weight_(miss_pressure_weight) {
    if (expert_count_ == 0 || layers.empty() ||
        layers.size() != private_capacities.size() ||
        markov_history == 0) {
      throw std::invalid_argument(
          "ExpertSSD global-pool dimensions are invalid");
    }
    if (retention_decay_ < 0.0 || retention_decay_ > 1.0 ||
        interval_alpha_ <= 0.0 || interval_alpha_ > 1.0 ||
        initial_interval_ <= 0.0 || age_weight_ < 0.0 ||
        deadline_weight_ < 0.0 || markov_boost_ < 0.0 ||
        miss_pressure_alpha_ < 0.0 ||
        miss_pressure_alpha_ > 1.0 || miss_pressure_weight_ < 0.0) {
      throw std::invalid_argument(
          "ExpertSSD global-pool policy parameters are invalid");
    }
    std::unordered_set<int64_t> unique_layers;
    for (size_t index = 0; index < layers.size(); ++index) {
      const auto layer = layers[index];
      const auto capacity = private_capacities[index];
      if (layer < 0 || !unique_layers.insert(layer).second || capacity <= 0 ||
          static_cast<size_t>(capacity) > expert_count_) {
        throw std::invalid_argument(
            "ExpertSSD global-pool layer contract is invalid");
      }
      layer_lookup_[layer] = index;
      LayerState state;
      state.layer = layer;
      state.capacity = static_cast<size_t>(capacity);
      state.expert_to_row.assign(expert_count_, -1);
      state.demand.assign(expert_count_, 0.0);
      state.last_seen.assign(expert_count_, -1);
      state.interval_ema.assign(expert_count_, initial_interval_);
      state.route_last_seen.assign(expert_count_, -1);
      state.route_interval_ema.assign(expert_count_, initial_interval_);
      state.markov = std::make_shared<ExpertSSDMarkovState>(
          expert_count_, markov_history);
      for (size_t row = state.capacity; row > 0; --row) {
        state.free_rows.push_back(static_cast<int32_t>(row - 1));
      }
      layers_.push_back(std::move(state));
    }
    const auto key_count = layers_.size() * expert_count_;
    reserved_keys_.assign(key_count, 0);
    shared_rows_.assign(key_count, -1);
    retention_.assign(key_count, 0.0);
    last_access_.assign(key_count, 0);
    for (size_t row = shared_capacity_; row > 0; --row) {
      shared_free_rows_.push_back(static_cast<int32_t>(row - 1));
    }
  }

  ExpertSSDRouteCachePlan plan(
      const mx::array& indices,
      const std::vector<uint8_t>& active_mask = {},
      bool reserve_rows = false) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_private_api();
    if (reserve_rows) require_identity();
    const auto selected = plan(0, indices, active_mask);
    auto& state = layers_.front();
    for (const auto expert : selected.evicted_experts) {
      const auto row = state.down_expert_to_row[expert];
      if (row < 0) {
        throw std::invalid_argument("ExpertSSD Down map lost a victim");
      }
      state.down_expert_to_row[expert] = -1;
      state.free_down_rows.push_back(row);
      state.legacy_retention[expert] = 0.0;
    }
    for (const auto expert : selected.unique) {
      if (state.down_expert_to_row[expert] >= 0) continue;
      if (state.free_down_rows.empty()) {
        throw std::invalid_argument("ExpertSSD Down free rows diverged");
      }
      state.down_expert_to_row[expert] = state.free_down_rows.back();
      state.free_down_rows.pop_back();
    }
    ExpertSSDRouteCachePlan output;
    output.routed = selected.routed;
    output.unique = selected.unique;
    output.counts = selected.counts;
    output.compact = selected.compact;
    output.gate_up_rows = selected.rows;
    output.resident_before = selected.resident_before;
    output.missing = selected.missing;
    output.missing_gate_up_rows = selected.missing_rows;
    output.evicted = selected.evicted_experts;
    output.hits = selected.hits;
    output.misses = selected.misses;
    for (const auto expert : selected.unique) {
      output.down_rows.push_back(state.down_expert_to_row[expert]);
    }
    for (const auto expert : selected.missing) {
      output.missing_down_rows.push_back(state.down_expert_to_row[expert]);
    }
    if (reserve_rows) output.reservation = reserve(0, selected.unique);
    return output;
  }

  void set_identity(const std::string& role, const std::vector<int64_t>& layers) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (role.empty() || layers.size() != layers_.size() || !role_.empty() || serial_ != 0) {
      throw std::invalid_argument("ExpertSSD identity must be set once before use");
    }
    std::unordered_set<int64_t> unique;
    for (size_t i = 0; i < layers.size(); ++i) {
      if (layers[i] < 0 || !unique.insert(layers[i]).second ||
          (!legacy_private_ && layers[i] != layers_[i].layer)) {
        throw std::invalid_argument("ExpertSSD identity layer geometry mismatch");
      }
    }
    role_ = role;
    logical_layers_ = layers;
  }

  uint64_t reserve(int64_t layer, const std::vector<int64_t>& experts) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_identity();
    const auto& state = layer_state(layer);
    const auto index = layer_lookup_.at(layer);
    std::unordered_set<size_t> keys;
    for (const auto expert : experts) {
      validate_expert(expert);
      const auto resident_key = key(index, expert);
      if (state.expert_to_row[expert] < 0 && shared_rows_[resident_key] < 0) {
        throw std::invalid_argument("ExpertSSD cannot reserve an absent expert");
      }
      keys.insert(resident_key);
    }
    if (keys.empty()) throw std::invalid_argument("ExpertSSD reservation is empty");
    const auto ticket = ++next_reservation_;
    reservations_.emplace(ticket, std::vector<size_t>(keys.begin(), keys.end()));
    for (const auto resident_key : keys) ++reserved_keys_[resident_key];
    return ticket;
  }

  void release(uint64_t ticket) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto iterator = reservations_.find(ticket);
    if (iterator == reservations_.end()) {
      throw std::invalid_argument("ExpertSSD reservation is foreign or released");
    }
    for (const auto resident_key : iterator->second) --reserved_keys_[resident_key];
    reservations_.erase(iterator);
  }

  uint64_t pin_identity_rows() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_private_api();
    if (shared_lease_active_) {
      throw std::invalid_argument("ExpertSSD identity rows are leased for prefill");
    }
    const auto& state = layers_.front();
    if (state.capacity != expert_count_) {
      throw std::invalid_argument("ExpertSSD identity pin requires a full resident bank");
    }
    std::vector<int64_t> experts;
    for (size_t expert = 0; expert < expert_count_; ++expert) {
      if (state.expert_to_row[expert] != static_cast<int32_t>(expert) ||
          state.down_expert_to_row[expert] != static_cast<int32_t>(expert)) {
        throw std::invalid_argument("ExpertSSD identity rows are not loaded");
      }
      experts.push_back(expert);
    }
    return reserve(0, experts);
  }

  void begin_private_prefill() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_private_api();
    require_identity();
    // The zero-shared-capacity lease protects private ownership and policy
    // without clearing it. The donor must be restored before the lease ends.
    lease_shared_rows();
  }

  void end_private_prefill() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_private_api();
    restore_leased_shared_rows({});
  }

  nb::dict reservation_metadata() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    nb::dict output;
    output["role"] = role_;
    output["layers"] = logical_layers_;
    output["transactions"] = reservations_.size();
    nb::list rows;
    for (size_t index = 0; index < reserved_keys_.size(); ++index) {
      if (reserved_keys_[index]) {
        rows.append(nb::make_tuple(role_, logical_layers_[index / expert_count_],
                                   index % expert_count_, reserved_keys_[index]));
      }
    }
    output["readers"] = rows;
    return output;
  }

  size_t try_all_hit(const mx::array& indices) {
    require_private_api();
    return try_all_hit(0, indices);
  }

  void replay_all_hit_routes(
      const std::vector<std::vector<int64_t>>& routes) {
    require_private_api();
    for (const auto& routed : routes) replay_all_hit_route(0, routed);
  }

  void restore(
      const std::vector<std::pair<int64_t, int64_t>>& expert_slots,
      const std::vector<std::pair<int64_t, int64_t>>& down_expert_slots,
      const std::vector<std::pair<int64_t, double>>& retention_scores,
      int64_t clock,
      double demand_scale,
      const std::vector<double>& demand,
      const std::vector<int64_t>& last_seen,
      const std::vector<double>& interval_ema) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    require_private_api();
    auto restored = layers_.front();
    if (expert_slots.size() > restored.capacity ||
        down_expert_slots.size() != expert_slots.size() ||
        demand.size() != expert_count_ || last_seen.size() != expert_count_ ||
        interval_ema.size() != expert_count_ || clock < 0 ||
        !std::isfinite(demand_scale) || demand_scale <= 0.0) {
      throw std::invalid_argument("ExpertSSD route-cache restore geometry is invalid");
    }
    std::fill(restored.expert_to_row.begin(), restored.expert_to_row.end(), -1);
    std::fill(restored.down_expert_to_row.begin(), restored.down_expert_to_row.end(), -1);
    std::fill(restored.legacy_retention.begin(), restored.legacy_retention.end(), 0.0);
    std::vector<uint8_t> used(restored.capacity, 0), down_used(restored.capacity, 0);
    for (const auto& [expert, row] : expert_slots) {
      validate_expert(expert);
      if (row < 0 || static_cast<size_t>(row) >= restored.capacity ||
          used[row] || restored.expert_to_row[expert] >= 0) {
        throw std::invalid_argument("ExpertSSD route-cache slot assignment is invalid");
      }
      restored.expert_to_row[expert] = static_cast<int32_t>(row);
      used[row] = 1;
    }
    for (const auto& [expert, row] : down_expert_slots) {
      validate_expert(expert);
      if (row < 0 || static_cast<size_t>(row) >= restored.capacity ||
          down_used[row] || restored.down_expert_to_row[expert] >= 0 ||
          restored.expert_to_row[expert] < 0) {
        throw std::invalid_argument("ExpertSSD route-cache Down set diverges from Gate/Up");
      }
      restored.down_expert_to_row[expert] = static_cast<int32_t>(row);
      down_used[row] = 1;
    }
    for (const auto& [expert, score] : retention_scores) {
      validate_expert(expert);
      if (!std::isfinite(score)) {
        throw std::invalid_argument("ExpertSSD route-cache retention score is invalid");
      }
      restored.legacy_retention[expert] = score;
    }
    restored.free_rows.clear();
    restored.free_down_rows.clear();
    for (size_t row = restored.capacity; row > 0; --row) {
      if (!used[row - 1]) restored.free_rows.push_back(static_cast<int32_t>(row - 1));
      if (!down_used[row - 1]) restored.free_down_rows.push_back(static_cast<int32_t>(row - 1));
    }
    restored.clock = clock;
    restored.demand_scale = demand_scale;
    restored.demand = demand;
    restored.last_seen = last_seen;
    restored.interval_ema = interval_ema;
    layers_.front() = std::move(restored);
    std::fill(last_access_.begin(), last_access_.end(), 0);
    for (const auto& [expert, row] : expert_slots) touch(0, expert);
  }

  nb::dict legacy_metadata() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_private_api();
    const auto& state = layers_.front();
    std::vector<int64_t> ordered;
    for (size_t expert = 0; expert < expert_count_; ++expert) {
      if (state.expert_to_row[expert] >= 0) ordered.push_back(expert);
    }
    std::sort(ordered.begin(), ordered.end(), [&](int64_t a, int64_t b) {
      return last_access_[a] < last_access_[b];
    });
    nb::list expert_slots, down_slots, scores;
    for (const auto expert : ordered) {
      expert_slots.append(nb::make_tuple(expert, state.expert_to_row[expert]));
      down_slots.append(nb::make_tuple(expert, state.down_expert_to_row[expert]));
    }
    for (size_t expert = 0; expert < expert_count_; ++expert) {
      if (state.legacy_retention[expert] != 0.0) {
        scores.append(nb::make_tuple(expert, state.legacy_retention[expert]));
      }
    }
    nb::dict output;
    output["expert_slots"] = expert_slots;
    output["down_expert_slots"] = down_slots;
    output["retention_scores"] = scores;
    output["clock"] = state.clock;
    output["demand_scale"] = state.demand_scale;
    output["demand"] = state.demand;
    output["last_seen"] = state.last_seen;
    output["interval_ema"] = state.interval_ema;
    output["hits"] = hits_;
    output["misses"] = misses_;
    output["evictions"] = evictions_;
    return output;
  }

  ExpertSSDGlobalPoolPlan plan(
      int64_t layer, const mx::array& indices,
      const std::vector<uint8_t>& active_mask = {},
      bool reserve_rows = false) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (reserve_rows) require_identity();
    if (shared_lease_active_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool cannot plan while shared rows are leased");
    }
    auto& state = layer_state(layer);
    ExpertSSDGlobalPoolPlan output;
    flatten(indices, output.routed);
    if (!active_mask.empty() && active_mask.size() != output.routed.size()) {
      throw std::invalid_argument("ExpertSSD active mask must match route size");
    }
    output.compact.reserve(output.routed.size());
    std::unordered_map<int64_t, int32_t> compact_ids;
    compact_ids.reserve(std::min<size_t>(output.routed.size(), expert_count_));
    for (const auto expert : output.routed) {
      validate_expert(expert);
      auto [iterator, inserted] = compact_ids.emplace(
          expert, static_cast<int32_t>(output.unique.size()));
      if (inserted) {
        output.unique.push_back(expert);
        output.counts.push_back(0);
      }
      output.compact.push_back(iterator->second);
      const auto position = output.compact.size() - 1;
      output.counts[iterator->second] +=
          active_mask.empty() || active_mask[position] != 0;
    }
    if (output.unique.size() > state.capacity + shared_capacity_) {
      throw std::invalid_argument(
          "ExpertSSD route exceeds two-tier cache capacity");
    }

    const auto layer_index = layer_lookup_.at(layer);
    std::vector<uint8_t> protected_experts(expert_count_, 0);
    for (const auto expert : output.unique) {
      protected_experts[expert] = 1;
      output.resident_before.push_back(
          state.expert_to_row[expert] >= 0 ||
          shared_rows_[key(layer_index, expert)] >= 0);
    }

    std::vector<int64_t> active_unique;
    std::vector<int32_t> active_counts;
    std::vector<uint8_t> active_seen(active_mask.empty() ? 0 : output.unique.size(), 0);
    for (size_t p = 0; p < active_mask.size(); ++p) {
      const auto i = output.compact[p];
      if (active_mask[p] && !active_seen[i]) {
        active_seen[i] = 1;
        active_unique.push_back(output.unique[i]);
        active_counts.push_back(output.counts[i]);
      }
    }
    for (size_t i = 0; !active_mask.empty() && i < output.unique.size(); ++i) {
      if (output.counts[i] == 0 && !output.resident_before[i]) {
        throw std::invalid_argument("ExpertSSD inactive filler must be resident");
      }
    }
    const auto& demand_unique = active_mask.empty() ? output.unique : active_unique;
    const auto& demand_counts = active_mask.empty() ? output.counts : active_counts;
    if (demand_unique.empty()) {
      throw std::invalid_argument("ExpertSSD route must have an active expert");
    }
    if (!reservations_.empty()) {
      size_t available = state.free_rows.size() + shared_free_rows_.size();
      size_t missing = 0;
      for (size_t expert = 0; expert < expert_count_; ++expert) {
        if (state.expert_to_row[expert] >= 0 && !protected_experts[expert] &&
            !reserved_keys_[key(layer_index, expert)]) ++available;
      }
      for (size_t index = 0; index < layers_.size(); ++index) {
        for (size_t expert = 0; expert < expert_count_; ++expert) {
          const auto resident_key = key(index, expert);
          if (shared_rows_[resident_key] >= 0 && !reserved_keys_[resident_key] &&
              !(index == layer_index && protected_experts[expert])) ++available;
        }
      }
      for (const auto expert : demand_unique) {
        if (state.expert_to_row[expert] < 0 && shared_rows_[key(layer_index, expert)] < 0) ++missing;
      }
      if (missing > available) {
        throw std::invalid_argument("ExpertSSD capacity is held by in-flight reservations");
      }
    }
    ++state.call_index;
    decay_retention(state);
    std::vector<int64_t> all_experts(expert_count_);
    std::iota(all_experts.begin(), all_experts.end(), 0);
    const auto markov_scores = state.markov->update_and_score(
        demand_unique, all_experts);
    record_route_intervals(state, demand_unique);
    record_lhd_accesses(state, demand_unique, demand_counts);
    refresh_layer_retention(layer_index, state, markov_scores);

    for (const auto expert : demand_unique) {
      auto private_row = state.expert_to_row[expert];
      auto shared_row = shared_rows_[key(layer_index, expert)];
      if (private_row >= 0 || shared_row >= 0) {
        ++output.hits;
        touch(layer_index, expert);
        continue;
      }
      ++output.misses;
      bool use_shared = false;
      int32_t row = -1;
      if (!state.free_rows.empty()) {
        row = state.free_rows.back();
        state.free_rows.pop_back();
      } else if (!shared_free_rows_.empty()) {
        use_shared = true;
        row = shared_free_rows_.back();
        shared_free_rows_.pop_back();
      } else {
        const auto private_victim = choose_private_victim(
            layer_index, state, protected_experts);
        const auto shared_victim = choose_shared_victim(
            layer_index, protected_experts);
        if (!private_victim.valid && !shared_victim.valid) {
          throw std::invalid_argument(
              "ExpertSSD global pool has no safe victim");
        }
        const auto victim = worse(private_victim, shared_victim)
            ? private_victim
            : shared_victim;
        use_shared = victim.shared;
        row = remove(victim);
        output.evicted_layers.push_back(layers_[victim.layer_index].layer);
        output.evicted_experts.push_back(victim.expert);
        if (victim.shared) {
          ++shared_evictions_;
        } else {
          ++private_evictions_;
        }
        ++evictions_;
      }
      if (use_shared) {
        shared_rows_[key(layer_index, expert)] = row;
      } else {
        state.expert_to_row[expert] = row;
      }
      retention_[key(layer_index, expert)] = score(
          state, expert, markov_scores[expert]);
      touch(layer_index, expert);
      output.missing.push_back(expert);
      output.missing_shared_bank.push_back(use_shared);
      output.missing_rows.push_back(row);
    }

    for (const auto expert : output.unique) {
      const auto private_row = state.expert_to_row[expert];
      const auto shared_row = shared_rows_[key(layer_index, expert)];
      if ((private_row >= 0) == (shared_row >= 0)) {
        throw std::invalid_argument(
            "ExpertSSD global-pool bank maps diverged");
      }
      output.shared_bank.push_back(shared_row >= 0);
      output.rows.push_back(shared_row >= 0 ? shared_row : private_row);
    }
    const auto miss_ratio = static_cast<double>(output.misses) /
        static_cast<double>(demand_unique.size());
    state.miss_pressure = miss_pressure_alpha_ * miss_ratio +
        (1.0 - miss_pressure_alpha_) * state.miss_pressure;
    hits_ += output.hits;
    misses_ += output.misses;
    if (reserve_rows) output.reservation = reserve(layer, output.unique);
    return output;
  }

  size_t try_all_hit(int64_t layer, const mx::array& indices) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto& state = layer_state(layer);
    const auto layer_index = layer_lookup_.at(layer);
    std::vector<int64_t> routed;
    flatten(indices, routed);
    if (routed.empty()) {
      throw std::invalid_argument(
          "ExpertSSD global-pool live all-hit route cannot be empty");
    }
    std::unordered_set<int64_t> unique;
    unique.reserve(std::min<size_t>(routed.size(), expert_count_));
    for (const auto expert : routed) {
      validate_expert(expert);
      unique.insert(expert);
      const auto private_row = state.expert_to_row[expert];
      const auto shared_row = shared_rows_[key(layer_index, expert)];
      if ((private_row >= 0) == (shared_row >= 0)) {
        if (private_row < 0) {
          return 0;
        }
        throw std::invalid_argument(
            "ExpertSSD global-pool bank maps diverged");
      }
    }
    if (unique.size() > state.capacity + shared_capacity_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool live route exceeds cache capacity");
    }

    replay_all_hit_route(layer, routed);
    return unique.size();
  }

  void replay_all_hit_route(
      int64_t layer,
      const std::vector<int64_t>& routed) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (shared_lease_active_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool cannot replay while shared rows are leased");
    }
    if (routed.empty()) {
      throw std::invalid_argument(
          "ExpertSSD global-pool all-hit route cannot be empty");
    }
    auto& state = layer_state(layer);
    const auto layer_index = layer_lookup_.at(layer);
    std::vector<int64_t> unique;
    std::vector<int32_t> counts;
    unique.reserve(routed.size());
    counts.reserve(routed.size());
    std::unordered_map<int64_t, int32_t> compact_ids;
    compact_ids.reserve(std::min<size_t>(routed.size(), expert_count_));
    for (const auto expert : routed) {
      validate_expert(expert);
      auto [iterator, inserted] = compact_ids.emplace(
          expert, static_cast<int32_t>(unique.size()));
      if (inserted) {
        unique.push_back(expert);
        counts.push_back(0);
      }
      counts[iterator->second] += 1;
    }
    if (unique.size() > state.capacity + shared_capacity_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool all-hit route exceeds cache capacity");
    }
    for (const auto expert : unique) {
      const auto private_row = state.expert_to_row[expert];
      const auto shared_row = shared_rows_[key(layer_index, expert)];
      if ((private_row >= 0) == (shared_row >= 0)) {
        throw std::invalid_argument(
            "ExpertSSD global-pool deferred route is no longer all-hit");
      }
    }

    // Reproduce the policy-only portion of plan() exactly. Unlike the
    // per-layer cache, the shared pool is globally authoritative, so this
    // touch is applied immediately rather than deferred across another
    // layer's possible victim decision.
    ++state.call_index;
    decay_retention(state);
    std::vector<int64_t> all_experts(expert_count_);
    std::iota(all_experts.begin(), all_experts.end(), 0);
    const auto markov_scores = state.markov->update_and_score(
        unique, all_experts);
    record_route_intervals(state, unique);
    record_lhd_accesses(state, unique, counts);
    refresh_layer_retention(layer_index, state, markov_scores);
    for (const auto expert : unique) {
      touch(layer_index, expert);
    }
    state.miss_pressure =
        (1.0 - miss_pressure_alpha_) * state.miss_pressure;
    hits_ += unique.size();
  }

  void restore_rows(
      const std::vector<std::vector<int64_t>>& private_assignments,
      const std::vector<std::vector<int64_t>>& shared_assignments) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    if (hits_ != 0 || misses_ != 0 || evictions_ != 0) {
      throw std::invalid_argument(
          "ExpertSSD global-pool restore requires unused state");
    }
    for (const auto& state : layers_) {
      if (std::any_of(
              state.expert_to_row.begin(),
              state.expert_to_row.end(),
              [](int32_t row) { return row >= 0; })) {
        throw std::invalid_argument(
            "ExpertSSD global-pool restore requires empty private rows");
      }
    }
    if (std::any_of(
            shared_rows_.begin(),
            shared_rows_.end(),
            [](int32_t row) { return row >= 0; })) {
      throw std::invalid_argument(
          "ExpertSSD global-pool restore requires empty shared rows");
    }

    std::vector<std::vector<uint8_t>> private_rows_used;
    private_rows_used.reserve(layers_.size());
    for (const auto& state : layers_) {
      private_rows_used.emplace_back(state.capacity, 0);
    }
    std::vector<uint8_t> shared_rows_used(shared_capacity_, 0);
    std::vector<uint8_t> resident_keys(layers_.size() * expert_count_, 0);

    const auto validate_assignment = [&]
        (const std::vector<int64_t>& assignment, bool shared)
        -> std::tuple<size_t, int64_t, int32_t> {
      if (assignment.size() != 3) {
        throw std::invalid_argument(
            "ExpertSSD global-pool assignment must be [layer, expert, row]");
      }
      const auto layer_iterator = layer_lookup_.find(assignment[0]);
      if (layer_iterator == layer_lookup_.end()) {
        throw std::invalid_argument(
            "ExpertSSD global-pool restore layer is unmanaged");
      }
      const auto layer_index = layer_iterator->second;
      const auto expert = assignment[1];
      validate_expert(expert);
      const auto row = assignment[2];
      const auto capacity = shared
          ? shared_capacity_
          : layers_[layer_index].capacity;
      if (row < 0 || static_cast<size_t>(row) >= capacity) {
        throw std::invalid_argument(
            "ExpertSSD global-pool restore row is invalid");
      }
      const auto resident_key = key(layer_index, expert);
      if (resident_keys[resident_key]) {
        throw std::invalid_argument(
            "ExpertSSD global-pool restore duplicates an expert");
      }
      resident_keys[resident_key] = 1;
      auto& rows_used = shared
          ? shared_rows_used
          : private_rows_used[layer_index];
      if (rows_used[row]) {
        throw std::invalid_argument(
            "ExpertSSD global-pool restore duplicates a row");
      }
      rows_used[row] = 1;
      return {layer_index, expert, static_cast<int32_t>(row)};
    };

    std::vector<std::tuple<size_t, int64_t, int32_t>> private_parsed;
    std::vector<std::tuple<size_t, int64_t, int32_t>> shared_parsed;
    private_parsed.reserve(private_assignments.size());
    shared_parsed.reserve(shared_assignments.size());
    for (const auto& assignment : private_assignments) {
      private_parsed.push_back(validate_assignment(assignment, false));
    }
    for (const auto& assignment : shared_assignments) {
      shared_parsed.push_back(validate_assignment(assignment, true));
    }

    for (const auto& [layer_index, expert, row] : private_parsed) {
      layers_[layer_index].expert_to_row[expert] = row;
    }
    for (const auto& [layer_index, expert, row] : shared_parsed) {
      shared_rows_[key(layer_index, expert)] = row;
    }
    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
      auto& state = layers_[layer_index];
      state.free_rows.clear();
      for (size_t row = state.capacity; row > 0; --row) {
        if (!private_rows_used[layer_index][row - 1]) {
          state.free_rows.push_back(static_cast<int32_t>(row - 1));
        }
      }
    }
    shared_free_rows_.clear();
    for (size_t row = shared_capacity_; row > 0; --row) {
      if (!shared_rows_used[row - 1]) {
        shared_free_rows_.push_back(static_cast<int32_t>(row - 1));
      }
    }
  }

  void lease_shared_rows() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    if (shared_lease_active_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool shared rows are already leased");
    }
    for (auto& row : shared_rows_) {
      row = -1;
    }
    shared_free_rows_.clear();
    for (size_t row = shared_capacity_; row > 0; --row) {
      shared_free_rows_.push_back(static_cast<int32_t>(row - 1));
    }
    shared_lease_active_ = true;
  }

  void restore_leased_shared_rows(
      const std::vector<std::vector<int64_t>>& shared_assignments) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    if (!shared_lease_active_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool shared rows are not leased");
    }
    if (std::any_of(
            shared_rows_.begin(),
            shared_rows_.end(),
            [](int32_t row) { return row >= 0; })) {
      throw std::invalid_argument(
          "ExpertSSD global-pool leased shared rows are not empty");
    }

    std::vector<uint8_t> rows_used(shared_capacity_, 0);
    std::vector<uint8_t> experts_used(layers_.size() * expert_count_, 0);
    std::vector<std::tuple<size_t, int64_t, int32_t>> parsed;
    parsed.reserve(shared_assignments.size());
    for (const auto& assignment : shared_assignments) {
      if (assignment.size() != 3) {
        throw std::invalid_argument(
            "ExpertSSD leased shared assignment must be [layer, expert, row]");
      }
      const auto layer_iterator = layer_lookup_.find(assignment[0]);
      if (layer_iterator == layer_lookup_.end()) {
        throw std::invalid_argument(
            "ExpertSSD leased shared restore layer is unmanaged");
      }
      const auto layer_index = layer_iterator->second;
      const auto expert = assignment[1];
      validate_expert(expert);
      const auto row = assignment[2];
      if (row < 0 || static_cast<size_t>(row) >= shared_capacity_) {
        throw std::invalid_argument(
            "ExpertSSD leased shared restore row is invalid");
      }
      if (rows_used[row]) {
        throw std::invalid_argument(
            "ExpertSSD leased shared restore duplicates a row");
      }
      const auto resident_key = key(layer_index, expert);
      if (experts_used[resident_key]) {
        throw std::invalid_argument(
            "ExpertSSD leased shared restore duplicates an expert");
      }
      if (layers_[layer_index].expert_to_row[expert] >= 0) {
        throw std::invalid_argument(
            "ExpertSSD leased shared restore duplicates a private expert");
      }
      rows_used[row] = 1;
      experts_used[resident_key] = 1;
      parsed.emplace_back(layer_index, expert, static_cast<int32_t>(row));
    }
    for (const auto& [layer_index, expert, row] : parsed) {
      shared_rows_[key(layer_index, expert)] = row;
    }
    shared_free_rows_.clear();
    for (size_t row = shared_capacity_; row > 0; --row) {
      if (!rows_used[row - 1]) {
        shared_free_rows_.push_back(static_cast<int32_t>(row - 1));
      }
    }
    shared_lease_active_ = false;
  }

  void restore_layer_policy(
      int64_t layer,
      int64_t clock,
      double demand_scale,
      const std::vector<double>& demand,
      const std::vector<int64_t>& last_seen,
      const std::vector<double>& interval_ema,
      const std::string& markov_payload) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    if (hits_ != 0 || misses_ != 0 || evictions_ != 0 || clock < 0 ||
        !std::isfinite(demand_scale) || demand_scale <= 0.0 ||
        demand.size() != expert_count_ || last_seen.size() != expert_count_ ||
        interval_ema.size() != expert_count_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool restored policy is invalid");
    }
    if (std::any_of(demand.begin(), demand.end(), [](double value) {
          return !std::isfinite(value) || value < 0.0;
        }) ||
        std::any_of(interval_ema.begin(), interval_ema.end(), [](double value) {
          return !std::isfinite(value) || value <= 0.0;
        }) ||
        std::any_of(last_seen.begin(), last_seen.end(), [clock](int64_t value) {
          return value < -1 || value > clock;
        })) {
      throw std::invalid_argument(
          "ExpertSSD global-pool restored policy vectors are invalid");
    }
    auto& state = layer_state(layer);
    const auto layer_index = layer_lookup_.at(layer);
    state.clock = clock;
    state.demand_scale = demand_scale;
    state.demand = demand;
    state.last_seen = last_seen;
    state.interval_ema = interval_ema;
    if (!markov_payload.empty()) {
      state.markov->restore(markov_payload);
    }
    for (size_t expert = 0; expert < expert_count_; ++expert) {
      const auto resident_key = key(layer_index, static_cast<int64_t>(expert));
      if (state.expert_to_row[expert] >= 0 || shared_rows_[resident_key] >= 0) {
        retention_[resident_key] = score(
            state, static_cast<int64_t>(expert), 0.0);
        const auto seen = state.last_seen[expert];
        last_access_[resident_key] =
            seen < 0 ? uint64_t{0} : static_cast<uint64_t>(seen);
        serial_ = std::max(serial_, last_access_[resident_key]);
      }
    }
  }

  std::string serialize_policy() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    std::string output("DSGP1", 5);
    append_policy(output, static_cast<uint32_t>(expert_count_));
    append_policy(output, static_cast<uint32_t>(layers_.size()));
    append_policy(output, static_cast<uint32_t>(shared_capacity_));
    append_policy(output, serial_);
    for (const auto& state : layers_) {
      append_policy(output, state.layer);
      append_policy(output, static_cast<uint32_t>(state.capacity));
      append_policy(output, state.clock);
      append_policy(output, state.demand_scale);
      append_policy_vector(output, state.demand);
      append_policy_vector(output, state.last_seen);
      append_policy_vector(output, state.interval_ema);
      append_policy(output, state.call_index);
      append_policy_vector(output, state.route_last_seen);
      append_policy_vector(output, state.route_interval_ema);
      append_policy(output, state.miss_pressure);
      const auto markov = state.markov->serialize();
      append_policy(output, static_cast<uint64_t>(markov.size()));
      output.append(markov);
    }
    append_policy_vector(output, retention_);
    append_policy_vector(output, last_access_);
    return output;
  }

  void validate_policy_snapshot(const std::string& payload) const {
    (void)parse_policy_snapshot(payload);
  }

  void restore_policy_snapshot(const std::string& payload) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_quiescent();
    if (hits_ != 0 || misses_ != 0 || evictions_ != 0) {
      throw std::invalid_argument(
          "ExpertSSD global-pool exact policy restore requires unused state");
    }
    auto parsed = parse_policy_snapshot(payload);
    serial_ = parsed.serial;
    retention_ = std::move(parsed.retention);
    last_access_ = std::move(parsed.last_access);
    for (size_t index = 0; index < layers_.size(); ++index) {
      auto& state = layers_[index];
      auto& saved = parsed.layers[index];
      state.clock = saved.clock;
      state.demand_scale = saved.demand_scale;
      state.demand = std::move(saved.demand);
      state.last_seen = std::move(saved.last_seen);
      state.interval_ema = std::move(saved.interval_ema);
      state.call_index = saved.call_index;
      state.route_last_seen = std::move(saved.route_last_seen);
      state.route_interval_ema = std::move(saved.route_interval_ema);
      state.miss_pressure = saved.miss_pressure;
      state.markov->restore(saved.markov);
    }
  }

  nb::dict metadata() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    nb::dict output;
    nb::list layer_entries;
    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
      const auto& state = layers_[layer_index];
      nb::dict entry;
      entry["layer"] = state.layer;
      nb::list private_rows;
      nb::list shared_rows;
      for (size_t expert = 0; expert < expert_count_; ++expert) {
        if (state.expert_to_row[expert] >= 0) {
          private_rows.append(nb::make_tuple(
              static_cast<int64_t>(expert), state.expert_to_row[expert]));
        }
        const auto shared_row = shared_rows_[key(layer_index, expert)];
        if (shared_row >= 0) {
          shared_rows.append(nb::make_tuple(
              static_cast<int64_t>(expert), shared_row));
        }
      }
      entry["private_rows"] = private_rows;
      entry["shared_rows"] = shared_rows;
      entry["clock"] = state.clock;
      entry["demand_scale"] = state.demand_scale;
      entry["demand"] = state.demand;
      entry["last_seen"] = state.last_seen;
      entry["interval_ema"] = state.interval_ema;
      entry["miss_pressure"] = state.miss_pressure;
      layer_entries.append(entry);
    }
    output["layers"] = layer_entries;
    output["shared_capacity"] = shared_capacity_;
    output["hits"] = hits_;
    output["misses"] = misses_;
    output["evictions"] = evictions_;
    output["private_evictions"] = private_evictions_;
    output["shared_evictions"] = shared_evictions_;
    output["shared_lease_active"] = shared_lease_active_;
    return output;
  }

  nb::dict snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_identity();
    require_quiescent();
    if (shared_lease_active_) {
      throw std::invalid_argument("ExpertSSD cannot snapshot a prefill lease");
    }
    nb::dict output;
    output["version"] = 1;
    output["role"] = role_;
    output["layers"] = logical_layers_;
    output["private_mode"] = legacy_private_;
    output["parameters"] = snapshot_parameters();
    nb::list owners;
    for (const auto& state : layers_) {
      nb::dict entry;
      entry["gate_up"] = state.expert_to_row;
      entry["down"] = state.down_expert_to_row;
      entry["free_gate_up"] = state.free_rows;
      entry["free_down"] = state.free_down_rows;
      entry["retention"] = state.legacy_retention;
      owners.append(entry);
    }
    output["ownership"] = owners;
    output["shared_rows"] = shared_rows_;
    output["free_shared"] = shared_free_rows_;
    const auto policy = serialize_policy();
    output["policy"] = nb::bytes(policy.data(), policy.size());
    return output;
  }

  void validate_snapshot(const nb::dict& saved, bool for_restore = false) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (for_restore) require_cold_snapshot();
    (void)parse_snapshot(saved);
  }

  void restore_snapshot(const nb::dict& saved) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    require_cold_snapshot();
    auto parsed = parse_snapshot(saved);
    restore_policy_snapshot(parsed.policy);
    for (size_t index = 0; index < layers_.size(); ++index) {
      auto& state = layers_[index];
      auto& rows = parsed.owners[index];
      state.expert_to_row = std::move(rows.gate_up);
      state.down_expert_to_row = std::move(rows.down);
      state.free_rows = std::move(rows.free_gate_up);
      state.free_down_rows = std::move(rows.free_down);
      state.legacy_retention = std::move(rows.retention);
    }
    shared_rows_ = std::move(parsed.shared_rows);
    shared_free_rows_ = std::move(parsed.free_shared);
  }

 private:
  void require_cold_snapshot() const {
    require_quiescent();
    if (shared_lease_active_ || hits_ || misses_ || evictions_ || serial_) {
      throw std::invalid_argument("ExpertSSD snapshot restore requires cold state");
    }
    for (const auto& state : layers_) {
      if (std::any_of(state.expert_to_row.begin(), state.expert_to_row.end(),
                      [](int32_t row) { return row >= 0; })) {
        throw std::invalid_argument("ExpertSSD snapshot restore requires empty rows");
      }
    }
    if (std::any_of(shared_rows_.begin(), shared_rows_.end(),
                    [](int32_t row) { return row >= 0; })) {
      throw std::invalid_argument("ExpertSSD snapshot restore requires empty shared rows");
    }
  }

  struct SavedOwnership {
    std::vector<int32_t> gate_up, down, free_gate_up, free_down;
    std::vector<double> retention;
  };

  struct SavedState {
    std::vector<SavedOwnership> owners;
    std::vector<int32_t> shared_rows, free_shared;
    std::string policy;
  };

  std::vector<double> snapshot_parameters() const {
    return {retention_decay_, interval_alpha_, initial_interval_, age_weight_,
            deadline_weight_, markov_boost_, static_cast<double>(markov_history_),
            miss_pressure_alpha_, miss_pressure_weight_};
  }

  static void validate_row_partition(
      const std::vector<int32_t>& rows, const std::vector<int32_t>& free,
      size_t capacity) {
    std::vector<bool> used(capacity, false);
    size_t count = 0;
    for (const auto row : rows) {
      if (row == -1) continue;
      if (row < 0 || static_cast<size_t>(row) >= capacity || used[row]) {
        throw std::invalid_argument("ExpertSSD snapshot has invalid row ownership");
      }
      used[row] = true;
      ++count;
    }
    for (const auto row : free) {
      if (row < 0 || static_cast<size_t>(row) >= capacity || used[row]) {
        throw std::invalid_argument("ExpertSSD snapshot has invalid free rows");
      }
      used[row] = true;
      ++count;
    }
    if (count != capacity) {
      throw std::invalid_argument("ExpertSSD snapshot loses physical rows");
    }
  }

  SavedState parse_snapshot(const nb::dict& saved) const {
    require_identity();
    if (nb::cast<int>(saved["version"]) != 1 ||
        nb::cast<std::string>(saved["role"]) != role_ ||
        nb::cast<std::vector<int64_t>>(saved["layers"]) != logical_layers_ ||
        nb::cast<bool>(saved["private_mode"]) != legacy_private_ ||
        nb::cast<std::vector<double>>(saved["parameters"]) != snapshot_parameters()) {
      throw std::invalid_argument("ExpertSSD snapshot identity or policy differs");
    }
    const auto policy = nb::cast<nb::bytes>(saved["policy"]);
    SavedState output;
    output.policy.assign(policy.c_str(), policy.size());
    (void)parse_policy_snapshot(output.policy);
    const auto owners = nb::cast<nb::list>(saved["ownership"]);
    if (owners.size() != layers_.size()) {
      throw std::invalid_argument("ExpertSSD snapshot layer count differs");
    }
    output.shared_rows = nb::cast<std::vector<int32_t>>(saved["shared_rows"]);
    output.free_shared = nb::cast<std::vector<int32_t>>(saved["free_shared"]);
    if (output.shared_rows.size() != layers_.size() * expert_count_) {
      throw std::invalid_argument("ExpertSSD snapshot shared geometry differs");
    }
    validate_row_partition(output.shared_rows, output.free_shared, shared_capacity_);
    for (size_t index = 0; index < layers_.size(); ++index) {
      const auto entry = nb::cast<nb::dict>(owners[index]);
      SavedOwnership rows{
          nb::cast<std::vector<int32_t>>(entry["gate_up"]),
          nb::cast<std::vector<int32_t>>(entry["down"]),
          nb::cast<std::vector<int32_t>>(entry["free_gate_up"]),
          nb::cast<std::vector<int32_t>>(entry["free_down"]),
          nb::cast<std::vector<double>>(entry["retention"])};
      if (rows.gate_up.size() != expert_count_ ||
          rows.down.size() != (legacy_private_ ? expert_count_ : 0) ||
          rows.retention.size() != rows.down.size() ||
          std::any_of(rows.retention.begin(), rows.retention.end(),
                      [](double value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("ExpertSSD snapshot projection geometry differs");
      }
      validate_row_partition(rows.gate_up, rows.free_gate_up, layers_[index].capacity);
      validate_row_partition(rows.down, rows.free_down, legacy_private_ ? layers_[index].capacity : 0);
      for (size_t expert = 0; expert < expert_count_; ++expert) {
        const bool resident = rows.gate_up[expert] >= 0;
        if ((resident && output.shared_rows[key(index, expert)] >= 0) ||
            (legacy_private_ && resident != (rows.down[expert] >= 0))) {
          throw std::invalid_argument("ExpertSSD snapshot projection ownership differs");
        }
      }
      output.owners.push_back(std::move(rows));
    }
    return output;
  }

  struct SavedLayerPolicy {
    int64_t clock{0};
    double demand_scale{1.0};
    std::vector<double> demand;
    std::vector<int64_t> last_seen;
    std::vector<double> interval_ema;
    int64_t call_index{-1};
    std::vector<int64_t> route_last_seen;
    std::vector<double> route_interval_ema;
    double miss_pressure{0.0};
    std::string markov;
  };

  struct SavedPolicy {
    uint64_t serial{0};
    std::vector<SavedLayerPolicy> layers;
    std::vector<double> retention;
    std::vector<uint64_t> last_access;
  };

  template <typename T>
  static void append_policy(std::string& output, const T& value) {
    output.append(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  template <typename T>
  static void append_policy_vector(
      std::string& output,
      const std::vector<T>& values) {
    output.append(
        reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(T));
  }

  template <typename T>
  static T read_policy(const std::string& payload, size_t& offset) {
    if (offset > payload.size() || payload.size() - offset < sizeof(T)) {
      throw std::invalid_argument(
          "ExpertSSD global-pool policy snapshot is truncated");
    }
    T value;
    std::memcpy(&value, payload.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
  }

  template <typename T>
  static std::vector<T> read_policy_vector(
      const std::string& payload,
      size_t& offset,
      size_t count) {
    if (count > (payload.size() - std::min(offset, payload.size())) / sizeof(T)) {
      throw std::invalid_argument(
          "ExpertSSD global-pool policy vector is truncated");
    }
    std::vector<T> values(count);
    if (count != 0) {
      std::memcpy(
          values.data(), payload.data() + offset, count * sizeof(T));
    }
    offset += count * sizeof(T);
    return values;
  }

  SavedPolicy parse_policy_snapshot(const std::string& payload) const {
    if (payload.size() < 5 || payload.compare(0, 5, "DSGP1") != 0) {
      throw std::invalid_argument(
          "ExpertSSD global-pool policy snapshot has wrong format");
    }
    size_t offset = 5;
    const auto expert_count = read_policy<uint32_t>(payload, offset);
    const auto layer_count = read_policy<uint32_t>(payload, offset);
    const auto shared_capacity = read_policy<uint32_t>(payload, offset);
    if (expert_count != expert_count_ || layer_count != layers_.size() ||
        shared_capacity != shared_capacity_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool policy dimensions are incompatible");
    }
    SavedPolicy output;
    output.serial = read_policy<uint64_t>(payload, offset);
    output.layers.reserve(layers_.size());
    for (const auto& current : layers_) {
      const auto layer = read_policy<int64_t>(payload, offset);
      const auto capacity = read_policy<uint32_t>(payload, offset);
      if (layer != current.layer || capacity != current.capacity) {
        throw std::invalid_argument(
            "ExpertSSD global-pool policy layer geometry differs");
      }
      SavedLayerPolicy saved;
      saved.clock = read_policy<int64_t>(payload, offset);
      saved.demand_scale = read_policy<double>(payload, offset);
      saved.demand = read_policy_vector<double>(
          payload, offset, expert_count_);
      saved.last_seen = read_policy_vector<int64_t>(
          payload, offset, expert_count_);
      saved.interval_ema = read_policy_vector<double>(
          payload, offset, expert_count_);
      saved.call_index = read_policy<int64_t>(payload, offset);
      saved.route_last_seen = read_policy_vector<int64_t>(
          payload, offset, expert_count_);
      saved.route_interval_ema = read_policy_vector<double>(
          payload, offset, expert_count_);
      saved.miss_pressure = read_policy<double>(payload, offset);
      const auto markov_size = read_policy<uint64_t>(payload, offset);
      if (markov_size > payload.size() - std::min(offset, payload.size())) {
        throw std::invalid_argument(
            "ExpertSSD global-pool Markov snapshot is truncated");
      }
      saved.markov.assign(payload.data() + offset, markov_size);
      offset += markov_size;

      if (saved.clock < 0 || !std::isfinite(saved.demand_scale) ||
          saved.demand_scale <= 0.0 || saved.call_index < -1 ||
          !std::isfinite(saved.miss_pressure) || saved.miss_pressure < 0.0 ||
          saved.miss_pressure > 1.0 ||
          std::any_of(saved.demand.begin(), saved.demand.end(), [](double value) {
            return !std::isfinite(value) || value < 0.0;
          }) ||
          std::any_of(
              saved.interval_ema.begin(),
              saved.interval_ema.end(),
              [](double value) {
                return !std::isfinite(value) || value <= 0.0;
              }) ||
          std::any_of(
              saved.route_interval_ema.begin(),
              saved.route_interval_ema.end(),
              [](double value) {
                return !std::isfinite(value) || value <= 0.0;
              }) ||
          std::any_of(
              saved.last_seen.begin(),
              saved.last_seen.end(),
              [&saved](int64_t value) {
                return value < -1 || value > saved.clock;
              }) ||
          std::any_of(
              saved.route_last_seen.begin(),
              saved.route_last_seen.end(),
              [&saved](int64_t value) {
                return value < -1 || value > saved.call_index;
              })) {
        throw std::invalid_argument(
            "ExpertSSD global-pool policy values are invalid");
      }
      ExpertSSDMarkovState markov_probe(expert_count_, markov_history_);
      markov_probe.restore(saved.markov);
      output.layers.push_back(std::move(saved));
    }
    const auto key_count = layers_.size() * expert_count_;
    output.retention = read_policy_vector<double>(
        payload, offset, key_count);
    output.last_access = read_policy_vector<uint64_t>(
        payload, offset, key_count);
    if (offset != payload.size() ||
        std::any_of(
            output.retention.begin(),
            output.retention.end(),
            [](double value) {
              return !std::isfinite(value) || value < 0.0;
            }) ||
        std::any_of(
            output.last_access.begin(),
            output.last_access.end(),
            [&output](uint64_t value) { return value > output.serial; })) {
      throw std::invalid_argument(
          "ExpertSSD global-pool policy tail is invalid");
    }
    return output;
  }

  struct LayerState {
    int64_t layer{-1};
    size_t capacity{0};
    std::vector<int32_t> expert_to_row;
    std::vector<int32_t> free_rows;
    std::vector<int32_t> down_expert_to_row;
    std::vector<int32_t> free_down_rows;
    std::vector<double> legacy_retention;
    std::shared_ptr<ExpertSSDMarkovState> markov;
    int64_t clock{0};
    double demand_scale{1.0};
    std::vector<double> demand;
    std::vector<int64_t> last_seen;
    std::vector<double> interval_ema;
    int64_t call_index{-1};
    std::vector<int64_t> route_last_seen;
    std::vector<double> route_interval_ema;
    double miss_pressure{0.0};
  };

  struct Victim {
    bool valid{false};
    bool shared{false};
    size_t layer_index{0};
    int64_t expert{-1};
  };

  void require_private_api() const {
    if (!legacy_private_ || layers_.size() != 1 || shared_capacity_ != 0) {
      throw std::invalid_argument("legacy ExpertSSD API requires one private bank");
    }
  }

  void require_identity() const {
    if (role_.empty()) throw std::invalid_argument("ExpertSSD reservation requires a role/layer identity");
  }

  void require_quiescent() const {
    if (!reservations_.empty()) throw std::invalid_argument("ExpertSSD has in-flight reservations");
  }

  LayerState& layer_state(int64_t layer) {
    const auto iterator = layer_lookup_.find(layer);
    if (iterator == layer_lookup_.end()) {
      throw std::invalid_argument(
          "ExpertSSD global-pool layer is unmanaged");
    }
    return layers_[iterator->second];
  }

  size_t key(size_t layer_index, int64_t expert) const {
    return layer_index * expert_count_ + static_cast<size_t>(expert);
  }

  void validate_expert(int64_t expert) const {
    if (expert < 0 || static_cast<size_t>(expert) >= expert_count_) {
      throw std::invalid_argument(
          "ExpertSSD global-pool expert id is out of range");
    }
  }

  template <typename T>
  void flatten_typed(
      const mx::array& indices,
      std::vector<int64_t>& output) const {
    const auto* data = indices.data<T>();
    const auto shape = indices.shape();
    const auto strides = indices.strides();
    output.reserve(indices.size());
    std::vector<size_t> coordinates(shape.size(), 0);
    for (size_t flat = 0; flat < indices.size(); ++flat) {
      size_t physical = 0;
      for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        physical += coordinates[dimension] * strides[dimension];
      }
      output.push_back(static_cast<int64_t>(data[physical]));
      for (int dimension = static_cast<int>(shape.size()) - 1;
           dimension >= 0;
           --dimension) {
        coordinates[dimension] += 1;
        if (coordinates[dimension] <
            static_cast<size_t>(shape[dimension])) {
          break;
        }
        coordinates[dimension] = 0;
      }
    }
  }

  void flatten(const mx::array& indices, std::vector<int64_t>& output) const {
    if (indices.dtype() == mx::int32) {
      flatten_typed<int32_t>(indices, output);
    } else if (indices.dtype() == mx::int64) {
      flatten_typed<int64_t>(indices, output);
    } else if (indices.dtype() == mx::uint32) {
      flatten_typed<uint32_t>(indices, output);
    } else if (indices.dtype() == mx::uint64) {
      flatten_typed<uint64_t>(indices, output);
    } else {
      throw std::invalid_argument(
          "ExpertSSD global-pool indices must be integral");
    }
  }

  void decay_retention(LayerState& state) const {
    if (retention_decay_ == 0.0) {
      std::fill(state.demand.begin(), state.demand.end(), 0.0);
      state.demand_scale = 1.0;
      return;
    }
    state.demand_scale *= retention_decay_;
    if (state.demand_scale < 1e-6) {
      for (auto& demand : state.demand) {
        demand *= state.demand_scale;
      }
      state.demand_scale = 1.0;
    }
  }

  void record_lhd_accesses(
      LayerState& state,
      const std::vector<int64_t>& experts,
      const std::vector<int32_t>& counts) const {
    for (size_t index = 0; index < experts.size(); ++index) {
      const auto expert = experts[index];
      ++state.clock;
      const auto previous = state.last_seen[expert];
      if (previous < 0) {
        state.interval_ema[expert] = initial_interval_;
      } else {
        const auto interval = std::max<int64_t>(1, state.clock - previous);
        state.interval_ema[expert] =
            interval_alpha_ * static_cast<double>(interval) +
            (1.0 - interval_alpha_) * state.interval_ema[expert];
      }
      state.last_seen[expert] = state.clock;
      state.demand[expert] +=
          static_cast<double>(counts[index]) / state.demand_scale;
    }
  }

  void record_route_intervals(
      LayerState& state,
      const std::vector<int64_t>& experts) const {
    for (const auto expert : experts) {
      const auto previous = state.route_last_seen[expert];
      if (previous < 0) {
        state.route_interval_ema[expert] = initial_interval_;
      } else {
        const auto interval = std::max<int64_t>(1, state.call_index - previous);
        state.route_interval_ema[expert] =
            interval_alpha_ * static_cast<double>(interval) +
            (1.0 - interval_alpha_) * state.route_interval_ema[expert];
      }
      state.route_last_seen[expert] = state.call_index;
    }
  }

  double score(
      const LayerState& state,
      int64_t expert,
      double markov_score) const {
    const auto demand = state.demand[expert] * state.demand_scale;
    const auto interval = std::max(1.0, state.interval_ema[expert]);
    const auto previous = state.last_seen[expert];
    const auto age = previous < 0
        ? int64_t{1}
        : std::max<int64_t>(1, state.clock - previous);
    const auto density = demand /
        (interval + age_weight_ * static_cast<double>(age));
    const auto route_previous = state.route_last_seen[expert];
    const auto remaining = route_previous < 0
        ? initial_interval_
        : std::max(
            1.0,
            static_cast<double>(route_previous) +
                state.route_interval_ema[expert] -
                static_cast<double>(state.call_index));
    const auto markov_confidence =
        1.0 + markov_boost_ * markov_score;
    const auto predicted_remaining = remaining / markov_confidence;
    const auto deadline_discount = std::pow(
        predicted_remaining, deadline_weight_);
    return density * markov_confidence *
        (1.0 + miss_pressure_weight_ * state.miss_pressure) /
        deadline_discount;
  }

  void refresh_layer_retention(
      size_t layer_index,
      const LayerState& state,
      const std::vector<float>& markov_scores) {
    for (size_t expert = 0; expert < expert_count_; ++expert) {
      if (state.expert_to_row[expert] >= 0 ||
          shared_rows_[key(layer_index, expert)] >= 0) {
        retention_[key(layer_index, expert)] = score(
            state, static_cast<int64_t>(expert), markov_scores[expert]);
      }
    }
  }

  void touch(size_t layer_index, int64_t expert) {
    ++serial_;
    last_access_[key(layer_index, expert)] = serial_;
  }

  bool worse(const Victim& lhs, const Victim& rhs) const {
    if (!lhs.valid) {
      return false;
    }
    if (!rhs.valid) {
      return true;
    }
    const auto lhs_key = key(lhs.layer_index, lhs.expert);
    const auto rhs_key = key(rhs.layer_index, rhs.expert);
    if (retention_[lhs_key] != retention_[rhs_key]) {
      return retention_[lhs_key] < retention_[rhs_key];
    }
    if (last_access_[lhs_key] != last_access_[rhs_key]) {
      return last_access_[lhs_key] < last_access_[rhs_key];
    }
    const auto lhs_layer = layers_[lhs.layer_index].layer;
    const auto rhs_layer = layers_[rhs.layer_index].layer;
    return lhs_layer != rhs_layer
        ? lhs_layer > rhs_layer
        : lhs.expert > rhs.expert;
  }

  Victim choose_private_victim(
      size_t layer_index,
      const LayerState& state,
      const std::vector<uint8_t>& protected_experts) const {
    Victim best;
    for (size_t expert = 0; expert < expert_count_; ++expert) {
      if (state.expert_to_row[expert] < 0 || protected_experts[expert] ||
          reserved_keys_[key(layer_index, expert)]) {
        continue;
      }
      Victim candidate{
          true, false, layer_index, static_cast<int64_t>(expert)};
      if (!best.valid || worse(candidate, best)) {
        best = candidate;
      }
    }
    return best;
  }

  Victim choose_shared_victim(
      size_t requesting_layer_index,
      const std::vector<uint8_t>& protected_experts) const {
    Victim best;
    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
      for (size_t expert = 0; expert < expert_count_; ++expert) {
        if (shared_rows_[key(layer_index, expert)] < 0 ||
            reserved_keys_[key(layer_index, expert)] ||
            (layer_index == requesting_layer_index &&
             protected_experts[expert])) {
          continue;
        }
        Victim candidate{
            true, true, layer_index, static_cast<int64_t>(expert)};
        if (!best.valid || worse(candidate, best)) {
          best = candidate;
        }
      }
    }
    return best;
  }

  int32_t remove(const Victim& victim) {
    const auto resident_key = key(victim.layer_index, victim.expert);
    int32_t row = -1;
    if (victim.shared) {
      row = shared_rows_[resident_key];
      shared_rows_[resident_key] = -1;
    } else {
      auto& state = layers_[victim.layer_index];
      row = state.expert_to_row[victim.expert];
      state.expert_to_row[victim.expert] = -1;
    }
    retention_[resident_key] = 0.0;
    last_access_[resident_key] = 0;
    if (row < 0) {
      throw std::invalid_argument(
          "ExpertSSD global-pool victim row is missing");
    }
    return row;
  }

  size_t expert_count_;
  mutable std::recursive_mutex mutex_;
  std::string role_;
  std::vector<int64_t> logical_layers_;
  std::vector<size_t> reserved_keys_;
  std::unordered_map<uint64_t, std::vector<size_t>> reservations_;
  inline static std::atomic<uint64_t> next_reservation_{0};
  size_t shared_capacity_;
  double retention_decay_;
  double interval_alpha_;
  double initial_interval_;
  double age_weight_;
  double deadline_weight_;
  double markov_boost_;
  size_t markov_history_;
  double miss_pressure_alpha_;
  double miss_pressure_weight_;
  std::vector<LayerState> layers_;
  std::unordered_map<int64_t, size_t> layer_lookup_;
  std::vector<int32_t> shared_rows_;
  std::vector<int32_t> shared_free_rows_;
  std::vector<double> retention_;
  std::vector<uint64_t> last_access_;
  uint64_t serial_{0};
  size_t hits_{0};
  size_t misses_{0};
  size_t evictions_{0};
  size_t private_evictions_{0};
  size_t shared_evictions_{0};
  bool shared_lease_active_{false};
  bool legacy_private_{false};
};

using ExpertSSDRouteCacheState = ExpertSSDCacheState;
using ExpertSSDGlobalPoolState = ExpertSSDCacheState;

mx::Dtype scalar_to_dtype(Scalar s) {
  if (auto pv = std::get_if<int64_t>(&s); pv) {
    return (*pv > std::numeric_limits<int>::max() ||
            *pv < std::numeric_limits<int>::min())
        ? mx::int64
        : mx::int32;
  } else if (std::holds_alternative<double>(s)) {
    return mx::float32;
  } else {
    return mx::bool_;
  }
}

double scalar_to_double(Scalar s) {
  if (auto pv = std::get_if<int64_t>(&s); pv) {
    return static_cast<double>(*pv);
  } else if (auto pv = std::get_if<double>(&s); pv) {
    return *pv;
  } else {
    return static_cast<double>(std::get<bool>(s));
  }
}

mx::Shape to_shape(const nb::object& shape) {
  if (nb::isinstance<nb::int_>(shape)) {
    return {check_shape_dim(nb::cast<int64_t>(shape))};
  }
  return nb::cast<mx::Shape>(shape);
}

mx::Dtype dtype_from_official_direct_string(const std::string& value) {
  if (value == "BF16") {
    return mx::bfloat16;
  } else if (value == "F16") {
    return mx::float16;
  } else if (value == "F32") {
    return mx::float32;
  } else if (value == "I8") {
    return mx::int8;
  } else if (value == "I32") {
    return mx::int32;
  } else if (value == "I64") {
    return mx::int64;
  } else if (value == "U8") {
    return mx::uint8;
  } else if (value == "U16") {
    return mx::uint16;
  } else if (value == "U32") {
    return mx::uint32;
  } else if (value == "U64") {
    return mx::uint64;
  }
  throw std::invalid_argument(
      "[_open_expert_safetensors_direct] unsupported destination dtype " +
      value);
}

struct OfficialDirectDestinationRows {
  std::vector<char*> bases;
  std::vector<size_t> row_nbytes;
  size_t capacity{0};
};

OfficialDirectDestinationRows validate_official_direct_destinations(
    const std::shared_ptr<mx::ExpertSafetensorsDirect>& direct,
    std::vector<mx::array>& destinations) {
  const auto& specs = direct->specs();
  if (destinations.size() != specs.size()) {
    std::ostringstream message;
    message << "[_expert_ssd_direct_load_into] expected " << specs.size()
            << " destination arrays, got " << destinations.size();
    throw std::invalid_argument(message.str());
  }
  OfficialDirectDestinationRows rows;
  rows.bases.reserve(destinations.size());
  rows.row_nbytes.reserve(destinations.size());
  for (size_t index = 0; index < destinations.size(); ++index) {
    auto& destination = destinations[index];
    const auto& spec = specs[index];
    if (destination.ndim() < 1 || destination.dtype() != spec.dtype ||
        static_cast<size_t>(destination.ndim()) != spec.shape.size() + 1 ||
        !std::equal(
            spec.shape.begin(),
            spec.shape.end(),
            destination.shape().begin() + 1)) {
      throw std::invalid_argument(
          "[_expert_ssd_direct_load_into] destination layout mismatch for " +
          spec.name);
    }
    const auto capacity = static_cast<size_t>(destination.shape(0));
    if (capacity == 0) {
      throw std::invalid_argument(
          "[_expert_ssd_direct_load_into] destination capacity is zero");
    }
    if (index == 0) {
      rows.capacity = capacity;
    } else if (capacity != rows.capacity) {
      throw std::invalid_argument(
          "[_expert_ssd_direct_load_into] destinations must share capacity");
    }
    if (!destination.is_available()) {
      destination.eval();
    }
    if (!destination.flags().row_contiguous) {
      throw std::invalid_argument(
          "[_expert_ssd_direct_load_into] destinations must be row-contiguous");
    }
    rows.bases.push_back(destination.data<char>());
    rows.row_nbytes.push_back(destination.nbytes() / capacity);
  }
  return rows;
}

struct ScaleXDestinationRows {
  std::array<char*, 3> bases{};
  std::array<size_t, 3> row_nbytes{};
  size_t capacity{0};
};

ScaleXDestinationRows validate_scalex_destinations(
    std::vector<mx::array>& destinations,
    mx::Dtype dtype,
    const std::array<size_t, 3>* expected_row_nbytes = nullptr) {
  if (destinations.size() != 3) {
    throw std::invalid_argument(
        "[ScaleXModeADirect] expected three destination arrays");
  }
  ScaleXDestinationRows rows;
  for (size_t tensor = 0; tensor < destinations.size(); ++tensor) {
    auto& destination = destinations[tensor];
    if (destination.ndim() < 1 || destination.dtype() != dtype) {
      throw std::invalid_argument(
          "[ScaleXModeADirect] destination dtype or rank changed");
    }
    const auto capacity = static_cast<size_t>(destination.shape(0));
    if (capacity == 0 || (tensor != 0 && capacity != rows.capacity)) {
      throw std::invalid_argument(
          "[ScaleXModeADirect] incompatible destination capacities");
    }
    rows.capacity = capacity;
    if (!destination.is_available()) {
      destination.eval();
    }
    if (!destination.flags().row_contiguous) {
      throw std::invalid_argument(
          "[ScaleXModeADirect] destinations must be row-contiguous");
    }
    rows.row_nbytes[tensor] = destination.nbytes() / capacity;
    if (expected_row_nbytes != nullptr &&
        rows.row_nbytes[tensor] != (*expected_row_nbytes)[tensor]) {
      throw std::invalid_argument(
          "[ScaleXModeADirect] destination row size changed");
    }
    rows.bases[tensor] = destination.data<char>();
  }
  return rows;
}

template <typename T>
nb::tuple official_direct_route_plan_impl(mx::array& indices) {
  const auto size = indices.size();
  const auto* data = indices.data<T>();
  const auto shape = indices.shape();
  const auto strides = indices.strides();
  std::vector<int32_t> remapped;
  remapped.reserve(size);
  std::vector<int64_t> unique;
  unique.reserve(std::min<size_t>(size, 256));
  std::unordered_map<int64_t, int32_t> mapping;
  std::unordered_map<int64_t, int32_t> counts;
  mapping.reserve(std::min<size_t>(size, 512));
  counts.reserve(std::min<size_t>(size, 512));

  std::vector<size_t> coordinates(shape.size(), 0);
  for (size_t flat = 0; flat < size; ++flat) {
    size_t physical = 0;
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
      physical += coordinates[dimension] * strides[dimension];
    }
    const auto expert = static_cast<int64_t>(data[physical]);
    auto [iterator, inserted] =
        mapping.emplace(expert, static_cast<int32_t>(unique.size()));
    if (inserted) {
      unique.push_back(expert);
    }
    remapped.push_back(iterator->second);
    counts[expert] += 1;
    for (int dimension = static_cast<int>(shape.size()) - 1;
         dimension >= 0;
         --dimension) {
      coordinates[dimension] += 1;
      if (coordinates[dimension] <
          static_cast<size_t>(shape[dimension])) {
        break;
      }
      coordinates[dimension] = 0;
    }
  }

  nb::list unique_output;
  nb::dict counts_output;
  for (auto expert : unique) {
    unique_output.append(nb::cast(expert));
    counts_output[nb::cast(expert)] = nb::cast(counts[expert]);
  }
  auto remapped_array =
      mx::array(remapped.begin(), indices.shape(), indices.dtype());
  return nb::make_tuple(unique_output, counts_output, remapped_array);
}

nb::tuple official_direct_route_plan(mx::array indices) {
  if (indices.ndim() == 0) {
    throw std::invalid_argument(
        "[_expert_ssd_route_plan] indices must have at least one dimension");
  }
  // A preceding ExpertSSD graph may be waiting on an IO-completion event
  // published by a Python executor callback. Do not hold the GIL while this
  // upstream graph completes or the callback cannot signal its Metal event.
  {
    nb::gil_scoped_release release;
    indices.eval();
  }
  for (auto stride : indices.strides()) {
    if (stride < 0) {
      throw std::invalid_argument(
          "[_expert_ssd_route_plan] negative strides are not supported");
    }
  }
  if (indices.dtype() == mx::int32) {
    return official_direct_route_plan_impl<int32_t>(indices);
  } else if (indices.dtype() == mx::int64) {
    return official_direct_route_plan_impl<int64_t>(indices);
  } else if (indices.dtype() == mx::uint32) {
    return official_direct_route_plan_impl<uint32_t>(indices);
  } else if (indices.dtype() == mx::uint64) {
    return official_direct_route_plan_impl<uint64_t>(indices);
  }
  throw std::invalid_argument(
      "[_expert_ssd_route_plan] indices must be int32, int64, uint32, or uint64");
}

double expert_ssd_internal_to_replica_speed_ratio() {
  static const double ratio = [] {
    const char* raw =
        std::getenv("MLX_EXPERT_SSD_INTERNAL_TO_REPLICA_SPEED_RATIO");
    if (raw == nullptr || *raw == '\0') {
      return 1.0;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(raw, &end);
    if (errno != 0 || end == raw || *end != '\0' || !std::isfinite(parsed) ||
        parsed <= 0.0) {
      throw std::invalid_argument(
          "MLX_EXPERT_SSD_INTERNAL_TO_REPLICA_SPEED_RATIO must be a "
          "positive finite number");
    }
    return parsed;
  }();
  return ratio;
}

size_t expert_ssd_internal_stripe_bytes_for_ratio(
    size_t total,
    long double ratio) {
  if (total < 2) {
    return total;
  }
  const long double share = ratio / (ratio + 1.0L);
  const size_t split = static_cast<size_t>(
      std::llround(static_cast<long double>(total) * share));
  return std::clamp(split, size_t{1}, total - 1);
}

size_t expert_ssd_internal_stripe_bytes(size_t total) {
  return expert_ssd_internal_stripe_bytes_for_ratio(
      total, expert_ssd_internal_to_replica_speed_ratio());
}

// Opt-in tracing only: one disjoint record per native worker task. No Python
// callbacks, shared trace lock, or clock reads on untraced refills.
struct ScaleXReadTrace {
  uint64_t started_ns{0}, finished_ns{0}, thread_id{0}, bytes{0};
  size_t item{0}, replica{0};
  bool success{false};
};

static uint64_t scalex_trace_clock_ns() {
  static const mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t value;
    mach_timebase_info(&value);
    return value;
  }();
  return (static_cast<__uint128_t>(mach_absolute_time()) * tb.numer) / tb.denom;
}

struct ScaleXTraceScope {
  ScaleXReadTrace* record;
  explicit ScaleXTraceScope(ScaleXReadTrace* record, size_t item, size_t replica)
      : record(record) {
    if (record) {
      record->item = item;
      record->replica = replica;
      pthread_threadid_np(nullptr, &record->thread_id);
      record->started_ns = scalex_trace_clock_ns();
    }
  }
  ~ScaleXTraceScope() {
    if (record) record->finished_ns = scalex_trace_clock_ns();
  }
};

template <class State>
static nb::list scalex_read_trace(const std::shared_ptr<State>& state) {
  if (!state) throw std::invalid_argument("ScaleX trace needs a batch state");
  std::lock_guard<std::mutex> lock(state->completion_mutex);
  if (!state->complete) throw std::runtime_error("ScaleX batch is not complete");
  nb::list result;
  for (const auto& r : state->read_trace) {
    if (!r.started_ns) continue;
    nb::dict row;
    row["item"] = r.item;
    row["expert"] = state->expert_ids[r.item];
    row["replica"] = r.replica;
    row["started_ns"] = r.started_ns;
    row["finished_ns"] = r.finished_ns;
    row["thread_id"] = r.thread_id;
    row["bytes"] = r.bytes;
    row["success"] = r.success;
    result.append(row);
  }
  return result;
}

struct ExpertSSDAsyncBatchState {
  const std::chrono::steady_clock::time_point started{
      std::chrono::steady_clock::now()};
  double elapsed_seconds{0.0};
  ExpertSSDAsyncBatchState(
      std::shared_ptr<mx::ExpertSafetensorsDirect> raw_direct,
      std::vector<size_t> expert_ids,
      std::vector<size_t> gate_up_slots,
      std::vector<size_t> down_slots,
      std::vector<mx::array> raw_destinations,
      OfficialDirectDestinationRows raw_rows,
      size_t worker_count,
      bool interactive_qos,
      std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
      uint64_t event_value)
      : expert_ids(std::move(expert_ids)),
        gate_up_slots(std::move(gate_up_slots)),
        down_slots(std::move(down_slots)),
        record_destinations(0),
        gate_destinations(0),
        down_destinations(0),
        up_destinations(0),
        worker_count(worker_count),
        interactive_qos(interactive_qos),
        event_state(std::move(event_state)),
        event_value(event_value),
        raw_direct(std::move(raw_direct)),
        raw_destinations(std::move(raw_destinations)),
        raw_rows(std::move(raw_rows)) {}

  ExpertSSDAsyncBatchState(
      std::shared_ptr<mx::ScaleXModeADirect> direct,
      std::vector<size_t> expert_ids,
      std::vector<size_t> gate_up_slots,
      std::vector<size_t> down_slots,
      mx::array record_destinations,
      mx::array gate_destinations,
      mx::array down_destinations,
      mx::array up_destinations,
      size_t worker_count,
      bool interactive_qos,
      std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
      uint64_t event_value,
      uint64_t wait_value = 0)
      : direct(std::move(direct)),
        expert_ids(std::move(expert_ids)),
        gate_up_slots(std::move(gate_up_slots)),
        down_slots(std::move(down_slots)),
        record_destinations(std::move(record_destinations)),
        gate_destinations(std::move(gate_destinations)),
        down_destinations(std::move(down_destinations)),
        up_destinations(std::move(up_destinations)),
        worker_count(worker_count),
        interactive_qos(interactive_qos),
        event_state(std::move(event_state)),
        event_value(event_value),
        wait_value(wait_value) {}

  std::shared_ptr<mx::ScaleXModeADirect> direct;
  std::vector<size_t> expert_ids;
  std::vector<size_t> gate_up_slots;
  std::vector<size_t> down_slots;
  std::vector<ScaleXReadTrace> read_trace;
  mx::array record_destinations;
  mx::array gate_destinations;
  mx::array down_destinations;
  mx::array up_destinations;
  char* record_base{nullptr};
  char* gate_base{nullptr};
  char* down_base{nullptr};
  char* up_base{nullptr};
  size_t record_row_nbytes{0};
  std::array<size_t, 3> row_nbytes{};
  size_t worker_count{0};
  bool interactive_qos{false};
  std::shared_ptr<mx::ExpertSSDIoEventState> event_state;
  uint64_t event_value{0};
  uint64_t wait_value{0};
  std::atomic<size_t> next{0};
  std::mutex error_mutex;
  std::exception_ptr error;
  std::mutex completion_mutex;
  std::condition_variable completion_condition;
  bool complete{false};
  std::shared_ptr<mx::ExpertSafetensorsDirect> raw_direct;
  std::vector<mx::array> raw_destinations;
  OfficialDirectDestinationRows raw_rows;
};

struct ScaleXModeAAsyncBatchState {
  ScaleXModeAAsyncBatchState(
      std::shared_ptr<mx::ScaleXModeADirect> direct,
      std::vector<size_t> expert_ids,
      std::vector<size_t> slots,
      std::vector<mx::array> scale_destinations,
      std::vector<mx::array> weight_destinations,
      size_t worker_count,
      bool interactive_qos,
      std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
      uint64_t wait_value,
      uint64_t event_value)
      : direct(std::move(direct)),
        expert_ids(std::move(expert_ids)),
        slots(std::move(slots)),
        scale_destinations(std::move(scale_destinations)),
        weight_destinations(std::move(weight_destinations)),
        worker_count(worker_count),
        interactive_qos(interactive_qos),
        event_state(std::move(event_state)),
        wait_value(wait_value),
        event_value(event_value) {}

  std::shared_ptr<mx::ScaleXModeADirect> direct;
  std::vector<size_t> expert_ids;
  std::vector<size_t> slots;
  std::vector<mx::array> scale_destinations;
  std::vector<mx::array> weight_destinations;
  std::array<char*, 3> scale_bases{};
  std::array<char*, 3> weight_bases{};
  std::array<size_t, 3> scale_row_nbytes{};
  std::array<size_t, 3> weight_row_nbytes{};
  size_t worker_count{0};
  bool interactive_qos{false};
  std::shared_ptr<mx::ExpertSSDIoEventState> event_state;
  uint64_t wait_value{0};
  uint64_t event_value{0};
  std::atomic<size_t> next{0};
  std::mutex error_mutex;
  std::exception_ptr error;
  std::mutex completion_mutex;
  std::condition_variable completion_condition;
  bool complete{false};
};

void scalex_mode_a_async_batch_worker(void* raw, size_t) {
  auto* state = static_cast<ScaleXModeAAsyncBatchState*>(raw);
  while (true) {
    const size_t item = state->next.fetch_add(1);
    if (item >= state->expert_ids.size()) {
      return;
    }
    try {
      const auto slot = state->slots[item];
      std::array<char*, 3> scale_pointers{};
      std::array<char*, 3> weight_pointers{};
      for (size_t tensor = 0; tensor < 3; ++tensor) {
        scale_pointers[tensor] = state->scale_bases[tensor] +
            slot * state->scale_row_nbytes[tensor];
        weight_pointers[tensor] = state->weight_bases[tensor] +
            slot * state->weight_row_nbytes[tensor];
      }
      state->direct->load_expert_into(
          state->expert_ids[item],
          scale_pointers,
          state->scale_row_nbytes,
          weight_pointers,
          state->weight_row_nbytes);
    } catch (...) {
      std::lock_guard<std::mutex> lock(state->error_mutex);
      if (!state->error) {
        state->error = std::current_exception();
      }
      return;
    }
  }
}

void scalex_mode_a_async_batch_run(void* raw) {
  std::unique_ptr<std::shared_ptr<ScaleXModeAAsyncBatchState>> owner(
      static_cast<std::shared_ptr<ScaleXModeAAsyncBatchState>*>(raw));
  auto state = *owner;
  if (state->wait_value != 0) {
    mx::expert_ssd_io_event_wait(state->event_state, state->wait_value);
  }
  const auto queue = dispatch_get_global_queue(
      state->interactive_qos ? QOS_CLASS_USER_INTERACTIVE
                             : QOS_CLASS_USER_INITIATED,
      0);
  dispatch_apply_f(
      std::min(state->worker_count, state->expert_ids.size()),
      queue,
      state.get(),
      scalex_mode_a_async_batch_worker);
  {
    std::lock_guard<std::mutex> lock(state->completion_mutex);
    state->complete = true;
  }
  state->completion_condition.notify_all();
  // Release dependent Metal work even when one native read failed. The
  // explicit host waiter rethrows before this destination bank is reused.
  mx::expert_ssd_io_event_signal(state->event_state, state->event_value);
}

void scalex_async_batch_worker(void* raw, size_t) {
  auto* state = static_cast<ExpertSSDAsyncBatchState*>(raw);
  const bool striped = state->direct && state->direct->replica_count() == 2;
  const size_t task_count = state->expert_ids.size() * (striped ? 2 : 1);
  while (true) {
    const size_t task = state->next.fetch_add(1);
    if (task >= task_count) {
      return;
    }
    const size_t item = striped ? task / 2 : task;
    const size_t replica = striped ? task % 2 : 0;
    ScaleXTraceScope timing(
        state->read_trace.empty() ? nullptr : &state->read_trace[task],
        item, replica);
    try {
      if (state->raw_direct) {
        std::vector<char*> pointers(state->raw_rows.bases.size());
        const auto& specs = state->raw_direct->specs();
        for (size_t tensor = 0; tensor < pointers.size(); ++tensor) {
          const bool down = specs[tensor].name.rfind("down_proj.", 0) == 0;
          const auto slot = down ? state->down_slots[item] : state->gate_up_slots[item];
          pointers[tensor] = state->raw_rows.bases[tensor] +
              slot * state->raw_rows.row_nbytes[tensor];
        }
        state->raw_direct->load_ordered_into(
            state->expert_ids[item], pointers, state->raw_rows.row_nbytes);
        if (timing.record) {
          timing.record->success = true;
          for (auto bytes : state->raw_rows.row_nbytes) timing.record->bytes += bytes;
        }
        continue;
      }
      const std::array<char*, 3> pointers{
          state->gate_base +
              state->gate_up_slots[item] * state->row_nbytes[0],
          state->down_base + state->down_slots[item] * state->row_nbytes[1],
          state->up_base +
              state->gate_up_slots[item] * state->row_nbytes[2]};
      char* record = state->record_base +
          state->gate_up_slots[item] * state->record_row_nbytes;
      if (striped) {
        const size_t total = state->direct->compressed_expert_nbytes(
            state->expert_ids[item], state->row_nbytes);
        const size_t split = expert_ssd_internal_stripe_bytes(total);
        const size_t begin = replica == 0 ? 0 : split;
        const size_t end = replica == 0 ? split : total;
        if (timing.record) timing.record->bytes = end - begin;
        state->direct->load_compressed_expert_slice_into_from(
            replica,
            state->expert_ids[item],
            begin,
            end,
            record,
            state->record_row_nbytes,
            pointers,
            state->row_nbytes);
      } else {
        if (timing.record) {
          timing.record->bytes = state->direct->compressed_expert_nbytes(
              state->expert_ids[item], state->row_nbytes);
        }
        state->direct->load_compressed_expert_into(
            state->expert_ids[item],
            record,
            state->record_row_nbytes,
            pointers,
            state->row_nbytes);
      }
      if (timing.record) timing.record->success = true;
    } catch (...) {
      std::lock_guard<std::mutex> lock(state->error_mutex);
      if (!state->error) {
        state->error = std::current_exception();
      }
      return;
    }
  }
}

void scalex_async_batch_run(void* raw) {
  std::unique_ptr<std::shared_ptr<ExpertSSDAsyncBatchState>> owner(
      static_cast<std::shared_ptr<ExpertSSDAsyncBatchState>*>(raw));
  auto state = *owner;
  if (state->wait_value != 0) {
    mx::expert_ssd_io_event_wait(state->event_state, state->wait_value);
  }
  const auto queue = dispatch_get_global_queue(
      state->interactive_qos ? QOS_CLASS_USER_INTERACTIVE
                             : QOS_CLASS_USER_INITIATED,
      0);
  dispatch_apply_f(
      std::min(
          state->worker_count,
          state->expert_ids.size() *
              (state->direct && state->direct->replica_count() == 2 ? 2 : 1)),
      queue,
      state.get(),
      scalex_async_batch_worker);
  if (state->direct && state->direct->replica_count() == 2) {
    try {
      {
        std::lock_guard<std::mutex> lock(state->error_mutex);
        if (state->error) {
          throw std::runtime_error(
              "[ExpertSSDAsyncBatchState] striped refill failed");
        }
      }
      for (size_t item = 0; item < state->expert_ids.size(); ++item) {
        state->direct->finalize_compressed_expert(
            state->expert_ids[item],
            state->record_base +
                state->gate_up_slots[item] * state->record_row_nbytes,
            state->record_row_nbytes);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(state->error_mutex);
      if (!state->error) {
        state->error = std::current_exception();
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(state->completion_mutex);
    state->elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - state->started).count();
    state->complete = true;
  }
  state->completion_condition.notify_all();
  // Always release Metal, including on an I/O exception. The cleanup waiter
  // rethrows the captured exception before the cache can be reused.
  mx::expert_ssd_io_event_signal(state->event_state, state->event_value);
}

struct ScaleXTwoBankAsyncBatchState {
  const std::chrono::steady_clock::time_point started{
      std::chrono::steady_clock::now()};
  double elapsed_seconds{0.0};
  ScaleXTwoBankAsyncBatchState(
      std::shared_ptr<mx::ScaleXModeADirect> direct,
      std::vector<size_t> expert_ids,
      std::vector<uint8_t> shared_bank,
      std::vector<size_t> rows,
      std::array<mx::array, 4> private_destinations,
      std::array<mx::array, 4> shared_destinations,
      size_t worker_count,
      bool interactive_qos,
      std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
      uint64_t event_value)
      : direct(std::move(direct)),
        expert_ids(std::move(expert_ids)),
        shared_bank(std::move(shared_bank)),
        rows(std::move(rows)),
        private_destinations(std::move(private_destinations)),
        shared_destinations(std::move(shared_destinations)),
        worker_count(worker_count),
        interactive_qos(interactive_qos),
        event_state(std::move(event_state)),
        event_value(event_value) {}

  std::shared_ptr<mx::ScaleXModeADirect> direct;
  std::vector<size_t> expert_ids;
  std::vector<uint8_t> shared_bank;
  std::vector<size_t> rows;
  std::vector<ScaleXReadTrace> read_trace;
  std::array<mx::array, 4> private_destinations;
  std::array<mx::array, 4> shared_destinations;
  std::array<char*, 4> private_bases{};
  std::array<char*, 4> shared_bases{};
  size_t private_record_row_nbytes{0};
  size_t shared_record_row_nbytes{0};
  std::array<size_t, 3> row_nbytes{};
  size_t worker_count{0};
  bool interactive_qos{false};
  std::shared_ptr<mx::ExpertSSDIoEventState> event_state;
  uint64_t event_value{0};
  std::atomic<size_t> next{0};
  std::mutex error_mutex;
  std::exception_ptr error;
  std::mutex completion_mutex;
  std::condition_variable completion_condition;
  bool complete{false};
};

void scalex_two_bank_batch_worker(void* raw, size_t) {
  auto* state = static_cast<ScaleXTwoBankAsyncBatchState*>(raw);
  const bool striped = state->direct->replica_count() == 2;
  const size_t task_count = state->expert_ids.size() * (striped ? 2 : 1);
  while (true) {
    const size_t task = state->next.fetch_add(1);
    if (task >= task_count) {
      return;
    }
    const size_t item = striped ? task / 2 : task;
    const size_t replica = striped ? task % 2 : 0;
    ScaleXTraceScope timing(
        state->read_trace.empty() ? nullptr : &state->read_trace[task],
        item, replica);
    try {
      const auto& bases = state->shared_bank[item]
          ? state->shared_bases
          : state->private_bases;
      const auto record_row_nbytes = state->shared_bank[item]
          ? state->shared_record_row_nbytes
          : state->private_record_row_nbytes;
      const auto row = state->rows[item];
      const std::array<char*, 3> pointers{
          bases[1] + row * state->row_nbytes[0],
          bases[2] + row * state->row_nbytes[1],
          bases[3] + row * state->row_nbytes[2]};
      char* record = bases[0] + row * record_row_nbytes;
      if (striped) {
        const size_t total = state->direct->compressed_expert_nbytes(
            state->expert_ids[item], state->row_nbytes);
        const size_t split = expert_ssd_internal_stripe_bytes(total);
        const size_t begin = replica == 0 ? 0 : split;
        const size_t end = replica == 0 ? split : total;
        if (timing.record) timing.record->bytes = end - begin;
        state->direct->load_compressed_expert_slice_into_from(
            replica,
            state->expert_ids[item],
            begin,
            end,
            record,
            record_row_nbytes,
            pointers,
            state->row_nbytes);
      } else {
        if (timing.record) {
          timing.record->bytes = state->direct->compressed_expert_nbytes(
              state->expert_ids[item], state->row_nbytes);
        }
        state->direct->load_compressed_expert_into(
            state->expert_ids[item],
            record,
            record_row_nbytes,
            pointers,
            state->row_nbytes);
      }
      if (timing.record) timing.record->success = true;
    } catch (...) {
      std::lock_guard<std::mutex> lock(state->error_mutex);
      if (!state->error) {
        state->error = std::current_exception();
      }
      return;
    }
  }
}

void scalex_two_bank_batch_run(void* raw) {
  std::unique_ptr<std::shared_ptr<ScaleXTwoBankAsyncBatchState>> owner(
      static_cast<std::shared_ptr<ScaleXTwoBankAsyncBatchState>*>(raw));
  auto state = *owner;
  const auto queue = dispatch_get_global_queue(
      state->interactive_qos ? QOS_CLASS_USER_INTERACTIVE
                             : QOS_CLASS_USER_INITIATED,
      0);
  dispatch_apply_f(
      std::min(
          state->worker_count,
          state->expert_ids.size() *
              (state->direct->replica_count() == 2 ? 2 : 1)),
      queue,
      state.get(),
      scalex_two_bank_batch_worker);
  if (state->direct->replica_count() == 2) {
    try {
      {
        std::lock_guard<std::mutex> lock(state->error_mutex);
        if (state->error) {
          throw std::runtime_error(
              "[ScaleXTwoBankAsyncBatchState] striped refill failed");
        }
      }
      for (size_t item = 0; item < state->expert_ids.size(); ++item) {
        const auto& bases = state->shared_bank[item]
            ? state->shared_bases
            : state->private_bases;
        const size_t record_row_nbytes = state->shared_bank[item]
            ? state->shared_record_row_nbytes
            : state->private_record_row_nbytes;
        state->direct->finalize_compressed_expert(
            state->expert_ids[item],
            bases[0] + state->rows[item] * record_row_nbytes,
            record_row_nbytes);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(state->error_mutex);
      if (!state->error) {
        state->error = std::current_exception();
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(state->completion_mutex);
    state->elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - state->started).count();
    state->complete = true;
  }
  state->completion_condition.notify_all();
  mx::expert_ssd_io_event_signal(state->event_state, state->event_value);
}

template <typename State>
std::optional<double> expert_ssd_async_poll(const std::shared_ptr<State>& state) {
  if (!state) {
    throw std::invalid_argument("[_expert_ssd_async_poll] state required");
  }
  std::lock_guard<std::mutex> lock(state->completion_mutex);
  if (!state->complete) return std::nullopt;
  std::lock_guard<std::mutex> error_lock(state->error_mutex);
  if (state->error) std::rethrow_exception(state->error);
  return state->elapsed_seconds;
}

template <typename State>
void expert_ssd_async_wait(const std::shared_ptr<State>& state) {
  {
    std::lock_guard<std::mutex> lock(state->completion_mutex);
    if (state->complete) {
      std::lock_guard<std::mutex> error_lock(state->error_mutex);
      if (state->error) std::rethrow_exception(state->error);
      return;
    }
  }
  std::exception_ptr error;
  {
    nb::gil_scoped_release release;
    std::unique_lock<std::mutex> lock(state->completion_mutex);
    state->completion_condition.wait(lock, [&]() { return state->complete; });
    std::lock_guard<std::mutex> error_lock(state->error_mutex);
    error = state->error;
  }
  if (error) std::rethrow_exception(error);
}

void init_ops(nb::module_& m) {
  m.def("_expert_ssd_async_poll", &expert_ssd_async_poll<ExpertSSDAsyncBatchState>,
        "state"_a);
  m.def("_expert_ssd_async_poll", &expert_ssd_async_poll<ScaleXTwoBankAsyncBatchState>,
        "state"_a);
  m.def("_scalex_mode_b_async_trace", &scalex_read_trace<ExpertSSDAsyncBatchState>);
  m.def("_scalex_mode_b_two_bank_async_trace",
        &scalex_read_trace<ScaleXTwoBankAsyncBatchState>);
  nb::class_<mx::ExpertSafetensorsDirect>(m, "_ExpertSafetensorsDirect");
  nb::class_<mx::ScaleXPrefixStore>(m, "_ScaleXPrefixStore");
  nb::class_<mx::ScaleXModeADirect>(m, "_ScaleXModeADirect");
  nb::class_<mx::SafetensorsRowDirect>(m, "_SafetensorsRowDirect");
  nb::class_<mx::ExpertSSDIoEventState>(m, "_ExpertSSDIoEventState");
  nb::class_<ExpertSSDAsyncBatchState>(m, "_ExpertSSDAsyncBatchState");
  m.attr("_ScaleXAsyncBatchState") = m.attr("_ExpertSSDAsyncBatchState");
  nb::class_<ScaleXModeAAsyncBatchState>(
      m, "_ScaleXModeAAsyncBatchState");
  nb::class_<ScaleXTwoBankAsyncBatchState>(
      m, "_ScaleXTwoBankAsyncBatchState");
  nb::class_<ExpertSSDMarkovState>(m, "_ExpertSSDMarkovState");
  nb::class_<ExpertSSDCacheState>(m, "_ExpertSSDCacheState");
  m.attr("_ExpertSSDRouteCacheState") = m.attr("_ExpertSSDCacheState");
  m.attr("_ExpertSSDGlobalPoolState") = m.attr("_ExpertSSDCacheState");
  m.def("_expert_ssd_cache_set_identity", [](std::shared_ptr<ExpertSSDCacheState> state,
                                           const std::string& role, const std::vector<int64_t>& layers) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    state->set_identity(role, layers);
  }, "state"_a, "role"_a, "layers"_a);
  m.def("_expert_ssd_cache_release", [](std::shared_ptr<ExpertSSDCacheState> state, uint64_t reservation) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    state->release(reservation);
  }, "state"_a, "reservation"_a);
  m.def("_expert_ssd_cache_pin_identity", [](std::shared_ptr<ExpertSSDCacheState> state) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    return state->pin_identity_rows();
  }, "state"_a);
  m.def("_expert_ssd_cache_begin_private_prefill", [](std::shared_ptr<ExpertSSDCacheState> state) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    state->begin_private_prefill();
  }, "state"_a);
  m.def("_expert_ssd_cache_end_private_prefill", [](std::shared_ptr<ExpertSSDCacheState> state) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    state->end_private_prefill();
  }, "state"_a);
  m.def("_expert_ssd_cache_reservations", [](std::shared_ptr<ExpertSSDCacheState> state) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    return state->reservation_metadata();
  }, "state"_a);
  m.def("_expert_ssd_cache_snapshot", [](std::shared_ptr<ExpertSSDCacheState> state) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    return state->snapshot();
  }, "state"_a);
  m.def("_expert_ssd_cache_snapshot_validate", [](std::shared_ptr<ExpertSSDCacheState> state, nb::dict snapshot, bool for_restore) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    state->validate_snapshot(snapshot, for_restore);
  }, "state"_a, "snapshot"_a, "for_restore"_a = false);
  m.def("_expert_ssd_cache_snapshot_restore", [](std::shared_ptr<ExpertSSDCacheState> state, nb::dict snapshot) {
    if (!state) throw std::invalid_argument("ExpertSSD state is required");
    state->restore_snapshot(snapshot);
  }, "state"_a, "snapshot"_a);
  m.def(
      "_expert_ssd_internal_stripe_bytes",
      [](size_t total, double ratio) {
        if (!std::isfinite(ratio) || ratio <= 0.0) {
          throw std::invalid_argument("ratio must be a positive finite number");
        }
        return expert_ssd_internal_stripe_bytes_for_ratio(total, ratio);
      },
      "total"_a,
      "ratio"_a,
      nb::sig(
          "def _expert_ssd_internal_stripe_bytes(total: int, ratio: float) -> int"));
  m.def(
      "_expert_ssd_markov_state_new",
      [](size_t expert_count, size_t history_limit) {
        return std::make_shared<ExpertSSDMarkovState>(
            expert_count, history_limit);
      },
      "expert_count"_a,
      "history_limit"_a,
      nb::sig(
          "def _expert_ssd_markov_state_new(expert_count: int, history_limit: int) -> _ExpertSSDMarkovState"));
  m.def(
      "_expert_ssd_markov_update",
      [](std::shared_ptr<ExpertSSDMarkovState> state,
         const std::vector<int64_t>& requested,
         const std::vector<int64_t>& candidates) {
        return state->update_and_score(requested, candidates);
      },
      "state"_a,
      "requested"_a,
      "candidates"_a,
      nb::sig(
          "def _expert_ssd_markov_update(state: _ExpertSSDMarkovState, requested: list[int], candidates: list[int]) -> list[float]"));
  m.def(
      "_expert_ssd_markov_snapshot",
      [](std::shared_ptr<ExpertSSDMarkovState> state) {
        auto payload = state->serialize();
        return nb::bytes(payload.data(), payload.size());
      },
      "state"_a,
      nb::sig(
          "def _expert_ssd_markov_snapshot(state: _ExpertSSDMarkovState) -> bytes"));
  m.def(
      "_expert_ssd_markov_restore",
      [](std::shared_ptr<ExpertSSDMarkovState> state, nb::bytes payload) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(payload.ptr(), &data, &size) != 0) {
          throw nb::python_error();
        }
        state->restore(std::string(data, static_cast<size_t>(size)));
      },
      "state"_a,
      "payload"_a,
      nb::sig(
          "def _expert_ssd_markov_restore(state: _ExpertSSDMarkovState, payload: bytes) -> None"));
  m.def(
      "_expert_ssd_route_cache_state_new",
      [](size_t expert_count,
         size_t capacity,
         double retention_decay,
         double interval_alpha,
         double initial_interval,
         double age_weight,
         double markov_boost,
         std::shared_ptr<ExpertSSDMarkovState> markov) {
        return std::make_shared<ExpertSSDRouteCacheState>(
            expert_count,
            capacity,
            retention_decay,
            interval_alpha,
            initial_interval,
            age_weight,
            markov_boost,
            std::move(markov));
      },
      "expert_count"_a,
      "capacity"_a,
      "retention_decay"_a,
      "interval_alpha"_a,
      "initial_interval"_a,
      "age_weight"_a,
      "markov_boost"_a,
      "markov"_a = nullptr,
      nb::sig(
          "def _expert_ssd_route_cache_state_new(expert_count: int, capacity: int, retention_decay: float, interval_alpha: float, initial_interval: float, age_weight: float, markov_boost: float, markov: _ExpertSSDMarkovState | None = None) -> _ExpertSSDRouteCacheState"));
  m.def(
      "_expert_ssd_route_cache_restore",
      [](std::shared_ptr<ExpertSSDRouteCacheState> state,
         const std::vector<std::pair<int64_t, int64_t>>& expert_slots,
         const std::vector<std::pair<int64_t, int64_t>>& down_expert_slots,
         const std::vector<std::pair<int64_t, double>>& retention_scores,
         int64_t lhd_clock,
         double demand_scale,
         const std::vector<double>& demand,
         const std::vector<int64_t>& last_seen,
         const std::vector<double>& interval_ema) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD route-cache state is required");
        }
        state->restore(
            expert_slots,
            down_expert_slots,
            retention_scores,
            lhd_clock,
            demand_scale,
            demand,
            last_seen,
            interval_ema);
      },
      "state"_a,
      "expert_slots"_a,
      "down_expert_slots"_a,
      "retention_scores"_a,
      "lhd_clock"_a,
      "demand_scale"_a,
      "demand"_a,
      "last_seen"_a,
      "interval_ema"_a,
      nb::sig(
          "def _expert_ssd_route_cache_restore(state: _ExpertSSDRouteCacheState, expert_slots: list[tuple[int, int]], down_expert_slots: list[tuple[int, int]], retention_scores: list[tuple[int, float]], lhd_clock: int, demand_scale: float, demand: list[float], last_seen: list[int], interval_ema: list[float]) -> None"));
  m.def(
      "_expert_ssd_route_cache_metadata",
      [](std::shared_ptr<ExpertSSDRouteCacheState> state) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD route-cache state is required");
        }
        return state->legacy_metadata();
      },
      "state"_a,
      nb::sig(
          "def _expert_ssd_route_cache_metadata(state: _ExpertSSDRouteCacheState) -> dict"));
  m.def(
      "_expert_ssd_route_cache_plan",
      [](std::shared_ptr<ExpertSSDRouteCacheState> state, mx::array indices,
         const std::vector<uint8_t>& active_mask, bool reserve) {
        if (!state || indices.ndim() == 0) {
          throw std::invalid_argument(
              "ExpertSSD route-cache state and routed indices are required");
        }
        for (auto stride : indices.strides()) {
          if (stride < 0) {
            throw std::invalid_argument(
                "ExpertSSD route-cache negative strides are unsupported");
          }
        }
        ExpertSSDRouteCachePlan plan;
        {
          // The planner owns the only host synchronization for this layer.
          // Release the GIL while upstream Metal work completes and while the
          // persistent native cache transaction mutates its policy state.
          nb::gil_scoped_release release;
          indices.eval();
          plan = state->plan(indices, active_mask, reserve);
        }
        nb::dict output;
        nb::list unique;
        nb::dict counts;
        nb::list routed;
        nb::list gate_up_rows;
        nb::list down_rows;
        nb::list resident_before;
        nb::list missing;
        nb::list missing_gate_up_rows;
        nb::list missing_down_rows;
        nb::list evicted;
        for (size_t index = 0; index < plan.unique.size(); ++index) {
          const auto expert = plan.unique[index];
          unique.append(nb::cast(expert));
          counts[nb::cast(expert)] = nb::cast(plan.counts[index]);
          gate_up_rows.append(nb::cast(plan.gate_up_rows[index]));
          down_rows.append(nb::cast(plan.down_rows[index]));
          resident_before.append(nb::cast(bool(plan.resident_before[index])));
        }
        for (const auto expert : plan.routed) {
          routed.append(nb::cast(expert));
        }
        for (size_t index = 0; index < plan.missing.size(); ++index) {
          missing.append(nb::cast(plan.missing[index]));
          missing_gate_up_rows.append(
              nb::cast(plan.missing_gate_up_rows[index]));
          missing_down_rows.append(
              nb::cast(plan.missing_down_rows[index]));
        }
        for (const auto expert : plan.evicted) {
          evicted.append(nb::cast(expert));
        }
        output["unique"] = unique;
        output["counts"] = counts;
        output["routed"] = routed;
        output["compact"] = mx::array(
            plan.compact.begin(), indices.shape(), indices.dtype());
        output["gate_up_rows"] = gate_up_rows;
        output["down_rows"] = down_rows;
        if (reserve) output["reservation"] = plan.reservation;
        output["resident_before"] = resident_before;
        output["missing"] = missing;
        output["missing_gate_up_rows"] = missing_gate_up_rows;
        output["missing_down_rows"] = missing_down_rows;
        output["evicted"] = evicted;
        output["hits"] = plan.hits;
        output["misses"] = plan.misses;
        return output;
      },
      "state"_a,
      "indices"_a,
      "active_mask"_a = std::vector<uint8_t>{},
      "reserve"_a = false,
      nb::sig(
          "def _expert_ssd_route_cache_plan(state: _ExpertSSDRouteCacheState, indices: array, active_mask: list[int] = [], reserve: bool = False) -> dict"));
  m.def(
      "_expert_ssd_route_cache_replay_all_hits",
      [](std::shared_ptr<ExpertSSDRouteCacheState> state,
         const std::vector<std::vector<int64_t>>& routes) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD route-cache state is required");
        }
        nb::gil_scoped_release release;
        state->replay_all_hit_routes(routes);
      },
      "state"_a,
      "routes"_a,
      nb::sig(
          "def _expert_ssd_route_cache_replay_all_hits(state: _ExpertSSDRouteCacheState, routes: list[list[int]]) -> None"));
  m.def(
      "_expert_ssd_route_cache_try_all_hit",
      [](std::shared_ptr<ExpertSSDRouteCacheState> state, mx::array indices) {
        if (!state || indices.ndim() == 0) {
          throw std::invalid_argument(
              "ExpertSSD route-cache state and routed indices are required");
        }
        for (auto stride : indices.strides()) {
          if (stride < 0) {
            throw std::invalid_argument(
                "ExpertSSD route-cache negative strides are unsupported");
          }
        }
        nb::gil_scoped_release release;
        indices.eval();
        return state->try_all_hit(indices);
      },
      "state"_a,
      "indices"_a,
      nb::sig(
          "def _expert_ssd_route_cache_try_all_hit(state: _ExpertSSDRouteCacheState, indices: array) -> int"));
  m.def(
      "_expert_ssd_global_pool_state_new",
      [](size_t expert_count,
         const std::vector<int64_t>& layers,
         const std::vector<int64_t>& private_capacities,
         size_t shared_capacity,
         double retention_decay,
         double interval_alpha,
         double initial_interval,
         double age_weight,
         double deadline_weight,
         double markov_boost,
         size_t markov_history,
         double miss_pressure_alpha,
         double miss_pressure_weight) {
        return std::make_shared<ExpertSSDGlobalPoolState>(
            expert_count,
            layers,
            private_capacities,
            shared_capacity,
            retention_decay,
            interval_alpha,
            initial_interval,
            age_weight,
            deadline_weight,
            markov_boost,
            markov_history,
            miss_pressure_alpha,
            miss_pressure_weight);
      },
      "expert_count"_a,
      "layers"_a,
      "private_capacities"_a,
      "shared_capacity"_a,
      "retention_decay"_a,
      "interval_alpha"_a,
      "initial_interval"_a,
      "age_weight"_a,
      "deadline_weight"_a,
      "markov_boost"_a,
      "markov_history"_a,
      "miss_pressure_alpha"_a,
      "miss_pressure_weight"_a,
      nb::sig(
          "def _expert_ssd_global_pool_state_new(expert_count: int, layers: list[int], private_capacities: list[int], shared_capacity: int, retention_decay: float, interval_alpha: float, initial_interval: float, age_weight: float, deadline_weight: float, markov_boost: float, markov_history: int, miss_pressure_alpha: float, miss_pressure_weight: float) -> _ExpertSSDGlobalPoolState"));
  m.def(
      "_expert_ssd_global_pool_plan",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state,
         int64_t layer,
         mx::array indices,
         const std::vector<uint8_t>& active_mask, bool reserve) {
        if (!state || indices.ndim() == 0) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state and routed indices are required");
        }
        for (auto stride : indices.strides()) {
          if (stride < 0) {
            throw std::invalid_argument(
                "ExpertSSD global-pool negative strides are unsupported");
          }
        }
        ExpertSSDGlobalPoolPlan plan;
        {
          nb::gil_scoped_release release;
          indices.eval();
          plan = state->plan(layer, indices, active_mask, reserve);
        }
        nb::dict output;
        nb::list unique;
        nb::dict counts;
        nb::list routed;
        nb::list shared_bank;
        nb::list rows;
        nb::list resident_before;
        nb::list missing;
        nb::list missing_shared_bank;
        nb::list missing_rows;
        nb::list evicted;
        for (size_t index = 0; index < plan.unique.size(); ++index) {
          const auto expert = plan.unique[index];
          unique.append(nb::cast(expert));
          counts[nb::cast(expert)] = nb::cast(plan.counts[index]);
          shared_bank.append(nb::cast(bool(plan.shared_bank[index])));
          rows.append(nb::cast(plan.rows[index]));
          resident_before.append(nb::cast(bool(plan.resident_before[index])));
        }
        for (const auto expert : plan.routed) {
          routed.append(nb::cast(expert));
        }
        for (size_t index = 0; index < plan.missing.size(); ++index) {
          missing.append(nb::cast(plan.missing[index]));
          missing_shared_bank.append(
              nb::cast(bool(plan.missing_shared_bank[index])));
          missing_rows.append(nb::cast(plan.missing_rows[index]));
        }
        for (size_t index = 0; index < plan.evicted_experts.size(); ++index) {
          evicted.append(nb::make_tuple(
              plan.evicted_layers[index], plan.evicted_experts[index]));
        }
        output["unique"] = unique;
        output["counts"] = counts;
        output["routed"] = routed;
        output["compact"] = mx::array(
            plan.compact.begin(), indices.shape(), indices.dtype());
        output["shared_bank"] = shared_bank;
        if (reserve) output["reservation"] = plan.reservation;
        output["rows"] = rows;
        output["resident_before"] = resident_before;
        output["missing"] = missing;
        output["missing_shared_bank"] = missing_shared_bank;
        output["missing_rows"] = missing_rows;
        output["evicted"] = evicted;
        output["hits"] = plan.hits;
        output["misses"] = plan.misses;
        return output;
      },
      "state"_a,
      "layer"_a,
      "indices"_a,
      "active_mask"_a = std::vector<uint8_t>{},
      "reserve"_a = false,
      nb::sig(
          "def _expert_ssd_global_pool_plan(state: _ExpertSSDGlobalPoolState, layer: int, indices: array, active_mask: list[int] = [], reserve: bool = False) -> dict"));
  m.def(
      "_expert_ssd_global_pool_replay_all_hit",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state,
         int64_t layer,
         const std::vector<int64_t>& routed) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        nb::gil_scoped_release release;
        state->replay_all_hit_route(layer, routed);
      },
      "state"_a,
      "layer"_a,
      "routed"_a,
      nb::sig(
          "def _expert_ssd_global_pool_replay_all_hit(state: _ExpertSSDGlobalPoolState, layer: int, routed: list[int]) -> None"));
  m.def(
      "_expert_ssd_global_pool_try_all_hit",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state,
         int64_t layer,
         mx::array indices) {
        if (!state || indices.ndim() == 0) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state and routed indices are required");
        }
        for (auto stride : indices.strides()) {
          if (stride < 0) {
            throw std::invalid_argument(
                "ExpertSSD global-pool negative strides are unsupported");
          }
        }
        nb::gil_scoped_release release;
        indices.eval();
        return state->try_all_hit(layer, indices);
      },
      "state"_a,
      "layer"_a,
      "indices"_a,
      nb::sig(
          "def _expert_ssd_global_pool_try_all_hit(state: _ExpertSSDGlobalPoolState, layer: int, indices: array) -> int"));
  m.def(
      "_expert_ssd_global_pool_restore_rows",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state,
         const std::vector<std::vector<int64_t>>& private_assignments,
         const std::vector<std::vector<int64_t>>& shared_assignments) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        nb::gil_scoped_release release;
        state->restore_rows(private_assignments, shared_assignments);
      },
      "state"_a,
      "private_assignments"_a,
      "shared_assignments"_a,
      nb::sig(
          "def _expert_ssd_global_pool_restore_rows(state: _ExpertSSDGlobalPoolState, private_assignments: list[list[int]], shared_assignments: list[list[int]]) -> None"));
  m.def(
      "_expert_ssd_global_pool_lease_shared_rows",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        nb::gil_scoped_release release;
        state->lease_shared_rows();
      },
      "state"_a,
      nb::sig(
          "def _expert_ssd_global_pool_lease_shared_rows(state: _ExpertSSDGlobalPoolState) -> None"));
  m.def(
      "_expert_ssd_global_pool_restore_leased_shared_rows",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state,
         const std::vector<std::vector<int64_t>>& shared_assignments) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        nb::gil_scoped_release release;
        state->restore_leased_shared_rows(shared_assignments);
      },
      "state"_a,
      "shared_assignments"_a,
      nb::sig(
          "def _expert_ssd_global_pool_restore_leased_shared_rows(state: _ExpertSSDGlobalPoolState, shared_assignments: list[list[int]]) -> None"));
  m.def(
      "_expert_ssd_global_pool_restore_layer_policy",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state,
         int64_t layer,
         int64_t clock,
         double demand_scale,
         const std::vector<double>& demand,
         const std::vector<int64_t>& last_seen,
         const std::vector<double>& interval_ema,
         nb::bytes markov_payload) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        const std::string payload(markov_payload.c_str(), markov_payload.size());
        nb::gil_scoped_release release;
        state->restore_layer_policy(
            layer,
            clock,
            demand_scale,
            demand,
            last_seen,
            interval_ema,
            payload);
      },
      "state"_a,
      "layer"_a,
      "clock"_a,
      "demand_scale"_a,
      "demand"_a,
      "last_seen"_a,
      "interval_ema"_a,
      "markov_payload"_a,
      nb::sig(
          "def _expert_ssd_global_pool_restore_layer_policy(state: _ExpertSSDGlobalPoolState, layer: int, clock: int, demand_scale: float, demand: list[float], last_seen: list[int], interval_ema: list[float], markov_payload: bytes) -> None"));
  m.def(
      "_expert_ssd_global_pool_policy_snapshot",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        const auto payload = state->serialize_policy();
        return nb::bytes(payload.data(), payload.size());
      },
      "state"_a,
      nb::sig(
          "def _expert_ssd_global_pool_policy_snapshot(state: _ExpertSSDGlobalPoolState) -> bytes"));
  m.def(
      "_expert_ssd_global_pool_policy_validate",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state, nb::bytes encoded) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        const std::string payload(encoded.c_str(), encoded.size());
        state->validate_policy_snapshot(payload);
      },
      "state"_a,
      "payload"_a,
      nb::sig(
          "def _expert_ssd_global_pool_policy_validate(state: _ExpertSSDGlobalPoolState, payload: bytes) -> None"));
  m.def(
      "_expert_ssd_global_pool_policy_restore",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state, nb::bytes encoded) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        const std::string payload(encoded.c_str(), encoded.size());
        nb::gil_scoped_release release;
        state->restore_policy_snapshot(payload);
      },
      "state"_a,
      "payload"_a,
      nb::sig(
          "def _expert_ssd_global_pool_policy_restore(state: _ExpertSSDGlobalPoolState, payload: bytes) -> None"));
  m.def(
      "_expert_ssd_global_pool_metadata",
      [](std::shared_ptr<ExpertSSDGlobalPoolState> state) {
        if (!state) {
          throw std::invalid_argument(
              "ExpertSSD global-pool state is required");
        }
        return state->metadata();
      },
      "state"_a,
      nb::sig(
          "def _expert_ssd_global_pool_metadata(state: _ExpertSSDGlobalPoolState) -> dict"));
  m.def(
      "_expert_ssd_io_event_state_new",
      []() { return mx::expert_ssd_io_event_state_new(); },
      nb::sig(
          "def _expert_ssd_io_event_state_new() -> _ExpertSSDIoEventState"));
  m.def(
      "_expert_ssd_io_gate",
      [](const mx::array& x,
         std::shared_ptr<mx::ExpertSSDIoEventState> state,
         uint64_t value) {
        return mx::expert_ssd_io_gate(x, state, value);
      },
      "x"_a,
      "state"_a,
      "value"_a,
      nb::sig(
          "def _expert_ssd_io_gate(x: array, state: _ExpertSSDIoEventState, value: int) -> array"));
  m.def(
      "_expert_ssd_io_event_signal",
      [](std::shared_ptr<mx::ExpertSSDIoEventState> state, uint64_t value) {
        mx::expert_ssd_io_event_signal(state, value);
      },
      "state"_a,
      "value"_a,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::sig(
          "def _expert_ssd_io_event_signal(state: _ExpertSSDIoEventState, value: int) -> None"));
  m.def(
      "_expert_ssd_io_event_wait",
      [](std::shared_ptr<mx::ExpertSSDIoEventState> state, uint64_t value) {
        nb::gil_scoped_release release;
        mx::expert_ssd_io_event_wait(state, value);
      },
      "state"_a,
      "value"_a,
      nb::sig(
          "def _expert_ssd_io_event_wait(state: _ExpertSSDIoEventState, value: int) -> None"));
  m.def(
      "_expert_ssd_gpu_event_signal",
      [](const mx::array& x,
         std::shared_ptr<mx::ExpertSSDIoEventState> state,
         uint64_t value) {
        return mx::expert_ssd_gpu_event_signal(x, state, value);
      },
      "x"_a,
      "state"_a,
      "value"_a,
      nb::sig(
          "def _expert_ssd_gpu_event_signal(x: array, state: _ExpertSSDIoEventState, value: int) -> array"));
  m.def(
      "_livseek_legacy_mxfp4_quantize",
      [](const mx::array& weight) {
        return mx::livseek_legacy_mxfp4_quantize(weight);
      },
      "weight"_a,
      nb::sig(
          "def _livseek_legacy_mxfp4_quantize(weight: array) -> list[array]"));
  m.def(
      "_expert_ssd_mxfp4_pair_qmv",
      [](const mx::array& x,
         const mx::array& up_weight,
         const mx::array& up_scales,
         const mx::array& gate_weight,
         const mx::array& gate_scales,
         const mx::array& routes) {
        return mx::expert_ssd_mxfp4_pair_qmv(
            x,
            up_weight,
            up_scales,
            gate_weight,
            gate_scales,
            routes);
      },
      "x"_a,
      "up_weight"_a,
      "up_scales"_a,
      "gate_weight"_a,
      "gate_scales"_a,
      "routes"_a,
      nb::sig(
          "def _expert_ssd_mxfp4_pair_qmv(x: array, up_weight: array, up_scales: array, gate_weight: array, gate_scales: array, routes: array) -> list[array]"));
  m.def(
      "_expert_ssd_mxfp4_two_row_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scales) {
        return mx::expert_ssd_mxfp4_two_row_qmv(x, weight, scales);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      nb::sig(
          "def _expert_ssd_mxfp4_two_row_qmv(x: array, weight: array, scales: array) -> array"));
  m.def(
      "_expert_ssd_mxfp4_grouped_two_row_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scales) {
        return mx::expert_ssd_mxfp4_grouped_two_row_qmv(x, weight, scales);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      nb::sig(
          "def _expert_ssd_mxfp4_grouped_two_row_qmv(x: array, weight: array, scales: array) -> array"));
  m.def(
      "_expert_ssd_mxfp8_two_row_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scales) {
        return mx::expert_ssd_mxfp8_two_row_qmv(x, weight, scales);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      nb::sig(
          "def _expert_ssd_mxfp8_two_row_qmv(x: array, weight: array, scales: array) -> array"));
  m.def(
      "_expert_ssd_mxfp4_three_row_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scales) {
        return mx::expert_ssd_mxfp4_three_row_qmv(x, weight, scales);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      nb::sig(
          "def _expert_ssd_mxfp4_three_row_qmv(x: array, weight: array, scales: array) -> array"));
  m.def(
      "_expert_ssd_mxfp4_grouped_three_row_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scales) {
        return mx::expert_ssd_mxfp4_grouped_three_row_qmv(
            x, weight, scales);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      nb::sig(
          "def _expert_ssd_mxfp4_grouped_three_row_qmv(x: array, weight: array, scales: array) -> array"));
  m.def(
      "_expert_ssd_two_row_gemv",
      [](const mx::array& x, const mx::array& weight) {
        return mx::expert_ssd_two_row_gemv(x, weight);
      },
      "x"_a,
      "weight"_a,
      nb::sig(
          "def _expert_ssd_two_row_gemv(x: array, weight: array) -> array"));
  m.def(
      "_expert_ssd_three_row_gemv",
      [](const mx::array& x, const mx::array& weight) {
        return mx::expert_ssd_three_row_gemv(x, weight);
      },
      "x"_a,
      "weight"_a,
      nb::sig(
          "def _expert_ssd_three_row_gemv(x: array, weight: array) -> array"));
  m.def(
      "_expert_ssd_mxfp4_masked_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scales,
         const mx::array& routes) {
        return mx::expert_ssd_mxfp4_masked_qmv(
            x,
            weight,
            scales,
            routes);
      },
      "x"_a,
      "weight"_a,
      "scales"_a,
      "routes"_a,
      nb::sig(
          "def _expert_ssd_mxfp4_masked_qmv(x: array, weight: array, scales: array, routes: array) -> array"));
  m.def(
      "_perfetto_trace_start",
      [](const std::string& output_path, size_t max_records) {
        if (output_path.empty() || max_records == 0) {
          throw std::invalid_argument(
              "[_perfetto_trace_start] output_path and max_records required");
        }
        return mx::perfetto_trace::start(output_path, max_records);
      },
      "output_path"_a,
      "max_records"_a,
      nb::sig(
          "def _perfetto_trace_start(output_path: str, max_records: int) -> int"),
      R"pbdoc(
        Begin a bounded native LivMLX trace. The returned timestamp uses the
        native monotonic clock and is used to align Python and Metal events.
      )pbdoc");
  m.def(
      "_perfetto_trace_stop",
      []() { mx::perfetto_trace::stop(); },
      nb::sig("def _perfetto_trace_stop() -> None"),
      R"pbdoc(
        Stop accepting new native LivMLX trace records without waiting for
        already submitted SSD reads or Metal command buffers.
      )pbdoc");
  m.def(
      "_perfetto_trace_finish",
      [](size_t timeout_ms) {
        mx::perfetto_trace::FinishResult result;
        {
          nb::gil_scoped_release release;
          result = mx::perfetto_trace::finish(timeout_ms);
        }
        nb::dict out;
        out["records"] = result.records;
        out["dropped"] = result.dropped;
        out["complete"] = result.complete;
        return out;
      },
      "timeout_ms"_a = 5000,
      nb::sig("def _perfetto_trace_finish(timeout_ms: int = 5000) -> dict"),
      R"pbdoc(
        Wait for in-flight native records, then write the native NDJSON
        sidecar. File IO occurs only after capture has stopped.
      )pbdoc");
  m.def(
      "_expert_ssd_scalex_mxfp4_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scale_records,
         const mx::array& routes,
         uint32_t projection) {
        return mx::expert_ssd_scalex_mxfp4_qmv(
            x, weight, scale_records, routes, projection);
      },
      "x"_a,
      "weight"_a,
      "scale_records"_a,
      "routes"_a,
      "projection"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_qmv(x: array, weight: array, scale_records: array, routes: array, projection: int) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_width2_pair_qmv",
      [](const mx::array& x, const mx::array& up, const mx::array& gate,
         const mx::array& records, const mx::array& routes) {
        return mx::expert_ssd_scalex_mxfp4_width2_pair_qmv(x, up, gate, records, routes);
      },
      "x"_a, "up_weight"_a, "gate_weight"_a, "scale_records"_a, "routes"_a);
  m.def(
      "_expert_ssd_scalex_mxfp4_width2_pair_qmv_two_bank",
      [](const mx::array& x,
         const mx::array& private_up,
         const mx::array& private_gate,
         const mx::array& private_records,
         const mx::array& shared_up,
         const mx::array& shared_gate,
         const mx::array& shared_records,
         const mx::array& routes,
         const mx::array& bank_routes) {
        return mx::expert_ssd_scalex_mxfp4_width2_pair_qmv_two_bank(
            x,
            private_up,
            private_gate,
            private_records,
            shared_up,
            shared_gate,
            shared_records,
            routes,
            bank_routes);
      },
      "x"_a,
      "private_up_weight"_a,
      "private_gate_weight"_a,
      "private_scale_records"_a,
      "shared_up_weight"_a,
      "shared_gate_weight"_a,
      "shared_scale_records"_a,
      "routes"_a,
      "bank_routes"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_width2_pair_qmv_two_bank(x: array, private_up_weight: array, private_gate_weight: array, private_scale_records: array, shared_up_weight: array, shared_gate_weight: array, shared_scale_records: array, routes: array, bank_routes: array) -> tuple[array, array]"));
  m.def(
      "_expert_ssd_scalex_mxfp4_grouped_qmv",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scale_records,
         const mx::array& routes,
         uint32_t projection,
         uint32_t top_k) {
        return mx::expert_ssd_scalex_mxfp4_grouped_qmv(
            x, weight, scale_records, routes, projection, top_k);
      },
      "x"_a,
      "weight"_a,
      "scale_records"_a,
      "routes"_a,
      "projection"_a,
      "top_k"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_grouped_qmv(x: array, weight: array, scale_records: array, routes: array, projection: int, top_k: int) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_qmv_two_bank",
      [](const mx::array& x,
         const mx::array& private_weight,
         const mx::array& private_scale_records,
         const mx::array& shared_weight,
         const mx::array& shared_scale_records,
         const mx::array& routes,
         const mx::array& bank_routes,
         uint32_t projection) {
        return mx::expert_ssd_scalex_mxfp4_qmv_two_bank(
            x,
            private_weight,
            private_scale_records,
            shared_weight,
            shared_scale_records,
            routes,
            bank_routes,
            projection);
      },
      "x"_a,
      "private_weight"_a,
      "private_scale_records"_a,
      "shared_weight"_a,
      "shared_scale_records"_a,
      "routes"_a,
      "bank_routes"_a,
      "projection"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_qmv_two_bank(x: array, private_weight: array, private_scale_records: array, shared_weight: array, shared_scale_records: array, routes: array, bank_routes: array, projection: int) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_qmv_split_routes",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scale_records,
         const mx::array& weight_routes,
         const mx::array& scale_routes,
         uint32_t projection) {
        return mx::expert_ssd_scalex_mxfp4_qmv_split_routes(
            x, weight, scale_records, weight_routes, scale_routes, projection);
      },
      "x"_a,
      "weight"_a,
      "scale_records"_a,
      "weight_routes"_a,
      "scale_routes"_a,
      "projection"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_qmv_split_routes(x: array, weight: array, scale_records: array, weight_routes: array, scale_routes: array, projection: int) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank",
      [](const mx::array& x,
         const mx::array& private_weight,
         const mx::array& private_scale_records,
         const mx::array& shared_weight,
         const mx::array& shared_scale_records,
         const mx::array& weight_routes,
         const mx::array& scale_routes,
         const mx::array& bank_routes,
         uint32_t projection) {
        return mx::expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank(
            x,
            private_weight,
            private_scale_records,
            shared_weight,
            shared_scale_records,
            weight_routes,
            scale_routes,
            bank_routes,
            projection);
      },
      "x"_a,
      "private_weight"_a,
      "private_scale_records"_a,
      "shared_weight"_a,
      "shared_scale_records"_a,
      "weight_routes"_a,
      "scale_routes"_a,
      "bank_routes"_a,
      "projection"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_qmv_split_routes_two_bank(x: array, private_weight: array, private_scale_records: array, shared_weight: array, shared_scale_records: array, weight_routes: array, scale_routes: array, bank_routes: array, projection: int) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_width2_down_reduce",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scale_records,
         const mx::array& weight_routes,
         const mx::array& scale_routes,
         const mx::array& scores,
         const mx::array& shared) {
        return mx::expert_ssd_scalex_mxfp4_width2_down_reduce(
            x,
            weight,
            scale_records,
            weight_routes,
            scale_routes,
            scores,
            shared);
      },
      "x"_a,
      "weight"_a,
      "scale_records"_a,
      "weight_routes"_a,
      "scale_routes"_a,
      "scores"_a,
      "shared"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_width2_down_reduce(x: array, weight: array, scale_records: array, weight_routes: array, scale_routes: array, scores: array, shared: array) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_width2_down_reduce_two_bank",
      [](const mx::array& x,
         const mx::array& private_weight,
         const mx::array& private_scale_records,
         const mx::array& shared_weight,
         const mx::array& shared_scale_records,
         const mx::array& weight_routes,
         const mx::array& scale_routes,
         const mx::array& bank_routes,
         const mx::array& scores,
         const mx::array& shared) {
        return mx::expert_ssd_scalex_mxfp4_width2_down_reduce_two_bank(
            x,
            private_weight,
            private_scale_records,
            shared_weight,
            shared_scale_records,
            weight_routes,
            scale_routes,
            bank_routes,
            scores,
            shared);
      },
      "x"_a,
      "private_weight"_a,
      "private_scale_records"_a,
      "shared_weight"_a,
      "shared_scale_records"_a,
      "weight_routes"_a,
      "scale_routes"_a,
      "bank_routes"_a,
      "scores"_a,
      "shared"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_width2_down_reduce_two_bank(x: array, private_weight: array, private_scale_records: array, shared_weight: array, shared_scale_records: array, weight_routes: array, scale_routes: array, bank_routes: array, scores: array, shared: array) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_width3_down_reduce",
      [](const mx::array& x,
         const mx::array& weight,
         const mx::array& scale_records,
         const mx::array& weight_routes,
         const mx::array& scale_routes,
         const mx::array& scores,
         const mx::array& shared) {
        return mx::expert_ssd_scalex_mxfp4_width3_down_reduce(
            x,
            weight,
            scale_records,
            weight_routes,
            scale_routes,
            scores,
            shared);
      },
      "x"_a,
      "weight"_a,
      "scale_records"_a,
      "weight_routes"_a,
      "scale_routes"_a,
      "scores"_a,
      "shared"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_width3_down_reduce(x: array, weight: array, scale_records: array, weight_routes: array, scale_routes: array, scores: array, shared: array) -> array"));
  m.def(
      "_expert_ssd_scalex_mxfp4_width3_down_reduce_two_bank",
      [](const mx::array& x,
         const mx::array& private_weight,
         const mx::array& private_scale_records,
         const mx::array& shared_weight,
         const mx::array& shared_scale_records,
         const mx::array& weight_routes,
         const mx::array& scale_routes,
         const mx::array& bank_routes,
         const mx::array& scores,
         const mx::array& shared) {
        return mx::expert_ssd_scalex_mxfp4_width3_down_reduce_two_bank(
            x,
            private_weight,
            private_scale_records,
            shared_weight,
            shared_scale_records,
            weight_routes,
            scale_routes,
            bank_routes,
            scores,
            shared);
      },
      "x"_a,
      "private_weight"_a,
      "private_scale_records"_a,
      "shared_weight"_a,
      "shared_scale_records"_a,
      "weight_routes"_a,
      "scale_routes"_a,
      "bank_routes"_a,
      "scores"_a,
      "shared"_a,
      nb::sig(
          "def _expert_ssd_scalex_mxfp4_width3_down_reduce_two_bank(x: array, private_weight: array, private_scale_records: array, shared_weight: array, shared_scale_records: array, weight_routes: array, scale_routes: array, bank_routes: array, scores: array, shared: array) -> array"));
  m.def(
      "_expert_ssd_scalex_conditional_m0",
      [](const mx::array& indices,
         const mx::array& x,
         const mx::array& scores,
         const mx::array& shared,
         const mx::array& scale_records,
         const mx::array& gate_weight,
         const mx::array& down_weight,
         const mx::array& up_weight,
         const mx::array& gate_directory,
         const mx::array& down_directory,
         const mx::array& gate_routes_scratch,
         const mx::array& down_routes_scratch,
         const mx::array& all_hit_scratch,
         const mx::array& up_scratch,
         const mx::array& gate_scratch,
         const mx::array& activated_scratch,
         const mx::array& routed_scratch,
         const mx::array& indirect_scratch,
         float swiglu_limit) {
        return mx::expert_ssd_scalex_conditional_m0(
            indices, x, scores, shared, scale_records, gate_weight,
            down_weight, up_weight, gate_directory, down_directory,
            gate_routes_scratch, down_routes_scratch, all_hit_scratch,
            up_scratch, gate_scratch, activated_scratch, routed_scratch,
            indirect_scratch,
            swiglu_limit);
      },
      "indices"_a,
      "x"_a,
      "scores"_a,
      "shared"_a,
      "scale_records"_a,
      "gate_weight"_a,
      "down_weight"_a,
      "up_weight"_a,
      "gate_directory"_a,
      "down_directory"_a,
      "gate_routes_scratch"_a,
      "down_routes_scratch"_a,
      "all_hit_scratch"_a,
      "up_scratch"_a,
      "gate_scratch"_a,
      "activated_scratch"_a,
      "routed_scratch"_a,
      "indirect_scratch"_a,
      "swiglu_limit"_a = 10.0f,
      nb::sig(
          "def _expert_ssd_scalex_conditional_m0(indices: array, x: array, scores: array, shared: array, scale_records: array, gate_weight: array, down_weight: array, up_weight: array, gate_directory: array, down_directory: array, gate_routes_scratch: array, down_routes_scratch: array, all_hit_scratch: array, up_scratch: array, gate_scratch: array, activated_scratch: array, routed_scratch: array, indirect_scratch: array, swiglu_limit: float = 10.0) -> array"));
  m.def(
      "_expert_ssd_scalex_conditional_m0_two_bank",
      [](const mx::array& indices,
         const mx::array& x,
         const mx::array& scores,
         const mx::array& shared,
         const mx::array& private_scale_records,
         const mx::array& private_gate_weight,
         const mx::array& private_down_weight,
         const mx::array& private_up_weight,
         const mx::array& shared_scale_records,
         const mx::array& shared_gate_weight,
         const mx::array& shared_down_weight,
         const mx::array& shared_up_weight,
         const mx::array& gate_directory,
         const mx::array& down_directory,
         const mx::array& bank_directory,
         const mx::array& gate_routes_scratch,
         const mx::array& down_routes_scratch,
         const mx::array& bank_routes_scratch,
         const mx::array& all_hit_scratch,
         const mx::array& up_scratch,
         const mx::array& gate_scratch,
         const mx::array& activated_scratch,
         const mx::array& routed_scratch,
         const mx::array& indirect_scratch,
         float swiglu_limit) {
        return mx::expert_ssd_scalex_conditional_m0_two_bank(
            indices, x, scores, shared,
            private_scale_records, private_gate_weight,
            private_down_weight, private_up_weight,
            shared_scale_records, shared_gate_weight,
            shared_down_weight, shared_up_weight,
            gate_directory, down_directory, bank_directory,
            gate_routes_scratch, down_routes_scratch, bank_routes_scratch,
            all_hit_scratch, up_scratch, gate_scratch, activated_scratch,
            routed_scratch, indirect_scratch, swiglu_limit);
      },
      "indices"_a,
      "x"_a,
      "scores"_a,
      "shared"_a,
      "private_scale_records"_a,
      "private_gate_weight"_a,
      "private_down_weight"_a,
      "private_up_weight"_a,
      "shared_scale_records"_a,
      "shared_gate_weight"_a,
      "shared_down_weight"_a,
      "shared_up_weight"_a,
      "gate_directory"_a,
      "down_directory"_a,
      "bank_directory"_a,
      "gate_routes_scratch"_a,
      "down_routes_scratch"_a,
      "bank_routes_scratch"_a,
      "all_hit_scratch"_a,
      "up_scratch"_a,
      "gate_scratch"_a,
      "activated_scratch"_a,
      "routed_scratch"_a,
      "indirect_scratch"_a,
      "swiglu_limit"_a = 10.0f,
      nb::sig(
          "def _expert_ssd_scalex_conditional_m0_two_bank(indices: array, x: array, scores: array, shared: array, private_scale_records: array, private_gate_weight: array, private_down_weight: array, private_up_weight: array, shared_scale_records: array, shared_gate_weight: array, shared_down_weight: array, shared_up_weight: array, gate_directory: array, down_directory: array, bank_directory: array, gate_routes_scratch: array, down_routes_scratch: array, bank_routes_scratch: array, all_hit_scratch: array, up_scratch: array, gate_scratch: array, activated_scratch: array, routed_scratch: array, indirect_scratch: array, swiglu_limit: float = 10.0) -> array"));
  m.def(
      "_expert_ssd_route_plan",
      &official_direct_route_plan,
      "indices"_a,
      nb::sig(
          "def _expert_ssd_route_plan(indices: array) -> tuple[list[int], dict[int, int], array]"),
      R"pbdoc(
        Materialize routed indices once and return first-seen experts, their
        selection counts, and compact remapped indices in one native pass.
      )pbdoc");
  m.def(
      "_open_expert_safetensors_direct",
      [](nb::object file,
         const std::vector<std::vector<std::tuple<
             std::string,
             std::string,
             std::vector<int64_t>,
             size_t>>>& raw_expert_specs,
         bool no_cache,
         bool read_ahead) {
        if (raw_expert_specs.empty()) {
          throw std::invalid_argument(
              "[_open_expert_safetensors_direct] expert specs must be non-empty");
        }
        std::vector<std::vector<mx::SafetensorsTensorSpec>> expert_specs;
        expert_specs.reserve(raw_expert_specs.size());
        for (const auto& raw_specs : raw_expert_specs) {
          std::vector<mx::SafetensorsTensorSpec> specs;
          specs.reserve(raw_specs.size());
          for (const auto& item : raw_specs) {
            const auto& [name, dtype_string, raw_shape, absolute_offset] = item;
            mx::Shape shape;
            shape.reserve(raw_shape.size());
            for (auto dim : raw_shape) {
              if (dim < 0 || dim > std::numeric_limits<int32_t>::max()) {
                throw std::invalid_argument(
                    "[_open_expert_safetensors_direct] invalid tensor shape");
              }
              shape.push_back(static_cast<int32_t>(dim));
            }
            specs.push_back(mx::SafetensorsTensorSpec{
                name,
                dtype_from_official_direct_string(dtype_string),
                std::move(shape),
                absolute_offset});
          }
          expert_specs.push_back(std::move(specs));
        }
        return std::make_shared<mx::ExpertSafetensorsDirect>(
            nb::cast<std::string>(nb::str(file)),
            std::move(expert_specs),
            no_cache,
            read_ahead);
      },
      "file"_a,
      "expert_specs"_a,
      "no_cache"_a = false,
      "read_ahead"_a = true,
      nb::sig(
          "def _open_expert_safetensors_direct(file: Union[str, pathlib.Path], expert_specs: list[list[tuple[str, str, list[int], int]]], no_cache: bool = False, read_ahead: bool = True) -> _ExpertSafetensorsDirect"),
      R"pbdoc(
        Open one official checkpoint shard and cache the exact absolute tensor
        offsets for every routed expert. Adjacent tensors are coalesced into
        positioned scatter reads; no converted slab is required.
      )pbdoc");
  m.def(
      "_open_scalex_prefix_store",
      [](nb::object file,
         size_t data_offset,
         size_t layers,
         size_t experts_per_layer,
         size_t prefix_nbytes) {
        return std::make_shared<mx::ScaleXPrefixStore>(
            nb::cast<std::string>(nb::str(file)),
            data_offset,
            layers,
            experts_per_layer,
            prefix_nbytes);
      },
      "file"_a,
      "data_offset"_a,
      "layers"_a,
      "experts_per_layer"_a,
      "prefix_nbytes"_a,
      nb::sig(
          "def _open_scalex_prefix_store(file: Union[str, pathlib.Path], data_offset: int, layers: int, experts_per_layer: int, prefix_nbytes: int) -> _ScaleXPrefixStore"));
  m.def(
      "_open_scalex_mode_a_direct",
      [](nb::object file,
         const std::vector<std::pair<size_t, size_t>>& raw_records,
         const std::vector<size_t>& decoded_tensor_nbytes,
         bool no_cache,
         bool read_ahead,
         std::shared_ptr<mx::ScaleXPrefixStore> prefix_store,
         size_t prefix_layer) {
        if (raw_records.empty() || decoded_tensor_nbytes.size() != 3) {
          throw std::invalid_argument(
              "[_open_scalex_mode_a_direct] expected records and three tensor sizes");
        }
        std::vector<mx::ScaleXModeARecordSpec> records;
        records.reserve(raw_records.size());
        for (const auto& [offset, length] : raw_records) {
          records.push_back({offset, length});
        }
        std::array<size_t, 3> tensor_nbytes{
            decoded_tensor_nbytes[0],
            decoded_tensor_nbytes[1],
            decoded_tensor_nbytes[2]};
        return std::make_shared<mx::ScaleXModeADirect>(
            nb::cast<std::string>(nb::str(file)),
            std::move(records),
            tensor_nbytes,
            no_cache,
            read_ahead,
            std::move(prefix_store),
            prefix_layer);
      },
      "file"_a,
      "records"_a,
      "decoded_tensor_nbytes"_a,
      "no_cache"_a = false,
      "read_ahead"_a = true,
      "prefix_store"_a = nullptr,
      "prefix_layer"_a = 0,
      nb::sig(
          "def _open_scalex_mode_a_direct(file: Union[str, pathlib.Path], records: list[tuple[int, int]], decoded_tensor_nbytes: list[int], no_cache: bool = False, read_ahead: bool = True, prefix_store: Optional[_ScaleXPrefixStore] = None, prefix_layer: int = 0) -> _ScaleXModeADirect"));
  m.def(
      "_open_scalex_mode_a_replicated_direct",
      [](const std::vector<std::string>& files,
         const std::vector<std::pair<size_t, size_t>>& raw_records,
         const std::vector<size_t>& decoded_tensor_nbytes,
         bool no_cache,
         bool read_ahead,
         std::shared_ptr<mx::ScaleXPrefixStore> prefix_store,
         size_t prefix_layer) {
        if (files.size() != 2 || raw_records.empty() ||
            decoded_tensor_nbytes.size() != 3) {
          throw std::invalid_argument(
              "[_open_scalex_mode_a_replicated_direct] expected two files, "
              "records, and three tensor sizes");
        }
        std::vector<mx::ScaleXModeARecordSpec> records;
        records.reserve(raw_records.size());
        for (const auto& [offset, length] : raw_records) {
          records.push_back({offset, length});
        }
        std::array<size_t, 3> tensor_nbytes{
            decoded_tensor_nbytes[0],
            decoded_tensor_nbytes[1],
            decoded_tensor_nbytes[2]};
        return std::make_shared<mx::ScaleXModeADirect>(
            files,
            std::move(records),
            tensor_nbytes,
            no_cache,
            read_ahead,
            std::move(prefix_store),
            prefix_layer);
      },
      "files"_a,
      "records"_a,
      "decoded_tensor_nbytes"_a,
      "no_cache"_a = false,
      "read_ahead"_a = true,
      "prefix_store"_a = nullptr,
      "prefix_layer"_a = 0,
      nb::sig(
          "def _open_scalex_mode_a_replicated_direct(files: list[str], records: list[tuple[int, int]], decoded_tensor_nbytes: list[int], no_cache: bool = False, read_ahead: bool = True, prefix_store: Optional[_ScaleXPrefixStore] = None, prefix_layer: int = 0) -> _ScaleXModeADirect"));
  m.def(
      "_scalex_mode_a_prefix_stats",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct) {
        if (!direct) {
          throw std::invalid_argument(
              "[_scalex_mode_a_prefix_stats] invalid handle");
        }
        const auto stats = direct->prefix_stats();
        nb::dict result;
        result["persisted"] = stats.persisted;
        result["prepare_calls"] = stats.prepare_calls;
        result["prepare_nanoseconds"] = stats.prepare_nanoseconds;
        result["store_bytes"] = stats.store_bytes;
        result["store_load_nanoseconds"] = stats.store_load_nanoseconds;
        return result;
      },
      "direct"_a,
      nb::sig(
          "def _scalex_mode_a_prefix_stats(direct: _ScaleXModeADirect) -> dict"));
  m.def(
      "_scalex_mode_a_replica_stats",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct) {
        if (!direct) {
          throw std::invalid_argument(
              "[_scalex_mode_a_replica_stats] invalid handle");
        }
        const auto stats = direct->replica_stats();
        nb::dict result;
        result["replica_count"] = stats.replica_count;
        for (size_t replica = 0; replica < stats.replica_count; ++replica) {
          const auto suffix = std::to_string(replica);
          result[("replica_" + suffix + "_reads").c_str()] =
              stats.reads[replica];
          result[("replica_" + suffix + "_bytes").c_str()] =
              stats.bytes[replica];
        }
        return result;
      },
      "direct"_a,
      nb::sig(
          "def _scalex_mode_a_replica_stats(direct: _ScaleXModeADirect) -> dict"));
  m.def(
      "_scalex_mode_a_advise_read",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         size_t expert_id,
         size_t total_bytes) {
        if (!direct) {
          throw std::invalid_argument(
              "[_scalex_mode_a_advise_read] invalid handle");
        }
        nb::gil_scoped_release release;
        if (direct->replica_count() == 2) {
          const size_t split = expert_ssd_internal_stripe_bytes(total_bytes);
          return direct->advise_read_slice_from(0, expert_id, 0, split) +
              direct->advise_read_slice_from(
                  1, expert_id, split, total_bytes);
        }
        return direct->advise_read(expert_id, total_bytes);
      },
      "direct"_a,
      "expert_id"_a,
      "total_bytes"_a,
      nb::sig(
          "def _scalex_mode_a_advise_read(direct: _ScaleXModeADirect, expert_id: int, total_bytes: int) -> int"));
  m.def(
      "_scalex_mode_a_advise_read_many",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         const std::vector<size_t>& expert_ids,
         const std::vector<size_t>& total_bytes) {
        if (!direct || expert_ids.size() != total_bytes.size()) {
          throw std::invalid_argument(
              "[_scalex_mode_a_advise_read_many] invalid batch");
        }
        std::vector<size_t> advised;
        advised.reserve(expert_ids.size());
        nb::gil_scoped_release release;
        for (size_t index = 0; index < expert_ids.size(); ++index) {
          if (direct->replica_count() == 2) {
            const size_t split =
                expert_ssd_internal_stripe_bytes(total_bytes[index]);
            advised.push_back(
                direct->advise_read_slice_from(
                    0, expert_ids[index], 0, split) +
                direct->advise_read_slice_from(
                    1, expert_ids[index], split, total_bytes[index]));
          } else {
            advised.push_back(
                direct->advise_read(expert_ids[index], total_bytes[index]));
          }
        }
        return advised;
      },
      "direct"_a,
      "expert_ids"_a,
      "total_bytes"_a,
      nb::sig(
          "def _scalex_mode_a_advise_read_many(direct: _ScaleXModeADirect, expert_ids: list[int], total_bytes: list[int]) -> list[int]"));
  m.def(
      "_scalex_mode_a_page_residency",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         size_t expert_id,
         size_t total_bytes) {
        if (!direct) {
          throw std::invalid_argument(
              "[_scalex_mode_a_page_residency] invalid handle");
        }
        nb::gil_scoped_release release;
        return direct->page_residency(expert_id, total_bytes);
      },
      "direct"_a,
      "expert_id"_a,
      "total_bytes"_a,
      nb::sig(
          "def _scalex_mode_a_page_residency(direct: _ScaleXModeADirect, expert_id: int, total_bytes: int) -> tuple[int, int]"));
  m.def(
      "_open_safetensors_row_direct",
      [](nb::object file,
         const std::string& dtype,
         int rows,
         int columns,
         size_t absolute_offset) {
        return std::make_shared<mx::SafetensorsRowDirect>(
            nb::cast<std::string>(nb::str(file)),
            dtype_from_official_direct_string(dtype),
            rows,
            columns,
            absolute_offset);
      },
      "file"_a,
      "dtype"_a,
      "rows"_a,
      "columns"_a,
      "absolute_offset"_a,
      nb::sig(
          "def _open_safetensors_row_direct(file: Union[str, pathlib.Path], dtype: str, rows: int, columns: int, absolute_offset: int) -> _SafetensorsRowDirect"));
  m.def(
      "_safetensors_row_direct_load",
      [](std::shared_ptr<mx::SafetensorsRowDirect> direct,
         const std::vector<size_t>& row_ids,
         mx::array destination) {
        if (!direct || destination.dtype() != direct->dtype() ||
            destination.ndim() < 2 ||
            destination.shape(-1) != direct->columns() ||
            destination.size() / direct->columns() != row_ids.size()) {
          throw std::invalid_argument(
              "[_safetensors_row_direct_load] destination layout mismatch");
        }
        if (!destination.is_available()) {
          destination.eval();
        }
        if (!destination.flags().row_contiguous) {
          throw std::invalid_argument(
              "[_safetensors_row_direct_load] destination must be row-contiguous");
        }
        nb::gil_scoped_release release;
        direct->load_rows_into(
            row_ids, destination.data<char>(), destination.nbytes());
      },
      "direct"_a,
      "row_ids"_a,
      "destination"_a,
      nb::sig(
          "def _safetensors_row_direct_load(direct: _SafetensorsRowDirect, row_ids: list[int], destination: array) -> None"));
  m.def(
      "_expert_safetensors_direct_read_range_count",
      &mx::ExpertSafetensorsDirect::read_range_count,
      "direct"_a,
      "expert_id"_a,
      nb::sig(
          "def _expert_safetensors_direct_read_range_count(direct: _ExpertSafetensorsDirect, expert_id: int) -> int"));
  m.def(
      "_expert_safetensors_direct_advise_read",
      [](std::shared_ptr<mx::ExpertSafetensorsDirect> direct,
         size_t expert_id) {
        if (!direct) {
          throw std::invalid_argument(
              "[_expert_safetensors_direct_advise_read] direct handle is null");
        }
        nb::gil_scoped_release release;
        return direct->advise_read(expert_id);
      },
      "direct"_a,
      "expert_id"_a,
      nb::sig(
          "def _expert_safetensors_direct_advise_read(direct: _ExpertSafetensorsDirect, expert_id: int) -> int"),
      R"pbdoc(
        Ask macOS to asynchronously warm the expert's source ranges in the
        file cache without copying bytes into a Metal-visible destination.
      )pbdoc");
  m.def(
      "_expert_safetensors_direct_advise_read_many",
      [](std::shared_ptr<mx::ExpertSafetensorsDirect> direct,
         const std::vector<size_t>& expert_ids) {
        if (!direct) {
          throw std::invalid_argument(
              "[_expert_safetensors_direct_advise_read_many] direct handle is null");
        }
        std::vector<size_t> advised;
        advised.reserve(expert_ids.size());
        nb::gil_scoped_release release;
        for (const auto expert_id : expert_ids) {
          advised.push_back(direct->advise_read(expert_id));
        }
        return advised;
      },
      "direct"_a,
      "expert_ids"_a,
      nb::sig(
          "def _expert_safetensors_direct_advise_read_many(direct: _ExpertSafetensorsDirect, expert_ids: list[int]) -> list[int]"));
  m.def(
      "_expert_ssd_direct_load_into",
      [](std::shared_ptr<mx::ExpertSafetensorsDirect> direct,
         size_t expert_id,
         size_t slot,
         std::vector<mx::array> destinations) {
        if (!direct) {
          throw std::invalid_argument(
              "[_expert_ssd_direct_load_into] direct handle is null");
        }
        auto rows =
            validate_official_direct_destinations(direct, destinations);
        if (slot >= rows.capacity) {
          throw std::out_of_range(
              "[_expert_ssd_direct_load_into] destination slot is out of range");
        }
        std::vector<char*> pointers(rows.bases.size());
        for (size_t index = 0; index < pointers.size(); ++index) {
          pointers[index] =
              rows.bases[index] + slot * rows.row_nbytes[index];
        }
        nb::gil_scoped_release release;
        direct->load_ordered_into(expert_id, pointers, rows.row_nbytes);
      },
      "direct"_a,
      "expert_id"_a,
      "slot"_a,
      "destinations"_a,
      nb::sig(
          "def _expert_ssd_direct_load_into(direct: _ExpertSafetensorsDirect, expert_id: int, slot: int, destinations: list[array]) -> None"),
      R"pbdoc(
        Read one expert directly from the official checkpoint into a row of
        pre-evaluated MLX slot arrays. This mutates the destination buffers;
        callers must not overwrite rows still used by in-flight computation.
      )pbdoc");
  m.def(
      "_expert_ssd_direct_load_into_many",
      [](std::shared_ptr<mx::ExpertSafetensorsDirect> direct,
         const std::vector<size_t>& expert_ids,
         const std::vector<size_t>& slots,
         std::vector<mx::array> destinations) {
        if (!direct || expert_ids.size() != slots.size()) {
          throw std::invalid_argument(
              "[_expert_ssd_direct_load_into_many] invalid handle or row lists");
        }
        auto rows =
            validate_official_direct_destinations(direct, destinations);
        for (auto slot : slots) {
          if (slot >= rows.capacity) {
            throw std::out_of_range(
                "[_expert_ssd_direct_load_into_many] destination slot is out of range");
          }
        }
        nb::gil_scoped_release release;
        std::vector<char*> pointers(rows.bases.size());
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          for (size_t index = 0; index < pointers.size(); ++index) {
            pointers[index] =
                rows.bases[index] + slots[item] * rows.row_nbytes[index];
          }
          direct->load_ordered_into(
              expert_ids[item], pointers, rows.row_nbytes);
        }
      },
      "direct"_a,
      "expert_ids"_a,
      "slots"_a,
      "destinations"_a,
      nb::sig(
          "def _expert_ssd_direct_load_into_many(direct: _ExpertSafetensorsDirect, expert_ids: list[int], slots: list[int], destinations: list[array]) -> None"));
  m.def(
      "_scalex_mode_a_load_into_many",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         const std::vector<size_t>& expert_ids,
         const std::vector<size_t>& slots,
         std::vector<mx::array> destinations) {
        if (!direct || expert_ids.size() != slots.size() ||
            destinations.size() != 3) {
          throw std::invalid_argument(
              "[_scalex_mode_a_load_into_many] invalid handle or row lists");
        }
        const auto& expected = direct->decoded_tensor_nbytes();
        std::array<char*, 3> bases{};
        std::array<size_t, 3> row_nbytes{};
        size_t capacity = 0;
        for (size_t tensor = 0; tensor < destinations.size(); ++tensor) {
          auto& destination = destinations[tensor];
          if (destination.ndim() < 1 || destination.dtype() != mx::uint8) {
            throw std::invalid_argument(
                "[_scalex_mode_a_load_into_many] destinations must be U8 row arrays");
          }
          const auto tensor_capacity = static_cast<size_t>(destination.shape(0));
          if (tensor_capacity == 0 ||
              (tensor != 0 && tensor_capacity != capacity)) {
            throw std::invalid_argument(
                "[_scalex_mode_a_load_into_many] incompatible destination capacities");
          }
          capacity = tensor_capacity;
          if (!destination.is_available()) {
            destination.eval();
          }
          if (!destination.flags().row_contiguous) {
            throw std::invalid_argument(
                "[_scalex_mode_a_load_into_many] destinations must be row-contiguous");
          }
          row_nbytes[tensor] = destination.nbytes() / capacity;
          if (row_nbytes[tensor] != expected[tensor]) {
            throw std::invalid_argument(
                "[_scalex_mode_a_load_into_many] destination row size changed");
          }
          bases[tensor] = destination.data<char>();
        }
        for (auto slot : slots) {
          if (slot >= capacity) {
            throw std::out_of_range(
                "[_scalex_mode_a_load_into_many] destination slot is out of range");
          }
        }
        nb::gil_scoped_release release;
        std::array<char*, 3> pointers{};
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          for (size_t tensor = 0; tensor < pointers.size(); ++tensor) {
            pointers[tensor] =
                bases[tensor] + slots[item] * row_nbytes[tensor];
          }
          direct->load_into(expert_ids[item], pointers, row_nbytes);
        }
      },
      "direct"_a,
      "expert_ids"_a,
      "slots"_a,
      "destinations"_a,
      nb::sig(
          "def _scalex_mode_a_load_into_many(direct: _ScaleXModeADirect, expert_ids: list[int], slots: list[int], destinations: list[array]) -> None"));
  m.def(
      "_scalex_mode_a_load_experts_into_many",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         const std::vector<size_t>& expert_ids,
         const std::vector<size_t>& slots,
         std::vector<mx::array> scale_destinations,
         std::vector<mx::array> weight_destinations) {
        if (!direct || expert_ids.size() != slots.size()) {
          throw std::invalid_argument(
              "[_scalex_mode_a_load_experts_into_many] invalid handle or row lists");
        }
        const auto scales = validate_scalex_destinations(
            scale_destinations,
            mx::uint8,
            &direct->decoded_tensor_nbytes());
        const auto weights = validate_scalex_destinations(
            weight_destinations, mx::uint32);
        if (weights.capacity != scales.capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_a_load_experts_into_many] scale/weight capacities differ");
        }
        for (auto slot : slots) {
          if (slot >= scales.capacity) {
            throw std::out_of_range(
                "[_scalex_mode_a_load_experts_into_many] destination slot is out of range");
          }
        }
        nb::gil_scoped_release release;
        std::array<char*, 3> scale_pointers{};
        std::array<char*, 3> weight_pointers{};
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          for (size_t tensor = 0; tensor < 3; ++tensor) {
            scale_pointers[tensor] = scales.bases[tensor] +
                slots[item] * scales.row_nbytes[tensor];
            weight_pointers[tensor] = weights.bases[tensor] +
                slots[item] * weights.row_nbytes[tensor];
          }
          direct->load_expert_into(
              expert_ids[item],
              scale_pointers,
              scales.row_nbytes,
              weight_pointers,
              weights.row_nbytes);
        }
      },
      "direct"_a,
      "expert_ids"_a,
      "slots"_a,
      "scale_destinations"_a,
      "weight_destinations"_a,
      nb::sig(
          "def _scalex_mode_a_load_experts_into_many(direct: _ScaleXModeADirect, expert_ids: list[int], slots: list[int], scale_destinations: list[array], weight_destinations: list[array]) -> None"));
  m.def(
      "_scalex_mode_a_load_experts_async",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         std::vector<size_t> expert_ids,
         std::vector<size_t> slots,
         std::vector<mx::array> scale_destinations,
         std::vector<mx::array> weight_destinations,
         size_t worker_count,
         bool interactive_qos,
         std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
         uint64_t wait_value,
         uint64_t event_value) {
        if (!direct || !event_state || event_value == 0 ||
            expert_ids.empty() || expert_ids.size() != slots.size() ||
            worker_count == 0 || wait_value >= event_value) {
          throw std::invalid_argument(
              "[_scalex_mode_a_load_experts_async] invalid arguments");
        }
        const auto scales = validate_scalex_destinations(
            scale_destinations,
            mx::uint8,
            &direct->decoded_tensor_nbytes());
        const auto weights = validate_scalex_destinations(
            weight_destinations, mx::uint32);
        if (weights.capacity != scales.capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_a_load_experts_async] scale/weight capacities differ");
        }
        for (auto slot : slots) {
          if (slot >= scales.capacity) {
            throw std::out_of_range(
                "[_scalex_mode_a_load_experts_async] destination slot is out of range");
          }
        }

        auto state = std::make_shared<ScaleXModeAAsyncBatchState>(
            std::move(direct),
            std::move(expert_ids),
            std::move(slots),
            std::move(scale_destinations),
            std::move(weight_destinations),
            worker_count,
            interactive_qos,
            std::move(event_state),
            wait_value,
            event_value);
        state->scale_bases = scales.bases;
        state->weight_bases = weights.bases;
        state->scale_row_nbytes = scales.row_nbytes;
        state->weight_row_nbytes = weights.row_nbytes;
        dispatch_async_f(
            dispatch_get_global_queue(
                interactive_qos ? QOS_CLASS_USER_INTERACTIVE
                                : QOS_CLASS_USER_INITIATED,
                0),
            new std::shared_ptr<ScaleXModeAAsyncBatchState>(state),
            scalex_mode_a_async_batch_run);
        return state;
      },
      "direct"_a,
      "expert_ids"_a,
      "slots"_a,
      "scale_destinations"_a,
      "weight_destinations"_a,
      "worker_count"_a,
      "interactive_qos"_a,
      "event_state"_a,
      "wait_value"_a,
      "event_value"_a,
      nb::sig(
          "def _scalex_mode_a_load_experts_async(direct: _ScaleXModeADirect, expert_ids: list[int], slots: list[int], scale_destinations: list[array], weight_destinations: list[array], worker_count: int, interactive_qos: bool, event_state: _ExpertSSDIoEventState, wait_value: int, event_value: int) -> _ScaleXModeAAsyncBatchState"));
  m.def(
      "_scalex_mode_a_async_wait",
      [](const std::shared_ptr<ScaleXModeAAsyncBatchState>& state) {
        if (!state) {
          throw std::invalid_argument(
              "[_scalex_mode_a_async_wait] state required");
        }
        expert_ssd_async_wait(state);
      },
      "state"_a,
      nb::sig(
          "def _scalex_mode_a_async_wait(state: _ScaleXModeAAsyncBatchState) -> None"));
  m.def(
      "_scalex_mode_b_load_experts_into_many",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         const std::vector<size_t>& expert_ids,
         const std::vector<size_t>& slots,
         mx::array record_destinations,
         std::vector<mx::array> weight_destinations) {
        if (!direct || expert_ids.size() != slots.size() ||
            record_destinations.ndim() != 2 ||
            record_destinations.dtype() != mx::uint8) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_experts_into_many] invalid handle, rows, or record destination");
        }
        if (!record_destinations.is_available()) {
          record_destinations.eval();
        }
        if (!record_destinations.flags().row_contiguous) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_experts_into_many] record destination must be row-contiguous");
        }
        const size_t capacity =
            static_cast<size_t>(record_destinations.shape(0));
        const size_t record_row_nbytes =
            record_destinations.nbytes() / capacity;
        if (capacity == 0 ||
            record_row_nbytes < direct->maximum_indexed_nbytes()) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_experts_into_many] compressed record row is too small");
        }
        const auto weights = validate_scalex_destinations(
            weight_destinations, mx::uint32);
        if (weights.capacity != capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_experts_into_many] record/weight capacities differ");
        }
        for (auto slot : slots) {
          if (slot >= capacity) {
            throw std::out_of_range(
                "[_scalex_mode_b_load_experts_into_many] destination slot is out of range");
          }
        }
        nb::gil_scoped_release release;
        auto* record_base = record_destinations.data<char>();
        std::array<char*, 3> weight_pointers{};
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          for (size_t tensor = 0; tensor < 3; ++tensor) {
            weight_pointers[tensor] = weights.bases[tensor] +
                slots[item] * weights.row_nbytes[tensor];
          }
          direct->load_compressed_expert_into(
              expert_ids[item],
              record_base + slots[item] * record_row_nbytes,
              record_row_nbytes,
              weight_pointers,
              weights.row_nbytes);
        }
      },
      "direct"_a,
      "expert_ids"_a,
      "slots"_a,
      "record_destinations"_a,
      "weight_destinations"_a,
      nb::sig(
          "def _scalex_mode_b_load_experts_into_many(direct: _ScaleXModeADirect, expert_ids: list[int], slots: list[int], record_destinations: array, weight_destinations: list[array]) -> None"));
  m.def(
      "_scalex_mode_b_load_full_split_into_many",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         const std::vector<size_t>& expert_ids,
         const std::vector<size_t>& gate_up_slots,
         const std::vector<size_t>& down_slots,
         mx::array record_destinations,
         mx::array gate_destinations,
         mx::array down_destinations,
         mx::array up_destinations,
         const std::vector<size_t>& weight_row_nbytes) {
        if (!direct || expert_ids.size() != gate_up_slots.size() ||
            expert_ids.size() != down_slots.size() ||
            weight_row_nbytes.size() != 3 || record_destinations.ndim() != 2 ||
            record_destinations.dtype() != mx::uint8 ||
            gate_destinations.dtype() != mx::uint32 ||
            down_destinations.dtype() != mx::uint32 ||
            up_destinations.dtype() != mx::uint32 ||
            gate_destinations.ndim() != 3 || down_destinations.ndim() != 3 ||
            up_destinations.ndim() != 3) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_into_many] invalid arguments");
        }
        const size_t gate_up_capacity = record_destinations.shape(0);
        const size_t down_capacity = down_destinations.shape(0);
        if (gate_up_capacity == 0 || down_capacity == 0 ||
            gate_destinations.shape(0) != gate_up_capacity ||
            up_destinations.shape(0) != gate_up_capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_into_many] capacities differ");
        }
        for (auto* destination :
             {&record_destinations,
              &gate_destinations,
              &down_destinations,
              &up_destinations}) {
          if (!destination->is_available()) {
            destination->eval();
          }
          if (!destination->flags().row_contiguous) {
            throw std::invalid_argument(
                "[_scalex_mode_b_load_full_split_into_many] destinations must be row-contiguous");
          }
        }
        const size_t record_row_nbytes =
            record_destinations.nbytes() / gate_up_capacity;
        if (record_row_nbytes < direct->maximum_indexed_nbytes() ||
            gate_destinations.nbytes() / gate_up_capacity !=
                weight_row_nbytes[0] ||
            down_destinations.nbytes() / down_capacity !=
                weight_row_nbytes[1] ||
            up_destinations.nbytes() / gate_up_capacity !=
                weight_row_nbytes[2]) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_into_many] row geometry changed");
        }
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          if (gate_up_slots[item] >= gate_up_capacity ||
              down_slots[item] >= down_capacity) {
            throw std::out_of_range(
                "[_scalex_mode_b_load_full_split_into_many] slot is out of range");
          }
        }
        nb::gil_scoped_release release;
        auto* record_base = record_destinations.data<char>();
        auto* gate_base = gate_destinations.data<char>();
        auto* down_base = down_destinations.data<char>();
        auto* up_base = up_destinations.data<char>();
        const std::array<size_t, 3> row_nbytes{
            weight_row_nbytes[0], weight_row_nbytes[1], weight_row_nbytes[2]};
        std::array<char*, 3> pointers{};
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          pointers = {
              gate_base + gate_up_slots[item] * row_nbytes[0],
              down_base + down_slots[item] * row_nbytes[1],
              up_base + gate_up_slots[item] * row_nbytes[2]};
          direct->load_compressed_expert_into(
              expert_ids[item],
              record_base + gate_up_slots[item] * record_row_nbytes,
              record_row_nbytes,
              pointers,
              row_nbytes);
        }
      },
      "direct"_a,
      "expert_ids"_a,
      "gate_up_slots"_a,
      "down_slots"_a,
      "record_destinations"_a,
      "gate_destinations"_a,
      "down_destinations"_a,
      "up_destinations"_a,
      "weight_row_nbytes"_a,
      nb::sig(
          "def _scalex_mode_b_load_full_split_into_many(direct: _ScaleXModeADirect, expert_ids: list[int], gate_up_slots: list[int], down_slots: list[int], record_destinations: array, gate_destinations: array, down_destinations: array, up_destinations: array, weight_row_nbytes: list[int]) -> None"));
  m.def(
      "_expert_ssd_raw_load_split_async",
      [](std::shared_ptr<mx::ExpertSafetensorsDirect> direct,
         std::vector<size_t> expert_ids,
         std::vector<size_t> gate_up_slots,
         std::vector<size_t> down_slots,
         std::vector<mx::array> destinations,
         size_t worker_count,
         bool interactive_qos,
         std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
         uint64_t event_value,
         bool trace_enabled) {
        if (!direct || !event_state || !event_value || !worker_count ||
            expert_ids.empty() || expert_ids.size() != gate_up_slots.size() ||
            expert_ids.size() != down_slots.size()) {
          throw std::invalid_argument("[ExpertSSD raw batch] invalid arguments");
        }
        auto rows = validate_official_direct_destinations(direct, destinations);
        for (size_t index = 0; index < expert_ids.size(); ++index) {
          if (gate_up_slots[index] >= rows.capacity || down_slots[index] >= rows.capacity) {
            throw std::out_of_range("[ExpertSSD raw batch] row is out of range");
          }
        }
        auto state = std::make_shared<ExpertSSDAsyncBatchState>(
            std::move(direct), std::move(expert_ids), std::move(gate_up_slots),
            std::move(down_slots), std::move(destinations), std::move(rows),
            worker_count, interactive_qos, std::move(event_state), event_value);
        if (trace_enabled) state->read_trace.resize(state->expert_ids.size());
        dispatch_async_f(
            dispatch_get_global_queue(
                interactive_qos ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_USER_INITIATED, 0),
            new std::shared_ptr<ExpertSSDAsyncBatchState>(state), scalex_async_batch_run);
        return state;
      },
      "direct"_a, "expert_ids"_a, "gate_up_slots"_a, "down_slots"_a,
      "destinations"_a, "worker_count"_a, "interactive_qos"_a,
      "event_state"_a, "event_value"_a, "trace_enabled"_a = false);
  m.def(
      "_scalex_mode_b_load_full_split_async",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         std::vector<size_t> expert_ids,
         std::vector<size_t> gate_up_slots,
         std::vector<size_t> down_slots,
         mx::array record_destinations,
         mx::array gate_destinations,
         mx::array down_destinations,
         mx::array up_destinations,
         const std::vector<size_t>& weight_row_nbytes,
         size_t worker_count,
         bool interactive_qos,
         std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
         uint64_t event_value,
         bool trace_enabled,
         uint64_t wait_value) {
        if (!direct || !event_state || event_value == 0 ||
            wait_value >= event_value ||
            expert_ids.empty() || expert_ids.size() != gate_up_slots.size() ||
            expert_ids.size() != down_slots.size() ||
            weight_row_nbytes.size() != 3 || worker_count == 0 ||
            record_destinations.ndim() != 2 ||
            record_destinations.dtype() != mx::uint8 ||
            gate_destinations.dtype() != mx::uint32 ||
            down_destinations.dtype() != mx::uint32 ||
            up_destinations.dtype() != mx::uint32 ||
            gate_destinations.ndim() != 3 || down_destinations.ndim() != 3 ||
            up_destinations.ndim() != 3) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_async] invalid arguments");
        }
        const size_t gate_up_capacity = record_destinations.shape(0);
        const size_t down_capacity = down_destinations.shape(0);
        if (gate_up_capacity == 0 || down_capacity == 0 ||
            gate_destinations.shape(0) != gate_up_capacity ||
            up_destinations.shape(0) != gate_up_capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_async] capacities differ");
        }
        for (auto* destination :
             {&record_destinations,
              &gate_destinations,
              &down_destinations,
              &up_destinations}) {
          if (!destination->is_available()) {
            destination->eval();
          }
          if (!destination->flags().row_contiguous) {
            throw std::invalid_argument(
                "[_scalex_mode_b_load_full_split_async] destinations must be row-contiguous");
          }
        }
        const size_t record_row_nbytes =
            record_destinations.nbytes() / gate_up_capacity;
        if (record_row_nbytes < direct->maximum_indexed_nbytes() ||
            gate_destinations.nbytes() / gate_up_capacity !=
                weight_row_nbytes[0] ||
            down_destinations.nbytes() / down_capacity !=
                weight_row_nbytes[1] ||
            up_destinations.nbytes() / gate_up_capacity !=
                weight_row_nbytes[2]) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_async] row geometry changed");
        }
        for (size_t item = 0; item < expert_ids.size(); ++item) {
          if (gate_up_slots[item] >= gate_up_capacity ||
              down_slots[item] >= down_capacity) {
            throw std::out_of_range(
                "[_scalex_mode_b_load_full_split_async] slot is out of range");
          }
        }

        auto state = std::make_shared<ExpertSSDAsyncBatchState>(
            std::move(direct),
            std::move(expert_ids),
            std::move(gate_up_slots),
            std::move(down_slots),
            std::move(record_destinations),
            std::move(gate_destinations),
            std::move(down_destinations),
            std::move(up_destinations),
            worker_count,
            interactive_qos,
            std::move(event_state),
            event_value,
            wait_value);
        state->record_row_nbytes = record_row_nbytes;
        if (trace_enabled) state->read_trace.resize(
            state->expert_ids.size() * (state->direct->replica_count() == 2 ? 2 : 1));
        state->row_nbytes = {
            weight_row_nbytes[0],
            weight_row_nbytes[1],
            weight_row_nbytes[2]};
        state->record_base = state->record_destinations.data<char>();
        state->gate_base = state->gate_destinations.data<char>();
        state->down_base = state->down_destinations.data<char>();
        state->up_base = state->up_destinations.data<char>();
        dispatch_async_f(
            dispatch_get_global_queue(
                interactive_qos ? QOS_CLASS_USER_INTERACTIVE
                                : QOS_CLASS_USER_INITIATED,
                0),
            new std::shared_ptr<ExpertSSDAsyncBatchState>(state),
            scalex_async_batch_run);
        return state;
      },
      "direct"_a,
      "expert_ids"_a,
      "gate_up_slots"_a,
      "down_slots"_a,
      "record_destinations"_a,
      "gate_destinations"_a,
      "down_destinations"_a,
      "up_destinations"_a,
      "weight_row_nbytes"_a,
      "worker_count"_a,
      "interactive_qos"_a,
      "event_state"_a,
      "event_value"_a,
      "trace_enabled"_a = false,
      "wait_value"_a = 0,
      nb::sig(
          "def _scalex_mode_b_load_full_split_async(direct: _ScaleXModeADirect, expert_ids: list[int], gate_up_slots: list[int], down_slots: list[int], record_destinations: array, gate_destinations: array, down_destinations: array, up_destinations: array, weight_row_nbytes: list[int], worker_count: int, interactive_qos: bool, event_state: _ExpertSSDIoEventState, event_value: int, trace_enabled: bool = False, wait_value: int = 0) -> _ScaleXAsyncBatchState"));
  m.def(
      "_scalex_mode_b_async_wait",
      [](const std::shared_ptr<ExpertSSDAsyncBatchState>& state) {
        if (!state) {
          throw std::invalid_argument(
              "[_scalex_mode_b_async_wait] state required");
        }
        expert_ssd_async_wait(state);
      },
      "state"_a,
      nb::sig(
          "def _scalex_mode_b_async_wait(state: _ScaleXAsyncBatchState) -> None"));
  m.def(
      "_scalex_mode_b_load_full_split_two_bank_async",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         std::vector<size_t> expert_ids,
         std::vector<uint8_t> shared_bank,
         std::vector<size_t> rows,
         mx::array private_record,
         mx::array private_gate,
         mx::array private_down,
         mx::array private_up,
         mx::array shared_record,
         mx::array shared_gate,
         mx::array shared_down,
         mx::array shared_up,
         const std::vector<size_t>& weight_row_nbytes,
         size_t worker_count,
         bool interactive_qos,
         std::shared_ptr<mx::ExpertSSDIoEventState> event_state,
         uint64_t event_value,
         bool trace_enabled) {
        if (!direct || !event_state || event_value == 0 ||
            expert_ids.empty() || expert_ids.size() != shared_bank.size() ||
            expert_ids.size() != rows.size() || weight_row_nbytes.size() != 3 ||
            worker_count == 0) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_two_bank_async] invalid arguments");
        }
        std::array<mx::array, 4> private_destinations{
            std::move(private_record),
            std::move(private_gate),
            std::move(private_down),
            std::move(private_up)};
        std::array<mx::array, 4> shared_destinations{
            std::move(shared_record),
            std::move(shared_gate),
            std::move(shared_down),
            std::move(shared_up)};
        const auto validate_bank = [&](std::array<mx::array, 4>& destinations) {
          if (destinations[0].ndim() != 2 ||
              destinations[0].dtype() != mx::uint8 ||
              destinations[1].dtype() != mx::uint32 ||
              destinations[2].dtype() != mx::uint32 ||
              destinations[3].dtype() != mx::uint32 ||
              destinations[1].ndim() != 3 || destinations[2].ndim() != 3 ||
              destinations[3].ndim() != 3) {
            throw std::invalid_argument(
                "[_scalex_mode_b_load_full_split_two_bank_async] invalid bank arrays");
          }
          const size_t capacity = destinations[0].shape(0);
          if (capacity == 0 || destinations[1].shape(0) != capacity ||
              destinations[2].shape(0) != capacity ||
              destinations[3].shape(0) != capacity) {
            throw std::invalid_argument(
                "[_scalex_mode_b_load_full_split_two_bank_async] bank capacities differ");
          }
          for (auto& destination : destinations) {
            if (!destination.is_available()) {
              destination.eval();
            }
            if (!destination.flags().row_contiguous) {
              throw std::invalid_argument(
                  "[_scalex_mode_b_load_full_split_two_bank_async] destinations must be row-contiguous");
            }
          }
          const size_t record_row_nbytes =
              destinations[0].nbytes() / capacity;
          if (record_row_nbytes < direct->maximum_indexed_nbytes() ||
              destinations[1].nbytes() / capacity != weight_row_nbytes[0] ||
              destinations[2].nbytes() / capacity != weight_row_nbytes[1] ||
              destinations[3].nbytes() / capacity != weight_row_nbytes[2]) {
            throw std::invalid_argument(
                "[_scalex_mode_b_load_full_split_two_bank_async] row geometry changed");
          }
          return record_row_nbytes;
        };
        const auto private_record_nbytes = validate_bank(private_destinations);
        const auto shared_record_nbytes = validate_bank(shared_destinations);
        const auto private_capacity = private_destinations[0].shape(0);
        const auto shared_capacity = shared_destinations[0].shape(0);
        for (size_t item = 0; item < rows.size(); ++item) {
          if (rows[item] >=
              (shared_bank[item] ? shared_capacity : private_capacity)) {
            throw std::out_of_range(
                "[_scalex_mode_b_load_full_split_two_bank_async] row is out of range");
          }
        }
        auto state = std::make_shared<ScaleXTwoBankAsyncBatchState>(
            std::move(direct),
            std::move(expert_ids),
            std::move(shared_bank),
            std::move(rows),
            std::move(private_destinations),
            std::move(shared_destinations),
            worker_count,
            interactive_qos,
            std::move(event_state),
            event_value);
        state->private_record_row_nbytes = private_record_nbytes;
        if (trace_enabled) state->read_trace.resize(
            state->expert_ids.size() * (state->direct->replica_count() == 2 ? 2 : 1));
        state->shared_record_row_nbytes = shared_record_nbytes;
        state->row_nbytes = {
            weight_row_nbytes[0],
            weight_row_nbytes[1],
            weight_row_nbytes[2]};
        for (size_t index = 0; index < 4; ++index) {
          state->private_bases[index] =
              state->private_destinations[index].data<char>();
          state->shared_bases[index] =
              state->shared_destinations[index].data<char>();
        }
        dispatch_async_f(
            dispatch_get_global_queue(
                interactive_qos ? QOS_CLASS_USER_INTERACTIVE
                                : QOS_CLASS_USER_INITIATED,
                0),
            new std::shared_ptr<ScaleXTwoBankAsyncBatchState>(state),
            scalex_two_bank_batch_run);
        return state;
      },
      "direct"_a,
      "expert_ids"_a,
      "shared_bank"_a,
      "rows"_a,
      "private_record"_a,
      "private_gate"_a,
      "private_down"_a,
      "private_up"_a,
      "shared_record"_a,
      "shared_gate"_a,
      "shared_down"_a,
      "shared_up"_a,
      "weight_row_nbytes"_a,
      "worker_count"_a,
      "interactive_qos"_a,
      "event_state"_a,
      "event_value"_a,
      nb::sig(
          "def _scalex_mode_b_load_full_split_two_bank_async(direct: _ScaleXModeADirect, expert_ids: list[int], shared_bank: list[bool], rows: list[int], private_record: array, private_gate: array, private_down: array, private_up: array, shared_record: array, shared_gate: array, shared_down: array, shared_up: array, weight_row_nbytes: list[int], worker_count: int, interactive_qos: bool, event_state: _ExpertSSDIoEventState, event_value: int, trace_enabled: bool = False) -> _ScaleXTwoBankAsyncBatchState"),
      "trace_enabled"_a = false);
  m.def(
      "_scalex_mode_b_two_bank_async_wait",
      [](const std::shared_ptr<ScaleXTwoBankAsyncBatchState>& state) {
        if (!state) {
          throw std::invalid_argument(
              "[_scalex_mode_b_two_bank_async_wait] state required");
        }
        expert_ssd_async_wait(state);
      },
      "state"_a,
      nb::sig(
          "def _scalex_mode_b_two_bank_async_wait(state: _ScaleXTwoBankAsyncBatchState) -> None"));
  m.def(
      "_scalex_mode_b_load_full_split_chunk",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         size_t expert_id,
         size_t gate_up_slot,
         size_t down_slot,
         size_t logical_offset,
         size_t maximum_bytes,
         mx::array record_destinations,
         mx::array gate_destinations,
         mx::array down_destinations,
         mx::array up_destinations,
         const std::vector<size_t>& weight_row_nbytes) {
        if (!direct || weight_row_nbytes.size() != 3 ||
            record_destinations.ndim() != 2 ||
            record_destinations.dtype() != mx::uint8 ||
            gate_destinations.dtype() != mx::uint32 ||
            down_destinations.dtype() != mx::uint32 ||
            up_destinations.dtype() != mx::uint32 ||
            gate_destinations.ndim() != 3 || down_destinations.ndim() != 3 ||
            up_destinations.ndim() != 3) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_chunk] invalid arguments");
        }
        const size_t gate_up_capacity = record_destinations.shape(0);
        const size_t down_capacity = down_destinations.shape(0);
        if (gate_up_capacity == 0 || down_capacity == 0 ||
            gate_up_slot >= gate_up_capacity || down_slot >= down_capacity ||
            gate_destinations.shape(0) != gate_up_capacity ||
            up_destinations.shape(0) != gate_up_capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_chunk] invalid capacities or rows");
        }
        for (auto* destination :
             {&record_destinations,
              &gate_destinations,
              &down_destinations,
              &up_destinations}) {
          if (!destination->is_available()) {
            destination->eval();
          }
          if (!destination->flags().row_contiguous) {
            throw std::invalid_argument(
                "[_scalex_mode_b_load_full_split_chunk] destinations must be row-contiguous");
          }
        }
        const size_t record_row_nbytes =
            record_destinations.nbytes() / gate_up_capacity;
        if (record_row_nbytes < direct->maximum_indexed_nbytes() ||
            gate_destinations.nbytes() / gate_up_capacity != weight_row_nbytes[0] ||
            down_destinations.nbytes() / down_capacity != weight_row_nbytes[1] ||
            up_destinations.nbytes() / gate_up_capacity != weight_row_nbytes[2]) {
          throw std::invalid_argument(
              "[_scalex_mode_b_load_full_split_chunk] row geometry changed");
        }
        auto* record_base = record_destinations.data<char>();
        auto* gate_base = gate_destinations.data<char>();
        auto* down_base = down_destinations.data<char>();
        auto* up_base = up_destinations.data<char>();
        const std::array<char*, 3> pointers{
            gate_base + gate_up_slot * weight_row_nbytes[0],
            down_base + down_slot * weight_row_nbytes[1],
            up_base + gate_up_slot * weight_row_nbytes[2]};
        const std::array<size_t, 3> row_nbytes{
            weight_row_nbytes[0], weight_row_nbytes[1], weight_row_nbytes[2]};
        nb::gil_scoped_release release;
        return direct->load_compressed_expert_chunk_into(
            expert_id,
            logical_offset,
            maximum_bytes,
            record_base + gate_up_slot * record_row_nbytes,
            record_row_nbytes,
            pointers,
            row_nbytes);
      },
      "direct"_a,
      "expert_id"_a,
      "gate_up_slot"_a,
      "down_slot"_a,
      "logical_offset"_a,
      "maximum_bytes"_a,
      "record_destinations"_a,
      "gate_destinations"_a,
      "down_destinations"_a,
      "up_destinations"_a,
      "weight_row_nbytes"_a,
      nb::sig(
          "def _scalex_mode_b_load_full_split_chunk(direct: _ScaleXModeADirect, expert_id: int, gate_up_slot: int, down_slot: int, logical_offset: int, maximum_bytes: int, record_destinations: array, gate_destinations: array, down_destinations: array, up_destinations: array, weight_row_nbytes: list[int]) -> int"));
  m.def(
      "_scalex_mode_b_copy_mapped_full_split",
      [](std::shared_ptr<mx::ScaleXModeADirect> direct,
         size_t expert_id,
         size_t gate_up_slot,
         size_t down_slot,
         mx::array record_destinations,
         mx::array gate_destinations,
         mx::array down_destinations,
         mx::array up_destinations,
         const std::vector<size_t>& weight_row_nbytes) {
        if (!direct || weight_row_nbytes.size() != 3 ||
            record_destinations.ndim() != 2 ||
            record_destinations.dtype() != mx::uint8 ||
            gate_destinations.dtype() != mx::uint32 ||
            down_destinations.dtype() != mx::uint32 ||
            up_destinations.dtype() != mx::uint32 ||
            gate_destinations.ndim() != 3 || down_destinations.ndim() != 3 ||
            up_destinations.ndim() != 3) {
          throw std::invalid_argument(
              "[_scalex_mode_b_copy_mapped_full_split] invalid arguments");
        }
        const size_t gate_up_capacity = record_destinations.shape(0);
        const size_t down_capacity = down_destinations.shape(0);
        if (gate_up_capacity == 0 || down_capacity == 0 ||
            gate_up_slot >= gate_up_capacity || down_slot >= down_capacity ||
            gate_destinations.shape(0) != gate_up_capacity ||
            up_destinations.shape(0) != gate_up_capacity) {
          throw std::invalid_argument(
              "[_scalex_mode_b_copy_mapped_full_split] invalid capacities or rows");
        }
        for (auto* destination :
             {&record_destinations,
              &gate_destinations,
              &down_destinations,
              &up_destinations}) {
          if (!destination->is_available()) {
            destination->eval();
          }
          if (!destination->flags().row_contiguous) {
            throw std::invalid_argument(
                "[_scalex_mode_b_copy_mapped_full_split] destinations must be row-contiguous");
          }
        }
        const size_t record_row_nbytes =
            record_destinations.nbytes() / gate_up_capacity;
        if (record_row_nbytes < direct->maximum_indexed_nbytes() ||
            gate_destinations.nbytes() / gate_up_capacity != weight_row_nbytes[0] ||
            down_destinations.nbytes() / down_capacity != weight_row_nbytes[1] ||
            up_destinations.nbytes() / gate_up_capacity != weight_row_nbytes[2]) {
          throw std::invalid_argument(
              "[_scalex_mode_b_copy_mapped_full_split] row geometry changed");
        }
        auto* record_base = record_destinations.data<char>();
        auto* gate_base = gate_destinations.data<char>();
        auto* down_base = down_destinations.data<char>();
        auto* up_base = up_destinations.data<char>();
        const std::array<char*, 3> pointers{
            gate_base + gate_up_slot * weight_row_nbytes[0],
            down_base + down_slot * weight_row_nbytes[1],
            up_base + gate_up_slot * weight_row_nbytes[2]};
        const std::array<size_t, 3> row_nbytes{
            weight_row_nbytes[0], weight_row_nbytes[1], weight_row_nbytes[2]};
        nb::gil_scoped_release release;
        direct->copy_mapped_compressed_expert_into(
            expert_id,
            record_base + gate_up_slot * record_row_nbytes,
            record_row_nbytes,
            pointers,
            row_nbytes);
      },
      "direct"_a,
      "expert_id"_a,
      "gate_up_slot"_a,
      "down_slot"_a,
      "record_destinations"_a,
      "gate_destinations"_a,
      "down_destinations"_a,
      "up_destinations"_a,
      "weight_row_nbytes"_a,
      nb::sig(
          "def _scalex_mode_b_copy_mapped_full_split(direct: _ScaleXModeADirect, expert_id: int, gate_up_slot: int, down_slot: int, record_destinations: array, gate_destinations: array, down_destinations: array, up_destinations: array, weight_row_nbytes: list[int]) -> None"));
  m.def(
      "_expert_ssd_copy_rows",
      [](std::vector<mx::array> sources,
         const std::vector<size_t>& source_rows,
         std::vector<mx::array> destinations,
         const std::vector<size_t>& destination_rows) {
        if (sources.empty() || sources.size() != destinations.size() ||
            source_rows.size() != destination_rows.size()) {
          throw std::invalid_argument(
              "[_expert_ssd_copy_rows] incompatible tensor or row lists");
        }
        struct RowCopy {
          const char* source;
          char* destination;
          size_t nbytes;
        };
        std::vector<RowCopy> copies;
        copies.reserve(sources.size() * source_rows.size());
        size_t copied_bytes = 0;
        for (size_t tensor = 0; tensor < sources.size(); ++tensor) {
          auto& source = sources[tensor];
          auto& destination = destinations[tensor];
          if (!source.is_available()) {
            source.eval();
          }
          if (!destination.is_available()) {
            destination.eval();
          }
          if (source.ndim() < 1 || destination.ndim() != source.ndim() ||
              source.dtype() != destination.dtype() ||
              !std::equal(
                  source.shape().begin() + 1,
                  source.shape().end(),
                  destination.shape().begin() + 1) ||
              !source.flags().row_contiguous ||
              !destination.flags().row_contiguous) {
            throw std::invalid_argument(
                "[_expert_ssd_copy_rows] source/destination layout mismatch");
          }
          const auto source_capacity = static_cast<size_t>(source.shape(0));
          const auto destination_capacity =
              static_cast<size_t>(destination.shape(0));
          const auto source_row_nbytes = source.nbytes() / source_capacity;
          const auto destination_row_nbytes =
              destination.nbytes() / destination_capacity;
          if (source_row_nbytes != destination_row_nbytes) {
            throw std::invalid_argument(
                "[_expert_ssd_copy_rows] row byte sizes differ");
          }
          for (size_t row = 0; row < source_rows.size(); ++row) {
            if (source_rows[row] >= source_capacity ||
                destination_rows[row] >= destination_capacity) {
              throw std::out_of_range(
                  "[_expert_ssd_copy_rows] row is out of range");
            }
            const auto* source_pointer =
                source.data<char>() + source_rows[row] * source_row_nbytes;
            auto* destination_pointer = destination.data<char>() +
                destination_rows[row] * destination_row_nbytes;
            if (source_pointer == destination_pointer) {
              throw std::invalid_argument(
                  "[_expert_ssd_copy_rows] source and destination rows alias");
            }
            copies.push_back(
                {source_pointer, destination_pointer, source_row_nbytes});
            copied_bytes += source_row_nbytes;
          }
        }
        {
          nb::gil_scoped_release release;
          for (const auto& copy : copies) {
            std::memcpy(copy.destination, copy.source, copy.nbytes);
          }
        }
        return copied_bytes;
      },
      "sources"_a,
      "source_rows"_a,
      "destinations"_a,
      "destination_rows"_a,
      nb::sig(
          "def _expert_ssd_copy_rows(sources: list[array], source_rows: list[int], destinations: list[array], destination_rows: list[int]) -> int"),
      R"pbdoc(
        Copy evaluated, row-contiguous expert tensors between disjoint MLX
        buffers. Callers must finish every GPU reader of both row sets first.
      )pbdoc");
  m.def(
      "_expert_ssd_wire_arrays",
      [](std::vector<mx::array> arrays) {
        size_t wired = 0;
        for (auto& array : arrays) {
          if (!array.is_available()) {
            array.eval();
          }
          if (!array.flags().row_contiguous) {
            throw std::invalid_argument(
                "[_expert_ssd_wire_arrays] arrays must be row-contiguous");
          }
          nb::gil_scoped_release release;
          if (::mlock(array.data<char>(), array.nbytes()) == 0) {
            wired += array.nbytes();
          }
        }
        return wired;
      },
      "arrays"_a,
      nb::sig("def _expert_ssd_wire_arrays(arrays: list[array]) -> int"),
      R"pbdoc(
        Best-effort wire evaluated slot-pool buffers so macOS cannot compress
        or page out resident expert rows between Metal layer uses. Returns the
        number of bytes successfully wired.
      )pbdoc");
  m.def(
      "_expert_ssd_unwire_arrays",
      [](std::vector<mx::array> arrays) {
        size_t unwired = 0;
        for (auto& array : arrays) {
          if (!array.is_available() || !array.flags().row_contiguous) {
            continue;
          }
          nb::gil_scoped_release release;
          if (::munlock(array.data<char>(), array.nbytes()) == 0) {
            unwired += array.nbytes();
          }
        }
        return unwired;
      },
      "arrays"_a,
      nb::sig("def _expert_ssd_unwire_arrays(arrays: list[array]) -> int"));
  m.def(
      "reshape",
      &mx::reshape,
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def reshape(a: array, /, shape: Sequence[int], *, stream: "
          "StreamOrDevice = None) -> array"),
      R"pbdoc(
        Reshape an array while preserving the size.

        Args:
            a (array): Input array.
            shape (tuple(int)): New shape.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The reshaped array.
      )pbdoc");
  m.def(
      "flatten",
      [](const mx::array& a,
         int start_axis,
         int end_axis,
         const mx::StreamOrDevice& s) {
        return mx::flatten(a, start_axis, end_axis);
      },
      nb::arg(),
      "start_axis"_a = 0,
      "end_axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def flatten(a: array, /, start_axis: int = 0, end_axis: int = "
          "-1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Flatten an array.

      The axes flattened will be between ``start_axis`` and ``end_axis``,
      inclusive. Negative axes are supported. After converting negative axis to
      positive, axes outside the valid range will be clamped to a valid value,
      ``start_axis`` to ``0`` and ``end_axis`` to ``ndim - 1``.

      Args:
          a (array): Input array.
          start_axis (int, optional): The first dimension to flatten. Defaults to ``0``.
          end_axis (int, optional): The last dimension to flatten. Defaults to ``-1``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

      Returns:
          array: The flattened array.

      Example:
          >>> a = mx.array([[1, 2], [3, 4]])
          >>> mx.flatten(a)
          array([1, 2, 3, 4], dtype=int32)
          >>>
          >>> mx.flatten(a, start_axis=0, end_axis=-1)
          array([1, 2, 3, 4], dtype=int32)
  )pbdoc");
  m.def(
      "unflatten",
      &mx::unflatten,
      nb::arg(),
      "axis"_a,
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def unflatten(a: array, /, axis: int, shape: Sequence[int], *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Unflatten an axis of an array to a shape.

      Args:
          a (array): Input array.
          axis (int): The axis to unflatten.
          shape (tuple(int)): The shape to unflatten to. At most one
            entry can be ``-1`` in which case the corresponding size will be
            inferred.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

      Returns:
          array: The unflattened array.

      Example:
          >>> a = mx.array([1, 2, 3, 4])
          >>> mx.unflatten(a, 0, (2, -1))
          array([[1, 2], [3, 4]], dtype=int32)
  )pbdoc");
  m.def(
      "squeeze",
      [](const mx::array& a, const IntOrVec& v, const mx::StreamOrDevice& s) {
        if (std::holds_alternative<std::monostate>(v)) {
          return mx::squeeze(a, s);
        } else if (auto pv = std::get_if<int>(&v); pv) {
          return mx::squeeze(a, *pv, s);
        } else {
          return mx::squeeze(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def squeeze(a: array, /, axis: None | int | Sequence[int] = "
          "None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Remove length one axes from an array.

        Args:
            a (array): Input array.
            axis (int or tuple(int), optional): Axes to remove. Defaults
              to ``None`` in which case all size one axes are removed.

        Returns:
            array: The output array with size one axes removed.
      )pbdoc");
  m.def(
      "flip",
      [](const mx::array& a, const IntOrVec& v, const mx::StreamOrDevice& s) {
        if (std::holds_alternative<std::monostate>(v)) {
          return mx::flip(a, s);
        } else if (auto pv = std::get_if<int>(&v); pv) {
          return mx::flip(a, *pv, s);
        } else {
          return mx::flip(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def flip(a: array, /, axis: None | int | Sequence[int] = None, "
          "*, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Reverse the order of elements along the given axis.

        Args:
            a (array): Input array.
            axis (int or tuple(int), optional): Axis or axes to flip over.
              Defaults to ``None`` in which case all axes are flipped.

        Returns:
            array: The flipped array.
      )pbdoc");
  m.def(
      "unstack",
      [](const mx::array& a, int axis, mx::StreamOrDevice s) {
        return mx::unstack(a, axis, s);
      },
      nb::arg(),
      nb::kw_only(),
      "axis"_a = 0,
      "stream"_a = nb::none(),
      nb::sig(
          "def unstack(x: array, /, *, axis: int = 0, stream: StreamOrDevice = None) -> list[array]"),
      R"pbdoc(
        Split an array into a sequence of arrays along the given axis.

        The inverse of :func:`stack`. The given axis is removed from each of
        the returned arrays.

        Args:
            x (array): Input array.
            axis (int, optional): Axis along which to unstack. Default: ``0``.

        Returns:
            list(array): A list of arrays, one for each index along ``axis``.
      )pbdoc");
  m.def(
      "expand_dims",
      [](const mx::array& a,
         const std::variant<int, std::vector<int>>& v,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&v); pv) {
          return mx::expand_dims(a, *pv, s);
        } else {
          return mx::expand_dims(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def expand_dims(a: array, /, axis: int | Sequence[int], "
          "*, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Add a size one dimension at the given axis.

        Args:
            a (array): Input array.
            axes (int or tuple(int)): The index of the inserted dimensions.

        Returns:
            array: The array with inserted dimensions.
      )pbdoc");
  m.def(
      "abs",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::abs(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def abs(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise absolute value.

        Args:
            a (array): Input array.

        Returns:
            array: The absolute value of ``a``.
      )pbdoc");
  m.def(
      "sign",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sign(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sign(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise sign.

        Args:
            a (array): Input array.

        Returns:
            array: The sign of ``a``.
      )pbdoc");
  m.def(
      "positive",
      &mx::positive,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def positive(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise unary plus. Returns a copy of the input.

        Args:
            a (array): Input array.

        Returns:
            array: A copy of ``a``.
      )pbdoc");
  m.def(
      "negative",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::negative(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def negative(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise negation.

        Args:
            a (array): Input array.

        Returns:
            array: The negative of ``a``.
      )pbdoc");
  m.def(
      "add",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::add(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def add(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise addition.

        Add two arrays with numpy-style broadcasting semantics. Either or both input arrays
        can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The sum of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "subtract",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::subtract(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def subtract(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise subtraction.

        Subtract one array from another with numpy-style broadcasting semantics. Either or both
        input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The difference ``a - b``.
      )pbdoc");
  m.def(
      "multiply",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::multiply(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def multiply(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise multiplication.

        Multiply two arrays with numpy-style broadcasting semantics. Either or both
        input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The multiplication ``a * b``.
      )pbdoc");
  m.def(
      "divide",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def divide(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise division.

        Divide two arrays with numpy-style broadcasting semantics. Either or both
        input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The quotient ``a / b``.
      )pbdoc");
  m.def(
      "divmod",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::divmod(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def divmod(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise quotient and remainder.

        The fuction ``divmod(a, b)`` is equivalent to but faster than
        ``(a // b, a % b)``. The function uses numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            tuple(array, array): The quotient ``a // b`` and remainder ``a % b``.
      )pbdoc");
  m.def(
      "floor_divide",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::floor_divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def floor_divide(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise integer division.

        If either array is a floating point type then it is equivalent to
        calling :func:`floor` after :func:`divide`.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The quotient ``a // b``.
      )pbdoc");
  m.def(
      "remainder",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::remainder(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def remainder(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise remainder of division.

        Computes the remainder of dividing a with b with numpy-style
        broadcasting semantics. Either or both input arrays can also be
        scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The remainder of ``a // b``.
      )pbdoc");
  m.def(
      "equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def equal(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise equality.

        Equality comparison on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a == b``.
      )pbdoc");
  m.def(
      "not_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::not_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def not_equal(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise not equal.

        Not equal comparison on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a != b``.
      )pbdoc");
  m.def(
      "less",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::less(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def less(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise less than.

        Strict less than on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a < b``.
      )pbdoc");
  m.def(
      "less_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::less_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def less_equal(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise less than or equal.

        Less than or equal on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a <= b``.
      )pbdoc");
  m.def(
      "greater",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::greater(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def greater(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise greater than.

        Strict greater than on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a > b``.
      )pbdoc");
  m.def(
      "greater_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::greater_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def greater_equal(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise greater or equal.

        Greater than or equal on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a >= b``.
      )pbdoc");
  m.def(
      "array_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         bool equal_nan,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::array_equal(a, b, equal_nan, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def array_equal(a: scalar | array, b: scalar | array, equal_nan: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Array equality check.

        Compare two arrays for equality. Returns ``True`` if and only if the arrays
        have the same shape and their values are equal. The arrays need not have
        the same type to be considered equal.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            equal_nan (bool): If ``True``, NaNs are considered equal.
              Defaults to ``False``.

        Returns:
            array: A scalar boolean array.
      )pbdoc");
  m.def(
      "matmul",
      &mx::matmul,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def matmul(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Matrix multiplication.

        Perform the (possibly batched) matrix multiplication of two arrays. This function supports
        broadcasting for arrays with more than two dimensions.

        - If the first array is 1-D then a 1 is prepended to its shape to make it
          a matrix. Similarly if the second array is 1-D then a 1 is appended to its
          shape to make it a matrix. In either case the singleton dimension is removed
          from the result.
        - A batched matrix multiplication is performed if the arrays have more than
          2 dimensions.  The matrix dimensions for the matrix product are the last
          two dimensions of each input.
        - All but the last two dimensions of each input are broadcast with one another using
          standard numpy-style broadcasting semantics.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The matrix product of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "trunc",
      &mx::trunc,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def trunc(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise truncation towards zero.

        Args:
            a (array): Input array.

        Returns:
            array: The truncated array.
      )pbdoc");
  m.def(
      "square",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::square(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def square(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise square.

        Args:
            a (array): Input array.

        Returns:
            array: The square of ``a``.
      )pbdoc");
  m.def(
      "sqrt",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sqrt(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise square root.

        Args:
            a (array): Input array.

        Returns:
            array: The square root of ``a``.
      )pbdoc");
  m.def(
      "rsqrt",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::rsqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def rsqrt(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise reciprocal and square root.

        Args:
            a (array): Input array.

        Returns:
            array: One over the square root of ``a``.
      )pbdoc");
  m.def(
      "reciprocal",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::reciprocal(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def reciprocal(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise reciprocal.

        Args:
            a (array): Input array.

        Returns:
            array: The reciprocal of ``a``.
      )pbdoc");
  m.def(
      "logical_not",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::logical_not(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_not(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise logical not.

        Args:
            a (array): Input array or scalar.

        Returns:
            array: The boolean array containing the logical not of ``a``.
      )pbdoc");
  m.def(
      "logical_and",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_and(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_and(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise logical and.

        Args:
            a (array): First input array or scalar.
            b (array): Second input array or scalar.

        Returns:
            array: The boolean array containing the logical and of ``a`` and ``b``.
    )pbdoc");

  m.def(
      "logical_or",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_or(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_or(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise logical or.

        Args:
            a (array): First input array or scalar.
            b (array): Second input array or scalar.

        Returns:
            array: The boolean array containing the logical or of ``a`` and ``b``.
    )pbdoc");
  m.def(
      "logical_xor",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_xor(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_xor(a: scalar | array, b: scalar | array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise logical exclusive or.

        Args:
            a (array): First input array or scalar.
            b (array): Second input array or scalar.

        Returns:
            array: The boolean array containing the logical xor of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "logaddexp",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::logaddexp(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logaddexp(a: scalar | array, b: scalar | array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise log-add-exp.

        This is a numerically stable log-add-exp of two arrays with numpy-style
        broadcasting semantics. Either or both input arrays can also be scalars.

        The computation is a numerically stable version of ``log(exp(a) + exp(b))``.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The log-add-exp of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "exp",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::exp(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def exp(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise exponential.

        Args:
            a (array): Input array.

        Returns:
            array: The exponential of ``a``.
      )pbdoc");
  m.def(
      "expm1",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::expm1(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def expm1(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise exponential minus 1.

        Computes ``exp(x) - 1`` with greater precision for small ``x``.

        Args:
            a (array): Input array.

        Returns:
            array: The expm1 of ``a``.
      )pbdoc");
  m.def(
      "erf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::erf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def erf(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise error function.

        .. math::
          \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt

        Args:
            a (array): Input array.

        Returns:
            array: The error function of ``a``.
      )pbdoc");
  m.def(
      "erfinv",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::erfinv(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def erfinv(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse of :func:`erf`.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse error function of ``a``.
      )pbdoc");
  m.def(
      "sin",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sin(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise sine.

        Args:
            a (array): Input array.

        Returns:
            array: The sine of ``a``.
      )pbdoc");
  m.def(
      "cos",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::cos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cos(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The cosine of ``a``.
      )pbdoc");
  m.def(
      "tan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::tan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tan(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The tangent of ``a``.
      )pbdoc");
  m.def(
      "arcsin",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arcsin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arcsin(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse sine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse sine of ``a``.
      )pbdoc");
  m.def(
      "arccos",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arccos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arccos(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse cosine of ``a``.
      )pbdoc");
  m.def(
      "arctan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arctan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctan(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse tangent of ``a``.
      )pbdoc");
  m.def(
      "arctan2",
      &mx::arctan2,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctan2(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse tangent of the ratio of two arrays.

        Args:
            a (array): Input array.
            b (array): Input array.

        Returns:
            array: The inverse tangent of the ratio of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "sinh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sinh(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise hyperbolic sine.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic sine of ``a``.
      )pbdoc");
  m.def(
      "cosh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::cosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cosh(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise hyperbolic cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic cosine of ``a``.
      )pbdoc");
  m.def(
      "tanh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::tanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tanh(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise hyperbolic tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic tangent of ``a``.
      )pbdoc");
  m.def(
      "arcsinh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arcsinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arcsinh(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse hyperbolic sine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic sine of ``a``.
      )pbdoc");
  m.def(
      "arccosh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arccosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arccosh(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse hyperbolic cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic cosine of ``a``.
      )pbdoc");
  m.def(
      "arctanh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arctanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctanh(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise inverse hyperbolic tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic tangent of ``a``.
      )pbdoc");
  m.def(
      "degrees",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::degrees(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def degrees(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Convert angles from radians to degrees.

      Args:
          a (array): Input array.

      Returns:
          array: The angles in degrees.
    )pbdoc");
  m.def(
      "radians",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::radians(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def radians(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Convert angles from degrees to radians.

      Args:
          a (array): Input array.

      Returns:
          array: The angles in radians.
    )pbdoc");
  m.def(
      "log",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise natural logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The natural logarithm of ``a``.
      )pbdoc");
  m.def(
      "log2",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log2(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log2(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise base-2 logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The base-2 logarithm of ``a``.
      )pbdoc");
  m.def(
      "log10",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log10(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log10(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise base-10 logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The base-10 logarithm of ``a``.
      )pbdoc");
  m.def(
      "log1p",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log1p(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log1p(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise natural log of one plus the array.

        Args:
            a (array): Input array.

        Returns:
            array: The natural logarithm of one plus ``a``.
      )pbdoc");
  m.def(
      "stop_gradient",
      &mx::stop_gradient,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stop_gradient(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Stop gradients from being computed.

        The operation is the identity but it prevents gradients from flowing
        through the array.

        Args:
            a (array): Input array.

        Returns:
            array:
              The unchanged input ``a`` but without gradient flowing
              through it.
      )pbdoc");
  m.def(
      "sigmoid",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sigmoid(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sigmoid(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise logistic sigmoid.

        The logistic sigmoid function is:

        .. math::
          \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

        Args:
            a (array): Input array.

        Returns:
            array: The logistic sigmoid of ``a``.
      )pbdoc");
  m.def(
      "power",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::power(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def power(a: scalar | array, b: scalar | array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise power operation.

        Raise the elements of a to the powers in elements of b with numpy-style
        broadcasting semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: Bases of ``a`` raised to powers in ``b``.
      )pbdoc");
  m.def(
      "arange",
      [](Scalar start,
         std::optional<Scalar> stop,
         const std::optional<Scalar>& step,
         const std::optional<mx::Dtype>& dtype_,
         mx::StreamOrDevice s) {
        if (!stop) {
          stop = start;
          start = 0;
        }
        // Determine the final dtype based on input types
        mx::Dtype dtype = dtype_
            ? *dtype_
            : mx::promote_types(
                  scalar_to_dtype(start),
                  step ? mx::promote_types(
                             scalar_to_dtype(*stop), scalar_to_dtype(*step))
                       : scalar_to_dtype(*stop));
        return mx::arange(
            scalar_to_double(start),
            scalar_to_double(*stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "start"_a.noconvert(),
      "stop"_a.noconvert(),
      "step"_a.noconvert() = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(start : int | float, stop : None | int | float, step : None | int | float, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Generates ranges of numbers.

      Generate numbers in the half-open interval ``[start, stop)`` in
      increments of ``step``.

      Args:
          start (float or int, optional): Starting value which defaults to ``0``.
          stop (float or int, optional): Stopping value.
          step (float or int, optional): Increment which defaults to ``1``.
          dtype (Dtype, optional): Specifies the data type of the output. If unspecified will default to ``float32`` if any of ``start``, ``stop``, or ``step`` are ``float``. Otherwise will default to ``int32``, or ``int64`` if any of ``start``, ``stop``, or ``step`` does not fit in ``int32``.

      Returns:
          array: The range of values.

      Note:
        Following the Numpy convention the actual increment used to
        generate numbers is ``dtype(start + step) - dtype(start)``.
        This can lead to unexpected results for example if `start + step`
        is a fractional value and the `dtype` is integral.
      )pbdoc");
  m.def(
      "arange",
      [](Scalar stop,
         const std::optional<Scalar>& step,
         const std::optional<mx::Dtype>& dtype_,
         mx::StreamOrDevice s) {
        mx::Dtype dtype = dtype_ ? *dtype_
            : step
            ? mx::promote_types(scalar_to_dtype(stop), scalar_to_dtype(*step))
            : scalar_to_dtype(stop);
        return mx::arange(
            0.0,
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "stop"_a.noconvert(),
      "step"_a.noconvert() = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(stop : int | float, step : None | int | float = None, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"));
  m.def(
      "bartlett",
      &mlx::core::bartlett,
      "M"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"pbdoc(
        Return the Bartlett window.
        
        The Bartlett window is a taper formed by using a weighted cosine.

        .. math::
          w(n) = 1 - \frac{2|n - (M-1)/2|}{M-1}
           \qquad 0 \le n \le M-1
        
        Args:
            M (int): Number of points in the output window.
            
        Returns:
            array: The window, with the maximum value normalized to one (the value one
                   appears only if the number of samples is odd).
    )pbdoc");
  m.def(
      "hanning",
      &mlx::core::hanning,
      "M"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"pbdoc(
        Return the Hanning window.
        
        The Hanning window is a taper formed by using a weighted cosine.

        .. math::
          w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{M-1}\right)
           \qquad 0 \le n \le M-1
        
        Args:
            M (int): Number of points in the output window.
            
        Returns:
            array: The window, with the maximum value normalized to one (the value one
                   appears only if the number of samples is odd).
    )pbdoc");
  m.def(
      "hamming",
      &mlx::core::hamming,
      "M"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def hamming(M: int, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the Hamming window.

        The Hamming window is a taper formed by using a weighted cosine.

        .. math::
           w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi n}{M-1}\right)
           \qquad 0 \le n \le M-1

        Args:
            M (int): Number of points in the output window.

        Returns:
            array: The window, with the maximum value normalized to one (the value one
                   appears only if the number of samples is odd).
    )pbdoc");
  m.def(
      "blackman",
      &mlx::core::blackman,
      "M"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def blackman(M: int, *, stream: StreamOrDevice = None) -> array"), // <--- J'ai rajouté ça
      R"pbdoc(
        Return the Blackman window.
        
        The Blackman window is a taper formed by using the first three terms of a summation of cosines.

        .. math::
          w(n) = 0.42 - 0.5 \cos\left(\frac{2\pi n}{M-1}\right) + 0.08 \cos\left(\frac{4\pi n}{M-1}\right)
           \qquad 0 \le n \le M-1
        
        Args:
            M (int): Number of points in the output window.
            
        Returns:
            array: The window, with the maximum value normalized to one (the value one
                   appears only if the number of samples is odd).
    )pbdoc");
  m.def(
      "linspace",
      [](Scalar start,
         Scalar stop,
         int num,
         bool endpoint,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::linspace(
            scalar_to_double(start),
            scalar_to_double(stop),
            num,
            endpoint,
            dtype.value_or(mx::float32),
            s);
      },
      "start"_a,
      "stop"_a,
      "num"_a = 50,
      "endpoint"_a = true,
      "dtype"_a.none() = mx::float32,
      "stream"_a = nb::none(),
      nb::sig(
          "def linspace(start: scalar, stop: scalar, num: int | None = 50, endpoint: bool = True, dtype: Dtype | None = float32, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Generate ``num`` evenly spaced numbers over interval ``[start, stop]``.

        Args:
            start (scalar): Starting value.
            stop (scalar): Stopping value.
            num (int, optional): Number of samples, defaults to ``50``.
            endpoint (bool, optional): If ``True``, ``stop`` is the last
              sample. Otherwise it is not included and the samples are spaced
              over the half-open interval ``[start, stop)``. Default: ``True``.
            dtype (Dtype, optional): Specifies the data type of the output,
              default to ``float32``.

        Returns:
            array: The range of values.
      )pbdoc");
  m.def(
      "kron",
      &mx::kron,
      nb::arg("a"),
      nb::arg("b"),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def kron(a: array, b: array, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Compute the Kronecker product of two arrays ``a`` and ``b``.

        Args:
          a (array): The first input array.
          b (array): The second input array.
          stream (StreamOrDevice, optional): Optional stream or
            device for execution. Default: ``None``.

        Returns:
          array: The Kronecker product of ``a`` and ``b``.

        Examples:
          >>> a = mx.array([[1, 2], [3, 4]])
          >>> b = mx.array([[0, 5], [6, 7]])
          >>> result = mx.kron(a, b)
          >>> print(result)
          array([[0, 5, 0, 10],
                 [6, 7, 12, 14],
                 [0, 15, 0, 20],
                 [18, 21, 24, 28]], dtype=int32)
      )pbdoc");
  m.def(
      "take",
      [](const mx::array& a,
         const std::variant<nb::int_, mx::array>& indices,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<nb::int_>(&indices); pv) {
          auto idx = nb::cast<int>(*pv);
          return axis ? mx::take(a, idx, axis.value(), s) : mx::take(a, idx, s);
        } else {
          auto indices_ = std::get<mx::array>(indices);
          return axis ? mx::take(a, indices_, axis.value(), s)
                      : mx::take(a, indices_, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take(a: array, /, indices: int | array, axis: int | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Take elements along an axis.

        The elements are taken from ``indices`` along the specified axis.
        If the axis is not specified the array is treated as a flattened
        1-D array prior to performing the take.

        As an example, if the ``axis=1`` this is equivalent to ``a[:, indices, ...]``.

        Args:
            a (array): Input array.
            indices (int or array): Integer index or input array with integral type.
            axis (int, optional): Axis along which to perform the take. If unspecified
              the array is treated as a flattened 1-D vector.

        Returns:
            array: The indexed values of ``a``.
      )pbdoc");
  m.def(
      "take_along_axis",
      [](const mx::array& a,
         const mx::array& indices,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::take_along_axis(a, indices, axis.value(), s);
        } else {
          return mx::take_along_axis(mx::reshape(a, {-1}, s), indices, 0, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take_along_axis(a: array, /, indices: array, axis: int | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Take values along an axis at the specified indices.

        Args:
            a (array): Input array.
            indices (array): Indices array. These should be broadcastable with
              the input array excluding the `axis` dimension.
            axis (int or None): Axis in the input to take the values from. If
              ``axis == None`` the array is flattened to 1D prior to the indexing
              operation.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "put_along_axis",
      [](const mx::array& a,
         const mx::array& indices,
         const mx::array& values,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::put_along_axis(a, indices, values, axis.value(), s);
        } else {
          return mx::reshape(
              mx::put_along_axis(
                  mx::reshape(a, {-1}, s), indices, values, 0, s),
              a.shape(),
              s);
        }
      },
      nb::arg(),
      "indices"_a,
      "values"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def put_along_axis(a: array, /, indices: array, values: array, axis: int | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Put values along an axis at the specified indices.

        Args:
            a (array): Destination array.
            indices (array): Indices array. These should be broadcastable with
              the input array excluding the `axis` dimension.
            values (array): Values array. These should be broadcastable with
              the indices.

            axis (int or None): Axis in the destination to put the values to. If
              ``axis == None`` the destination is flattened prior to the put
              operation.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "full",
      [](const nb::object& shape,
         const ScalarOrArray& vals,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::full(to_shape(shape), to_array(vals, dtype), s);
      },
      "shape"_a,
      "vals"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def full(shape: int | Sequence[int], vals: scalar | array, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Construct an array with the given value.

        Constructs an array of size ``shape`` filled with ``vals``. If ``vals``
        is an :obj:`array` it must be broadcastable to the given ``shape``.

        Args:
            shape (int or list(int)): The shape of the output array.
            vals (float or int or array): Values to fill the array with.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type is inferred from ``vals``.

        Returns:
            array: The output array with the specified shape and values.
      )pbdoc");
  m.def(
      "full_like",
      [](const mx::array& a,
         const ScalarOrArray& vals,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(a.dtype());
        return mx::full_like(a, to_array(vals, t), t, s);
      },
      nb::arg(),
      "vals"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def full_like(a: array, vals: scalar | array, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An array filled with ``vals`` with the same shape as the input.

        Args:
            a (array): The input to take the shape from.
            vals (float or int or array): Values to fill the array with.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the type of the input is used.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "zeros",
      [](const nb::object& shape,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(mx::float32);
        return mx::zeros(to_shape(shape), t, s);
      },
      "shape"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zeros(shape: int | Sequence[int], dtype: Dtype | None = float32, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Construct an array of zeros.

        Args:
            shape (int or list(int)): The shape of the output array.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type defaults to ``float32``.

        Returns:
            array: The array of zeros with the specified shape.
      )pbdoc");
  m.def(
      "asarray",
      [](const nb::object& a,
         std::optional<mx::Dtype> dtype,
         std::optional<bool> copy) { return create_array(a, dtype, copy); },
      nb::arg(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "copy"_a = nb::none(),
      nb::sig(
          "def asarray(a: scalar | array | Sequence | DLPackCompatible, dtype: "
          "Dtype | None = None, *, copy: bool | None = None) -> array"),
      R"pbdoc(
        Convert the input to an array.

        Args:
            a: Input data.
            dtype (Dtype, optional): The desired data-type for the array.
            copy (bool, optional): Whether to copy the input. If ``True``,
              always copy. If ``False``, never copy. If ``None``, share memory
              when possible and copy otherwise. Zero-copy DLPack imports
              preserve the DLPack strides.

        Returns:
            array: An array interpretation of the input.

        Raises:
            ValueError: If ``copy`` is ``False`` and a copy is required.
      )pbdoc");
  m.def(
      "from_dlpack",
      [](nb::ndarray<nb::ro> x, std::optional<bool> copy) {
        return nd_array_to_mlx(x, std::nullopt, std::nullopt, copy);
      },
      nb::arg(),
      nb::kw_only(),
      "copy"_a = nb::none(),
      nb::sig(
          "def from_dlpack(x: DLPackCompatible, /, *, copy: bool | None = None) -> array"),
      R"pbdoc(
        Create an array from an object that supports DLPack.

        Args:
            x: Input object implementing ``__dlpack__`` and
              ``__dlpack_device__``.
            copy (bool, optional): Whether to copy the input. If ``True``,
              always copy. If ``False``, never copy. If ``None``, share memory
              when possible and copy otherwise. Zero-copy imports preserve the
              DLPack strides.

        Returns:
            array: An array containing the input data.
      )pbdoc");
  m.def(
      "zeros_like",
      [](const mx::array& a,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::zeros_like(a, dtype.value_or(a.dtype()), s);
      },
      nb::arg(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zeros_like(a: array, /, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An array of zeros like the input.

        Args:
            a (array): The input to take the shape from.
            dtype (Dtype, optional): Output data type. If ``None``, the output
              type defaults to the input array's data type.

        Returns:
            array: The output array filled with zeros.
      )pbdoc");
  m.def(
      "ones",
      [](const nb::object& shape,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(mx::float32);
        return mx::ones(to_shape(shape), t, s);
      },
      "shape"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ones(shape: int | Sequence[int], dtype: Dtype | None = float32, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Construct an array of ones.

        Args:
            shape (int or list(int)): The shape of the output array.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type defaults to ``float32``.

        Returns:
            array: The array of ones with the specified shape.
      )pbdoc");
  m.def(
      "ones_like",
      [](const mx::array& a,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::ones_like(a, dtype.value_or(a.dtype()), s);
      },
      nb::arg(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ones_like(a: array, /, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An array of ones like the input.

        Args:
            a (array): The input to take the shape from.
            dtype (Dtype, optional): Output data type. If ``None``, the output
              type defaults to the input array's data type.

        Returns:
            array: The output array filled with ones.
      )pbdoc");
  m.def(
      "eye",
      [](int n,
         std::optional<int> m,
         int k,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::eye(n, m.value_or(n), k, dtype.value_or(mx::float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def eye(n: int, m: int | None = None, k: int = 0, dtype: Dtype | None = float32, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Create an identity matrix or a general diagonal matrix.

        Args:
            n (int): The number of rows in the output.
            m (int, optional): The number of columns in the output. Defaults to n.
            k (int, optional): Index of the diagonal. Defaults to 0 (main diagonal).
            dtype (Dtype, optional): Data type of the output array. Defaults to float32.
            stream (Stream, optional): Stream or device. Defaults to None.

        Returns:
            array: An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.
      )pbdoc");
  m.def(
      "identity",
      [](int n, std::optional<mx::Dtype> dtype, mx::StreamOrDevice s) {
        return mx::identity(n, dtype.value_or(mx::float32), s);
      },
      "n"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def identity(n: int, dtype: Dtype | None = float32, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Create a square identity matrix.

        Args:
            n (int): The number of rows and columns in the output.
            dtype (Dtype, optional): Data type of the output array. Defaults to float32.
            stream (Stream, optional): Stream or device. Defaults to None.

        Returns:
            array: An identity matrix of size n x n.
      )pbdoc");
  m.def(
      "tri",
      [](int n,
         std::optional<int> m,
         int k,
         std::optional<mx::Dtype> type,
         mx::StreamOrDevice s) {
        return mx::tri(n, m.value_or(n), k, type.value_or(mx::float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tri(n: int, m: int, k: int, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An array with ones at and below the given diagonal and zeros elsewhere.

        Args:
          n (int): The number of rows in the output.
          m (int, optional): The number of cols in the output. Defaults to ``None``.
          k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
          dtype (Dtype, optional): Data type of the output array. Defaults to ``float32``.
          stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
          array: Array with its lower triangle filled with ones and zeros elsewhere
      )pbdoc");
  m.def(
      "tril",
      &mx::tril,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tril(x: array, k: int, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Zeros the array above the given diagonal.

        Args:
          x (array): input array.
          k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
          stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
          array: Array zeroed above the given diagonal
      )pbdoc");
  m.def(
      "triu",
      &mx::triu,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def triu(x: array, k: int, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Zeros the array below the given diagonal.

        Args:
          x (array): input array.
          k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
          stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
          array: Array zeroed below the given diagonal
    )pbdoc");
  m.def(
      "allclose",
      &mx::allclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def allclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Approximate comparison of two arrays.

        Infinite values are considered equal if they have the same sign, NaN values are not equal unless ``equal_nan`` is ``True``.

        The arrays are considered equal if:

        .. code-block::

         all(abs(a - b) <= (atol + rtol * abs(b)))

        Note unlike :func:`array_equal`, this function supports numpy-style
        broadcasting.

        Args:
            a (array): Input array.
            b (array): Input array.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
            equal_nan (bool): If ``True``, NaNs are considered equal.
              Defaults to ``False``.

        Returns:
            array: The boolean output scalar indicating if the arrays are close.
      )pbdoc");
  m.def(
      "isclose",
      &mx::isclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def isclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns a boolean array where two arrays are element-wise equal within a tolerance.

        Infinite values are considered equal if they have the same sign, NaN values are
        not equal unless ``equal_nan`` is ``True``.

        Two values are considered equal if:

        .. code-block::

         abs(a - b) <= (atol + rtol * abs(b))

        Note unlike :func:`array_equal`, this function supports numpy-style
        broadcasting.

        Args:
            a (array): Input array.
            b (array): Input array.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
            equal_nan (bool): If ``True``, NaNs are considered equal.
              Defaults to ``False``.

        Returns:
            array: The boolean output scalar indicating if the arrays are close.
      )pbdoc");
  m.def(
      "all",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An `and` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "any",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def any(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An `or` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "minimum",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::minimum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def minimum(a: scalar | array, b: scalar | array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise minimum.

        Take the element-wise min of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The min of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "maximum",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::maximum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def maximum(a: scalar | array, b: scalar | array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise maximum.

        Take the element-wise max of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The max of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "floor",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::floor(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def floor(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise floor.

        Args:
            a (array): Input array.

        Returns:
            array: The floor of ``a``.
      )pbdoc");
  m.def(
      "ceil",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::ceil(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ceil(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise ceil.

        Args:
            a (array): Input array.

        Returns:
            array: The ceil of ``a``.
      )pbdoc");
  m.def(
      "isnan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isnan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isnan(a: array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are NaN.

        Args:
            a (array): Input array.

        Returns:
            array: The boolean array indicating which elements are NaN.
      )pbdoc");
  m.def(
      "isinf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isinf(a: array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are +/- inifnity.

        Args:
            a (array): Input array.

        Returns:
            array: The boolean array indicating which elements are +/- infinity.
      )pbdoc");
  m.def(
      "isfinite",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isfinite(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isfinite(a: array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are finite.

        An element is finite if it is not infinite or NaN.

        Args:
            a (array): Input array.

        Returns:
            array: The boolean array indicating which elements are finite.
      )pbdoc");
  m.def(
      "isposinf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isposinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isposinf(a: array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are positive infinity.

        Args:
            a (array): Input array.
            stream (StreamOrDevice): Optional stream or device.

        Returns:
            array: The boolean array indicating which elements are positive infinity.
      )pbdoc");
  m.def(
      "isneginf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isneginf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isneginf(a: array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are negative infinity.

        Args:
            a (array): Input array.
            stream (StreamOrDevice): Optional stream or device.

        Returns:
            array: The boolean array indicating which elements are negative infinity.
      )pbdoc");
  m.def(
      "moveaxis",
      &mx::moveaxis,
      nb::arg(),
      "source"_a,
      "destination"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def moveaxis(a: array, /, source: int, destination: int, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Move an axis to a new position.

        Args:
            a (array): Input array.
            source (int): Specifies the source axis.
            destination (int): Specifies the destination axis.

        Returns:
            array: The array with the axis moved.
      )pbdoc");
  m.def(
      "swapaxes",
      &mx::swapaxes,
      nb::arg(),
      "axis1"_a,
      "axis2"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def swapaxes(a: array, /, axis1 : int, axis2: int, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Swap two axes of an array.

        Args:
            a (array): Input array.
            axis1 (int): Specifies the first axis.
            axis2 (int): Specifies the second axis.

        Returns:
            array: The array with swapped axes.
      )pbdoc");
  m.def(
      "transpose",
      [](const mx::array& a,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value()) {
          return mx::transpose(a, *axes, s);
        } else {
          return mx::transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def transpose(a: array, /, axes: Sequence[int] | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Transpose the dimensions of the array.

        Args:
            a (array): Input array.
            axes (list(int), optional): Specifies the source axis for each axis
              in the new array. The default is to reverse the axes.

        Returns:
            array: The transposed array.
      )pbdoc");
  m.def(
      "permute_dims",
      [](const mx::array& a,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value()) {
          return mx::transpose(a, *axes, s);
        } else {
          return mx::transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def permute_dims(a: array, /, axes: Sequence[int] | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        See :func:`transpose`.
      )pbdoc");
  m.def(
      "sum",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      "array"_a,
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sum(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Sum reduce the array over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "count_nonzero",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (std::holds_alternative<std::monostate>(axis)) {
          return mx::count_nonzero(a, keepdims, s);
        } else if (auto pv = std::get_if<int>(&axis); pv) {
          return mx::count_nonzero(a, *pv, keepdims, s);
        } else {
          return mx::count_nonzero(
              a, std::get<std::vector<int>>(axis), keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "keepdims"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def count_nonzero(a: array, /, *, axis: None | int | Sequence[int] = None, keepdims: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Count the number of non-zero elements along the given axis.

        Args:
            a (array): Input array.
            axis (int or tuple(int), optional): Axis or axes to count over.
              Defaults to ``None`` in which case the whole array is counted.
            keepdims (bool, optional): Keep the reduced axes as size one.
              Default: ``False``.

        Returns:
            array: The counts as an ``int32`` array.
      )pbdoc");
  m.def(
      "prod",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def prod(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        An product reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "min",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def min(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        A `min` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "max",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def max(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        A `max` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "logcumsumexp",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::logcumsumexp(a, *axis, reverse, inclusive, s);
        } else {
          return mx::logcumsumexp(
              mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def logcumsumexp(a: array, /, axis: int | None = None, *, reverse: bool = False, inclusive: bool = True, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the cumulative logsumexp of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative logsumexp
            over. If unspecified the cumulative logsumexp of the flattened array is
            returned.
          reverse (bool): Perform the cumulative logsumexp in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "logsumexp",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logsumexp(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        A `log-sum-exp` reduction over the given axes.

        The log-sum-exp reduction is a numerically stable version of:

        .. code-block::

          log(sum(exp(a), axis))

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "mean",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def mean(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Compute the mean(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array of means.
      )pbdoc");
  m.def(
      "median",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::median(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def median(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Compute the median(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array of medians.
      )pbdoc");
  m.def(
      "var",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         mx::StreamOrDevice s) {
        return mx::var(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def var(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, ddof: int = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Compute the variance(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.
            ddof (int, optional): The divisor to compute the variance
              is ``N - ddof``, defaults to 0.

        Returns:
            array: The output array of variances.
      )pbdoc");
  m.def(
      "std",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         mx::StreamOrDevice s) {
        return mx::std(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def std(a: array, /, axis: None | int | Sequence[int] = None, keepdims: bool = False, ddof: int = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Compute the standard deviation(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.
            ddof (int, optional): The divisor to compute the variance
              is ``N - ddof``, defaults to 0.

        Returns:
            array: The output array of standard deviations.
      )pbdoc");
  m.def(
      "split",
      [](const mx::array& a,
         const std::variant<int, mx::Shape>& indices_or_sections,
         int axis,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
          return mx::split(a, *pv, axis, s);
        } else {
          return mx::split(
              a, std::get<mx::Shape>(indices_or_sections), axis, s);
        }
      },
      nb::arg(),
      "indices_or_sections"_a,
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def split(a: array, /, indices_or_sections: int | Sequence[int], axis: int = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Split an array along a given axis.

        Args:
            a (array): Input array.
            indices_or_sections (int or list(int)): If ``indices_or_sections``
              is an integer the array is split into that many sections of equal
              size. An error is raised if this is not possible. If
              ``indices_or_sections`` is a list, then the indices are the split
              points, and the array is divided into
              ``len(indices_or_sections) + 1`` sub-arrays.
            axis (int, optional): Axis to split along, defaults to `0`.

        Returns:
            list(array): A list of split arrays.

        Example:

          >>> a = mx.array([1, 2, 3, 4], dtype=mx.int32)
          >>> mx.split(a, 2)
          [array([1, 2], dtype=int32), array([3, 4], dtype=int32)]
          >>> mx.split(a, [1, 3])
          [array([1], dtype=int32), array([2, 3], dtype=int32), array([4], dtype=int32)]

      )pbdoc");
  m.def(
      "argmin",
      [](const mx::array& a,
         std::optional<int> axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argmin(a, *axis, keepdims, s);
        } else {
          return mx::argmin(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argmin(a: array, /, axis: None | int = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Indices of the minimum values along the axis.

        Args:
            a (array): Input array.
            axis (int, optional): Optional axis to reduce over. If unspecified
              this defaults to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The ``uint32`` array with the indices of the minimum values.
      )pbdoc");
  m.def(
      "argmax",
      [](const mx::array& a,
         std::optional<int> axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argmax(a, *axis, keepdims, s);
        } else {
          return mx::argmax(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argmax(a: array, /, axis: None | int = None, keepdims: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Indices of the maximum values along the axis.

        Args:
            a (array): Input array.
            axis (int, optional): Optional axis to reduce over. If unspecified
              this defaults to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The ``uint32`` array with the indices of the maximum values.
      )pbdoc");
  m.def(
      "sort",
      [](const mx::array& a, std::optional<int> axis, mx::StreamOrDevice s) {
        if (axis) {
          return mx::sort(a, *axis, s);
        } else {
          return mx::sort(a, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sort(a: array, /, axis: None | int = -1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns a sorted copy of the array.

        The sort is stable, meaning equal elements preserve their relative
        order. ``NaN`` values are placed at the end.

        Args:
            a (array): Input array.
            axis (int or None, optional): Optional axis to sort over.
              If ``None``, this sorts over the flattened array.
              If unspecified, it defaults to -1 (sorting over the last axis).

        Returns:
            array: The sorted array.
      )pbdoc");
  m.def(
      "argsort",
      [](const mx::array& a, std::optional<int> axis, mx::StreamOrDevice s) {
        if (axis) {
          return mx::argsort(a, *axis, s);
        } else {
          return mx::argsort(a, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argsort(a: array, /, axis: None | int = -1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns the indices that sort the array.

        The sort is stable, meaning equal elements preserve their relative
        order. ``NaN`` values are placed at the end.

        Args:
            a (array): Input array.
            axis (int or None, optional): Optional axis to sort over.
              If ``None``, this sorts over the flattened array.
              If unspecified, it defaults to -1 (sorting over the last axis).

        Returns:
            array: The ``uint32`` array containing indices that sort the input.
      )pbdoc");
  m.def(
      "partition",
      [](const mx::array& a,
         int kth,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::partition(a, kth, *axis, s);
        } else {
          return mx::partition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def partition(a: array, /, kth: int, axis: None | int = -1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns a partitioned copy of the array such that the smaller ``kth``
        elements are first.

        The ordering of the elements in partitions is undefined.

        Args:
            a (array): Input array.
            kth (int): Element at the ``kth`` index will be in its sorted
              position in the output. All elements before the kth index will
              be less or equal to the ``kth`` element and all elements after
              will be greater or equal to the ``kth`` element in the output.
            axis (int or None, optional): Optional axis to partition over.
              If ``None``, this partitions over the flattened array.
              If unspecified, it defaults to ``-1``.

        Returns:
            array: The partitioned array.
      )pbdoc");
  m.def(
      "argpartition",
      [](const mx::array& a,
         int kth,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argpartition(a, kth, *axis, s);
        } else {
          return mx::argpartition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argpartition(a: array, /, kth: int, axis: None | int = -1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns the indices that partition the array.

        The ordering of the elements within a partition in given by the indices
        is undefined.

        Args:
            a (array): Input array.
            kth (int): Element index at the ``kth`` position in the output will
              give the sorted position. All indices before the ``kth`` position
              will be of elements less or equal to the element at the ``kth``
              index and all indices after will be of elements greater or equal
              to the element at the ``kth`` index.
            axis (int or None, optional): Optional axis to partition over.
              If ``None``, this partitions over the flattened array.
              If unspecified, it defaults to ``-1``.

        Returns:
            array: The ``uint32`` array containing indices that partition the input.
      )pbdoc");
  m.def(
      "searchsorted",
      &mx::searchsorted,
      nb::arg(),
      nb::arg(),
      "side"_a = "left",
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def searchsorted(sorted_sequence: array, values: array, /, side: str = 'left', *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Find the indices that keep ``sorted_sequence`` sorted when inserting ``values``.

        Args:
            sorted_sequence (array): A 1-D array sorted in ascending order.
            values (array): The values to insert. May have any shape.
            side (str, optional): Either ``'left'`` or ``'right'``. With
              ``'left'`` the first suitable index is returned, so the result is
              the number of elements strictly less than the value. With
              ``'right'`` the last is returned, so the result is the number of
              elements less than or equal to it. The two differ only where a
              value is already present. Default: ``'left'``.

        Returns:
            array: A ``uint32`` array with the same shape as ``values``, holding
            indices in ``[0, sorted_sequence.size]``.

        Example:
            >>> a = mx.array([1, 2, 2, 4])
            >>> mx.searchsorted(a, mx.array([0, 2, 3, 5]))
            array([0, 1, 3, 4], dtype=uint32)
            >>> mx.searchsorted(a, mx.array([0, 2, 3, 5]), side="right")
            array([0, 3, 3, 4], dtype=uint32)
      )pbdoc");
  m.def(
      "topk",
      [](const mx::array& a,
         int k,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::topk(a, k, *axis, s);
        } else {
          return mx::topk(a, k, s);
        }
      },
      nb::arg(),
      "k"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def topk(a: array, /, k: int, axis: None | int = -1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns the ``k`` largest elements from the input along a given axis.

        The elements will not necessarily be in sorted order.

        Args:
            a (array): Input array.
            k (int): ``k`` top elements to be returned
            axis (int or None, optional): Optional axis to select over.
              If ``None``, this selects the top ``k`` elements over the
              flattened array. If unspecified, it defaults to ``-1``.

        Returns:
            array: The top ``k`` elements from the input.
      )pbdoc");
  m.def(
      "broadcast_to",
      [](const ScalarOrArray& a, const mx::Shape& shape, mx::StreamOrDevice s) {
        return mx::broadcast_to(to_array(a), shape, s);
      },
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def broadcast_to(a: scalar | array, /, shape: Sequence[int], *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Broadcast an array to the given shape.

        The broadcasting semantics are the same as Numpy.

        Args:
            a (array): Input array.
            shape (list(int)): The shape to broadcast to.

        Returns:
            array: The output array with the new shape.
      )pbdoc");
  m.def(
      "broadcast_arrays",
      [](const nb::args& args, mx::StreamOrDevice s) {
        return broadcast_arrays(nb::cast<std::vector<mx::array>>(args), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def broadcast_arrays(*arrays: array, stream: StreamOrDevice = None) -> tuple[array, ...]"),
      R"pbdoc(
        Broadcast arrays against one another.

        The broadcasting semantics are the same as Numpy.

        Args:
            *arrays (array): The input arrays.

        Returns:
            tuple(array): The output arrays with the broadcasted shape.
      )pbdoc");
  m.def(
      "softmax",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool precise,
         mx::StreamOrDevice s) {
        return mx::softmax(a, get_reduce_axes(axis, a.ndim()), precise, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "precise"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def softmax(a: array, /, axis: None | int | Sequence[int] = None, *, precise: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Perform the softmax along the given axis.

        This operation is a numerically stable version of:

        .. code-block::

          exp(a) / sum(exp(a), axis, keepdims=True)

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or axes to compute
             the softmax over. If unspecified this performs the softmax over
             the full array.
            precise (bool, optional): Accumulate in ``float32`` for inputs of
              lower precision. Otherwise the accumulation type matches the
              input, which can lose precision over long reduction axes.
              Default: ``False``.

        Returns:
            array: The output of the softmax.
      )pbdoc");
  m.def(
      "concatenate",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::concatenate(arrays, *axis, s);
        } else {
          return mx::concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concatenate(arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Concatenate the arrays along the given axis.

        Args:
            arrays (list(array)): Input :obj:`list` or :obj:`tuple` of arrays.
            axis (int, optional): Optional axis to concatenate along. If
              unspecified defaults to ``0``.

        Returns:
            array: The concatenated array.
      )pbdoc");
  m.def(
      "concat",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::concatenate(arrays, *axis, s);
        } else {
          return mx::concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concat(arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        See :func:`concatenate`.
      )pbdoc");
  m.def(
      "stack",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::stack(arrays, axis.value(), s);
        } else {
          return mx::stack(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stack(arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Stacks the arrays along a new axis.

        Args:
            arrays (list(array)): A list of arrays to stack.
            axis (int, optional): The axis in the result array along which the
              input arrays are stacked. Defaults to ``0``.
            stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
            array: The resulting stacked array.
      )pbdoc");
  m.def(
      "meshgrid",
      [](nb::args arrays_,
         bool sparse,
         std::string indexing,
         mx::StreamOrDevice s) {
        std::vector<mx::array> arrays =
            nb::cast<std::vector<mx::array>>(arrays_);
        return nb::tuple(nb::cast(mx::meshgrid(arrays, sparse, indexing, s)));
      },
      "arrays"_a,
      "sparse"_a = false,
      "indexing"_a = "xy",
      "stream"_a = nb::none(),
      nb::sig(
          "def meshgrid(*arrays: array, sparse: bool | None = False, indexing: str | None = 'xy', stream: StreamOrDevice = None) -> tuple[array, ...]"),
      R"pbdoc(
        Generate multidimensional coordinate grids from 1-D coordinate arrays

        Args:
            *arrays (array): Input arrays.
            sparse (bool, optional): If ``True``, a sparse grid is returned in which each output
              array has a single non-zero element. If ``False``, a dense grid is returned.
              Defaults to ``False``.
            indexing (str, optional): Cartesian ('xy') or matrix ('ij') indexing of the output arrays.
              Defaults to ``'xy'``.

        Returns:
            tuple(array): The output arrays.
      )pbdoc");
  m.def(
      "repeat",
      [](const mx::array& array,
         int repeats,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::repeat(array, repeats, axis.value(), s);
        } else {
          return mx::repeat(array, repeats, s);
        }
      },
      nb::arg(),
      "repeats"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def repeat(array: array, repeats: int, axis: int | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Repeat an array along a specified axis.

        Args:
            array (array): Input array.
            repeats (int): The number of repetitions for each element.
            axis (int, optional): The axis in which to repeat the array along. If
              unspecified it uses the flattened array of the input and repeats
              along axis 0.
            stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
            array: The resulting repeated array.
      )pbdoc");
  m.def(
      "clip",
      [](const mx::array& a,
         const std::optional<ScalarOrArray>& min,
         const std::optional<ScalarOrArray>& max,
         mx::StreamOrDevice s) {
        std::optional<mx::array> min_ = std::nullopt;
        std::optional<mx::array> max_ = std::nullopt;
        if (min) {
          min_ = to_arrays(a, min.value()).second;
        }
        if (max) {
          max_ = to_arrays(a, max.value()).second;
        }
        return mx::clip(a, min_, max_, s);
      },
      nb::arg(),
      "a_min"_a.none(),
      "a_max"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def clip(a: array, /, a_min: scalar | array | None, a_max: scalar | array | None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Clip the values of the array between the given minimum and maximum.

        If either ``a_min`` or ``a_max`` are ``None``, then corresponding edge
        is ignored. At least one of ``a_min`` and ``a_max`` cannot be ``None``.
        The input ``a`` and the limits must broadcast with one another.

        Args:
            a (array): Input array.
            a_min (scalar or array or None): Minimum value to clip to.
            a_max (scalar or array or None): Maximum value to clip to.

        Returns:
            array: The clipped array.
      )pbdoc");
  m.def(
      "pad",
      [](const mx::array& a,
         const std::variant<
             int,
             std::tuple<int>,
             std::pair<int, int>,
             std::vector<std::pair<int, int>>>& pad_width,
         const std::string& mode,
         const ScalarOrArray& constant_value,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&pad_width); pv) {
          return mx::pad(a, *pv, to_array(constant_value, a.dtype()), mode, s);
        } else if (auto pv = std::get_if<std::tuple<int>>(&pad_width); pv) {
          return mx::pad(
              a,
              std::get<0>(*pv),
              to_array(constant_value, a.dtype()),
              mode,
              s);
        } else if (auto pv = std::get_if<std::pair<int, int>>(&pad_width); pv) {
          return mx::pad(a, *pv, to_array(constant_value, a.dtype()), mode, s);
        } else {
          auto v = std::get<std::vector<std::pair<int, int>>>(pad_width);
          if (v.size() == 1) {
            return mx::pad(
                a, v[0], to_array(constant_value, a.dtype()), mode, s);
          } else {
            return mx::pad(a, v, to_array(constant_value, a.dtype()), mode, s);
          }
        }
      },
      nb::arg(),
      "pad_width"_a,
      "mode"_a = "constant",
      "constant_values"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def pad(a: array, pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]], mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'constant', constant_values: scalar | array = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Pad an array with a constant value

        Args:
            a (array): Input array.
            pad_width (int, tuple(int), tuple(int, int) or list(tuple(int, int))): Number of padded
              values to add to the edges of each axis:``((before_1, after_1),
              (before_2, after_2), ..., (before_N, after_N))``. If a single pair
              of integers is passed then ``(before_i, after_i)`` are all the same.
              If a single integer or tuple with a single integer is passed then
              all axes are extended by the same number on each side.
            mode: Padding mode. One of the following strings:
              "constant" (default): Pads with a constant value.
              "edge": Pads with the edge values of array.
              "reflect": Pads with the reflection of the array, without repeating the edge values.
              "symmetric": Pads with the reflection of the array, repeating the edge values.
            constant_values (array or scalar, optional): Optional constant value
              to pad the edges of the array with.

        Returns:
            array: The padded array.
      )pbdoc");
  m.def(
      "as_strided",
      [](const mx::array& a,
         std::optional<mx::Shape> shape,
         std::optional<mx::Strides> strides,
         size_t offset,
         mx::StreamOrDevice s) {
        auto a_shape = (shape) ? *shape : a.shape();
        mx::Strides a_strides;
        if (strides) {
          a_strides = *strides;
        } else {
          a_strides = mx::Strides(a_shape.size(), 1);
          for (int i = a_shape.size() - 1; i > 0; i--) {
            a_strides[i - 1] = a_shape[i] * a_strides[i];
          }
        }
        return mx::as_strided(a, a_shape, a_strides, offset, s);
      },
      nb::arg(),
      "shape"_a = nb::none(),
      "strides"_a = nb::none(),
      "offset"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def as_strided(a: array, /, shape: Sequence[int] | None = None, strides: Sequence[int] | None = None, offset: int = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Create a view into the array with the given shape and strides.

        The resulting array will always be as if the provided array was row
        contiguous regardless of the provided arrays storage order and current
        strides.

        .. note::
           Note that this function should be used with caution as it changes
           the shape and strides of the array directly. This can lead to the
           resulting array pointing to invalid memory locations which can
           result into crashes.

        Args:
          a (array): Input array
          shape (list(int), optional): The shape of the resulting array. If
            None it defaults to ``a.shape()``.
          strides (list(int), optional): The strides of the resulting array. If
            None it defaults to the reverse exclusive cumulative product of
            ``a.shape()``.
          offset (int): Skip that many elements from the beginning of the input
            array.

        Returns:
          array: The output array which is the strided view of the input.
      )pbdoc");
  m.def(
      "astype",
      [](const mx::array& a, mx::Dtype dtype, mx::StreamOrDevice s) {
        return mx::astype(a, dtype, s);
      },
      nb::arg(),
      "dtype"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def astype(a: array, dtype: Dtype, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Cast the array to a specified type.

        Args:
          a (array): Input array.
          dtype (Dtype): Type to which the array is cast.

        Returns:
          array: The array with type ``dtype``.
      )pbdoc");
  m.def(
      "cumsum",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cumsum(a, *axis, reverse, inclusive, dtype, s);
        }
        return mx::cumsum(a, reverse, inclusive, dtype, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "dtype"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cumsum(a: array, /, axis: int | None = None, *, reverse: bool = False, inclusive: bool = True, dtype: Dtype | None = None, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the cumulative sum of the elements along the given axis.

        Args:
          a (array): Input array.
          axis (int, optional): Optional axis to compute the cumulative sum
            over. If unspecified the cumulative sum of the flattened array is
            returned.
          reverse (bool): Perform the cumulative sum in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.
          dtype (Dtype, optional): Cast the input to this type before summing.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "cumprod",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cumprod(a, *axis, reverse, inclusive, dtype, s);
        }
        return mx::cumprod(a, reverse, inclusive, dtype, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "dtype"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cumprod(a: array, /, axis: int | None = None, *, reverse: bool = False, inclusive: bool = True, dtype: Dtype | None = None, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the cumulative product of the elements along the given axis.

        Args:
          a (array): Input array.
          axis (int, optional): Optional axis to compute the cumulative product
            over. If unspecified the cumulative product of the flattened array
            is returned.
          reverse (bool): Perform the cumulative product in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.
          dtype (Dtype, optional): Cast the input to this type before multiplying.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "cummax",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cummax(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cummax(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cummax(a: array, /, axis: int | None = None, *, reverse: bool = False, inclusive: bool = True, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the cumulative maximum of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative maximum
            over. If unspecified the cumulative maximum of the flattened array is
            returned.
          reverse (bool): Perform the cumulative maximum in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "cummin",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cummin(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cummin(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cummin(a: array, /, axis: int | None = None, *, reverse: bool = False, inclusive: bool = True, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the cumulative minimum of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative minimum
            over. If unspecified the cumulative minimum of the flattened array is
            returned.
          reverse (bool): Perform the cumulative minimum in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "diff",
      &mx::diff,
      nb::arg(),
      "n"_a = 1,
      "axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def diff(a: array, /, n: int = 1, axis: int = -1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        The n-th discrete difference along the given axis.

        Args:
            a (array): Input array.
            n (int, optional): The number of times to difference. Default: ``1``.
            axis (int, optional): The axis along which to difference.
              Default: ``-1``.

        Returns:
            array: The n-th differences.
      )pbdoc");
  m.def(
      "conj",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::conjugate(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def conj(a: array, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the elementwise complex conjugate of the input.
        Alias for `mx.conjugate`.

        Args:
          a (array): Input array

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "conjugate",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::conjugate(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conjugate(a: array, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the elementwise complex conjugate of the input.
        Alias for `mx.conj`.

        Args:
          a (array): Input array

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "convolve",
      [](const mx::array& a,
         const mx::array& v,
         const std::string& mode,
         mx::StreamOrDevice s) {
        if (a.ndim() != 1 || v.ndim() != 1) {
          throw std::invalid_argument("[convolve] Inputs must be 1D.");
        }

        if (a.size() == 0 || v.size() == 0) {
          throw std::invalid_argument("[convolve] Inputs cannot be empty.");
        }

        mx::array in = a.size() < v.size() ? v : a;
        mx::array wt = a.size() < v.size() ? a : v;
        wt = mx::slice(wt, {wt.shape(0) - 1}, {-wt.shape(0) - 1}, {-1}, s);

        in = mx::reshape(in, {1, -1, 1}, s);
        wt = mx::reshape(wt, {1, -1, 1}, s);

        int padding = 0;

        if (mode == "full") {
          padding = wt.size() - 1;
        } else if (mode == "valid") {
          padding = 0;
        } else if (mode == "same") {
          // Odd sizes use symmetric padding
          if (wt.size() % 2) {
            padding = wt.size() / 2;
          } else { // Even sizes use asymmetric padding
            int pad_l = wt.size() / 2;
            int pad_r = std::max(0, pad_l - 1);
            in = mx::pad(
                in,
                {{0, 0}, {pad_l, pad_r}, {0, 0}},
                mx::array(0),
                "constant",
                s);
          }

        } else {
          throw std::invalid_argument("[convolve] Invalid mode.");
        }

        mx::array out = mx::conv1d(
            in,
            wt,
            /*stride = */ 1,
            /*padding = */ padding,
            /*dilation = */ 1,
            /*groups = */ 1,
            s);

        return mx::reshape(out, {-1}, s);
      },
      nb::arg(),
      nb::arg(),
      "mode"_a = "full",
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          R"(def convolve(a: array, v: array, /, mode: str = "full", *, stream: StreamOrDevice = None) -> array)"),
      R"pbdoc(
        The discrete convolution of 1D arrays.

        If ``v`` is longer than ``a``, then they are swapped.
        The conv filter is flipped following signal processing convention.

        Args:
            a (array): 1D Input array.
            v (array): 1D Input array.
            mode (str, optional): {'full', 'valid', 'same'}

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv1d",
      &mx::conv1d,
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        1D convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, L, C_in)``.
            weight (array): Weight array of shape ``(C_out, K, C_in)``.
            stride (int, optional): Kernel stride. Default: ``1``.
            padding (int, optional): Input padding. Default: ``0``.
            dilation (int, optional): Kernel dilation. Default: ``1``.
            groups (int, optional): Input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv2d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::pair<int, int> stride_pair{1, 1};
        std::pair<int, int> padding_pair{0, 0};
        std::pair<int, int> dilation_pair{1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_pair = std::pair<int, int>{*pv, *pv};
        } else {
          stride_pair = std::get<std::pair<int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          padding_pair = std::get<std::pair<int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_pair = std::pair<int, int>{*pv, *pv};
        } else {
          dilation_pair = std::get<std::pair<int, int>>(dilation);
        }

        return mx::conv2d(
            input, weight, stride_pair, padding_pair, dilation_pair, groups, s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv2d(input: array, weight: array, /, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        2D convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, KH, KW, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv3d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::tuple<int, int, int>>& stride,
         const std::variant<int, std::tuple<int, int, int>>& padding,
         const std::variant<int, std::tuple<int, int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::tuple<int, int, int> stride_tuple{1, 1, 1};
        std::tuple<int, int, int> padding_tuple{0, 0, 0};
        std::tuple<int, int, int> dilation_tuple{1, 1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          stride_tuple = std::get<std::tuple<int, int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          padding_tuple = std::get<std::tuple<int, int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          dilation_tuple = std::get<std::tuple<int, int, int>>(dilation);
        }

        return mx::conv3d(
            input,
            weight,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv3d(input: array, weight: array, /, stride: int | tuple[int, int, int] = 1, padding: int | tuple[int, int, int] = 0, dilation: int | tuple[int, int, int] = 1, groups: int = 1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        3D convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, D, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, KD, KH, KW, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_transpose1d",
      &mx::conv_transpose1d,
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "output_padding"_a = 0,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, groups: int = 1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        1D transposed convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, L, C_in)``.
            weight (array): Weight array of shape ``(C_out, K, C_in)``.
            stride (int, optional): Kernel stride. Default: ``1``.
            padding (int, optional): Input padding. Default: ``0``.
            dilation (int, optional): Kernel dilation. Default: ``1``.
            output_padding (int, optional): Output padding. Default: ``0``.
            groups (int, optional): Input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_transpose2d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         const std::variant<int, std::pair<int, int>>& output_padding,
         int groups,
         mx::StreamOrDevice s) {
        std::pair<int, int> stride_pair{1, 1};
        std::pair<int, int> padding_pair{0, 0};
        std::pair<int, int> dilation_pair{1, 1};
        std::pair<int, int> output_padding_pair{0, 0};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_pair = std::pair<int, int>{*pv, *pv};
        } else {
          stride_pair = std::get<std::pair<int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          padding_pair = std::get<std::pair<int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_pair = std::pair<int, int>{*pv, *pv};
        } else {
          dilation_pair = std::get<std::pair<int, int>>(dilation);
        }

        if (auto pv = std::get_if<int>(&output_padding); pv) {
          output_padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          output_padding_pair = std::get<std::pair<int, int>>(output_padding);
        }

        return mx::conv_transpose2d(
            input,
            weight,
            stride_pair,
            padding_pair,
            dilation_pair,
            output_padding_pair,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "output_padding"_a = 0,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose2d(input: array, weight: array, /, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, output_padding: int | tuple[int, int] = 0, groups: int = 1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        2D transposed convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, KH, KW, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            output_padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
                output padding. All spatial dimensions get the same output
                padding if only one number is specified. Default: ``0``.
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_transpose3d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::tuple<int, int, int>>& stride,
         const std::variant<int, std::tuple<int, int, int>>& padding,
         const std::variant<int, std::tuple<int, int, int>>& dilation,
         const std::variant<int, std::tuple<int, int, int>>& output_padding,
         int groups,
         mx::StreamOrDevice s) {
        std::tuple<int, int, int> stride_tuple{1, 1, 1};
        std::tuple<int, int, int> padding_tuple{0, 0, 0};
        std::tuple<int, int, int> dilation_tuple{1, 1, 1};
        std::tuple<int, int, int> output_padding_tuple{0, 0, 0};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          stride_tuple = std::get<std::tuple<int, int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          padding_tuple = std::get<std::tuple<int, int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          dilation_tuple = std::get<std::tuple<int, int, int>>(dilation);
        }

        if (auto pv = std::get_if<int>(&output_padding); pv) {
          output_padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          output_padding_tuple =
              std::get<std::tuple<int, int, int>>(output_padding);
        }

        return mx::conv_transpose3d(
            input,
            weight,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
            output_padding_tuple,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "output_padding"_a = 0,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose3d(input: array, weight: array, /, stride: int | tuple[int, int, int] = 1, padding: int | tuple[int, int, int] = 0, dilation: int | tuple[int, int, int] = 1, output_padding: int | tuple[int, int, int] = 0, groups: int = 1, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        3D transposed convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, D, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, KD, KH, KW, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            output_padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
                output padding. All spatial dimensions get the same output
                padding if only one number is specified. Default: ``0``.
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_general",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::vector<int>>& stride,
         const std::variant<
             int,
             std::vector<int>,
             std::pair<std::vector<int>, std::vector<int>>>& padding,
         const std::variant<int, std::vector<int>>& kernel_dilation,
         const std::variant<int, std::vector<int>>& input_dilation,
         int groups,
         bool flip,
         mx::StreamOrDevice s) {
        std::vector<int> stride_vec;
        std::vector<int> padding_lo_vec;
        std::vector<int> padding_hi_vec;
        std::vector<int> kernel_dilation_vec;
        std::vector<int> input_dilation_vec;

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_vec.push_back(*pv);
        } else {
          stride_vec = std::get<std::vector<int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_lo_vec.push_back(*pv);
          padding_hi_vec.push_back(*pv);
        } else if (auto pv = std::get_if<std::vector<int>>(&padding); pv) {
          padding_lo_vec = *pv;
          padding_hi_vec = *pv;
        } else {
          auto [pl, ph] =
              std::get<std::pair<std::vector<int>, std::vector<int>>>(padding);
          padding_lo_vec = pl;
          padding_hi_vec = ph;
        }

        if (auto pv = std::get_if<int>(&kernel_dilation); pv) {
          kernel_dilation_vec.push_back(*pv);
        } else {
          kernel_dilation_vec = std::get<std::vector<int>>(kernel_dilation);
        }

        if (auto pv = std::get_if<int>(&input_dilation); pv) {
          input_dilation_vec.push_back(*pv);
        } else {
          input_dilation_vec = std::get<std::vector<int>>(input_dilation);
        }

        return mx::conv_general(
            /* array input = */ std::move(input),
            /* array weight = */ std::move(weight),
            /* std::vector<int> stride = */ std::move(stride_vec),
            /* std::vector<int> padding_lo = */ std::move(padding_lo_vec),
            /* std::vector<int> padding_hi = */ std::move(padding_hi_vec),
            /* std::vector<int> kernel_dilation = */
            std::move(kernel_dilation_vec),
            /* std::vector<int> input_dilation = */
            std::move(input_dilation_vec),
            /* int groups = */ groups,
            /* bool flip = */ flip,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "kernel_dilation"_a = 1,
      "input_dilation"_a = 1,
      "groups"_a = 1,
      "flip"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_general(input: array, weight: array, /, stride: int | Sequence[int] = 1, padding: int | Sequence[int] | tuple[Sequence[int], Sequence[int]] = 0, kernel_dilation: int | Sequence[int] = 1, input_dilation: int | Sequence[int] = 1, groups: int = 1, flip: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        General convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, ..., C_in)``.
            weight (array): Weight array of shape ``(C_out, ..., C_in)``.
            stride (int or list(int), optional): :obj:`list` with kernel strides.
                All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int, list(int), or tuple(list(int), list(int)), optional):
                :obj:`list` with input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            kernel_dilation (int or list(int), optional): :obj:`list` with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            input_dilation (int or list(int), optional): :obj:`list` with
                input dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): Input feature groups. Default: ``1``.
            flip (bool, optional): Flip the order in which the spatial dimensions of
                the weights are processed. Performs the cross-correlation operator when
                ``flip`` is ``False`` and the convolution operator otherwise.
                Default: ``False``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "save",
      &mlx_save_helper,
      "file"_a,
      "arr"_a,
      nb::sig("def save(file: file | str | pathlib.Path, arr: array) -> None"),
      R"pbdoc(
        Save the array to a binary file in ``.npy`` format.

        Args:
            file (str, pathlib.Path, file): File to which the array is saved
            arr (array): Array to be saved.
      )pbdoc");
  m.def(
      "savez",
      [](nb::object file, nb::args args, const nb::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /* compressed= */ false);
      },
      "file"_a,
      "args"_a,
      "kwargs"_a,
      nb::sig("def savez(file: file | str | pathlib.Path, *args, **kwargs)"),
      R"pbdoc(
        Save several arrays to a binary file in uncompressed ``.npz``
        format.

        .. code-block:: python

            import mlx.core as mx

            x = mx.ones((10, 10))
            mx.savez("my_path.npz", x=x)

            import mlx.nn as nn
            from mlx.utils import tree_flatten

            model = nn.TransformerEncoder(6, 128, 4)
            flat_params = tree_flatten(model.parameters())
            mx.savez("model.npz", **dict(flat_params))

        Args:
            file (file, str, pathlib.Path): Path to file to which the arrays are saved.
            *args (arrays): Arrays to be saved.
            **kwargs (arrays): Arrays to be saved. Each array will be saved
              with the associated keyword as the output file name.
      )pbdoc");
  m.def(
      "savez_compressed",
      [](nb::object file, nb::args args, const nb::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /*compressed=*/true);
      },
      nb::arg(),
      "args"_a,
      "kwargs"_a,
      nb::sig(
          "def savez_compressed(file: file | str | pathlib.Path, *args, **kwargs)"),
      R"pbdoc(
        Save several arrays to a binary file in compressed ``.npz`` format.

        Args:
            file (file, str, pathlib.Path): Path to file to which the arrays are saved.
            *args (arrays): Arrays to be saved.
            **kwargs (arrays): Arrays to be saved. Each array will be saved
              with the associated keyword as the output file name.
      )pbdoc");
  m.def(
      "load",
      &mlx_load_helper,
      nb::arg(),
      "format"_a = nb::none(),
      "return_metadata"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def load(file: file | str | pathlib.Path, /, format: str | None = None, return_metadata: bool = False, *, stream: StreamOrDevice = None) -> array | dict[str, array] | tuple[dict[str, array], dict[str, Any]]"),
      R"pbdoc(
        Load array(s) from a binary file.

        The supported formats are ``.npy``, ``.npz``, ``.safetensors``, and
        ``.gguf``.

        Args:
            file (file, str, pathlib.Path): File in which the array is saved.
            format (str, optional): Format of the file. If ``None``, the
              format is inferred from the file extension. Supported formats:
              ``npy``, ``npz``, and ``safetensors``. Default: ``None``.
            return_metadata (bool, optional): Load the metadata for formats
              which support matadata. The metadata will be returned as an
              additional dictionary. Default: ``False``.
        Returns:
            array, dict, or tuple:
                A single array if loading from a ``.npy`` file or a dict
                mapping names to arrays if loading from a ``.npz`` or
                ``.safetensors`` file. If ``return_metadata`` is ``True`` a
                tuple ``(arrays, metadata)`` will be returned where the second
                element is a dictionary containing the metadata.

        Warning:

          When loading unsupported quantization formats from GGUF, tensors
          will automatically cast to ``mx.float16``
      )pbdoc");
  m.def(
      "save_safetensors",
      &mlx_save_safetensor_helper,
      "file"_a,
      "arrays"_a,
      "metadata"_a = nb::none(),
      nb::sig(
          "def save_safetensors(file: file | str | pathlib.Path, arrays: dict[str, array], metadata: dict[str, str] | None = None)"),
      R"pbdoc(
        Save array(s) to a binary file in ``.safetensors`` format.

        See the `Safetensors documentation
        <https://huggingface.co/docs/safetensors/index>`_ for more
        information on the format.

        Args:
            file (file, str, pathlib.Path): File in which the array is saved.
            arrays (dict(str, array)): The dictionary of names to arrays to
              be saved.
            metadata (dict(str, str), optional): The dictionary of
              metadata to be saved.
      )pbdoc");
  m.def(
      "save_gguf",
      &mlx_save_gguf_helper,
      "file"_a,
      "arrays"_a,
      "metadata"_a = nb::none(),
      nb::sig(
          "def save_gguf(file: file | str | pathlib.Path, arrays: dict[str, array], metadata: dict[str, array | str | list[str]])"),
      R"pbdoc(
        Save array(s) to a binary file in ``.gguf`` format.

        See the `GGUF documentation
        <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>`_ for
        more information on the format.

        Args:
            file (file, str, pathlib.Path): File in which the array is saved.
            arrays (dict(str, array)): The dictionary of names to arrays to
              be saved.
            metadata (dict(str, Union[array, str, list(str)])): The dictionary
               of metadata to be saved. The values can be a scalar or 1D
               obj:`array`, a :obj:`str`, or a :obj:`list` of :obj:`str`.
      )pbdoc");
  m.def(
      "where",
      [](const ScalarOrArray& condition,
         const ScalarOrArray& x_,
         const ScalarOrArray& y_,
         mx::StreamOrDevice s) {
        auto [x, y] = to_arrays(x_, y_);
        return mx::where(to_array(condition), x, y, s);
      },
      "condition"_a,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def where(condition: scalar | array, x: scalar | array, y: scalar | array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Select from ``x`` or ``y`` according to ``condition``.

        The condition and input arrays must be the same shape or
        broadcastable with each another.

        Args:
          condition (array): The condition array.
          x (array): The input selected from where condition is ``True``.
          y (array): The input selected from where condition is ``False``.

        Returns:
            array: The output containing elements selected from
            ``x`` and ``y``.
      )pbdoc");
  m.def(
      "nan_to_num",
      [](const ScalarOrArray& a,
         float nan,
         std::optional<float>& posinf,
         std::optional<float>& neginf,
         mx::StreamOrDevice s) {
        return mx::nan_to_num(to_array(a), nan, posinf, neginf, s);
      },
      nb::arg(),
      "nan"_a = 0.0f,
      "posinf"_a = nb::none(),
      "neginf"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def nan_to_num(a: scalar | array, nan: float = 0, posinf: float | None = None, neginf: float | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Replace NaN and Inf values with finite numbers.

        Args:
            a (array): Input array
            nan (float, optional): Value to replace NaN with. Default: ``0``.
            posinf (float, optional): Value to replace positive infinities
              with. If ``None``, defaults to largest finite value for the
              given data type. Default: ``None``.
            neginf (float, optional): Value to replace negative infinities
              with. If ``None``, defaults to the negative of the largest
              finite value for the given data type. Default: ``None``.

        Returns:
            array: Output array with NaN and Inf replaced.
    )pbdoc");
  m.def(
      "round",
      [](const ScalarOrArray& a, int decimals, mx::StreamOrDevice s) {
        return mx::round(to_array(a), decimals, s);
      },
      nb::arg(),
      "decimals"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def round(a: array, /, decimals: int = 0, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Round to the given number of decimals.

        Basically performs:

        .. code-block:: python

          s = 10**decimals
          x = round(x * s) / s

        Args:
          a (array): Input array
          decimals (int): Number of decimal places to round to. (default: 0)

        Returns:
          array: An array of the same type as ``a`` rounded to the
          given number of decimals.
      )pbdoc");
  m.def(
      "quantized_matmul",
      &mx::quantized_matmul,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a = nb::none(),
      "transpose"_a = true,
      "group_size"_a = nb::none(),
      "bits"_a = nb::none(),
      "mode"_a = "affine",
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantized_matmul(x: array, w: array, /, scales: array, biases: array | None = None, transpose: bool = True, group_size: int | None = None, bits: int | None = None, mode: str = 'affine', *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Perform the matrix multiplication with the quantized matrix ``w``. The
        quantization uses one floating point scale and bias per ``group_size`` of
        elements. Each element in ``w`` takes ``bits`` bits and is packed in an
        unsigned 32 bit integer.

        Args:
          x (array): Input array
          w (array): Quantized matrix packed in unsigned integers
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array, optional): The biases to use per ``group_size``
            elements of ``w``. Default: ``None``.
          transpose (bool, optional): Defines whether to multiply with the
            transposed ``w`` or not, namely whether we are performing
            ``x @ w.T`` or ``x @ w``. Default: ``True``.
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. See supported values and defaults in the
            :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the quantized array. See supported values and defaults in the
            :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
          mode (str, optional): The quantization mode. Default: ``"affine"``.

        Returns:
          array: The result of the multiplication of ``x`` with ``w``.
      )pbdoc");
  m.def(
      "quantize",
      &mx::quantize,
      nb::arg(),
      "group_size"_a = nb::none(),
      "bits"_a = nb::none(),
      "mode"_a = "affine",
      "global_scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantize(w: array, /, group_size: int | None = None, bits: int | None = None, mode: str = 'affine', *, global_scale: array | None = None, stream: StreamOrDevice = None) -> tuple[array, array, array]"),
      R"pbdoc(
        Quantize the array ``w``.

        Note, every ``group_size`` elements in a row of ``w`` are quantized
        together. Hence, the last dimension of ``w`` should be divisible by
        ``group_size``.

        .. warning::

          ``quantize`` only supports inputs with two or more dimensions with
          the last dimension divisible by ``group_size``

        The supported quantization modes are ``"affine"``, ``"mxfp4"``,
        ``"mxfp8"``, and ``"nvfp4"``. They are described in more detail below.

        Args:
          w (array): Array to be quantized
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. See supported values and defaults in the
            :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the quantized array. See supported values and defaults in the
            :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
          mode (str, optional): The quantization mode. Default: ``"affine"``.
          global_scale (array, optional): The per-input float32 scale used for
            ``"nvfp4"`` quantization if provided. Default: ``None``.

        Returns:
          tuple: A tuple with either two or three elements containing:

          * w_q (array): The quantized version of ``w``
          * scales (array): The quantization scales
          * biases (array): The quantization biases (returned for ``mode=="affine"``).

        Notes:
          .. _quantize-modes:

          .. table:: Quantization modes

            ======  ======================   ==========================  =============  =====
            mode    group size               bits                        scale type     bias
            ======  ======================   ==========================  =============  =====
            affine  32, 64\ :sup:`*`, 128    2, 3, 4\ :sup:`*`, 5, 6, 8  same as input  yes
            mxfp4   32\ :sup:`*`             4\ :sup:`*`                 e8m0           no
            mxfp8   32\ :sup:`*`             8\ :sup:`*`                 e8m0           no
            nvfp4   16\ :sup:`*`             4\ :sup:`*`                 e4m3           no
            ======  ======================   ==========================  =============  =====

          :sup:`*` indicates the default value when unspecified.

          The ``"affine"`` mode quantizes groups of :math:`g` consecutive
          elements in a row of ``w``. For each group the quantized
          representation of each element :math:`\hat{w_i}` is computed as follows:

          .. math::

            \begin{aligned}
              \alpha &= \max_i w_i \\
              \beta &= \min_i w_i \\
              s &= \frac{\alpha - \beta}{2^b - 1} \\
              \hat{w_i} &= \textrm{round}\left( \frac{w_i - \beta}{s}\right).
            \end{aligned}

          After the above computation, :math:`\hat{w_i}` fits in :math:`b` bits
          and is packed in an unsigned 32-bit integer from the lower to upper
          bits. For instance, for 4-bit quantization we fit 8 elements in an
          unsigned 32 bit integer where the 1st element occupies the 4 least
          significant bits, the 2nd bits 4-7 etc.

          To dequantize the elements of ``w``, we also save :math:`s` and
          :math:`\beta` which are the returned ``scales`` and
          ``biases`` respectively.

          The ``"mxfp4"``, ``"mxfp8"``, and ``"nvfp4"`` modes similarly
          quantize groups of :math:`g` elements of ``w``. For the ``"mx"``
          modes, the group size must be ``32``.  For ``"nvfp4"`` the group
          size must be 16. The elements are quantized to 4-bit or 8-bit
          precision floating-point values: E2M1 for ``"fp4"`` and E4M3 for
          ``"fp8"``. There is a shared 8-bit scale per group. The ``"mx"``
          modes use an E8M0 scale and the ``"nv"`` mode uses an E4M3 scale.
          Unlike ``affine`` quantization, these modes does not have a bias
          value.

          More details on the ``"mx"`` formats can
          be found in the `specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_.
      )pbdoc");
  m.def(
      "dequantize",
      &mx::dequantize,
      nb::arg(),
      "scales"_a,
      "biases"_a = nb::none(),
      "group_size"_a = nb::none(),
      "bits"_a = nb::none(),
      "mode"_a = "affine",
      "global_scale"_a = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def dequantize(w: array, /, scales: array, biases: array | None = None, group_size: int | None = None, bits: int | None = None, mode: str = 'affine', global_scale: array | None = None, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Dequantize the matrix ``w`` using quantization parameters.

        Args:
          w (array): Matrix to be dequantized
          scales (array): The scales to use per ``group_size`` elements of ``w``.
          biases (array, optional): The biases to use per ``group_size``
             elements of ``w``. Default: ``None``.
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. See supported values and defaults in the
            :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the quantized array. See supported values and defaults in the
            :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
          mode (str, optional): The quantization mode. Default: ``"affine"``.
          global_scale (array, optional): The per-input float32 scale used for
            ``"nvfp4"`` quantization if provided. Default: ``None``.
          dtype (Dtype, optional): The data type of the dequantized output. If
            ``None`` the return type is inferred from the scales and biases
            when possible and otherwise defaults to ``bfloat16``.
            Default: ``None``.

        Returns:
          array: The dequantized version of ``w``

        Notes:
          The currently supported quantization modes are ``"affine"``,
          ``"mxfp4``, ``"mxfp8"``, and ``"nvfp4"``.

          For ``affine`` quantization, given the notation in :func:`quantize`,
          we compute :math:`w_i` from :math:`\hat{w_i}` and corresponding :math:`s`
          and :math:`\beta` as follows

          .. math::

            w_i = s \hat{w_i} + \beta
      )pbdoc");
  m.def(
      "gather_qmm",
      &mx::gather_qmm,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a = nb::none(),
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      "transpose"_a = true,
      "group_size"_a = nb::none(),
      "bits"_a = nb::none(),
      "mode"_a = "affine",
      nb::kw_only(),
      "sorted_indices"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_qmm(x: array, w: array, /, scales: array, biases: array | None = None, lhs_indices: array | None = None, rhs_indices: array | None = None, transpose: bool = True, group_size: int | None = None, bits: int | None = None, mode: str = 'affine', *, sorted_indices: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Perform quantized matrix multiplication with matrix-level gather.

        This operation is the quantized equivalent to :func:`gather_mm`.
        Similar to :func:`gather_mm`, the indices ``lhs_indices`` and
        ``rhs_indices`` contain flat indices along the batch dimensions (i.e.
        all but the last two dimensions) of ``x`` and ``w`` respectively.

        Note that ``scales`` and ``biases`` must have the same batch dimensions
        as ``w`` since they represent the same quantized matrix.

        Args:
            x (array): Input array
            w (array): Quantized matrix packed in unsigned integers
            scales (array): The scales to use per ``group_size`` elements of ``w``
            biases (array, optional): The biases to use per ``group_size``
              elements of ``w``. Default: ``None``.
            lhs_indices (array, optional): Integer indices for ``x``. Default: ``None``.
            rhs_indices (array, optional): Integer indices for ``w``. Default: ``None``.
            transpose (bool, optional): Defines whether to multiply with the
              transposed ``w`` or not, namely whether we are performing
              ``x @ w.T`` or ``x @ w``. Default: ``True``.
            group_size (int, optional): The size of the group in ``w`` that shares a
              scale and bias. See supported values and defaults in the
              :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
            bits (int, optional): The number of bits occupied by each element of
              ``w`` in the quantized array. See supported values and defaults in the
              :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
            mode (str, optional): The quantization mode. Default: ``"affine"``.
            sorted_indices (bool, optional): May allow a faster implementation
              if the passed indices are sorted. Default: ``False``.

        Returns:
            array: The result of the multiplication of ``x`` with ``w``
              after gathering using ``lhs_indices`` and ``rhs_indices``.
      )pbdoc");
  m.def(
      "gather_qqmm",
      &mx::gather_qqmm,
      nb::arg(),
      nb::arg(),
      "scales"_a = nb::none(),
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      "group_size"_a = nb::none(),
      "bits"_a = nb::none(),
      "mode"_a = "nvfp4",
      "global_scale_x"_a = nb::none(),
      "global_scale_w"_a = nb::none(),
      nb::kw_only(),
      "sorted_indices"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_qqmm(x: array, w: array, /, scales: array | None = None, lhs_indices: array | None = None, rhs_indices: array | None = None, group_size: int | None = None, bits: int | None = None, mode: str = 'nvfp4', global_scale_x: array | None = None, global_scale_w: array | None = None, *, sorted_indices: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Fused :func:`qqmm` with matrix-level gather.

        Similar to :func:`gather_mm`, the indices ``lhs_indices`` and
        ``rhs_indices`` contain flat indices along the batch dimensions (i.e.
        all but the last two dimensions) of ``x`` and ``w`` respectively.

        Args:
            x (array): Input array.
            w (array): Weight matrix. If quantized, it is packed in unsigned integers.
            scales (array, optional): The scales to use per ``group_size`` elements of
              ``w`` if ``w`` is quantized. Default: ``None``.
            lhs_indices (array, optional): Integer indices for ``x``. Default: ``None``.
            rhs_indices (array, optional): Integer indices for ``w``. Default: ``None``.
            group_size (int, optional): Number of elements in ``x`` and ``w`` that
              share a scale. See supported values and defaults in the
              :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
            bits (int, optional): Number of bits used to represent each element of
              ``x`` and ``w``. See supported values and defaults in the
              :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
            mode (str, optional): The quantization mode. Default: ``"nvfp4"``.
              Supported modes are ``nvfp4`` and ``mxfp8``. See the
              :ref:`table of quantization modes <quantize-modes>` for details.
            global_scale_x (array, optional): The per-input float32 scale used for x
                with ``"nvfp4"`` quantization. Default: ``None``.
            global_scale_w (array, optional): The per-input float32 scale used for w
                with ``"nvfp4"`` quantization. Default: ``None``.
            sorted_indices (bool, optional): May allow a faster implementation
              if the passed indices are sorted. Default: ``False``.

        Returns:
            array: The result of the multiplication of quantized ``x`` with quantized ``w``.
            needed).
      )pbdoc");
  m.def(
      "segmented_mm",
      &mx::segmented_mm,
      nb::arg(),
      nb::arg(),
      "segments"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def segmented_mm(a: array, b: array, /, segments: array, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Perform a matrix multiplication but segment the inner dimension and
        save the result for each segment separately.

        Args:
          a (array): Input array of shape ``MxK``.
          b (array): Input array of shape ``KxN``.
          segments (array): The offsets into the inner dimension for each segment.

        Returns:
          array: The result per segment of shape ``MxN``.
      )pbdoc");
  m.def(
      "tensordot",
      [](const mx::array& a,
         const mx::array& b,
         const std::variant<int, std::vector<std::vector<int>>>& axes,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&axes); pv) {
          return mx::tensordot(a, b, *pv, s);
        } else {
          auto& x = std::get<std::vector<std::vector<int>>>(axes);
          if (x.size() != 2) {
            throw std::invalid_argument(
                "[tensordot] axes must be a list of two lists.");
          }
          return mx::tensordot(a, b, x[0], x[1], s);
        }
      },
      nb::arg(),
      nb::arg(),
      "axes"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tensordot(a: array, b: array, /, axes: int | list[Sequence[int]] = 2, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Compute the tensor dot product along the specified axes.

        Args:
            a (array): Input array
            b (array): Input array
            axes (int or list(list(int)), optional): The number of dimensions to
              sum over. If an integer is provided, then sum over the last
              ``axes`` dimensions of ``a`` and the first ``axes`` dimensions of
              ``b``. If a list of lists is provided, then sum over the
              corresponding dimensions of ``a`` and ``b``. Default: 2.

        Returns:
            array: The tensor dot product.
      )pbdoc");
  m.def(
      "inner",
      &mx::inner,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def inner(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the last axes.

      Args:
        a (array): Input array
        b (array): Input array

      Returns:
        array: The inner product.
    )pbdoc");
  m.def(
      "vecdot",
      &mx::vecdot,
      nb::arg(),
      nb::arg(),
      "axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def vecdot(a: array, b: array, /, *, axis: int = -1, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Compute the vector dot product of two arrays along an axis.

      Args:
        a (array): Input array
        b (array): Input array
        axis (int, optional): Axis over which to compute the dot product. Default: ``-1``.

      Returns:
        array: The vector dot product.
    )pbdoc");
  m.def(
      "outer",
      &mx::outer,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def outer(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand.

      Args:
        a (array): Input array
        b (array): Input array

      Returns:
        array: The outer product.
    )pbdoc");
  m.def(
      "tile",
      [](const mx::array& a,
         const std::variant<int, std::vector<int>>& reps,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&reps); pv) {
          return mx::tile(a, {*pv}, s);
        } else {
          return mx::tile(a, std::get<std::vector<int>>(reps), s);
        }
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tile(a: array, reps: int | Sequence[int], /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Construct an array by repeating ``a`` the number of times given by ``reps``.

      Args:
        a (array): Input array
        reps (int or list(int)): The number of times to repeat ``a`` along each axis.

      Returns:
        array: The tiled array.
    )pbdoc");
  m.def(
      "addmm",
      &mx::addmm,
      nb::arg(),
      nb::arg(),
      nb::arg(),
      "alpha"_a = 1.0f,
      "beta"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def addmm(c: array, a: array, b: array, /, alpha: float = 1.0, beta: float = 1.0,  *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Matrix multiplication with addition and optional scaling.

        Perform the (possibly batched) matrix multiplication of two arrays and add to the result
        with optional scaling factors.

        Args:
            c (array): Input array or scalar.
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            alpha (float, optional): Scaling factor for the
                matrix product of ``a`` and ``b`` (default: ``1``)
            beta (float, optional): Scaling factor for ``c`` (default: ``1``)

        Returns:
            array: ``alpha * (a @ b)  + beta * c``
      )pbdoc");
  m.def(
      "block_masked_mm",
      &mx::block_masked_mm,
      nb::arg(),
      nb::arg(),
      "block_size"_a = 64,
      "mask_out"_a = nb::none(),
      "mask_lhs"_a = nb::none(),
      "mask_rhs"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def block_masked_mm(a: array, b: array, /, block_size: int = 64, mask_out: array | None = None, mask_lhs: array | None = None, mask_rhs: array | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Matrix multiplication with block masking.

        Perform the (possibly batched) matrix multiplication of two arrays and with blocks
        of size ``block_size x block_size`` optionally masked out.

        Assuming ``a`` with shape (..., `M`, `K`) and b with shape (..., `K`, `N`)

        * ``lhs_mask`` must have shape (..., :math:`\lceil` `M` / ``block_size`` :math:`\rceil`, :math:`\lceil` `K` / ``block_size`` :math:`\rceil`)

        * ``rhs_mask`` must have shape (..., :math:`\lceil` `K` / ``block_size`` :math:`\rceil`, :math:`\lceil` `N` / ``block_size`` :math:`\rceil`)

        * ``out_mask`` must have shape (..., :math:`\lceil` `M` / ``block_size`` :math:`\rceil`, :math:`\lceil` `N` / ``block_size`` :math:`\rceil`)

        Note: Only ``block_size=64`` and ``block_size=32`` are currently supported

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            block_size (int): Size of blocks to be masked. Must be ``32`` or ``64``. Default: ``64``.
            mask_out (array, optional): Mask for output. Default: ``None``.
            mask_lhs (array, optional): Mask for ``a``. Default: ``None``.
            mask_rhs (array, optional): Mask for ``b``. Default: ``None``.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "gather_mm",
      &mx::gather_mm,
      nb::arg(),
      nb::arg(),
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      nb::kw_only(),
      "sorted_indices"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_mm(a: array, b: array, /, lhs_indices: array, rhs_indices: array, *, sorted_indices: bool = False, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Matrix multiplication with matrix-level gather.

        Performs a gather of the operands with the given indices followed by a
        (possibly batched) matrix multiplication of two arrays.  This operation
        is more efficient than explicitly applying a :func:`take` followed by a
        :func:`matmul`.

        The indices ``lhs_indices`` and ``rhs_indices`` contain flat indices
        along the batch dimensions (i.e. all but the last two dimensions) of
        ``a`` and ``b`` respectively.

        For ``a`` with shape ``(A1, A2, ..., AS, M, K)``, ``lhs_indices``
        contains indices from the range ``[0, A1 * A2 * ... * AS)``

        For ``b`` with shape ``(B1, B2, ..., BS, M, K)``, ``rhs_indices``
        contains indices from the range ``[0, B1 * B2 * ... * BS)``

        If only one index is passed and it is sorted, the ``sorted_indices``
        flag can be passed for a possible faster implementation.

        Args:
            a (array): Input array.
            b (array): Input array.
            lhs_indices (array, optional): Integer indices for ``a``. Default: ``None``
            rhs_indices (array, optional): Integer indices for ``b``. Default: ``None``
            sorted_indices (bool, optional): May allow a faster implementation
              if the passed indices are sorted. Default: ``False``.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "diagonal",
      &mx::diagonal,
      "a"_a,
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "stream"_a = nb::none(),
      nb::sig(
          "def diagonal(a: array, offset: int = 0, axis1: int = 0, axis2: int = 1, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return specified diagonals.

        If ``a`` is 2-D, then a 1-D array containing the diagonal at the given
        ``offset`` is returned.

        If ``a`` has more than two dimensions, then ``axis1`` and ``axis2``
        determine the 2D subarrays from which diagonals are extracted. The new
        shape is the original shape with ``axis1`` and ``axis2`` removed and a
        new dimension inserted at the end corresponding to the diagonal.

        Args:
          a (array): Input array
          offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Default: ``0``.
          axis1 (int, optional): The first axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``0``.
          axis2 (int, optional): The second axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``1``.

        Returns:
            array: The diagonals of the array.
      )pbdoc");
  m.def(
      "diag",
      &mx::diag,
      nb::arg(),
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def diag(a: array, /, k: int = 0, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Extract a diagonal or construct a diagonal matrix.
        If ``a`` is 1-D then a diagonal matrix is constructed with ``a`` on the
        :math:`k`-th diagonal. If ``a`` is 2-D then the :math:`k`-th diagonal is
        returned.

        Args:
            a (array): 1-D or 2-D input array.
            k (int, optional): The diagonal to extract or construct.
                Default: ``0``.

        Returns:
            array: The extracted diagonal or the constructed diagonal matrix.
        )pbdoc");
  m.def(
      "trace",
      [](const mx::array& a,
         int offset,
         int axis1,
         int axis2,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (!dtype.has_value()) {
          return mx::trace(a, offset, axis1, axis2, s);
        }
        return mx::trace(a, offset, axis1, axis2, dtype.value(), s);
      },
      nb::arg(),
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def trace(a: array, /, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Dtype | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Return the sum along a specified diagonal in the given array.

        Args:
          a (array): Input array
          offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Default: ``0``.
          axis1 (int, optional): The first axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``0``.
          axis2 (int, optional): The second axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``1``.
          dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type is inferred from the input array.

        Returns:
            array: Sum of specified diagonal.
        )pbdoc");
  m.def(
      "atleast_1d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_1d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_1d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_1d(*arys: array, stream: StreamOrDevice = None) -> array | list[array]"),
      R"pbdoc(
        Convert all arrays to have at least one dimension.

        Args:
            *arys: Input arrays.
            stream (StreamOrDevice, optional): The stream to execute the operation on.

        Returns:
            array or list(array): An array or list of arrays with at least one dimension.
        )pbdoc");
  m.def(
      "atleast_2d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_2d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_2d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_2d(*arys: array, stream: StreamOrDevice = None) -> array | list[array]"),
      R"pbdoc(
        Convert all arrays to have at least two dimensions.

        Args:
            *arys: Input arrays.
            stream (StreamOrDevice, optional): The stream to execute the operation on.

        Returns:
            array or list(array): An array or list of arrays with at least two dimensions.
        )pbdoc");
  m.def(
      "atleast_3d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_3d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_3d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_3d(*arys: array, stream: StreamOrDevice = None) -> array | list[array]"),
      R"pbdoc(
        Convert all arrays to have at least three dimensions.

        Args:
            *arys: Input arrays.
            stream (StreamOrDevice, optional): The stream to execute the operation on.

        Returns:
            array or list(array): An array or list of arrays with at least three dimensions.
        )pbdoc");
  m.def(
      "issubdtype",
      [](const nb::object& d1, const nb::object& d2) {
        auto dispatch_second = [](const auto& t1, const auto& d2) {
          if (nb::isinstance<mx::Dtype>(d2)) {
            return mx::issubdtype(t1, nb::cast<mx::Dtype>(d2));
          } else if (nb::isinstance<mx::Dtype::Category>(d2)) {
            return mx::issubdtype(t1, nb::cast<mx::Dtype::Category>(d2));
          } else {
            throw std::invalid_argument(
                "[issubdtype] Received invalid type for second input.");
          }
        };
        if (nb::isinstance<mx::Dtype>(d1)) {
          return dispatch_second(nb::cast<mx::Dtype>(d1), d2);
        } else if (nb::isinstance<mx::Dtype::Category>(d1)) {
          return dispatch_second(nb::cast<mx::Dtype::Category>(d1), d2);
        } else {
          throw std::invalid_argument(
              "[issubdtype] Received invalid type for first input.");
        }
      },
      ""_a,
      ""_a,
      nb::sig(
          "def issubdtype(arg1: Dtype | DtypeCategory, arg2: Dtype | DtypeCategory) -> bool"),
      R"pbdoc(
        Check if a :obj:`Dtype` or :obj:`DtypeCategory` is a subtype
        of another.

        Args:
            arg1 (Union[Dtype, DtypeCategory]: First dtype or category.
            arg2 (Union[Dtype, DtypeCategory]: Second dtype or category.

        Returns:
            bool:
               A boolean indicating if the first input is a subtype of the
               second input.

        Example:

          >>> ints = mx.array([1, 2, 3], dtype=mx.int32)
          >>> mx.issubdtype(ints.dtype, mx.integer)
          True
          >>> mx.issubdtype(ints.dtype, mx.floating)
          False

          >>> floats = mx.array([1, 2, 3], dtype=mx.float32)
          >>> mx.issubdtype(floats.dtype, mx.integer)
          False
          >>> mx.issubdtype(floats.dtype, mx.floating)
          True

          Similar types of different sizes are not subdtypes of each other:

          >>> mx.issubdtype(mx.float64, mx.float32)
          False
          >>> mx.issubdtype(mx.float32, mx.float64)
          False

          but both are subtypes of `floating`:

          >>> mx.issubdtype(mx.float64, mx.floating)
          True
          >>> mx.issubdtype(mx.float32, mx.floating)
          True

          For convenience, dtype-like objects are allowed too:

          >>> mx.issubdtype(mx.float32, mx.inexact)
          True
          >>> mx.issubdtype(mx.signedinteger, mx.floating)
          False
      )pbdoc");
  m.def(
      "result_type",
      [](const nb::args& arrays_and_dtypes) {
        auto to_dtype = [](const nb::handle& v) -> mx::Dtype {
          if (nb::isinstance<mx::array>(v)) {
            return nb::cast<mx::array>(v).dtype();
          } else if (nb::isinstance<mx::Dtype>(v)) {
            return nb::cast<mx::Dtype>(v);
          } else {
            throw std::invalid_argument(
                "[result_type] Inputs must be arrays or dtypes.");
          }
        };
        if (arrays_and_dtypes.size() == 0) {
          throw std::invalid_argument(
              "[result_type] At least one array or dtype is required.");
        }
        mx::Dtype t = to_dtype(arrays_and_dtypes[0]);
        for (size_t i = 1; i < arrays_and_dtypes.size(); ++i) {
          t = mx::promote_types(t, to_dtype(arrays_and_dtypes[i]));
        }
        return t;
      },
      nb::sig("def result_type(*arrays_and_dtypes: array | Dtype) -> Dtype"),
      R"pbdoc(
        The type that results from applying type promotion to the inputs.

        Args:
            *arrays_and_dtypes (array or Dtype): A variable number of arrays
              or dtypes.

        Returns:
            Dtype: The result type.
      )pbdoc");
  m.def(
      "can_cast",
      [](const nb::object& from_, const mx::Dtype& to) {
        mx::Dtype from_dtype = mx::bool_;
        if (nb::isinstance<mx::array>(from_)) {
          from_dtype = nb::cast<mx::array>(from_).dtype();
        } else if (nb::isinstance<mx::Dtype>(from_)) {
          from_dtype = nb::cast<mx::Dtype>(from_);
        } else {
          throw std::invalid_argument(
              "[can_cast] `from_` must be an array or a dtype.");
        }
        return mx::promote_types(from_dtype, to) == to;
      },
      "from_"_a,
      "to"_a,
      nb::sig("def can_cast(from_: array | Dtype, to: Dtype) -> bool"),
      R"pbdoc(
        Determine if one data type can be cast to another according to type
        promotion rules.

        ``from_`` can be cast to ``to`` if promoting the two together gives
        back ``to``.

        Args:
            from_ (array or Dtype): The source array or dtype.
            to (Dtype): The destination dtype.

        Returns:
            bool: Whether the cast can be performed.
      )pbdoc");
  m.def(
      "isdtype",
      [](const mx::Dtype& dtype, const nb::object& kind) {
        auto check_one = [&dtype](const nb::handle& k) -> bool {
          if (nb::isinstance<mx::Dtype>(k)) {
            return dtype == nb::cast<mx::Dtype>(k);
          } else if (nb::isinstance<nb::str>(k)) {
            auto s = nb::cast<std::string>(k);
            if (s == "bool") {
              return dtype == mx::bool_;
            } else if (s == "signed integer") {
              return mx::issubdtype(dtype, mx::signedinteger);
            } else if (s == "unsigned integer") {
              return mx::issubdtype(dtype, mx::unsignedinteger);
            } else if (s == "integral") {
              return mx::issubdtype(dtype, mx::integer);
            } else if (s == "real floating") {
              return mx::issubdtype(dtype, mx::floating);
            } else if (s == "complex floating") {
              return mx::issubdtype(dtype, mx::complexfloating);
            } else if (s == "numeric") {
              return mx::issubdtype(dtype, mx::number);
            } else {
              std::ostringstream msg;
              msg << "[isdtype] Unknown data type kind: '" << s << "'.";
              throw std::invalid_argument(msg.str());
            }
          } else {
            throw std::invalid_argument(
                "[isdtype] `kind` must be a dtype, a string, or a tuple of "
                "dtypes and strings.");
          }
        };
        if (nb::isinstance<nb::tuple>(kind)) {
          for (auto k : nb::cast<nb::tuple>(kind)) {
            if (check_one(k)) {
              return true;
            }
          }
          return false;
        }
        return check_one(kind);
      },
      "dtype"_a,
      "kind"_a,
      nb::sig(
          "def isdtype(dtype: Dtype, kind: Dtype | str | tuple[Dtype | str, ...]) -> bool"),
      R"pbdoc(
        Test whether a dtype belongs to one or more data type kinds.

        Args:
            dtype (Dtype): The dtype to test.
            kind (Dtype, str, or tuple): A dtype, a kind string, or a tuple
              of dtypes and kind strings. Supported kind strings are
              ``"bool"``, ``"signed integer"``, ``"unsigned integer"``,
              ``"integral"``, ``"real floating"``, ``"complex floating"``,
              and ``"numeric"``.

        Returns:
            bool: ``True`` if ``dtype`` matches any of the given kinds.
      )pbdoc");
  m.def(
      "bitwise_and",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_and(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_and(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise bitwise and.

        Take the bitwise and of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise and ``a & b``.
      )pbdoc");
  m.def(
      "bitwise_or",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_or(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_or(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise bitwise or.

        Take the bitwise or of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise or``a | b``.
      )pbdoc");
  m.def(
      "bitwise_xor",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_xor(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_xor(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise bitwise xor.

        Take the bitwise exclusive or of two arrays with numpy-style
        broadcasting semantics. Either or both input arrays can also be
        scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise xor ``a ^ b``.
      )pbdoc");
  m.def(
      "left_shift",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::left_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def left_shift(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise left shift.

        Shift the bits of the first input to the left by the second using
        numpy-style broadcasting semantics. Either or both input arrays can
        also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise left shift ``a << b``.
      )pbdoc");
  m.def(
      "right_shift",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::right_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def right_shift(a: scalar | array, b: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise right shift.

        Shift the bits of the first input to the right by the second using
        numpy-style broadcasting semantics. Either or both input arrays can
        also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise right shift ``a >> b``.
      )pbdoc");
  m.def(
      "bitwise_invert",
      [](const ScalarOrArray& a_, mx::StreamOrDevice s) {
        auto a = to_array(a_);
        return mx::bitwise_invert(a, s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_invert(a: scalar | array, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Element-wise bitwise inverse.

        Take the bitwise complement of the input.

        Args:
            a (array): Input array or scalar.

        Returns:
            array: The bitwise inverse ``~a``.
      )pbdoc");
  m.def(
      "view",
      [](const ScalarOrArray& a, const mx::Dtype& dtype, mx::StreamOrDevice s) {
        return mx::view(to_array(a), dtype, s);
      },
      nb::arg(),
      "dtype"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def view(a: scalar | array, dtype: Dtype, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        View the array as a different type.

        The output shape changes along the last axis if the input array's
        type and the input ``dtype`` do not have the same size.

        Note: the view op does not imply that the input and output arrays share
        their underlying data. The view only gaurantees that the binary
        representation of each element (or group of elements) is the same.

        Args:
            a (array): Input array or scalar.
            dtype (Dtype): The data type to change to.

        Returns:
            array: The array with the new type.
      )pbdoc");
  m.def(
      "hadamard_transform",
      &mx::hadamard_transform,
      nb::arg(),
      "scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def hadamard_transform(a: array, scale: float | None = None, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Perform the Walsh-Hadamard transform along the final axis.

        Equivalent to:

        .. code-block:: python

           from scipy.linalg import hadamard

           y = (hadamard(len(x)) @ x) * scale

        Supports sizes ``n = m*2^k`` for ``m`` in ``(1, 12, 20, 28)`` and ``2^k
        <= 8192`` for float32 and ``2^k <= 16384`` for float16/bfloat16.

        Args:
            a (array): Input array or scalar.
            scale (float): Scale the output by this factor.
              Defaults to ``1/sqrt(a.shape[-1])`` so that the Hadamard matrix is orthonormal.

        Returns:
            array: The transformed array.
      )pbdoc");
  m.def(
      "einsum_path",
      [](const std::string& equation, const nb::args& operands) {
        auto arrays_list = nb::cast<std::vector<mx::array>>(operands);
        auto [path, str] = mx::einsum_path(equation, arrays_list);
        // Convert to list of tuples
        std::vector<nb::tuple> tuple_path;
        for (auto& p : path) {
          tuple_path.push_back(nb::tuple(nb::cast(p)));
        }
        return std::make_pair(tuple_path, str);
      },
      "subscripts"_a,
      "operands"_a,
      nb::sig("def einsum_path(subscripts: str, *operands)"),
      R"pbdoc(

      Compute the contraction order for the given Einstein summation.

      Args:
        subscripts (str): The Einstein summation convention equation.
        *operands (array): The input arrays.

      Returns:
        tuple(list(tuple(int, int)), str):
          The einsum path and a string containing information about the
          chosen path.
    )pbdoc");
  m.def(
      "einsum",
      [](const std::string& subscripts,
         const nb::args& operands,
         mx::StreamOrDevice s) {
        auto arrays_list = nb::cast<std::vector<mx::array>>(operands);
        return mx::einsum(subscripts, arrays_list, s);
      },
      "subscripts"_a,
      "operands"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def einsum(subscripts: str, *operands, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(

      Perform the Einstein summation convention on the operands.

      Args:
        subscripts (str): The Einstein summation convention equation.
        *operands (array): The input arrays.

      Returns:
        array: The output array.
    )pbdoc");
  m.def(
      "roll",
      [](const mx::array& a,
         const std::variant<int, mx::Shape>& shift,
         const IntOrVec& axis,
         mx::StreamOrDevice s) {
        return std::visit(
            [&](auto sh, auto ax) -> mx::array {
              if constexpr (std::is_same_v<decltype(ax), std::monostate>) {
                return mx::roll(a, sh, s);
              } else {
                return mx::roll(a, sh, ax, s);
              }
            },
            shift,
            axis);
      },
      nb::arg(),
      "shift"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def roll(a: array, shift: int | tuple[int], axis: None | int | tuple[int] = None, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Roll array elements along a given axis.

        Elements that are rolled beyond the end of the array are introduced at
        the beggining and vice-versa.

        If the axis is not provided the array is flattened, rolled and then the
        shape is restored.

        Args:
          a (array): Input array
          shift (int or tuple(int)): The number of places by which elements
            are shifted. If positive the array is rolled to the right, if
            negative it is rolled to the left. If an int is provided but the
            axis is a tuple then the same value is used for all axes.
          axis (int or tuple(int), optional): The axis or axes along which to
            roll the elements.
      )pbdoc");
  m.def(
      "real",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::real(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def real(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns the real part of a complex array.

        Args:
            a (array): Input array.

        Returns:
            array: The real part of ``a``.
      )pbdoc");
  m.def(
      "imag",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::imag(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def imag(a: array, /, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Returns the imaginary part of a complex array.

        Args:
            a (array): Input array.

        Returns:
            array: The imaginary part of ``a``.
      )pbdoc");
  m.def(
      "slice",
      [](const mx::array& a,
         const mx::array& start_indices,
         std::vector<int> axes,
         mx::Shape slice_size,
         mx::StreamOrDevice s) {
        return mx::slice(
            a, start_indices, std::move(axes), std::move(slice_size), s);
      },
      nb::arg(),
      "start_indices"_a,
      "axes"_a,
      "slice_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def slice(a: array, start_indices: array, axes: Sequence[int], slice_size: Sequence[int], *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Extract a sub-array from the input array.

        Args:
          a (array): Input array
          start_indices (array): The index location to start the slice at.
          axes (tuple(int)): The axes corresponding to the indices in ``start_indices``.
          slice_size (tuple(int)): The size of the slice.

        Returns:
          array: The sliced output array.

        Example:

          >>> a = mx.array([[1, 2, 3], [4, 5, 6]])
          >>> mx.slice(a, start_indices=mx.array(1), axes=(0,), slice_size=(1, 2))
          array([[4, 5]], dtype=int32)
          >>>
          >>> mx.slice(a, start_indices=mx.array(1), axes=(1,), slice_size=(2, 1))
          array([[2],
                 [5]], dtype=int32)
      )pbdoc");
  m.def(
      "slice_update",
      [](const mx::array& src,
         const mx::array& update,
         const mx::array& start_indices,
         std::vector<int> axes,
         mx::StreamOrDevice s) {
        return mx::slice_update(src, update, start_indices, axes, s);
      },
      nb::arg(),
      "update"_a,
      "start_indices"_a,
      "axes"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def slice_update(a: array, update: array, start_indices: array, axes: Sequence[int], *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
        Update a sub-array of the input array.

        Args:
          a (array): The input array to update
          update (array): The update array.
          start_indices (array): The index location to start the slice at.
          axes (tuple(int)): The axes corresponding to the indices in ``start_indices``.

        Returns:
          array: The output array with the same shape and type as the input.

        Example:

          >>> a = mx.zeros((3, 3))
          >>> mx.slice_update(a, mx.ones((1, 2)), start_indices=mx.array(1, 1), axes=(0, 1))
          array([[0, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0]], dtype=float32)
      )pbdoc");
  m.def(
      "contiguous",
      &mx::contiguous,
      nb::arg(),
      "allow_col_major"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def contiguous(a: array, /, allow_col_major: bool = False, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Force an array to be row contiguous. Copy if necessary.

      Args:
        a (array): The input to make contiguous
        allow_col_major (bool): Consider column major as contiguous and don't copy

      Returns:
        array: The row or col contiguous output.
    )pbdoc");
  m.def(
      "broadcast_shapes",
      [](const nb::args& shapes) {
        if (shapes.size() == 0)
          throw std::invalid_argument(
              "[broadcast_shapes] Must provide at least one shape.");

        mx::Shape result = nb::cast<mx::Shape>(shapes[0]);
        for (size_t i = 1; i < shapes.size(); ++i) {
          if (!nb::isinstance<mx::Shape>(shapes[i]) &&
              !nb::isinstance<nb::tuple>(shapes[i]))
            throw std::invalid_argument(
                "[broadcast_shapes] Expects a sequence of shapes (tuple or list of ints).");
          result = mx::broadcast_shapes(result, nb::cast<mx::Shape>(shapes[i]));
        }

        return nb::tuple(nb::cast(result));
      },
      nb::sig("def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int]"),
      R"pbdoc(
        Broadcast shapes.

        Returns the shape that results from broadcasting the supplied array shapes
        against each other.

        Args:
            *shapes (Sequence[int]): The shapes to broadcast.

        Returns:
            tuple: The broadcasted shape.

        Raises:
            ValueError: If the shapes cannot be broadcast.

        Example:
            >>> mx.broadcast_shapes((1,), (3, 1))
            (3, 1)
            >>> mx.broadcast_shapes((6, 7), (5, 6, 1), (7,))
            (5, 6, 7)
            >>> mx.broadcast_shapes((5, 1, 4), (1, 3, 1))
            (5, 3, 4)
      )pbdoc");
  m.def(
      "depends",
      [](const nb::object& inputs_, const nb::object& deps_) {
        bool return_vec = false;
        std::vector<mx::array> inputs;
        std::vector<mx::array> deps;
        if (nb::isinstance<mx::array>(inputs_)) {
          inputs = {nb::cast<mx::array>(inputs_)};
        } else {
          return_vec = true;
          inputs = {nb::cast<std::vector<mx::array>>(inputs_)};
        }
        if (nb::isinstance<mx::array>(deps_)) {
          deps = {nb::cast<mx::array>(deps_)};
        } else {
          deps = {nb::cast<std::vector<mx::array>>(deps_)};
        }
        auto out = depends(inputs, deps);
        if (return_vec) {
          return nb::cast(out);
        } else {
          return nb::cast(out[0]);
        }
      },
      nb::arg(),
      nb::arg(),
      nb::sig(
          "def depends(inputs: array | Sequence[array], dependencies: array | Sequence[array])"),
      R"pbdoc(
        Insert dependencies between arrays in the graph. The outputs are
        identical to ``inputs`` but with dependencies on ``dependencies``.

        Args:
            inputs (array or Sequence[array]): The input array or arrays.
            dependencies (array or Sequence[array]): The array or arrays
              to insert dependencies on.

        Returns:
            array or Sequence[array]: The outputs which depend on dependencies.
      )pbdoc");
  m.def(
      "qqmm",
      &mx::qqmm,
      nb::arg(), // x
      nb::arg(), // w_q
      "scales"_a = nb::none(), // scales w
      "group_size"_a = nb::none(),
      "bits"_a = nb::none(),
      "mode"_a = "nvfp4",
      "global_scale_x"_a = nb::none(),
      "global_scale_w"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def qqmm(x: array, w: array, scales: array | None = None, group_size: int | None = None, bits: int | None = None, mode: str = 'nvfp4', global_scale_x: array | None = None, global_scale_w: array | None = None, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Perform a matrix multiplication using a possibly quantized weight matrix
      ``w`` and a non-quantized input ``x``. The input ``x`` is quantized on the
      fly. The weight matrix ``w`` is used as-is if it is already quantized;
      otherwise, it is quantized on the fly.

      If ``w`` is quantized, ``scales`` must be provided, and ``group_size``,
      ``bits``, and ``mode`` must match the parameters that were used to quantize
      ``w``.

      Notes:
        If ``w`` is expected to receive gradients, it must be provided in
        non-quantized form.

        If ``x`` and `w`` are not quantized, their data types must be ``float32``,
        ``float16``, or ``bfloat16``.
        If ``w`` is quantized, it must be packed in unsigned integers.
        ``global_scale_x`` and ``global_scale_w`` are only used for ``nvfp4`` quantization.

      Args:
        x (array): Input array.
        w (array): Weight matrix. If quantized, it is packed in unsigned integers.
        scales (array, optional): The scales to use per ``group_size`` elements of
          ``w`` if ``w`` is quantized. Default: ``None``.
        group_size (int, optional): Number of elements in ``x`` and ``w`` that
          share a scale. See supported values and defaults in the
          :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
        bits (int, optional): Number of bits used to represent each element of
          ``x`` and ``w``. See supported values and defaults in the
          :ref:`table of quantization modes <quantize-modes>`. Default: ``None``.
        mode (str, optional): The quantization mode. Default: ``"nvfp4"``.
          Supported modes are ``nvfp4`` and ``mxfp8``. See the
          :ref:`table of quantization modes <quantize-modes>` for details.
        global_scale (array, optional): The per-input float32 scale used for x
            with ``"nvfp4"`` quantization. Default: ``None``.
        global_scale_w (array, optional): The per-input float32 scale used for w
            with ``"nvfp4"`` quantization. Default: ``None``.
      Returns:
        array: The result of the multiplication of quantized ``x`` with quantized ``w``.
        needed).
  )pbdoc");
  m.def(
      "from_fp8",
      &mx::from_fp8,
      nb::arg(),
      "dtype"_a = mx::bfloat16,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def from_fp8(x: array, dtype: Dtype = bfloat16, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Convert the array from fp8 (e4m3) to another floating-point type.

      Args:
        x (array): The input fp8 array with type ``uint8``.
        dtype (Dtype): The data type to convert to. Default: ``bfloat16``.

      Returns:
        array: The array converted from fp8.
  )pbdoc");
  m.def(
      "to_fp8",
      &mx::to_fp8,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def to_fp8(x: array, *, stream: StreamOrDevice = None) -> array"),
      R"pbdoc(
      Convert the array to fp8 (e4m3) from another floating-point type.

      Args:
        x (array): The input array.

      Returns:
        array: The array converted to fp8 with type ``uint8``.
  )pbdoc");
  // Array API standard aliases (https://data-apis.org/array-api/latest/).
  m.attr("acos") = m.attr("arccos");
  m.attr("acosh") = m.attr("arccosh");
  m.attr("asin") = m.attr("arcsin");
  m.attr("asinh") = m.attr("arcsinh");
  m.attr("atan") = m.attr("arctan");
  m.attr("atanh") = m.attr("arctanh");
  m.attr("atan2") = m.attr("arctan2");
  m.attr("bitwise_left_shift") = m.attr("left_shift");
  m.attr("bitwise_right_shift") = m.attr("right_shift");
  m.attr("cumulative_prod") = m.attr("cumprod");
  m.attr("cumulative_sum") = m.attr("cumsum");
  m.attr("empty") = m.attr("zeros");
  m.attr("empty_like") = m.attr("zeros_like");
  m.attr("matrix_transpose") = m.attr("transpose");
  m.attr("pow") = m.attr("power");
}
