// Copyright © 2023 Apple Inc.

#include <json.hpp>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stack>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

#include "mlx/backend/cuda/cuda.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"
#define ST_F8_E8M0 "F8_E8M0"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

namespace {

size_t tensor_nbytes(const SafetensorsTensorSpec& spec) {
  size_t nbytes = spec.dtype.size();
  for (auto dim : spec.shape) {
    if (dim < 0 ||
        (dim != 0 && nbytes > std::numeric_limits<size_t>::max() /
                static_cast<size_t>(dim))) {
      throw std::invalid_argument(
          "[ExpertSafetensorsDirect] invalid or overflowing tensor shape");
    }
    nbytes *= static_cast<size_t>(dim);
  }
  return nbytes;
}

#ifndef _WIN32
void pread_exact(int fd, char* destination, size_t nbytes, size_t offset) {
  size_t completed = 0;
  while (completed < nbytes) {
    auto result = ::pread(
        fd,
        destination + completed,
        nbytes - completed,
        static_cast<off_t>(offset + completed));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      std::ostringstream message;
      message << "[ExpertSafetensorsDirect] positioned read failed at offset "
              << (offset + completed);
      throw std::runtime_error(message.str());
    }
    completed += static_cast<size_t>(result);
  }
}

uint32_t read_u32_le(const unsigned char* source) {
  return static_cast<uint32_t>(source[0]) |
      (static_cast<uint32_t>(source[1]) << 8) |
      (static_cast<uint32_t>(source[2]) << 16) |
      (static_cast<uint32_t>(source[3]) << 24);
}

std::vector<unsigned char>& scalex_encoded_scratch(size_t nbytes) {
  // Expert refills run on a bounded native worker pool. Retaining one grow-only
  // buffer per worker avoids tens of thousands of allocator round trips while
  // keeping scratch private to the calling thread.
  thread_local std::vector<unsigned char> scratch;
  scratch.resize(nbytes);
  return scratch;
}

void decode_scalex_mode_a_into(
    const std::vector<unsigned char>& encoded,
    const std::array<char*, 3>& destinations,
    const std::array<size_t, 3>& destination_nbytes) {
  if (encoded.size() < 24 || std::memcmp(encoded.data(), "LSA1", 4) != 0) {
    throw std::runtime_error("[ScaleXModeADirect] invalid Mode-A magic");
  }
  const auto codec = encoded[4];
  if (codec > 2) {
    throw std::runtime_error("[ScaleXModeADirect] invalid Mode-A codec");
  }
  const size_t raw_size = read_u32_le(encoded.data() + 12);
  const size_t primary_size = read_u32_le(encoded.data() + 16);
  const size_t exception_count = read_u32_le(encoded.data() + 20);
  const size_t expected_raw = std::accumulate(
      destination_nbytes.begin(), destination_nbytes.end(), size_t{0});
  if (raw_size != expected_raw ||
      exception_count >
          (std::numeric_limits<size_t>::max() - 24 - primary_size) / 5 ||
      24 + primary_size + 5 * exception_count != encoded.size()) {
    throw std::runtime_error("[ScaleXModeADirect] invalid Mode-A record size");
  }

  const auto* primary = encoded.data() + 24;
  size_t global_start = 0;
  if (codec == 0) {
    if (primary_size != raw_size || exception_count != 0) {
      throw std::runtime_error("[ScaleXModeADirect] invalid raw Mode-A record");
    }
    for (size_t tensor = 0; tensor < destinations.size(); ++tensor) {
      std::memcpy(
          destinations[tensor],
          primary + global_start,
          destination_nbytes[tensor]);
      global_start += destination_nbytes[tensor];
    }
    return;
  }

  const size_t bits = codec == 1 ? 1 : 2;
  const size_t expected_primary = (raw_size * bits + 7) / 8;
  if (primary_size != expected_primary) {
    throw std::runtime_error("[ScaleXModeADirect] invalid Mode-A primary size");
  }
  const size_t values_per_byte = codec == 1 ? 8 : 4;
  const bool tensor_boundaries_aligned = std::all_of(
      destination_nbytes.begin(),
      destination_nbytes.end(),
      [values_per_byte](size_t value) {
        return value % values_per_byte == 0;
      });
  if (!tensor_boundaries_aligned) {
    const size_t code_mask = codec == 1 ? 1 : 3;
    for (size_t tensor = 0; tensor < destinations.size(); ++tensor) {
      auto* destination =
          reinterpret_cast<unsigned char*>(destinations[tensor]);
      for (size_t local = 0; local < destination_nbytes[tensor]; ++local) {
        const size_t index = global_start + local;
        const size_t bit_offset = index * bits;
        const auto code =
            (primary[bit_offset / 8] >> (bit_offset % 8)) & code_mask;
        destination[local] = encoded[5 + code];
      }
      global_start += destination_nbytes[tensor];
    }
  } else if (codec == 1) {
    struct OneBitLookup {
      bool initialized{false};
      std::array<unsigned char, 2> palette{};
      std::array<uint64_t, 256> values{};
    };
    thread_local OneBitLookup cache;
    const std::array<unsigned char, 2> palette{encoded[5], encoded[6]};
    if (!cache.initialized || cache.palette != palette) {
      for (size_t packed = 0; packed < cache.values.size(); ++packed) {
        uint64_t expanded = 0;
        for (size_t code = 0; code < 8; ++code) {
          expanded |= static_cast<uint64_t>(
                          palette[(packed >> code) & 1])
              << (8 * code);
        }
        cache.values[packed] = expanded;
      }
      cache.palette = palette;
      cache.initialized = true;
    }
    for (size_t tensor = 0; tensor < destinations.size(); ++tensor) {
      if (destination_nbytes[tensor] % 8 != 0 || global_start % 8 != 0) {
        throw std::runtime_error(
            "[ScaleXModeADirect] one-bit tensor boundary is not byte aligned");
      }
      auto* destination =
          reinterpret_cast<unsigned char*>(destinations[tensor]);
      const auto* tensor_primary = primary + global_start / 8;
      const size_t packed_bytes = destination_nbytes[tensor] / 8;
      for (size_t index = 0; index < packed_bytes; ++index) {
        const uint64_t expanded = cache.values[tensor_primary[index]];
        std::memcpy(destination + 8 * index, &expanded, sizeof(expanded));
      }
      global_start += destination_nbytes[tensor];
    }
  } else {
    struct TwoBitLookup {
      bool initialized{false};
      std::array<unsigned char, 4> palette{};
      std::array<uint32_t, 256> values{};
    };
    thread_local TwoBitLookup cache;
    const std::array<unsigned char, 4> palette{
        encoded[5], encoded[6], encoded[7], encoded[8]};
    if (!cache.initialized || cache.palette != palette) {
      for (size_t packed = 0; packed < cache.values.size(); ++packed) {
        uint32_t expanded = 0;
        for (size_t code = 0; code < 4; ++code) {
          expanded |= static_cast<uint32_t>(
                          palette[(packed >> (2 * code)) & 3])
              << (8 * code);
        }
        cache.values[packed] = expanded;
      }
      cache.palette = palette;
      cache.initialized = true;
    }
    for (size_t tensor = 0; tensor < destinations.size(); ++tensor) {
      if (destination_nbytes[tensor] % 4 != 0 || global_start % 4 != 0) {
        throw std::runtime_error(
            "[ScaleXModeADirect] two-bit tensor boundary is not byte aligned");
      }
      auto* destination =
          reinterpret_cast<unsigned char*>(destinations[tensor]);
      const auto* tensor_primary = primary + global_start / 4;
      const size_t packed_bytes = destination_nbytes[tensor] / 4;
      for (size_t index = 0; index < packed_bytes; ++index) {
        const uint32_t expanded = cache.values[tensor_primary[index]];
        std::memcpy(destination + 4 * index, &expanded, sizeof(expanded));
      }
      global_start += destination_nbytes[tensor];
    }
  }

  const auto* positions = primary + primary_size;
  const auto* values = positions + 4 * exception_count;
  uint32_t previous = 0;
  for (size_t index = 0; index < exception_count; ++index) {
    const uint32_t position = read_u32_le(positions + 4 * index);
    if (position >= raw_size || (index != 0 && position <= previous)) {
      throw std::runtime_error(
          "[ScaleXModeADirect] invalid Mode-A exception position");
    }
    size_t tensor = 0;
    size_t local = position;
    while (tensor < destination_nbytes.size() &&
           local >= destination_nbytes[tensor]) {
      local -= destination_nbytes[tensor++];
    }
    if (tensor >= destinations.size()) {
      throw std::runtime_error(
          "[ScaleXModeADirect] Mode-A exception exceeds destinations");
    }
    destinations[tensor][local] = static_cast<char>(values[index]);
    previous = position;
  }
}
#endif

} // namespace

ExpertSafetensorsDirect::ExpertSafetensorsDirect(
    std::string file,
    std::vector<std::vector<SafetensorsTensorSpec>> specs_by_expert,
    bool no_cache,
    bool read_ahead)
    : file_(std::move(file)), specs_by_expert_(std::move(specs_by_expert)) {
#ifdef _WIN32
  throw std::runtime_error(
      "[ExpertSafetensorsDirect] official-direct IO currently requires POSIX preadv");
#else
  if (specs_by_expert_.empty() || specs_by_expert_.front().empty()) {
    throw std::invalid_argument(
        "[ExpertSafetensorsDirect] expert tensor specs must be non-empty");
  }
  fd_ = ::open(file_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error(
        "[ExpertSafetensorsDirect] failed to open file " + file_);
  }
#ifdef __APPLE__
  if ((no_cache && ::fcntl(fd_, F_NOCACHE, 1) != 0) ||
      (!read_ahead && ::fcntl(fd_, F_RDAHEAD, 0) != 0)) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[ExpertSafetensorsDirect] failed to apply macOS file-cache policy");
  }
#else
  if (no_cache || !read_ahead) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[ExpertSafetensorsDirect] file-cache policy is supported only on macOS");
  }
#endif
  struct stat info {};
  if (::fstat(fd_, &info) != 0 || info.st_size < 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[ExpertSafetensorsDirect] failed to stat file " + file_);
  }
  file_nbytes_ = static_cast<size_t>(info.st_size);

  try {
    const auto& canonical = specs_by_expert_.front();
    ranges_by_expert_.reserve(specs_by_expert_.size());
    for (size_t expert_id = 0; expert_id < specs_by_expert_.size(); ++expert_id) {
      const auto& specs = specs_by_expert_[expert_id];
      if (specs.size() != canonical.size()) {
        throw std::invalid_argument(
            "[ExpertSafetensorsDirect] experts must share a tensor count");
      }
      std::vector<size_t> sizes(specs.size());
      std::vector<size_t> order(specs.size());
      std::iota(order.begin(), order.end(), size_t{0});
      for (size_t index = 0; index < specs.size(); ++index) {
        const auto& spec = specs[index];
        const auto& layout = canonical[index];
        if (spec.name.empty() || spec.name != layout.name ||
            spec.dtype != layout.dtype || spec.shape != layout.shape) {
          throw std::invalid_argument(
              "[ExpertSafetensorsDirect] experts must share canonical tensor layouts");
        }
        sizes[index] = tensor_nbytes(spec);
        if (spec.absolute_offset > file_nbytes_ ||
            sizes[index] > file_nbytes_ - spec.absolute_offset) {
          std::ostringstream message;
          message << "[ExpertSafetensorsDirect] tensor '" << spec.name
                  << "' exceeds file bounds for expert " << expert_id;
          throw std::invalid_argument(message.str());
        }
      }
      std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return specs[left].absolute_offset < specs[right].absolute_offset;
      });

      std::vector<ReadRange> ranges;
      for (auto index : order) {
        const auto offset = specs[index].absolute_offset;
        const auto size = sizes[index];
        if (!ranges.empty()) {
          auto& previous = ranges.back();
          const auto previous_end =
              previous.absolute_offset + previous.byte_length;
          if (offset < previous_end) {
            throw std::invalid_argument(
                "[ExpertSafetensorsDirect] source tensor ranges overlap");
          }
          if (offset == previous_end) {
            previous.byte_length += size;
            previous.tensor_indices.push_back(index);
            continue;
          }
        }
        ranges.push_back(ReadRange{offset, size, {index}});
      }
      ranges_by_expert_.push_back(std::move(ranges));
    }
  } catch (...) {
    ::close(fd_);
    fd_ = -1;
    throw;
  }
#endif
}

ExpertSafetensorsDirect::~ExpertSafetensorsDirect() {
#ifndef _WIN32
  if (fd_ >= 0) {
    ::close(fd_);
  }
#endif
}

ScaleXModeADirect::ScaleXModeADirect(
    std::string file,
    std::vector<ScaleXModeARecordSpec> records,
    std::array<size_t, 3> decoded_tensor_nbytes,
    bool no_cache,
    bool read_ahead)
    : file_(std::move(file)),
      records_(std::move(records)),
      decoded_tensor_nbytes_(decoded_tensor_nbytes) {
#ifdef _WIN32
  throw std::runtime_error(
      "[ScaleXModeADirect] decode-on-arrival requires POSIX pread");
#else
  if (records_.empty() ||
      std::any_of(
          decoded_tensor_nbytes_.begin(),
          decoded_tensor_nbytes_.end(),
          [](size_t value) { return value == 0; })) {
    throw std::invalid_argument(
        "[ScaleXModeADirect] records and decoded tensor sizes must be non-empty");
  }
  fd_ = ::open(file_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("[ScaleXModeADirect] failed to open file " + file_);
  }
#ifdef __APPLE__
  if ((no_cache && ::fcntl(fd_, F_NOCACHE, 1) != 0) ||
      (!read_ahead && ::fcntl(fd_, F_RDAHEAD, 0) != 0)) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[ScaleXModeADirect] failed to apply macOS file-cache policy");
  }
#else
  if (no_cache || !read_ahead) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[ScaleXModeADirect] file-cache policy is supported only on macOS");
  }
#endif
  struct stat info {};
  if (::fstat(fd_, &info) != 0 || info.st_size < 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("[ScaleXModeADirect] failed to stat file " + file_);
  }
  file_nbytes_ = static_cast<size_t>(info.st_size);
  for (const auto& record : records_) {
    if (record.encoded_nbytes < 24 || record.absolute_offset > file_nbytes_ ||
        record.encoded_nbytes > file_nbytes_ - record.absolute_offset) {
      ::close(fd_);
      fd_ = -1;
      throw std::invalid_argument(
          "[ScaleXModeADirect] encoded record exceeds file bounds");
    }
  }
#endif
}

ScaleXModeADirect::~ScaleXModeADirect() {
#ifndef _WIN32
  if (fd_ >= 0) {
    ::close(fd_);
  }
#endif
}

void ScaleXModeADirect::load_into(
    size_t expert_id,
    const std::array<char*, 3>& destinations,
    const std::array<size_t, 3>& destination_nbytes) const {
#ifdef _WIN32
  throw std::runtime_error(
      "[ScaleXModeADirect] decode-on-arrival requires POSIX pread");
#else
  if (expert_id >= records_.size()) {
    throw std::out_of_range("[ScaleXModeADirect] expert id is out of range");
  }
  if (destination_nbytes != decoded_tensor_nbytes_ ||
      std::any_of(
          destinations.begin(), destinations.end(), [](char* value) {
            return value == nullptr;
          })) {
    throw std::invalid_argument(
        "[ScaleXModeADirect] destination tensor layout mismatch");
  }
  const auto& record = records_[expert_id];
  auto& encoded = scalex_encoded_scratch(record.encoded_nbytes);
  pread_exact(
      fd_,
      reinterpret_cast<char*>(encoded.data()),
      encoded.size(),
      record.absolute_offset);
  decode_scalex_mode_a_into(encoded, destinations, destination_nbytes);
#endif
}

void ScaleXModeADirect::load_expert_into(
    size_t expert_id,
    const std::array<char*, 3>& scale_destinations,
    const std::array<size_t, 3>& scale_destination_nbytes,
    const std::array<char*, 3>& weight_destinations,
    const std::array<size_t, 3>& weight_destination_nbytes) const {
#ifdef _WIN32
  throw std::runtime_error(
      "[ScaleXModeADirect] combined expert reads require POSIX preadv");
#else
  if (expert_id >= records_.size()) {
    throw std::out_of_range("[ScaleXModeADirect] expert id is out of range");
  }
  if (scale_destination_nbytes != decoded_tensor_nbytes_ ||
      std::any_of(
          scale_destinations.begin(), scale_destinations.end(), [](char* value) {
            return value == nullptr;
          }) ||
      std::any_of(
          weight_destinations.begin(), weight_destinations.end(), [](char* value) {
            return value == nullptr;
          }) ||
      std::any_of(
          weight_destination_nbytes.begin(),
          weight_destination_nbytes.end(),
          [](size_t value) { return value == 0; })) {
    throw std::invalid_argument(
        "[ScaleXModeADirect] combined destination tensor layout mismatch");
  }
  const auto& record = records_[expert_id];
  const size_t weight_bytes = std::accumulate(
      weight_destination_nbytes.begin(),
      weight_destination_nbytes.end(),
      size_t{0});
  if (record.absolute_offset > file_nbytes_ ||
      record.encoded_nbytes > file_nbytes_ - record.absolute_offset ||
      weight_bytes >
          file_nbytes_ - record.absolute_offset - record.encoded_nbytes) {
    throw std::runtime_error(
        "[ScaleXModeADirect] combined expert range exceeds file bounds");
  }

  auto& encoded = scalex_encoded_scratch(record.encoded_nbytes);
  std::array<struct iovec, 4> vectors{};
  vectors[0].iov_base = encoded.data();
  vectors[0].iov_len = encoded.size();
  for (size_t tensor = 0; tensor < weight_destinations.size(); ++tensor) {
    vectors[tensor + 1].iov_base = weight_destinations[tensor];
    vectors[tensor + 1].iov_len = weight_destination_nbytes[tensor];
  }
  const size_t total_bytes = record.encoded_nbytes + weight_bytes;
  ssize_t result;
  do {
    result = ::preadv(
        fd_,
        vectors.data(),
        static_cast<int>(vectors.size()),
        static_cast<off_t>(record.absolute_offset));
  } while (result < 0 && errno == EINTR);
  if (result != static_cast<ssize_t>(total_bytes)) {
    pread_exact(
        fd_,
        reinterpret_cast<char*>(encoded.data()),
        encoded.size(),
        record.absolute_offset);
    size_t offset = record.absolute_offset + record.encoded_nbytes;
    for (size_t tensor = 0; tensor < weight_destinations.size(); ++tensor) {
      pread_exact(
          fd_,
          weight_destinations[tensor],
          weight_destination_nbytes[tensor],
          offset);
      offset += weight_destination_nbytes[tensor];
    }
  }
  decode_scalex_mode_a_into(
      encoded, scale_destinations, scale_destination_nbytes);
#endif
}

SafetensorsRowDirect::SafetensorsRowDirect(
    std::string file,
    Dtype dtype,
    int rows,
    int columns,
    size_t absolute_offset)
    : file_(std::move(file)),
      dtype_(dtype),
      rows_(rows),
      columns_(columns),
      absolute_offset_(absolute_offset) {
#ifdef _WIN32
  throw std::runtime_error(
      "[SafetensorsRowDirect] official-direct rows currently require POSIX pread");
#else
  if (rows_ <= 0 || columns_ <= 0) {
    throw std::invalid_argument(
        "[SafetensorsRowDirect] row tensor dimensions must be positive");
  }
  row_nbytes_ = static_cast<size_t>(columns_) * dtype_.size();
  fd_ = ::open(file_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error(
        "[SafetensorsRowDirect] failed to open file " + file_);
  }
  struct stat info {};
  if (::fstat(fd_, &info) != 0 || info.st_size < 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[SafetensorsRowDirect] failed to stat file " + file_);
  }
  const auto file_nbytes = static_cast<size_t>(info.st_size);
  const auto tensor_nbytes = static_cast<size_t>(rows_) * row_nbytes_;
  if (absolute_offset_ > file_nbytes ||
      tensor_nbytes > file_nbytes - absolute_offset_) {
    ::close(fd_);
    fd_ = -1;
    throw std::invalid_argument(
        "[SafetensorsRowDirect] tensor exceeds file bounds");
  }
#endif
}

SafetensorsRowDirect::~SafetensorsRowDirect() {
#ifndef _WIN32
  if (fd_ >= 0) {
    ::close(fd_);
  }
#endif
}

void SafetensorsRowDirect::load_rows_into(
    const std::vector<size_t>& row_ids,
    char* destination,
    size_t destination_nbytes) const {
#ifdef _WIN32
  throw std::runtime_error(
      "[SafetensorsRowDirect] official-direct rows currently require POSIX pread");
#else
  if (destination == nullptr || row_ids.empty() ||
      destination_nbytes != row_ids.size() * row_nbytes_) {
    throw std::invalid_argument(
        "[SafetensorsRowDirect] invalid destination row buffer");
  }
  for (size_t item = 0; item < row_ids.size(); ++item) {
    const auto row = row_ids[item];
    if (row >= static_cast<size_t>(rows_)) {
      throw std::out_of_range(
          "[SafetensorsRowDirect] source row is out of range");
    }
    pread_exact(
        fd_,
        destination + item * row_nbytes_,
        row_nbytes_,
        absolute_offset_ + row * row_nbytes_);
  }
#endif
}

size_t ExpertSafetensorsDirect::read_range_count(size_t expert_id) const {
  if (expert_id >= ranges_by_expert_.size()) {
    throw std::out_of_range(
        "[ExpertSafetensorsDirect] expert id is out of range");
  }
  return ranges_by_expert_[expert_id].size();
}

size_t ExpertSafetensorsDirect::advise_read(size_t expert_id) const {
  if (expert_id >= ranges_by_expert_.size()) {
    throw std::out_of_range(
        "[ExpertSafetensorsDirect] expert id is out of range");
  }
#ifdef __APPLE__
  size_t advised = 0;
  for (const auto& range : ranges_by_expert_[expert_id]) {
    if (range.byte_length > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "[ExpertSafetensorsDirect] advisory range exceeds macOS limit");
    }
    struct radvisory advice {
      static_cast<off_t>(range.absolute_offset),
          static_cast<int>(range.byte_length)
    };
    if (::fcntl(fd_, F_RDADVISE, &advice) != 0) {
      throw std::runtime_error(
          "[ExpertSafetensorsDirect] macOS asynchronous read advice failed");
    }
    advised += range.byte_length;
  }
  return advised;
#else
  throw std::runtime_error(
      "[ExpertSafetensorsDirect] asynchronous read advice requires macOS");
#endif
}

void ExpertSafetensorsDirect::load_ordered_into(
    size_t expert_id,
    const std::vector<char*>& destinations,
    const std::vector<size_t>& destination_nbytes) const {
#ifdef _WIN32
  throw std::runtime_error(
      "[ExpertSafetensorsDirect] official-direct IO currently requires POSIX preadv");
#else
  if (expert_id >= specs_by_expert_.size()) {
    throw std::out_of_range(
        "[ExpertSafetensorsDirect] expert id is out of range");
  }
  const auto& specs = specs_by_expert_[expert_id];
  if (destinations.size() != specs.size() ||
      destination_nbytes.size() != specs.size()) {
    throw std::invalid_argument(
        "[ExpertSafetensorsDirect] destination count does not match tensor specs");
  }
  std::vector<size_t> sizes(specs.size());
  for (size_t index = 0; index < specs.size(); ++index) {
    sizes[index] = tensor_nbytes(specs[index]);
    if (destinations[index] == nullptr ||
        destination_nbytes[index] != sizes[index]) {
      std::ostringstream message;
      message << "[ExpertSafetensorsDirect] invalid destination for tensor '"
              << specs[index].name << "'";
      throw std::invalid_argument(message.str());
    }
  }

  for (const auto& range : ranges_by_expert_[expert_id]) {
    std::vector<struct iovec> vectors(range.tensor_indices.size());
    for (size_t position = 0; position < range.tensor_indices.size(); ++position) {
      const auto tensor_index = range.tensor_indices[position];
      vectors[position].iov_base = destinations[tensor_index];
      vectors[position].iov_len = sizes[tensor_index];
    }
    ssize_t result;
    do {
      result = ::preadv(
          fd_,
          vectors.data(),
          static_cast<int>(vectors.size()),
          static_cast<off_t>(range.absolute_offset));
    } while (result < 0 && errno == EINTR);
    if (result == static_cast<ssize_t>(range.byte_length)) {
      continue;
    }
    // A short scatter read is rare, but exact positioned reads make recovery
    // deterministic and overwrite any partially filled destinations.
    for (auto tensor_index : range.tensor_indices) {
      pread_exact(
          fd_,
          destinations[tensor_index],
          sizes[tensor_index],
          specs[tensor_index].absolute_offset);
    }
  }
#endif
}

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else if (str == ST_F8_E4M3) {
    return uint8;
  } else if (str == ST_F8_E8M0) {
    return uint8;
  } else {
    std::ostringstream msg;
    msg << "[safetensor] unsupported dtype" << str;
    throw std::runtime_error(msg.str());
  }
}

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    std::ostringstream msg;
    msg << "[load_safetensors] Failed to open " << in_stream->label();
    throw std::runtime_error(msg.str());
  }

  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);

  uint64_t jsonHeaderLength = 0;
  // This is the same limit as in the original Rust Safetensors code.
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    std::ostringstream msg;
    msg << "[load_safetensors] Invalid json header length "
        << in_stream->label();
    throw std::runtime_error(msg.str());
  }

  // Determine file size to be able to validate that reads are within the
  // bounds of the file (at least at creation time)
  in_stream->seek(0, std::ios_base::end);
  size_t file_size = in_stream->tell();
  in_stream->seek(8, std::ios_base::beg);

  // Load the json metadata
  if (file_size < jsonHeaderLength + 8) {
    std::ostringstream msg;
    msg << "[load_safetensors] The JSON header is " << jsonHeaderLength
        << " bytes long but the file is only " << file_size << " bytes. "
        << "Perhaps an incomplete download or corrupt file?";
    throw std::runtime_error(msg.str());
  }
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream->read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    std::ostringstream msg;
    msg << "[load_safetensors] Invalid json metadata " << in_stream->label();
    throw std::runtime_error(msg.str());
  }
  size_t offset = jsonHeaderLength + 8;

  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    if (data_offsets.size() != 2) {
      std::ostringstream msg;
      msg << "[load_safetensors] Tensor '" << item.key()
          << "' data_offsets must have exactly 2 entries but has "
          << data_offsets.size();
      throw std::runtime_error(msg.str());
    }
    {
      size_t expected_nbytes = type.size();
      for (auto dim : shape) {
        expected_nbytes *= static_cast<size_t>(dim);
      }
      if (data_offsets[1] < data_offsets[0] ||
          data_offsets[1] - data_offsets[0] != expected_nbytes) {
        std::ostringstream msg;
        msg << "[load_safetensors] Tensor '" << item.key()
            << "' invalid data offsets (" << data_offsets[0] << ", "
            << data_offsets[1] << "). Expecting " << expected_nbytes
            << " bytes.";
        throw std::runtime_error(msg.str());
      }
    }
    if (offset + data_offsets[1] > file_size) {
      std::ostringstream msg;
      msg << "[load_safetensors] Tensor '" << item.key()
          << "' invalid data offsets (" << data_offsets[0] << ", "
          << data_offsets[1] << ") exceeding the size of the file. "
          << "Perhaps an incomplete download or corrupt file?";
      throw std::runtime_error(msg.str());
    }
    res.insert(
        {item.key(),
         array(
             shape,
             type,
             std::make_shared<Load>(
                 stream, in_stream, offset + data_offsets.at(0), false),
             std::vector<array>{})});
  }
  return {res, metadata_map};
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
  return load_safetensors(std::make_shared<io::ParallelFileReader>(file), s);
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    std::ostringstream msg;
    msg << "[save_safetensors] Failed to open " << out_stream->label();
    throw std::runtime_error(msg.str());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    if (arr.nbytes() == 0) {
      std::ostringstream msg;
      msg << "[save_safetensors] Cannot serialize an empty array ('" << key
          << "')";
      throw std::invalid_argument(msg.str());
    }

    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    out_stream->write(arr.data<char>(), arr.nbytes());
  }
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)), a, metadata);
}

} // namespace mlx::core
