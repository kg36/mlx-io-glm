// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_map>
#include <variant>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/io/load.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {
using GGUFMetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>>;
using SafetensorsLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>>;

/** Byte-preserving description of one tensor in an official safetensors file. */
struct SafetensorsTensorSpec {
  std::string name;
  Dtype dtype;
  Shape shape;
  size_t absolute_offset;
};

/**
 * Direct-to-buffer reader for routed experts in an official checkpoint shard.
 *
 * Tensor layouts are supplied once for every expert. Adjacent source tensors
 * are coalesced into positioned scatter reads, while destination buffers stay
 * in the caller's canonical tensor order. No converted weight artifact or MLX
 * graph node is created on the load path.
 */
class MLX_API ExpertSafetensorsDirect {
 public:
  ExpertSafetensorsDirect(
      std::string file,
      std::vector<std::vector<SafetensorsTensorSpec>> specs_by_expert,
      bool no_cache = false,
      bool read_ahead = true);
  ~ExpertSafetensorsDirect();

  ExpertSafetensorsDirect(const ExpertSafetensorsDirect&) = delete;
  ExpertSafetensorsDirect& operator=(const ExpertSafetensorsDirect&) = delete;

  void load_ordered_into(
      size_t expert_id,
      const std::vector<char*>& destinations,
      const std::vector<size_t>& destination_nbytes) const;

  const std::vector<SafetensorsTensorSpec>& specs() const {
    return specs_by_expert_.front();
  }
  size_t num_experts() const {
    return specs_by_expert_.size();
  }
  size_t read_range_count(size_t expert_id) const;
  size_t advise_read(size_t expert_id) const;

 private:
  struct ReadRange {
    size_t absolute_offset;
    size_t byte_length;
    std::vector<size_t> tensor_indices;
  };

  std::string file_;
  size_t file_nbytes_{0};
  int fd_{-1};
  std::vector<std::vector<SafetensorsTensorSpec>> specs_by_expert_;
  std::vector<std::vector<ReadRange>> ranges_by_expert_;
};

/** Positioned row reader for a two-dimensional official safetensors tensor. */
class MLX_API SafetensorsRowDirect {
 public:
  SafetensorsRowDirect(
      std::string file,
      Dtype dtype,
      int rows,
      int columns,
      size_t absolute_offset);
  ~SafetensorsRowDirect();

  SafetensorsRowDirect(const SafetensorsRowDirect&) = delete;
  SafetensorsRowDirect& operator=(const SafetensorsRowDirect&) = delete;

  void load_rows_into(
      const std::vector<size_t>& row_ids,
      char* destination,
      size_t destination_nbytes) const;

  Dtype dtype() const {
    return dtype_;
  }
  int rows() const {
    return rows_;
  }
  int columns() const {
    return columns_;
  }
  size_t row_nbytes() const {
    return row_nbytes_;
  }

 private:
  std::string file_;
  Dtype dtype_;
  int rows_;
  int columns_;
  size_t absolute_offset_;
  size_t row_nbytes_;
  int fd_{-1};
};

/** Save array to out stream in .npy format */
MLX_API void save(std::shared_ptr<io::Writer> out_stream, array a);

/** Save array to file in .npy format */
MLX_API void save(std::string file, array a);

/** Load array from reader in .npy format */
MLX_API array
load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

/** Load array from file in .npy format */
MLX_API array load(std::string file, StreamOrDevice s = {});

/** Load array map from .safetensors file format */
MLX_API SafetensorsLoad
load_safetensors(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});
MLX_API SafetensorsLoad
load_safetensors(const std::string& file, StreamOrDevice s = {});

MLX_API void save_safetensors(
    std::shared_ptr<io::Writer> in_stream,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});
MLX_API void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});

/** Load array map and metadata from .gguf file format */

MLX_API GGUFLoad load_gguf(const std::string& file, StreamOrDevice s = {});

MLX_API void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, GGUFMetaData> meta_data = {});

} // namespace mlx::core
