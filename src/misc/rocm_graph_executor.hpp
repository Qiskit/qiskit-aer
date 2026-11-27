/**
 * This code is part of Qiskit.
 *
 * (C) Copyright AMD 2025.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_rocm_graph_executor_hpp_
#define _aer_rocm_graph_executor_hpp_

#ifdef AER_THRUST_ROCM

#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <unordered_map>

namespace AER {
namespace ROCm {

//============================================================================
// HIP Graph Executor (ROCm 7.0+)
// Captures and replays GPU kernel sequences for better performance
// Particularly beneficial for repeated quantum circuit executions
//============================================================================

#ifdef AER_ROCM_7_PLUS

class HIPGraphExecutor {
private:
  int device_id_;
  hipStream_t stream_;
  hipGraph_t graph_;
  hipGraphExec_t graph_exec_;
  bool graph_created_;
  bool graph_instantiated_;
  bool is_capturing_;
  
  // Statistics
  uint64_t graph_launches_;
  uint64_t total_kernel_calls_;
  
public:
  HIPGraphExecutor(int device_id = 0, hipStream_t stream = nullptr)
      : device_id_(device_id),
        stream_(stream),
        graph_(nullptr),
        graph_exec_(nullptr),
        graph_created_(false),
        graph_instantiated_(false),
        is_capturing_(false),
        graph_launches_(0),
        total_kernel_calls_(0) {
    
    hipSetDevice(device_id_);
    
    // Create stream if not provided
    if (stream_ == nullptr) {
      hipStreamCreate(&stream_);
    }
  }
  
  ~HIPGraphExecutor() {
    destroy_graph();
    if (stream_ != nullptr) {
      hipStreamDestroy(stream_);
    }
  }
  
  // Start capturing operations into a graph
  bool begin_capture() {
    if (is_capturing_) {
      return false; // Already capturing
    }
    
    hipSetDevice(device_id_);
    
    // Start stream capture
    hipError_t err = hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) {
      return false;
    }
    
    is_capturing_ = true;
    return true;
  }
  
  // End capture and create graph
  bool end_capture() {
    if (!is_capturing_) {
      return false; // Not capturing
    }
    
    hipSetDevice(device_id_);
    
    // End stream capture
    hipError_t err = hipStreamEndCapture(stream_, &graph_);
    if (err != hipSuccess) {
      is_capturing_ = false;
      return false;
    }
    
    graph_created_ = true;
    is_capturing_ = false;
    
    // Instantiate the graph
    return instantiate_graph();
  }
  
  // Instantiate graph for execution
  bool instantiate_graph() {
    if (!graph_created_ || graph_instantiated_) {
      return false;
    }
    
    hipSetDevice(device_id_);
    
    // Instantiate graph
    hipError_t err = hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    if (err != hipSuccess) {
      return false;
    }
    
    graph_instantiated_ = true;
    return true;
  }
  
  // Launch the captured graph
  bool launch_graph() {
    if (!graph_instantiated_) {
      return false;
    }
    
    hipSetDevice(device_id_);
    
    // Launch graph
    hipError_t err = hipGraphLaunch(graph_exec_, stream_);
    if (err != hipSuccess) {
      return false;
    }
    
    graph_launches_++;
    return true;
  }
  
  // Synchronize after graph launch
  bool synchronize() {
    hipSetDevice(device_id_);
    hipError_t err = hipStreamSynchronize(stream_);
    return (err == hipSuccess);
  }
  
  // Destroy graph and free resources
  void destroy_graph() {
    hipSetDevice(device_id_);
    
    if (graph_instantiated_ && graph_exec_ != nullptr) {
      hipGraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
      graph_instantiated_ = false;
    }
    
    if (graph_created_ && graph_ != nullptr) {
      hipGraphDestroy(graph_);
      graph_ = nullptr;
      graph_created_ = false;
    }
  }
  
  // Reset for new capture
  void reset() {
    destroy_graph();
    graph_launches_ = 0;
  }
  
  // Check if capturing
  bool is_capturing() const { return is_capturing_; }
  bool is_graph_ready() const { return graph_instantiated_; }
  
  // Get statistics
  uint64_t get_graph_launches() const { return graph_launches_; }
  
  // Get stream for kernel launches during capture
  hipStream_t get_stream() const { return stream_; }
};

//============================================================================
// Graph Cache for repeated circuit patterns
// Caches graphs by circuit signature for reuse
//============================================================================
class HIPGraphCache {
private:
  int device_id_;
  std::unordered_map<std::string, std::shared_ptr<HIPGraphExecutor>> cache_;
  size_t max_cache_size_;
  uint64_t cache_hits_;
  uint64_t cache_misses_;
  
public:
  HIPGraphCache(int device_id = 0, size_t max_cache_size = 100)
      : device_id_(device_id),
        max_cache_size_(max_cache_size),
        cache_hits_(0),
        cache_misses_(0) {}
  
  // Get or create graph executor for a circuit signature
  std::shared_ptr<HIPGraphExecutor> get_or_create(const std::string& signature,
                                                   hipStream_t stream = nullptr) {
    auto it = cache_.find(signature);
    if (it != cache_.end()) {
      cache_hits_++;
      return it->second;
    }
    
    // Create new graph executor
    cache_misses_++;
    auto executor = std::make_shared<HIPGraphExecutor>(device_id_, stream);
    
    // Evict oldest if cache full
    if (cache_.size() >= max_cache_size_) {
      cache_.erase(cache_.begin());
    }
    
    cache_[signature] = executor;
    return executor;
  }
  
  // Clear cache
  void clear() {
    cache_.clear();
  }
  
  // Statistics
  size_t size() const { return cache_.size(); }
  uint64_t get_cache_hits() const { return cache_hits_; }
  uint64_t get_cache_misses() const { return cache_misses_; }
  double get_hit_rate() const {
    uint64_t total = cache_hits_ + cache_misses_;
    return total > 0 ? static_cast<double>(cache_hits_) / total : 0.0;
  }
};

#else // ROCm < 7.0 - Stub implementation

class HIPGraphExecutor {
public:
  HIPGraphExecutor(int device_id = 0, hipStream_t stream = nullptr) {}
  bool begin_capture() { return false; }
  bool end_capture() { return false; }
  bool launch_graph() { return false; }
  bool synchronize() { return false; }
  void reset() {}
  bool is_capturing() const { return false; }
  bool is_graph_ready() const { return false; }
  hipStream_t get_stream() const { return nullptr; }
};

class HIPGraphCache {
public:
  HIPGraphCache(int device_id = 0, size_t max_cache_size = 100) {}
  std::shared_ptr<HIPGraphExecutor> get_or_create(const std::string& signature,
                                                   hipStream_t stream = nullptr) {
    return nullptr;
  }
  void clear() {}
  size_t size() const { return 0; }
};

#endif // AER_ROCM_7_PLUS

} // namespace ROCm
} // namespace AER

#endif // AER_THRUST_ROCM
#endif // _aer_rocm_graph_executor_hpp_
