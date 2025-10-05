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

#ifndef _aer_rocm_memory_manager_hpp_
#define _aer_rocm_memory_manager_hpp_

#ifdef AER_THRUST_ROCM

#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace AER {
namespace ROCm {

//============================================================================
// ROCm 7.x Enhanced Memory Manager
// Provides unified memory support, memory pooling, and MI300 optimizations
//============================================================================

enum class MemoryType {
  DEVICE,          // Standard device memory (HBM/GDDR)
  MANAGED,         // Unified memory (ROCm 6.0+)
  FINE_GRAINED,    // Fine-grained memory for CPU-GPU sharing (MI300)
  COARSE_GRAINED,  // Coarse-grained memory for GPU-only (default)
};

struct MemoryAllocationInfo {
  void* ptr;
  size_t size;
  MemoryType type;
  int device_id;
  bool is_pooled;
};

class ROCmMemoryManager {
private:
  int device_id_;
  bool supports_managed_memory_;
  bool supports_fine_grained_;
  bool supports_xnack_;  // Page migration support
  
  // Memory pool for reusing allocations
  std::vector<MemoryAllocationInfo> memory_pool_;
  size_t pool_size_limit_;
  bool pooling_enabled_;
  
  // Statistics
  size_t total_allocated_;
  size_t total_freed_;
  size_t peak_usage_;
  size_t current_usage_;
  
  // Check device capabilities
  void detect_capabilities() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id_);
    
    // Check for unified memory support (ROCm 6.0+)
    supports_managed_memory_ = prop.managedMemory;
    
    // Check for fine-grained memory (MI300, MI250)
    // Fine-grained allows coherent CPU-GPU access
    supports_fine_grained_ = (prop.gcnArch >= 940); // gfx940+ (MI300)
    
    // Check for XNACK (page migration support)
    // XNACK enables automatic page migration for managed memory
#ifdef AER_ROCM_7_PLUS
    hipDeviceAttribute_t attr;
    hipDeviceGetAttribute(&attr, hipDeviceAttributePageableMemoryAccess, device_id_);
    supports_xnack_ = (attr != 0);
#else
    supports_xnack_ = false;
#endif
  }

public:
  ROCmMemoryManager(int device_id = 0, bool enable_pooling = true,
                    size_t pool_size_limit_mb = 1024)
      : device_id_(device_id),
        supports_managed_memory_(false),
        supports_fine_grained_(false),
        supports_xnack_(false),
        pool_size_limit_(pool_size_limit_mb * 1024 * 1024),
        pooling_enabled_(enable_pooling),
        total_allocated_(0),
        total_freed_(0),
        peak_usage_(0),
        current_usage_(0) {
    hipSetDevice(device_id_);
    detect_capabilities();
  }
  
  ~ROCmMemoryManager() {
    // Free all pooled memory
    clear_pool();
  }
  
  // Allocate memory with automatic type selection
  void* allocate(size_t size, MemoryType preferred_type = MemoryType::DEVICE) {
    hipSetDevice(device_id_);
    
    // Try to reuse from pool first
    if (pooling_enabled_) {
      void* pooled_ptr = try_allocate_from_pool(size, preferred_type);
      if (pooled_ptr != nullptr) {
        current_usage_ += size;
        if (current_usage_ > peak_usage_) {
          peak_usage_ = current_usage_;
        }
        return pooled_ptr;
      }
    }
    
    void* ptr = nullptr;
    hipError_t err = hipSuccess;
    
    // Allocate based on type and capabilities
    switch (preferred_type) {
      case MemoryType::MANAGED:
        if (supports_managed_memory_) {
#ifdef AER_ROCM_6_PLUS
          err = hipMallocManaged(&ptr, size);
          if (err == hipSuccess) {
            // Hint: primarily accessed by GPU
            hipMemAdvise(ptr, size, hipMemAdviseSetAccessedBy, device_id_);
            break;
          }
#endif
        }
        // Fallback to device memory if managed not supported
        [[fallthrough]];
        
      case MemoryType::FINE_GRAINED:
        if (supports_fine_grained_) {
#ifdef AER_ROCM_6_PLUS
          // Allocate fine-grained memory (coherent CPU-GPU access)
          err = hipExtMallocWithFlags(&ptr, size, hipDeviceMallocFinegrained);
          if (err == hipSuccess) {
            break;
          }
#endif
        }
        // Fallback to coarse-grained
        [[fallthrough]];
        
      case MemoryType::COARSE_GRAINED:
      case MemoryType::DEVICE:
      default:
        // Standard device memory (most common)
        err = hipMalloc(&ptr, size);
        break;
    }
    
    if (err != hipSuccess || ptr == nullptr) {
      throw std::runtime_error("ROCm memory allocation failed: " + 
                               std::string(hipGetErrorString(err)));
    }
    
    // Update statistics
    total_allocated_ += size;
    current_usage_ += size;
    if (current_usage_ > peak_usage_) {
      peak_usage_ = current_usage_;
    }
    
    return ptr;
  }
  
  // Free memory (return to pool if enabled)
  void free(void* ptr, size_t size, MemoryType type = MemoryType::DEVICE) {
    if (ptr == nullptr) return;
    
    hipSetDevice(device_id_);
    
    // Try to add to pool for reuse
    if (pooling_enabled_ && current_usage_ < pool_size_limit_) {
      MemoryAllocationInfo info{ptr, size, type, device_id_, true};
      memory_pool_.push_back(info);
      current_usage_ -= size;
      return;
    }
    
    // Actually free the memory
    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) {
      // Log but don't throw in destructor contexts
    }
    
    total_freed_ += size;
    current_usage_ -= size;
  }
  
  // Prefetch memory to GPU (ROCm 6.0+)
  void prefetch_to_gpu(void* ptr, size_t size) {
#ifdef AER_ROCM_6_PLUS
    if (supports_managed_memory_ && supports_xnack_) {
      hipSetDevice(device_id_);
      hipMemPrefetchAsync(ptr, size, device_id_, nullptr);
    }
#endif
  }
  
  // Prefetch memory to CPU (ROCm 6.0+)
  void prefetch_to_cpu(void* ptr, size_t size) {
#ifdef AER_ROCM_6_PLUS
    if (supports_managed_memory_ && supports_xnack_) {
      hipMemPrefetchAsync(ptr, size, hipCpuDeviceId, nullptr);
    }
#endif
  }
  
  // Advise memory access patterns (ROCm 6.0+)
  void advise_read_mostly(void* ptr, size_t size) {
#ifdef AER_ROCM_6_PLUS
    if (supports_managed_memory_) {
      hipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, device_id_);
    }
#endif
  }
  
  void advise_preferred_location(void* ptr, size_t size, int preferred_device) {
#ifdef AER_ROCM_6_PLUS
    if (supports_managed_memory_) {
      hipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, preferred_device);
    }
#endif
  }
  
  // Clear memory pool
  void clear_pool() {
    hipSetDevice(device_id_);
    for (auto& info : memory_pool_) {
      hipFree(info.ptr);
      total_freed_ += info.size;
    }
    memory_pool_.clear();
  }
  
  // Get statistics
  size_t get_total_allocated() const { return total_allocated_; }
  size_t get_total_freed() const { return total_freed_; }
  size_t get_peak_usage() const { return peak_usage_; }
  size_t get_current_usage() const { return current_usage_; }
  size_t get_pool_size() const {
    size_t total = 0;
    for (const auto& info : memory_pool_) {
      total += info.size;
    }
    return total;
  }
  
  // Capability checks
  bool supports_managed_memory() const { return supports_managed_memory_; }
  bool supports_fine_grained() const { return supports_fine_grained_; }
  bool supports_xnack() const { return supports_xnack_; }
  
  // Enable/disable pooling
  void enable_pooling(bool enable) { pooling_enabled_ = enable; }
  bool is_pooling_enabled() const { return pooling_enabled_; }

private:
  // Try to allocate from pool
  void* try_allocate_from_pool(size_t size, MemoryType type) {
    for (auto it = memory_pool_.begin(); it != memory_pool_.end(); ++it) {
      if (it->size >= size && it->type == type) {
        void* ptr = it->ptr;
        memory_pool_.erase(it);
        return ptr;
      }
    }
    return nullptr;
  }
};

//============================================================================
// Memory Pool Allocator for Thrust
// Integrates with Thrust for efficient allocation
//============================================================================
template <typename T>
class ROCmPoolAllocator {
private:
  std::shared_ptr<ROCmMemoryManager> manager_;
  
public:
  using value_type = T;
  
  ROCmPoolAllocator(std::shared_ptr<ROCmMemoryManager> manager)
      : manager_(manager) {}
  
  T* allocate(size_t n) {
    return static_cast<T*>(manager_->allocate(n * sizeof(T)));
  }
  
  void deallocate(T* ptr, size_t n) {
    manager_->free(ptr, n * sizeof(T));
  }
};

} // namespace ROCm
} // namespace AER

#endif // AER_THRUST_ROCM
#endif // _aer_rocm_memory_manager_hpp_
