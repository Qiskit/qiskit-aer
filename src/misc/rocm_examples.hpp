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

/**
 * ROCm 7.x Feature Integration Example
 * 
 * This file demonstrates how to use the enhanced ROCm 7.x features in Qiskit Aer:
 * - Advanced memory management with unified memory
 * - HIP Graph execution for repeated circuits
 * - Wavefront-optimized kernels
 * - Multi-GPU coordination with XGMI
 */

#ifdef AER_THRUST_ROCM

#include "misc/rocm_memory_manager.hpp"
#include "misc/rocm_graph_executor.hpp"
#include "misc/rocm_wavefront_utils.hpp"
#include "misc/rocm_multi_gpu.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

namespace AER {
namespace ROCm {
namespace Examples {

//============================================================================
// Example 1: Enhanced Memory Management
//============================================================================

void example_memory_management() {
  std::cout << "\n=== Example 1: Enhanced Memory Management ===\n\n";
  
  // Create memory manager with pooling enabled
  auto mem_manager = std::make_shared<ROCmMemoryManager>(0, true, 2048);
  
  std::cout << "Memory Manager Capabilities:\n";
  std::cout << "  Managed Memory: " << (mem_manager->supports_managed_memory() ? "Yes" : "No") << "\n";
  std::cout << "  Fine-Grained Memory: " << (mem_manager->supports_fine_grained() ? "Yes" : "No") << "\n";
  std::cout << "  XNACK (Page Migration): " << (mem_manager->supports_xnack() ? "Yes" : "No") << "\n\n";
  
  // Allocate device memory (standard)
  size_t array_size = 1024 * 1024 * 16;  // 16 MB
  void* device_ptr = mem_manager->allocate(array_size, MemoryType::DEVICE);
  std::cout << "Allocated 16 MB device memory\n";
  
  // If managed memory is supported, allocate unified memory
  if (mem_manager->supports_managed_memory()) {
    void* managed_ptr = mem_manager->allocate(array_size, MemoryType::MANAGED);
    std::cout << "Allocated 16 MB managed memory\n";
    
    // Prefetch to GPU for better performance
    mem_manager->prefetch_to_gpu(managed_ptr, array_size);
    std::cout << "Prefetched managed memory to GPU\n";
    
    mem_manager->free(managed_ptr, array_size, MemoryType::MANAGED);
  }
  
  // Memory statistics
  std::cout << "\nMemory Statistics:\n";
  std::cout << "  Total Allocated: " << (mem_manager->get_total_allocated() / (1024*1024)) << " MB\n";
  std::cout << "  Current Usage: " << (mem_manager->get_current_usage() / (1024*1024)) << " MB\n";
  std::cout << "  Peak Usage: " << (mem_manager->get_peak_usage() / (1024*1024)) << " MB\n";
  std::cout << "  Pool Size: " << (mem_manager->get_pool_size() / (1024*1024)) << " MB\n";
  
  mem_manager->free(device_ptr, array_size);
}

//============================================================================
// Example 2: HIP Graph Execution (ROCm 7.0+)
//============================================================================

#ifdef AER_ROCM_7_PLUS

// Simple kernel for demonstration
__global__ void vector_add_kernel(float* c, const float* a, const float* b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void example_hip_graph() {
  std::cout << "\n=== Example 2: HIP Graph Execution ===\n\n";
  
  const int n = 1024 * 1024;
  const size_t bytes = n * sizeof(float);
  
  // Allocate memory
  float *d_a, *d_b, *d_c;
  hipMalloc(&d_a, bytes);
  hipMalloc(&d_b, bytes);
  hipMalloc(&d_c, bytes);
  
  // Create graph executor
  HIPGraphExecutor graph_executor(0);
  
  // Start capturing operations
  if (graph_executor.begin_capture()) {
    std::cout << "Started graph capture\n";
    
    // Launch kernels that will be captured
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    for (int i = 0; i < 10; i++) {
      hipLaunchKernelGGL(vector_add_kernel, dim3(blocks), dim3(threads), 0,
                         graph_executor.get_stream(), d_c, d_a, d_b, n);
    }
    
    // End capture and create graph
    if (graph_executor.end_capture()) {
      std::cout << "Graph captured and instantiated\n";
      
      // Launch graph multiple times (much faster than individual kernel launches)
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 100; i++) {
        graph_executor.launch_graph();
      }
      graph_executor.synchronize();
      auto end = std::chrono::high_resolution_clock::now();
      
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << "Launched graph 100 times in " << duration << " ms\n";
      std::cout << "Average: " << (duration / 100.0) << " ms per graph launch\n";
    }
  }
  
  // Cleanup
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
}

#endif

//============================================================================
// Example 3: Wavefront-Optimized Kernel
//============================================================================

// Example: Parallel reduction using wavefront primitives
template <typename T>
__global__ void wavefront_reduction_kernel(T* output, const T* input, int n) {
  extern __shared__ T shared_data[];
  
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Load data
  T value = (gid < n) ? input[gid] : 0;
  
  // Wavefront-level reduction (very fast)
  value = AER::ROCm::wavefront_reduce_sum(value);
  
  // First thread of each wavefront writes to shared memory
  int lane_id = AER::ROCm::get_lane_id();
  int wavefront_id = tid / AER::ROCm::get_wavefront_size();
  
  if (lane_id == 0) {
    shared_data[wavefront_id] = value;
  }
  __syncthreads();
  
  // Final reduction across wavefronts
  if (wavefront_id == 0) {
    int num_wavefronts = (blockDim.x + AER::ROCm::get_wavefront_size() - 1) / 
                         AER::ROCm::get_wavefront_size();
    value = (tid < num_wavefronts) ? shared_data[tid] : 0;
    value = AER::ROCm::wavefront_reduce_sum(value);
    
    if (tid == 0) {
      output[blockIdx.x] = value;
    }
  }
}

void example_wavefront_kernels() {
  std::cout << "\n=== Example 3: Wavefront-Optimized Kernels ===\n\n";
  
  const int n = 1024 * 1024;
  const size_t bytes = n * sizeof(float);
  
  std::cout << "Wavefront Size: " << AER_AMD_WAVEFRONT_SIZE << "\n";
  
#ifdef AER_AMD_ARCH_CDNA
  std::cout << "Architecture: CDNA (MI100/MI200/MI300)\n";
  std::cout << "Optimal Block Size: 256 (4 wavefronts)\n";
#elif defined(AER_AMD_ARCH_RDNA)
  std::cout << "Architecture: RDNA (RX 6000/7000)\n";
  std::cout << "Optimal Block Size: 128 (4 wavefronts)\n";
#endif
  
  // Allocate memory
  float *d_input, *d_output;
  hipMalloc(&d_input, bytes);
  
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipMalloc(&d_output, blocks * sizeof(float));
  
  // Launch wavefront-optimized reduction
  int shared_mem = (threads / AER_AMD_WAVEFRONT_SIZE) * sizeof(float);
  
  auto start = std::chrono::high_resolution_clock::now();
  hipLaunchKernelGGL(wavefront_reduction_kernel<float>, dim3(blocks), dim3(threads),
                     shared_mem, 0, d_output, d_input, n);
  hipDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Reduction completed in " << duration << " microseconds\n";
  
  // Cleanup
  hipFree(d_input);
  hipFree(d_output);
}

//============================================================================
// Example 4: Multi-GPU Coordination with XGMI
//============================================================================

void example_multi_gpu() {
  std::cout << "\n=== Example 4: Multi-GPU Coordination ===\n";
  
  MultiGPUCoordinator coordinator;
  
  if (!coordinator.initialize()) {
    std::cout << "No GPUs found or initialization failed\n";
    return;
  }
  
  // Print topology
  coordinator.print_topology();
  
  // Get optimal GPU pairs for distributed work
  auto pairs = coordinator.get_optimal_gpu_pairs();
  
  if (!pairs.empty()) {
    std::cout << "Optimal GPU Pairing for Distributed Work:\n";
    for (const auto& pair : pairs) {
      std::cout << "  GPU" << pair.first << " <-> GPU" << pair.second;
      
      const auto& topology = coordinator.get_topology();
      if (topology.xgmi_matrix[pair.first][pair.second]) {
        std::cout << " (XGMI, " << topology.xgmi_hops[pair.first][pair.second] << " hops)";
      } else {
        std::cout << " (PCIe)";
      }
      std::cout << "\n";
    }
  }
  
  // Distribute work based on GPU capabilities
  size_t total_qubits = 1000000;
  auto distribution = coordinator.distribute_work(total_qubits);
  
  std::cout << "\nWork Distribution (for " << total_qubits << " items):\n";
  for (size_t i = 0; i < distribution.size(); i++) {
    std::cout << "  GPU" << i << ": " << distribution[i] << " items\n";
  }
  
  // Example: XGMI-aware data transfer
  if (coordinator.get_num_devices() >= 2) {
    XGMITransferManager transfer_manager(coordinator);
    
    const size_t transfer_size = 1024 * 1024 * 1024;  // 1 GB
    double transfer_time_ns = transfer_manager.estimate_transfer_time_ns(0, 1, transfer_size);
    
    std::cout << "\nEstimated transfer time for 1 GB (GPU0 -> GPU1): "
              << (transfer_time_ns / 1e6) << " ms\n";
    
    const auto& topology = coordinator.get_topology();
    if (topology.xgmi_matrix[0][1]) {
      double bandwidth_gbps = topology.gpus[0].xgmi_bandwidth_gbps;
      std::cout << "XGMI Bandwidth: " << bandwidth_gbps << " GB/s\n";
    }
  }
}

//============================================================================
// Run all examples
//============================================================================

void run_all_examples() {
  std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
  std::cout << "║      Qiskit Aer - ROCm 7.x Feature Examples           ║\n";
  std::cout << "╚════════════════════════════════════════════════════════╝\n";
  
  // Example 1: Memory Management
  example_memory_management();
  
  // Example 2: HIP Graph (ROCm 7.0+ only)
#ifdef AER_ROCM_7_PLUS
  example_hip_graph();
#else
  std::cout << "\n=== Example 2: HIP Graph Execution ===\n";
  std::cout << "HIP Graph requires ROCm 7.0+. Current version: " << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "\n";
#endif
  
  // Example 3: Wavefront Kernels
  example_wavefront_kernels();
  
  // Example 4: Multi-GPU
  example_multi_gpu();
  
  std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
  std::cout << "║              All examples completed!                   ║\n";
  std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
}

} // namespace Examples
} // namespace ROCm
} // namespace AER

#endif // AER_THRUST_ROCM
