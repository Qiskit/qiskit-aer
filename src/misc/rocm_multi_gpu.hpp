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

#ifndef _aer_rocm_multi_gpu_hpp_
#define _aer_rocm_multi_gpu_hpp_

#ifdef AER_THRUST_ROCM

#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <algorithm>

namespace AER {
namespace ROCm {

//============================================================================
// GPU Topology Information
// Describes the connectivity and capabilities of AMD GPUs in the system
//============================================================================

struct GPUInfo {
  int device_id;
  std::string name;
  size_t total_memory;
  size_t free_memory;
  int compute_capability;  // GCN arch number (e.g., 908, 90a, 940)
  int pcie_bus_id;
  int pcie_device_id;
  bool supports_peer_access;
  bool supports_xgmi;  // AMD Infinity Fabric
  int xgmi_links;      // Number of XGMI links
  double xgmi_bandwidth_gbps;  // Peak bandwidth per link
};

struct GPUTopology {
  std::vector<GPUInfo> gpus;
  std::vector<std::vector<bool>> peer_access_matrix;
  std::vector<std::vector<bool>> xgmi_matrix;
  std::vector<std::vector<int>> xgmi_hops;  // Number of hops between GPUs
};

//============================================================================
// Multi-GPU Coordinator for AMD GPUs
// Optimized for AMD Infinity Fabric (XGMI) on MI200/MI300
//============================================================================

class MultiGPUCoordinator {
private:
  int num_devices_;
  std::vector<int> device_ids_;
  GPUTopology topology_;
  bool initialized_;
  
  // Detect GPU topology
  void detect_topology() {
    hipGetDeviceCount(&num_devices_);
    
    topology_.gpus.resize(num_devices_);
    topology_.peer_access_matrix.resize(num_devices_, std::vector<bool>(num_devices_, false));
    topology_.xgmi_matrix.resize(num_devices_, std::vector<bool>(num_devices_, false));
    topology_.xgmi_hops.resize(num_devices_, std::vector<int>(num_devices_, -1));
    
    for (int i = 0; i < num_devices_; i++) {
      hipDeviceProp_t prop;
      hipGetDeviceProperties(&prop, i);
      
      GPUInfo& info = topology_.gpus[i];
      info.device_id = i;
      info.name = prop.name;
      info.total_memory = prop.totalGlobalMem;
      info.compute_capability = prop.gcnArch;
      info.pcie_bus_id = prop.pciBusID;
      info.pcie_device_id = prop.pciDeviceID;
      
      // Get free memory
      hipSetDevice(i);
      hipMemGetInfo(&info.free_memory, &info.total_memory);
      
      // Check peer access and XGMI
      for (int j = 0; j < num_devices_; j++) {
        if (i == j) {
          topology_.peer_access_matrix[i][j] = true;
          topology_.xgmi_hops[i][j] = 0;
          continue;
        }
        
        int can_access = 0;
        hipDeviceCanAccessPeer(&can_access, i, j);
        topology_.peer_access_matrix[i][j] = (can_access != 0);
        
        // Detect XGMI links (MI200/MI300)
        // XGMI provides much higher bandwidth than PCIe
#ifdef AER_ROCM_6_PLUS
        uint32_t link_type = 0;
        hipDeviceGetP2PAttribute(&link_type, hipDevP2PAttrLinkType, i, j);
        
        if (link_type == 2) {  // XGMI link
          topology_.xgmi_matrix[i][j] = true;
          
          // Get number of hops
          uint32_t hops = 0;
          hipDeviceGetP2PAttribute(&hops, hipDevP2PAttrHdpMemFlushCntl, i, j);
          topology_.xgmi_hops[i][j] = hops;
          
          // Estimate bandwidth (simplified)
          if (prop.gcnArch >= 940) {
            // MI300: Up to 896 GB/s per link
            info.xgmi_bandwidth_gbps = 896.0;
          } else if (prop.gcnArch == 0x90a) {
            // MI250/MI210: Up to 200 GB/s per link
            info.xgmi_bandwidth_gbps = 200.0;
          } else {
            info.xgmi_bandwidth_gbps = 50.0;  // Default
          }
        } else {
          topology_.xgmi_hops[i][j] = 10;  // PCIe, high hop count
        }
#endif
      }
    }
  }
  
public:
  MultiGPUCoordinator() : num_devices_(0), initialized_(false) {
    initialize();
  }
  
  bool initialize() {
    if (initialized_) return true;
    
    hipGetDeviceCount(&num_devices_);
    if (num_devices_ <= 0) {
      return false;
    }
    
    device_ids_.resize(num_devices_);
    for (int i = 0; i < num_devices_; i++) {
      device_ids_[i] = i;
    }
    
    detect_topology();
    enable_peer_access();
    
    initialized_ = true;
    return true;
  }
  
  // Enable peer access between all compatible GPUs
  void enable_peer_access() {
    for (int i = 0; i < num_devices_; i++) {
      hipSetDevice(i);
      for (int j = 0; j < num_devices_; j++) {
        if (i != j && topology_.peer_access_matrix[i][j]) {
          // Enable peer access (ignore errors if already enabled)
          hipDeviceEnablePeerAccess(j, 0);
        }
      }
    }
  }
  
  // Get number of GPUs
  int get_num_devices() const { return num_devices_; }
  
  // Get topology information
  const GPUTopology& get_topology() const { return topology_; }
  
  // Check if two GPUs are connected via XGMI
  bool has_xgmi_link(int device1, int device2) const {
    if (device1 < 0 || device1 >= num_devices_ ||
        device2 < 0 || device2 >= num_devices_) {
      return false;
    }
    return topology_.xgmi_matrix[device1][device2];
  }
  
  // Get optimal GPU pairing for distributed computation
  // Returns pairs of GPUs with best connectivity (prefer XGMI)
  std::vector<std::pair<int, int>> get_optimal_gpu_pairs() const {
    std::vector<std::pair<int, int>> pairs;
    std::vector<bool> paired(num_devices_, false);
    
    // First, pair GPUs with XGMI links
    for (int i = 0; i < num_devices_; i++) {
      if (paired[i]) continue;
      
      int best_partner = -1;
      int min_hops = 999;
      
      for (int j = i + 1; j < num_devices_; j++) {
        if (paired[j]) continue;
        
        if (topology_.xgmi_matrix[i][j]) {
          int hops = topology_.xgmi_hops[i][j];
          if (hops < min_hops) {
            min_hops = hops;
            best_partner = j;
          }
        }
      }
      
      if (best_partner != -1) {
        pairs.push_back({i, best_partner});
        paired[i] = true;
        paired[best_partner] = true;
      }
    }
    
    // Then pair remaining GPUs via PCIe
    for (int i = 0; i < num_devices_; i++) {
      if (paired[i]) continue;
      
      for (int j = i + 1; j < num_devices_; j++) {
        if (paired[j]) continue;
        
        if (topology_.peer_access_matrix[i][j]) {
          pairs.push_back({i, j});
          paired[i] = true;
          paired[j] = true;
          break;
        }
      }
    }
    
    return pairs;
  }
  
  // Distribute work across GPUs based on their capabilities
  std::vector<size_t> distribute_work(size_t total_work) const {
    std::vector<size_t> distribution(num_devices_);
    
    // Calculate work distribution based on available memory
    size_t total_memory = 0;
    for (const auto& gpu : topology_.gpus) {
      total_memory += gpu.free_memory;
    }
    
    size_t assigned_work = 0;
    for (int i = 0; i < num_devices_; i++) {
      if (i == num_devices_ - 1) {
        // Last GPU gets remainder
        distribution[i] = total_work - assigned_work;
      } else {
        double fraction = static_cast<double>(topology_.gpus[i].free_memory) / total_memory;
        distribution[i] = static_cast<size_t>(total_work * fraction);
        assigned_work += distribution[i];
      }
    }
    
    return distribution;
  }
  
  // Synchronize all devices
  void synchronize_all() {
    for (int i = 0; i < num_devices_; i++) {
      hipSetDevice(i);
      hipDeviceSynchronize();
    }
  }
  
  // Print topology information
  void print_topology() const {
    std::cout << "\n=== AMD GPU Topology ===\n";
    std::cout << "Number of GPUs: " << num_devices_ << "\n\n";
    
    for (int i = 0; i < num_devices_; i++) {
      const auto& gpu = topology_.gpus[i];
      std::cout << "GPU " << i << ": " << gpu.name << "\n";
      std::cout << "  Memory: " << (gpu.total_memory / (1024*1024*1024)) << " GB\n";
      std::cout << "  Free: " << (gpu.free_memory / (1024*1024*1024)) << " GB\n";
      std::cout << "  GCN Arch: gfx" << std::hex << gpu.compute_capability << std::dec << "\n";
      
      // Show connectivity
      std::cout << "  Connected to: ";
      for (int j = 0; j < num_devices_; j++) {
        if (i != j && topology_.xgmi_matrix[i][j]) {
          std::cout << "GPU" << j << "(XGMI-" << topology_.xgmi_hops[i][j] << ") ";
        } else if (i != j && topology_.peer_access_matrix[i][j]) {
          std::cout << "GPU" << j << "(PCIe) ";
        }
      }
      std::cout << "\n\n";
    }
    
    // Show XGMI connectivity matrix
    if (num_devices_ > 1) {
      std::cout << "XGMI Connectivity Matrix:\n";
      std::cout << "     ";
      for (int i = 0; i < num_devices_; i++) {
        std::cout << "GPU" << i << "  ";
      }
      std::cout << "\n";
      
      for (int i = 0; i < num_devices_; i++) {
        std::cout << "GPU" << i << " ";
        for (int j = 0; j < num_devices_; j++) {
          if (i == j) {
            std::cout << " --   ";
          } else if (topology_.xgmi_matrix[i][j]) {
            std::cout << " YES  ";
          } else {
            std::cout << " NO   ";
          }
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }
};

//============================================================================
// XGMI-Aware Data Transfer Manager
// Optimizes data transfers using AMD Infinity Fabric when available
//============================================================================

class XGMITransferManager {
private:
  MultiGPUCoordinator& coordinator_;
  
public:
  XGMITransferManager(MultiGPUCoordinator& coordinator)
      : coordinator_(coordinator) {}
  
  // Transfer data between GPUs with optimal path
  bool transfer(void* dst, int dst_device, const void* src, int src_device,
                size_t size, hipStream_t stream = nullptr) {
    
    const auto& topology = coordinator_.get_topology();
    
    // Check if XGMI link exists
    if (topology.xgmi_matrix[src_device][dst_device]) {
      // Direct XGMI transfer (very fast)
      hipError_t err = hipMemcpyPeerAsync(dst, dst_device, src, src_device,
                                          size, stream);
      return (err == hipSuccess);
    } else if (topology.peer_access_matrix[src_device][dst_device]) {
      // PCIe peer transfer
      hipError_t err = hipMemcpyPeerAsync(dst, dst_device, src, src_device,
                                          size, stream);
      return (err == hipSuccess);
    } else {
      // No peer access, use host as intermediary
      void* host_buffer = malloc(size);
      if (!host_buffer) return false;
      
      hipSetDevice(src_device);
      hipMemcpy(host_buffer, src, size, hipMemcpyDeviceToHost);
      
      hipSetDevice(dst_device);
      hipMemcpy(dst, host_buffer, size, hipMemcpyHostToDevice);
      
      free(host_buffer);
      return true;
    }
  }
  
  // Get estimated transfer time (nanoseconds)
  double estimate_transfer_time_ns(int src_device, int dst_device, size_t size) {
    const auto& topology = coordinator_.get_topology();
    
    if (topology.xgmi_matrix[src_device][dst_device]) {
      // XGMI bandwidth
      double bandwidth_gbps = topology.gpus[src_device].xgmi_bandwidth_gbps;
      double size_gb = size / (1024.0 * 1024.0 * 1024.0);
      return (size_gb / bandwidth_gbps) * 1e9;  // Convert to nanoseconds
    } else {
      // PCIe Gen4 x16: ~32 GB/s bidirectional
      double pcie_bandwidth_gbps = 32.0;
      double size_gb = size / (1024.0 * 1024.0 * 1024.0);
      return (size_gb / pcie_bandwidth_gbps) * 1e9;
    }
  }
};

} // namespace ROCm
} // namespace AER

#endif // AER_THRUST_ROCM
#endif // _aer_rocm_multi_gpu_hpp_
