/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _tensor_net_contractor_cuTensorNet_hpp_
#define _tensor_net_contractor_cuTensorNet_hpp_

#ifdef AER_THRUST_CUDA

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <cutensornet.h>
DISABLE_WARNING_POP

#include "misc/wrap_thrust.hpp"

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "framework/utils.hpp"
#include <complex>

#include "simulators/tensor_network/tensor.hpp"
#include "simulators/tensor_network/tensor_net_contractor.hpp"

#include "simulators/statevector/chunk/thrust_kernels.hpp"

namespace AER {
namespace TensorNetwork {

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSORNET_STATUS_SUCCESS) {                                   \
      std::stringstream str;                                                   \
      str << "ERROR TensorNet::contractor : "                                  \
          << cutensornetGetErrorString(err);                                   \
      throw std::runtime_error(str.str());                                     \
    }                                                                          \
  };

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::stringstream str;                                                   \
      str << "ERROR TensorNet::contractor : " << cudaGetErrorString(err);      \
      throw std::runtime_error(str.str());                                     \
    }                                                                          \
  };

// tensor data on each device
template <typename data_t = double>
class RawTensorData {
protected:
  int device_id_;
  cudaStream_t stream_;
  uint_t num_tensors_;
  uint_t num_additional_tensors_;
  thrust::device_vector<thrust::complex<data_t>>
      dev_tensor_data_; // tensor data array on device
  thrust::device_vector<thrust::complex<data_t>>
      dev_additional_tensor_data_; // additional tensor data on device
  thrust::device_vector<thrust::complex<data_t>> dev_out_; // output buffer
  thrust::device_vector<double> sampling_rnds_;
  thrust::device_vector<uint_t> sampling_out_;
  thrust::device_vector<unsigned char> dev_work_; // work buffer
  std::vector<void *> dev_data_ptr_; // array of pointer to each tensor
  uint_t tensor_size_;
  uint_t additional_tensor_size_;
  uint_t out_size_;
  uint_t work_size_limit_;
  uint_t work_size_;
  uint_t sampling_buffer_size_;

  // handles of cuTensorNet
  cutensornetHandle_t hTensorNet_;
  cutensornetNetworkDescriptor_t tn_desc_;
  cutensornetContractionOptimizerConfig_t optimizer_config_;
  cutensornetContractionOptimizerInfo_t optimizer_info_;
  cutensornetContractionPlan_t plan_;
  cutensornetWorkspaceDescriptor_t work_desc_;
  cutensornetContractionAutotunePreference_t autotunePref_;

  void release_cuTensorNet(void);

  void assert_error(const char *name, const char *desc) {
    std::stringstream str;
    str << "ERROR TensorNet::contractor in " << name << " : " << desc;
    throw std::runtime_error(str.str());
  }

public:
  RawTensorData();
  ~RawTensorData();
  RawTensorData(const RawTensorData &obj) {}

  void set_device(int idev) { device_id_ = idev; }
  bool work_allocated(void) { return (work_size_ > 0); }

  void reserve_arrays(uint_t num_tensors);

  void allocate_tensors(uint_t size);
  void allocate_additional_tensors(uint_t size);
  void allocate_output(uint_t size);
  void allocate_work(uint_t size);

  void copy_tensors(const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
                    bool add_sp_tensors);
  void copy_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors);
  void update_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors);

  void copy_tensors_from_device(const RawTensorData<data_t> &src);
  void copy_optimization_from_device(const RawTensorData<data_t> &src);

  void remove_additional_tensors(uint_t num_add);

  void create_contraction_descriptor(uint_t num_tensors,
                                     std::vector<int32_t *> &modes,
                                     std::vector<int32_t> &num_modes,
                                     std::vector<int64_t *> &extents,
                                     std::vector<int64_t *> &strides,
                                     std::vector<int32_t> &modes_out,
                                     std::vector<int64_t> &extents_out);
  uint_t optimize_contraction(void);
  void create_contraction_plan(bool use_autotune);

  void contract(uint_t islice_begin, uint_t islice_end);

  void get_output(std::vector<std::complex<data_t>> &out);
  void update_output(std::vector<std::complex<data_t>> &out);

  // accumulate output on different device
  void accumulate_output(const RawTensorData<data_t> &src);

  // get trace of output
  double trace_output(uint_t num_qubits);

  // sampling measure
  double sample_measure(reg_t &samples, std::vector<double> &rnds,
                        uint_t num_qubits);

  void allocate_sampling_buffers(uint_t size);
  void deallocate_sampling_buffers(void);
};

template <typename data_t>
RawTensorData<data_t>::RawTensorData() {
  device_id_ = 0;
  num_tensors_ = 0;
  num_additional_tensors_ = 0;

  tensor_size_ = 0;
  additional_tensor_size_ = 0;
  out_size_ = 0;
  work_size_ = 0;

  hTensorNet_ = nullptr;
  stream_ = nullptr;
  autotunePref_ = nullptr;
}

template <typename data_t>
RawTensorData<data_t>::~RawTensorData() {
  cudaSetDevice(device_id_);

  release_cuTensorNet();

  dev_tensor_data_.clear();
  dev_tensor_data_.shrink_to_fit();

  dev_additional_tensor_data_.clear();
  dev_additional_tensor_data_.shrink_to_fit();

  dev_out_.clear();
  dev_out_.shrink_to_fit();

  dev_work_.clear();
  dev_work_.shrink_to_fit();

  dev_data_ptr_.clear();

  sampling_rnds_.clear();
  sampling_rnds_.shrink_to_fit();

  sampling_out_.clear();
  sampling_out_.shrink_to_fit();

  if (stream_)
    cudaStreamDestroy(stream_);
}

template <typename data_t>
void RawTensorData<data_t>::release_cuTensorNet(void) {
  if (hTensorNet_) {
    HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(tn_desc_));
    HANDLE_ERROR(cutensornetDestroyContractionPlan(plan_));
    HANDLE_ERROR(
        cutensornetDestroyContractionOptimizerConfig(optimizer_config_));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizer_info_));
    HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(work_desc_));
    if (autotunePref_)
      HANDLE_ERROR(
          cutensornetDestroyContractionAutotunePreference(autotunePref_));
    HANDLE_ERROR(cutensornetDestroy(hTensorNet_));
  }
  hTensorNet_ = nullptr;
}

template <typename data_t>
void RawTensorData<data_t>::reserve_arrays(uint_t num_tensors) {
  num_tensors_ = num_tensors;
  dev_data_ptr_.reserve(num_tensors);
}

template <typename data_t>
void RawTensorData<data_t>::allocate_tensors(uint_t size) {
  cudaSetDevice(device_id_);

  cudaError_t err;
  if (!stream_) {
    err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess)
      assert_error("allocate_tensor: cudaStreamCreateWithFlags",
                   cudaGetErrorString(err));
  }

  if (tensor_size_ < size) {
    dev_tensor_data_.resize(size);
    tensor_size_ = size;
  }
}

template <typename data_t>
void RawTensorData<data_t>::allocate_additional_tensors(uint_t size) {
  cudaSetDevice(device_id_);

  if (additional_tensor_size_ < size) {
    dev_additional_tensor_data_.resize(size);
    additional_tensor_size_ = size;
  }
}

template <typename data_t>
void RawTensorData<data_t>::allocate_output(uint_t size) {
  cudaSetDevice(device_id_);

  if (out_size_ < size) {
    dev_out_.resize(size);
    out_size_ = size;
  }
}

template <typename data_t>
void RawTensorData<data_t>::allocate_work(uint_t size) {
  cudaSetDevice(device_id_);
  if (work_size_ < size) {
    dev_work_.resize(size);
    work_size_ = size;
  }
}

template <typename data_t>
void RawTensorData<data_t>::copy_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    bool add_sp_tensors) {
  cudaSetDevice(device_id_);

  uint_t size = 0;
  for (int_t i = 0; i < tensors.size(); i++) {
    if (add_sp_tensors || !tensors[i]->sp_tensor()) {
      std::complex<data_t> *ptr =
          (std::complex<data_t> *)thrust::raw_pointer_cast(
              dev_tensor_data_.data()) +
          size;
      dev_data_ptr_.push_back(ptr);
      cudaError_t err = cudaMemcpyAsync(ptr, tensors[i]->tensor().data(),
                                        sizeof(std::complex<data_t>) *
                                            tensors[i]->tensor().size(),
                                        cudaMemcpyHostToDevice, stream_);
      if (err != cudaSuccess)
        assert_error("copy_tensors: cudaMemcpyAsync", cudaGetErrorString(err));
      size += tensors[i]->tensor().size();
    }
  }
  //  cudaStreamSynchronize(stream_);
}

template <typename data_t>
void RawTensorData<data_t>::copy_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  cudaSetDevice(device_id_);

  uint_t size = 0;
  for (int_t i = 0; i < tensors.size(); i++) {
    std::complex<data_t> *ptr =
        (std::complex<data_t> *)thrust::raw_pointer_cast(
            dev_additional_tensor_data_.data()) +
        size;
    dev_data_ptr_.push_back(ptr);
    cudaError_t err = cudaMemcpyAsync(ptr, tensors[i]->tensor().data(),
                                      sizeof(std::complex<data_t>) *
                                          tensors[i]->tensor().size(),
                                      cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess)
      assert_error("copy_additional_tensors: cudaMemcpyAsync",
                   cudaGetErrorString(err));
    size += tensors[i]->tensor().size();
  }
  num_additional_tensors_ = tensors.size();
}

template <typename data_t>
void RawTensorData<data_t>::update_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  cudaSetDevice(device_id_);

  uint_t size = 0;
  for (int_t i = 0; i < tensors.size(); i++) {
    std::complex<data_t> *ptr =
        (std::complex<data_t> *)thrust::raw_pointer_cast(
            dev_additional_tensor_data_.data()) +
        size;
    dev_data_ptr_.push_back(ptr);
    cudaError_t err = cudaMemcpyAsync(ptr, tensors[i]->tensor().data(),
                                      sizeof(std::complex<data_t>) *
                                          tensors[i]->tensor().size(),
                                      cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess)
      assert_error("update_additional_tensors: cudaMemcpyAsync",
                   cudaGetErrorString(err));
    size += tensors[i]->tensor().size();
  }
}

template <typename data_t>
void RawTensorData<data_t>::copy_tensors_from_device(
    const RawTensorData<data_t> &src) {
  allocate_tensors(src.tensor_size_);
  if (src.additional_tensor_size_ > 0)
    allocate_additional_tensors(src.additional_tensor_size_);
  allocate_output(src.out_size_);

  num_tensors_ = src.num_tensors_;
  num_additional_tensors_ = src.num_additional_tensors_;

  int peer;
  cudaDeviceCanAccessPeer(&peer, device_id_, src.device_id_);
  if (peer) {
    if (cudaDeviceEnablePeerAccess(src.device_id_, 0) != cudaSuccess)
      cudaGetLastError();

    cudaMemcpyAsync(thrust::raw_pointer_cast(dev_tensor_data_.data()),
                    thrust::raw_pointer_cast(src.dev_tensor_data_.data()),
                    tensor_size_ * sizeof(thrust::complex<data_t>),
                    cudaMemcpyDeviceToDevice, src.stream_);
    if (src.additional_tensor_size_ > 0) {
      cudaMemcpyAsync(
          thrust::raw_pointer_cast(dev_additional_tensor_data_.data()),
          thrust::raw_pointer_cast(src.dev_additional_tensor_data_.data()),
          additional_tensor_size_ * sizeof(thrust::complex<data_t>),
          cudaMemcpyDeviceToDevice, src.stream_);
    }
  } else {
    cudaMemcpyPeerAsync(
        thrust::raw_pointer_cast(dev_tensor_data_.data()), device_id_,
        thrust::raw_pointer_cast(src.dev_tensor_data_.data()), src.device_id_,
        tensor_size_ * sizeof(thrust::complex<data_t>), src.stream_);
    if (src.additional_tensor_size_ > 0) {
      cudaMemcpyPeerAsync(
          thrust::raw_pointer_cast(dev_additional_tensor_data_.data()),
          device_id_,
          thrust::raw_pointer_cast(src.dev_additional_tensor_data_.data()),
          src.device_id_,
          additional_tensor_size_ * sizeof(thrust::complex<data_t>),
          src.stream_);
    }
  }

  // calculate pointers
  dev_data_ptr_.resize(src.dev_data_ptr_.size());
  for (int_t i = 0; i < num_tensors_ - num_additional_tensors_; i++) {
    dev_data_ptr_[i] =
        thrust::raw_pointer_cast(dev_tensor_data_.data()) +
        ((uint_t)src.dev_data_ptr_[i] - (uint_t)src.dev_data_ptr_[0]);
  }
  for (int_t i = 0; i < num_additional_tensors_; i++) {
    dev_data_ptr_[num_tensors_ + i] =
        thrust::raw_pointer_cast(dev_additional_tensor_data_.data()) +
        ((uint_t)src.dev_data_ptr_[num_tensors_ + i] -
         (uint_t)thrust::raw_pointer_cast(
             src.dev_additional_tensor_data_.data()));
  }

  cudaStreamSynchronize(src.stream_);
}

template <typename data_t>
void RawTensorData<data_t>::copy_optimization_from_device(
    const RawTensorData<data_t> &src) {
  // copy optimizer
  cutensornetStatus_t err;
  size_t packed_size;
  cudaSetDevice(src.device_id_);
  err = cutensornetContractionOptimizerInfoGetPackedSize(
      src.hTensorNet_, src.optimizer_info_, &packed_size);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetContractionOptimizerInfoGetPackedSize",
                 cutensornetGetErrorString(err));

  std::vector<unsigned char> tmp(packed_size);
  err = cutensornetContractionOptimizerInfoPackData(
      src.hTensorNet_, src.optimizer_info_, tmp.data(), packed_size);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetContractionOptimizerInfoPackData",
                 cutensornetGetErrorString(err));

  cudaSetDevice(device_id_);
  err = cutensornetCreateContractionOptimizerInfoFromPackedData(
      hTensorNet_, tn_desc_, tmp.data(), packed_size, &optimizer_info_);
}

template <typename data_t>
void RawTensorData<data_t>::remove_additional_tensors(uint_t num_add) {
  dev_data_ptr_.erase(dev_data_ptr_.end() - num_add, dev_data_ptr_.end());

  num_additional_tensors_ = 0;
}

template <typename data_t>
void RawTensorData<data_t>::create_contraction_descriptor(
    uint_t num_tensors, std::vector<int32_t *> &modes,
    std::vector<int32_t> &num_modes, std::vector<int64_t *> &extents,
    std::vector<int64_t *> &strides, std::vector<int32_t> &modes_out,
    std::vector<int64_t> &extents_out) {
  cutensornetStatus_t err;
  cudaSetDevice(device_id_);

  if (!hTensorNet_) {
    err = cutensornetCreate(&hTensorNet_);
    if (err != CUTENSORNET_STATUS_SUCCESS)
      assert_error("cutensornetCreate", cutensornetGetErrorString(err));
  }

  cudaDataType_t dtype;
  cutensornetComputeType_t ctype;

  if (sizeof(data_t) == 8) { // double precision
    dtype = CUDA_C_64F;
    ctype = CUTENSORNET_COMPUTE_64F;
  } else {
    dtype = CUDA_C_32F;
    ctype = CUTENSORNET_COMPUTE_TF32;
  }

  // network descriptor
  err = cutensornetCreateNetworkDescriptor(
      hTensorNet_, num_modes.size(), num_modes.data(), extents.data(),
      strides.data(), modes.data(), nullptr, extents_out.size(),
      extents_out.data(), nullptr, modes_out.data(), dtype, ctype, &tn_desc_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetCreateNetworkDescriptor",
                 cutensornetGetErrorString(err));
}

template <typename data_t>
uint_t RawTensorData<data_t>::optimize_contraction(void) {
  cutensornetStatus_t err;
  cudaSetDevice(device_id_);

  size_t freeMem, totalMem;
  int nid = omp_get_num_threads();

  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  work_size_limit_ = (freeMem / nid) * 0.9;

  /*******************************
   * Find "optimal" contraction order and slicing
   *******************************/
  err = cutensornetCreateContractionOptimizerConfig(hTensorNet_,
                                                    &optimizer_config_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetCreateContractionOptimizerConfig",
                 cutensornetGetErrorString(err));

  // Set the value of the partitioner imbalance factor, if desired
  int32_t num_hypersamples = 8;
  err = cutensornetContractionOptimizerConfigSetAttribute(
      hTensorNet_, optimizer_config_,
      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
      &num_hypersamples, sizeof(num_hypersamples));
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetContractionOptimizerConfigSetAttribute",
                 cutensornetGetErrorString(err));

  err = cutensornetCreateContractionOptimizerInfo(hTensorNet_, tn_desc_,
                                                  &optimizer_info_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetCreateContractionOptimizerInfo",
                 cutensornetGetErrorString(err));

  err = cutensornetContractionOptimize(hTensorNet_, tn_desc_, optimizer_config_,
                                       work_size_limit_, optimizer_info_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetContractionOptimize",
                 cutensornetGetErrorString(err));

  uint_t num_slices = 0;
  err = cutensornetContractionOptimizerInfoGetAttribute(
      hTensorNet_, optimizer_info_,
      CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &num_slices,
      sizeof(num_slices));
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetContractionOptimizerInfoGetAttribute",
                 cutensornetGetErrorString(err));

  return num_slices;
}

template <typename data_t>
void RawTensorData<data_t>::create_contraction_plan(bool use_autotune) {
  cutensornetStatus_t err;
  cudaSetDevice(device_id_);

  /*******************************
   * Initialize all pair-wise contraction plans (for cuTENSOR)
   *******************************/
  err = cutensornetCreateWorkspaceDescriptor(hTensorNet_, &work_desc_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetCreateWorkspaceDescriptor",
                 cutensornetGetErrorString(err));

  int64_t requiredWorkspaceSize = 0;
  err = cutensornetWorkspaceComputeContractionSizes(
      hTensorNet_, tn_desc_, optimizer_info_, work_desc_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetWorkspaceComputeSizes",
                 cutensornetGetErrorString(err));

  err = cutensornetWorkspaceGetMemorySize(
      hTensorNet_, work_desc_, CUTENSORNET_WORKSIZE_PREF_MIN,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
      &requiredWorkspaceSize);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetWorkspaceGetSize", cutensornetGetErrorString(err));

  allocate_work(requiredWorkspaceSize);

  err = cutensornetWorkspaceSetMemory(
      hTensorNet_, work_desc_, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, thrust::raw_pointer_cast(dev_work_.data()),
      work_size_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetWorkspaceSet", cutensornetGetErrorString(err));

  err = cutensornetCreateContractionPlan(hTensorNet_, tn_desc_, optimizer_info_,
                                         work_desc_, &plan_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetCreateContractionPlan",
                 cutensornetGetErrorString(err));

  /*******************************
   * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest
   *kernel
   *******************************/
  if (use_autotune) {
    err = cutensornetCreateContractionAutotunePreference(hTensorNet_,
                                                         &autotunePref_);
    if (err != CUTENSORNET_STATUS_SUCCESS)
      assert_error("cutensornetCreateContractionAutotunePreference",
                   cutensornetGetErrorString(err));

    const int numAutotuningIterations = 5; // may be 0
    err = cutensornetContractionAutotunePreferenceSetAttribute(
        hTensorNet_, autotunePref_,
        CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
        &numAutotuningIterations, sizeof(numAutotuningIterations));
    if (err != CUTENSORNET_STATUS_SUCCESS)
      assert_error("cutensornetContractionAutotunePreferenceSetAttribute",
                   cutensornetGetErrorString(err));

    // modify the plan again to find the best pair-wise contractions
    err = cutensornetContractionAutotune(
        hTensorNet_, plan_, dev_data_ptr_.data(),
        thrust::raw_pointer_cast(dev_out_.data()), work_desc_, autotunePref_,
        stream_);
    if (err != CUTENSORNET_STATUS_SUCCESS)
      assert_error("cutensornetContractionAutotune",
                   cutensornetGetErrorString(err));
  }
}

template <typename data_t>
void RawTensorData<data_t>::contract(uint_t islice_begin, uint_t islice_end) {
  cutensornetSliceGroup_t group;
  cutensornetStatus_t err;

  cudaSetDevice(device_id_);

  err = cutensornetCreateSliceGroupFromIDRange(hTensorNet_, islice_begin,
                                               islice_end, 1, &group);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetCreateSliceGroupFromIDRange",
                 cutensornetGetErrorString(err));

  int32_t accum = 0;
  if (islice_end - islice_begin > 1) {
    // clear output
    cudaMemsetAsync(thrust::raw_pointer_cast(dev_out_.data()), 0,
                    sizeof(std::complex<data_t>) * out_size_, stream_);
    accum = 1;
  }

  // do contraction
  err = cutensornetContractSlices(hTensorNet_, plan_, dev_data_ptr_.data(),
                                  thrust::raw_pointer_cast(dev_out_.data()),
                                  accum, work_desc_, group, stream_);
  if (err != CUTENSORNET_STATUS_SUCCESS)
    assert_error("cutensornetContractSlices", cutensornetGetErrorString(err));

  // no synchronization here, synchronize after copying output to host
}

template <typename data_t>
void RawTensorData<data_t>::get_output(std::vector<std::complex<data_t>> &out) {
  if (out.size() < out_size_)
    out.resize(out_size_);

  cudaSetDevice(device_id_);
  cudaMemcpyAsync(out.data(), thrust::raw_pointer_cast(dev_out_.data()),
                  sizeof(std::complex<data_t>) * out_size_,
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
}

template <typename data_t>
void RawTensorData<data_t>::update_output(
    std::vector<std::complex<data_t>> &out) {
  cudaSetDevice(device_id_);
  cudaMemcpyAsync(thrust::raw_pointer_cast(dev_out_.data()), out.data(),
                  sizeof(std::complex<data_t>) * out_size_,
                  cudaMemcpyHostToDevice, stream_);
  cudaStreamSynchronize(stream_);
}

template <typename data_t>
void RawTensorData<data_t>::accumulate_output(
    const RawTensorData<data_t> &src) {
  int peer;

  cudaSetDevice(device_id_);
  cudaDeviceCanAccessPeer(&peer, device_id_, src.device_id_);
  if (peer) {
    if (cudaDeviceEnablePeerAccess(src.device_id_, 0) != cudaSuccess)
      cudaGetLastError();

    thrust::plus<thrust::complex<data_t>> op;
    thrust::transform(thrust::cuda::par.on(stream_), dev_out_.begin(),
                      dev_out_.begin() + out_size_, src.dev_out_.begin(),
                      dev_out_.begin(), op);
  } else {
    thrust::device_vector<thrust::complex<data_t>> tmp;
    tmp.resize(out_size_);
    cudaMemcpyPeerAsync(thrust::raw_pointer_cast(tmp.data()), device_id_,
                        thrust::raw_pointer_cast(src.dev_out_.data()),
                        src.device_id_,
                        out_size_ * sizeof(thrust::complex<data_t>), stream_);
    cudaStreamSynchronize(stream_);
    thrust::plus<thrust::complex<data_t>> op;
    thrust::transform(thrust::cuda::par.on(stream_), dev_out_.begin(),
                      dev_out_.begin() + out_size_, tmp.begin(),
                      dev_out_.begin(), op);

    cudaStreamSynchronize(stream_); // need sync to delete tmp
    tmp.clear();
    tmp.shrink_to_fit();
  }
}

template <typename data_t>
double RawTensorData<data_t>::trace_output(uint_t num_qubits) {
  thrust::complex<data_t> ret;
  cudaSetDevice(device_id_);

  uint_t stride = (1ull << num_qubits) + 1;
  QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
      (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev_out_.data()),
      (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev_out_.data()) +
          out_size_,
      stride);

  thrust::plus<thrust::complex<data_t>> op;
  ret = thrust::reduce(thrust::cuda::par.on(stream_), iter.begin(), iter.end());

  return ret.real();
}

template <typename data_t>
void RawTensorData<data_t>::allocate_sampling_buffers(uint_t size) {
  cudaSetDevice(device_id_);
  sampling_rnds_.resize(size);
  sampling_out_.resize(size);
  sampling_buffer_size_ = size;
}

template <typename data_t>
void RawTensorData<data_t>::deallocate_sampling_buffers(void) {
  cudaSetDevice(device_id_);
  sampling_rnds_.clear();
  sampling_rnds_.shrink_to_fit();

  sampling_out_.clear();
  sampling_out_.shrink_to_fit();
}

// device function to update(-) rnds by sampled probabilities
template <typename data_t>
class sampling_update_rnd_func {
protected:
  thrust::complex<data_t> *data_;
  uint_t stride_;
  uint_t *index_;
  double *rnds_;

public:
  sampling_update_rnd_func(thrust::complex<data_t> *p, uint_t st, uint_t *idx,
                           double *rnd) {
    data_ = p;
    stride_ = st;
    index_ = idx;
    rnds_ = rnd;
  }
  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t pos;
    thrust::complex<data_t> d;
    pos = index_[i];
    if (pos > 0) {
      double t = rnds_[i];
      pos = (pos - 1) * stride_;
      d = data_[pos];
      t -= d.real();
      rnds_[i] = t;
    }
  }
};

template <typename data_t>
double RawTensorData<data_t>::sample_measure(reg_t &samples,
                                             std::vector<double> &rnds,
                                             uint_t num_qubits) {
  if (samples.size() < rnds.size())
    samples.resize(rnds.size());

  cudaSetDevice(device_id_);

  uint_t stride = (1ull << num_qubits) + 1;
  QV::Chunk::strided_range<thrust::complex<data_t> *> iter(
      (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev_out_.data()),
      (thrust::complex<data_t> *)thrust::raw_pointer_cast(dev_out_.data()) +
          out_size_,
      stride);

  // reduce trace
  thrust::inclusive_scan(thrust::cuda::par.on(stream_), iter.begin(),
                         iter.end(), iter.begin(),
                         thrust::plus<thrust::complex<data_t>>());

  uint_t pos = 0;
  while (pos < rnds.size()) {
    uint_t nsamples = sampling_buffer_size_;
    if (pos + nsamples > rnds.size())
      nsamples = rnds.size() - pos;

    cudaMemcpyAsync((double *)thrust::raw_pointer_cast(sampling_rnds_.data()),
                    &rnds[pos], nsamples * sizeof(double),
                    cudaMemcpyHostToDevice, stream_);

    thrust::lower_bound(thrust::cuda::par.on(stream_), iter.begin(), iter.end(),
                        sampling_rnds_.begin(), sampling_rnds_.end(),
                        sampling_out_.begin(),
                        QV::Chunk::complex_less<data_t>());

    // update rnds
    auto ci = thrust::counting_iterator<uint_t>(0);
    thrust::for_each_n(
        thrust::cuda::par.on(stream_), ci, nsamples,
        sampling_update_rnd_func<data_t>(
            (thrust::complex<data_t> *)thrust::raw_pointer_cast(
                dev_out_.data()),
            stride, (uint_t *)thrust::raw_pointer_cast(sampling_out_.data()),
            (double *)thrust::raw_pointer_cast(sampling_rnds_.data())));

    cudaMemcpyAsync(&samples[pos],
                    (uint_t *)thrust::raw_pointer_cast(sampling_out_.data()),
                    nsamples * sizeof(uint_t), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(&rnds[pos],
                    (double *)thrust::raw_pointer_cast(sampling_rnds_.data()),
                    nsamples * sizeof(double), cudaMemcpyDeviceToHost, stream_);

    pos += nsamples;
  }

  cudaStreamSynchronize(stream_);
  thrust::complex<data_t> t = dev_out_[out_size_ - 1]; // return reduced trace
  return t.real();
}

// tensor network contractor for cuTensorNet
template <typename data_t = double>
class TensorNetContractor_cuTensorNet : public TensorNetContractor<data_t> {
protected:
  uint_t num_tensors_;
  uint_t num_additional_tensors_;
  std::vector<int32_t *> modes_;
  std::vector<int32_t> num_modes_;
  std::vector<int64_t *> extents_;
  std::vector<int64_t *> strides_;

  std::vector<int32_t> modes_out_;
  std::vector<int64_t> extents_out_;
  uint_t out_size_;

  uint_t num_slices_;
  uint_t islice_begin_;
  uint_t islice_end_;

  std::vector<RawTensorData<data_t>> tensor_data_;

  int num_devices_ = 1;
  int num_devices_used_ = 1;
  int nprocs_ = 1;
  int myrank_ = 0;

  reg_t target_gpus_;

public:
  TensorNetContractor_cuTensorNet();
  ~TensorNetContractor_cuTensorNet();

  void set_device(int idev) override {}
  void allocate_additional_tensors(uint_t size) override;

  void set_network(const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
                   bool add_sp_tensors = true) override;
  void set_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) override;
  void update_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) override;
  void set_output(std::vector<int32_t> &modes,
                  std::vector<int64_t> &extents) override;

  uint_t num_slices(void) override { return num_slices_; }

  void setup_contraction(bool use_autotune = false) override;

  void contract(std::vector<std::complex<data_t>> &out) override;
  double contract_and_trace(uint_t num_qubits) override;

  double contract_and_sample_measure(reg_t &samples, std::vector<double> &rnds,
                                     uint_t num_qubits) override;

  void
  allocate_sampling_buffers(uint_t size = AER_TENSOR_NET_MAX_SAMPLING) override;
  void deallocate_sampling_buffers(void) override;

  void set_target_gpus(reg_t &t) override { target_gpus_ = t; }

protected:
  void remove_additional_tensors(void);

  void assert_error(const char *name, const char *desc) {
    std::stringstream str;
    str << "ERROR TensorNet::contractor in " << name << " : " << desc;
    throw std::runtime_error(str.str());
  }

  void contract_all(void);
};

template <typename data_t>
TensorNetContractor_cuTensorNet<data_t>::TensorNetContractor_cuTensorNet() {
  num_additional_tensors_ = 0;
}

template <typename data_t>
TensorNetContractor_cuTensorNet<data_t>::~TensorNetContractor_cuTensorNet() {
  tensor_data_.clear();
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::set_network(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
    bool add_sp_tensors) {
  uint_t size = 0;

  // allocate tensor data storage for each device
  if (cudaGetDeviceCount(&num_devices_) != cudaSuccess)
    cudaGetLastError();
  if (target_gpus_.size() > 0) {
    num_devices_ = target_gpus_.size();
  } else {
    target_gpus_.resize(num_devices_);
    for (int_t i = 0; i < num_devices_; i++)
      target_gpus_[i] = i;
  }

  tensor_data_.clear();
  tensor_data_.resize(num_devices_);
  for (int_t i = 0; i < num_devices_; i++) {
    tensor_data_[i].set_device(target_gpus_[i]);
  }

  // count number of tensors
  if (!add_sp_tensors) {
    num_tensors_ = 0;
    for (int_t i = 0; i < tensors.size(); i++) {
      if (!tensors[i]->sp_tensor()) {
        num_tensors_++;
      }
    }
  } else {
    num_tensors_ = tensors.size();
  }

  modes_.reserve(num_tensors_);
  num_modes_.reserve(num_tensors_);
  extents_.reserve(num_tensors_);
  strides_.reserve(num_tensors_);

  // convert tensor network data to cuTensorNet format
  for (int_t i = 0; i < tensors.size(); i++) {
    if (add_sp_tensors || !tensors[i]->sp_tensor()) {
      modes_.push_back(tensors[i]->modes().data());
      num_modes_.push_back(tensors[i]->modes().size());
      extents_.push_back((int64_t *)tensors[i]->extents().data());
      strides_.push_back(nullptr);

      size += tensors[i]->tensor().size();
    }
  }

  // copy tensors to 1st GPU (use other GPUs if number of slices > 1)
  tensor_data_[0].reserve_arrays(num_tensors_);
  tensor_data_[0].allocate_tensors(size);
  tensor_data_[0].copy_tensors(tensors, add_sp_tensors);

  num_devices_used_ = 1;
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::allocate_additional_tensors(
    uint_t size) {
  tensor_data_[0].allocate_additional_tensors(size);
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::set_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  remove_additional_tensors();

  num_additional_tensors_ = tensors.size();
  for (int_t i = 0; i < num_additional_tensors_; i++) {
    modes_.push_back(tensors[i]->modes().data());
    num_modes_.push_back(tensors[i]->modes().size());
    extents_.push_back((int64_t *)tensors[i]->extents().data());
    strides_.push_back(nullptr);
  }

  tensor_data_[0].copy_additional_tensors(tensors);
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::update_additional_tensors(
    const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) {
  for (int i = 0; i < num_devices_used_; i++)
    tensor_data_[i].update_additional_tensors(tensors);
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::remove_additional_tensors(void) {
  if (num_additional_tensors_ > 0) {
    modes_.erase(modes_.end() - num_additional_tensors_, modes_.end());
    num_modes_.erase(num_modes_.end() - num_additional_tensors_,
                     num_modes_.end());
    extents_.erase(extents_.end() - num_additional_tensors_, extents_.end());
    strides_.erase(strides_.end() - num_additional_tensors_, strides_.end());

    for (int i = 0; i < num_devices_used_; i++)
      tensor_data_[i].remove_additional_tensors(num_additional_tensors_);

    num_additional_tensors_ = 0;
  }
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::set_output(
    std::vector<int32_t> &modes, std::vector<int64_t> &extents) {
  modes_out_ = modes;
  extents_out_ = extents;

  out_size_ = 1;
  for (int_t i = 0; i < extents_out_.size(); i++)
    out_size_ *= extents_out_[i];

  tensor_data_[0].allocate_output(out_size_);
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::setup_contraction(
    bool use_autotune) {

  // for MPI distribution
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  num_devices_used_ = 1;

  // setup first device
  tensor_data_[0].create_contraction_descriptor(num_tensors_, modes_,
                                                num_modes_, extents_, strides_,
                                                modes_out_, extents_out_);
  num_slices_ = tensor_data_[0].optimize_contraction();
  tensor_data_[0].create_contraction_plan(use_autotune);

  islice_begin_ = myrank_ * num_slices_ / nprocs_;
  islice_end_ = (myrank_ + 1) * num_slices_ / nprocs_;

  // distribute with multi-GPUs
  if ((islice_end_ - islice_begin_) > num_devices_ && num_devices_ > 1) {
    for (int_t i = 1; i < num_devices_; i++) {
      uint_t ns = (i + 1) * (islice_end_ - islice_begin_) / num_devices_;
      ns -= i * (islice_end_ - islice_begin_) / num_devices_;

      if (ns > 0) {
        // setup for the device
        tensor_data_[i].copy_tensors_from_device(
            tensor_data_[0]); // copy data from the first device
        tensor_data_[i].create_contraction_descriptor(
            num_tensors_, modes_, num_modes_, extents_, strides_, modes_out_,
            extents_out_);
        tensor_data_[i].copy_optimization_from_device(
            tensor_data_[0]); // copy optimizer from the first device
        tensor_data_[i].create_contraction_plan(use_autotune);
      }
    }
    num_devices_used_ = num_devices_;
  }
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::contract_all(void) {
  for (int_t idev = 0; idev < num_devices_used_; idev++) {
    uint_t is, ie;
    is = islice_begin_ +
         (islice_end_ - islice_begin_) * idev / num_devices_used_;
    ie = islice_begin_ +
         (islice_end_ - islice_begin_) * (idev + 1) / num_devices_used_;

    tensor_data_[idev].contract(is, ie);
  }
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::contract(
    std::vector<std::complex<data_t>> &out) {
  contract_all();

  for (int_t idev = 1; idev < num_devices_used_; idev++) { // naive accumulation
    tensor_data_[idev].accumulate_output(tensor_data_[idev]);
  }
  tensor_data_[0].get_output(out);

#ifdef AER_MPI
  // accumulate among all MPI processes
  if (nprocs_ > 1) {
    uint_t n = out.size();
    std::vector<std::complex<data_t>> tmp(n);
    if (sizeof(data_t) == 16)
      MPI_Allreduce(out.data(), tmp.data(), n * 2, MPI_DOUBLE_PRECISION,
                    MPI_SUM, MPI_COMM_WORLD);
    else
      MPI_Allreduce(out.data(), tmp.data(), n * 2, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
    out = tmp;
  }
#endif
}

template <typename data_t>
double
TensorNetContractor_cuTensorNet<data_t>::contract_and_trace(uint_t num_qubits) {
  double ret = 0.0;
  contract_all();

  for (int_t idev = 0; idev < num_devices_used_; idev++) {
    ret += tensor_data_[idev].trace_output(num_qubits);
  }

#ifdef AER_MPI
  // reduce among all MPI processes
  if (nprocs_ > 1) {
    double sum = ret;
    MPI_Allreduce(&sum, &ret, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);
  }
#endif

  return ret;
}

template <typename data_t>
double TensorNetContractor_cuTensorNet<data_t>::contract_and_sample_measure(
    reg_t &samples, std::vector<double> &rnds, uint_t num_qubits) {
  contract_all();

  // accumulate on device 0
  for (int_t idev = 1; idev < num_devices_used_; idev++) { // naive accumulation
    tensor_data_[idev].accumulate_output(tensor_data_[idev]);
  }

#ifdef AER_MPI
  // accumulate among all MPI processes
  if (nprocs_ > 1) {
    std::vector<std::complex<data_t>> out;
    tensor_data_[0].get_output(out);
    uint_t n = out.size();
    std::vector<std::complex<data_t>> tmp(n);
    if (sizeof(data_t) == 16)
      MPI_Allreduce(out.data(), tmp.data(), n * 2, MPI_DOUBLE_PRECISION,
                    MPI_SUM, MPI_COMM_WORLD);
    else
      MPI_Allreduce(out.data(), tmp.data(), n * 2, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
    tensor_data_[0].update_output(out);
  }
#endif

  // sample on device 0
  return tensor_data_[0].sample_measure(samples, rnds, num_qubits);
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::allocate_sampling_buffers(
    uint_t size) {
  // sample on device 0 only
  tensor_data_[0].allocate_sampling_buffers(size);
}

template <typename data_t>
void TensorNetContractor_cuTensorNet<data_t>::deallocate_sampling_buffers(
    void) {
  tensor_data_[0].deallocate_sampling_buffers();
}

//------------------------------------------------------------------------------
} // namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif // AER_THRUST_CUDA

#endif //_tensor_net_contractor_cuTensorNet_hpp_
