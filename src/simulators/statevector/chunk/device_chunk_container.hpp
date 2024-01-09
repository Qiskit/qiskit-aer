/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_device_chunk_container_hpp_
#define _qv_device_chunk_container_hpp_

#include "simulators/statevector/chunk/chunk_container.hpp"
#include "simulators/statevector/chunk/thrust_kernels.hpp"

#include "misc/gpu_static_properties.hpp"

#ifdef AER_THRUST_CUDA
namespace thrust_gpu = thrust::cuda;
#endif
#ifdef AER_THRUST_ROCM
namespace thrust_gpu = thrust::hip;
#endif

namespace AER {
namespace QV {
namespace Chunk {

// reserve 512MB of memory for Thrust internal use
#define RESERVE_FOR_THRUST (1ull << 28)

// max storage reserved for sampling measure
#define AER_MAX_SAMPLING_SHOTS 1024

//============================================================================
// device chunk container class
//============================================================================
template <typename data_t>
class DeviceChunkContainer : public ChunkContainer<data_t> {
protected:
  // device vector to chunks and buffers
  AERDeviceVector<thrust::complex<data_t>> data_;
  // storage for large matrix
  mutable AERDeviceVector<thrust::complex<double>> matrix_;
  // storage for additional parameters
  mutable AERDeviceVector<uint_t> params_;
  // buffer for reduction
  AERDeviceVector<double> reduce_buffer_;
  // buffer used for measure probability
  AERDeviceVector<double> probability_buffer_;

  AERDeviceVector<uint_t> cregs_;
  AERHostVector<uint_t> cregs_host_;
  int device_id_;                 // device index
  std::vector<bool> peer_access_; // to which device accepts peer access
  uint_t matrix_buffer_size_;     // matrix buffer size per chunk
  uint_t params_buffer_size_;     // params buffer size per chunk
  uint_t num_matrices_; // number of matrices for chunks (1 shared matrix for
                        // multi-chunk execution)
  uint_t reduce_buffer_size_;

  bool multi_shots_; // multi-shot parallelization

  bool creg_host_update_;
  bool creg_dev_update_;

  // for register blocking
  thrust::host_vector<uint_t> blocked_qubits_holder_;
  uint_t max_blocked_gates_;
  reg_t num_blocked_gates_;
  reg_t num_blocked_matrix_;
  reg_t num_blocked_qubits_;

#ifdef AER_THRUST_GPU
  std::vector<cudaStream_t> stream_; // asynchronous execution
#endif

public:
  DeviceChunkContainer() {
    device_id_ = 0;
    matrix_buffer_size_ = 0;
    params_buffer_size_ = 0;
    num_matrices_ = 1;
    multi_shots_ = false;
    creg_host_update_ = true;
    creg_dev_update_ = false;
  }
  ~DeviceChunkContainer();

  uint_t size(void) { return data_.size(); }
  int device(void) { return device_id_; }

  AERDeviceVector<thrust::complex<data_t>> &vector(void) { return data_; }

  bool peer_access(int i_dest) {
    if (i_dest < 0) {
#ifdef AER_ATS
      // for IBM AC922
      return true;
#else
      return false;
#endif
    }
    return peer_access_[i_dest];
  }

  thrust::complex<data_t> &operator[](uint_t i) {
    return raw_reference_cast(data_[i]);
  }

  uint_t Allocate(int idev, int chunk_bits, int num_qubits, uint_t chunks,
                  uint_t buffers, bool multi_shots, int matrix_bit,
                  int max_shots, bool density_matrix) override;
  void Deallocate(void) override;

  void StoreMatrix(const std::vector<std::complex<double>> &mat,
                   uint_t iChunk) const override;
  void StoreMatrix(const std::complex<double> *mat, uint_t iChunk,
                   uint_t size) const override;
  void StoreUintParams(const std::vector<uint_t> &prm,
                       uint_t iChunk) const override;
  void ResizeMatrixBuffers(int bits, int max_shots) override;

  void calculate_matrix_buffer_size(int bits, int shots);

  void set_device(void) const {
#ifdef AER_THRUST_GPU
    cudaSetDevice(device_id_);
#endif
  }

#ifdef AER_THRUST_GPU
  cudaStream_t stream(uint_t iChunk) const {
    if (iChunk >= this->num_chunks_)
      return stream_[(num_matrices_ + iChunk - this->num_chunks_)];
    if (num_matrices_ == 1)
      return stream_[0];
    return stream_[iChunk];
  }
#endif

  void Set(uint_t i, const thrust::complex<data_t> &t) { data_[i] = t; }
  thrust::complex<data_t> Get(uint_t i) const { return data_[i]; }

  void CopyIn(Chunk<data_t> &src, uint_t iChunk) override;
  void CopyOut(Chunk<data_t> &src, uint_t iChunk) override;
  void CopyIn(thrust::complex<data_t> *src, uint_t iChunk,
              uint_t size) override;
  void CopyOut(thrust::complex<data_t> *dest, uint_t iChunk,
               uint_t size) override;
  void Swap(Chunk<data_t> &src, uint_t iChunk, uint_t dest_offset = 0,
            uint_t src_offset = 0, uint_t size = 0,
            bool write_back = true) override;

  void Zero(uint_t iChunk, uint_t count) override;

  reg_t sample_measure(uint_t iChunk, const std::vector<double> &rnds,
                       uint_t stride = 1, bool dot = true,
                       uint_t count = 1) const override;

  thrust::complex<data_t> *chunk_pointer(uint_t iChunk) const {
    return (thrust::complex<data_t> *)thrust::raw_pointer_cast(data_.data()) +
           (iChunk << this->chunk_bits_);
  }
  thrust::complex<data_t> *buffer_pointer(void) const {
    return (thrust::complex<data_t> *)thrust::raw_pointer_cast(data_.data()) +
           (this->num_chunks_ << this->chunk_bits_);
  }

  thrust::complex<double> *matrix_pointer(uint_t iChunk) const {
    if (iChunk >= this->num_chunks_) { // for buffer chunks
      return ((thrust::complex<double> *)thrust::raw_pointer_cast(
                 matrix_.data())) +
             ((num_matrices_ + iChunk - this->num_chunks_) *
              matrix_buffer_size_);
    } else {
      if (num_matrices_ == 1)
        return ((thrust::complex<double> *)thrust::raw_pointer_cast(
            matrix_.data()));
      else
        return ((thrust::complex<double> *)thrust::raw_pointer_cast(
                   matrix_.data())) +
               (iChunk * matrix_buffer_size_);
    }
  }

  uint_t *param_pointer(uint_t iChunk) const {
    if (iChunk >= this->num_chunks_) { // for buffer chunks
      return ((uint_t *)thrust::raw_pointer_cast(params_.data())) +
             ((num_matrices_ + iChunk - this->num_chunks_) *
              params_buffer_size_);
    } else {
      if (num_matrices_ == 1)
        return ((uint_t *)thrust::raw_pointer_cast(params_.data()));
      else
        return ((uint_t *)thrust::raw_pointer_cast(params_.data())) +
               (iChunk * params_buffer_size_);
    }
  }

  double *reduce_buffer(uint_t iChunk) const {
    return ((double *)thrust::raw_pointer_cast(reduce_buffer_.data()) +
            iChunk * reduce_buffer_size_);
  }
  uint_t reduce_buffer_size() const { return reduce_buffer_size_; }
  double *probability_buffer(uint_t iChunk) const {
    return ((double *)thrust::raw_pointer_cast(probability_buffer_.data()) +
            iChunk * QV_PROBABILITY_BUFFER_SIZE);
  }

  void copy_to_probability_buffer(std::vector<double> &buf, int pos);
  void copy_reduce_buffer(std::vector<double> &ret, uint_t iChunk,
                          uint_t num_val) const override;

  void allocate_creg(uint_t num_mem, uint_t num_reg);
  int measured_cbit(uint_t iChunk, int qubit) {
    uint_t n64, i64, ibit;
    if ((uint_t)qubit >= this->num_creg_bits_)
      return -1;
    n64 = (this->num_creg_bits_ + 63) >> 6;
    i64 = qubit >> 6;
    ibit = qubit & 63;
    if (iChunk == 0 && creg_host_update_) {
      creg_host_update_ = false;
#ifdef AER_THRUST_GPU
      cudaMemcpyAsync(thrust::raw_pointer_cast(cregs_host_.data()),
                      thrust::raw_pointer_cast(cregs_.data()),
                      sizeof(uint_t) * num_matrices_ * n64,
                      cudaMemcpyDeviceToHost, stream_[0]);
      cudaStreamSynchronize(stream_[0]);
#else
      thrust::copy_n(cregs_.begin(), this->num_chunks_ * n64,
                     cregs_host_.begin());
#endif
    }

    return (cregs_host_[iChunk * n64 + i64] >> ibit) & 1;
  }

  void write_cbit(uint_t iChunk, int qubit, int val) {
    uint_t n64, i64, ibit;
    if (qubit >= this->num_creg_bits_)
      return;
    n64 = (this->num_creg_bits_ + 63) >> 6;
    i64 = qubit >> 6;
    ibit = qubit & 63;
    if (iChunk == 0 && creg_host_update_) {
      creg_host_update_ = false;
#ifdef AER_THRUST_GPU
      cudaMemcpyAsync(thrust::raw_pointer_cast(cregs_host_.data()),
                      thrust::raw_pointer_cast(cregs_.data()),
                      sizeof(uint_t) * num_matrices_ * n64,
                      cudaMemcpyDeviceToHost, stream_[0]);
      cudaStreamSynchronize(stream_[0]);
#else
      thrust::copy_n(cregs_.begin(), this->num_chunks_ * n64,
                     cregs_host_.begin());
#endif
    }

    cregs_host_[iChunk * n64 + i64] =
        (cregs_host_[iChunk * n64 + i64] & (~(1ull << ibit))) |
        (((uint_t)val & 1) << ibit);
    creg_dev_update_ = true;
  }
  void store_cbits(void) {
    if (creg_dev_update_) {
      uint_t n64;
      n64 = (this->num_creg_bits_ + 63) >> 6;
      creg_dev_update_ = false;
      creg_host_update_ = false;
#ifdef AER_THRUST_GPU
      cudaMemcpyAsync(thrust::raw_pointer_cast(cregs_.data()),
                      thrust::raw_pointer_cast(cregs_host_.data()),
                      sizeof(uint_t) * num_matrices_ * n64,
                      cudaMemcpyHostToDevice, stream_[0]);
#else
      thrust::copy_n(cregs_host_.begin(), this->num_chunks_ * n64,
                     cregs_.begin());
#endif
    }
  }

  uint_t *creg_buffer(uint_t iChunk) const {
    uint_t n64;
    n64 = (this->num_creg_bits_ + 63) >> 6;
    return ((uint_t *)thrust::raw_pointer_cast(cregs_.data()) + iChunk * n64);
  }
  void request_creg_update(void) { creg_host_update_ = true; }

  void synchronize(uint_t iChunk) const {
#ifdef AER_THRUST_GPU
    set_device();
    cudaStreamSynchronize(stream(iChunk));
#endif
  }

  // set qubits to be blocked
  void set_blocked_qubits(uint_t iChunk, const reg_t &qubits);

  // do all gates stored in queue
  void apply_blocked_gates(uint_t iChunk);

  // queue gate for blocked execution
  void queue_blocked_gate(uint_t iChunk, char gate, uint_t qubit, uint_t mask,
                          const std::complex<double> *pMat = NULL);
};

template <typename data_t>
DeviceChunkContainer<data_t>::~DeviceChunkContainer(void) {
  Deallocate();
}

template <typename data_t>
uint_t DeviceChunkContainer<data_t>::Allocate(int idev, int chunk_bits,
                                              int num_qubits, uint_t chunks,
                                              uint_t buffers, bool multi_shots,
                                              int matrix_bit, int max_shots,
                                              bool density_matrix) {
  uint_t nc = chunks;
  uint_t i;

  this->chunk_bits_ = chunk_bits;
  this->num_qubits_ = num_qubits;

  this->density_matrix_ = density_matrix;

  device_id_ = idev;
  set_device();

#ifdef AER_THRUST_GPU
  int ip, nd;
  cudaGetDeviceCount(&nd);
  peer_access_.resize(nd);
  for (i = 0; i < nd; i++) {
    ip = 1;
    if (i != device_id_) {
      cudaDeviceCanAccessPeer(&ip, device_id_, i);
    }
    if (ip) {
      if (cudaDeviceEnablePeerAccess(i, 0) != cudaSuccess)
        cudaGetLastError();
      peer_access_[i] = true;
    } else
      peer_access_[i] = false;
  }
#else
  peer_access_.resize(1);
  peer_access_[0] = true;
#endif

  this->num_buffers_ = buffers;

  if (multi_shots) { // mult-shot parallelization for small qubits
    multi_shots_ = true;
    nc = chunks;
    num_matrices_ = chunks;
  } else {
    multi_shots_ = false;
    num_matrices_ = 1;
    nc = chunks;
  }

  matrix_buffer_size_ = 0;
  params_buffer_size_ = 0;
  max_blocked_gates_ = QV_MAX_BLOCKED_GATES;
  calculate_matrix_buffer_size(matrix_bit, max_shots);

  reduce_buffer_size_ = 2;

#ifdef AER_THRUST_GPU
  size_t param_size = sizeof(thrust::complex<double>) * matrix_buffer_size_ +
                      sizeof(uint_t) * params_buffer_size_;

  if (chunk_bits < 10)
    reduce_buffer_size_ = 1;
  else
    reduce_buffer_size_ = (1ull << (chunk_bits - 10));
  reduce_buffer_size_ *= 2;

  param_size += sizeof(double) * reduce_buffer_size_;
  if (multi_shots)
    param_size += sizeof(double) * QV_PROBABILITY_BUFFER_SIZE +
                  sizeof(uint_t) * ((this->num_creg_bits_ + 63) >> 6);

  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  freeMem -= RESERVE_FOR_THRUST;
  while (freeMem <= ((((nc + buffers) * (uint_t)sizeof(thrust::complex<data_t>))
                      << chunk_bits) +
                     sizeof(double) * reduce_buffer_size_ * nc +
                     param_size * (num_matrices_ + buffers))) {
    nc--;
    if (num_matrices_ > 1)
      num_matrices_--;
    if (nc == 0) {
      break;
    }
  }

#endif
  ResizeMatrixBuffers(matrix_bit, max_shots);

  this->num_chunks_ = nc;
  data_.resize((nc + buffers) << chunk_bits);

  // init number of bits for chunk count
  uint_t nc_tmp = this->num_chunks_;
  this->num_pow2_qubits_ = this->chunk_bits_;
  while ((nc_tmp & 1) == 0) {
    this->num_pow2_qubits_++;
    nc_tmp >>= 1;
  }

  uint_t size = num_matrices_ + this->num_buffers_;

#ifdef AER_THRUST_GPU
  stream_.resize(size);
  for (int i = 0; i < size; i++)
    cudaStreamCreateWithFlags(&stream_[i], cudaStreamNonBlocking);

  if (chunk_bits < 10) {
    reduce_buffer_size_ = 1;
  } else {
    reduce_buffer_size_ = (1ull << (chunk_bits - 10));
  }
#else
  reduce_buffer_size_ = 1;
#endif

  reduce_buffer_size_ *= 2;
  reduce_buffer_.resize(reduce_buffer_size_ * nc);
  probability_buffer_.resize(nc * QV_PROBABILITY_BUFFER_SIZE);

  creg_host_update_ = false;
  this->num_creg_bits_ = num_qubits;

  num_blocked_gates_.resize(size);
  num_blocked_matrix_.resize(size);
  num_blocked_qubits_.resize(size);
  for (i = 0; i < size; i++) {
    num_blocked_gates_[i] = 0;
    num_blocked_matrix_[i] = 0;
  }
  blocked_qubits_holder_.resize(QV_MAX_REGISTERS * size);

  // allocate chunk classes
  ChunkContainer<data_t>::allocate_chunks();

  return nc;
}

template <typename data_t>
void DeviceChunkContainer<data_t>::allocate_creg(uint_t num_mem,
                                                 uint_t num_reg) {
  // allocate memory + register in the same array (reg first)
  this->num_creg_bits_ = num_mem + num_reg + QV_NUM_INTERNAL_REGS;
  this->num_cregisters_ = num_reg;
  this->num_cmemory_ = num_mem;

  uint_t n64 = (this->num_creg_bits_ + 63) >> 6;
  if (cregs_.size() != num_matrices_ * n64) {
    cregs_.resize(num_matrices_ * n64);
    cregs_host_.resize(num_matrices_ * n64);
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Deallocate(void) {
  set_device();

  data_.clear();
  data_.shrink_to_fit();
  matrix_.clear();
  matrix_.shrink_to_fit();
  params_.clear();
  params_.shrink_to_fit();
  reduce_buffer_.clear();
  reduce_buffer_.shrink_to_fit();
  probability_buffer_.clear();
  probability_buffer_.shrink_to_fit();
  cregs_.clear();
  cregs_.shrink_to_fit();
  cregs_host_.clear();
  cregs_host_.shrink_to_fit();

  peer_access_.clear();
  num_blocked_gates_.clear();
  num_blocked_matrix_.clear();
  num_blocked_qubits_.clear();
  blocked_qubits_holder_.clear();

#ifdef AER_THRUST_GPU
  for (int i = 0; i < stream_.size(); i++)
    cudaStreamDestroy(stream_[i]);
  stream_.clear();
#endif
  ChunkContainer<data_t>::deallocate_chunks();
}

template <typename data_t>
void DeviceChunkContainer<data_t>::calculate_matrix_buffer_size(int bits,
                                                                int shots) {
  uint_t size;

  // matrix buffer size
  this->matrix_bits_ = bits;
  // adjust matrix_bits_ so that all shots can be stored on GPU
  if (shots > 1) {
    if (shots > AER_MAX_SAMPLING_SHOTS)
      shots = AER_MAX_SAMPLING_SHOTS;
    uint_t b = this->matrix_bits_;
    while ((1ull << (b * 2)) < (uint_t)shots) {
      b++;
    }
    this->matrix_bits_ = b;
  }
  size = 1ull << (this->matrix_bits_ * 2);

  if (max_blocked_gates_ * 4 > size) {
    size = max_blocked_gates_ * 4;
  }
  matrix_buffer_size_ = size;

  // param buffer size
  size = bits * 3;
  uint_t qb = this->num_qubits_ * 4;
  if (this->density_matrix_)
    qb /= 2;
  if (size < qb)
    size = qb;
  if (size < 4)
    size = 4;
  if (QV_MAX_REGISTERS + max_blocked_gates_ * 4 > size) {
    size = QV_MAX_REGISTERS + max_blocked_gates_ * 4;
  }
  params_buffer_size_ = size;

  if (shots > 1 && params_buffer_size_ < (uint_t)shots) {
    params_buffer_size_ = shots;
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::ResizeMatrixBuffers(int bits,
                                                       int max_shots) {
  uint_t n = num_matrices_ + this->num_buffers_;

  if ((uint_t)bits != this->matrix_bits_) {
    calculate_matrix_buffer_size(bits, max_shots);
  }

  if (matrix_.size() < n * matrix_buffer_size_)
    matrix_.resize(n * matrix_buffer_size_);

  if (params_.size() < n * params_buffer_size_)
    params_.resize(n * params_buffer_size_);
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreMatrix(
    const std::vector<std::complex<double>> &mat, uint_t iChunk) const {
  set_device();

#ifdef AER_THRUST_GPU
  cudaMemcpyAsync(matrix_pointer(iChunk), &mat[0],
                  mat.size() * sizeof(thrust::complex<double>),
                  cudaMemcpyHostToDevice, stream(iChunk));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream str;
    str << "ChunkContainer::StoreMatrix size = " << mat.size()
        << " iChunk = " << iChunk << " : " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }
#else
  uint_t offset;
  if (iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunks_) * matrix_buffer_size_;
  else {
    if (num_matrices_ == 1)
      offset = 0;
    else
      offset = iChunk * matrix_buffer_size_;
  }
  thrust::copy_n(mat.begin(), mat.size(), matrix_.begin() + offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreMatrix(const std::complex<double> *mat,
                                               uint_t iChunk,
                                               uint_t size) const {
  set_device();

#ifdef AER_THRUST_GPU
  cudaMemcpyAsync(matrix_pointer(iChunk), mat,
                  size * sizeof(thrust::complex<double>),
                  cudaMemcpyHostToDevice, stream(iChunk));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream str;
    str << "ChunkContainer::StoreMatrix size = " << size
        << " iChunk = " << iChunk << " : " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  uint_t offset;
  if (iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunks_) * matrix_buffer_size_;
  else {
    if (num_matrices_ == 1)
      offset = 0;
    else
      offset = iChunk * matrix_buffer_size_;
  }
  thrust::copy_n(mat, mat + size, matrix_.begin() + offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreUintParams(
    const std::vector<uint_t> &prm, uint_t iChunk) const {
  set_device();

#ifdef AER_THRUST_GPU
  cudaMemcpyAsync(param_pointer(iChunk), &prm[0], prm.size() * sizeof(uint_t),
                  cudaMemcpyHostToDevice, stream(iChunk));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream str;
    str << "ChunkContainer::StoreUintParams size = " << prm.size() << "/"
        << params_.size() << " iChunk = " << iChunk << " : "
        << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }
#else
  uint_t offset;
  if (iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunks_) * params_buffer_size_;
  else {
    if (num_matrices_ == 1)
      offset = 0;
    else
      offset = iChunk * params_buffer_size_;
  }
  thrust::copy_n(prm.begin(), prm.size(), params_.begin() + offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyIn(Chunk<data_t> &src, uint_t iChunk) {
  uint_t size = 1ull << this->chunk_bits_;
#ifdef AER_THRUST_GPU
  if (src.device() >= 0) {
    if (peer_access(src.device())) {
      cudaMemcpyAsync(chunk_pointer(iChunk), src.pointer(),
                      size * sizeof(thrust::complex<data_t>),
                      cudaMemcpyDeviceToDevice, stream(iChunk));
    } else {
      cudaMemcpyPeerAsync(chunk_pointer(iChunk), device_id_, src.pointer(),
                          src.device(), size, stream(iChunk));
    }
  } else {
    cudaMemcpyAsync(chunk_pointer(iChunk), src.pointer(),
                    size * sizeof(thrust::complex<data_t>),
                    cudaMemcpyHostToDevice, stream(iChunk));
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream str;
    str << "DeviceChunkContainer::CopyIn : " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  thrust::copy_n(src.pointer(), size,
                 data_.begin() + (iChunk << this->chunk_bits_));
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyOut(Chunk<data_t> &dest, uint_t iChunk) {
  uint_t size = 1ull << this->chunk_bits_;

#ifdef AER_THRUST_GPU
  if (dest.device() >= 0) {
    if (peer_access(dest.device())) {
      cudaMemcpyAsync(dest.pointer(), chunk_pointer(iChunk),
                      size * sizeof(thrust::complex<data_t>),
                      cudaMemcpyDeviceToDevice, stream(iChunk));
    } else {
      cudaMemcpyPeerAsync(dest.pointer(), dest.device(), chunk_pointer(iChunk),
                          device_id_, size, stream(iChunk));
    }
  } else {
    cudaMemcpyAsync(dest.pointer(), chunk_pointer(iChunk),
                    size * sizeof(thrust::complex<data_t>),
                    cudaMemcpyDeviceToHost, stream(iChunk));
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream str;
    str << "DeviceChunkContainer::CopyOut " << dest.pointer() << ","
        << chunk_pointer(iChunk) << " size = " << size << " : "
        << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_), size,
                 dest.pointer());
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyIn(thrust::complex<data_t> *src,
                                          uint_t iChunk, uint_t size) {
  uint_t this_size = 1ull << this->chunk_bits_;
  if (this_size < size) {
    std::stringstream str;
    str << "DeviceChunkContainer::CopyIn chunk size " << this_size
        << " is less than " << size;
    throw std::runtime_error(str.str());
  }

  synchronize(iChunk);
  thrust::copy_n(src, size, data_.begin() + (iChunk << this->chunk_bits_));
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyOut(thrust::complex<data_t> *dest,
                                           uint_t iChunk, uint_t size) {
  uint_t this_size = 1ull << this->chunk_bits_;
  if (this_size < size) {
    std::stringstream str;
    str << "DeviceChunkContainer::CopyOut chunk size " << this_size
        << " is less than " << size;
    throw std::runtime_error(str.str());
  }
  synchronize(iChunk);
  thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_), size, dest);
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Swap(Chunk<data_t> &src, uint_t iChunk,
                                        uint_t dest_offset, uint_t src_offset,
                                        uint_t size_in, bool write_back) {
  uint_t size = size_in;
  if (size == 0)
    size = 1ull << this->chunk_bits_;

  set_device();
#ifdef AER_THRUST_GPU
  if (src.device() >= 0) {
    if (peer_access(src.device())) {
      this->Execute(BufferSwap_func<data_t>(chunk_pointer(iChunk) + dest_offset,
                                            src.pointer() + src_offset, size,
                                            write_back),
                    iChunk, 0, 1);
    } else {
      thrust::complex<data_t> *pBuffer = buffer_pointer();
      thrust::complex<data_t> *pSrc = src.pointer();
      cudaMemcpyPeerAsync(pBuffer + dest_offset, device_id_, pSrc + src_offset,
                          src.device(), size * sizeof(thrust::complex<data_t>),
                          stream(iChunk));
      this->Execute(BufferSwap_func<data_t>(chunk_pointer(iChunk) + dest_offset,
                                            pBuffer + dest_offset, size, true),
                    iChunk, 0, 1);
      cudaMemcpyPeerAsync(
          pSrc + src_offset, src.device(), pBuffer + dest_offset, device_id_,
          size * sizeof(thrust::complex<data_t>), stream(iChunk));
    }
  } else {
    thrust::complex<data_t> *pBuffer = buffer_pointer();
    thrust::complex<data_t> *pSrc = src.pointer();
    cudaMemcpyAsync(pBuffer + dest_offset, pSrc + src_offset,
                    size * sizeof(thrust::complex<data_t>),
                    cudaMemcpyHostToDevice, stream(this->num_chunks_));
    this->Execute(BufferSwap_func<data_t>(chunk_pointer(iChunk) + dest_offset,
                                          pBuffer + dest_offset, size, true),
                  iChunk, 0, 1);
    cudaMemcpyAsync(pSrc + src_offset, pBuffer + dest_offset,
                    size * sizeof(thrust::complex<data_t>),
                    cudaMemcpyDeviceToHost, stream(this->num_chunks_));
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream str;
    str << "DeviceChunkContainer::Swap : " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  thrust::swap_ranges(thrust::device, chunk_pointer(iChunk) + dest_offset,
                      chunk_pointer(iChunk) + dest_offset + size,
                      src.pointer() + src_offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Zero(uint_t iChunk, uint_t count) {
  set_device();
#ifdef AER_THRUST_GPU
  thrust::fill_n(thrust_gpu::par.on(stream(iChunk)),
                 data_.begin() + (iChunk << this->chunk_bits_), count, 0.0);
#else
  if (this->omp_threads_ > 1)
    thrust::fill_n(thrust::device,
                   data_.begin() + (iChunk << this->chunk_bits_), count, 0.0);
  else
    thrust::fill_n(thrust::seq, data_.begin() + (iChunk << this->chunk_bits_),
                   count, 0.0);
#endif
}

template <typename data_t>
reg_t DeviceChunkContainer<data_t>::sample_measure(
    uint_t iChunk, const std::vector<double> &rnds, uint_t stride, bool dot,
    uint_t count) const {
  const int_t SHOTS = rnds.size();
  reg_t samples(SHOTS, 0);

  set_device();

  strided_range<thrust::complex<data_t> *> iter(
      chunk_pointer(iChunk), chunk_pointer(iChunk + count), stride);

#ifdef AER_THRUST_GPU

  if (dot)
    thrust::transform_inclusive_scan(thrust_gpu::par.on(stream(iChunk)),
                                     iter.begin(), iter.end(), iter.begin(),
                                     complex_dot_scan<data_t>(),
                                     thrust::plus<thrust::complex<data_t>>());
  else
    thrust::inclusive_scan(thrust_gpu::par.on(stream(iChunk)), iter.begin(),
                           iter.end(), iter.begin(),
                           thrust::plus<thrust::complex<data_t>>());

  uint_t i, nshots, size;
  uint_t iBuf = 0;
  if (multi_shots_ && count == 1) {
    iBuf = iChunk;
    size = matrix_buffer_size_ * 2;
    if (size > params_buffer_size_)
      size = params_buffer_size_;
  } else {
    size = matrix_.size() * 2;
    if (size > params_.size())
      size = params_.size();
  }

  double *pRnd = (double *)matrix_pointer(iBuf);
  uint_t *pSmp = param_pointer(iBuf);
  thrust::device_ptr<double> rnd_dev_ptr = thrust::device_pointer_cast(pRnd);

  for (i = 0; i < SHOTS; i += size) {
    nshots = size;
    if (i + nshots > SHOTS)
      nshots = SHOTS - i;

    cudaMemcpyAsync(pRnd, &rnds[i], nshots * sizeof(double),
                    cudaMemcpyHostToDevice, stream(iChunk));

    thrust::lower_bound(thrust_gpu::par.on(stream(iChunk)), iter.begin(),
                        iter.end(), rnd_dev_ptr, rnd_dev_ptr + nshots,
                        params_.begin() + (iBuf * params_buffer_size_),
                        complex_less<data_t>());

    cudaMemcpyAsync(&samples[i], pSmp, nshots * sizeof(uint_t),
                    cudaMemcpyDeviceToHost, stream(iChunk));
  }
  cudaStreamSynchronize(stream(iChunk));
#else
  if (this->omp_threads_ > 1) {
    if (dot)
      thrust::transform_inclusive_scan(thrust::device, iter.begin(), iter.end(),
                                       iter.begin(), complex_dot_scan<data_t>(),
                                       thrust::plus<thrust::complex<data_t>>());
    else
      thrust::inclusive_scan(thrust::device, iter.begin(), iter.end(),
                             iter.begin(),
                             thrust::plus<thrust::complex<data_t>>());

    thrust::lower_bound(thrust::device, iter.begin(), iter.end(), rnds.begin(),
                        rnds.begin() + SHOTS, samples.begin(),
                        complex_less<data_t>());
  } else {
    if (dot)
      thrust::transform_inclusive_scan(thrust::seq, iter.begin(), iter.end(),
                                       iter.begin(), complex_dot_scan<data_t>(),
                                       thrust::plus<thrust::complex<data_t>>());
    else
      thrust::inclusive_scan(thrust::seq, iter.begin(), iter.end(),
                             iter.begin(),
                             thrust::plus<thrust::complex<data_t>>());

    thrust::lower_bound(thrust::seq, iter.begin(), iter.end(), rnds.begin(),
                        rnds.begin() + SHOTS, samples.begin(),
                        complex_less<data_t>());
  }
#endif

  return samples;
}

// set qubits to be blocked
template <typename data_t>
void DeviceChunkContainer<data_t>::set_blocked_qubits(uint_t iChunk,
                                                      const reg_t &qubits) {
  if (num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_) {
    // only the first chunk can store (multi-chunk mode)
    return;
  }
  uint_t iBlock;
  if (iChunk >= this->num_chunks_) { // for buffer chunks
    iBlock = num_matrices_ + iChunk - this->num_chunks_;
  } else {
    iBlock = iChunk;
  }

  if (num_blocked_gates_[iBlock] > 0) {
    apply_blocked_gates(iChunk);
  }

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  uint_t i;
  for (i = 0; i < qubits.size(); i++) {
    blocked_qubits_holder_[iBlock * QV_MAX_REGISTERS + i] = qubits_sorted[i];
  }
#ifdef AER_THRUST_GPU
  set_device();
  cudaMemcpyAsync(param_pointer(iChunk), (uint_t *)&qubits_sorted[0],
                  qubits.size() * sizeof(uint_t), cudaMemcpyHostToDevice,
                  stream(iChunk));
#endif

  num_blocked_gates_[iBlock] = 0;
  num_blocked_matrix_[iBlock] = 0;
  num_blocked_qubits_[iBlock] = qubits.size();
}

template <typename data_t>
class GeneralMatrixMult2x2 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0_, m1_, m2_, m3_;
  uint_t mask_;
  uint_t offset_;

public:
  GeneralMatrixMult2x2(const cvector_t<double> &mat, int q, uint_t mask) {
    m0_ = mat[0];
    m1_ = mat[1];
    m2_ = mat[2];
    m3_ = mat[3];

    mask_ = mask;

    offset_ = 1ull << q;
  }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec;

    vec = this->data_;

    i1 = i & (offset_ - 1);
    i0 = (i - i1) << 1;
    i0 += i1;
    i1 = i0 + offset_;

    q0 = vec[i0];
    q1 = vec[i1];

    if ((i0 & mask_) == mask_)
      vec[i0] = m0_ * q0 + m2_ * q1;
    if ((i1 & mask_) == mask_)
      vec[i1] = m1_ * q0 + m3_ * q1;
  }
  const char *name(void) { return "general_mult2x2"; }
};

// queue gate for blocked execution
template <typename data_t>
void DeviceChunkContainer<data_t>::queue_blocked_gate(
    uint_t iChunk, char gate, uint_t qubit, uint_t mask,
    const std::complex<double> *pMat) {
  if (num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_) {
    // only the first chunk can store (multi-chunk mode)
    return;
  }

  cvector_t<double> mat(4, 0.0);
  uint_t iBlock;
  if (iChunk >= this->num_chunks_) { // for buffer chunks
    iBlock = num_matrices_ + iChunk - this->num_chunks_;
  } else {
    iBlock = iChunk;
  }

  if (num_blocked_gates_[iBlock] >= max_blocked_gates_) {
    apply_blocked_gates(iChunk);
  }

#ifdef AER_THRUST_GPU
  BlockedGateParams params;

  params.mask_ = mask;
  params.gate_ = gate;
  params.qubit_ = 0;
  for (uint_t i = 0; i < num_blocked_qubits_[iBlock]; i++) {
    if (blocked_qubits_holder_[iBlock * QV_MAX_REGISTERS + i] == qubit) {
      params.qubit_ = i;
      break;
    }
  }
  set_device();
  cudaMemcpyAsync((BlockedGateParams *)(param_pointer(iChunk) +
                                        num_blocked_qubits_[iBlock]) +
                      num_blocked_gates_[iBlock],
                  &params, sizeof(BlockedGateParams), cudaMemcpyHostToDevice,
                  stream(iChunk));

  if (pMat != NULL) {
    if (gate == 'd') { // diagonal matrix
      mat[0] = pMat[0];
      mat[1] = pMat[1];
      cudaMemcpyAsync(matrix_pointer(iChunk) + num_blocked_matrix_[iBlock],
                      (thrust::complex<double> *)&mat[0],
                      2 * sizeof(thrust::complex<double>),
                      cudaMemcpyHostToDevice, stream(iChunk));
      num_blocked_matrix_[iBlock] += 2;
    } else if (gate == 'p') { // phase
      mat[0] = pMat[0];
      cudaMemcpyAsync(matrix_pointer(iChunk) + num_blocked_matrix_[iBlock],
                      (thrust::complex<double> *)&mat[0],
                      1 * sizeof(thrust::complex<double>),
                      cudaMemcpyHostToDevice, stream(iChunk));
      num_blocked_matrix_[iBlock] += 1;
    } else { // otherwise, 2x2 matrix
      mat[0] = pMat[0];
      mat[1] = pMat[2];
      mat[2] = pMat[3];
      mat[3] = pMat[1];
      cudaMemcpyAsync(matrix_pointer(iChunk) + num_blocked_matrix_[iBlock],
                      (thrust::complex<double> *)&mat[0],
                      4 * sizeof(thrust::complex<double>),
                      cudaMemcpyHostToDevice, stream(iChunk));
      num_blocked_matrix_[iBlock] += 4;
    }
  }
  num_blocked_gates_[iBlock]++;

#else
  // for statevector_cpu, apply now

  switch (gate) {
  case 'x':
    mat[1] = 1.0;
    mat[2] = 1.0;
    break;
  case 'y':
    mat[1] = std::complex<double>(0.0, -1.0);
    mat[2] = std::complex<double>(0.0, 1.0);
    break;
  case 'p':
    if (pMat) {
      mat[0] = pMat[0];
      mat[3] = pMat[0];
    }
    break;
  case 'd':
    if (pMat) {
      mat[0] = pMat[0];
      mat[3] = pMat[1];
    }
    break;
  default:
    if (pMat) {
      mat[0] = pMat[0];
      mat[1] = pMat[1];
      mat[2] = pMat[2];
      mat[3] = pMat[3];
    }
    break;
  }
  uint_t count = 1;
  if (num_matrices_ == 1)
    count = this->chunk_bits_;
  this->Execute(GeneralMatrixMult2x2<data_t>(mat, qubit, mask), iChunk, 0,
                count);

#endif
}

#ifdef AER_THRUST_GPU

template <typename data_t>
__global__ void
dev_apply_register_blocked_gates(thrust::complex<data_t> *data, int num_gates,
                                 int num_qubits, int num_matrix, uint_t *qubits,
                                 BlockedGateParams *params,
                                 thrust::complex<double> *matrix) {
  uint_t i, idx, ii, t, offset;
  uint_t j, laneID, iPair;
  thrust::complex<data_t> q, qp, qt;
  thrust::complex<double> m0, m1;
  data_t qr, qi;
  int nElem;
  thrust::complex<double> *matrix_load;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  laneID = i & (_WS - 1);

  // index for this thread
  idx = 0;
  ii = i >> num_qubits;
  for (j = 0; j < num_qubits; j++) {
    offset = (1ull << qubits[j]);
    t = ii & (offset - 1);
    idx += t;
    ii = (ii - t) << 1;

    if (((laneID >> j) & 1) != 0) {
      idx += offset;
    }
  }
  idx += ii;

  q = data[idx];

  // prefetch
  if (threadIdx.x < num_matrix)
    m0 = matrix[threadIdx.x];

  for (j = 0; j < num_gates; j++) {
    iPair = laneID ^ (1ull << params[j].qubit_);

    matrix_load = matrix;
    nElem = 0;

    switch (params[j].gate_) {
    case 'x':
      m0 = 0.0;
      m1 = 1.0;
      break;
    case 'y':
      m0 = 0.0;
      if (iPair > laneID)
        m1 = thrust::complex<double>(0.0, -1.0);
      else
        m1 = thrust::complex<double>(0.0, 1.0);
      break;
    case 'p':
      nElem = 1;
      matrix += 1;
      m1 = 0.0;
      break;
    case 'd':
      nElem = 2;
      matrix += 2;
      m1 = 0.0;
      break;
    default:
      nElem = 4;
      matrix += 4;
      break;
    }

    if (iPair < laneID) {
      matrix_load += (nElem >> 1);
    }
    if (nElem > 0)
      m0 = *(matrix_load);
    if (nElem > 2)
      m1 = *(matrix_load + 1);

    // warp shuffle to get pair amplitude
    qr = __shfl_sync(0xffffffff, q.real(), iPair, 32);
    qi = __shfl_sync(0xffffffff, q.imag(), iPair, 32);
    qp = thrust::complex<data_t>(qr, qi);
    qt = m0 * q + m1 * qp;

    if ((idx & params[j].mask_) == params[j].mask_) { // handling control bits
      q = qt;
    }
  }

  data[idx] = q;
}

template <typename data_t>
__global__ void
dev_apply_shared_memory_blocked_gates(thrust::complex<data_t> *data,
                                      int num_gates, int num_qubits,
                                      uint_t *qubits, BlockedGateParams *params,
                                      thrust::complex<double> *matrix) {
  __shared__ thrust::complex<data_t> buf[1024];
  uint_t i, idx, ii, t, offset;
  uint_t j, laneID, iPair;
  thrust::complex<data_t> q, qp;
  thrust::complex<double> m0, m1;
  data_t qr, qi;
  int nElem;
  thrust::complex<double> *matrix_load;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  laneID = threadIdx.x;

  // index for this thread
  idx = 0;
  ii = i >> num_qubits;
  for (j = 0; j < num_qubits; j++) {
    offset = (1ull << qubits[j]);
    t = ii & (offset - 1);
    idx += t;
    ii = (ii - t) << 1;

    if (((laneID >> j) & 1) != 0) {
      idx += offset;
    }
  }
  idx += ii;

  q = data[idx];

  for (j = 0; j < num_gates; j++) {
    iPair = laneID ^ (1ull << params[j].qubit_);

    if (params[j].qubit_ < 5) {
      // warp shuffle to get pair amplitude
      qr = q.real();
      qi = q.imag();
      qr = __shfl_sync(0xffffffff, qr, iPair & (_WS - 1), 32);
      qi = __shfl_sync(0xffffffff, qi, iPair & (_WS - 1), 32);
      qp = thrust::complex<data_t>(qr, qi);
    } else {
      __syncthreads();
      buf[laneID] = q;
      __syncthreads();
      qp = buf[iPair];
    }

    matrix_load = matrix;
    nElem = 0;

    switch (params[j].gate_) {
    case 'x':
      m0 = 0.0;
      m1 = 1.0;
      break;
    case 'y':
      m0 = 0.0;
      if (iPair > laneID)
        m1 = thrust::complex<double>(0.0, -1.0);
      else
        m1 = thrust::complex<double>(0.0, 1.0);
      break;
    case 'p':
      nElem = 1;
      matrix += 1;
      m1 = 0.0;
      break;
    case 'd':
      nElem = 2;
      matrix += 2;
      m1 = 0.0;
      break;
    default:
      nElem = 4;
      matrix += 4;
      break;
    }

    if (iPair < laneID) {
      matrix_load += (nElem >> 1);
    }
    if (nElem > 0)
      m0 = *(matrix_load);
    if (nElem > 2)
      m1 = *(matrix_load + 1);

    if ((idx & params[j].mask_) == params[j].mask_) { // handling control bits
      q = m0 * q + m1 * qp;
    }
  }

  data[idx] = q;
}

#endif

// do all gates stored in queue
template <typename data_t>
void DeviceChunkContainer<data_t>::apply_blocked_gates(uint_t iChunk) {
  if (num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_) {
    // only the first chunk can apply
    return;
  }
  uint_t iBlock;
  if (iChunk >= this->num_chunks_) { // for buffer chunks
    iBlock = num_matrices_ + iChunk - this->num_chunks_;
  } else {
    iBlock = iChunk;
  }

  if (num_blocked_gates_[iBlock] == 0)
    return;

#ifdef AER_THRUST_GPU

  uint_t size;
  uint_t *pQubits;
  BlockedGateParams *pParams;
  thrust::complex<double> *pMatrix;

  set_device();

  pQubits = param_pointer(iChunk);
  pParams = (BlockedGateParams *)(param_pointer(iChunk) +
                                  num_blocked_qubits_[iBlock]);
  pMatrix = matrix_pointer(iChunk);

  if (num_matrices_ == 1) {
    size = this->num_chunks_ << this->chunk_bits_;
  } else {
    size = 1ull << this->chunk_bits_;
  }
  uint_t nt, nb;
  nt = size;
  nb = 1;
  if (nt > 1024) {
    nb = (nt + 1024 - 1) / 1024;
    nt = 1024;
  }

  if (num_blocked_qubits_[iBlock] < 6) {
    // using register blocking (<=5 qubits)
    dev_apply_register_blocked_gates<data_t>
        <<<nb, nt,
           num_blocked_matrix_[iChunk] * sizeof(thrust::complex<double>),
           stream(iChunk)>>>(chunk_pointer(iChunk), num_blocked_gates_[iBlock],
                             num_blocked_qubits_[iBlock],
                             num_blocked_matrix_[iBlock], pQubits, pParams,
                             pMatrix);
  } else {
    // using shared memory blocking (<=10 qubits)
    dev_apply_shared_memory_blocked_gates<data_t>
        <<<nb, nt, 1024 * sizeof(thrust::complex<data_t>), stream(iChunk)>>>(
            chunk_pointer(iChunk), num_blocked_gates_[iBlock],
            num_blocked_qubits_[iBlock], pQubits, pParams, pMatrix);
  }

#endif

  num_blocked_gates_[iBlock] = 0;
  num_blocked_matrix_[iBlock] = 0;
}

template <typename data_t>
void DeviceChunkContainer<data_t>::copy_to_probability_buffer(
    std::vector<double> &buf, int pos) {
#ifdef AER_THRUST_GPU
  set_device();
  cudaMemcpyAsync(probability_buffer(0) + pos * this->num_chunks_, &buf[0],
                  buf.size() * sizeof(double), cudaMemcpyHostToDevice,
                  stream_[0]);
#else
  thrust::copy_n(buf.begin(), buf.size(), probability_buffer_.begin());
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::copy_reduce_buffer(std::vector<double> &ret,
                                                      uint_t iChunk,
                                                      uint_t num_val) const {
  uint_t count = ret.size();
  std::vector<double> tmp(count * reduce_buffer_size_);
#ifdef AER_THRUST_CUDA
  set_device();
  cudaMemcpyAsync(&tmp[0], reduce_buffer(iChunk),
                  reduce_buffer_size_ * count * sizeof(double),
                  cudaMemcpyDeviceToHost, stream(iChunk));
  cudaStreamSynchronize(stream(iChunk));
#else
  thrust::copy_n(reduce_buffer_.begin() + iChunk * reduce_buffer_size_,
                 count * reduce_buffer_size_, tmp.begin());
#endif

  for (uint_t i = 0; i < count; i++) {
    for (uint_t j = 0; j < num_val; j++)
      ret[i * num_val + j] = tmp[i * reduce_buffer_size_ + j];
  }
}

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
