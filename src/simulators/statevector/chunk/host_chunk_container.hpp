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

#ifndef _qv_host_chunk_container_hpp_
#define _qv_host_chunk_container_hpp_

#include "simulators/statevector/chunk/chunk_container.hpp"

namespace AER {
namespace QV {
namespace Chunk {

//============================================================================
// host chunk container class
//============================================================================
template <typename data_t>
class HostChunkContainer : public ChunkContainer<data_t> {
protected:
  AERHostVector<thrust::complex<data_t>>
      data_; // host vector for chunks + buffers
  mutable std::vector<thrust::complex<double> *> matrix_; // pointer to matrix
  mutable std::vector<uint_t *> params_; // pointer to additional parameters
public:
  HostChunkContainer() {}
  ~HostChunkContainer();

  uint_t size(void) { return data_.size(); }

  AERHostVector<thrust::complex<data_t>> &vector(void) { return data_; }

  thrust::complex<data_t> &operator[](uint_t i) { return data_[i]; }

  uint_t Allocate(int idev, int chunk_bits, int num_qubits, uint_t chunks,
                  uint_t buffers, bool multi_shots, int matrix_bit,
                  int max_shots, bool density_matrix) override;
  void Deallocate(void) override;

  void StoreMatrix(const std::vector<std::complex<double>> &mat,
                   uint_t iChunk) const override {
    matrix_[iChunk] = (thrust::complex<double> *)&mat[0];
  }
  void StoreMatrix(const std::complex<double> *mat, uint_t iChunk,
                   uint_t size) const override {
    matrix_[iChunk] = (thrust::complex<double> *)mat;
  }

  void StoreUintParams(const std::vector<uint_t> &prm,
                       uint_t iChunk) const override {
    params_[iChunk] = (uint_t *)&prm[0];
  }
  void ResizeMatrixBuffers(int bits, int max_shots) {}

  void Set(uint_t i, const thrust::complex<data_t> &t) override {
    data_[i] = t;
  }
  thrust::complex<data_t> Get(uint_t i) const override { return data_[i]; }

  thrust::complex<data_t> *chunk_pointer(uint_t iChunk) const override {
    return (thrust::complex<data_t> *)thrust::raw_pointer_cast(data_.data()) +
           (iChunk << this->chunk_bits_);
  }
  thrust::complex<data_t> *buffer_pointer(void) const {
    return (thrust::complex<data_t> *)thrust::raw_pointer_cast(data_.data()) +
           (this->num_chunks_ << this->chunk_bits_);
  }

  thrust::complex<double> *matrix_pointer(uint_t iChunk) const override {
    return matrix_[iChunk];
  }

  uint_t *param_pointer(uint_t iChunk) const override {
    return params_[iChunk];
  }

  bool peer_access(int i_dest) {
#ifdef AER_ATS
    // for IBM AC922
    return true;
#else
    return false;
#endif
  }

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
};

template <typename data_t>
HostChunkContainer<data_t>::~HostChunkContainer(void) {
  Deallocate();
}

template <typename data_t>
uint_t HostChunkContainer<data_t>::Allocate(int idev, int chunk_bits,
                                            int num_qubits, uint_t chunks,
                                            uint_t buffers, bool multi_shots,
                                            int matrix_bit, int max_shots,
                                            bool density_matrix) {
  uint_t nc = chunks;

  ChunkContainer<data_t>::chunk_bits_ = chunk_bits;
  ChunkContainer<data_t>::num_qubits_ = num_qubits;
  ChunkContainer<data_t>::density_matrix_ = density_matrix;

  ChunkContainer<data_t>::num_buffers_ = buffers;
  ChunkContainer<data_t>::num_chunks_ = nc;
  if (nc + buffers > 0)
    data_.resize((nc + buffers) << chunk_bits);
  if (nc + buffers > 0) {
    matrix_.resize(nc + buffers);
    params_.resize(nc + buffers);
  }

  // allocate chunk classes
  if (nc + buffers > 0)
    ChunkContainer<data_t>::allocate_chunks();

  return nc;
}

template <typename data_t>
void HostChunkContainer<data_t>::Deallocate(void) {
  data_.clear();
  data_.shrink_to_fit();
  matrix_.clear();
  matrix_.shrink_to_fit();
  params_.clear();
  params_.shrink_to_fit();

  ChunkContainer<data_t>::deallocate_chunks();
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyIn(Chunk<data_t> &src, uint_t iChunk) {
  uint_t size = 1ull << this->chunk_bits_;

  if (src.device() >= 0) {
    src.set_device();
    auto src_cont =
        std::static_pointer_cast<DeviceChunkContainer<data_t>>(src.container());
    thrust::copy_n(src_cont->vector().begin() +
                       (src.pos() << this->chunk_bits_),
                   size, data_.begin() + (iChunk << this->chunk_bits_));
  } else {
    auto src_cont =
        std::static_pointer_cast<HostChunkContainer<data_t>>(src.container());

    thrust::copy_n(src_cont->vector().begin() +
                       (src.pos() << this->chunk_bits_),
                   size, data_.begin() + (iChunk << this->chunk_bits_));
  }
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyOut(Chunk<data_t> &dest, uint_t iChunk) {
  uint_t size = 1ull << this->chunk_bits_;
  if (dest.device() >= 0) {
    dest.set_device();
    auto dest_cont = std::static_pointer_cast<DeviceChunkContainer<data_t>>(
        dest.container());
    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_), size,
                   dest_cont->vector().begin() +
                       (dest.pos() << this->chunk_bits_));
  } else {
    auto dest_cont =
        std::static_pointer_cast<HostChunkContainer<data_t>>(dest.container());

    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_), size,
                   dest_cont->vector().begin() +
                       (dest.pos() << this->chunk_bits_));
  }
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyIn(thrust::complex<data_t> *src,
                                        uint_t iChunk, uint_t size) {
  uint_t this_size = 1ull << this->chunk_bits_;
  if (this_size < size)
    throw std::runtime_error("CopyIn chunk size is less than provided size");

  thrust::copy_n(src, size, data_.begin() + (iChunk << this->chunk_bits_));
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyOut(thrust::complex<data_t> *dest,
                                         uint_t iChunk, uint_t size) {
  uint_t this_size = 1ull << this->chunk_bits_;
  if (this_size < size)
    throw std::runtime_error("CopyIn chunk size is less than provided size");

  thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_), size, dest);
}

template <typename data_t>
void HostChunkContainer<data_t>::Swap(Chunk<data_t> &src, uint_t iChunk,
                                      uint_t dest_offset, uint_t src_offset,
                                      uint_t size_in, bool write_back) {
  uint_t size = size_in;
  if (size == 0)
    size = 1ull << this->chunk_bits_;
  //  if(src.device() >= 0){
  //    src.swap(*this,dest_offset,src_offset,size_in,write_back);
  //  }
  //  else{
  auto src_cont =
      std::static_pointer_cast<HostChunkContainer<data_t>>(src.container());

  this->Execute(BufferSwap_func<data_t>(chunk_pointer(iChunk) + dest_offset,
                                        src.pointer() + src_offset, size,
                                        write_back),
                iChunk, 0, 1);
  //    thrust::swap_ranges(thrust::omp::par,data_.begin() + (iChunk <<
  //    this->chunk_bits_) + dest_offset,data_.begin() + (iChunk <<
  //    this->chunk_bits_) + dest_offset + size,src_cont->vector().begin() +
  //    (src.pos() << this->chunk_bits_) + src_offset);
  //  }
}

template <typename data_t>
void HostChunkContainer<data_t>::Zero(uint_t iChunk, uint_t count) {
#ifndef AER_THRUST_ROCM_DISABLE_THRUST_OMP
  thrust::fill_n(thrust::omp::par,
                 data_.begin() + (iChunk << this->chunk_bits_), count, 0.0);
#endif
}

template <typename data_t>
reg_t HostChunkContainer<data_t>::sample_measure(
    uint_t iChunk, const std::vector<double> &rnds, uint_t stride, bool dot,
    uint_t count) const {
  const int_t SHOTS = rnds.size();
  reg_t samples(SHOTS, 0);
  thrust::host_vector<uint_t> vSmp(SHOTS);
  int i;

  strided_range<thrust::complex<data_t> *> iter(
      chunk_pointer(iChunk), chunk_pointer(iChunk + count), stride);

#ifndef AER_THRUST_ROCM_DISABLE_THRUST_OMP
  if (dot)
    thrust::transform_inclusive_scan(thrust::omp::par, iter.begin(), iter.end(),
                                     iter.begin(), complex_dot_scan<data_t>(),
                                     thrust::plus<thrust::complex<data_t>>());
  else
    thrust::inclusive_scan(thrust::omp::par, iter.begin(), iter.end(),
                           iter.begin(),
                           thrust::plus<thrust::complex<data_t>>());
  thrust::lower_bound(thrust::omp::par, iter.begin(), iter.end(), rnds.begin(),
                      rnds.begin() + SHOTS, vSmp.begin(),
                      complex_less<data_t>());
#endif

  for (i = 0; i < SHOTS; i++) {
    samples[i] = vSmp[i];
  }
  vSmp.clear();

  return samples;
}

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
