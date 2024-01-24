/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_chunk_hpp_
#define _qv_chunk_hpp_

#include "simulators/statevector/chunk/device_chunk_container.hpp"
#include "simulators/statevector/chunk/host_chunk_container.hpp"

#ifdef AER_CUSTATEVEC
#include "simulators/statevector/chunk/cuStateVec_chunk_container.hpp"
#endif

namespace AER {
namespace QV {
namespace Chunk {

//============================================================================
// chunk class
//============================================================================
template <typename data_t>
class Chunk {
protected:
  mutable std::weak_ptr<ChunkContainer<data_t>>
      chunk_container_;                  // pointer to chunk container
  std::shared_ptr<Chunk<data_t>> cache_; // pointer to cache chunk on device
  uint_t chunk_pos_;                     // position in container
  int place_;                            // container ID
  uint_t num_qubits_;                    // total number of qubits
  uint_t chunk_index_;                   // global chunk index
  bool mapped_;                          // mapped to qubitvector
public:
  Chunk() {
    chunk_pos_ = 0;
    place_ = -1;
    num_qubits_ = 0;
    chunk_index_ = 0;
    mapped_ = false;
    cache_ = nullptr;
  }

  Chunk(std::weak_ptr<ChunkContainer<data_t>> cc, uint_t pos) {
    chunk_container_ = cc;
    chunk_pos_ = pos;
    place_ = chunk_container_.lock()->place();
    num_qubits_ = 0;
    chunk_index_ = 0;
    mapped_ = false;
    cache_ = nullptr;
  }
  Chunk(Chunk<data_t>
            &chunk) // map chunk from exisiting chunk (used fo cache chunk)
  {
    chunk_container_ = chunk.chunk_container_;
    chunk_pos_ = chunk.chunk_pos_;
    place_ = chunk.place_;
    num_qubits_ = chunk.num_qubits_;
    chunk_index_ = chunk.chunk_index_;
    mapped_ = true;
    cache_ = nullptr;
  }
  ~Chunk() {
    if (cache_)
      cache_.reset();
  }

  void set_device(void) const { chunk_container_.lock()->set_device(); }
  int device(void) { return chunk_container_.lock()->device(); }

  std::shared_ptr<ChunkContainer<data_t>> container() {
    return chunk_container_.lock();
  }

  void map(std::weak_ptr<ChunkContainer<data_t>> cc, uint_t pos) {
    chunk_container_ = cc;
    chunk_pos_ = pos;
    place_ = chunk_container_.lock()->place();
    mapped_ = true;
  }

  uint_t pos(void) { return chunk_pos_; }
  int place(void) { return place_; }
  void set_place(int ip) { place_ = ip; }
  void map_cache(Chunk<data_t> &chunk) {
    cache_ = std::make_shared<Chunk<data_t>>(chunk);
  }
  void unmap_cache(void) {
    if (cache_) {
      cache_->unmap();
      cache_.reset();
      cache_ = nullptr;
    }
  }

  bool is_mapped(void) { return mapped_; }
  void unmap(void) {
    mapped_ = false;
    if (cache_)
      unmap_cache();
  }

  void set_num_qubits(uint_t qubits) { num_qubits_ = qubits; }
  void set_chunk_index(uint_t id) { chunk_index_ = id; }
  uint_t chunk_index(void) { return chunk_index_; }

  uint_t matrix_bits(void) { return chunk_container_.lock()->matrix_bits(); }

  void Set(uint_t i, const thrust::complex<data_t> &t) {
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->synchronize(chunk_pos_);
    sel_chunk_container->Set(
        i + (chunk_pos_ << sel_chunk_container->chunk_bits()), t);
  }
  thrust::complex<data_t> Get(uint_t i) const {
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->synchronize(chunk_pos_);
    return sel_chunk_container->Get(
        i + (chunk_pos_ << sel_chunk_container->chunk_bits()));
  }

  thrust::complex<data_t> &operator[](uint_t i) {
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->synchronize(chunk_pos_);
    return (*sel_chunk_container)[i + (chunk_pos_
                                       << sel_chunk_container->chunk_bits())];
  }

  thrust::complex<data_t> *pointer(void) {
    return chunk_container_.lock()->chunk_pointer(chunk_pos_);
  }
  thrust::complex<data_t> *buffer_pointer(void) {
    return chunk_container_.lock()->buffer_pointer();
  }

  void StoreMatrix(const std::vector<std::complex<double>> &mat) {
    if (cache_) {
      cache_->StoreMatrix(mat);
    } else {
      chunk_container_.lock()->StoreMatrix(mat, chunk_pos_);
    }
  }
  void StoreMatrix(const std::complex<double> *mat, uint_t size) {
    if (cache_) {
      cache_->StoreMatrix(mat, size);
    } else {
      chunk_container_.lock()->StoreMatrix(mat, chunk_pos_, size);
    }
  }
  void StoreUintParams(const std::vector<uint_t> &prm) {
    if (cache_) {
      cache_->StoreUintParams(prm);
    } else {
      chunk_container_.lock()->StoreUintParams(prm, chunk_pos_);
    }
  }

  void ResizeMatrixBuffers(int bits, int max_shots) {
    // synchronize all kernel execution before changing matrix buffer size
    chunk_container_.lock()->synchronize(chunk_pos_);
    chunk_container_.lock()->ResizeMatrixBuffers(bits, max_shots);
  }

  void CopyIn(Chunk<data_t> &src) {
    chunk_container_.lock()->CopyIn(src, chunk_pos_);
  }
  void CopyOut(Chunk<data_t> &dest) {
    chunk_container_.lock()->CopyOut(dest, chunk_pos_);
  }
  void CopyIn(thrust::complex<data_t> *src, uint_t size) {
    chunk_container_.lock()->CopyIn(src, chunk_pos_, size);
  }
  void CopyOut(thrust::complex<data_t> *dest, uint_t size) {
    chunk_container_.lock()->CopyOut(dest, chunk_pos_, size);
  }
  void Swap(Chunk<data_t> &src, uint_t dest_offset = 0, uint_t src_offset = 0,
            uint_t size = 0, bool write_back = true) {
    chunk_container_.lock()->Swap(src, chunk_pos_, dest_offset, src_offset,
                                  size, write_back);
  }

  template <typename Function>
  void Execute(Function func, uint_t count) {
    if (cache_) {
      cache_->chunk_container_.lock()->Execute(func, cache_->chunk_pos_,
                                               chunk_index_, count);
    } else {
      chunk_container_.lock()->Execute(func, chunk_pos_, chunk_index_, count);
    }
  }

  template <typename Function>
  void ExecuteSum(double *pSum, Function func, uint_t count) const {
    if (cache_) {
      cache_->ExecuteSum(pSum, func, count);
    } else {
      chunk_container_.lock()->ExecuteSum(pSum, func, chunk_pos_, count);
    }
  }

  template <typename Function>
  void ExecuteSum2(double *pSum, Function func, uint_t count) const {
    if (cache_) {
      cache_->ExecuteSum2(pSum, func, count);
    } else {
      chunk_container_.lock()->ExecuteSum2(pSum, func, chunk_pos_, count);
    }
  }

  reg_t sample_measure(const std::vector<double> &rnds, uint_t stride = 1,
                       bool dot = true, uint_t count = 1) const {
    return chunk_container_.lock()->sample_measure(chunk_pos_, rnds, stride,
                                                   dot, count);
  }

  double norm(uint_t count) const {
    return chunk_container_.lock()->norm(chunk_pos_, count);
  }
  double trace(uint_t row, uint_t count) const {
    return chunk_container_.lock()->trace(chunk_pos_, row, count);
  }

#ifdef AER_THRUST_GPU
  cudaStream_t stream(void) {
    return std::static_pointer_cast<DeviceChunkContainer<data_t>>(
               chunk_container_.lock())
        ->stream(chunk_pos_);
  }
#endif

  thrust::complex<double> *matrix_pointer(void) {
    return chunk_container_.lock()->matrix_pointer(chunk_pos_);
  }
  uint_t *param_pointer(void) {
    return chunk_container_.lock()->param_pointer(chunk_pos_);
  }
  double *reduce_buffer(void) {
    if (cache_) {
      return cache_->reduce_buffer();
    }
    return chunk_container_.lock()->reduce_buffer(chunk_pos_);
  }
  uint_t reduce_buffer_size(void) {
    if (cache_) {
      return cache_->reduce_buffer_size();
    }
    return chunk_container_.lock()->reduce_buffer_size();
  }
  double *probability_buffer(void) {
    if (cache_) {
      return cache_->probability_buffer();
    }
    return chunk_container_.lock()->probability_buffer(chunk_pos_);
  }
  void copy_reduce_buffer(std::vector<double> &ret, uint_t num_val) const {
    if (cache_) {
      return cache_->copy_reduce_buffer(ret, num_val);
    }
    return chunk_container_.lock()->copy_reduce_buffer(ret, chunk_pos_,
                                                       num_val);
  }

  void synchronize(void) const {
    if (cache_) {
      cache_->synchronize();
    } else {
      chunk_container_.lock()->synchronize(chunk_pos_);
    }
  }

  // set qubits to be blocked
  void set_blocked_qubits(const reg_t &qubits) {
    chunk_container_.lock()->set_blocked_qubits(chunk_pos_, qubits);
  }

  // do all gates stored in queue
  void apply_blocked_gates(void) {
    chunk_container_.lock()->apply_blocked_gates(chunk_pos_);
  }

  // queue gate for blocked execution
  void queue_blocked_gate(char gate, uint_t qubit, uint_t mask,
                          const std::complex<double> *pMat = nullptr) {
    chunk_container_.lock()->queue_blocked_gate(chunk_pos_, gate, qubit, mask,
                                                pMat);
  }

  int measured_cbit(int qubit) {
    return chunk_container_.lock()->measured_cbit(chunk_pos_, qubit);
  }

  void set_conditional(int_t bit) {
    // top chunk only sets conditional bit
    if (chunk_pos_ == 0)
      chunk_container_.lock()->set_conditional(bit);
  }
  int_t get_conditional(void) {
    return chunk_container_.lock()->get_conditional();
  }
  void keep_conditional(bool keep) {
    // top chunk only sets conditional bit
    if (chunk_pos_ == 0)
      chunk_container_.lock()->keep_conditional(keep);
  }

  // apply matrix
  void apply_matrix(const reg_t &qubits, const int_t control_bits,
                    const cvector_t<double> &mat, const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_matrix(
          cache_->chunk_pos_, qubits, control_bits, mat, chunk_index_, count);
    else
      chunk_container_.lock()->apply_matrix(chunk_pos_, qubits, control_bits,
                                            mat, chunk_index_, count);
  }
  void apply_batched_matrix(const reg_t &qubits, const int_t control_bits,
                            const cvector_t<double> &mat,
                            const uint_t num_shots_per_matrix,
                            const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_batched_matrix(
          cache_->chunk_pos_, qubits, control_bits, mat, num_shots_per_matrix,
          chunk_index_, count);
    else
      chunk_container_.lock()->apply_batched_matrix(
          chunk_pos_, qubits, control_bits, mat, num_shots_per_matrix,
          chunk_index_, count);
  }

  // apply diagonal matrix
  void apply_diagonal_matrix(const reg_t &qubits, const int_t control_bits,
                             const cvector_t<double> &diag,
                             const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_diagonal_matrix(
          cache_->chunk_pos_, qubits, control_bits, diag, chunk_index_, count);
    else
      chunk_container_.lock()->apply_diagonal_matrix(
          chunk_pos_, qubits, control_bits, diag, chunk_index_, count);
  }
  void apply_batched_diagonal_matrix(const reg_t &qubits,
                                     const int_t control_bits,
                                     const cvector_t<double> &diag,
                                     const uint_t num_shots_per_matrix,
                                     const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_batched_diagonal_matrix(
          cache_->chunk_pos_, qubits, control_bits, diag, num_shots_per_matrix,
          chunk_index_, count);
    else
      chunk_container_.lock()->apply_batched_diagonal_matrix(
          chunk_pos_, qubits, control_bits, diag, num_shots_per_matrix,
          chunk_index_, count);
  }

  // apply (controlled) X
  void apply_X(const reg_t &qubits, const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_X(cache_->chunk_pos_, qubits,
                                               chunk_index_, count);
    else
      chunk_container_.lock()->apply_X(chunk_pos_, qubits, chunk_index_, count);
  }
  // apply (controlled) Y
  void apply_Y(const reg_t &qubits, const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_Y(cache_->chunk_pos_, qubits,
                                               chunk_index_, count);
    else
      chunk_container_.lock()->apply_Y(chunk_pos_, qubits, chunk_index_, count);
  }
  // apply (controlled) phase
  void apply_phase(const reg_t &qubits, const int_t control_bits,
                   const std::complex<double> phase, const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_phase(
          cache_->chunk_pos_, qubits, control_bits, phase, chunk_index_, count);
    else
      chunk_container_.lock()->apply_phase(chunk_pos_, qubits, control_bits,
                                           phase, chunk_index_, count);
  }
  // apply (controlled) swap gate
  void apply_swap(const reg_t &qubits, const int_t control_bits,
                  const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_swap(
          cache_->chunk_pos_, qubits, control_bits, chunk_index_, count);
    else
      chunk_container_.lock()->apply_swap(chunk_pos_, qubits, control_bits,
                                          chunk_index_, count);
  }
  // apply multiple swap gates
  void apply_multi_swaps(const reg_t &qubits, const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_multi_swaps(
          cache_->chunk_pos_, qubits, chunk_index_, count);
    else
      chunk_container_.lock()->apply_multi_swaps(chunk_pos_, qubits,
                                                 chunk_index_, count);
  }
  // apply permutation
  void apply_permutation(const reg_t &qubits,
                         const std::vector<std::pair<uint_t, uint_t>> &pairs,
                         const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_permutation(
          cache_->chunk_pos_, qubits, pairs, chunk_index_, count);
    else
      chunk_container_.lock()->apply_permutation(chunk_pos_, qubits, pairs,
                                                 chunk_index_, count);
  }

  // apply rotation around axis
  void apply_rotation(const reg_t &qubits, const Rotation r, const double theta,
                      const uint_t count) {
    if (cache_)
      cache_->chunk_container_.lock()->apply_rotation(
          cache_->chunk_pos_, qubits, r, theta, chunk_index_, count);
    else
      chunk_container_.lock()->apply_rotation(chunk_pos_, qubits, r, theta,
                                              chunk_index_, count);
  }

  // get probabilities of chunk
  void probabilities(std::vector<double> &probs, const reg_t &qubits) const {
    chunk_container_.lock()->probabilities(probs, chunk_pos_, qubits);
  }
  // get norm of matrix multiplication
  double expval_matrix(const reg_t &qubits, const cvector_t<double> &mat,
                       const uint_t count) const {
    return chunk_container_.lock()->expval_matrix(chunk_pos_, qubits, mat,
                                                  count);
  }
  // Pauli expectation values
  double expval_pauli(const reg_t &qubits, const std::string &pauli,
                      const complex_t initial_phase) const {
    return chunk_container_.lock()->expval_pauli(chunk_pos_, qubits, pauli,
                                                 initial_phase);
  }
  void batched_expval_pauli(const uint_t count, const reg_t &qubits,
                            const std::string &pauli, bool variance,
                            std::complex<double> param, bool first,
                            const complex_t initial_phase) const {
    chunk_container_.lock()->batched_expval_pauli(chunk_pos_, count, qubits,
                                                  pauli, variance, param, first,
                                                  initial_phase);
  }
};

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
