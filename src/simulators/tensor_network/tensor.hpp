/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _tensor_hpp_
#define _tensor_hpp_

#include <complex>
#include <vector>

#include "framework/types.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace TensorNetwork {

// class for tensor expression
template <typename data_t>
class Tensor {
protected:
  int base_; // basis of bit, 2 for qubit, 3 for qutrit (not supported yet)
  int rank_; // rank(dimension) of tensor
  std::vector<int64_t> extents_; // number of elements in each rank ( = base_)
  int64_t size_;                 // size of tensor (number of elements)
  std::vector<std::complex<data_t>> tensor_; // tensor elements (row major)
  std::vector<int32_t> modes_;               // indices of connected tensors
  reg_t qubits_;                             // bits for each rank
  bool sp_tensor_;                           // this tensor is superop matrix
public:
  Tensor() {
    rank_ = 0;
    size_ = 0;
    base_ = 2; // qubit for default
    sp_tensor_ = false;
  }
  Tensor(Tensor<data_t> &t);

  ~Tensor() {}

  int rank() { return rank_; }
  std::vector<int64_t> &extents() { return extents_; }
  std::vector<std::complex<data_t>> &tensor() { return tensor_; }
  std::vector<int32_t> &modes() { return modes_; }
  reg_t &qubits() { return qubits_; }

  // create tensor from matrix
  void set(const reg_t &qubits, std::vector<std::complex<data_t>> &mat);
  void set(int qubit, std::vector<std::complex<data_t>> &mat);
  void set(const reg_t &qubits, std::complex<data_t> *mat, uint64_t size);

  // create conjugate tensor
  void set_conj(const reg_t &qubits, std::vector<std::complex<data_t>> &mat);

  void set_rank(int nr);

  // multiply matrix
  void mult_matrix(std::vector<std::complex<data_t>> &mat);
  void mult_matrix_conj(std::vector<std::complex<data_t>> &mat);

  // make conjugate tensor
  void conjugate_tensor(void);

  bool &sp_tensor(void) { return sp_tensor_; }
};

template <typename data_t>
Tensor<data_t>::Tensor(Tensor<data_t> &t) {
  base_ = t.base_;
  rank_ = t.rank_;
  extents_ = t.extents_;
  size_ = t.size_;
  tensor_ = t.tensor_;
  modes_ = t.modes_;
  qubits_ = t.qubits_;

  sp_tensor_ = t.sp_tensor_;
}

template <typename data_t>
void Tensor<data_t>::set(const reg_t &qubits,
                         std::vector<std::complex<data_t>> &mat) {
  tensor_ = mat;
  size_ = mat.size();
  rank_ = 0;

  int64_t t = size_;
  if (base_ == 2) {
    while (t > 1) {
      rank_++;
      t >>= 1;
    }
  } else {
    while (t > 1) {
      rank_++;
      t /= base_;
    }
  }

  modes_.resize(rank_);
  extents_.resize(rank_);
  for (int i = 0; i < rank_; i++)
    extents_[i] = base_;

  qubits_ = qubits;
}

template <typename data_t>
void Tensor<data_t>::set(int qubit, std::vector<std::complex<data_t>> &mat) {
  tensor_ = mat;
  size_ = mat.size();
  rank_ = 0;

  int64_t t = size_;
  if (base_ == 2) {
    while (t > 1) {
      rank_++;
      t >>= 1;
    }
  } else {
    while (t > 1) {
      rank_++;
      t /= base_;
    }
  }

  modes_.resize(rank_);
  extents_.resize(rank_);
  for (int i = 0; i < rank_; i++)
    extents_[i] = base_;

  qubits_.push_back(qubit);
}

template <typename data_t>
void Tensor<data_t>::set(const reg_t &qubits, std::complex<data_t> *mat,
                         uint_t size) {
  tensor_.resize(size);
  for (int i = 0; i < size; i++)
    tensor_[i] = mat[i];

  size_ = size;
  rank_ = 0;

  int64_t t = size_;
  if (base_ == 2) {
    while (t > 1) {
      rank_++;
      t >>= 1;
    }
  } else {
    while (t > 1) {
      rank_++;
      t /= base_;
    }
  }

  modes_.resize(rank_);
  extents_.resize(rank_);
  for (int i = 0; i < rank_; i++)
    extents_[i] = base_;

  qubits_ = qubits;
}

template <typename data_t>
void Tensor<data_t>::set_conj(const reg_t &qubits,
                              std::vector<std::complex<data_t>> &mat) {
  set(qubits, mat);

  for (uint_t i = 0; i < tensor_.size(); i++)
    tensor_[i] = std::conj(tensor_[i]);
  sp_tensor_ = true;
}

template <typename data_t>
void Tensor<data_t>::set_rank(int nr) {
  rank_ = nr;
  size_ = 1ull << nr;

  tensor_.resize(size_);
  modes_.resize(rank_);
  extents_.resize(rank_);
  for (int i = 0; i < rank_; i++)
    extents_[i] = base_;
}

template <typename data_t>
void Tensor<data_t>::mult_matrix(std::vector<std::complex<data_t>> &mat) {
  int i, j, k;

  if (tensor_.size() == mat.size()) {
    for (i = 0; i < rank_; i++) {
      std::vector<std::complex<data_t>> t(base_, 0.0);
      for (j = 0; j < base_; j++) {
        for (k = 0; k < base_; k++)
          t[k] += tensor_[i + j * base_] * mat[j + k * base_];
      }
      for (k = 0; k < base_; k++)
        tensor_[i + k * base_] = t[k];
    }
  }
}

template <typename data_t>
void Tensor<data_t>::mult_matrix_conj(std::vector<std::complex<data_t>> &mat) {
  int i, j, k;

  if (tensor_.size() == mat.size()) {
    for (i = 0; i < rank_; i++) {
      std::vector<std::complex<data_t>> t(base_, 0.0);
      for (j = 0; j < base_; j++) {
        for (k = 0; k < base_; k++)
          t[k] += tensor_[i + j * base_] * std::conj(mat[j + k * base_]);
      }
      for (k = 0; k < base_; k++)
        tensor_[i + k * base_] = t[k];
    }
  }
}

template <typename data_t>
void Tensor<data_t>::conjugate_tensor(void) {
  for (int i = 0; i < tensor_.size(); i++) {
    tensor_[i] = std::conj(tensor_[i]);
  }
}

//------------------------------------------------------------------------------
} // namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif // _tensor_hpp_
