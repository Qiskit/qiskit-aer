/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _mps_size_estimator_hpp_
#define _mps_size_estimator_hpp_

#include "framework/operations.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace MatrixProductState {

// size estimation of MPS simulation by calculating bond dimensions
class MPSSizeEstimator {
protected:
  uint_t num_qubits_;
  reg_t bond_dimensions_;
  std::vector<std::pair<uint_t, uint_t>> tensor_size_;
  reg_t qubit_map_;
  reg_t qubit_order_;

public:
  MPSSizeEstimator(void) {}
  MPSSizeEstimator(uint_t nq) { initialize(nq); }

  void initialize(uint_t nq);

  uint_t estimate(const std::vector<Operations::Op> &ops);

protected:
  void apply_qubits(const reg_t &qubits);

  void reorder_qubit(uint_t qubit, uint_t target);

  void update(uint_t a);
};

void MPSSizeEstimator::initialize(uint_t nq) {
  num_qubits_ = nq;
  bond_dimensions_.resize(nq);
  tensor_size_.resize(nq);
  qubit_map_.resize(nq);
  qubit_order_.resize(nq);

  for (uint_t i = 0; i < nq; i++) {
    tensor_size_[i].first = 1;
    tensor_size_[i].second = 1;

    qubit_map_[i] = i;
    qubit_order_[i] = i;

    bond_dimensions_[i] = 1;
  }
}

uint_t MPSSizeEstimator::estimate(const std::vector<Operations::Op> &ops) {
  uint_t n = ops.size();
  for (uint_t i = 0; i < n; i++) {
    switch (ops[i].type) {
    case Operations::OpType::gate:
    case Operations::OpType::matrix:
    case Operations::OpType::diagonal_matrix:
      if (ops[i].qubits.size() > 1)
        apply_qubits(ops[i].qubits);
      break;
    default:
      break;
    }
  }
  uint_t max_bond = 0;
  for (uint_t i = 0; i < num_qubits_ - 1; i++) {
    if (max_bond < bond_dimensions_[i])
      max_bond = bond_dimensions_[i];
  }
  return num_qubits_ * (32 * max_bond * max_bond + 8 * max_bond);
}

void MPSSizeEstimator::apply_qubits(const reg_t &qubits) {
  reg_t sorted(qubits.size());

  for (uint_t i = 0; i < qubits.size(); i++) {
    sorted[i] = qubit_map_[qubits[i]];
  }
  std::sort(sorted.begin(), sorted.end());

  for (uint_t i = 1; i < qubits.size(); i++) {
    reorder_qubit(sorted[i - 1], sorted[i]);
  }

  for (uint_t i = 0; i < qubits.size() - 1; i++) {
    update(sorted[i]);
  }
}

void MPSSizeEstimator::reorder_qubit(uint_t qubit, uint_t target) {
  while (target > qubit + 1) {
    uint_t q0, q1;
    q0 = qubit_order_[target - 1];
    q1 = qubit_order_[target];
    qubit_map_[q0] = target;
    qubit_map_[q1] = target - 1;
    std::swap(qubit_order_[target], qubit_order_[target - 1]);

    update(target - 1);

    target--;
  }
}

void MPSSizeEstimator::update(uint_t a) {
  uint_t rows = tensor_size_[a].first;
  uint_t cols = tensor_size_[a + 1].second;

  bond_dimensions_[a] = std::min(rows * 2, cols * 2);

  tensor_size_[a].first = rows;
  tensor_size_[a].second = bond_dimensions_[a];
  tensor_size_[a + 1].first = bond_dimensions_[a];
  tensor_size_[a + 1].second = cols;
}

//-------------------------------------------------------------------------
} // namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
