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

#include <limits>

#include "framework/operations.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace MatrixProductState {

// size estimation of MPS simulation by calculating bond dimensions
class MPSSizeEstimator {
protected:
  uint_t num_qubits_ = 0;
  reg_t bond_dimensions_;
  std::vector<std::pair<uint_t, uint_t>> tensor_size_;
  reg_t qubit_map_;
  reg_t qubit_order_;

public:
  MPSSizeEstimator(void) {}
  MPSSizeEstimator(uint_t nq) { initialize(nq); }

  void initialize(uint_t nq);

  uint_t
  estimate(const std::vector<Operations::Op> &ops,
           const stringmap_t<Gates> &gateset, bool account_pauli_cache = false,
           uint_t max_bond_dimension = std::numeric_limits<uint_t>::max());
  uint_t estimate_mb(const std::vector<Operations::Op> &ops,
                     const stringmap_t<Gates> &gateset,
                     uint_t max_bond_dimension);

protected:
  void apply_qubits(const reg_t &qubits, bool model_pauli_cache = false);

  void reorder_qubit(uint_t qubit, uint_t target);

  void update(uint_t a);

private:
  uint_t bond_limit_ = std::numeric_limits<uint_t>::max();
  static bool is_partial_pauli_save(const Operations::Op &op,
                                    uint_t num_qubits);
  static uint_t saturated_add(uint_t a, uint_t b);
  static uint_t saturated_multiply(uint_t a, uint_t b);
  uint_t pauli_cache_bytes() const;
  uint_t estimated_tensor_bytes() const;
  void import_bond_dimensions(const mps_container_t &mps);
  void estimate_span(uint_t first, uint_t last);
  void set_bond_dimension(uint_t cut, uint_t dimension);
};

inline uint_t
MPSSizeEstimator::estimate_mb(const std::vector<Operations::Op> &ops,
                              const stringmap_t<Gates> &gateset,
                              uint_t max_bond_dimension) {
  if (num_qubits_ < 2)
    return 0;
  const bool account_pauli_cache =
      std::any_of(ops.begin(), ops.end(), [this](const auto &op) {
        return is_partial_pauli_save(op, num_qubits_);
      });
  const auto bytes =
      estimate(ops, gateset, account_pauli_cache, max_bond_dimension);
  constexpr uint_t bytes_per_mib = uint_t{1} << 20;
  // Keep the historical rounding for circuits without a cache reservation.
  if (!account_pauli_cache)
    return bytes / bytes_per_mib;
  const auto rounded = bytes / bytes_per_mib + (bytes % bytes_per_mib != 0);
  // The circuit executor narrows its per-circuit MiB estimate to int.
  return std::min(rounded,
                  static_cast<uint_t>(std::numeric_limits<int>::max()));
}

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

uint_t MPSSizeEstimator::estimate(const std::vector<Operations::Op> &ops,
                                  const stringmap_t<Gates> &gateset,
                                  bool account_pauli_cache,
                                  uint_t max_bond_dimension) {
  bond_limit_ = account_pauli_cache ? std::max(uint_t{1}, max_bond_dimension)
                                    : std::numeric_limits<uint_t>::max();
  uint_t peak_pauli_cache = 0;
  uint_t peak_tensor_bytes = 0;
  for (const auto &op : ops) {
    double pi2, pi2_int;
    switch (op.type) {
    case Operations::OpType::gate:
      if (op.qubits.size() > 1) {
        auto it = gateset.find(op.name);
        if (it != gateset.end()) {
          switch (it->second) {
          case Gates::rxx:
          case Gates::ryy:
          case Gates::rzx:
            pi2 = std::real(op.params[0]) / M_PI;
            pi2_int = (double)std::round(pi2);
            if (!AER::Linalg::almost_equal(pi2, pi2_int))
              apply_qubits(op.qubits, account_pauli_cache);
            break;
          default:
            if (account_pauli_cache && it->second != Gates::pauli)
              apply_qubits(op.qubits, true);
            break;
          }
        }
      }
      break;
    case Operations::OpType::matrix:
      if (op.qubits.size() > 1)
        apply_qubits(op.qubits, account_pauli_cache);
      break;
    case Operations::OpType::diagonal_matrix:
    case Operations::OpType::kraus:
    case Operations::OpType::initialize:
      if (account_pauli_cache && op.qubits.size() > 1)
        apply_qubits(op.qubits, true);
      break;
    case Operations::OpType::set_statevec:
      if (account_pauli_cache && num_qubits_ > 1) {
        initialize(num_qubits_);
        estimate_span(0, num_qubits_ - 1);
      }
      break;
    case Operations::OpType::set_mps:
      if (account_pauli_cache) {
        import_bond_dimensions(op.mps);
        uint_t bytes = 0;
        for (const auto &site : op.mps.first) {
          bytes = saturated_add(
              bytes, saturated_multiply(
                         sizeof(complex_t),
                         saturated_add(site.first.size(), site.second.size())));
        }
        for (const auto &bond : op.mps.second)
          bytes = saturated_add(
              bytes, saturated_multiply(sizeof(double), bond.size()));
        peak_tensor_bytes = std::max(peak_tensor_bytes, bytes);
      }
      break;
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      if (account_pauli_cache && is_partial_pauli_save(op, num_qubits_)) {
        peak_pauli_cache = std::max(peak_pauli_cache, pauli_cache_bytes());
        peak_tensor_bytes =
            std::max(peak_tensor_bytes, estimated_tensor_bytes());
      }
      break;
    default:
      break;
    }
  }
  return saturated_add(std::max(estimated_tensor_bytes(), peak_tensor_bytes),
                       peak_pauli_cache);
}

inline uint_t MPSSizeEstimator::estimated_tensor_bytes() const {
  uint_t max_bond = 0;
  for (uint_t i = 0; i + 1 < num_qubits_; i++) {
    if (max_bond < bond_dimensions_[i])
      max_bond = bond_dimensions_[i];
  }
  const auto gamma_bytes =
      saturated_multiply(32, saturated_multiply(max_bond, max_bond));
  const auto lambda_bytes = saturated_multiply(8, max_bond);
  return saturated_multiply(num_qubits_,
                            saturated_add(gamma_bytes, lambda_bytes));
}

inline bool MPSSizeEstimator::is_partial_pauli_save(const Operations::Op &op,
                                                    uint_t num_qubits) {
  // A save of every qubit always spans both ends, even after qubit routing.
  return (op.type == Operations::OpType::save_expval ||
          op.type == Operations::OpType::save_expval_var) &&
         !op.qubits.empty() && op.qubits.size() < num_qubits;
}

inline uint_t MPSSizeEstimator::saturated_add(uint_t a, uint_t b) {
  constexpr auto limit = std::numeric_limits<uint_t>::max();
  return b > limit - a ? limit : a + b;
}

inline uint_t MPSSizeEstimator::saturated_multiply(uint_t a, uint_t b) {
  constexpr auto limit = std::numeric_limits<uint_t>::max();
  return a != 0 && b > limit / a ? limit : a * b;
}

inline uint_t MPSSizeEstimator::pauli_cache_bytes() const {
  // Two square complex matrices at each cut, including both boundaries.
  uint_t elements_per_chain = 2;
  for (uint_t cut = 0; cut + 1 < num_qubits_; ++cut) {
    const auto bond = bond_dimensions_[cut];
    elements_per_chain =
        saturated_add(elements_per_chain, saturated_multiply(bond, bond));
  }
  const auto matrices =
      saturated_multiply(2 * sizeof(complex_t), elements_per_chain);
  const auto descriptors =
      saturated_multiply(2 * sizeof(cmatrix_t), saturated_add(num_qubits_, 1));
  return saturated_add(matrices, descriptors);
}

inline void
MPSSizeEstimator::import_bond_dimensions(const mps_container_t &mps) {
  for (uint_t site = 0; site < num_qubits_; ++site) {
    const auto &gamma = mps.first.at(site).first;
    tensor_size_[site] = {gamma.GetRows(), gamma.GetColumns()};
    bond_dimensions_[site] = gamma.GetColumns();
    qubit_map_[site] = site;
    qubit_order_[site] = site;
  }
}

void MPSSizeEstimator::apply_qubits(const reg_t &qubits,
                                    bool model_pauli_cache) {
  reg_t sorted(qubits.size());

  for (uint_t i = 0; i < qubits.size(); i++) {
    sorted[i] = qubit_map_[qubits[i]];
  }
  std::sort(sorted.begin(), sorted.end());
  if (model_pauli_cache && sorted.size() > 2)
    estimate_span(sorted.front(), sorted.back());

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
  const auto rank = std::min(tensor_size_[a].first, tensor_size_[a + 1].second);
  set_bond_dimension(a, std::min(bond_limit_, saturated_multiply(2, rank)));
}

inline void MPSSizeEstimator::estimate_span(uint_t first, uint_t last) {
  // Each site doubles the rank available from the span's exterior bond.
  uint_t rank = tensor_size_[first].first;
  for (uint_t cut = first; cut < last; ++cut) {
    rank = std::min(bond_limit_, saturated_multiply(2, rank));
    set_bond_dimension(cut, rank);
  }
  rank = tensor_size_[last].second;
  for (uint_t site = last; site > first; --site) {
    rank = std::min(bond_limit_, saturated_multiply(2, rank));
    set_bond_dimension(site - 1, std::min(rank, bond_dimensions_[site - 1]));
  }
}

inline void MPSSizeEstimator::set_bond_dimension(uint_t cut, uint_t dimension) {
  bond_dimensions_[cut] = dimension;
  tensor_size_[cut].second = dimension;
  tensor_size_[cut + 1].first = dimension;
}

//-------------------------------------------------------------------------
} // namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
