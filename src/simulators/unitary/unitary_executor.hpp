/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019. 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _unitary_executor_hpp
#define _unitary_executor_hpp

#include "simulators/parallel_state_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace QubitUnitary {

//-------------------------------------------------------------------------
// Parallel executor for QubitUnitar
//-------------------------------------------------------------------------

template <class state_t>
class Executor : public CircuitExecutor::ParallelStateExecutor<state_t> {
  using Base = CircuitExecutor::ParallelStateExecutor<state_t>;

protected:
public:
  Executor() {}
  virtual ~Executor() {}

  auto move_to_matrix(void);
  auto copy_to_matrix(void);

protected:
  void set_config(const Config &config) override;

  // apply parallel operations
  bool apply_parallel_op(const Operations::Op &op, ExperimentResult &result,
                         RngEngine &rng, bool final_op) override;

  void initialize_qreg(uint_t num_qubits) override;

  //-----------------------------------------------------------------------
  // Apply Instructions
  //-----------------------------------------------------------------------
  // swap between chunks
  void apply_chunk_swap(const reg_t &qubits) override;

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the unitary matrix for the simulator
  void apply_save_unitary(const Operations::Op &op, ExperimentResult &result,
                          bool last_op);

  // Helper function for computing expectation value
  double expval_pauli(const reg_t &qubits, const std::string &pauli) override;

  // scale for unitary = 2
  // this function is used in the base class to scale chunk qubits for
  // multi-chunk distribution
  uint_t qubit_scale(void) override { return 2; }
};

template <class state_t>
void Executor<state_t>::set_config(const Config &config) {
  Base::set_config(config);
}

template <class state_t>
void Executor<state_t>::initialize_qreg(uint_t num_qubits) {
  uint_t iChunk;
  for (iChunk = 0; iChunk < Base::states_.size(); iChunk++) {
    Base::states_[iChunk].qreg().set_num_qubits(Base::chunk_bits_);
  }

  if (Base::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(iChunk)
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        uint_t irow, icol;
        irow = (Base::global_state_index_ + iChunk) >>
               ((Base::num_qubits_ - Base::chunk_bits_));
        icol = (Base::global_state_index_ + iChunk) -
               (irow << ((Base::num_qubits_ - Base::chunk_bits_)));
        if (irow == icol) {
          Base::states_[iChunk].qreg().initialize();
          Base::states_[iChunk].apply_global_phase();
        } else
          Base::states_[iChunk].qreg().zero();
      }
    }
  } else {
    for (iChunk = 0; iChunk < Base::states_.size(); iChunk++) {
      uint_t irow, icol;
      irow = (Base::global_state_index_ + iChunk) >>
             ((Base::num_qubits_ - Base::chunk_bits_));
      icol = (Base::global_state_index_ + iChunk) -
             (irow << ((Base::num_qubits_ - Base::chunk_bits_)));
      if (irow == icol) {
        Base::states_[iChunk].qreg().initialize();
        Base::states_[iChunk].apply_global_phase();
      } else
        Base::states_[iChunk].qreg().zero();
    }
  }
}

template <class state_t>
bool Executor<state_t>::apply_parallel_op(const Operations::Op &op,
                                          ExperimentResult &result,
                                          RngEngine &rng, bool final_op) {
  // temporary : this is for statevector
  if (Base::states_[0].creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::bfunc:
      Base::states_[0].creg().apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      Base::states_[0].creg().apply_roerror(op, rng);
      break;
    case Operations::OpType::set_unitary:
      Base::initialize_from_matrix(op.mats[0]);
      break;
    case Operations::OpType::save_state:
    case Operations::OpType::save_unitary:
      apply_save_unitary(op, result, final_op);
      break;
    default:
      return false;
    }
  }
  return true;
}

template <class state_t>
auto Executor<state_t>::move_to_matrix(void) {
  return Base::apply_to_matrix(false);
}

template <class state_t>
auto Executor<state_t>::copy_to_matrix(void) {
  return Base::apply_to_matrix(true);
}

template <class state_t>
void Executor<state_t>::apply_save_unitary(const Operations::Op &op,
                                           ExperimentResult &result,
                                           bool last_op) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full unitary can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "unitary" : op.string_params[0];

  if (last_op) {
    result.save_data_pershot(Base::states_[0].creg(), key, move_to_matrix(),
                             Operations::OpType::save_unitary, op.save_type);
  } else {
    result.save_data_pershot(Base::states_[0].creg(), key, copy_to_matrix(),
                             Operations::OpType::save_unitary, op.save_type);
  }
}

template <class state_t>
double Executor<state_t>::expval_pauli(const reg_t &qubits,
                                       const std::string &pauli) {
  throw std::runtime_error(
      "Unitary simulator does not support Pauli expectation values.");
}

// swap between chunks
template <class state_t>
void Executor<state_t>::apply_chunk_swap(const reg_t &qubits) {
  uint_t q0, q1;
  q0 = qubits[0];
  q1 = qubits[1];

  std::swap(Base::qubit_map_[q0], Base::qubit_map_[q1]);

  if (qubits[0] >= Base::chunk_bits_) {
    q0 += Base::chunk_bits_;
  }
  if (qubits[1] >= Base::chunk_bits_) {
    q1 += Base::chunk_bits_;
  }
  reg_t qs0 = {{q0, q1}};
  Base::apply_chunk_swap(qs0);
}

//------------------------------------------------------------------------------
} // namespace QubitUnitary
} // end namespace AER
//------------------------------------------------------------------------------
#endif
