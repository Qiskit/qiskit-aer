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

#ifndef _densitymatrix_executor_hpp_
#define _densitymatrix_executor_hpp_

#include "simulators/batch_shots_executor.hpp"
#include "simulators/parallel_state_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace DensityMatrix {

using ResultItr = std::vector<ExperimentResult>::iterator;
//-------------------------------------------------------------------------
// batched-shots executor for density matrix
//-------------------------------------------------------------------------
template <class state_t>
class Executor : public CircuitExecutor::ParallelStateExecutor<state_t>,
                 public CircuitExecutor::BatchShotsExecutor<state_t> {
  using Base = CircuitExecutor::MultiStateExecutor<state_t>;
  using BasePar = CircuitExecutor::ParallelStateExecutor<state_t>;
  using BaseBatch = CircuitExecutor::BatchShotsExecutor<state_t>;
  using Base::sample_measure;

protected:
public:
  Executor() {}
  virtual ~Executor() {}

protected:
  void set_config(const Config &config) override;

  bool shot_branching_supported(void) override { return true; }

  // apply parallel operations
  bool apply_parallel_op(const Operations::Op &op, ExperimentResult &result,
                         RngEngine &rng, bool final_op) override;

  // apply op to multiple shots , return flase if op is not supported to execute
  // in a batch
  bool apply_batched_op(const int_t istate, const Operations::Op &op,
                        ResultItr result, std::vector<RngEngine> &rng,
                        bool final_op = false) override;

  bool apply_branching_op(CircuitExecutor::Branch &root,
                          const Operations::Op &op, ResultItr result,
                          bool final_op) override;

  // Initializes an n-qubit state to the all |0> state
  void initialize_qreg(uint_t num_qubits) override;

  auto move_to_matrix();
  auto copy_to_matrix();

  template <typename list_t>
  void initialize_from_vector(const list_t &vec);

  void run_circuit_with_sampling(Circuit &circ, const Config &config,
                                 RngEngine &init_rng,
                                 ResultItr result) override;
  void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                         const Config &config, RngEngine &init_rng,
                         ResultItr result_it, bool sample_noise) override;

  bool allocate_states(uint_t num_states, const Config &config) override {
    return BasePar::allocate_states(num_states, config);
  }
  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                     const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by tracing out qubits
  void apply_reset(const reg_t &qubits);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kraus);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current full density matrix
  void apply_save_state(const Operations::Op &op, ExperimentResult &result,
                        bool last_op = false);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result,
                                 bool last_op = false);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes squared
  void apply_save_amplitudes_sq(const Operations::Op &op,
                                ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  // Return the reduced density matrix for the simulator
  cmatrix_t reduced_density_matrix(const reg_t &qubits, bool last_op = false);
  cmatrix_t reduced_density_matrix_helper(const reg_t &qubits,
                                          const reg_t &qubits_sorted);

  // Helper functions for shot-branching
  void apply_save_density_matrix(CircuitExecutor::Branch &root,
                                 const Operations::Op &op, ResultItr result,
                                 bool final_op);
  void apply_save_state(CircuitExecutor::Branch &root, const Operations::Op &op,
                        ResultItr result, bool final_op);
  void apply_save_probs(CircuitExecutor::Branch &root, const Operations::Op &op,
                        ResultItr result);
  void apply_save_amplitudes(CircuitExecutor::Branch &root,
                             const Operations::Op &op, ResultItr result);
  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  rvector_t measure_probs(const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_with_prob(const reg_t &qubits,
                                                     RngEngine &rng);

  void measure_reset_update(const std::vector<uint_t> &qubits,
                            const uint_t final_state, const uint_t meas_state,
                            const double meas_prob);

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                    RngEngine &rng) const override;

  rvector_t sample_measure_with_prob(CircuitExecutor::Branch &root,
                                     const reg_t &qubits);
  void measure_reset_update(CircuitExecutor::Branch &root,
                            const std::vector<uint_t> &qubits,
                            const int_t final_state,
                            const rvector_t &meas_probs);
  void apply_measure(CircuitExecutor::Branch &root, const reg_t &qubits,
                     const reg_t &cmemory, const reg_t &cregister);

  std::vector<reg_t> sample_measure(state_t &state, const reg_t &qubits,
                                    uint_t shots,
                                    std::vector<RngEngine> &rng) const override;

  //-----------------------------------------------------------------------
  // Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
  // swap between chunks
  void apply_chunk_swap(const reg_t &qubits) override;

  // apply multiple swaps between chunks
  void apply_multi_chunk_swap(const reg_t &qubits) override;

  // scale for density matrix = 2
  // this function is used in the base class to scale chunk qubits for
  // multi-chunk distribution
  uint_t qubit_scale(void) override { return 2; }
};

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------
template <class densmat_t>
void Executor<densmat_t>::initialize_qreg(uint_t num_qubits) {
  for (uint_t i = 0; i < Base::states_.size(); i++) {
    Base::states_[i].qreg().set_num_qubits(BasePar::chunk_bits_);
  }

  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        if (Base::global_state_index_ + iChunk == 0) {
          Base::states_[iChunk].qreg().initialize();
        } else {
          Base::states_[iChunk].qreg().zero();
        }
      }
    }
  } else {
    for (uint_t i = 0; i < Base::states_.size(); i++) {
      if (Base::global_state_index_ + i == 0) {
        Base::states_[i].qreg().initialize();
      } else {
        Base::states_[i].qreg().zero();
      }
    }
  }
}

template <class densmat_t>
template <typename list_t>
void Executor<densmat_t>::initialize_from_vector(const list_t &vec) {
  if ((1ull << (Base::num_qubits_ * 2)) == vec.size()) {
    BasePar::initialize_from_vector(vec);
  } else if ((1ull << (Base::num_qubits_ * 2)) == vec.size() * vec.size()) {
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t iChunk = Base::top_state_of_group_[ig];
             iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
          uint_t irow_chunk = ((iChunk + Base::global_state_index_) >>
                               ((Base::num_qubits_ - BasePar::chunk_bits_)))
                              << (BasePar::chunk_bits_);
          uint_t icol_chunk =
              ((iChunk + Base::global_state_index_) &
               ((1ull << ((Base::num_qubits_ - BasePar::chunk_bits_))) - 1))
              << (BasePar::chunk_bits_);

          // copy part of state for this chunk
          uint_t i;
          list_t vec1(1ull << BasePar::chunk_bits_);
          list_t vec2(1ull << BasePar::chunk_bits_);

          for (i = 0; i < (1ull << BasePar::chunk_bits_); i++) {
            vec1[i] = vec[(irow_chunk << BasePar::chunk_bits_) + i];
            vec2[i] = std::conj(vec[(icol_chunk << BasePar::chunk_bits_) + i]);
          }
          Base::states_[iChunk].qreg().initialize_from_vector(
              AER::Utils::tensor_product(vec1, vec2));
        }
      }
    } else {
      for (uint_t iChunk = 0; iChunk < Base::states_.size(); iChunk++) {
        uint_t irow_chunk = ((iChunk + Base::global_state_index_) >>
                             ((Base::num_qubits_ - BasePar::chunk_bits_)))
                            << (BasePar::chunk_bits_);
        uint_t icol_chunk =
            ((iChunk + Base::global_state_index_) &
             ((1ull << ((Base::num_qubits_ - BasePar::chunk_bits_))) - 1))
            << (BasePar::chunk_bits_);

        // copy part of state for this chunk
        uint_t i;
        list_t vec1(1ull << BasePar::chunk_bits_);
        list_t vec2(1ull << BasePar::chunk_bits_);

        for (i = 0; i < (1ull << BasePar::chunk_bits_); i++) {
          vec1[i] = vec[(irow_chunk << BasePar::chunk_bits_) + i];
          vec2[i] = std::conj(vec[(icol_chunk << BasePar::chunk_bits_) + i]);
        }
        Base::states_[iChunk].qreg().initialize_from_vector(
            AER::Utils::tensor_product(vec1, vec2));
      }
    }
  } else {
    throw std::runtime_error(
        "DensityMatrixChunk::initialize input vector is incorrect length. "
        "Expected: " +
        std::to_string((1ull << (Base::num_qubits_ * 2))) +
        " Received: " + std::to_string(vec.size()));
  }
}

template <class densmat_t>
auto Executor<densmat_t>::move_to_matrix() {
  return BasePar::apply_to_matrix(false);
}

template <class densmat_t>
auto Executor<densmat_t>::copy_to_matrix() {
  return BasePar::apply_to_matrix(true);
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class densmat_t>
void Executor<densmat_t>::set_config(const Config &config) {
  BasePar::set_config(config);
  BaseBatch::set_config(config);
}

template <class state_t>
void Executor<state_t>::run_circuit_with_sampling(Circuit &circ,
                                                  const Config &config,
                                                  RngEngine &init_rng,
                                                  ResultItr result_it) {
  Noise::NoiseModel dummy_noise;
  if (BasePar::multiple_chunk_required(config, circ, dummy_noise)) {
    return BasePar::run_circuit_with_sampling(circ, config, init_rng,
                                              result_it);
  } else {
    return BaseBatch::run_circuit_with_sampling(circ, config, init_rng,
                                                result_it);
  }
}

template <class state_t>
void Executor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    RngEngine &init_rng, ResultItr result_it, bool sample_noise) {
  state_t dummy_state;
  if (BasePar::multiple_chunk_required(config, circ, noise)) {
    return BasePar::run_circuit_shots(circ, noise, config, init_rng, result_it,
                                      sample_noise);
  } else {
    return BaseBatch::run_circuit_shots(circ, noise, config, init_rng,
                                        result_it, sample_noise);
  }
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class densmat_t>
bool Executor<densmat_t>::apply_parallel_op(const Operations::Op &op,
                                            ExperimentResult &result,
                                            RngEngine &rng, bool final_ops) {
  if (Base::states_[0].creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::reset:
      apply_reset(op.qubits);
      break;
    case Operations::OpType::measure:
      apply_measure(op.qubits, op.memory, op.registers, rng);
      break;
    case Operations::OpType::bfunc:
      BasePar::apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BasePar::apply_roerror(op, rng);
      break;
    case Operations::OpType::kraus:
      apply_kraus(op.qubits, op.mats);
      break;
    case Operations::OpType::set_statevec:
      initialize_from_vector(op.params);
      break;
    case Operations::OpType::set_densmat:
      BasePar::initialize_from_matrix(op.mats[0]);
      break;
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      BasePar::apply_save_expval(op, result);
      break;
    case Operations::OpType::save_state:
      apply_save_state(op, result, final_ops);
      break;
    case Operations::OpType::save_densmat:
      apply_save_density_matrix(op, result, final_ops);
      break;
    case Operations::OpType::save_probs:
    case Operations::OpType::save_probs_ket:
      apply_save_probs(op, result);
      break;
    case Operations::OpType::save_amps_sq:
      apply_save_amplitudes_sq(op, result);
      break;
    default:
      return false;
    }
  }
  return true;
}

template <class state_t>
bool Executor<state_t>::apply_batched_op(const int_t istate,
                                         const Operations::Op &op,
                                         ResultItr result,
                                         std::vector<RngEngine> &rng,
                                         bool final_op) {
  if (op.conditional) {
    Base::states_[istate].qreg().set_conditional(op.conditional_reg);
  }

  switch (op.type) {
  case Operations::OpType::barrier:
  case Operations::OpType::nop:
  case Operations::OpType::qerror_loc:
    break;
  case Operations::OpType::reset:
    Base::states_[istate].apply_reset(op.qubits);
    break;
  case Operations::OpType::measure:
    Base::states_[istate].qreg().apply_batched_measure(op.qubits, rng,
                                                       op.memory, op.registers);
    break;
  case Operations::OpType::bfunc:
    Base::states_[istate].qreg().apply_bfunc(op);
    break;
  case Operations::OpType::roerror:
    Base::states_[istate].qreg().apply_roerror(op, rng);
    break;
  case Operations::OpType::gate:
    Base::states_[istate].apply_gate(op);
    break;
  case Operations::OpType::matrix:
    Base::states_[istate].apply_matrix(op.qubits, op.mats[0]);
    break;
  case Operations::OpType::diagonal_matrix:
    Base::states_[istate].apply_diagonal_unitary_matrix(op.qubits, op.params);
    break;
  case Operations::OpType::superop:
    Base::states_[istate].qreg().apply_superop_matrix(
        op.qubits, Utils::vectorize_matrix(op.mats[0]));
    break;
  case Operations::OpType::kraus:
    Base::states_[istate].apply_kraus(op.qubits, op.mats);
    break;
  default:
    // other operations should be called to indivisual chunks by apply_op
    return false;
  }
  return true;
}

template <class state_t>
bool Executor<state_t>::apply_branching_op(CircuitExecutor::Branch &root,
                                           const Operations::Op &op,
                                           ResultItr result, bool final_op) {
  RngEngine dummy;
  if (Base::states_[root.state_index()].creg().check_conditional(op)) {
    switch (op.type) {
      // ops with branching
      //      case Operations::OpType::reset:
      //        apply_reset(root, op.qubits);
      //        break;
    case Operations::OpType::measure:
      apply_measure(root, op.qubits, op.memory, op.registers);
      break;
    // save ops
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      Base::apply_save_expval(root, op, result);
      break;
    case Operations::OpType::save_state:
      apply_save_state(root, op, result, final_op);
      break;
    case Operations::OpType::save_densmat:
      apply_save_density_matrix(root, op, result, final_op);
      break;
    case Operations::OpType::save_probs:
    case Operations::OpType::save_probs_ket:
      apply_save_probs(root, op, result);
      break;
    case Operations::OpType::save_amps_sq:
      apply_save_amplitudes(root, op, result);
      break;
    default:
      return false;
    }
  }
  return true;
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class densmat_t>
void Executor<densmat_t>::apply_save_probs(const Operations::Op &op,
                                           ExperimentResult &result) {
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    result.save_data_average(
        Base::states_[0].creg(), op.string_params[0],
        Utils::vec2ket(probs, Base::json_chop_threshold_, 16), op.type,
        op.save_type);
  } else {
    result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_amplitudes_sq(const Operations::Op &op,
                                                   ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes_sq instructions (empty params).");
  }
  const uint_t size = op.int_params.size();
  rvector_t amps_sq(size);

  int_t iChunk;
#pragma omp parallel for if (BasePar::chunk_omp_parallel_) private(iChunk)
  for (iChunk = 0; iChunk < (int_t)Base::states_.size(); iChunk++) {
    uint_t irow, icol;
    irow = (Base::global_state_index_ + iChunk) >>
           ((Base::num_qubits_ - BasePar::chunk_bits_));
    icol = (Base::global_state_index_ + iChunk) -
           (irow << ((Base::num_qubits_ - BasePar::chunk_bits_)));
    if (irow != icol)
      continue;

    for (uint_t i = 0; i < size; ++i) {
      uint_t idx = BasePar::mapped_index(op.int_params[i]);
      if (idx >= (irow << BasePar::chunk_bits_) &&
          idx < ((irow + 1) << BasePar::chunk_bits_))
        amps_sq[i] = Base::states_[iChunk].qreg().probability(
            idx - (irow << BasePar::chunk_bits_));
    }
  }
#ifdef AER_MPI
  BasePar::reduce_sum(amps_sq);
#endif

  result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                           std::move(amps_sq), op.type, op.save_type);
}

template <class densmat_t>
double Executor<densmat_t>::expval_pauli(const reg_t &qubits,
                                         const std::string &pauli) {
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;
  std::string pauli_in_chunk;
  std::string pauli_out_chunk;
  int_t i, n;
  double expval(0.);

  // get inner/outer chunk pauli string
  n = pauli.size();
  for (i = 0; i < n; i++) {
    if (qubits[i] < BasePar::chunk_bits_) {
      qubits_in_chunk.push_back(qubits[i]);
      pauli_in_chunk.push_back(pauli[n - i - 1]);
    } else {
      qubits_out_chunk.push_back(qubits[i]);
      pauli_out_chunk.push_back(pauli[n - i - 1]);
    }
  }

  int_t nrows = 1ull << ((Base::num_qubits_ - BasePar::chunk_bits_));

  if (qubits_out_chunk.size() > 0) { // there are bits out of chunk
    std::complex<double> phase = 1.0;

    std::reverse(pauli_out_chunk.begin(), pauli_out_chunk.end());
    std::reverse(pauli_in_chunk.begin(), pauli_in_chunk.end());

    uint_t x_mask, z_mask, num_y, x_max;
    std::tie(x_mask, z_mask, num_y, x_max) =
        AER::QV::pauli_masks_and_phase(qubits_out_chunk, pauli_out_chunk);

    z_mask >>= (BasePar::chunk_bits_);
    if (x_mask != 0) {
      x_mask >>= (BasePar::chunk_bits_);
      x_max -= (BasePar::chunk_bits_);

      AER::QV::add_y_phase(num_y, phase);

      const uint_t mask_u = ~((1ull << (x_max + 1)) - 1);
      const uint_t mask_l = (1ull << x_max) - 1;

      for (i = 0; i < nrows / 2; i++) {
        uint_t irow = ((i << 1) & mask_u) | (i & mask_l);
        uint_t iChunk = (irow ^ x_mask) + irow * nrows;

        if (Base::state_index_begin_[Base::distributed_rank_] <= iChunk &&
            Base::state_index_end_[Base::distributed_rank_] >
                iChunk) { // on this process
          double sign = 2.0;
          if (z_mask && (AER::Utils::popcount(irow & z_mask) & 1))
            sign = -2.0;
          expval += sign * Base::states_[iChunk - Base::global_state_index_]
                               .qreg()
                               .expval_pauli_non_diagonal_chunk(
                                   qubits_in_chunk, pauli_in_chunk, phase);
        }
      }
    } else {
      for (i = 0; i < nrows; i++) {
        uint_t iChunk = i * (nrows + 1);
        if (Base::state_index_begin_[Base::distributed_rank_] <= iChunk &&
            Base::state_index_end_[Base::distributed_rank_] >
                iChunk) { // on this process
          double sign = 1.0;
          if (z_mask && (AER::Utils::popcount(i & z_mask) & 1))
            sign = -1.0;
          expval +=
              sign * Base::states_[iChunk - Base::global_state_index_]
                         .qreg()
                         .expval_pauli(qubits_in_chunk, pauli_in_chunk, 1.0);
        }
      }
    }
  } else { // all bits are inside chunk
    for (i = 0; i < nrows; i++) {
      uint_t iChunk = i * (nrows + 1);
      if (Base::state_index_begin_[Base::distributed_rank_] <= iChunk &&
          Base::state_index_end_[Base::distributed_rank_] >
              iChunk) { // on this process
        expval += Base::states_[iChunk - Base::global_state_index_]
                      .qreg()
                      .expval_pauli(qubits, pauli, 1.0);
      }
    }
  }

#ifdef AER_MPI
  BasePar::reduce_sum(expval);
#endif
  return expval;
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_density_matrix(const Operations::Op &op,
                                                    ExperimentResult &result,
                                                    bool last_op) {
  result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                           reduced_density_matrix(op.qubits, last_op), op.type,
                           op.save_type);
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_state(const Operations::Op &op,
                                           ExperimentResult &result,
                                           bool last_op) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name + " was not applied to all qubits."
                                          " Only the full state can be saved.");
  }
  // Renamp single data type to average
  Operations::DataSubType save_type;
  switch (op.save_type) {
  case Operations::DataSubType::single:
    save_type = Operations::DataSubType::average;
    break;
  case Operations::DataSubType::c_single:
    save_type = Operations::DataSubType::c_average;
    break;
  default:
    save_type = op.save_type;
  }

  // Default key
  std::string key = (op.string_params[0] == "_method_") ? "density_matrix"
                                                        : op.string_params[0];
  if (last_op) {
    result.save_data_average(Base::states_[0].creg(), key, move_to_matrix(),
                             Operations::OpType::save_densmat, save_type);
  } else {
    result.save_data_average(Base::states_[0].creg(), key, copy_to_matrix(),
                             Operations::OpType::save_densmat, save_type);
  }
}

template <class densmat_t>
cmatrix_t Executor<densmat_t>::reduced_density_matrix(const reg_t &qubits,
                                                      bool last_op) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    std::complex<double> sum = 0.0;
    for (uint_t i = 0; i < Base::states_.size(); i++) {
      sum += Base::states_[i].qreg().trace();
    }
#ifdef AER_MPI
    BasePar::reduce_sum(sum);
#endif
    reduced_state[0] = sum;
  } else {
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    if ((qubits.size() == Base::num_qubits_) && (qubits == qubits_sorted)) {
      if (last_op) {
        reduced_state = move_to_matrix();
      } else {
        reduced_state = copy_to_matrix();
      }
    } else {
      reduced_state = reduced_density_matrix_helper(qubits, qubits_sorted);
    }
  }
  return reduced_state;
}

template <class densmat_t>
cmatrix_t
Executor<densmat_t>::reduced_density_matrix_helper(const reg_t &qubits,
                                                   const reg_t &qubits_sorted) {
  uint_t iChunk;
  uint_t size = 1ull << (BasePar::chunk_bits_ * 2);
  uint_t mask = (1ull << (BasePar::chunk_bits_)) - 1;
  uint_t num_threads = Base::states_[0].qreg().get_omp_threads();

  size_t size_required =
      (sizeof(std::complex<double>) << (qubits.size() * 2)) +
      (sizeof(std::complex<double>) << (BasePar::chunk_bits_ * 2)) *
          Base::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store density matrix"));
  }
  cmatrix_t reduced_state(1ull << qubits.size(), 1ull << qubits.size(), true);

  if (Base::distributed_rank_ == 0) {
    auto tmp = Base::states_[0].copy_to_matrix();
    for (iChunk = 0; iChunk < Base::num_global_states_; iChunk++) {
      int_t i;
      uint_t irow_chunk =
          (iChunk >> ((Base::num_qubits_ - BasePar::chunk_bits_)))
          << BasePar::chunk_bits_;
      uint_t icol_chunk =
          (iChunk &
           ((1ull << ((Base::num_qubits_ - BasePar::chunk_bits_))) - 1))
          << BasePar::chunk_bits_;

      if (iChunk < Base::num_local_states_)
        tmp = Base::states_[iChunk].qreg().copy_to_matrix();
#ifdef AER_MPI
      else
        BasePar::recv_data(tmp.data(), size, 0, iChunk);
#endif
#pragma omp parallel for if (num_threads > 1) num_threads(num_threads)
      for (i = 0; i < (int_t)size; i++) {
        uint_t irow = (i >> (BasePar::chunk_bits_)) + irow_chunk;
        uint_t icol = (i & mask) + icol_chunk;
        uint_t irow_out = 0;
        uint_t icol_out = 0;
        uint_t j;
        for (j = 0; j < qubits.size(); j++) {
          if ((irow >> qubits[j]) & 1) {
            irow &= ~(1ull << qubits[j]);
            irow_out += (1ull << j);
          }
          if ((icol >> qubits[j]) & 1) {
            icol &= ~(1ull << qubits[j]);
            icol_out += (1ull << j);
          }
        }
        if (irow == icol) { // only diagonal base can be reduced
          uint_t idx = ((irow_out) << qubits.size()) + icol_out;
#pragma omp critical
          reduced_state[idx] += tmp[i];
        }
      }
    }
  } else {
#ifdef AER_MPI
    // send matrices to process 0
    for (iChunk = 0; iChunk < Base::num_global_states_; iChunk++) {
      uint_t iProc = BasePar::get_process_by_chunk(iChunk);
      if (iProc == Base::distributed_rank_) {
        auto tmp = Base::states_[iChunk - Base::global_state_index_]
                       .qreg()
                       .copy_to_matrix();
        BasePar::send_data(tmp.data(), size, iChunk, 0);
      }
    }
#endif
  }

  return reduced_state;
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_density_matrix(
    CircuitExecutor::Branch &root, const Operations::Op &op, ResultItr result,
    bool final_op) {
  cmatrix_t mat;
  mat = Base::states_[root.state_index()].reduced_density_matrix(op.qubits,
                                                                 final_op);

  std::vector<bool> copied(Base::num_bind_params_, false);
  for (uint_t i = 0; i < root.num_shots(); i++) {
    uint_t ip = root.param_index(i);
    if (!copied[ip]) {
      (result + ip)
          ->save_data_average(Base::states_[root.state_index()].creg(),
                              op.string_params[0], mat, op.type, op.save_type);
      copied[ip] = true;
    }
  }
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_state(CircuitExecutor::Branch &root,
                                           const Operations::Op &op,
                                           ResultItr result, bool final_op) {
  if (op.qubits.size() !=
      Base::states_[root.state_index()].qreg().num_qubits()) {
    throw std::invalid_argument(op.name + " was not applied to all qubits."
                                          " Only the full state can be saved.");
  }
  // Renamp single data type to average
  Operations::DataSubType save_type;
  switch (op.save_type) {
  case Operations::DataSubType::single:
    save_type = Operations::DataSubType::average;
    break;
  case Operations::DataSubType::c_single:
    save_type = Operations::DataSubType::c_average;
    break;
  default:
    save_type = op.save_type;
  }

  // Default key
  std::string key = (op.string_params[0] == "_method_") ? "density_matrix"
                                                        : op.string_params[0];

  std::vector<bool> copied(Base::num_bind_params_, false);
  if (final_op) {
    auto state = Base::states_[root.state_index()].move_to_matrix();
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(Base::states_[root.state_index()].creg(), key,
                                state, OpType::save_densmat, save_type);
        copied[ip] = true;
      }
    }
  } else {
    auto state = Base::states_[root.state_index()].copy_to_matrix();

    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(Base::states_[root.state_index()].creg(), key,
                                state, OpType::save_densmat, save_type);
        copied[ip] = true;
      }
    }
  }
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_probs(CircuitExecutor::Branch &root,
                                           const Operations::Op &op,
                                           ResultItr result) {
  // get probs as hexadecimal
  auto probs =
      Base::states_[root.state_index()].qreg().probabilities(op.qubits);

  std::vector<bool> copied(Base::num_bind_params_, false);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(
                Base::states_[root.state_index()].creg(), op.string_params[0],
                Utils::vec2ket(probs, Base::json_chop_threshold_, 16), op.type,
                op.save_type);
        copied[ip] = true;
      }
    }
  } else {
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(Base::states_[root.state_index()].creg(),
                                op.string_params[0], probs, op.type,
                                op.save_type);
        copied[ip] = true;
      }
    }
  }
}

template <class densmat_t>
void Executor<densmat_t>::apply_save_amplitudes(CircuitExecutor::Branch &root,
                                                const Operations::Op &op,
                                                ResultItr result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  rvector_t amps_sq(size, 0);
  for (int_t i = 0; i < size; ++i) {
    amps_sq[i] =
        Base::states_[root.state_index()].qreg().probability(op.int_params[i]);
  }
  std::vector<bool> copied(Base::num_bind_params_, false);
  for (uint_t i = 0; i < root.num_shots(); i++) {
    uint_t ip = root.param_index(i);
    if (!copied[ip]) {
      (result + ip)
          ->save_data_average(Base::states_[root.state_index()].creg(),
                              op.string_params[0], amps_sq, op.type,
                              op.save_type);
      copied[ip] = true;
    }
  }
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

template <class densmat_t>
void Executor<densmat_t>::apply_measure(const reg_t &qubits,
                                        const reg_t &cmemory,
                                        const reg_t &cregister,
                                        RngEngine &rng) {
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BasePar::store_measure(outcome, cmemory, cregister);
}

template <class densmat_t>
rvector_t Executor<densmat_t>::measure_probs(const reg_t &qubits) const {
  uint_t dim = 1ull << qubits.size();
  rvector_t sum(dim, 0.0);
  uint_t i, j, k;
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;

  for (i = 0; i < qubits.size(); i++) {
    if (qubits[i] < BasePar::chunk_bits_) {
      qubits_in_chunk.push_back(qubits[i]);
    } else {
      qubits_out_chunk.push_back(qubits[i]);
    }
  }

  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(i, j, k)
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (i = Base::top_state_of_group_[ig];
           i < Base::top_state_of_group_[ig + 1]; i++) {
        uint_t irow, icol;
        irow = (Base::global_state_index_ + i) >>
               ((Base::num_qubits_ - BasePar::chunk_bits_));
        icol = (Base::global_state_index_ + i) -
               (irow << ((Base::num_qubits_ - BasePar::chunk_bits_)));

        if (irow == icol) { // diagonal chunk
          if (qubits_in_chunk.size() > 0) {
            auto chunkSum =
                Base::states_[i].qreg().probabilities(qubits_in_chunk);
            if (qubits_in_chunk.size() == qubits.size()) {
              for (j = 0; j < dim; j++) {
#pragma omp atomic
                sum[j] += chunkSum[j];
              }
            } else {
              for (j = 0; j < chunkSum.size(); j++) {
                int idx = 0;
                int i_in = 0;
                for (k = 0; k < qubits.size(); k++) {
                  if (qubits[k] < (BasePar::chunk_bits_)) {
                    idx += (((j >> i_in) & 1) << k);
                    i_in++;
                  } else {
                    if ((((i + Base::global_state_index_)
                          << (BasePar::chunk_bits_)) >>
                         qubits[k]) &
                        1) {
                      idx += 1ull << k;
                    }
                  }
                }
#pragma omp atomic
                sum[idx] += chunkSum[j];
              }
            }
          } else { // there is no bit in chunk
            auto tr = std::real(Base::states_[i].qreg().trace());
            int idx = 0;
            for (k = 0; k < qubits_out_chunk.size(); k++) {
              if ((((i + Base::global_state_index_)
                    << (BasePar::chunk_bits_)) >>
                   qubits_out_chunk[k]) &
                  1) {
                idx += 1ull << k;
              }
            }
#pragma omp atomic
            sum[idx] += tr;
          }
        }
      }
    }
  } else {
    for (i = 0; i < Base::states_.size(); i++) {
      uint_t irow, icol;
      irow = (Base::global_state_index_ + i) >>
             ((Base::num_qubits_ - BasePar::chunk_bits_));
      icol = (Base::global_state_index_ + i) -
             (irow << ((Base::num_qubits_ - BasePar::chunk_bits_)));

      if (irow == icol) { // diagonal chunk
        if (qubits_in_chunk.size() > 0) {
          auto chunkSum =
              Base::states_[i].qreg().probabilities(qubits_in_chunk);
          if (qubits_in_chunk.size() == qubits.size()) {
            for (j = 0; j < dim; j++) {
              sum[j] += chunkSum[j];
            }
          } else {
            for (j = 0; j < chunkSum.size(); j++) {
              int idx = 0;
              int i_in = 0;
              for (k = 0; k < qubits.size(); k++) {
                if (qubits[k] < (BasePar::chunk_bits_)) {
                  idx += (((j >> i_in) & 1) << k);
                  i_in++;
                } else {
                  if ((((i + Base::global_state_index_)
                        << (BasePar::chunk_bits_)) >>
                       qubits[k]) &
                      1) {
                    idx += 1ull << k;
                  }
                }
              }
              sum[idx] += chunkSum[j];
            }
          }
        } else { // there is no bit in chunk
          auto tr = std::real(Base::states_[i].qreg().trace());
          int idx = 0;
          for (k = 0; k < qubits_out_chunk.size(); k++) {
            if ((((i + Base::global_state_index_) << (BasePar::chunk_bits_)) >>
                 qubits_out_chunk[k]) &
                1) {
              idx += 1ull << k;
            }
          }
          sum[idx] += tr;
        }
      }
    }
  }

#ifdef AER_MPI
  BasePar::reduce_sum(sum);
#endif

  return sum;
}

template <class densmat_t>
void Executor<densmat_t>::apply_reset(const reg_t &qubits) {
  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        Base::states_[iChunk].qreg().apply_reset(qubits);
      }
    }
  } else {
    for (uint_t i = 0; i < Base::states_.size(); i++)
      Base::states_[i].qreg().apply_reset(qubits);
  }
}

template <class densmat_t>
std::pair<uint_t, double>
Executor<densmat_t>::sample_measure_with_prob(const reg_t &qubits,
                                              RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class densmat_t>
void Executor<densmat_t>::measure_reset_update(const reg_t &qubits,
                                               const uint_t final_state,
                                               const uint_t meas_state,
                                               const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++)
          Base::states_[i].qreg().apply_diagonal_unitary_matrix(qubits, mdiag);
      }
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        Base::states_[i].qreg().apply_diagonal_unitary_matrix(qubits, mdiag);
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if (qubits[0] < BasePar::chunk_bits_) {
        if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
          for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
            for (uint_t i = Base::top_state_of_group_[ig];
                 i < Base::top_state_of_group_[ig + 1]; i++)
              Base::states_[i].qreg().apply_x(qubits[0]);
          }
        } else {
          for (uint_t i = 0; i < Base::states_.size(); i++)
            Base::states_[i].qreg().apply_x(qubits[0]);
        }
      } else {
        BasePar::apply_chunk_x(qubits[0]);
        BasePar::apply_chunk_x(qubits[0] + BasePar::chunk_bits_);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++)
          Base::states_[i].qreg().apply_diagonal_unitary_matrix(qubits, mdiag);
      }
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        Base::states_[i].qreg().apply_diagonal_unitary_matrix(qubits, mdiag);
    }

    // If it doesn't agree with the reset state update
    // TODO This function could be optimized as a permutation update
    if (final_state != meas_state) {
      // build vectorized permutation matrix
      cvector_t perm(dim * dim, 0.);
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (size_t j = 0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      reg_t qubits_in_chunk;
      reg_t qubits_out_chunk;

      for (uint_t i = 0; i < qubits.size(); i++) {
        if (qubits[i] < BasePar::chunk_bits_) {
          qubits_in_chunk.push_back(qubits[i]);
        } else {
          qubits_out_chunk.push_back(qubits[i]);
        }
      }
      if (qubits_in_chunk.size() > 0) { // in chunk exchange
        if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
          for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
            for (uint_t i = Base::top_state_of_group_[ig];
                 i < Base::top_state_of_group_[ig + 1]; i++)
              Base::states_[i].qreg().apply_unitary_matrix(qubits, perm);
          }
        } else {
          for (uint_t i = 0; i < Base::states_.size(); i++)
            Base::states_[i].qreg().apply_unitary_matrix(qubits, perm);
        }
      }
      if (qubits_out_chunk.size() > 0) { // out of chunk exchange
        for (uint_t i = 0; i < qubits_out_chunk.size(); i++) {
          BasePar::apply_chunk_x(qubits_out_chunk[i]);
          BasePar::apply_chunk_x(qubits_out_chunk[i] +
                                 (Base::num_qubits_ - BasePar::chunk_bits_));
        }
      }
    }
  }
}

template <class densmat_t>
std::vector<reg_t> Executor<densmat_t>::sample_measure(const reg_t &qubits,
                                                       uint_t shots,
                                                       RngEngine &rng) const {
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  for (uint_t i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));
  reg_t allbit_samples(shots, 0);

  uint_t i, j;
  std::vector<double> chunkSum(Base::states_.size() + 1, 0);
  double sum, localSum;
  // calculate per chunk sum
  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(i)
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (i = Base::top_state_of_group_[ig];
           i < Base::top_state_of_group_[ig + 1]; i++) {
        uint_t irow, icol;
        irow = (Base::global_state_index_ + i) >>
               ((Base::num_qubits_ - BasePar::chunk_bits_));
        icol = (Base::global_state_index_ + i) -
               (irow << ((Base::num_qubits_ - BasePar::chunk_bits_)));
        if (irow == icol) // only diagonal chunk has probabilities
          chunkSum[i] = std::real(Base::states_[i].qreg().trace());
        else
          chunkSum[i] = 0.0;
      }
    }
  } else {
    for (i = 0; i < Base::states_.size(); i++) {
      uint_t irow, icol;
      irow = (Base::global_state_index_ + i) >>
             ((Base::num_qubits_ - BasePar::chunk_bits_));
      icol = (Base::global_state_index_ + i) -
             (irow << ((Base::num_qubits_ - BasePar::chunk_bits_)));
      if (irow == icol) // only diagonal chunk has probabilities
        chunkSum[i] = std::real(Base::states_[i].qreg().trace());
      else
        chunkSum[i] = 0.0;
    }
  }
  localSum = 0.0;
  for (i = 0; i < Base::states_.size(); i++) {
    sum = localSum;
    localSum += chunkSum[i];
    chunkSum[i] = sum;
  }
  chunkSum[Base::states_.size()] = localSum;

  double globalSum = 0.0;
  if (Base::nprocs_ > 1) {
    std::vector<double> procTotal(Base::nprocs_);

    for (i = 0; i < Base::nprocs_; i++) {
      procTotal[i] = localSum;
    }
    BasePar::gather_value(procTotal);

    for (i = 0; i < Base::myrank_; i++) {
      globalSum += procTotal[i];
    }
  }

  reg_t local_samples(shots, 0);

  // get rnds positions for each chunk
  for (i = 0; i < Base::states_.size(); i++) {
    uint_t irow, icol;
    irow = (Base::global_state_index_ + i) >>
           ((Base::num_qubits_ - BasePar::chunk_bits_));
    icol = (Base::global_state_index_ + i) -
           (irow << ((Base::num_qubits_ - BasePar::chunk_bits_)));
    if (irow != icol)
      continue;

    uint_t nIn;
    std::vector<uint_t> vIdx;
    std::vector<double> vRnd;

    // find rnds in this chunk
    nIn = 0;
    for (j = 0; j < shots; j++) {
      if (rnds[j] >= chunkSum[i] + globalSum &&
          rnds[j] < chunkSum[i + 1] + globalSum) {
        vRnd.push_back(rnds[j] - (globalSum + chunkSum[i]));
        vIdx.push_back(j);
        nIn++;
      }
    }

    if (nIn > 0) {
      auto chunkSamples = Base::states_[i].qreg().sample_measure(vRnd);
      uint_t ir;
      ir = (Base::global_state_index_ + i) >>
           ((Base::num_qubits_ - BasePar::chunk_bits_));

      for (j = 0; j < chunkSamples.size(); j++) {
        local_samples[vIdx[j]] = (ir << BasePar::chunk_bits_) + chunkSamples[j];
      }
    }
  }

#ifdef AER_MPI
  BasePar::reduce_sum(local_samples);
#endif
  allbit_samples = local_samples;

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, Base::num_qubits_);
    reg_t sample;
    sample.reserve(qubits.size());
    for (uint_t qubit : qubits) {
      sample.push_back(allbit_sample[qubit]);
    }
    all_samples.push_back(sample);
  }
  return all_samples;
}

template <class state_t>
rvector_t
Executor<state_t>::sample_measure_with_prob(CircuitExecutor::Branch &root,
                                            const reg_t &qubits) {
  rvector_t probs =
      Base::states_[root.state_index()].qreg().probabilities(qubits);
  uint_t nshots = root.num_shots();
  reg_t shot_branch(nshots);

  for (uint_t i = 0; i < nshots; i++) {
    shot_branch[i] = root.rng_shots()[i].rand_int(probs);
  }

  // branch shots
  root.creg() = Base::states_[root.state_index()].creg();
  root.branch_shots(shot_branch, probs.size());

  return probs;
}

template <class state_t>
void Executor<state_t>::measure_reset_update(CircuitExecutor::Branch &root,
                                             const std::vector<uint_t> &qubits,
                                             const int_t final_state,
                                             const rvector_t &meas_probs) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    for (int_t i = 0; i < 2; i++) {
      cvector_t mdiag(2, 0.);
      mdiag[i] = 1. / std::sqrt(meas_probs[i]);

      Operations::Op op;
      op.type = OpType::diagonal_matrix;
      op.qubits = qubits;
      op.params = mdiag;
      root.branches()[i]->add_op_after_branch(op);

      if (final_state >= 0 && final_state != i) {
        Operations::Op op2;
        op2.type = OpType::gate;
        op2.name = "x";
        op2.qubits = qubits;
        root.branches()[i]->add_op_after_branch(op2);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    for (uint_t i = 0; i < dim; i++) {
      cvector_t mdiag(dim, 0.);
      mdiag[i] = 1. / std::sqrt(meas_probs[i]);

      Operations::Op op;
      op.type = OpType::diagonal_matrix;
      op.qubits = qubits;
      op.params = mdiag;
      root.branches()[i]->add_op_after_branch(op);

      if (final_state >= 0 && final_state != (int_t)i) {
        // build vectorized permutation matrix
        cvector_t perm(dim * dim, 0.);
        perm[final_state * dim + i] = 1.;
        perm[i * dim + final_state] = 1.;
        for (size_t j = 0; j < dim; j++) {
          if ((int_t)j != final_state && j != i)
            perm[j * dim + j] = 1.;
        }
        Operations::Op op2;
        op2.type = OpType::matrix;
        op2.qubits = qubits;
        op2.mats.push_back(Utils::devectorize_matrix(perm));
        root.branches()[i]->add_op_after_branch(op2);
      }
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_measure(CircuitExecutor::Branch &root,
                                      const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister) {
  rvector_t probs = sample_measure_with_prob(root, qubits);

  // save result to cregs
  for (uint_t i = 0; i < probs.size(); i++) {
    const reg_t outcome = Utils::int2reg(i, 2, qubits.size());
    root.branches()[i]->creg().store_measure(outcome, cmemory, cregister);
  }

  measure_reset_update(root, qubits, -1, probs);
}

template <class state_t>
std::vector<reg_t>
Executor<state_t>::sample_measure(state_t &state, const reg_t &qubits,
                                  uint_t shots,
                                  std::vector<RngEngine> &rng) const {
  uint_t i;
  std::vector<double> rnds;
  rnds.reserve(shots);

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng[i].rand(0, 1));

  bool flg = state.qreg().enable_batch(false);
  auto allbit_samples = state.qreg().sample_measure(rnds);
  state.qreg().enable_batch(flg);

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, Base::num_qubits_);
    reg_t sample;
    sample.reserve(qubits.size());
    for (uint_t qubit : qubits) {
      sample.push_back(allbit_sample[qubit]);
    }
    all_samples.push_back(sample);
  }
  return all_samples;
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class densmat_t>
void Executor<densmat_t>::apply_kraus(const reg_t &qubits,
                                      const std::vector<cmatrix_t> &kmats) {
  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        Base::states_[iChunk].qreg().apply_superop_matrix(
            qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
      }
    }
  } else {
    for (uint_t i = 0; i < Base::states_.size(); i++)
      Base::states_[i].qreg().apply_superop_matrix(
          qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
  }
}

//-----------------------------------------------------------------------
// Functions for multi-chunk distribution
//-----------------------------------------------------------------------
// swap between chunks
template <class densmat_t>
void Executor<densmat_t>::apply_chunk_swap(const reg_t &qubits) {
  uint_t q0, q1;
  q0 = qubits[0];
  q1 = qubits[1];

  std::swap(BasePar::qubit_map_[q0], BasePar::qubit_map_[q1]);

  if (qubits[0] >= BasePar::chunk_bits_) {
    q0 += BasePar::chunk_bits_;
  }
  if (qubits[1] >= BasePar::chunk_bits_) {
    q1 += BasePar::chunk_bits_;
  }
  reg_t qs0 = {{q0, q1}};
  BasePar::apply_chunk_swap(qs0);

  if (qubits[0] >= BasePar::chunk_bits_) {
    q0 += (Base::num_qubits_ - BasePar::chunk_bits_);
  } else {
    q0 += BasePar::chunk_bits_;
  }
  if (qubits[1] >= BasePar::chunk_bits_) {
    q1 += (Base::num_qubits_ - BasePar::chunk_bits_);
  } else {
    q1 += BasePar::chunk_bits_;
  }
  reg_t qs1 = {{q0, q1}};
  BasePar::apply_chunk_swap(qs1);
}

template <class densmat_t>
void Executor<densmat_t>::apply_multi_chunk_swap(const reg_t &qubits) {
  reg_t qubits_density;

  for (uint_t i = 0; i < qubits.size(); i += 2) {
    uint_t q0, q1;
    q0 = qubits[i * 2];
    q1 = qubits[i * 2 + 1];

    std::swap(BasePar::qubit_map_[q0], BasePar::qubit_map_[q1]);

    if (q1 >= BasePar::chunk_bits_) {
      q1 += BasePar::chunk_bits_;
    }
    qubits_density.push_back(q0);
    qubits_density.push_back(q1);

    q0 += BasePar::chunk_bits_;
    if (q1 >= BasePar::chunk_bits_) {
      q1 += (Base::num_qubits_ - BasePar::chunk_bits_ * 2);
    }
  }

  BasePar::apply_multi_chunk_swap(qubits_density);
}

//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
