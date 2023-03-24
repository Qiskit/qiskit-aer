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

#ifndef _densitymatrix_parallel_executor_hpp_
#define _densitymatrix_parallel_executor_hpp_

#include "simulators/parallel_executor.hpp"

namespace AER {

namespace DensityMatrix {

//=========================================================================
// Parallel executor for DensityMatrix
//=========================================================================
template <class densmat_t>
class ParallelExecutor : public Executor::ParallelExecutor<densmat_t> {
  using BaseExecutor = Executor::ParallelExecutor<densmat_t>;

protected:
public:
  ParallelExecutor() {}
  virtual ~ParallelExecutor() {}

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------
  void set_config(const Config &config) override;

  // apply parallel operations
  void apply_parallel_op(const Operations::Op &op, ExperimentResult &result,
                         RngEngine &rng, bool final_op) override;

  // Initializes an n-qubit state to the all |0> state
  void initialize_qreg(uint_t num_qubits) override;

  auto move_to_matrix();
  auto copy_to_matrix();

protected:
  template <typename list_t>
  void initialize_from_vector(const list_t &vec);

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

  // scale for density matrix = 2
  // this function is used in the base class to scale chunk qubits for
  // multi-chunk distribution
  uint_t qubit_scale(void) override { return 2; }

  //-----------------------------------------------------------------------
  // Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
  // swap between chunks
  void apply_chunk_swap(const reg_t &qubits) override;

  // apply multiple swaps between chunks
  void apply_multi_chunk_swap(const reg_t &qubits) override;
};

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------
template <class densmat_t>
void ParallelExecutor<densmat_t>::initialize_qreg(uint_t num_qubits) {
  for (int_t i = 0; i < BaseExecutor::states_.size(); i++) {
    BaseExecutor::states_[i].qreg().set_num_qubits(BaseExecutor::chunk_bits_);
  }

  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
           iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
        if (BaseExecutor::global_state_index_ + iChunk == 0) {
          BaseExecutor::states_[iChunk].qreg().initialize();
        } else {
          BaseExecutor::states_[iChunk].qreg().zero();
        }
      }
    }
  } else {
    for (int_t i = 0; i < BaseExecutor::states_.size(); i++) {
      if (BaseExecutor::global_state_index_ + i == 0) {
        BaseExecutor::states_[i].qreg().initialize();
      } else {
        BaseExecutor::states_[i].qreg().zero();
      }
    }
  }
}

template <class densmat_t>
template <typename list_t>
void ParallelExecutor<densmat_t>::initialize_from_vector(const list_t &vec) {
  if ((1ull << (BaseExecutor::num_qubits_ * 2)) == vec.size()) {
    BaseExecutor::initialize_from_vector(vec);
  } else if ((1ull << (BaseExecutor::num_qubits_ * 2)) ==
             vec.size() * vec.size()) {
    int_t iChunk;
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
             iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
          uint_t irow_chunk =
              ((iChunk + BaseExecutor::global_state_index_) >>
               ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)))
              << (BaseExecutor::chunk_bits_);
          uint_t icol_chunk = ((iChunk + BaseExecutor::global_state_index_) &
                               ((1ull << ((BaseExecutor::num_qubits_ -
                                           BaseExecutor::chunk_bits_))) -
                                1))
                              << (BaseExecutor::chunk_bits_);

          // copy part of state for this chunk
          uint_t i, row, col;
          list_t vec1(1ull << BaseExecutor::chunk_bits_);
          list_t vec2(1ull << BaseExecutor::chunk_bits_);

          for (i = 0; i < (1ull << BaseExecutor::chunk_bits_); i++) {
            vec1[i] = vec[(irow_chunk << BaseExecutor::chunk_bits_) + i];
            vec2[i] =
                std::conj(vec[(icol_chunk << BaseExecutor::chunk_bits_) + i]);
          }
          BaseExecutor::states_[iChunk].qreg().initialize_from_vector(
              AER::Utils::tensor_product(vec1, vec2));
        }
      }
    } else {
      for (iChunk = 0; iChunk < BaseExecutor::states_.size(); iChunk++) {
        uint_t irow_chunk =
            ((iChunk + BaseExecutor::global_state_index_) >>
             ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)))
            << (BaseExecutor::chunk_bits_);
        uint_t icol_chunk = ((iChunk + BaseExecutor::global_state_index_) &
                             ((1ull << ((BaseExecutor::num_qubits_ -
                                         BaseExecutor::chunk_bits_))) -
                              1))
                            << (BaseExecutor::chunk_bits_);

        // copy part of state for this chunk
        uint_t i, row, col;
        list_t vec1(1ull << BaseExecutor::chunk_bits_);
        list_t vec2(1ull << BaseExecutor::chunk_bits_);

        for (i = 0; i < (1ull << BaseExecutor::chunk_bits_); i++) {
          vec1[i] = vec[(irow_chunk << BaseExecutor::chunk_bits_) + i];
          vec2[i] =
              std::conj(vec[(icol_chunk << BaseExecutor::chunk_bits_) + i]);
        }
        BaseExecutor::states_[iChunk].qreg().initialize_from_vector(
            AER::Utils::tensor_product(vec1, vec2));
      }
    }
  } else {
    throw std::runtime_error(
        "DensityMatrixChunk::initialize input vector is incorrect length. "
        "Expected: " +
        std::to_string((1ull << (BaseExecutor::num_qubits_ * 2))) +
        " Received: " + std::to_string(vec.size()));
  }
}

template <class densmat_t>
auto ParallelExecutor<densmat_t>::move_to_matrix() {
  return BaseExecutor::apply_to_matrix(false);
}

template <class densmat_t>
auto ParallelExecutor<densmat_t>::copy_to_matrix() {
  return BaseExecutor::apply_to_matrix(true);
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class densmat_t>
void ParallelExecutor<densmat_t>::set_config(const Config &config) {
  BaseExecutor::set_config(config);
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_parallel_op(const Operations::Op &op,
                                                    ExperimentResult &result,
                                                    RngEngine &rng,
                                                    bool final_ops) {
  if (BaseExecutor::states_[0].creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::barrier:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::reset:
      apply_reset(op.qubits);
      break;
    case Operations::OpType::measure:
      apply_measure(op.qubits, op.memory, op.registers, rng);
      break;
    case Operations::OpType::bfunc:
      BaseExecutor::apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BaseExecutor::apply_roerror(op, rng);
      break;
    case Operations::OpType::kraus:
      apply_kraus(op.qubits, op.mats);
      break;
    case Operations::OpType::set_statevec:
      initialize_from_vector(op.params);
      break;
    case Operations::OpType::set_densmat:
      BaseExecutor::initialize_from_matrix(op.mats[0]);
      break;
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      BaseExecutor::apply_save_expval(op, result);
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
      throw std::invalid_argument(
          "DensityMatrix::State::invalid instruction \'" + op.name + "\'.");
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_save_probs(const Operations::Op &op,
                                                   ExperimentResult &result) {
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    result.save_data_average(
        BaseExecutor::states_[0].creg(), op.string_params[0],
        Utils::vec2ket(probs, BaseExecutor::json_chop_threshold_, 16), op.type,
        op.save_type);
  } else {
    result.save_data_average(BaseExecutor::states_[0].creg(),
                             op.string_params[0], std::move(probs), op.type,
                             op.save_type);
  }
}

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_save_amplitudes_sq(
    const Operations::Op &op, ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes_sq instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  rvector_t amps_sq(size);

  int_t iChunk;
#pragma omp parallel for if (BaseExecutor::chunk_omp_parallel_) private(iChunk)
  for (iChunk = 0; iChunk < BaseExecutor::states_.size(); iChunk++) {
    uint_t irow, icol;
    irow = (BaseExecutor::global_state_index_ + iChunk) >>
           ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
    icol = (BaseExecutor::global_state_index_ + iChunk) -
           (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));
    if (irow != icol)
      continue;

    for (int_t i = 0; i < size; ++i) {
      uint_t idx = BaseExecutor::mapped_index(op.int_params[i]);
      if (idx >= (irow << BaseExecutor::chunk_bits_) &&
          idx < ((irow + 1) << BaseExecutor::chunk_bits_))
        amps_sq[i] = BaseExecutor::states_[iChunk].qreg().probability(
            idx - (irow << BaseExecutor::chunk_bits_));
    }
  }
#ifdef AER_MPI
  BaseExecutor::reduce_sum(amps_sq);
#endif

  result.save_data_average(BaseExecutor::states_[0].creg(), op.string_params[0],
                           std::move(amps_sq), op.type, op.save_type);
}

template <class densmat_t>
double ParallelExecutor<densmat_t>::expval_pauli(const reg_t &qubits,
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
    if (qubits[i] < BaseExecutor::chunk_bits_) {
      qubits_in_chunk.push_back(qubits[i]);
      pauli_in_chunk.push_back(pauli[n - i - 1]);
    } else {
      qubits_out_chunk.push_back(qubits[i]);
      pauli_out_chunk.push_back(pauli[n - i - 1]);
    }
  }

  int_t nrows =
      1ull << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));

  if (qubits_out_chunk.size() > 0) { // there are bits out of chunk
    std::complex<double> phase = 1.0;

    std::reverse(pauli_out_chunk.begin(), pauli_out_chunk.end());
    std::reverse(pauli_in_chunk.begin(), pauli_in_chunk.end());

    uint_t x_mask, z_mask, num_y, x_max;
    std::tie(x_mask, z_mask, num_y, x_max) =
        AER::QV::pauli_masks_and_phase(qubits_out_chunk, pauli_out_chunk);

    z_mask >>= (BaseExecutor::chunk_bits_);
    if (x_mask != 0) {
      x_mask >>= (BaseExecutor::chunk_bits_);
      x_max -= (BaseExecutor::chunk_bits_);

      AER::QV::add_y_phase(num_y, phase);

      const uint_t mask_u = ~((1ull << (x_max + 1)) - 1);
      const uint_t mask_l = (1ull << x_max) - 1;

      for (i = 0; i < nrows / 2; i++) {
        uint_t irow = ((i << 1) & mask_u) | (i & mask_l);
        uint_t iChunk = (irow ^ x_mask) + irow * nrows;

        if (BaseExecutor::state_index_begin_[BaseExecutor::distributed_rank_] <=
                iChunk &&
            BaseExecutor::state_index_end_[BaseExecutor::distributed_rank_] >
                iChunk) { // on this process
          double sign = 2.0;
          if (z_mask && (AER::Utils::popcount(irow & z_mask) & 1))
            sign = -2.0;
          expval +=
              sign *
              BaseExecutor::states_[iChunk - BaseExecutor::global_state_index_]
                  .qreg()
                  .expval_pauli_non_diagonal_chunk(qubits_in_chunk,
                                                   pauli_in_chunk, phase);
        }
      }
    } else {
      for (i = 0; i < nrows; i++) {
        uint_t iChunk = i * (nrows + 1);
        if (BaseExecutor::state_index_begin_[BaseExecutor::distributed_rank_] <=
                iChunk &&
            BaseExecutor::state_index_end_[BaseExecutor::distributed_rank_] >
                iChunk) { // on this process
          double sign = 1.0;
          if (z_mask && (AER::Utils::popcount(i & z_mask) & 1))
            sign = -1.0;
          expval +=
              sign *
              BaseExecutor::states_[iChunk - BaseExecutor::global_state_index_]
                  .qreg()
                  .expval_pauli(qubits_in_chunk, pauli_in_chunk, 1.0);
        }
      }
    }
  } else { // all bits are inside chunk
    for (i = 0; i < nrows; i++) {
      uint_t iChunk = i * (nrows + 1);
      if (BaseExecutor::state_index_begin_[BaseExecutor::distributed_rank_] <=
              iChunk &&
          BaseExecutor::state_index_end_[BaseExecutor::distributed_rank_] >
              iChunk) { // on this process
        expval +=
            BaseExecutor::states_[iChunk - BaseExecutor::global_state_index_]
                .qreg()
                .expval_pauli(qubits, pauli, 1.0);
      }
    }
  }

#ifdef AER_MPI
  BaseExecutor::reduce_sum(expval);
#endif
  return expval;
}

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_save_density_matrix(
    const Operations::Op &op, ExperimentResult &result, bool last_op) {
  result.save_data_average(BaseExecutor::states_[0].creg(), op.string_params[0],
                           reduced_density_matrix(op.qubits, last_op), op.type,
                           op.save_type);
}

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_save_state(const Operations::Op &op,
                                                   ExperimentResult &result,
                                                   bool last_op) {
  if (op.qubits.size() != BaseExecutor::num_qubits_) {
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
    result.save_data_average(BaseExecutor::states_[0].creg(), key,
                             move_to_matrix(), Operations::OpType::save_densmat,
                             save_type);
  } else {
    result.save_data_average(BaseExecutor::states_[0].creg(), key,
                             copy_to_matrix(), Operations::OpType::save_densmat,
                             save_type);
  }
}

template <class densmat_t>
cmatrix_t
ParallelExecutor<densmat_t>::reduced_density_matrix(const reg_t &qubits,
                                                    bool last_op) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    std::complex<double> sum = 0.0;
    for (int_t i = 0; i < BaseExecutor::states_.size(); i++) {
      sum += BaseExecutor::states_[i].qreg().trace();
    }
#ifdef AER_MPI
    BaseExecutor::reduce_sum(sum);
#endif
    reduced_state[0] = sum;
  } else {
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    if ((qubits.size() == BaseExecutor::num_qubits_) &&
        (qubits == qubits_sorted)) {
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
cmatrix_t ParallelExecutor<densmat_t>::reduced_density_matrix_helper(
    const reg_t &qubits, const reg_t &qubits_sorted) {
  int_t iChunk;
  uint_t size = 1ull << (BaseExecutor::chunk_bits_ * 2);
  uint_t mask = (1ull << (BaseExecutor::chunk_bits_)) - 1;
  uint_t num_threads = BaseExecutor::states_[0].qreg().get_omp_threads();

  size_t size_required =
      (sizeof(std::complex<double>) << (qubits.size() * 2)) +
      (sizeof(std::complex<double>) << (BaseExecutor::chunk_bits_ * 2)) *
          BaseExecutor::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store density matrix"));
  }
  cmatrix_t reduced_state(1ull << qubits.size(), 1ull << qubits.size(), true);

  if (BaseExecutor::distributed_rank_ == 0) {
    auto tmp = BaseExecutor::states_[0].copy_to_matrix();
    for (iChunk = 0; iChunk < BaseExecutor::num_global_states_; iChunk++) {
      int_t i;
      uint_t irow_chunk =
          (iChunk >> ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)))
          << BaseExecutor::chunk_bits_;
      uint_t icol_chunk = (iChunk & ((1ull << ((BaseExecutor::num_qubits_ -
                                                BaseExecutor::chunk_bits_))) -
                                     1))
                          << BaseExecutor::chunk_bits_;

      if (iChunk < BaseExecutor::num_local_states_)
        tmp = BaseExecutor::states_[iChunk].qreg().copy_to_matrix();
#ifdef AER_MPI
      else
        BaseExecutor::recv_data(tmp.data(), size, 0, iChunk);
#endif
#pragma omp parallel for if (num_threads > 1) num_threads(num_threads)
      for (i = 0; i < size; i++) {
        uint_t irow = (i >> (BaseExecutor::chunk_bits_)) + irow_chunk;
        uint_t icol = (i & mask) + icol_chunk;
        uint_t irow_out = 0;
        uint_t icol_out = 0;
        int j;
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
    for (iChunk = 0; iChunk < BaseExecutor::num_global_states_; iChunk++) {
      uint_t iProc = BaseExecutor::get_process_by_chunk(iChunk);
      if (iProc == BaseExecutor::distributed_rank_) {
        auto tmp =
            BaseExecutor::states_[iChunk - BaseExecutor::global_state_index_]
                .qreg()
                .copy_to_matrix();
        BaseExecutor::send_data(tmp.data(), size, iChunk, 0);
      }
    }
#endif
  }

  return reduced_state;
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_measure(const reg_t &qubits,
                                                const reg_t &cmemory,
                                                const reg_t &cregister,
                                                RngEngine &rng) {
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BaseExecutor::store_measure(outcome, cmemory, cregister);
}

template <class densmat_t>
rvector_t
ParallelExecutor<densmat_t>::measure_probs(const reg_t &qubits) const {
  uint_t dim = 1ull << qubits.size();
  rvector_t sum(dim, 0.0);
  int_t i, j, k;
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;

  for (i = 0; i < qubits.size(); i++) {
    if (qubits[i] < BaseExecutor::chunk_bits_) {
      qubits_in_chunk.push_back(qubits[i]);
    } else {
      qubits_out_chunk.push_back(qubits[i]);
    }
  }

  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for private(i, j, k)
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (i = BaseExecutor::top_state_of_group_[ig];
           i < BaseExecutor::top_state_of_group_[ig + 1]; i++) {
        uint_t irow, icol;
        irow = (BaseExecutor::global_state_index_ + i) >>
               ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
        icol =
            (BaseExecutor::global_state_index_ + i) -
            (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));

        if (irow == icol) { // diagonal chunk
          if (qubits_in_chunk.size() > 0) {
            auto chunkSum =
                BaseExecutor::states_[i].qreg().probabilities(qubits_in_chunk);
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
                  if (qubits[k] < (BaseExecutor::chunk_bits_)) {
                    idx += (((j >> i_in) & 1) << k);
                    i_in++;
                  } else {
                    if ((((i + BaseExecutor::global_state_index_)
                          << (BaseExecutor::chunk_bits_)) >>
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
            auto tr = std::real(BaseExecutor::states_[i].qreg().trace());
            int idx = 0;
            for (k = 0; k < qubits_out_chunk.size(); k++) {
              if ((((i + BaseExecutor::global_state_index_)
                    << (BaseExecutor::chunk_bits_)) >>
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
    for (i = 0; i < BaseExecutor::states_.size(); i++) {
      uint_t irow, icol;
      irow = (BaseExecutor::global_state_index_ + i) >>
             ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
      icol =
          (BaseExecutor::global_state_index_ + i) -
          (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));

      if (irow == icol) { // diagonal chunk
        if (qubits_in_chunk.size() > 0) {
          auto chunkSum =
              BaseExecutor::states_[i].qreg().probabilities(qubits_in_chunk);
          if (qubits_in_chunk.size() == qubits.size()) {
            for (j = 0; j < dim; j++) {
              sum[j] += chunkSum[j];
            }
          } else {
            for (j = 0; j < chunkSum.size(); j++) {
              int idx = 0;
              int i_in = 0;
              for (k = 0; k < qubits.size(); k++) {
                if (qubits[k] < (BaseExecutor::chunk_bits_)) {
                  idx += (((j >> i_in) & 1) << k);
                  i_in++;
                } else {
                  if ((((i + BaseExecutor::global_state_index_)
                        << (BaseExecutor::chunk_bits_)) >>
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
          auto tr = std::real(BaseExecutor::states_[i].qreg().trace());
          int idx = 0;
          for (k = 0; k < qubits_out_chunk.size(); k++) {
            if ((((i + BaseExecutor::global_state_index_)
                  << (BaseExecutor::chunk_bits_)) >>
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
  BaseExecutor::reduce_sum(sum);
#endif

  return sum;
}

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_reset(const reg_t &qubits) {
  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
           iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
        BaseExecutor::states_[iChunk].qreg().apply_reset(qubits);
      }
    }
  } else {
    for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
      BaseExecutor::states_[i].qreg().apply_reset(qubits);
  }
}

template <class densmat_t>
std::pair<uint_t, double>
ParallelExecutor<densmat_t>::sample_measure_with_prob(const reg_t &qubits,
                                                      RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class densmat_t>
void ParallelExecutor<densmat_t>::measure_reset_update(const reg_t &qubits,
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
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
          BaseExecutor::states_[i].qreg().apply_diagonal_unitary_matrix(qubits,
                                                                        mdiag);
      }
    } else {
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        BaseExecutor::states_[i].qreg().apply_diagonal_unitary_matrix(qubits,
                                                                      mdiag);
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if (qubits[0] < BaseExecutor::chunk_bits_) {
        if (BaseExecutor::chunk_omp_parallel_ &&
            BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
          for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
            for (int_t i = BaseExecutor::top_state_of_group_[ig];
                 i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
              BaseExecutor::states_[i].qreg().apply_x(qubits[0]);
          }
        } else {
          for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
            BaseExecutor::states_[i].qreg().apply_x(qubits[0]);
        }
      } else {
        BaseExecutor::apply_chunk_x(qubits[0]);
        BaseExecutor::apply_chunk_x(qubits[0] + BaseExecutor::chunk_bits_);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
          BaseExecutor::states_[i].qreg().apply_diagonal_unitary_matrix(qubits,
                                                                        mdiag);
      }
    } else {
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        BaseExecutor::states_[i].qreg().apply_diagonal_unitary_matrix(qubits,
                                                                      mdiag);
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

      for (int_t i = 0; i < qubits.size(); i++) {
        if (qubits[i] < BaseExecutor::chunk_bits_) {
          qubits_in_chunk.push_back(qubits[i]);
        } else {
          qubits_out_chunk.push_back(qubits[i]);
        }
      }
      if (qubits_in_chunk.size() > 0) { // in chunk exchange
        if (BaseExecutor::chunk_omp_parallel_ &&
            BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
          for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
            for (int_t i = BaseExecutor::top_state_of_group_[ig];
                 i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
              BaseExecutor::states_[i].qreg().apply_unitary_matrix(qubits,
                                                                   perm);
          }
        } else {
          for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
            BaseExecutor::states_[i].qreg().apply_unitary_matrix(qubits, perm);
        }
      }
      if (qubits_out_chunk.size() > 0) { // out of chunk exchange
        for (int_t i = 0; i < qubits_out_chunk.size(); i++) {
          BaseExecutor::apply_chunk_x(qubits_out_chunk[i]);
          BaseExecutor::apply_chunk_x(
              qubits_out_chunk[i] +
              (BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
        }
      }
    }
  }
}

template <class densmat_t>
std::vector<reg_t>
ParallelExecutor<densmat_t>::sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) const {
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  for (uint_t i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));
  reg_t allbit_samples(shots, 0);

  int_t i, j;
  std::vector<double> chunkSum(BaseExecutor::states_.size() + 1, 0);
  double sum, localSum;
  // calculate per chunk sum
  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for private(i)
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (i = BaseExecutor::top_state_of_group_[ig];
           i < BaseExecutor::top_state_of_group_[ig + 1]; i++) {
        uint_t irow, icol;
        irow = (BaseExecutor::global_state_index_ + i) >>
               ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
        icol =
            (BaseExecutor::global_state_index_ + i) -
            (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));
        if (irow == icol) // only diagonal chunk has probabilities
          chunkSum[i] = std::real(BaseExecutor::states_[i].qreg().trace());
        else
          chunkSum[i] = 0.0;
      }
    }
  } else {
    for (i = 0; i < BaseExecutor::states_.size(); i++) {
      uint_t irow, icol;
      irow = (BaseExecutor::global_state_index_ + i) >>
             ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
      icol =
          (BaseExecutor::global_state_index_ + i) -
          (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));
      if (irow == icol) // only diagonal chunk has probabilities
        chunkSum[i] = std::real(BaseExecutor::states_[i].qreg().trace());
      else
        chunkSum[i] = 0.0;
    }
  }
  localSum = 0.0;
  for (i = 0; i < BaseExecutor::states_.size(); i++) {
    sum = localSum;
    localSum += chunkSum[i];
    chunkSum[i] = sum;
  }
  chunkSum[BaseExecutor::states_.size()] = localSum;

  double globalSum = 0.0;
  if (BaseExecutor::nprocs_ > 1) {
    std::vector<double> procTotal(BaseExecutor::nprocs_);

    for (i = 0; i < BaseExecutor::nprocs_; i++) {
      procTotal[i] = localSum;
    }
    BaseExecutor::gather_value(procTotal);

    for (i = 0; i < BaseExecutor::myrank_; i++) {
      globalSum += procTotal[i];
    }
  }

  reg_t local_samples(shots, 0);

  // get rnds positions for each chunk
  for (i = 0; i < BaseExecutor::states_.size(); i++) {
    uint_t irow, icol;
    irow = (BaseExecutor::global_state_index_ + i) >>
           ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
    icol = (BaseExecutor::global_state_index_ + i) -
           (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));
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
      auto chunkSamples = BaseExecutor::states_[i].qreg().sample_measure(vRnd);
      uint_t ir;
      ir = (BaseExecutor::global_state_index_ + i) >>
           ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));

      for (j = 0; j < chunkSamples.size(); j++) {
        local_samples[vIdx[j]] =
            (ir << BaseExecutor::chunk_bits_) + chunkSamples[j];
      }
    }
  }

#ifdef AER_MPI
  BaseExecutor::reduce_sum(local_samples);
#endif
  allbit_samples = local_samples;

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, BaseExecutor::num_qubits_);
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
void ParallelExecutor<densmat_t>::apply_kraus(
    const reg_t &qubits, const std::vector<cmatrix_t> &kmats) {
  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
           iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
        BaseExecutor::states_[iChunk].qreg().apply_superop_matrix(
            qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
      }
    }
  } else {
    for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
      BaseExecutor::states_[i].qreg().apply_superop_matrix(
          qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
  }
}

//-----------------------------------------------------------------------
// Functions for multi-chunk distribution
//-----------------------------------------------------------------------
// swap between chunks
template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_chunk_swap(const reg_t &qubits) {
  uint_t q0, q1;
  q0 = qubits[0];
  q1 = qubits[1];

  std::swap(BaseExecutor::qubit_map_[q0], BaseExecutor::qubit_map_[q1]);

  if (qubits[0] >= BaseExecutor::chunk_bits_) {
    q0 += BaseExecutor::chunk_bits_;
  }
  if (qubits[1] >= BaseExecutor::chunk_bits_) {
    q1 += BaseExecutor::chunk_bits_;
  }
  reg_t qs0 = {{q0, q1}};
  BaseExecutor::apply_chunk_swap(qs0);

  if (qubits[0] >= BaseExecutor::chunk_bits_) {
    q0 += (BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_);
  } else {
    q0 += BaseExecutor::chunk_bits_;
  }
  if (qubits[1] >= BaseExecutor::chunk_bits_) {
    q1 += (BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_);
  } else {
    q1 += BaseExecutor::chunk_bits_;
  }
  reg_t qs1 = {{q0, q1}};
  BaseExecutor::apply_chunk_swap(qs1);
}

template <class densmat_t>
void ParallelExecutor<densmat_t>::apply_multi_chunk_swap(const reg_t &qubits) {
  reg_t qubits_density;

  for (int_t i = 0; i < qubits.size(); i += 2) {
    uint_t q0, q1;
    q0 = qubits[i * 2];
    q1 = qubits[i * 2 + 1];

    std::swap(BaseExecutor::qubit_map_[q0], BaseExecutor::qubit_map_[q1]);

    if (q1 >= BaseExecutor::chunk_bits_) {
      q1 += BaseExecutor::chunk_bits_;
    }
    qubits_density.push_back(q0);
    qubits_density.push_back(q1);

    q0 += BaseExecutor::chunk_bits_;
    if (q1 >= BaseExecutor::chunk_bits_) {
      q1 += (BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_ * 2);
    }
  }

  BaseExecutor::apply_multi_chunk_swap(qubits_density);
}

//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
