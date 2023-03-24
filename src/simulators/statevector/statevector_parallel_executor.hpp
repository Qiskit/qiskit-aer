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

#ifndef _statevector_parallel_executor_hpp_
#define _statevector_parallel_executor_hpp_

#include "simulators/parallel_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace Statevector {

//-------------------------------------------------------------------------
// Parallel executor for statevector
//-------------------------------------------------------------------------
template <class statevec_t>
class ParallelExecutor : public Executor::ParallelExecutor<statevec_t> {
  using BaseExecutor = Executor::ParallelExecutor<statevec_t>;

protected:
public:
  ParallelExecutor() {}
  virtual ~ParallelExecutor() {}

protected:
  void set_config(const Config &config) override;

  // apply parallel operations
  void apply_parallel_op(const Operations::Op &op, ExperimentResult &result,
                         RngEngine &rng, bool final_op) override;

  // Initializes an n-qubit state to the all |0> state
  void initialize_qreg(uint_t num_qubits) override;

  auto move_to_vector(void);
  auto copy_to_vector(void);

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------
  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                     const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const reg_t &qubits, const cvector_t &params,
                        RngEngine &rng);

  void initialize_from_vector(const cvector_t &params);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the statevector simulator
  // If `last_op` is True this will use move semantics to move the simulator
  // state to the results, otherwise it will use copy semantics to leave
  // the current simulator state unchanged.
  void apply_save_statevector(const Operations::Op &op,
                              ExperimentResult &result, bool last_op);

  // Save the current state of the statevector simulator as a ket-form map.
  void apply_save_statevector_dict(const Operations::Op &op,
                                   ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result);

  // Helper function for computing expectation value
  double expval_pauli(const reg_t &qubits, const std::string &pauli) override;
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

  // Return the reduced density matrix for the simulator
  cmatrix_t density_matrix(const reg_t &qubits);

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                    RngEngine &rng) const override;
};

template <class state_t>
void ParallelExecutor<state_t>::set_config(const Config &config) {
  BaseExecutor::set_config(config);
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_parallel_op(const Operations::Op &op,
                                                     ExperimentResult &result,
                                                     RngEngine &rng,
                                                     bool final_op) {
  // temporary : this is for statevector
  if (BaseExecutor::states_[0].creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::barrier:
    case Operations::OpType::nop:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::reset:
      apply_reset(op.qubits, rng);
      break;
    case Operations::OpType::initialize:
      apply_initialize(op.qubits, op.params, rng);
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
      apply_kraus(op.qubits, op.mats, rng);
      break;
    case Operations::OpType::set_statevec:
      initialize_from_vector(op.params);
      break;
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      BaseExecutor::apply_save_expval(op, result);
      break;
    case Operations::OpType::save_densmat:
      apply_save_density_matrix(op, result);
      break;
    case Operations::OpType::save_state:
    case Operations::OpType::save_statevec:
      apply_save_statevector(op, result, final_op);
      break;
    case Operations::OpType::save_statevec_dict:
      apply_save_statevector_dict(op, result);
      break;
    case Operations::OpType::save_probs:
    case Operations::OpType::save_probs_ket:
      apply_save_probs(op, result);
      break;
    case Operations::OpType::save_amps:
    case Operations::OpType::save_amps_sq:
      apply_save_amplitudes(op, result);
      break;
    default:
      throw std::invalid_argument("ParallelExecutor::invalid instruction \'" +
                                  op.name + "\'.");
    }
  }
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::initialize_qreg(uint_t num_qubits) {
  int_t i;

  for (i = 0; i < BaseExecutor::states_.size(); i++) {
    BaseExecutor::states_[i].qreg().set_num_qubits(BaseExecutor::chunk_bits_);
  }

  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
           iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
        if (BaseExecutor::global_state_index_ + iChunk == 0 ||
            this->num_qubits_ == this->chunk_bits_) {
          BaseExecutor::states_[iChunk].qreg().initialize();
        } else {
          BaseExecutor::states_[iChunk].qreg().zero();
        }
      }
    }
  } else {
    for (i = 0; i < BaseExecutor::states_.size(); i++) {
      if (BaseExecutor::global_state_index_ + i == 0 ||
          this->num_qubits_ == this->chunk_bits_) {
        BaseExecutor::states_[i].qreg().initialize();
      } else {
        BaseExecutor::states_[i].qreg().zero();
      }
    }
  }

  BaseExecutor::apply_global_phase();
}

template <class statevec_t>
auto ParallelExecutor<statevec_t>::move_to_vector(void) {
  size_t size_required =
      2 * (sizeof(std::complex<double>) << BaseExecutor::num_qubits_) +
      (sizeof(std::complex<double>) << BaseExecutor::chunk_bits_) *
          BaseExecutor::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store states"));
  }
  int_t iChunk;
  auto state = BaseExecutor::states_[0].qreg().move_to_vector();
  state.resize(BaseExecutor::num_local_states_ << BaseExecutor::chunk_bits_);

#pragma omp parallel for if (BaseExecutor::chunk_omp_parallel_) private(iChunk)
  for (iChunk = 1; iChunk < BaseExecutor::states_.size(); iChunk++) {
    auto tmp = BaseExecutor::states_[iChunk].qreg().move_to_vector();
    uint_t j, offset = iChunk << BaseExecutor::chunk_bits_;
    for (j = 0; j < tmp.size(); j++) {
      state[offset + j] = tmp[j];
    }
  }

#ifdef AER_MPI
  BaseExecutor::gather_state(state);
#endif
  return state;
}

template <class statevec_t>
auto ParallelExecutor<statevec_t>::copy_to_vector(void) {
  size_t size_required =
      2 * (sizeof(std::complex<double>) << BaseExecutor::num_qubits_) +
      (sizeof(std::complex<double>) << BaseExecutor::chunk_bits_) *
          BaseExecutor::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store states"));
  }
  int_t iChunk;
  auto state = BaseExecutor::states_[0].qreg().copy_to_vector();
  state.resize(BaseExecutor::num_local_states_ << BaseExecutor::chunk_bits_);

#pragma omp parallel for if (BaseExecutor::chunk_omp_parallel_) private(iChunk)
  for (iChunk = 1; iChunk < BaseExecutor::states_.size(); iChunk++) {
    auto tmp = BaseExecutor::states_[iChunk].qreg().copy_to_vector();
    uint_t j, offset = iChunk << BaseExecutor::chunk_bits_;
    for (j = 0; j < tmp.size(); j++) {
      state[offset + j] = tmp[j];
    }
  }

#ifdef AER_MPI
  BaseExecutor::gather_state(state);
#endif
  return state;
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_save_probs(const Operations::Op &op,
                                                    ExperimentResult &result) {
  // get probs as hexadecimal
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
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

template <class statevec_t>
double ParallelExecutor<statevec_t>::expval_pauli(const reg_t &qubits,
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

  if (qubits_out_chunk.size() > 0) { // there are bits out of chunk
    std::complex<double> phase = 1.0;

    std::reverse(pauli_out_chunk.begin(), pauli_out_chunk.end());
    std::reverse(pauli_in_chunk.begin(), pauli_in_chunk.end());

    uint_t x_mask, z_mask, num_y, x_max;
    std::tie(x_mask, z_mask, num_y, x_max) =
        AER::QV::pauli_masks_and_phase(qubits_out_chunk, pauli_out_chunk);

    AER::QV::add_y_phase(num_y, phase);

    if (x_mask != 0) { // pairing state is out of chunk
      bool on_same_process = true;
#ifdef AER_MPI
      int proc_bits = 0;
      uint_t procs = BaseExecutor::distributed_procs_;
      while (procs > 1) {
        if ((procs & 1) != 0) {
          proc_bits = -1;
          break;
        }
        proc_bits++;
        procs >>= 1;
      }
      if (x_mask & (~((1ull << (BaseExecutor::num_qubits_ - proc_bits)) - 1)) !=
                       0) { // data exchange between processes is required
        on_same_process = false;
      }
#endif

      x_mask >>= BaseExecutor::chunk_bits_;
      z_mask >>= BaseExecutor::chunk_bits_;
      x_max -= BaseExecutor::chunk_bits_;

      const uint_t mask_u = ~((1ull << (x_max + 1)) - 1);
      const uint_t mask_l = (1ull << x_max) - 1;
      if (on_same_process) {
        auto apply_expval_pauli_chunk = [this, x_mask, z_mask, x_max, mask_u,
                                         mask_l, qubits_in_chunk,
                                         pauli_in_chunk, phase](int_t iGroup) {
          double expval = 0.0;
          for (int_t iChunk = BaseExecutor::top_state_of_group_[iGroup];
               iChunk < BaseExecutor::top_state_of_group_[iGroup + 1];
               iChunk++) {
            uint_t pair_chunk = iChunk ^ x_mask;
            if (iChunk < pair_chunk) {
              uint_t z_count, z_count_pair;
              z_count = AER::Utils::popcount(iChunk & z_mask);
              z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

              expval +=
                  BaseExecutor::states_[iChunk -
                                        BaseExecutor::global_state_index_]
                      .qreg()
                      .expval_pauli(qubits_in_chunk, pauli_in_chunk,
                                    BaseExecutor::states_[pair_chunk].qreg(),
                                    z_count, z_count_pair, phase);
            }
          }
          return expval;
        };
        expval += Utils::apply_omp_parallel_for_reduction(
            (BaseExecutor::chunk_omp_parallel_ &&
             BaseExecutor::num_groups_ > 0),
            0, BaseExecutor::num_global_states_ / 2, apply_expval_pauli_chunk);
      } else {
        for (int_t i = 0; i < BaseExecutor::num_global_states_ / 2; i++) {
          uint_t iChunk = ((i << 1) & mask_u) | (i & mask_l);
          uint_t pair_chunk = iChunk ^ x_mask;
          uint_t iProc = BaseExecutor::get_process_by_chunk(pair_chunk);
          if (BaseExecutor::state_index_begin_
                      [BaseExecutor::distributed_rank_] <= iChunk &&
              BaseExecutor::state_index_end_[BaseExecutor::distributed_rank_] >
                  iChunk) { // on this process
            uint_t z_count, z_count_pair;
            z_count = AER::Utils::popcount(iChunk & z_mask);
            z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

            if (iProc == BaseExecutor::distributed_rank_) { // pair is on the
                                                            // same process
              expval +=
                  BaseExecutor::states_[iChunk -
                                        BaseExecutor::global_state_index_]
                      .qreg()
                      .expval_pauli(
                          qubits_in_chunk, pauli_in_chunk,
                          BaseExecutor::states_
                              [pair_chunk - BaseExecutor::global_state_index_]
                                  .qreg(),
                          z_count, z_count_pair, phase);
            } else {
              BaseExecutor::recv_chunk(
                  iChunk - BaseExecutor::global_state_index_, pair_chunk);
              // refer receive buffer to calculate expectation value
              expval += BaseExecutor::states_[iChunk -
                                              BaseExecutor::global_state_index_]
                            .qreg()
                            .expval_pauli(
                                qubits_in_chunk, pauli_in_chunk,
                                BaseExecutor::states_
                                    [iChunk - BaseExecutor::global_state_index_]
                                        .qreg(),
                                z_count, z_count_pair, phase);
            }
          } else if (iProc == BaseExecutor::distributed_rank_) { // pair is on
                                                                 // this process
            BaseExecutor::send_chunk(iChunk - BaseExecutor::global_state_index_,
                                     pair_chunk);
          }
        }
      }
    } else { // no exchange between chunks
      z_mask >>= BaseExecutor::chunk_bits_;
      if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for reduction(+ : expval)
        for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
          double e_tmp = 0.0;
          for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
               iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
            double sign = 1.0;
            if (z_mask &&
                (AER::Utils::popcount(
                     (iChunk + BaseExecutor::global_state_index_) & z_mask) &
                 1))
              sign = -1.0;
            e_tmp += sign * BaseExecutor::states_[iChunk].qreg().expval_pauli(
                                qubits_in_chunk, pauli_in_chunk);
          }
          expval += e_tmp;
        }
      } else {
        for (i = 0; i < BaseExecutor::states_.size(); i++) {
          double sign = 1.0;
          if (z_mask && (AER::Utils::popcount(
                             (i + BaseExecutor::global_state_index_) & z_mask) &
                         1))
            sign = -1.0;
          expval += sign * BaseExecutor::states_[i].qreg().expval_pauli(
                               qubits_in_chunk, pauli_in_chunk);
        }
      }
    }
  } else { // all bits are inside chunk
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for reduction(+ : expval)
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        double e_tmp = 0.0;
        for (int_t iChunk = BaseExecutor::top_state_of_group_[ig];
             iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++)
          e_tmp +=
              BaseExecutor::states_[iChunk].qreg().expval_pauli(qubits, pauli);
        expval += e_tmp;
      }
    } else {
      for (i = 0; i < BaseExecutor::states_.size(); i++)
        expval += BaseExecutor::states_[i].qreg().expval_pauli(qubits, pauli);
    }
  }

#ifdef AER_MPI
  BaseExecutor::reduce_sum(expval);
#endif
  return expval;
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_save_statevector(
    const Operations::Op &op, ExperimentResult &result, bool last_op) {
  if (op.qubits.size() != BaseExecutor::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "statevector" : op.string_params[0];

  if (last_op) {
    auto v = move_to_vector();
    result.save_data_pershot(BaseExecutor::states_[0].creg(), key, std::move(v),
                             Operations::OpType::save_statevec, op.save_type);
  } else {
    result.save_data_pershot(BaseExecutor::states_[0].creg(), key,
                             copy_to_vector(),
                             Operations::OpType::save_statevec, op.save_type);
  }
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_save_statevector_dict(
    const Operations::Op &op, ExperimentResult &result) {
  if (op.qubits.size() != BaseExecutor::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  auto vec = copy_to_vector();
  std::map<std::string, complex_t> result_state_ket;
  for (size_t k = 0; k < vec.size(); ++k) {
    if (std::abs(vec[k]) >= BaseExecutor::json_chop_threshold_) {
      std::string key = Utils::int2hex(k);
      result_state_ket.insert({key, vec[k]});
    }
  }
  result.save_data_pershot(BaseExecutor::states_[0].creg(), op.string_params[0],
                           std::move(result_state_ket), op.type, op.save_type);
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_save_density_matrix(
    const Operations::Op &op, ExperimentResult &result) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    double sum = 0.0;
    if (BaseExecutor::chunk_omp_parallel_) {
#pragma omp parallel for reduction(+ : sum)
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        sum += BaseExecutor::states_[i].qreg().norm();
    } else {
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        sum += BaseExecutor::states_[i].qreg().norm();
    }
#ifdef AER_MPI
    BaseExecutor::reduce_sum(sum);
#endif
    reduced_state[0] = sum;
  } else {
    reduced_state = density_matrix(op.qubits);
  }

  result.save_data_average(BaseExecutor::states_[0].creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_save_amplitudes(
    const Operations::Op &op, ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (int_t i = 0; i < size; ++i) {
      uint_t idx = BaseExecutor::mapped_index(op.int_params[i]);
      uint_t iChunk = idx >> BaseExecutor::chunk_bits_;
      amps[i] = 0.0;
      if (iChunk >= BaseExecutor::global_state_index_ &&
          iChunk < BaseExecutor::global_state_index_ +
                       BaseExecutor::states_.size()) {
        amps[i] =
            BaseExecutor::states_[iChunk - BaseExecutor::global_state_index_]
                .qreg()
                .get_state(idx - (iChunk << BaseExecutor::chunk_bits_));
      }
#ifdef AER_MPI
      complex_t amp = amps[i];
      BaseExecutor::reduce_sum(amp);
      amps[i] = amp;
#endif
    }
    result.save_data_pershot(BaseExecutor::states_[0].creg(),
                             op.string_params[0], std::move(amps), op.type,
                             op.save_type);
  } else {
    rvector_t amps_sq(size, 0);
    for (int_t i = 0; i < size; ++i) {
      uint_t idx = BaseExecutor::mapped_index(op.int_params[i]);
      uint_t iChunk = idx >> BaseExecutor::chunk_bits_;
      if (iChunk >= BaseExecutor::global_state_index_ &&
          iChunk < BaseExecutor::global_state_index_ +
                       BaseExecutor::states_.size()) {
        amps_sq[i] =
            BaseExecutor::states_[iChunk - BaseExecutor::global_state_index_]
                .qreg()
                .probability(idx - (iChunk << BaseExecutor::chunk_bits_));
      }
    }
#ifdef AER_MPI
    BaseExecutor::reduce_sum(amps_sq);
#endif
    result.save_data_average(BaseExecutor::states_[0].creg(),
                             op.string_params[0], std::move(amps_sq), op.type,
                             op.save_type);
  }
}

template <class statevec_t>
cmatrix_t ParallelExecutor<statevec_t>::density_matrix(const reg_t &qubits) {
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  auto vec = copy_to_vector();

  // Return full density matrix
  cmatrix_t densmat(DIM, DIM);
  if ((N == BaseExecutor::num_qubits_) && (qubits == qubits_sorted)) {
    const int_t mask = QV::MASKS[N];
#pragma omp parallel for
    for (int_t rowcol = 0; rowcol < int_t(DIM * DIM); ++rowcol) {
      const int_t row = rowcol >> N;
      const int_t col = rowcol & mask;
      densmat(row, col) = complex_t(vec[row]) * complex_t(std::conj(vec[col]));
    }
  } else {
    const size_t END = 1ULL << (BaseExecutor::num_qubits_ - N);
    // Initialize matrix values with first block
    {
      const auto inds = QV::indexes(qubits, qubits_sorted, 0);
      for (size_t row = 0; row < DIM; ++row)
        for (size_t col = 0; col < DIM; ++col) {
          densmat(row, col) =
              complex_t(vec[inds[row]]) * complex_t(std::conj(vec[inds[col]]));
        }
    }
    // Accumulate remaining blocks
    for (size_t k = 1; k < END; k++) {
      // store entries touched by U
      const auto inds = QV::indexes(qubits, qubits_sorted, k);
      for (size_t row = 0; row < DIM; ++row)
        for (size_t col = 0; col < DIM; ++col) {
          densmat(row, col) +=
              complex_t(vec[inds[row]]) * complex_t(std::conj(vec[inds[col]]));
        }
    }
  }
  return densmat;
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_measure(const reg_t &qubits,
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

template <class statevec_t>
rvector_t
ParallelExecutor<statevec_t>::measure_probs(const reg_t &qubits) const {
  uint_t dim = 1ull << qubits.size();
  rvector_t sum(dim, 0.0);
  int_t i, j, k;
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;

  Chunk::get_qubits_inout(BaseExecutor::chunk_bits_, qubits, qubits_in_chunk,
                          qubits_out_chunk);

  if (qubits_in_chunk.size() > 0) {
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for private(i, j, k)
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++) {
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
                if (qubits[k] < BaseExecutor::chunk_bits_) {
                  idx += (((j >> i_in) & 1) << k);
                  i_in++;
                } else {
                  if ((((i + BaseExecutor::global_state_index_)
                        << BaseExecutor::chunk_bits_) >>
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
        }
      }
    } else {
      for (i = 0; i < BaseExecutor::states_.size(); i++) {
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
              if (qubits[k] < BaseExecutor::chunk_bits_) {
                idx += (((j >> i_in) & 1) << k);
                i_in++;
              } else {
                if ((((i + BaseExecutor::global_state_index_)
                      << BaseExecutor::chunk_bits_) >>
                     qubits[k]) &
                    1) {
                  idx += 1ull << k;
                }
              }
            }
            sum[idx] += chunkSum[j];
          }
        }
      }
    }
  } else { // there is no bit in chunk
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for private(i, j, k)
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++) {
          auto nr = std::real(BaseExecutor::states_[i].qreg().norm());
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
          sum[idx] += nr;
        }
      }
    } else {
      for (i = 0; i < BaseExecutor::states_.size(); i++) {
        auto nr = std::real(BaseExecutor::states_[i].qreg().norm());
        int idx = 0;
        for (k = 0; k < qubits_out_chunk.size(); k++) {
          if ((((i + BaseExecutor::global_state_index_)
                << (BaseExecutor::chunk_bits_)) >>
               qubits_out_chunk[k]) &
              1) {
            idx += 1ull << k;
          }
        }
        sum[idx] += nr;
      }
    }
  }

#ifdef AER_MPI
  BaseExecutor::reduce_sum(sum);
#endif

  return sum;
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_reset(const reg_t &qubits,
                                               RngEngine &rng) {
  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Apply update to reset state
  measure_reset_update(qubits, 0, meas.first, meas.second);
}

template <class statevec_t>
std::pair<uint_t, double>
ParallelExecutor<statevec_t>::sample_measure_with_prob(const reg_t &qubits,
                                                       RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);

  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::measure_reset_update(
    const std::vector<uint_t> &qubits, const uint_t final_state,
    const uint_t meas_state, const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t ic = BaseExecutor::top_state_of_group_[ig];
             ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
          BaseExecutor::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    } else {
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t ic = BaseExecutor::top_state_of_group_[ig];
             ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
          BaseExecutor::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      BaseExecutor::apply_chunk_x(qubits[0]);
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
        for (int_t ic = BaseExecutor::top_state_of_group_[ig];
             ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
          BaseExecutor::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    } else {
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t ic = BaseExecutor::top_state_of_group_[ig];
             ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
          BaseExecutor::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    }

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      reg_t qubits_in_chunk;
      reg_t qubits_out_chunk;

      Chunk::get_qubits_inout(BaseExecutor::chunk_bits_, qubits,
                              qubits_in_chunk, qubits_out_chunk);

      if (qubits_in_chunk.size() == qubits.size()) { // all bits are inside
                                                     // chunk
        // build vectorized permutation matrix
        cvector_t perm(dim * dim, 0.);
        perm[final_state * dim + meas_state] = 1.;
        perm[meas_state * dim + final_state] = 1.;
        for (size_t j = 0; j < dim; j++) {
          if (j != final_state && j != meas_state)
            perm[j * dim + j] = 1.;
        }
        // apply permutation to swap state
        if (BaseExecutor::chunk_omp_parallel_ &&
            BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
          for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
            for (int_t ic = BaseExecutor::top_state_of_group_[ig];
                 ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
              BaseExecutor::states_[ic].qreg().apply_matrix(qubits, perm);
          }
        } else {
          for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
            for (int_t ic = BaseExecutor::top_state_of_group_[ig];
                 ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
              BaseExecutor::states_[ic].qreg().apply_matrix(qubits, perm);
          }
        }
      } else {
        for (int_t i = 0; i < qubits.size(); i++) {
          if (((final_state >> i) & 1) != ((meas_state >> i) & 1)) {
            BaseExecutor::apply_chunk_x(qubits[i]);
          }
        }
      }
    }
  }
}

template <class statevec_t>
std::vector<reg_t>
ParallelExecutor<statevec_t>::sample_measure(const reg_t &qubits, uint_t shots,
                                             RngEngine &rng) const {
  int_t i, j;
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  reg_t allbit_samples(shots, 0);

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  std::vector<double> chunkSum(BaseExecutor::states_.size() + 1, 0);
  double sum, localSum;

  // calculate per chunk sum
  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (int_t ic = BaseExecutor::top_state_of_group_[ig];
           ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++) {
        bool batched = BaseExecutor::states_[ic].qreg().enable_batch(
            true); // return sum of all chunks in group
        chunkSum[ic] = BaseExecutor::states_[ic].qreg().norm();
        BaseExecutor::states_[ic].qreg().enable_batch(batched);
      }
    }
  } else {
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (int_t ic = BaseExecutor::top_state_of_group_[ig];
           ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++) {
        bool batched = BaseExecutor::states_[ic].qreg().enable_batch(
            true); // return sum of all chunks in group
        chunkSum[ic] = BaseExecutor::states_[ic].qreg().norm();
        BaseExecutor::states_[ic].qreg().enable_batch(batched);
      }
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

      for (j = 0; j < chunkSamples.size(); j++) {
        local_samples[vIdx[j]] = ((BaseExecutor::global_state_index_ + i)
                                  << BaseExecutor::chunk_bits_) +
                                 chunkSamples[j];
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

template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_initialize(const reg_t &qubits,
                                                    const cvector_t &params,
                                                    RngEngine &rng) {
  auto sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());
  if (qubits.size() == BaseExecutor::num_qubits_) {
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      initialize_from_vector(params);
      return;
    }
  }
  // Apply reset to qubits
  apply_reset(qubits, rng);

  // Apply initialize_component
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;
  Chunk::get_qubits_inout(BaseExecutor::chunk_bits_, qubits, qubits_in_chunk,
                          qubits_out_chunk);

  if (qubits_out_chunk.size() == 0) { // no qubits outside of chunk
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
          BaseExecutor::states_[i].qreg().initialize_component(qubits, params);
      }
    } else {
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        BaseExecutor::states_[i].qreg().initialize_component(qubits, params);
    }
  } else {
    // scatter base states
    if (qubits_in_chunk.size() > 0) {
      // scatter inside chunks
      const size_t dim = 1ULL << qubits_in_chunk.size();
      cvector_t perm(dim * dim, 0.);
      for (int_t i = 0; i < dim; i++) {
        perm[i] = 1.0;
      }

      if (BaseExecutor::chunk_omp_parallel_) {
#pragma omp parallel for
        for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
          BaseExecutor::states_[i].qreg().apply_matrix(qubits_in_chunk, perm);
      } else {
        for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
          BaseExecutor::states_[i].qreg().apply_matrix(qubits_in_chunk, perm);
      }
    }
    if (qubits_out_chunk.size() > 0) {
      // then scatter outside chunk
      auto sorted_qubits_out = qubits_out_chunk;
      std::sort(sorted_qubits_out.begin(), sorted_qubits_out.end());

      for (int_t i = 0;
           i < (1ull << (BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_ -
                         qubits_out_chunk.size()));
           i++) {
        uint_t baseChunk = 0;
        uint_t j, ii, t;
        ii = i;
        for (j = 0; j < qubits_out_chunk.size(); j++) {
          t = ii & ((1ull << qubits_out_chunk[j]) - 1);
          baseChunk += t;
          ii = (ii - t) << 1;
        }
        baseChunk += ii;
        baseChunk >>= BaseExecutor::chunk_bits_;

        for (j = 1; j < (1ull << qubits_out_chunk.size()); j++) {
          int_t ic = baseChunk;
          for (t = 0; t < qubits_out_chunk.size(); t++) {
            if ((j >> t) & 1)
              ic += (1ull << (qubits_out_chunk[t] - BaseExecutor::chunk_bits_));
          }

          if (ic >= BaseExecutor::state_index_begin_
                        [BaseExecutor::distributed_rank_] &&
              ic < BaseExecutor::state_index_end_
                       [BaseExecutor::distributed_rank_]) { // on this process
            if (baseChunk >= BaseExecutor::state_index_begin_
                                 [BaseExecutor::distributed_rank_] &&
                baseChunk <
                    BaseExecutor::state_index_end_
                        [BaseExecutor::distributed_rank_]) { // base chunk is on
                                                             // this process
              BaseExecutor::states_[ic].qreg().initialize_from_data(
                  BaseExecutor::states_[baseChunk].qreg().data(),
                  1ull << BaseExecutor::chunk_bits_);
            } else {
              BaseExecutor::recv_chunk(ic, baseChunk);
              // using swap chunk function to release send/recv buffers for
              // Thrust
              reg_t swap(2);
              swap[0] = BaseExecutor::chunk_bits_;
              swap[1] = BaseExecutor::chunk_bits_;
              BaseExecutor::states_[ic].qreg().apply_chunk_swap(swap,
                                                                baseChunk);
            }
          } else if (baseChunk >= BaseExecutor::state_index_begin_
                                      [BaseExecutor::distributed_rank_] &&
                     baseChunk <
                         BaseExecutor::state_index_end_
                             [BaseExecutor::distributed_rank_]) { // base chunk
                                                                  // is on this
                                                                  // process
            BaseExecutor::send_chunk(
                baseChunk - BaseExecutor::global_state_index_, ic);
          }
        }
      }
    }

    // initialize by params
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
          BaseExecutor::states_[i].qreg().apply_diagonal_matrix(qubits, params);
      }
    } else {
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        BaseExecutor::states_[i].qreg().apply_diagonal_matrix(qubits, params);
    }
  }
}

template <class statevec_t>
void ParallelExecutor<statevec_t>::initialize_from_vector(
    const cvector_t &params) {
  uint_t local_offset = BaseExecutor::global_state_index_
                        << BaseExecutor::chunk_bits_;

#pragma omp parallel for if (BaseExecutor::chunk_omp_parallel_)
  for (int_t i = 0; i < BaseExecutor::states_.size(); i++) {
    // copy part of state for this chunk
    cvector_t tmp(1ull << BaseExecutor::chunk_bits_);
    std::copy(params.begin() + local_offset + (i << BaseExecutor::chunk_bits_),
              params.begin() + local_offset +
                  ((i + 1) << BaseExecutor::chunk_bits_),
              tmp.begin());
    BaseExecutor::states_[i].qreg().initialize_from_vector(tmp);
  }
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class statevec_t>
void ParallelExecutor<statevec_t>::apply_kraus(
    const reg_t &qubits, const std::vector<cmatrix_t> &kmats, RngEngine &rng) {
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early

  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double r = rng.rand(0., 1.);
  double accum = 0.;
  double p;
  bool complete = false;

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {

    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);

    p = 0.0;
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for reduction(+ : p)
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t i = BaseExecutor::top_state_of_group_[ig];
             i < BaseExecutor::top_state_of_group_[ig + 1]; i++)
          p += BaseExecutor::states_[i].qreg().norm(qubits, vmat);
      }
    } else {
      for (int_t i = 0; i < BaseExecutor::states_.size(); i++)
        p += BaseExecutor::states_[i].qreg().norm(qubits, vmat);
    }

#ifdef AER_MPI
    BaseExecutor::reduce_sum(p);
#endif
    accum += p;

    // check if we need to apply this operator
    if (accum > r) {
      // rescale vmat so projection is normalized
      Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(p));
      // apply Kraus projection operator
      if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
        for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
          for (int_t ic = BaseExecutor::top_state_of_group_[ig];
               ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
            BaseExecutor::states_[ic].qreg().apply_matrix(qubits, vmat);
        }
      } else {
        for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
          for (int_t ic = BaseExecutor::top_state_of_group_[ig];
               ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
            BaseExecutor::states_[ic].qreg().apply_matrix(qubits, vmat);
        }
      }
      complete = true;
      break;
    }
  }

  // check if we haven't applied a kraus operator yet
  if (complete == false) {
    // Compute probability from accumulated
    complex_t renorm = 1 / std::sqrt(1. - accum);
    auto vmat = Utils::vectorize_matrix(renorm * kmats.back());
    if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t ic = BaseExecutor::top_state_of_group_[ig];
             ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
          BaseExecutor::states_[ic].qreg().apply_matrix(qubits, vmat);
      }
    } else {
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t ic = BaseExecutor::top_state_of_group_[ig];
             ic < BaseExecutor::top_state_of_group_[ig + 1]; ic++)
          BaseExecutor::states_[ic].qreg().apply_matrix(qubits, vmat);
      }
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace Statevector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
