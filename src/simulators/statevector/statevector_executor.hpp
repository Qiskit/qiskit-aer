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

#ifndef _statevector_executor_hpp_
#define _statevector_executor_hpp_

#include "simulators/batch_shots_executor.hpp"
#include "simulators/parallel_state_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace Statevector {

using ResultItr = std::vector<ExperimentResult>::iterator;

//-------------------------------------------------------------------------
// Executor for statevector
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

  auto move_to_vector(void);
  auto copy_to_vector(void);

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

  void apply_reset(CircuitExecutor::Branch &root, const reg_t &qubits);
  void apply_initialize(CircuitExecutor::Branch &root, const reg_t &qubits,
                        const cvector_t &params);
  void apply_kraus(CircuitExecutor::Branch &root, const reg_t &qubits,
                   const std::vector<cmatrix_t> &kmats);

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

  // Helper functions for shot-branching
  void apply_save_density_matrix(CircuitExecutor::Branch &root,
                                 const Operations::Op &op, ResultItr result);
  void apply_save_probs(CircuitExecutor::Branch &root, const Operations::Op &op,
                        ResultItr result);
  void apply_save_statevector(CircuitExecutor::Branch &root,
                              const Operations::Op &op, ResultItr result,
                              bool last_op);
  void apply_save_statevector_dict(CircuitExecutor::Branch &root,
                                   const Operations::Op &op, ResultItr result);
  void apply_save_amplitudes(CircuitExecutor::Branch &root,
                             const Operations::Op &op, ResultItr result);

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

  // Return the reduced density matrix for the simulator
  cmatrix_t density_matrix(const reg_t &qubits);

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                    RngEngine &rng) const override;
};

template <class state_t>
void Executor<state_t>::set_config(const Config &config) {
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
  if (BasePar::multiple_chunk_required(config, circ, noise)) {
    return BasePar::run_circuit_shots(circ, noise, config, init_rng, result_it,
                                      sample_noise);
  } else {
    return BaseBatch::run_circuit_shots(circ, noise, config, init_rng,
                                        result_it, sample_noise);
  }
}

template <class state_t>
bool Executor<state_t>::apply_parallel_op(const Operations::Op &op,
                                          ExperimentResult &result,
                                          RngEngine &rng, bool final_op) {
  // temporary : this is for statevector
  if (Base::states_[0].creg().check_conditional(op)) {
    switch (op.type) {
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
      BasePar::apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BasePar::apply_roerror(op, rng);
      break;
    case Operations::OpType::kraus:
      apply_kraus(op.qubits, op.mats, rng);
      break;
    case Operations::OpType::set_statevec:
      initialize_from_vector(op.params);
      break;
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      BasePar::apply_save_expval(op, result);
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

  // parameterization
  if (op.has_bind_params) {
    if (op.type == Operations::OpType::diagonal_matrix)
      Base::states_[istate].qreg().apply_batched_diagonal_matrix(
          op.qubits, op.params, Base::num_bind_params_,
          Base::num_shots_per_bind_param_);
    else
      Base::states_[istate].qreg().apply_batched_matrix(
          op.qubits, op.params, Base::num_bind_params_,
          Base::num_shots_per_bind_param_);
    return true;
  }

  switch (op.type) {
  case Operations::OpType::barrier:
  case Operations::OpType::nop:
  case Operations::OpType::qerror_loc:
    break;
  case Operations::OpType::reset:
    Base::states_[istate].qreg().apply_batched_reset(op.qubits, rng);
    break;
  case Operations::OpType::initialize:
    Base::states_[istate].qreg().apply_batched_reset(op.qubits, rng);
    Base::states_[istate].qreg().initialize_component(op.qubits, op.params);
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
    Base::states_[istate].apply_matrix(op);
    break;
  case Operations::OpType::diagonal_matrix:
    Base::states_[istate].qreg().apply_diagonal_matrix(op.qubits, op.params);
    break;
  case Operations::OpType::multiplexer:
    Base::states_[istate].apply_multiplexer(
        op.regs[0], op.regs[1],
        op.mats); // control qubits ([0]) & target qubits([1])
    break;
  case Operations::OpType::kraus:
    Base::states_[istate].qreg().apply_batched_kraus(op.qubits, op.mats, rng);
    break;
  case Operations::OpType::save_expval:
  case Operations::OpType::save_expval_var:
    BaseBatch::apply_batched_expval(istate, op, result);
    break;
  case Operations::OpType::sim_op:
    if (op.name == "begin_register_blocking") {
      Base::states_[istate].qreg().enter_register_blocking(op.qubits);
    } else if (op.name == "end_register_blocking") {
      Base::states_[istate].qreg().leave_register_blocking();
    } else {
      return false;
    }
    break;
  case Operations::OpType::set_statevec:
    Base::states_[istate].qreg().initialize_from_vector(op.params);
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
    case Operations::OpType::reset:
      apply_reset(root, op.qubits);
      break;
    case Operations::OpType::initialize:
      apply_initialize(root, op.qubits, op.params);
      break;
    case Operations::OpType::measure:
      apply_measure(root, op.qubits, op.memory, op.registers);
      break;
    case Operations::OpType::kraus:
      apply_kraus(root, op.qubits, op.mats);
      break;
    // save ops
    case Operations::OpType::save_expval:
    case Operations::OpType::save_expval_var:
      Base::apply_save_expval(root, op, result);
      break;
    case Operations::OpType::save_densmat:
      apply_save_density_matrix(root, op, result);
      break;
    case Operations::OpType::save_probs:
    case Operations::OpType::save_probs_ket:
      apply_save_probs(root, op, result);
      break;
    case Operations::OpType::save_state:
    case Operations::OpType::save_statevec:
      apply_save_statevector(root, op, result, final_op);
      break;
    case Operations::OpType::save_statevec_dict:
      apply_save_statevector_dict(root, op, result);
      break;
    case Operations::OpType::save_amps:
    case Operations::OpType::save_amps_sq:
      apply_save_amplitudes(root, op, result);
      break;
    default:
      return false;
    }
  }
  return true;
}

template <class state_t>
void Executor<state_t>::initialize_qreg(uint_t num_qubits) {
  uint_t i;

  for (i = 0; i < Base::states_.size(); i++) {
    Base::states_[i].qreg().set_num_qubits(BasePar::chunk_bits_);
  }

  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        if (Base::global_state_index_ + iChunk == 0 ||
            this->num_qubits_ == this->chunk_bits_) {
          Base::states_[iChunk].qreg().initialize();
          Base::states_[iChunk].apply_global_phase();
        } else {
          Base::states_[iChunk].qreg().zero();
        }
      }
    }
  } else {
    for (i = 0; i < Base::states_.size(); i++) {
      if (Base::global_state_index_ + i == 0 ||
          this->num_qubits_ == this->chunk_bits_) {
        Base::states_[i].qreg().initialize();
        Base::states_[i].apply_global_phase();
      } else {
        Base::states_[i].qreg().zero();
      }
    }
  }
}

template <class state_t>
auto Executor<state_t>::move_to_vector(void) {
  size_t size_required =
      2 * (sizeof(std::complex<double>) << Base::num_qubits_) +
      (sizeof(std::complex<double>) << BasePar::chunk_bits_) *
          Base::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store states"));
  }
  int_t iChunk;
  auto state = Base::states_[0].qreg().move_to_vector();
  state.resize(Base::num_local_states_ << BasePar::chunk_bits_);

#pragma omp parallel for if (BasePar::chunk_omp_parallel_) private(iChunk)
  for (iChunk = 1; iChunk < (int_t)Base::states_.size(); iChunk++) {
    auto tmp = Base::states_[iChunk].qreg().move_to_vector();
    uint_t j, offset = iChunk << BasePar::chunk_bits_;
    for (j = 0; j < tmp.size(); j++) {
      state[offset + j] = tmp[j];
    }
  }

#ifdef AER_MPI
  BasePar::gather_state(state);
#endif
  return state;
}

template <class state_t>
auto Executor<state_t>::copy_to_vector(void) {
  size_t size_required =
      2 * (sizeof(std::complex<double>) << Base::num_qubits_) +
      (sizeof(std::complex<double>) << BasePar::chunk_bits_) *
          Base::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store states"));
  }
  int_t iChunk;
  auto state = Base::states_[0].qreg().copy_to_vector();
  state.resize(Base::num_local_states_ << BasePar::chunk_bits_);

#pragma omp parallel for if (BasePar::chunk_omp_parallel_) private(iChunk)
  for (iChunk = 1; iChunk < (int_t)Base::states_.size(); iChunk++) {
    auto tmp = Base::states_[iChunk].qreg().copy_to_vector();
    uint_t j, offset = iChunk << BasePar::chunk_bits_;
    for (j = 0; j < tmp.size(); j++) {
      state[offset + j] = tmp[j];
    }
  }

#ifdef AER_MPI
  BasePar::gather_state(state);
#endif
  return state;
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class state_t>
void Executor<state_t>::apply_save_probs(const Operations::Op &op,
                                         ExperimentResult &result) {
  // get probs as hexadecimal
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    result.save_data_average(
        Base::states_[0].creg(), op.string_params[0],
        Utils::vec2ket(probs, Base::json_chop_threshold_, 16), op.type,
        op.save_type);
  } else {
    result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}

template <class state_t>
double Executor<state_t>::expval_pauli(const reg_t &qubits,
                                       const std::string &pauli) {
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;
  std::string pauli_in_chunk;
  std::string pauli_out_chunk;
  uint_t n;
  double expval(0.);

  // get inner/outer chunk pauli string
  n = pauli.size();
  for (uint_t i = 0; i < n; i++) {
    if (qubits[i] < BasePar::chunk_bits_) {
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
      uint_t proc_bits = 0;
      uint_t procs = Base::distributed_procs_;
      while (procs > 1) {
        if ((procs & 1) != 0) {
          proc_bits = 0;
          break;
        }
        proc_bits++;
        procs >>= 1;
      }
      if ((x_mask & (~((1ull << (Base::num_qubits_ - proc_bits)) - 1))) !=
          0) { // data exchange between processes is required
        on_same_process = false;
      }
#endif

      x_mask >>= BasePar::chunk_bits_;
      z_mask >>= BasePar::chunk_bits_;
      x_max -= BasePar::chunk_bits_;

      const uint_t mask_u = ~((1ull << (x_max + 1)) - 1);
      const uint_t mask_l = (1ull << x_max) - 1;
      if (on_same_process) {
        auto apply_expval_pauli_chunk = [this, x_mask, z_mask, x_max, mask_u,
                                         mask_l, qubits_in_chunk,
                                         pauli_in_chunk, phase](int_t iGroup) {
          double expval_t = 0.0;
          for (uint_t iChunk = Base::top_state_of_group_[iGroup];
               iChunk < Base::top_state_of_group_[iGroup + 1]; iChunk++) {
            uint_t pair_chunk = iChunk ^ x_mask;
            if (iChunk < pair_chunk) {
              uint_t z_count, z_count_pair;
              z_count = AER::Utils::popcount(iChunk & z_mask);
              z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

              expval_t += Base::states_[iChunk - Base::global_state_index_]
                              .qreg()
                              .expval_pauli(qubits_in_chunk, pauli_in_chunk,
                                            Base::states_[pair_chunk].qreg(),
                                            z_count, z_count_pair, phase);
            }
          }
          return expval_t;
        };
        expval += Utils::apply_omp_parallel_for_reduction(
            (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1), 0,
            Base::num_global_states_ / 2, apply_expval_pauli_chunk);
      } else {
        for (uint_t i = 0; i < Base::num_global_states_ / 2; i++) {
          uint_t iChunk = ((i << 1) & mask_u) | (i & mask_l);
          uint_t pair_chunk = iChunk ^ x_mask;
          uint_t iProc = BasePar::get_process_by_chunk(pair_chunk);
          if (Base::state_index_begin_[Base::distributed_rank_] <= iChunk &&
              Base::state_index_end_[Base::distributed_rank_] >
                  iChunk) { // on this process
            uint_t z_count, z_count_pair;
            z_count = AER::Utils::popcount(iChunk & z_mask);
            z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

            if (iProc == Base::distributed_rank_) { // pair is on the
                                                    // same process
              expval +=
                  Base::states_[iChunk - Base::global_state_index_]
                      .qreg()
                      .expval_pauli(
                          qubits_in_chunk, pauli_in_chunk,
                          Base::states_[pair_chunk - Base::global_state_index_]
                              .qreg(),
                          z_count, z_count_pair, phase);
            } else {
              BasePar::recv_chunk(iChunk - Base::global_state_index_,
                                  pair_chunk);
              // refer receive buffer to calculate expectation value
              expval +=
                  Base::states_[iChunk - Base::global_state_index_]
                      .qreg()
                      .expval_pauli(
                          qubits_in_chunk, pauli_in_chunk,
                          Base::states_[iChunk - Base::global_state_index_]
                              .qreg(),
                          z_count, z_count_pair, phase);
            }
          } else if (iProc == Base::distributed_rank_) { // pair is on
                                                         // this process
            BasePar::send_chunk(iChunk - Base::global_state_index_, pair_chunk);
          }
        }
      }
    } else { // no exchange between chunks
      z_mask >>= BasePar::chunk_bits_;
      if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for reduction(+ : expval)
        for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
          double e_tmp = 0.0;
          for (uint_t iChunk = Base::top_state_of_group_[ig];
               iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
            double sign = 1.0;
            if (z_mask && (AER::Utils::popcount(
                               (iChunk + Base::global_state_index_) & z_mask) &
                           1))
              sign = -1.0;
            e_tmp += sign * Base::states_[iChunk].qreg().expval_pauli(
                                qubits_in_chunk, pauli_in_chunk);
          }
          expval += e_tmp;
        }
      } else {
        for (uint_t i = 0; i < Base::states_.size(); i++) {
          double sign = 1.0;
          if (z_mask &&
              (AER::Utils::popcount((i + Base::global_state_index_) & z_mask) &
               1))
            sign = -1.0;
          expval += sign * Base::states_[i].qreg().expval_pauli(qubits_in_chunk,
                                                                pauli_in_chunk);
        }
      }
    }
  } else { // all bits are inside chunk
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for reduction(+ : expval)
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        double e_tmp = 0.0;
        for (uint_t iChunk = Base::top_state_of_group_[ig];
             iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
          e_tmp += Base::states_[iChunk].qreg().expval_pauli(qubits, pauli);
        expval += e_tmp;
      }
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        expval += Base::states_[i].qreg().expval_pauli(qubits, pauli);
    }
  }

#ifdef AER_MPI
  BasePar::reduce_sum(expval);
#endif
  return expval;
}

template <class state_t>
void Executor<state_t>::apply_save_statevector(const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "statevector" : op.string_params[0];

  if (last_op) {
    auto v = move_to_vector();
    result.save_data_pershot(Base::states_[0].creg(), key, std::move(v),
                             Operations::OpType::save_statevec, op.save_type);
  } else {
    result.save_data_pershot(Base::states_[0].creg(), key, copy_to_vector(),
                             Operations::OpType::save_statevec, op.save_type);
  }
}

template <class state_t>
void Executor<state_t>::apply_save_statevector_dict(const Operations::Op &op,
                                                    ExperimentResult &result) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  auto vec = copy_to_vector();
  std::map<std::string, complex_t> result_state_ket;
  for (size_t k = 0; k < vec.size(); ++k) {
    if (std::abs(vec[k]) >= Base::json_chop_threshold_) {
      std::string key = Utils::int2hex(k);
      result_state_ket.insert({key, vec[k]});
    }
  }
  result.save_data_pershot(Base::states_[0].creg(), op.string_params[0],
                           std::move(result_state_ket), op.type, op.save_type);
}

template <class state_t>
void Executor<state_t>::apply_save_density_matrix(const Operations::Op &op,
                                                  ExperimentResult &result) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    double sum = 0.0;
    if (BasePar::chunk_omp_parallel_) {
#pragma omp parallel for reduction(+ : sum)
      for (int_t i = 0; i < (int_t)Base::states_.size(); i++)
        sum += Base::states_[i].qreg().norm();
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        sum += Base::states_[i].qreg().norm();
    }
#ifdef AER_MPI
    BasePar::reduce_sum(sum);
#endif
    reduced_state[0] = sum;
  } else {
    reduced_state = density_matrix(op.qubits);
  }

  result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

template <class state_t>
void Executor<state_t>::apply_save_amplitudes(const Operations::Op &op,
                                              ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (int_t i = 0; i < size; ++i) {
      uint_t idx = BasePar::mapped_index(op.int_params[i]);
      uint_t iChunk = idx >> BasePar::chunk_bits_;
      amps[i] = 0.0;
      if (iChunk >= Base::global_state_index_ &&
          iChunk < Base::global_state_index_ + Base::states_.size()) {
        amps[i] =
            Base::states_[iChunk - Base::global_state_index_].qreg().get_state(
                idx - (iChunk << BasePar::chunk_bits_));
      }
#ifdef AER_MPI
      complex_t amp = amps[i];
      BasePar::reduce_sum(amp);
      amps[i] = amp;
#endif
    }
    result.save_data_pershot(Base::states_[0].creg(), op.string_params[0],
                             std::move(amps), op.type, op.save_type);
  } else {
    rvector_t amps_sq(size, 0);
    for (int_t i = 0; i < size; ++i) {
      uint_t idx = BasePar::mapped_index(op.int_params[i]);
      uint_t iChunk = idx >> BasePar::chunk_bits_;
      if (iChunk >= Base::global_state_index_ &&
          iChunk < Base::global_state_index_ + Base::states_.size()) {
        amps_sq[i] = Base::states_[iChunk - Base::global_state_index_]
                         .qreg()
                         .probability(idx - (iChunk << BasePar::chunk_bits_));
      }
    }
#ifdef AER_MPI
    BasePar::reduce_sum(amps_sq);
#endif
    result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                             std::move(amps_sq), op.type, op.save_type);
  }
}

template <class state_t>
cmatrix_t Executor<state_t>::density_matrix(const reg_t &qubits) {
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  auto vec = copy_to_vector();

  // Return full density matrix
  cmatrix_t densmat(DIM, DIM);
  if ((N == Base::num_qubits_) && (qubits == qubits_sorted)) {
    const int_t mask = QV::MASKS[N];
#pragma omp parallel for
    for (int_t rowcol = 0; rowcol < int_t(DIM * DIM); ++rowcol) {
      const int_t row = rowcol >> N;
      const int_t col = rowcol & mask;
      densmat(row, col) = complex_t(vec[row]) * complex_t(std::conj(vec[col]));
    }
  } else {
    const size_t END = 1ULL << (Base::num_qubits_ - N);
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

template <class state_t>
void Executor<state_t>::apply_measure(const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister, RngEngine &rng) {
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BasePar::store_measure(outcome, cmemory, cregister);
}

template <class state_t>
rvector_t Executor<state_t>::measure_probs(const reg_t &qubits) const {
  uint_t dim = 1ull << qubits.size();
  rvector_t sum(dim, 0.0);
  uint_t i, j, k;
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;

  Chunk::get_qubits_inout(BasePar::chunk_bits_, qubits, qubits_in_chunk,
                          qubits_out_chunk);

  if (qubits_in_chunk.size() > 0) {
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(i, j, k)
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++) {
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
                if (qubits[k] < BasePar::chunk_bits_) {
                  idx += (((j >> i_in) & 1) << k);
                  i_in++;
                } else {
                  if ((((i + Base::global_state_index_)
                        << BasePar::chunk_bits_) >>
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
      for (i = 0; i < Base::states_.size(); i++) {
        auto chunkSum = Base::states_[i].qreg().probabilities(qubits_in_chunk);

        if (qubits_in_chunk.size() == qubits.size()) {
          for (j = 0; j < dim; j++) {
            sum[j] += chunkSum[j];
          }
        } else {
          for (j = 0; j < chunkSum.size(); j++) {
            int idx = 0;
            int i_in = 0;
            for (k = 0; k < qubits.size(); k++) {
              if (qubits[k] < BasePar::chunk_bits_) {
                idx += (((j >> i_in) & 1) << k);
                i_in++;
              } else {
                if ((((i + Base::global_state_index_)
                      << BasePar::chunk_bits_) >>
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
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(i, j, k)
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++) {
          auto nr = std::real(Base::states_[i].qreg().norm());
          int idx = 0;
          for (k = 0; k < qubits_out_chunk.size(); k++) {
            if ((((i + Base::global_state_index_) << (BasePar::chunk_bits_)) >>
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
      for (i = 0; i < Base::states_.size(); i++) {
        auto nr = std::real(Base::states_[i].qreg().norm());
        uint_t idx = 0;
        for (k = 0; k < qubits_out_chunk.size(); k++) {
          if ((((i + Base::global_state_index_) << (BasePar::chunk_bits_)) >>
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
  BasePar::reduce_sum(sum);
#endif

  return sum;
}

template <class state_t>
void Executor<state_t>::apply_reset(const reg_t &qubits, RngEngine &rng) {
  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Apply update to reset state
  measure_reset_update(qubits, 0, meas.first, meas.second);
}

template <class state_t>
std::pair<uint_t, double>
Executor<state_t>::sample_measure_with_prob(const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);

  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class state_t>
void Executor<state_t>::measure_reset_update(const std::vector<uint_t> &qubits,
                                             const uint_t final_state,
                                             const uint_t meas_state,
                                             const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t ic = Base::top_state_of_group_[ig];
             ic < Base::top_state_of_group_[ig + 1]; ic++)
          Base::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    } else {
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t ic = Base::top_state_of_group_[ig];
             ic < Base::top_state_of_group_[ig + 1]; ic++)
          Base::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      BasePar::apply_chunk_x(qubits[0]);
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
        for (uint_t ic = Base::top_state_of_group_[ig];
             ic < Base::top_state_of_group_[ig + 1]; ic++)
          Base::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    } else {
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t ic = Base::top_state_of_group_[ig];
             ic < Base::top_state_of_group_[ig + 1]; ic++)
          Base::states_[ic].apply_diagonal_matrix(qubits, mdiag);
      }
    }

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      reg_t qubits_in_chunk;
      reg_t qubits_out_chunk;

      Chunk::get_qubits_inout(BasePar::chunk_bits_, qubits, qubits_in_chunk,
                              qubits_out_chunk);

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
        if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
          for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
            for (uint_t ic = Base::top_state_of_group_[ig];
                 ic < Base::top_state_of_group_[ig + 1]; ic++)
              Base::states_[ic].qreg().apply_matrix(qubits, perm);
          }
        } else {
          for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
            for (uint_t ic = Base::top_state_of_group_[ig];
                 ic < Base::top_state_of_group_[ig + 1]; ic++)
              Base::states_[ic].qreg().apply_matrix(qubits, perm);
          }
        }
      } else {
        for (int_t i = 0; i < (int_t)qubits.size(); i++) {
          if (((final_state >> i) & 1) != ((meas_state >> i) & 1)) {
            BasePar::apply_chunk_x(qubits[i]);
          }
        }
      }
    }
  }
}

template <class state_t>
std::vector<reg_t> Executor<state_t>::sample_measure(const reg_t &qubits,
                                                     uint_t shots,
                                                     RngEngine &rng) const {
  uint_t i, j;
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  reg_t allbit_samples(shots, 0);

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  std::vector<double> chunkSum(Base::states_.size() + 1, 0);
  double sum, localSum;

  // calculate per chunk sum
  if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t ic = Base::top_state_of_group_[ig];
           ic < Base::top_state_of_group_[ig + 1]; ic++) {
        bool batched = Base::states_[ic].qreg().enable_batch(
            true); // return sum of all chunks in group
        chunkSum[ic] = Base::states_[ic].qreg().norm();
        Base::states_[ic].qreg().enable_batch(batched);
      }
    }
  } else {
    for (uint_t ig = 0; ig < Base::num_groups_; ig++) {
      for (uint_t ic = Base::top_state_of_group_[ig];
           ic < Base::top_state_of_group_[ig + 1]; ic++) {
        bool batched = Base::states_[ic].qreg().enable_batch(
            true); // return sum of all chunks in group
        chunkSum[ic] = Base::states_[ic].qreg().norm();
        Base::states_[ic].qreg().enable_batch(batched);
      }
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

      for (j = 0; j < chunkSamples.size(); j++) {
        local_samples[vIdx[j]] =
            ((Base::global_state_index_ + i) << BasePar::chunk_bits_) +
            chunkSamples[j];
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
void Executor<state_t>::apply_initialize(const reg_t &qubits,
                                         const cvector_t &params_in,
                                         RngEngine &rng) {
  auto sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());
  // apply global phase here
  cvector_t tmp;
  if (Base::states_[0].has_global_phase()) {
    tmp.resize(params_in.size());
    std::complex<double> global_phase = Base::states_[0].global_phase();
    auto apply_global_phase = [&tmp, &params_in, global_phase](int_t i) {
      tmp[i] = params_in[i] * global_phase;
    };
    Utils::apply_omp_parallel_for(
        (qubits.size() > (uint_t)Base::omp_qubit_threshold_), 0,
        params_in.size(), apply_global_phase, Base::parallel_state_update_);
  }
  const cvector_t &params = tmp.empty() ? params_in : tmp;
  if (qubits.size() == Base::num_qubits_) {
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
  Chunk::get_qubits_inout(BasePar::chunk_bits_, qubits, qubits_in_chunk,
                          qubits_out_chunk);

  if (qubits_out_chunk.size() == 0) { // no qubits outside of chunk
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++)
          Base::states_[i].qreg().initialize_component(qubits, params);
      }
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        Base::states_[i].qreg().initialize_component(qubits, params);
    }
  } else {
    // scatter base states
    if (qubits_in_chunk.size() > 0) {
      // scatter inside chunks
      const size_t dim = 1ULL << qubits_in_chunk.size();
      cvector_t perm(dim * dim, 0.);
      for (uint_t i = 0; i < dim; i++) {
        perm[i] = 1.0;
      }

      if (BasePar::chunk_omp_parallel_) {
#pragma omp parallel for
        for (int_t i = 0; i < (int_t)Base::states_.size(); i++)
          Base::states_[i].qreg().apply_matrix(qubits_in_chunk, perm);
      } else {
        for (uint_t i = 0; i < Base::states_.size(); i++)
          Base::states_[i].qreg().apply_matrix(qubits_in_chunk, perm);
      }
    }
    if (qubits_out_chunk.size() > 0) {
      // then scatter outside chunk
      auto sorted_qubits_out = qubits_out_chunk;
      std::sort(sorted_qubits_out.begin(), sorted_qubits_out.end());

      for (uint_t i = 0;
           i < (1ull << (Base::num_qubits_ - BasePar::chunk_bits_ -
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
        baseChunk >>= BasePar::chunk_bits_;

        for (j = 1; j < (1ull << qubits_out_chunk.size()); j++) {
          uint_t ic = baseChunk;
          for (t = 0; t < qubits_out_chunk.size(); t++) {
            if ((j >> t) & 1)
              ic += (1ull << (qubits_out_chunk[t] - BasePar::chunk_bits_));
          }

          if (ic >= Base::state_index_begin_[Base::distributed_rank_] &&
              ic < Base::state_index_end_[Base::distributed_rank_]) { // on this
                                                                      // process
            if (baseChunk >=
                    Base::state_index_begin_[Base::distributed_rank_] &&
                baseChunk < Base::state_index_end_
                                [Base::distributed_rank_]) { // base chunk is on
                                                             // this process
              Base::states_[ic].qreg().initialize_from_data(
                  Base::states_[baseChunk].qreg().data(),
                  1ull << BasePar::chunk_bits_);
            } else {
              BasePar::recv_chunk(ic, baseChunk);
              // using swap chunk function to release send/recv buffers for
              // Thrust
              reg_t swap(2);
              swap[0] = BasePar::chunk_bits_;
              swap[1] = BasePar::chunk_bits_;
              Base::states_[ic].qreg().apply_chunk_swap(swap, baseChunk);
            }
          } else if (baseChunk >=
                         Base::state_index_begin_[Base::distributed_rank_] &&
                     baseChunk < Base::state_index_end_
                                     [Base::distributed_rank_]) { // base chunk
                                                                  // is on this
                                                                  // process
            BasePar::send_chunk(baseChunk - Base::global_state_index_, ic);
          }
        }
      }
    }

    // initialize by params
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++)
          Base::states_[i].qreg().apply_diagonal_matrix(qubits, params);
      }
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        Base::states_[i].qreg().apply_diagonal_matrix(qubits, params);
    }
  }
}

template <class state_t>
void Executor<state_t>::initialize_from_vector(const cvector_t &params) {
  uint_t local_offset = Base::global_state_index_ << BasePar::chunk_bits_;

#pragma omp parallel for if (BasePar::chunk_omp_parallel_)
  for (int_t i = 0; i < (int_t)Base::states_.size(); i++) {
    // copy part of state for this chunk
    cvector_t tmp(1ull << BasePar::chunk_bits_);
    std::copy(params.begin() + local_offset + (i << BasePar::chunk_bits_),
              params.begin() + local_offset + ((i + 1) << BasePar::chunk_bits_),
              tmp.begin());
    Base::states_[i].qreg().initialize_from_vector(tmp);
  }
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class state_t>
void Executor<state_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats,
                                    RngEngine &rng) {
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
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for reduction(+ : p)
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t i = Base::top_state_of_group_[ig];
             i < Base::top_state_of_group_[ig + 1]; i++)
          p += Base::states_[i].qreg().norm(qubits, vmat);
      }
    } else {
      for (uint_t i = 0; i < Base::states_.size(); i++)
        p += Base::states_[i].qreg().norm(qubits, vmat);
    }

#ifdef AER_MPI
    BasePar::reduce_sum(p);
#endif
    accum += p;

    // check if we need to apply this operator
    if (accum > r) {
      // rescale vmat so projection is normalized
      Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(p));
      // apply Kraus projection operator
      if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
        for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
          for (uint_t ic = Base::top_state_of_group_[ig];
               ic < Base::top_state_of_group_[ig + 1]; ic++)
            Base::states_[ic].qreg().apply_matrix(qubits, vmat);
        }
      } else {
        for (uint_t ig = 0; ig < Base::num_groups_; ig++) {
          for (uint_t ic = Base::top_state_of_group_[ig];
               ic < Base::top_state_of_group_[ig + 1]; ic++)
            Base::states_[ic].qreg().apply_matrix(qubits, vmat);
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
    if (BasePar::chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t ic = Base::top_state_of_group_[ig];
             ic < Base::top_state_of_group_[ig + 1]; ic++)
          Base::states_[ic].qreg().apply_matrix(qubits, vmat);
      }
    } else {
      for (uint_t ig = 0; ig < Base::num_groups_; ig++) {
        for (uint_t ic = Base::top_state_of_group_[ig];
             ic < Base::top_state_of_group_[ig + 1]; ic++)
          Base::states_[ic].qreg().apply_matrix(qubits, vmat);
      }
    }
  }
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
        op2.name = "mcx";
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
        for (uint_t j = 0; j < dim; j++) {
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
void Executor<state_t>::apply_reset(CircuitExecutor::Branch &root,
                                    const reg_t &qubits) {
  rvector_t probs = sample_measure_with_prob(root, qubits);

  measure_reset_update(root, qubits, 0, probs);
}

template <class state_t>
void Executor<state_t>::apply_initialize(CircuitExecutor::Branch &root,
                                         const reg_t &qubits,
                                         const cvector_t &params_in) {
  // apply global phase here
  cvector_t tmp;
  if (Base::states_[root.state_index()].has_global_phase()) {
    tmp.resize(params_in.size());
    std::complex<double> global_phase =
        Base::states_[root.state_index()].global_phase();
    auto apply_global_phase = [&tmp, params_in, global_phase](int_t i) {
      tmp[i] = params_in[i] * global_phase;
    };
    Utils::apply_omp_parallel_for(
        (qubits.size() > (uint_t)Base::omp_qubit_threshold_), 0,
        params_in.size(), apply_global_phase, Base::parallel_state_update_);
  }
  const cvector_t &params = tmp.empty() ? params_in : tmp;
  if (qubits.size() == Base::num_qubits_) {
    auto sorted_qubits = qubits;
    std::sort(sorted_qubits.begin(), sorted_qubits.end());
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      Base::states_[root.state_index()].initialize_from_vector(params);
      return;
    }
  }

  if (root.additional_ops().size() == 0) {
    apply_reset(root, qubits);

    Operations::Op op;
    op.type = OpType::initialize;
    op.name = "initialize";
    op.qubits = qubits;
    op.params = params;
    for (uint_t i = 0; i < root.num_branches(); i++) {
      root.branches()[i]->add_op_after_branch(op);
    }
    return; // initialization will be done in next call because of shot
            // branching in reset
  }

  Base::states_[root.state_index()].qreg().initialize_component(qubits, params);
}

template <class state_t>
void Executor<state_t>::apply_kraus(CircuitExecutor::Branch &root,
                                    const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats) {
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early

  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double accum = 0.;
  double p;

  reg_t shot_branch;
  uint_t nshots;
  rvector_t rshots, pmats;
  uint_t nshots_multiplied = 0;

  nshots = root.num_shots();
  shot_branch.resize(nshots);
  rshots.resize(nshots);
  for (uint_t i = 0; i < nshots; i++) {
    shot_branch[i] = kmats.size() - 1;
    rshots[i] = root.rng_shots()[i].rand(0., 1.);
  }
  pmats.resize(kmats.size());

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {
    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);

    p = Base::states_[root.state_index()].qreg().norm(qubits, vmat);
    accum += p;

    // check if we need to apply this operator
    pmats[j] = p;
    for (uint_t i = 0; i < nshots; i++) {
      if (shot_branch[i] >= kmats.size() - 1) {
        if (accum > rshots[i]) {
          shot_branch[i] = j;
          nshots_multiplied++;
        }
      }
    }
    if (nshots_multiplied >= nshots) {
      break;
    }
  }

  pmats[pmats.size() - 1] = 1. - accum;

  root.creg() = Base::states_[root.state_index()].creg();
  root.branch_shots(shot_branch, kmats.size());
  for (uint_t i = 0; i < kmats.size(); i++) {
    Operations::Op op;
    op.type = OpType::matrix;
    op.qubits = qubits;
    op.mats.push_back(kmats[i]);
    p = 1 / std::sqrt(pmats[i]);
    for (uint_t j = 0; j < op.mats[0].size(); j++)
      op.mats[0][j] *= p;
    root.branches()[i]->add_op_after_branch(op);
  }
}

template <class state_t>
void Executor<state_t>::apply_save_density_matrix(CircuitExecutor::Branch &root,
                                                  const Operations::Op &op,
                                                  ResultItr result) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    reduced_state[0] = Base::states_[root.state_index()].qreg().norm();
  } else {
    reduced_state = Base::states_[root.state_index()].density_matrix(op.qubits);
  }

  std::vector<bool> copied(Base::num_bind_params_, false);
  for (uint_t i = 0; i < root.num_shots(); i++) {
    uint_t ip = root.param_index(i);
    if (!copied[ip]) {
      (result + ip)
          ->save_data_average(Base::states_[root.state_index()].creg(),
                              op.string_params[0], reduced_state, op.type,
                              op.save_type);
      copied[ip] = true;
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_save_probs(CircuitExecutor::Branch &root,
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

template <class state_t>
void Executor<state_t>::apply_save_statevector(CircuitExecutor::Branch &root,
                                               const Operations::Op &op,
                                               ResultItr result, bool last_op) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "statevector" : op.string_params[0];

  if (last_op) {
    const auto v = Base::states_[root.state_index()].move_to_vector();
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      (result + ip)
          ->save_data_pershot(Base::states_[root.state_index()].creg(), key, v,
                              OpType::save_statevec, op.save_type);
    }
  } else {
    const auto v = Base::states_[root.state_index()].copy_to_vector();
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      (result + ip)
          ->save_data_pershot(Base::states_[root.state_index()].creg(), key, v,
                              OpType::save_statevec, op.save_type);
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_save_statevector_dict(
    CircuitExecutor::Branch &root, const Operations::Op &op, ResultItr result) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  auto state_ket = Base::states_[root.state_index()].qreg().vector_ket(
      Base::json_chop_threshold_);
  std::map<std::string, complex_t> result_state_ket;
  for (auto const &it : state_ket) {
    result_state_ket[it.first] = it.second;
  }
  for (uint_t i = 0; i < root.num_shots(); i++) {
    uint_t ip = root.param_index(i);
    (result + ip)
        ->save_data_pershot(
            Base::states_[root.state_index()].creg(), op.string_params[0],
            (const std::map<std::string, complex_t> &)result_state_ket, op.type,
            op.save_type);
  }
}

template <class state_t>
void Executor<state_t>::apply_save_amplitudes(CircuitExecutor::Branch &root,
                                              const Operations::Op &op,
                                              ResultItr result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (int_t i = 0; i < size; ++i) {
      amps[i] =
          Base::states_[root.state_index()].qreg().get_state(op.int_params[i]);
    }
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      (result + ip)
          ->save_data_pershot(
              Base::states_[root.state_index()].creg(), op.string_params[0],
              (const Vector<complex_t> &)amps, op.type, op.save_type);
    }
  } else {
    rvector_t amps_sq(size, 0);
    for (int_t i = 0; i < size; ++i) {
      amps_sq[i] = Base::states_[root.state_index()].qreg().probability(
          op.int_params[i]);
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

//-------------------------------------------------------------------------
} // end namespace Statevector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
