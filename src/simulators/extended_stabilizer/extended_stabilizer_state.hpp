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

#ifndef _aer_chsimulator_state_hpp
#define _aer_chsimulator_state_hpp

#include <complex>
#include <vector>

#include "simulators/state.hpp"
#include "framework/json.hpp"
#include "framework/types.hpp"

#include "chlib/core.hpp"
#include "chlib/chstabilizer.hpp"
#include "ch_runner.hpp"
#include "gates.hpp"

namespace AER{
namespace ExtendedStabilizer {

using chpauli_t = CHSimulator::pauli_t;
using chstate_t = CHSimulator::Runner;
using Gates = CHSimulator::Gates;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::measure,
    Operations::OpType::reset, Operations::OpType::barrier,
    Operations::OpType::roerror, Operations::OpType::bfunc,
    Operations::OpType::qerror_loc,
    Operations::OpType::save_statevec,
    }, //Operations::OpType::save_expval, Operations::OpType::save_expval_var},
  // Gates
  {"CX", "u0", "u1", "p", "cx", "cz", "swap", "id", "x", "y", "z", "h",
    "s", "sdg", "sx", "sxdg", "t", "tdg", "ccx", "ccz", "delay", "pauli"});

uint_t zero = 0ULL;
uint_t toff_branch_max = 7ULL;

enum class SamplingMethod {
  metropolis,
  resampled_metropolis,
  norm_estimation
};

class State: public QuantumState::State<chstate_t>
{
public:
  using BaseState = QuantumState::State<chstate_t>;
  
  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  std::string name() const override {return "extended_stabilizer";}

  //Apply a sequence of operations to the cicuit. For each operation,
  //we loop over the terms in the decomposition in parallel
  template <typename InputIterator>
  void apply_ops(InputIterator first, InputIterator last,
                  ExperimentResult &result,
                  RngEngine &rng,
                  bool final_ops = false);

  // Apply a single operation
  // If the op is not in allowed_ops an exeption will be raised.
  void apply_op(QuantumState::RegistersBase& state,
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine &rng,
                        bool final_op = false) override;

  size_t required_memory_mb(uint_t num_qubits,
                                    QuantumState::OpItr first, QuantumState::OpItr last)
                                    const override;

  std::vector<reg_t> sample_measure(QuantumState::RegistersBase& state_in, const reg_t& qubits,
                                    uint_t shots,
                                    RngEngine &rng) override;

protected:
  void initialize_qreg_state(QuantumState::RegistersBase& state_in, const uint_t num_qubits) override;

  void initialize_qreg_state(QuantumState::RegistersBase& state_in, const chstate_t &state) override;

  void set_state_config(QuantumState::RegistersBase& state_in, const json_t &config) override;

  template <typename InputIterator>
  void apply_ops_state(QuantumState::Registers<chstate_t>& state, InputIterator first, InputIterator last,
                  ExperimentResult &result,
                  RngEngine &rng,
                  bool final_ops = false);

  //Alongside the sample measure optimisaiton, we can parallelise
  //circuit applicaiton over the states. This reduces the threading overhead
  //as we only have to fork once per circuit.
  template <typename InputIterator>
  void apply_ops_parallel(QuantumState::Registers<chstate_t>& state, InputIterator first, InputIterator last,
                          ExperimentResult &result,
                          RngEngine &rng);

  //Small routine that eschews any parallelisation/decomposition and applies a stabilizer
  //circuit to a single state. This is used to optimize a circuit with a large
  //initial clifford fraction, or for running stabilizer circuits.
  template <typename InputIterator>
  void apply_stabilizer_circuit(QuantumState::Registers<chstate_t>& state, InputIterator first, InputIterator last,
                                ExperimentResult &result,
                                RngEngine &rng);

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  // TODO: Investigate OMP synchronisation over stattes to remove these different versions
  // One option would be tasks, but the memory overhead isn't clear
  void apply_gate(QuantumState::Registers<chstate_t>& state,const Operations::Op &op, RngEngine &rng);
  void apply_gate(QuantumState::Registers<chstate_t>& state,const Operations::Op &op, RngEngine &rng, uint_t rank);

  // Apply a multi-qubit Pauli gate
  void apply_pauli(QuantumState::Registers<chstate_t>& state,const reg_t &qubits, const std::string& pauli, uint_t rank);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function then "measure" 
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  void apply_measure(QuantumState::Registers<chstate_t>& state,const reg_t &qubits,
                     const reg_t &cmemory,
                     const reg_t &cregister,
                     RngEngine &rng);

  // Reset the specified qubits to the |0> state by measuring the
  // projectors Id+Z_{i} for each qubit i
  void apply_reset(QuantumState::Registers<chstate_t>& state, const reg_t &qubits, AER::RngEngine &rng);

  const static stringmap_t<Gates> gateset_;

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Compute and save the statevector for the current simulator state
  void apply_save_statevector(QuantumState::Registers<chstate_t>& state,const Operations::Op &op,
                              ExperimentResult &result);

  // Compute and save the expval for the current simulator state
  void apply_save_expval(QuantumState::Registers<chstate_t>& state,const Operations::Op &op,
                              ExperimentResult &result,
                              RngEngine &rng);

   // Helper function for computing expectation value
   double expval_pauli(QuantumState::Registers<chstate_t>& state,const reg_t &qubits,
                       const std::string& pauli,
                       RngEngine &rng);

  // Helper function for computing expectation value
  virtual double expval_pauli(QuantumState::RegistersBase& state,const reg_t &qubits,
                              const std::string& pauli) override;

  //-----------------------------------------------------------------------
  //Parameters and methods specific to the Stabilizer Rank Decomposition
  //-----------------------------------------------------------------------

  //Allowed error in the stabilizer rank decomposition.
  //The required number of states scales as \delta^{-2}
  //for allowed error \delta
  double approximation_error_ = 0.05;

  uint_t norm_estimation_samples_ = 100;

  uint_t norm_estimation_repetitions_ = 3;

  // How long the metropolis algorithm runs before
  // we consider it to be well mixed and sample form the
  // output distribution
  uint_t metropolis_mixing_steps_ = 5000;

  //Minimum number of states before we try to parallelise
  uint_t omp_threshold_rank_ = 100;

  double snapshot_chop_threshold_ = 1e-10;

  uint_t probabilities_snapshot_samples_ = 3000;

  SamplingMethod sampling_method_ = SamplingMethod::resampled_metropolis;

  template <typename InputIterator>
  uint_t compute_chi(InputIterator first, InputIterator last) const;

  // Add the given operation to the extent
  void compute_extent(const Operations::Op &op, double &xi) const;

  template <typename InputIterator>
  std::pair<uint_t, uint_t> decomposition_parameters(InputIterator first, InputIterator last);
  
  template <typename InputIterator>
  std::pair<bool, size_t> check_stabilizer_opt(InputIterator first, InputIterator last) const;

  template <typename InputIterator>
  bool check_measurement_opt(InputIterator first, InputIterator last) const;

};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

const stringmap_t<Gates> State::gateset_({
  // Single qubit gates
  {"delay", Gates::id},  // Delay gate
  {"id", Gates::id},     // Pauli-Identity gate
  {"x", Gates::x},       // Pauli-X gate
  {"y", Gates::y},       // Pauli-Y gate
  {"z", Gates::z},       // Pauli-Z gate
  {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
  {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
  {"sx", Gates::sx},     // sqrt(X) gate
  {"sxdg", Gates::sxdg}, // Inverse sqrt(X) gate
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  // Waltz Gates
  {"u0", Gates::u0},     // idle gate in multiples of X90
  {"u1", Gates::u1},     // zero-X90 pulse waltz gate
  {"p", Gates::u1},      // zero-X90 pulse waltz gate
  // Two-qubit gates
  {"CX", Gates::cx},     // Controlled-X gate (CNOT)
  {"cx", Gates::cx},     // Controlled-X gate (CNOT)
  {"cz", Gates::cz},     // Controlled-Z gate
  {"swap", Gates::swap}, // SWAP gate
  // Three-qubit gates
  {"ccx", Gates::ccx},    // Controlled-CX gate (Toffoli)
  {"ccz", Gates::ccz},    // Constrolled-CZ gate (H3 Toff H3)
  // Multi-qubit Pauli
  {"pauli", Gates::pauli} // Multi-qubit Pauli gate
});

//-------------------------------------------------------------------------
// Implementation: Initialisation and Config
//-------------------------------------------------------------------------

void State::initialize_qreg_state(QuantumState::RegistersBase& state_in, const uint_t num_qubits)
{
  QuantumState::Registers<chstate_t>& state = dynamic_cast<QuantumState::Registers<chstate_t>&>(state_in);
  if(state.qregs().size() == 0)
    state.allocate(1);
  state.qreg().initialize(num_qubits);
  state.qreg().initialize_omp(BaseState::threads_, omp_threshold_rank_);
}

void State::initialize_qreg_state(QuantumState::RegistersBase& state_in, const chstate_t &chstate)
{
  QuantumState::Registers<chstate_t>& state = dynamic_cast<QuantumState::Registers<chstate_t>&>(state_in);
  if(state.qregs().size() == 0)
    state.allocate(1);

  if(state.qreg().get_n_qubits() != BaseState::num_qubits_)
  {
    throw std::invalid_argument("CH::State::initialize: initial state does not match qubit number.");
  }

  state.qreg() = chstate;
  state.qreg().initialize_omp(BaseState::threads_, omp_threshold_rank_);
}

void State::set_state_config(QuantumState::RegistersBase& state_in, const json_t &config)
{
  // Set the error upper bound in the stabilizer rank approximation
  JSON::get_value(approximation_error_, "extended_stabilizer_approximation_error", config);
  // Set the number of samples used in the norm estimation routine
  JSON::get_value(norm_estimation_samples_, "extended_stabilizer_norm_estimation_default_samples", config);
  // Set the desired number of repetitions of the norm estimation step. If not explicitly set, we
  // compute a default basd on the approximation error
  norm_estimation_repetitions_ = std::llrint(std::log2(1. / approximation_error_));
  JSON::get_value(norm_estimation_repetitions_, "extended_stabilizer_norm_estimation_repetitions", config);
  // Set the number of steps used in the metropolis sampler before we
  // consider the distribution as approximating the output
  JSON::get_value(metropolis_mixing_steps_, "extended_stabilizer_metropolis_mixing_time", config);
  //Set the threshold of the decomposition before we use omp
  JSON::get_value(omp_threshold_rank_, "extended_stabilizer_parallel_threshold", config);
  //Set the truncation threshold for the probabilities snapshot.
  JSON::get_value(snapshot_chop_threshold_, "zero_threshold", config);
  //Set the number of samples for the probabilities snapshot
  JSON::get_value(probabilities_snapshot_samples_, "extended_stabilizer_probabilities_snapshot_samples", config);
  //Set the measurement strategy
  std::string sampling_method_str = "resampled_metropolis";
  JSON::get_value(sampling_method_str, "extended_stabilizer_sampling_method", config);
  if (sampling_method_str == "metropolis") {
    sampling_method_ = SamplingMethod::metropolis;
  }
  else if (sampling_method_str == "resampled_metropolis")
  {
    sampling_method_ = SamplingMethod::resampled_metropolis;
  }
  else if (sampling_method_str == "norm_estimation") {
    sampling_method_ = SamplingMethod::norm_estimation;
  }
  else {
    throw std::runtime_error(
      std::string("Unrecognised sampling method ") + sampling_method_str +
      std::string("for the extended stabilizer simulator.")
    );
  }
}

template <typename InputIterator>
std::pair<uint_t, uint_t> State::decomposition_parameters(InputIterator first, InputIterator last)
{
  double xi=1.;
  unsigned three_qubit_gate_count = 0;
  for (auto op = first; op != last; op++)
  {
    if (op->type == Operations::OpType::gate)
    {
      compute_extent(op, xi);
      auto it = CHSimulator::gate_types_.find(op->name);
      if (it->second == CHSimulator::Gatetypes::non_clifford && op->qubits.size() == 3)
      { //We count the number of 3 qubit gates for normalisation purposes
        three_qubit_gate_count++;
      }
    }
  }
  uint_t chi=1;
  if (xi >1)
  {
    double err_scaling = std::pow(approximation_error_, -2);
    chi = std::llrint(std::ceil(xi*err_scaling));
  }
  return std::pair<uint_t, uint_t>({chi, three_qubit_gate_count});
}

template <typename InputIterator>
std::pair<bool, size_t> State::check_stabilizer_opt(InputIterator first, InputIterator last) const
{
  for(auto op = first; op != last; op++)
  {
    if (op->type != Operations::OpType::gate)
    {
      continue;
    }
    auto it = CHSimulator::gate_types_.find(op->name);
    if(it == CHSimulator::gate_types_.end())
    {
      throw std::invalid_argument("CHState::check_measurement_opt doesn't recognise a the operation \'"+
                                   op->name + "\'.");
    }
    if(it->second == CHSimulator::Gatetypes::non_clifford)
    {
      return std::pair<bool, size_t>({false, op - first});
    }
  }
  return std::pair<bool, size_t>({true, 0});
}

template <typename InputIterator>
bool State::check_measurement_opt(InputIterator first, InputIterator last) const
{
  for (auto op = first; op != last; op++)
  {
    if (op->conditional)
    {
      return false;
    }
    if (op->type == Operations::OpType::measure ||
        op->type == Operations::OpType::bfunc ||
        op->type == Operations::OpType::save_statevec ||
        op->type == Operations::OpType::save_expval)
    {
      return false;
    }
  }
  return true;
}

//-------------------------------------------------------------------------
// Implementation: Operations
//-------------------------------------------------------------------------
void State::apply_op(QuantumState::RegistersBase& state_in, const Operations::Op &op, ExperimentResult &result,
                     RngEngine &rng, bool final_op) 
{
  QuantumState::Registers<chstate_t>& state = dynamic_cast<QuantumState::Registers<chstate_t>&>(state_in);
  apply_ops_state(state, &op, &op+1, result, rng, final_op);
}

template <typename InputIterator>
void State::apply_ops(InputIterator first, InputIterator last, ExperimentResult &result,
                      RngEngine &rng, bool final_ops)
{
  apply_ops_state(BaseState::state_,first, last, result, rng, final_ops);
}

template <typename InputIterator>
void State::apply_ops_state(QuantumState::Registers<chstate_t>& state, InputIterator first, InputIterator last, ExperimentResult &result,
                      RngEngine &rng, bool final_ops)
{
  std::pair<bool, size_t> stabilizer_opts = check_stabilizer_opt(first, last);
  bool is_stabilizer = stabilizer_opts.first;
  if(is_stabilizer)
  {
    apply_stabilizer_circuit(state, first, last, result, rng);
  }
  else
  {
    //Split the circuit into stabilizer and non-stabilizer fractions
    size_t first_non_clifford = stabilizer_opts.second;
    if (first_non_clifford > 0)
    {
      //Apply the stabilizer circuit first. This optimisaiton avoids duplicating the application
      //of the initial stabilizer circuit chi times.
      apply_stabilizer_circuit(state, first, first+first_non_clifford, result, rng);
    }

    auto it_nonstab_begin = first+first_non_clifford;

    uint_t chi = compute_chi(it_nonstab_begin, last);
    double delta = std::pow(approximation_error_, -2);
    state.qreg().initialize_decomposition(chi, delta);
    //Check for measurement optimisaitons
    bool measurement_opt = check_measurement_opt(first, last);

    if(measurement_opt)
    {
      apply_ops_parallel(state, it_nonstab_begin, last, result, rng);
    }
    else
    {
      for (auto it = it_nonstab_begin; it != last; it++)
      {
        const auto op = *it;
        if(state.creg().check_conditional(op)) {
          switch (op.type) {
            case Operations::OpType::gate:
              apply_gate(state, op, rng);
              break;
            case Operations::OpType::reset:
              apply_reset(state, op.qubits, rng);
              break;
            case Operations::OpType::barrier:
            case Operations::OpType::qerror_loc:
              break;
            case Operations::OpType::measure:
              apply_measure(state, op.qubits, op.memory, op.registers, rng);
              break;
            case Operations::OpType::roerror:
              state.creg().apply_roerror(op, rng);
              break;
            case Operations::OpType::bfunc:
              state.creg().apply_bfunc(op);
              break;
            case Operations::OpType::save_statevec:
              apply_save_statevector(state, op, result);
              break;
            // Disabled until can fix bug in expval
            // case Operations::OpType::save_expval:
            // case Operations::OpType::save_expval_var:
            //   apply_save_expval(op, result, rng);
            //   break;
            default:
              throw std::invalid_argument("CH::State::apply_ops does not support operations of the type \'" + 
                                          op.name + "\'.");
              break;
          }
        }
      }
    }
  }
}

std::vector<reg_t> State::sample_measure(QuantumState::RegistersBase& state_in, const reg_t& qubits,
                           uint_t shots,
                           RngEngine &rng)
{
  QuantumState::Registers<chstate_t>& state = dynamic_cast<QuantumState::Registers<chstate_t>&>(state_in);

  std::vector<uint_t> output_samples;
  if(state.qreg().get_num_states() == 1)
  {
    output_samples = state.qreg().stabilizer_sampler(shots, rng);
  }
  else
  {
    if (sampling_method_ == SamplingMethod::metropolis)
    {
      output_samples = state.qreg().metropolis_estimation(metropolis_mixing_steps_, shots, rng);
    }
    else if (sampling_method_ == SamplingMethod::resampled_metropolis)
    {
      output_samples.reserve(shots);
      for (uint_t i=0; i<shots; i++)
      {
        output_samples.push_back(
          state.qreg().metropolis_estimation(metropolis_mixing_steps_, rng)
        );
      }
    }
    else
    {
      output_samples.reserve(shots);
      for(uint_t i=0; i<shots; i++)
      {
        output_samples.push_back(
          state.qreg().ne_single_sample(norm_estimation_samples_, norm_estimation_repetitions_, true, qubits, rng)
        );
      }
    }
  }
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for(uint_t sample: output_samples)
  {
    reg_t sample_bits(qubits.size(), 0ULL);
    for(size_t i=0; i<qubits.size(); i++)
    {
      if((sample >> qubits[i]) & 1ULL)
      {
        sample_bits[i] = 1ULL;
      }
    }
    all_samples.push_back(sample_bits);
  }
  return all_samples;
}


//-------------------------------------------------------------------------
// Implemenation: Protected Methods
//-------------------------------------------------------------------------

//Method with slighty optimized parallelisation for the case of a sample_measure circuit
template <typename InputIterator>
void State::apply_ops_parallel(QuantumState::Registers<chstate_t>& state,InputIterator first, InputIterator last, ExperimentResult &result, RngEngine &rng)
{
  const int_t NUM_STATES = state.qreg().get_num_states();
  #pragma omp parallel for if(state.qreg().check_omp_threshold() && BaseState::threads_>1) num_threads(BaseState::threads_)
  for(int_t i=0; i < NUM_STATES; i++)
  {
    if(!state.qreg().check_eps(i))
    {
      continue;
    }
    for(auto it = first; it != last; it++)
    {
      switch (it->type)
      {
        case Operations::OpType::gate:
          apply_gate(state, *it, rng, i);
          break;
        case Operations::OpType::barrier:
        case Operations::OpType::qerror_loc:
          break;
        default:
          throw std::invalid_argument("CH::State::apply_ops_parallel does not support operations of the type \'" + 
                                       it->name + "\'.");
          break;
      }
    }
  }
}

template <typename InputIterator>
void State::apply_stabilizer_circuit(QuantumState::Registers<chstate_t>& state,InputIterator first, InputIterator last,
                                     ExperimentResult &result, RngEngine &rng)
{
  for (auto it = first; it != last; ++it)
  {
    const Operations::Op op = *it;
    if(state.creg().check_conditional(op)) {
      switch (op.type)
      {
        case Operations::OpType::gate:
          apply_gate(state,op, rng, 0);
          break;
        case Operations::OpType::reset:
          apply_reset(state,op.qubits, rng);
          break;
        case Operations::OpType::barrier:
        case Operations::OpType::qerror_loc:
          break;
        case Operations::OpType::measure:
          apply_measure(state, op.qubits, op.memory, op.registers, rng);
          break;
        case Operations::OpType::roerror:
          state.creg().apply_roerror(op, rng);
          break;
        case Operations::OpType::bfunc:
          state.creg().apply_bfunc(op);
          break;
        case Operations::OpType::save_statevec:
          apply_save_statevector(state, op, result);
          break;
        case Operations::OpType::save_expval:
        case Operations::OpType::save_expval_var:
          apply_save_expval(state, op, result, rng);
          break;
        default:
          throw std::invalid_argument("CH::State::apply_stabilizer_circuit does not support operations of the type \'" + 
                                      op.name + "\'.");
          break;
      }
    }
  }
}

void State::apply_measure(QuantumState::Registers<chstate_t>& state,const reg_t &qubits, const reg_t &cmemory, const reg_t &cregister, RngEngine &rng)
{
  uint_t out_string;
  // Flag if the Pauli projector is applied already as part of the sampling
  bool do_projector_correction = true;
  // Prepare an output register for the qubits we are measurig
  reg_t outcome(qubits.size(), 0ULL);
  if(state.qreg().get_num_states() == 1)
  {
    //For a single state, we use the efficient sampler defined in Sec IV.A ofarxiv:1808.00128
    out_string = state.qreg().stabilizer_sampler(rng);
  }
  else
  {
    if (sampling_method_ == SamplingMethod::norm_estimation)
    {
      do_projector_correction = false;
      //Run the norm estimation routine
      out_string = state.qreg().ne_single_sample(
        norm_estimation_samples_, norm_estimation_repetitions_, false, qubits, rng
      );
    }
    else
    {
      // We use the metropolis algorithm to sample an output string non-destructively
      // This is a single measure step so we do the same for metropolis or resampled_metropolis
      out_string = state.qreg().metropolis_estimation(metropolis_mixing_steps_, rng);
    }
  }
  if (do_projector_correction)
  {
    //We prepare the Pauli projector corresponding to the measurement result
    std::vector<chpauli_t>paulis(qubits.size(), chpauli_t());
    for (uint_t i=0; i<qubits.size(); i++)
    {
      //Create a Pauli projector onto 1+-Z_{i} on qubit i
      paulis[i].Z = (1ULL << qubits[i]);
      if ((out_string >> qubits[i]) & 1ULL)
      {
        //Additionally, store the output bit for this qubit
        paulis[i].e = 2;
      }
    }
    //Project the decomposition onto the measurement outcome
    state.qreg().apply_pauli_projector(paulis);
  }
  for (uint_t i=0; i<qubits.size(); i++)
  {
    //Create a Pauli projector onto 1+-Z_{i} on qubit i
    if ((out_string >> qubits[i]) & 1ULL)
    {
      //Additionally, store the output bit for this qubit
      outcome[i] = 1ULL;
    }
  }
  // Convert the output string to a reg_t. and store
  state.creg().store_measure(outcome, cmemory, cregister);
}

void State::apply_reset(QuantumState::Registers<chstate_t>& state,const reg_t &qubits, AER::RngEngine &rng)
{
  uint_t measure_string;
  const int_t NUM_STATES = state.qreg().get_num_states();
  if(state.qreg().get_num_states() == 1)
  {
    measure_string = state.qreg().stabilizer_sampler(rng);
  }
  else
  {
    measure_string = state.qreg().metropolis_estimation(metropolis_mixing_steps_, rng);
  }

  std::vector<chpauli_t> paulis(qubits.size(), chpauli_t());
  for(size_t i=0; i<qubits.size(); i++)
  {
    uint_t qubit = qubits[i];
    uint_t shift = 1ULL << qubit;
    paulis[i].Z = shift;
    if(!!(measure_string & shift))
    {
      paulis[i].e = 2;
    }
  }
  state.qreg().apply_pauli_projector(paulis);
  #pragma omp parallel for if(BaseState::threads_ > 1 && state.qreg().check_omp_threshold()) num_threads(BaseState::threads_)
  for(int_t i=0; i< NUM_STATES; i++)
  {
    for (auto qubit: qubits)
    {
      if ((measure_string>>qubit) & 1ULL)
      {
        state.qreg().apply_x(qubit, i);
      }
    }
  }
}

void State::apply_gate(QuantumState::Registers<chstate_t>& state,const Operations::Op &op, RngEngine &rng)
{
  const int_t NUM_STATES = state.qreg().get_num_states();
  #pragma omp parallel for if (BaseState::threads_ > 1 && state.qreg().check_omp_threshold()) num_threads(BaseState::threads_)
  for(int_t i=0; i < NUM_STATES; i++)
  {
    if(state.qreg().check_eps(i))
    {
      apply_gate(state, op, rng, i);
    }
  }
}

void State::apply_gate(QuantumState::Registers<chstate_t>& state,const Operations::Op &op, RngEngine &rng, uint_t rank)
{
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
  {
    throw std::invalid_argument("CH::State: Invalid gate operation \'"
                                +op.name + "\'.");
  }
  switch(it->second)
  {
    case Gates::x:
      state.qreg().apply_x(op.qubits[0], rank);
      break;
    case Gates::y:
      state.qreg().apply_y(op.qubits[0], rank);
      break;
    case Gates::z:
      state.qreg().apply_z(op.qubits[0], rank);
      break;
    case Gates::s:
      state.qreg().apply_s(op.qubits[0], rank);
      break;
    case Gates::sdg:
      state.qreg().apply_sdag(op.qubits[0], rank);
      break;
    case Gates::h:
      state.qreg().apply_h(op.qubits[0], rank);
      break;
    case Gates::sx:
      BaseState::add_global_phase(M_PI / 4.);
      state.qreg().apply_sx(op.qubits[0], rank);
      break;
    case Gates::sxdg:
      BaseState::add_global_phase(-M_PI / 4.);
      state.qreg().apply_sxdg(op.qubits[0], rank);
      break;
    case Gates::cx:
      state.qreg().apply_cx(op.qubits[0], op.qubits[1], rank);
      break;
    case Gates::cz:
      state.qreg().apply_cz(op.qubits[0], op.qubits[1], rank);
      break;
    case Gates::swap:
      state.qreg().apply_swap(op.qubits[0], op.qubits[1], rank);
      break;
    case Gates::t:
      state.qreg().apply_t(op.qubits[0], rng.rand(), rank);
      break;
    case Gates::tdg:
      state.qreg().apply_tdag(op.qubits[0], rng.rand(), rank);
      break;
    case Gates::ccx:
      state.qreg().apply_ccx(op.qubits[0], op.qubits[1], op.qubits[2], rng.rand_int(zero, toff_branch_max), rank);
      break;
    case Gates::ccz:
      state.qreg().apply_ccz(op.qubits[0], op.qubits[1], op.qubits[2], rng.rand_int(zero, toff_branch_max), rank);
      break;
    case Gates::u1:
      state.qreg().apply_u1(op.qubits[0], op.params[0], rng.rand(), rank);
      break;
    case Gates::pauli:
      apply_pauli(state, op.qubits, op.string_params[0], rank);
      break;
    default: //u0 or Identity
      break;
  }
}

void State::apply_pauli(QuantumState::Registers<chstate_t>& state, const reg_t &qubits, const std::string& pauli, uint_t rank) {
  const auto size = qubits.size();
  for (size_t i = 0; i < qubits.size(); ++i) {
    const auto qubit = qubits[size - 1 - i];
    switch (pauli[i]) {
      case 'I':
        break;
      case 'X':
        state.qreg().apply_x(qubit, rank);
        break;
      case 'Y':
        state.qreg().apply_y(qubit, rank);
        break;
      case 'Z':
        state.qreg().apply_z(qubit, rank);
        break;
      default:
        throw std::invalid_argument("invalid Pauli \'" + std::to_string(pauli[i]) + "\'.");
    }
  }
}

void State::apply_save_statevector(QuantumState::Registers<chstate_t>& state,const Operations::Op &op,
                                   ExperimentResult &result) 
{
  if (op.qubits.size() != state.qreg().get_n_qubits()) {
    throw std::invalid_argument(
        "Save statevector was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  auto statevec = state.qreg().statevector();
  if (BaseState::has_global_phase_) {
    statevec *= BaseState::global_phase_;
  }
  result.save_data_pershot(
    state.creg(), op.string_params[0],
    std::move(statevec),
    op.type, op.save_type);
}

void State::apply_save_expval(QuantumState::Registers<chstate_t>& state,const Operations::Op &op,
                                ExperimentResult &result,
                                RngEngine& rng) 
{
  // Check empty edge case
  if (op.expval_params.empty()) {
    throw std::invalid_argument(
        "Invalid save expval instruction (Pauli components are empty).");
  }
  bool variance = (op.type == Operations::OpType::save_expval_var);

  // Accumulate expval components
  double expval(0.);
  double sq_expval(0.);

  for (const auto &param : op.expval_params) {
    // param is tuple (pauli, coeff, sq_coeff)
    const auto val = expval_pauli(state, op.qubits, std::get<0>(param), rng);
    expval += std::get<1>(param) * val;
    if (variance) {
      sq_expval += std::get<2>(param) * val;
    }
  }
  if (variance) {
    std::vector<double> expval_var(2);
    expval_var[0] = expval;  // mean
    expval_var[1] = sq_expval - expval * expval;  // variance
    result.save_data_average(state.creg(), op.string_params[0], expval_var, op.type, op.save_type);
  } else {
    result.save_data_average(state.creg(), op.string_params[0], expval, op.type, op.save_type);
  }
}


double State::expval_pauli(QuantumState::Registers<chstate_t>& state, const reg_t &qubits,
                           const std::string& pauli,
                           RngEngine &rng) 
{
    // Compute expval components
    auto state_cpy = state.qreg();
    auto phi_norm = state_cpy.norm_estimation(norm_estimation_samples_, norm_estimation_repetitions_, rng);
    std::vector<chpauli_t>paulis(1, chpauli_t());
    for (uint_t pos = 0; pos < qubits.size(); ++pos) {
      switch (pauli[pauli.size() - 1 - pos]) {
        case 'I':
          break;
        case 'X':
          paulis[0].X += (1ULL << qubits[pos]);
          break;
        case 'Y':
          paulis[0].X += (1ULL << qubits[pos]);
          paulis[0].Z += (1ULL << qubits[pos]);
          break;
        case 'Z':
          paulis[0].Z += (1ULL << qubits[pos]);
          break;
        default: {
          std::stringstream msg;
          msg << "QubitVectorState::invalid Pauli string \'" << pauli[pos]
              << "\'.";
          throw std::invalid_argument(msg.str());
        }
      }
    }
    auto g_norm = state_cpy.norm_estimation(norm_estimation_samples_, norm_estimation_repetitions_, paulis, rng);
    return (2*g_norm - phi_norm);
}

double State::expval_pauli(QuantumState::RegistersBase& state, const reg_t &qubits,
                           const std::string& pauli) {
    // empty implementation of base class virtual method
    // since in the extended stabilizer, expval relies on RNG
    return 0;
}

//-------------------------------------------------------------------------
// Implementation: Utility
//-------------------------------------------------------------------------

inline void to_json(json_t &js, cvector_t vec)
{
  js = json_t();
  for (uint_t j=0; j < vec.size(); j++) {
    js.push_back(vec[j]);
  }
}

template <typename InputIterator>
uint_t State::compute_chi(InputIterator first, InputIterator last) const
{
  double xi = 1;
  for (auto op = first; op != last; op++)
  {
    compute_extent(*op, xi);
  }
  double err_scaling = std::pow(approximation_error_, -2);
  return std::llrint(std::ceil(xi*err_scaling));
}

void State::compute_extent(const Operations::Op &op, double &xi) const
{
  if(op.type == Operations::OpType::gate)
  {
    auto it = gateset_.find(op.name);
    if (it == gateset_.end())
    {
      throw std::invalid_argument("CH::State: Invalid gate operation \'"
                                  +op.name + "\'.");
    }
    switch (it->second)
    {
      case Gates::t:
        xi *= CHSimulator::t_extent;
        break;
      case Gates::tdg:
        xi *= CHSimulator::t_extent;
        break;
      case Gates::ccx:
        xi *= CHSimulator::ccx_extent;
        break;
      case Gates::ccz:
        xi *= CHSimulator::ccx_extent;
        break;
      case Gates::u1:
        xi *= CHSimulator::u1_extent(std::real(op.params[0]));
        break;
      default:
        break;
    }
  }
}

size_t State::required_memory_mb(uint_t num_qubits,
                                 QuantumState::OpItr first, QuantumState::OpItr last) const
{
  size_t required_chi = compute_chi(first, last);
  // 5 vectors of num_qubits*8byte words
  // Plus 2*CHSimulator::scalar_t which has 3 4 byte words
  // Plus 2*CHSimulator::pauli_t which has 2 8 byte words and one 4 byte word;
  double mb_per_state = 5e-5*num_qubits;//
  size_t required_mb = std::llrint(std::ceil(mb_per_state*required_chi));

  if (sampling_method_ == SamplingMethod::norm_estimation)
  {
    required_mb *= 2;
  }
  return required_mb;
}

}
}

#endif
