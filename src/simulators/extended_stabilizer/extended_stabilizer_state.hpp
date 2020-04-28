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

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::measure,
    Operations::OpType::reset, Operations::OpType::barrier,
    Operations::OpType::roerror, Operations::OpType::bfunc,
    Operations::OpType::snapshot},
  // Gates
  {"CX", "u0", "u1", "cx", "cz", "swap", "id", "x", "y", "z", "h",
    "s", "sdg", "t", "tdg", "ccx", "ccz"},
  // Snapshots
  {"statevector", "probabilities", "memory", "register"}
);

using chpauli_t = CHSimulator::pauli_t;
using chstate_t = CHSimulator::Runner;
using Gates = CHSimulator::Gates;

uint_t zero = 0ULL;
uint_t toff_branch_max = 7ULL;

enum class Snapshots {
  state, 
  statevector,
  probabilities,
  cmemory,
  cregister,
  probs
};

class State: public Base::State<chstate_t>
{
public:
  using BaseState = Base::State<chstate_t>;
  
  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  virtual std::string name() const override {return "extended_stabilizer";}

  //Apply a sequence of operations to the cicuit. For each operation,
  //we loop over the terms in the decomposition in parallel
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  virtual void initialize_qreg(uint_t num_qubits) override;

  virtual void initialize_qreg(uint_t num_qubits, const chstate_t &state) override;

  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const override;

  virtual void set_config(const json_t &config) override;

  virtual std::vector<reg_t> sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;

protected:

  //Alongside the sample measure optimisaiton, we can parallelise
  //circuit applicaiton over the states. This reduces the threading overhead
  //as we only have to fork once per circuit.
  void apply_ops_parallel(const std::vector<Operations::Op> &ops,
                                  RngEngine &rng);

  //Small routine that eschews any parallelisation/decomposition and applies a stabilizer
  //circuit to a single state. This is used to optimize a circuit with a large
  //initial clifford fraction, or for running stabilizer circuits.
  void apply_stabilizer_circuit(const std::vector<Operations::Op> &ops,
                                      ExperimentData &data,
                                      RngEngine &rng);
  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  // TODO: Investigate OMP synchronisation over stattes to remove these different versions
  // One option would be tasks, but the memory overhead isn't clear
  void apply_gate(const Operations::Op &op, RngEngine &rng);
  void apply_gate(const Operations::Op &op, RngEngine &rng, uint_t rank);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function then "measure" 
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  void apply_measure(const reg_t &qubits,
                     const reg_t &cmemory,
                     const reg_t &cregister,
                     RngEngine &rng);

  // Reset the specified qubits to the |0> state by measuring the
  // projectors Id+Z_{i} for each qubit i
  void apply_reset(const reg_t &qubits, AER::RngEngine &rng);

  //Take a snapshot of the simulation state
  //TODO: Improve the CHSimulator::to_json method.
  void apply_snapshot(const Operations::Op &op, ExperimentData &data, RngEngine &rng);
  //Convert a decomposition to a state-vector
  void statevector_snapshot(const Operations::Op &op, ExperimentData &data, RngEngine &rng);
  //Compute probabilities from a stabilizer rank decomposition
  //TODO: Check ordering/output format...
  void probabilities_snapshot(const Operations::Op &op, ExperimentData &data, RngEngine &rng);

  const static stringmap_t<Gates> gateset_;
  const static stringmap_t<Snapshots> snapshotset_;

  //-----------------------------------------------------------------------
  //Parameters and methods specific to the Stabilizer Rank Decomposition
  //-----------------------------------------------------------------------

  //Allowed error in the stabilizer rank decomposition.
  //The required number of states scales as \delta^{-2}
  //for allowed error \delta
  double approximation_error_ = 0.05;

  uint_t norm_estimation_samples_ = 100;
  // How long the metropolis algorithm runs before
  // we consider it to be well mixed and sample form the
  // output distribution
  uint_t metropolis_mixing_steps_ = 5000;

  //Minimum number of states before we try to parallelise
  uint_t omp_threshold_rank_ = 100;

  double snapshot_chop_threshold_ = 1e-10;

  double probabilities_snapshot_samples_ = 3000.;

  // Compute the required stabilizer rank of the circuit
  uint_t compute_chi(const std::vector<Operations::Op> &ops) const;
  // Add the given operation to the extent
  void compute_extent(const Operations::Op &op, double &xi) const;

  //Compute the required chi, and count the number of three qubit gates
  std::pair<uint_t, uint_t> decomposition_parameters(const std::vector<Operations::Op> &ops);
  
  //Check if this is a stabilizer circuit, for the locaiton of the first non-CLifford gate
  std::pair<bool, size_t> check_stabilizer_opt(const std::vector<Operations::Op> &ops) const;

  //Check if we can use the sample_measure optimisation
  bool check_measurement_opt(const std::vector<Operations::Op> &ops) const;
};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

const stringmap_t<Gates> State::gateset_({
  // Single qubit gates
  {"id", Gates::id},     // Pauli-Identity gate
  {"x", Gates::x},       // Pauli-X gate
  {"y", Gates::y},       // Pauli-Y gate
  {"z", Gates::z},       // Pauli-Z gate
  {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
  {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  // Waltz Gates
  {"u0", Gates::u0},     // idle gate in multiples of X90
  {"u1", Gates::u1},     // zero-X90 pulse waltz gate
  // Two-qubit gates
  {"CX", Gates::cx},     // Controlled-X gate (CNOT)
  {"cx", Gates::cx},     // Controlled-X gate (CNOT)
  {"cz", Gates::cz},     // Controlled-Z gate
  {"swap", Gates::swap}, // SWAP gate
  // Three-qubit gates
  {"ccx", Gates::ccx},    // Controlled-CX gate (Toffoli)
  {"ccz", Gates::ccz}     // Constrolled-CZ gate (H3 Toff H3)
});

const stringmap_t<Snapshots> State::snapshotset_({
  {"state", Snapshots::state},
  {"statevector", Snapshots::statevector},
  {"probabilities", Snapshots::probabilities},
  {"memory", Snapshots::cmemory},
  {"register", Snapshots::cregister}
});

//-------------------------------------------------------------------------
// Implementation: Initialisation and Config
//-------------------------------------------------------------------------

void State::initialize_qreg(uint_t num_qubits)
{
  BaseState::qreg_.initialize(num_qubits);
  BaseState::qreg_.initialize_omp(BaseState::threads_, omp_threshold_rank_);
}

void State::initialize_qreg(uint_t num_qubits, const chstate_t &state)
{
  if(BaseState::qreg_.get_n_qubits() != num_qubits)
  {
    throw std::invalid_argument("CH::State::initialize: initial state does not match qubit number.");
  }
  BaseState::qreg_ = state;
  BaseState::qreg_.initialize_omp(BaseState::threads_, omp_threshold_rank_);
}

void State::set_config(const json_t &config)
{
  // Set the error upper bound in the stabilizer rank approximation
  JSON::get_value(approximation_error_, "extended_stabilizer_approximation_error", config);
  // Set the number of samples used in the norm estimation routine
  JSON::get_value(norm_estimation_samples_, "extended_stabilizer_norm_estimation_samples", config);
  // Set the number of steps used in the metropolis sampler before we
  // consider the distribution as approximating the output
  JSON::get_value(metropolis_mixing_steps_, "extended_stabilizer_mixing_time", config);
  //Set the threshold of the decomposition before we use omp
  JSON::get_value(omp_threshold_rank_, "extended_stabilizer_parallel_threshold", config);
  //Set the truncation threshold for the probabilities snapshot.
  JSON::get_value(snapshot_chop_threshold_, "zero_threshold", config);
  //Set the number of samples for the probabilities snapshot
  JSON::get_value(probabilities_snapshot_samples_, "probabilities_snapshot_samples", config);
}

std::pair<uint_t, uint_t> State::decomposition_parameters(const std::vector<Operations::Op> &ops)
{
  double xi=1.;
  unsigned three_qubit_gate_count = 0;
  for (const auto op: ops)
  {
    if (op.type == Operations::OpType::gate)
    {
      compute_extent(op, xi);
      auto it = CHSimulator::gate_types_.find(op.name);
      if (it->second == CHSimulator::Gatetypes::non_clifford && op.qubits.size() == 3)
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

std::pair<bool, size_t> State::check_stabilizer_opt(const std::vector<Operations::Op> &ops) const
{
  for(auto op = ops.cbegin(); op != ops.cend(); op++)
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
      return std::pair<bool, size_t>({false, op - ops.cbegin()});
    }
  }
  return std::pair<bool, size_t>({true, 0});
}

bool State::check_measurement_opt(const std::vector<Operations::Op> &ops) const
{
  for (const auto op: ops)
  {
    if (op.conditional || op.old_conditional)
    {
      return false;
    }
    if (op.type == Operations::OpType::measure || op.type == Operations::OpType::bfunc ||
        op.type == Operations::OpType::snapshot)
    {
      return false;
    }
  }
  return true;
}

//-------------------------------------------------------------------------
// Implementation: Operations
//-------------------------------------------------------------------------

void State::apply_ops(const std::vector<Operations::Op> &ops, ExperimentData &data,
                         RngEngine &rng)
{
  std::pair<bool, size_t> stabilizer_opts = check_stabilizer_opt(ops);
  bool is_stabilizer = stabilizer_opts.first;
  if(is_stabilizer)
  {
    apply_stabilizer_circuit(ops, data, rng);
  }
  else
  {
    //Split the circuit into stabilizer and non-stabilizer fractions
    size_t first_non_clifford = stabilizer_opts.second;
    if (first_non_clifford > 0)
    {
      //Apply the stabilizer circuit first. This optimisaiton avoids duplicating the application
      //of the initial stabilizer circuit chi times.
      std::vector<Operations::Op> stabilizer_circuit(ops.cbegin(), ops.cbegin()+first_non_clifford);
      apply_stabilizer_circuit(stabilizer_circuit, data, rng);
    }
    std::vector<Operations::Op> non_stabilizer_circuit(ops.cbegin()+first_non_clifford, ops.cend());
    uint_t chi = compute_chi(non_stabilizer_circuit);
    BaseState::qreg_.initialize_decomposition(chi);
    //Check for measurement optimisaitons
    bool measurement_opt = check_measurement_opt(ops);
    if(measurement_opt)
    {
      apply_ops_parallel(non_stabilizer_circuit, rng);
    }
    else
    {
      for (const auto op: non_stabilizer_circuit)
      {
        if(BaseState::creg_.check_conditional(op)) {
          switch (op.type) {
            case Operations::OpType::gate:
              apply_gate(op, rng);
              break;
            case Operations::OpType::reset:
              apply_reset(op.qubits, rng);
              break;
            case Operations::OpType::barrier:
              break;
            case Operations::OpType::measure:
              apply_measure(op.qubits, op.memory, op.registers, rng);
              break;
            case Operations::OpType::roerror:
              BaseState::creg_.apply_roerror(op, rng);
              break;
            case Operations::OpType::bfunc:
              BaseState::creg_.apply_bfunc(op);
              break;
            case Operations::OpType::snapshot:
              apply_snapshot(op, data, rng);
              break;
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

std::vector<reg_t> State::sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng)
{
  std::vector<uint_t> output_samples;
  if(BaseState::qreg_.get_num_states() == 1)
  {
    output_samples = BaseState::qreg_.stabilizer_sampler(shots, rng);
  }
  else
  {
    output_samples = BaseState::qreg_.metropolis_estimation(metropolis_mixing_steps_, shots, rng);
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
void State::apply_ops_parallel(const std::vector<Operations::Op> &ops, RngEngine &rng)
{
  const int_t NUM_STATES = BaseState::qreg_.get_num_states();
  #pragma omp parallel for if(BaseState::qreg_.check_omp_threshold() && BaseState::threads_>1) num_threads(BaseState::threads_)
  for(int_t i=0; i < NUM_STATES; i++)
  {
    if(!BaseState::qreg_.check_eps(i))
    {
      continue;
    }
    for(const auto op: ops)
    {
      switch (op.type)
      {
        case Operations::OpType::gate:
          apply_gate(op, rng, i);
          break;
        case Operations::OpType::barrier:
          break;
        default:
          throw std::invalid_argument("CH::State::apply_ops_parallel does not support operations of the type \'" + 
                                       op.name + "\'.");
          break;
      }
    }
  }
}

void State::apply_stabilizer_circuit(const std::vector<Operations::Op> &ops,
                                      ExperimentData &data, RngEngine &rng)
{
  for (const auto op: ops)
  {
    switch (op.type)
    {
      case Operations::OpType::gate:
        if(BaseState::creg_.check_conditional(op))
        {
          apply_gate(op, rng, 0);
        }
        break;
      case Operations::OpType::reset:
        apply_reset(op.qubits, rng);
        break;
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::measure:
        apply_measure(op.qubits, op.memory, op.registers, rng);
        break;
      case Operations::OpType::roerror:
        BaseState::creg_.apply_roerror(op, rng);
        break;
      case Operations::OpType::bfunc:
        BaseState::creg_.apply_bfunc(op);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, data, rng);
        break;
      default:
        throw std::invalid_argument("CH::State::apply_stabilizer_circuit does not support operations of the type \'" + 
                                     op.name + "\'.");
        break;
    }
  }
}

void State::apply_measure(const reg_t &qubits, const reg_t &cmemory, const reg_t &cregister, RngEngine &rng)
{
  uint_t full_string;
  if(BaseState::qreg_.get_num_states() == 1)
  {
    //For a single state, we use the efficient sampler defined in Sec IV.A ofarxiv:1808.00128
    full_string = BaseState::qreg_.stabilizer_sampler(rng);
  }
  else
  {
    //We use the metropolis algorithm to sample an output string non-destructively
    full_string = BaseState::qreg_.metropolis_estimation(metropolis_mixing_steps_, rng);
  }
  //We prepare the Pauli projector corresponding to the measurement result
  std::vector<chpauli_t>paulis(qubits.size(), chpauli_t());
  // Prepare an output register for the qubits we are measurig
  reg_t outcome(qubits.size(), 0ULL);
  for (uint_t i=0; i<qubits.size(); i++)
  {
    //Create a Pauli projector onto 1+-Z_{i} on qubit i
    paulis[i].Z = (1ULL << qubits[i]);
    if ((full_string >> qubits[i]) & 1ULL)
    {
      //Additionally, store the output bit for this qubit
      outcome[i]= 1ULL;
      paulis[i].e = 2;
    }
  }
  // Convert the output string to a reg_t. and store
  BaseState::creg_.store_measure(outcome, cmemory, cregister);
  //Project the decomposition onto the measurement outcome
  BaseState::qreg_.apply_pauli_projector(paulis);
}

void State::apply_reset(const reg_t &qubits, AER::RngEngine &rng)
{
  uint_t measure_string;
  const int_t NUM_STATES = BaseState::qreg_.get_num_states();
  if(BaseState::qreg_.get_num_states() == 1)
  {
    measure_string = BaseState::qreg_.stabilizer_sampler(rng);
  }
  else
  {
    measure_string = BaseState::qreg_.metropolis_estimation(metropolis_mixing_steps_, rng);
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
  BaseState::qreg_.apply_pauli_projector(paulis);
  #pragma omp parallel for if(BaseState::threads_ > 1 && BaseState::qreg_.check_omp_threshold()) num_threads(BaseState::threads_)
  for(int_t i=0; i< NUM_STATES; i++)
  {
    for (auto qubit: qubits)
    {
      if ((measure_string>>qubit) & 1ULL)
      {
        BaseState::qreg_.apply_x(qubit, i);
      }
    }
  }
}

void State::apply_gate(const Operations::Op &op, RngEngine &rng)
{
  const int_t NUM_STATES = BaseState::qreg_.get_num_states();
  #pragma omp parallel for if (BaseState::threads_ > 1 && BaseState::qreg_.check_omp_threshold()) num_threads(BaseState::threads_)
  for(int_t i=0; i < NUM_STATES; i++)
  {
    if(BaseState::qreg_.check_eps(i))
    {
      apply_gate(op, rng, i);
    }
  }
}

void State::apply_gate(const Operations::Op &op, RngEngine &rng, uint_t rank)
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
      BaseState::qreg_.apply_x(op.qubits[0], rank);
      break;
    case Gates::y:
      BaseState::qreg_.apply_y(op.qubits[0], rank);
      break;
    case Gates::z:
      BaseState::qreg_.apply_z(op.qubits[0], rank);
      break;
    case Gates::s:
      BaseState::qreg_.apply_s(op.qubits[0], rank);
      break;
    case Gates::sdg:
      BaseState::qreg_.apply_sdag(op.qubits[0], rank);
      break;
    case Gates::h:
      BaseState::qreg_.apply_h(op.qubits[0], rank);
      break;
    case Gates::cx:
      BaseState::qreg_.apply_cx(op.qubits[0], op.qubits[1], rank);
      break;
    case Gates::cz:
      BaseState::qreg_.apply_cz(op.qubits[0], op.qubits[1], rank);
      break;
    case Gates::swap:
      BaseState::qreg_.apply_swap(op.qubits[0], op.qubits[1], rank);
      break;
    case Gates::t:
      BaseState::qreg_.apply_t(op.qubits[0], rng.rand(), rank);
      break;
    case Gates::tdg:
      BaseState::qreg_.apply_tdag(op.qubits[0], rng.rand(), rank);
      break;
    case Gates::ccx:
      BaseState::qreg_.apply_ccx(op.qubits[0], op.qubits[1], op.qubits[2], rng.rand_int(zero, toff_branch_max), rank);
      break;
    case Gates::ccz:
      BaseState::qreg_.apply_ccz(op.qubits[0], op.qubits[1], op.qubits[2], rng.rand_int(zero, toff_branch_max), rank);
      break;
    case Gates::u1:
      BaseState::qreg_.apply_u1(op.qubits[0], op.params[0], rng.rand(), rank);
      break;
    default: //u0 or Identity
      break;
  }
}

void State::apply_snapshot(const Operations::Op &op, ExperimentData &data, RngEngine &rng)
{
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
  {
    throw std::invalid_argument("CH::State::invlaid snapshot instruction \'"+
                                op.name + "\'.");
  }
  switch(it->second)
  {
    case Snapshots::state:
      BaseState::snapshot_state(op, data, "extended_stabilizer_state");
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(op, data);
      break;
    case Snapshots::statevector:
      statevector_snapshot(op, data, rng);
      break;
    case Snapshots::probabilities:
      probabilities_snapshot(op, data, rng);
      break;
    default:
      throw std::invalid_argument("CH::State::invlaid snapshot instruction \'"+
                              op.name + "\'.");
      break;
  }
}

void State::statevector_snapshot(const Operations::Op &op, ExperimentData &data, RngEngine &rng)
{
  cvector_t statevector;
  BaseState::qreg_.state_vector(statevector, rng);
  double sum = 0.;
  for(uint_t i=0; i<statevector.size(); i++)
  {
    sum += std::pow(std::abs(statevector[i]), 2);
  }
  data.add_pershot_snapshot("statevector", op.string_params[0], statevector);
}

void State::probabilities_snapshot(const Operations::Op &op, ExperimentData &data, RngEngine &rng)
{
  rvector_t probs;
  if (op.qubits.size() == 0)
  {
    probs.push_back(BaseState::qreg_.norm_estimation(norm_estimation_samples_, rng));
  }
  else
  {
    probs = rvector_t(1ULL<<op.qubits.size(), 0.);
    int_t dim = probs.size();
    uint_t mask = 0ULL;
    for(auto qubit: op.qubits)
    {
      mask ^= (1ULL << qubit);
    }
    std::vector<uint_t> samples;
    if(BaseState::qreg_.get_num_states() == 1)
    {
      samples = BaseState::qreg_.stabilizer_sampler(probabilities_snapshot_samples_, rng);
    }
    else
    {
      samples = BaseState::qreg_.metropolis_estimation(metropolis_mixing_steps_, probabilities_snapshot_samples_,
                                                      rng);
    }
    #pragma omp parallel for if(BaseState::qreg_.check_omp_threshold() && BaseState::threads_>1) num_threads(BaseState::threads_)
    for(int_t i=0; i < dim; i++)
    {
      uint_t target = 0ULL;
      for(uint_t j=0; j<op.qubits.size(); j++)
      {
        if((dim >> j) & 1ULL)
        {
          target ^= (1ULL << op.qubits[j]);
        }
      }
      for(uint_t j=0; j<probabilities_snapshot_samples_; j++)
      {
        if((samples[j] & mask) == target)
        {
          probs[i] += 1;
        }
      }
      probs[i] /= probabilities_snapshot_samples_;
    }
  }
  data.add_average_snapshot("probabilities", op.string_params[0],
                            BaseState::creg_.memory_hex(),
                            Utils::vec2ket(probs, snapshot_chop_threshold_, 16),
                            false);
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

//
uint_t State::compute_chi(const std::vector<Operations::Op> &ops) const
{
  double xi = 1;
  for (const auto op: ops)
  {
    compute_extent(op, xi);
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
                                 const std::vector<Operations::Op> &ops)
                                 const
{
  size_t required_chi = compute_chi(ops);
  // 5 vectors of num_qubits*8byte words
  // Plus 2*CHSimulator::scalar_t which has 3 4 byte words
  // Plus 2*CHSimulator::pauli_t which has 2 8 byte words and one 4 byte word;
  double mb_per_state = 5e-5*num_qubits;//
  size_t required_mb = std::llrint(std::ceil(mb_per_state*required_chi));
  return required_mb;
  //Todo: Update this function to account for snapshots
}

}
}

#endif
