#ifndef _aer_extended_stabilizer_compute_hpp
#define _aer_extended_stabilizer_compute_hpp

#define _USE_MATH_DEFINES
#include <cmath>

#include <complex>
#include <vector>
#include <algorithm>

#include "simulators/state.hpp"
#include "framework/json.hpp"
#include "framework/types.hpp"

#include "ag_state.hpp"
#include "simulators/stabilizer/pauli.hpp"

namespace AER{
namespace CliffPhaseCompute{

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::save_specific_prob},
  // Gates
  {"CX", "cx", "cz", "swap", "id", "x", "y", "z", "h",
   "s", "sdg", "t","p", "rz", "u1"},
  // Snapshots
  {}
);



enum class Gates {
id, x, y, z, h, s, sdg, sx, t, tdg, cx, cz, swap,  p, rz, u1,
};


using agstate_t = CliffPhaseCompute::AGState;


class State: public Base::State<agstate_t>{
public:
  using BaseState = Base::State<agstate_t>;
  
  State() : BaseState(StateOpSet) {};
  virtual ~State() = default;

  std::string name() const override {return "clifford_phase_compute";}

  //Apply a sequence of operations to the cicuit.
  //We just store these operations in _circuit because we can't implement them
  //until we know how many non-Clifford gates and what the measurements are
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false) override;
  
  void apply_gate(const Operations::Op &op);

  void initialize_qreg(uint_t num_qubits) override;
  void initialize_qreg(uint_t num_qubits, const agstate_t &state) override;
  size_t required_memory_mb(uint_t num_qubits, const std::vector<Operations::Op> &ops) const override;

  void apply_save_specific_prob(const Operations::Op &op, ExperimentResult &result);
  double expval_pauli(const reg_t &qubits, const std::string& pauli);
private:  
  const static stringmap_t<Gates> gateset_;
  size_t num_code_qubits; //out AG state has code+magic qubits
  double compute_probability(std::vector<size_t> measured_qubits, std::vector<uint_t> outcomes);
};


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
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"rz", Gates::rz}, // Pauli-Z rotation gate
  {"p", Gates::rz},   // Parameterized phase gate
  {"u1", Gates::rz}, 
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  // Two-qubit gates
  {"CX", Gates::cx},     // Controlled-X gate (CNOT)
  {"cx", Gates::cx},     // Controlled-X gate (CNOT)
  {"CZ", Gates::cz},     // Controlled-Z gate
  {"cz", Gates::cz},     // Controlled-Z gate
  {"swap", Gates::swap}, // SWAP gate
  // Three-qubit gates
});


void State::apply_ops(const std::vector<Operations::Op> &ops, ExperimentResult &result, RngEngine &rng, bool final_ops){

  for(const auto &op: ops){
    switch(op.type){
    case Operations::OpType::gate:
      this->apply_gate(op);
      break;
    case Operations::OpType::save_specific_prob:
      this->apply_save_specific_prob(op, result);
      break;
    default:
      throw std::invalid_argument("Compute::State::invalid instruction \'" + op.name + "\'.");  
    } 
  }
}

void State::apply_gate(const Operations::Op &op){  
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
  {
    throw std::invalid_argument("Compute::State: Invalid gate operation \'"
                                +op.name + "\'.");
  }
  switch(it->second){
  case Gates::id:
    break;
  case Gates::x:
    this->qreg_.applyX(op.qubits[0]);
    break;
  case Gates::y:
    this->qreg_.applyY(op.qubits[0]);
    break;
  case Gates::z:
    this->qreg_.applyZ(op.qubits[0]);
    break;
  case Gates::s:
    this->qreg_.applyS(op.qubits[0]);
    break;
  case Gates::sdg:
    this->qreg_.applyZ(op.qubits[0]);
    this->qreg_.applyS(op.qubits[0]);
    break;
  case Gates::h:    
    this->qreg_.applyH(op.qubits[0]);
    break;
  case Gates::t:
    this->qreg_.gadgetized_phase_gate(op.qubits[0], T_ANGLE);
    break;
  case Gates::tdg:
    this->qreg_.gadgetized_phase_gate(op.qubits[0], -T_ANGLE);
    break;
  case Gates::rz:
    this->qreg_.gadgetized_phase_gate(op.qubits[0], op.params[0].real());
  case Gates::cx:
    this->qreg_.applyCX(op.qubits[0],op.qubits[1]);
    break;
  case Gates::cz:
    this->qreg_.applyCZ(op.qubits[0],op.qubits[1]);
    break;
  case Gates::swap:
    this->qreg_.applySwap(op.qubits[0],op.qubits[1]);
    break;
  default: //u0 or Identity
    break;
  }
}

void State::apply_save_specific_prob(const Operations::Op &op, ExperimentResult &result){  
  std::vector<double> v;
  double p = this->compute_probability(op.qubits, op.int_params);
  v.push_back(p);
  BaseState::save_data_average(result, op.string_params[0], std::move(v), Operations::DataSubType::list);
}

// This function converts an unsigned binary number to reflected binary Gray code.
uint_t BinaryToGray(uint_t num)
{
  return num ^ (num >> 1); // The operator >> is shift right. The operator ^ is exclusive or.
}


double compute_algorithm_all_phases_T(AGState &state){
  
  uint_t full_mask = 0u;
  for(size_t i = 0; i < state.num_stabilizers; i++){
    full_mask |= (ONE << i);
  }

  double acc = 1.;
  Pauli::Pauli row(state.num_qubits);
  unsigned char phase = 0;
  
  for(uint_t mask = 1u; mask <= full_mask; mask++){
    uint_t mask_with_bit_to_flip = BinaryToGray(mask) ^ BinaryToGray(mask - 1);
    size_t bit_to_flip = 0;
    for(size_t j = 0; j < state.num_stabilizers; j++){
      if((mask_with_bit_to_flip >> j) & ONE){
	bit_to_flip = j;
	break;
      }
    }

    phase += state.phases[bit_to_flip] + (Pauli::Pauli::phase_exponent(row, state.table[bit_to_flip]) / 2); //phases for stabilizers are always 1 or -1
    phase %= 2;
    row += state.table[bit_to_flip];

    size_t XCount = 0;
    size_t YCount = 0;
    size_t ZCount = 0;
    
    for(size_t j = 0; j < state.num_qubits; j++){
      //if((row[j] == 0) && (row[j+state.n] == 0)){
      //    ICount += 1;
      //}
      if(row.X[j] && !row.Z[j]){
	XCount += 1;
      }
      if(!row.X[j] && row.Z[j]){
	ZCount += 1;
	break;
      }
      if(row.X[j] && row.Z[j]){
	YCount += 1;
      }
    }
    
    if(ZCount == 0){
      if(((phase + YCount) % 2) == 0){
	acc += powl(1./2., (XCount + YCount)/2.);;
      }else{
	acc -= powl(1./2., (XCount + YCount)/2.);;
      }
    }      
  }

  if(full_mask == 0u){
    return 1.;
  }
  return acc;
}

double compute_algorithm_arbitrary_phases(AGState &state){
  uint_t full_mask = 0u;
  for(size_t i = 0; i < state.num_stabilizers; i++){
    full_mask |= (ONE << i);
  }
  double acc = 1.;
  Pauli::Pauli row(state.num_qubits);
  unsigned char phase = 0;
  
  for(uint_t mask = 1u; mask <= full_mask; mask++){
    uint_t mask_with_bit_to_flip = BinaryToGray(mask) ^ BinaryToGray(mask - 1);
    size_t bit_to_flip = 0;
    for(size_t j = 0; j < state.num_stabilizers; j++){
      if((mask_with_bit_to_flip >> j) & ONE){
	bit_to_flip = j;
	break;
      }
    }
    
    phase += (Pauli::Pauli::phase_exponent(row, state.table[bit_to_flip]) / 2); //phases for stabilizers are always 0 or 2
    row += state.table[bit_to_flip];
    
    double prod = 1.;
    for(size_t j = 0; j < state.num_qubits; j++){
      //if((row[j] == 0) && (row[j+state.n] == 0)){
      //    ICount += 1;
      //}
      if(row.X[j] && !row.Z[j]){
	prod *= cos(state.magic_phases[j]);
      }
      if(!row.X[j] && row.Z[j]){
	prod = 0.;
	break;
      }
      if(row.X[j] && row.Z[j]){
	prod *= -sin(state.magic_phases[j]);
      }
    }
    if(phase){
      acc -= prod;
    }else{
      acc += prod;
    }
  }
  
  if(full_mask == 0u){
    return 1.;
  }
  return acc;
}


double State::compute_probability(std::vector<size_t> measured_qubits, std::vector<uint_t> outcomes){

  AGState copied_ag(this->qreg_); //copy constructor TODO check this

  //first reorder things so the first w qubits are measured
  std::vector<size_t> measured_qubits_sorted(measured_qubits);

  std::vector<size_t> qubit_indexes;
  for(size_t i = 0; i < this->qreg_.num_qubits; i++){
    qubit_indexes.push_back(i);
  }  
  std::sort(measured_qubits_sorted.begin(), measured_qubits_sorted.end());
  
  for(size_t i = 0; i < measured_qubits.size(); i++){
    size_t w = measured_qubits_sorted[i];    
    size_t idx1 = 0;
    size_t idx2 = 0;
    for(size_t j = 0; j < qubit_indexes.size(); j++){
      if(qubit_indexes[j] == w){
	idx1 = j;
	break;
      }
    }
    for(size_t j = 0; j < measured_qubits.size(); j++){
      if(measured_qubits[j] == w){
	idx2 = j;
	break;
      }
    }
    
    if(idx1 != idx2){
      std::swap(qubit_indexes[idx1], qubit_indexes[idx2]);
      copied_ag.applySwap(idx1, idx2);
    }
  }

  //from this point on we will assume we're looking for the measurement outcome 0 on all measured qubits
  //so apply X gates to measured qubits where we're looking for outcome 1 to correct this
  //now all the measured qubits are at the start and the magic qubits are at the end
  for(size_t i = 0; i < outcomes.size(); i++){
    if(outcomes[i] == 1){
      copied_ag.applyX(i);
    }
  }

  size_t w = measured_qubits.size();
  size_t t = copied_ag.magic_phases.size();

  //now all the measured qubits are at the start and the magic qubits are at the end

  std::pair<bool, size_t> v_pair = copied_ag.apply_constraints(w, t);
  if(!v_pair.first){
    return 0.;
  }
  size_t v = v_pair.second;

  //at this point we can delete all the non-magic qubits
  for(size_t q = 0; q < t; q++){
    copied_ag.applySwap(q, q+(copied_ag.num_qubits - t));
  }
  
  for(size_t s = 0; s < copied_ag.num_stabilizers; s++){
    copied_ag.table[s].X.resize(t);
    copied_ag.table[s].Z.resize(t);
  }

  copied_ag.num_qubits = t;
  copied_ag.apply_T_constraints();
  copied_ag.delete_identity_magic_qubits();

  //we can make the compute algorithm much faster if all of our non-Clifford gates are in fact T gates (pi/4 rotations)

  bool all_phases_are_T = true;
  for(size_t i = 0; i < copied_ag.num_qubits; i++){
    if(fabs(copied_ag.magic_phases[i] - T_ANGLE) > AG_CHOP_THRESHOLD){
      all_phases_are_T  = false;
      break;
    }
  }

  
  if(copied_ag.num_qubits == 0){
    return powl(2., (double)v - (double)w);
  }
  if(all_phases_are_T){
    return compute_algorithm_all_phases_T(copied_ag) * powl(2., (double)v - w);
  } else {
    return compute_algorithm_arbitrary_phases(copied_ag) * powl(2., (double)v - w);
  }  
}

void State::initialize_qreg(uint_t num_qubits){
  this->qreg_.initialize(num_qubits);
  this->num_code_qubits = num_qubits;
}
void State::initialize_qreg(uint_t num_qubits, const agstate_t &state){
  if(BaseState::qreg_.num_qubits != num_qubits){
    throw std::invalid_argument("CH::State::initialize: initial state does not match qubit number.");
  }
  BaseState::qreg_ = state;  
}

size_t State::required_memory_mb(uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  return 0; //TODO update this!
}


double State::expval_pauli(const reg_t &qubits, const std::string& pauli){
  return 0; //TODO fix this
}

} //close namespace CliffPhaseCompute
} //close namespace AER

#endif
