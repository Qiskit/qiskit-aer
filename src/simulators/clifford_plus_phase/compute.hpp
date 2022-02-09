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

  //Apply a sequence of operations to the cicuit
  template <typename InputIterator>
  void apply_ops(InputIterator first, InputIterator last,
		 ExperimentResult &result,
		 RngEngine &rng,
		 bool final_ops = false);
  
  virtual void apply_op(const Operations::Op &op,
		ExperimentResult &result,
		RngEngine &rng,
		bool final_op = false) override;
  
  void apply_gate(const Operations::Op &op);

  void initialize_qreg(uint_t num_qubits) override;
  void initialize_qreg(uint_t num_qubits, const agstate_t &state) override;
  size_t required_memory_mb(uint_t num_qubits, const std::vector<Operations::Op> &ops) const override;
  void apply_save_specific_prob(const Operations::Op &op, ExperimentResult &result);
  double expval_pauli(const reg_t &qubits, const std::string& pauli);
private:  
  const static stringmap_t<Gates> gateset_;
  double compute_algorithm_all_phases_T(AGState &state);
  double compute_algorithm_arbitrary_phases(AGState &state);
  size_t num_code_qubits; //our AG state has code+magic qubits
  double compute_probability(std::vector<uint_t> measured_qubits, std::vector<uint_t> outcomes);
  template <typename InputIterator>
  uint_t count_magic_gates(InputIterator first, InputIterator last) const;
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

template <typename InputIterator>
void State::apply_ops(InputIterator first, InputIterator last, ExperimentResult &result, RngEngine &rng, bool final_ops){
  //std::cout << "applying gates" << std::endl;
  for(auto it = first; it != last; ++it){
    apply_op(*it, result, rng, final_ops);
  }
}

void State::apply_op(const Operations::Op &op, ExperimentResult &result,
                     RngEngine &rng, bool final_op) {
  //std::cout << "applying op ";
  switch(op.type){
  case Operations::OpType::gate:
    //std::cout << "gate" << std::endl;
    //std::cout << op << std::endl;
    this->apply_gate(op);
    break;
  case Operations::OpType::save_specific_prob:
    //std::cout << "save" << std::endl;
    this->apply_save_specific_prob(op, result);
    break;
  default:
    throw std::invalid_argument("Compute::State::invalid instruction \'" + op.name + "\'.");  
  }
}

void State::apply_gate(const Operations::Op &op){
  //std::cout << "apply_gate" << std::endl;
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
  {
    throw std::invalid_argument("Compute::State: Invalid gate operation \'"
                                +op.name + "\'.");
  }
  switch(it->second){
  case Gates::id:
    //std::cout << "id" << std::endl;
    break;
  case Gates::x:
    //std::cout << "x" << std::endl;
    this->qreg_.applyX(op.qubits[0]);
    break;
  case Gates::y:
    //std::cout << "y" << std::endl;
    this->qreg_.applyY(op.qubits[0]);
    break;
  case Gates::z:
    //std::cout << "z" << std::endl;
    this->qreg_.applyZ(op.qubits[0]);
    break;
  case Gates::s:
    //std::cout << "s" << std::endl;
    this->qreg_.applyS(op.qubits[0]);
    break;
  case Gates::sdg:
    //std::cout << "sdg" << std::endl;
    this->qreg_.applyZ(op.qubits[0]);
    this->qreg_.applyS(op.qubits[0]);
    break;
  case Gates::h:
    //std::cout << "h" << std::endl;
    this->qreg_.applyH(op.qubits[0]);
    break;
  case Gates::t:
    //std::cout << "t" << std::endl;
    this->qreg_.gadgetized_phase_gate(op.qubits[0], T_ANGLE);
    break;
  case Gates::tdg:
    //std::cout << "tdg" << std::endl;
    this->qreg_.gadgetized_phase_gate(op.qubits[0], -T_ANGLE);
    break;
  case Gates::rz:
    //std::cout << "rz" << std::endl;
    this->qreg_.gadgetized_phase_gate(op.qubits[0], op.params[0].real());
    break;
  case Gates::cx:
    //std::cout << "cx[" << op.qubits[0] << ", " << op.qubits[1] << "]" <<std::endl;
    this->qreg_.applyCX(op.qubits[0],op.qubits[1]);
    break;
  case Gates::cz:
    //std::cout << "cz" << std::endl;
    this->qreg_.applyCZ(op.qubits[0],op.qubits[1]);
    break;
  case Gates::swap:
    //std::cout << "swap" << std::endl;
    this->qreg_.applySwap(op.qubits[0],op.qubits[1]);
    break;
  default: //u0 or Identity
    break;
  }
}

void State::apply_save_specific_prob(const Operations::Op &op, ExperimentResult &result){
  double p = this->compute_probability(op.qubits, op.int_params);
  save_data_average(result, op.string_params[0], p, op.type, op.save_type);
}

// This function converts an unsigned binary number to reflected binary Gray code.
uint_t BinaryToGray(uint_t num)
{
  return num ^ (num >> 1); // The operator >> is shift right. The operator ^ is exclusive or.
}


double State::compute_algorithm_all_phases_T(AGState &state){
  //std::cout << "State::compute_algorithm_all_phases_T" << std::endl;
  if(state.num_stabilizers > 63){
    //we use an int_t == int_fast64_t variable to store our loop counter, this means we can deal with at most 63 stabilizers
    //in the case of 63 stabilizers we will iterate over all length 63 bitstrings
    //this is the most we can store in a signed 64 bit integer
    //the integer has to be signed due to openMP's requirements
    //realistically a computation with 63 stabilizers = 2^63 - 1 iterations is not going to terminate anyway so this restriction shouldn't matter
    std::stringstream msg;
    msg << "CliffPhaseCompute::State::compute_algorithm_all_phases_T called with " << state.num_stabilizers << " stabilizers. Maximum possible is 63."; 
    throw std::runtime_error(msg.str());
  }
  
  uint_t full_mask = 0;
  for(size_t i = 0; i < state.num_stabilizers; i++){
    full_mask |= (ONE << i);
  }

  double acc = 0.; 
  Pauli::Pauli row(state.num_qubits);
  unsigned char phase = 0;
  
  bool row_initialised = false;  
  uint_t num_threads_ = BaseState::threads_;

  int_t chunk_size = 0;
  #pragma omp parallel for num_threads(num_threads_)
  for (int_t thread = 0; thread < num_threads_; ++thread) {
  // identify a chunk from i and num_threads
  }
  
  #pragma omp parallel for if(num_threads_ > 1) \
    num_threads(num_threads_) firstprivate(row_initialised, row, phase) shared(state) reduction(+:acc) 
  for(uint_t mask = 0; mask <= full_mask; mask++){    
    if(row_initialised){
      uint_t mask_with_bit_to_flip = BinaryToGray((uint_t)mask) ^ BinaryToGray((uint_t)(mask - 1));
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
    }else{
      int_t mask_with_bit_to_flip = BinaryToGray(mask); // note BinaryToGray(0u) == 0u
      for(size_t j = 0; j < state.num_stabilizers; j++){
	if((mask_with_bit_to_flip >> j) & ONE){
	  phase += state.phases[j] + (Pauli::Pauli::phase_exponent(row, state.table[j]) / 2); //phases for stabilizers are always 1 or -1
	  phase %= 2;
	  row += state.table[j];
	}	
      }
      row_initialised = true;
    }
    size_t XCount = 0;
    size_t YCount = 0;
    size_t ZCount = 0;
    
    for(size_t j = 0; j < state.num_qubits; j++){
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

double State::compute_algorithm_arbitrary_phases(AGState &state){

  if(state.num_stabilizers > 63){
    //we use an int_t == int_fast64_t variable to store our loop counter, this means we can deal with at most 63 stabilizers
    //in the case of 63 stabilizers we will iterate over all length 63 bitstrings
    //this is the most we can store in a signed 64 bit integer
    //the integer has to be signed due to openMP's requirements
    //realistically a computation with 63 stabilizers = 2^63 - 1 iterations is not going to terminate anyway so this restriction shouldn't matter
    std::stringstream msg;
    msg << "CliffPhaseCompute::State::compute_algorithm_arbitrary_phases called with " << state.num_stabilizers << " stabilizers. Maximum possible is 63."; 
    throw std::runtime_error(msg.str());
  }

  
  //std::cout << "1" << std::endl;
  uint_t full_mask = 0u;
  for(size_t i = 0; i < state.num_stabilizers; i++){
    full_mask |= (ONE << i);
  }
  double acc = 1.;
  Pauli::Pauli row(state.num_qubits);
  unsigned char phase = 0;
  //std::cout << "2" << std::endl;
  for(uint_t mask = 1u; mask <= full_mask; mask++){
    uint_t mask_with_bit_to_flip = BinaryToGray((uint_t)mask) ^ BinaryToGray((uint_t)(mask - 1));
    size_t bit_to_flip = 0;
    for(size_t j = 0; j < state.num_stabilizers; j++){
      if((mask_with_bit_to_flip >> j) & ONE){
        bit_to_flip = j;
        break;
      }
    }
    
    phase += Pauli::Pauli::phase_exponent(row, state.table[bit_to_flip]) / 2; //phases for stabilizers are always 0 or 2
    phase += state.phases[bit_to_flip];
    phase %= 2;
    row += state.table[bit_to_flip];
    
    double prod = (phase==1) ? -1. : 1.;
    for(size_t j = 0; j < state.num_qubits; j++){
      if(row.X[j] && !row.Z[j]){
        prod *= cos(state.magic_phases[j]);
      }
      if(!row.X[j] && row.Z[j]){
        prod = 0.;
        break;
      }
      if(row.X[j] && row.Z[j]){
        prod *= (-sin(state.magic_phases[j]));
      }
    }
    acc += prod;
  }
  //std::cout << "3" << std::endl;
  if(full_mask == 0u){
    acc=1;
  }
  return acc;
}


double State::compute_probability(std::vector<uint_t> measured_qubits, std::vector<uint_t> outcomes){
  //std::cout << "State::compute_probability" << std::endl;
  AGState copied_ag(this->qreg_); //copy constructor TODO check this

  //first reorder things so the first w qubits are measured
  std::vector<uint_t> measured_qubits_sorted(measured_qubits);
  
  std::vector<uint_t> qubit_indexes;
  for(uint_t i = 0; i < this->qreg_.num_qubits; i++){
    qubit_indexes.push_back(i);
  }  
  std::sort(measured_qubits_sorted.begin(), measured_qubits_sorted.end());
  for(uint_t i = 0; i < measured_qubits.size(); i++){
    uint_t w = measured_qubits_sorted[i];
    uint_t idx1 = 0;
    uint_t idx2 = 0;
    for(uint_t j = 0; j < qubit_indexes.size(); j++){
      if(qubit_indexes[j] == w){
        idx1 = j;
        break;
      }
    }
    for(uint_t j = 0; j < measured_qubits.size(); j++){
      if(measured_qubits[j] == w){
        idx2 = j;
        break;
      }
    }
    
    if(idx1 != idx2){
      //std::cout << "swapping " << idx1 << " and " << idx2 << std::endl;
      //std::swap(qubit_indexes[idx1], qubit_indexes[idx2]);
      copied_ag.applySwap(idx1, idx2);
    }
  }

  //now all the measured qubits are at the start and the magic qubits are at the end
  //from this point on we will assume we're looking for the measurement outcome 0 on all measured qubits
  //so apply X gates to measured qubits where we're looking for outcome 1 to correct this  
  for(uint_t i = 0; i < outcomes.size(); i++){
    //std::cout << "outcomes[" << i << "] = " << outcomes[i] << " ";
    if(outcomes[i] == 1){      
      copied_ag.applyX(i);
    }
  }
  //std::cout << std::endl;

  uint_t w = measured_qubits.size();
  uint_t t = copied_ag.magic_phases.size();

  //now all the measured qubits are at the start and the magic qubits are at the end

  std::pair<bool, uint_t> v_pair = copied_ag.apply_constraints(w, t);

  if(!v_pair.first){
    return 0.;
  }
  uint_t v = v_pair.second;

  //at this point we can delete all the non-magic qubits
  for(uint_t q = 0; q < t; q++){
    copied_ag.applySwap(q, q+(copied_ag.num_qubits - t));
  }
  
  for(uint_t s = 0; s < copied_ag.num_stabilizers; s++){
    copied_ag.table[s].X.resize(t);
    copied_ag.table[s].Z.resize(t);
  }

  copied_ag.num_qubits = t;
  copied_ag.apply_T_constraints();
  copied_ag.delete_identity_magic_qubits();

  //we can make the compute algorithm much faster if all of our non-Clifford gates are in fact T gates (pi/4 rotations)

  bool all_phases_are_T = true;
  for(uint_t i = 0; i < copied_ag.num_qubits; i++){
    if(fabs(copied_ag.magic_phases[i] - T_ANGLE) > AG_CHOP_THRESHOLD){
      all_phases_are_T  = false;
      break;
    }
  }
  //std::cout << copied_ag.magic_phases.size() << std::endl;
  //std::cout << copied_ag.magic_phases[0] << ", " << cos(copied_ag.magic_phases[0]) << ", " << -sin(copied_ag.magic_phases[0]) << std::endl;
  if(copied_ag.num_qubits == 0){
    return pow(2., ((double)v) - ((double)w));
  }
  //std::cout << "a" << std::endl;
  if(all_phases_are_T){
    //std::cout << "a" << std::endl;
    return compute_algorithm_all_phases_T(copied_ag) * pow(2., ((double)v) - ((double)w));
  } else {
    //std::cout << "b" << std::endl;
    return compute_algorithm_arbitrary_phases(copied_ag) * pow(2., ((double)v) - ((double)w));
  }  
}

void State::initialize_qreg(uint_t num_qubits) {
  this->qreg_.initialize(num_qubits);
  this->num_code_qubits = num_qubits;
}
void State::initialize_qreg(uint_t num_qubits, const agstate_t &state) {
  if(BaseState::qreg_.num_qubits != num_qubits){
    throw std::invalid_argument("CH::State::initialize: initial state does not match qubit number.");
  }
  BaseState::qreg_ = state;  
}

template <typename InputIterator>
uint_t State::count_magic_gates(InputIterator first, InputIterator last) const {
  uint_t count = 0;
  for (auto op = first; op != last; op++)
  {
    auto it = gateset_.find(op->name);
    if (it != gateset_.end())
    {
      if(it->second == Gates::t || it->second == Gates::rz || it->second == Gates::tdg){
	count += 1;
      }      
    }
  }
  return count;
}


size_t State::required_memory_mb(uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  uint_t t = count_magic_gates(ops.cbegin(), ops.cend());

  uint_t total_qubits = num_qubits + t;

  // we store a stabilizer tableau with n+t stabilizers on n+t qubits
  // each stabilizer (each "row" in the table) is a Pauli::Pauli on n+t qubits
  // each Pauli:: Pauli stores two BV::BinaryVectors of length n+t and in addition needs one byte of space we use to store the phase
  // each BinaryVector stores 8 * ceil((n+t)/64) bytes
  //note that this is an upper bound, in reality the compress algorithm will give us a state with fewer qubits and stabilizers
  
  //in addition each working thread gets a Pauli (two binary vectors and a phase) of space to do its work in
  uint_t bv_size_bytes = 8*((num_qubits+t)/64 + ((((num_qubits+t) % 64)) ? 1 : 0)); //add an extra 8 byte int that we partially use if there is a remainder
  uint_t pauli_size_bytes = 2*bv_size_bytes + 1;
  
  // State::compute_probability copies the stabilizer tableau before operating on it so double the memory required for that
  size_t mb = (pauli_size_bytes * (2*(num_qubits+t)+BaseState::threads_) + t*sizeof(double))/(1<<20);

  return mb;
}


double State::expval_pauli(const reg_t &qubits, const std::string& pauli) {
  return 0; //TODO fix this
}

} //close namespace CliffPhaseCompute
} //close namespace AER

#endif
