#ifndef _aer_extended_stabilizer_estimator_ag_hpp
#define _aer_extended_stabilizer_estimator_ag_hpp

#include <vector>
#include <utility>
#include <algorithm>

#include "framework/types.hpp"
#include "framework/operations.hpp"
#include "simulators/stabilizer/pauli.hpp"

namespace AER{
namespace CliffPhaseCompute{

const double T_ANGLE = M_PI/4.;
const double AG_CHOP_THRESHOLD = 1e-10; //if phases are closer to Clifford phases than this we say they are Clifford
const uint_t ONE = 1u;
/*
 * We need a variant of the stabilizer tableau described in 
 * Improved Simulation of Stabilizer Circuits
 * Scott Aaronson, Daniel Gottesman (2004)
 * 
 * We only store the stabilizer part of the tableau because we don't need the destabilizer part
 * We also add some extra subroutines they don't have which are useful for the Estimate and Compute algorithms
 */

class AGState{
public:
  //initialize such that the jth stabilizer is the Pauli Z_j
  AGState(uint_t num_qubits, uint_t num_stabilizers) : num_qubits(num_qubits), num_stabilizers(num_stabilizers) {};
  AGState() : num_qubits(0), num_stabilizers(0) {};
  AGState(uint_t num_qubits) : num_qubits(num_qubits), num_stabilizers(num_qubits) {};

  void initialize();
  void initialize(uint_t num_qubits);
  
  size_t num_qubits;
  size_t num_stabilizers; //we can represent mixed states so we may have fewer stabilizers than qubits

  std::vector<Pauli::Pauli> table;
  std::vector< unsigned char > phases;

  std::vector<double> magic_phases; //each T or non-Clifford Z-rotation gate we add comes with a phase in [0, pi/4)
  virtual ~AGState() = default; //TODO I'm not actually certain the default destructor is correct here

  void Print();

  // CX controlled on a and targetted on b
  void applyCX(size_t a, size_t b);
  // CZ controlled on a and targetted on b
  void applyCZ(size_t a, size_t b);
  void applyH(size_t a);
  void applyS(size_t a);
  void applyX(size_t a);
  void applyY(size_t a);
  void applyZ(size_t a);
  void applySwap(size_t a, size_t b);
  //this adds a new magic qubit initialised in the |0> state and applies the CX gate required for out gadget
  void gadgetized_phase_gate(size_t a, double phase);
  
  // "add" row i onto row h
  // this group operation is equivalent to the multiplication of Pauli matrices
  void rowsum(size_t h, size_t i);

  /* Does not change the table at all
   * updates the "tableau row" given to be row * table_i
   * returns the update to the phase of row (0 or 1)
   */
  bool rowsum2(Pauli::Pauli row, bool phase, size_t i);

  //return the bool you get at index (stabilizer, column) of the matrix obtained by joining the X and Z matrix as two blocks
  //i.e. if column < num_qubits return X[stabilizer][column] otherwise return Z[stabilizer][column-num_qubits]
  bool tableau_element(size_t stabilizer, size_t column);
  
  std::pair<bool,size_t> first_non_zero_in_col(size_t col, size_t startpoint);
  std::pair<bool,size_t> first_non_zero_in_row(size_t col, size_t startpoint);

  void swap_rows(size_t i, size_t j);
  void delete_last_row();
/*
 * create a QCircuit that brings the state represented by this table to the state |0><0|^k \otimes I^(n-k) / (2^(n-k))
 * Explicitly num_stabilizers stabilisers on num_qubits qubits with the jth stabilzier being a +z on the jth qubit
 * Note that in addition to returning the circuit that does the simplification this method also brings the state into the simplified form
 * if you need the state after calling this method then either use the circuit to rebuild it, or make a copy
 */ 
  std::vector<Operations::Op> simplifying_unitary();

  /*
   * attempt to find a Pauli X operator acting on qubit q in a stabilizer with an index at least a and at most this->num_stabilizer-2 (we ignore the last stabilizer)
   */  
  std::pair<bool,size_t> find_x_on_q(size_t q, size_t a);
  /*
   * attempt to find a Pauli Y operator acting on qubit q in a stabilizer with an index at least a and at most this->num_stabilizer-2 (we ignore the last stabilizer)
   */  
  std::pair<bool,size_t> find_y_on_q(size_t q, size_t a);
  /*
   * attempt to find a Pauli Z operator acting on qubit q in a stabilizer with an index at least a and at most this->num_stabilizer-2 (we ignore the last stabilizer)
   */  
  std::pair<bool,size_t> find_z_on_q(size_t q, size_t a);

  bool independence_test(int q);

  /*
   * Apply constraints arising from the fact that we measure the first w qubits and project the last t onto T gates
   * In particular we remove stabilisers if qubits [0, w) get killed by taking the expectation value <0| P |0>
   * and we remove stabilisers if qubits in [w, this->num_qubits-t) aren't the identity
   * we do not remove any qubits from the table
   */
  std::pair<bool, size_t> apply_constraints(size_t w, size_t t);
  size_t apply_T_constraints();
  /*
   * Go through our stabilizer table and delete every qubit for which every stabilizer is the identity on that qubit
   * In other words delete every column from the X and Z matrices if both are 0 for every element in that column
   * intended only for use when we have restricted our table to only have magic qubits
   * also deletes magic_phases elements to reflect deletion of the magic qubits
   */
  void delete_identity_magic_qubits();
  
};

// Implementation
void AGState::initialize() 
{
  this->table = std::vector<Pauli::Pauli>();

  for(size_t i = 0; i < this->num_stabilizers; i++){
    this->table.push_back(Pauli::Pauli(num_qubits));
    this->table[i].Z.set1(i);
    this->phases.push_back(0);
  }
}

// Implementation
void AGState::initialize(uint_t num_qubits)
{
  this->num_qubits = num_qubits;
  this->num_stabilizers = num_qubits;
  this->initialize();
}


void AGState::Print(){
  for(size_t i = 0; i < this->num_stabilizers; i++){
    for(size_t j = 0; j < this->num_qubits; j++){
      std::cout << this->table[i].X[j];
    }
    std::cout  << "|";
    for(size_t j = 0; j < this->num_qubits; j++){
      std::cout << this->table[i].Z[j];
    }
    std::cout << " " << this->phases[i] << std::endl;
  }
}

void AGState::applyCX(size_t a, size_t b){
  for(size_t i = 0; i < this->num_stabilizers; i++){
    this->phases[i] ^= this->table[i].X[a] & this->table[i].Z[b] & (this->table[i].X[b] ^ this->table[i].Z[a] ^ true);
    this->table[i].X.xorAt(this->table[i].X[a], b);
    this->table[i].Z.xorAt(this->table[i].Z[b], a);
  }
}

void AGState::applyCZ(size_t a, size_t b){
  for(size_t i = 0; i < this->num_stabilizers; i++){
    this->phases[i] ^= this->table[i].X[a] & this->table[i].X[b] & (this->table[i].Z[a] ^ this->table[i].Z[b]);      
    this->table[i].Z.xorAt(this->table[i].X[b], a);
    this->table[i].Z.xorAt(this->table[i].X[a], b);
  }
}

void AGState::applySwap(size_t a, size_t b){
  this->applyCX(a,b);
  this->applyCX(b,a);
  this->applyCX(a,b);
  // for(size_t i = 0; i < this->num_stabilizers; i++){
  //   this->table[i].X.xorAt(this->table[i].X[a], b);
  //   this->table[i].X.xorAt(this->table[i].X[b], a);
  //   this->table[i].X.xorAt(this->table[i].X[a], b);
    
  //   this->table[i].Z.xorAt(this->table[i].Z[b], a);
  //   this->table[i].Z.xorAt(this->table[i].Z[a], b);
  //   this->table[i].Z.xorAt(this->table[i].Z[b], a);
  // }
}

void AGState::applyH(size_t a){
  bool scratch;  
  for(size_t i = 0; i < this->num_stabilizers; i++){
    this->phases[i] ^= this->table[i].X[a] & this->table[i].Z[a];
    scratch = this->table[i].X[a];
    this->table[i].X.setValue(this->table[i].Z[a], a);
    this->table[i].Z.setValue(scratch, a);   
  }
}

void AGState::applyS(size_t a){
  for(size_t i = 0; i < this->num_stabilizers; i++){
    this->phases[i] ^= this->table[i].X[a] & this->table[i].Z[a];
    this->table[i].Z.xorAt(this->table[i].X[a], a);
  }
}

void AGState::applyX(size_t a){
  //TODO write a proper X implementation
  this->applyH(a);
  this->applyS(a);
  this->applyS(a);
  this->applyH(a);
}

void AGState::applyY(size_t a){
  //TODO write a proper Y implementation
  this->applyS(a);
  this->applyX(a);
  this->applyS(a);
  this->applyS(a);
  this->applyS(a);  
}

void AGState::applyZ(size_t a){
  //TODO write a proper Z implementation
  this->applyS(a);
  this->applyS(a);
}

void AGState::gadgetized_phase_gate(size_t a, double phase){
  for(size_t i = 0; i < this->num_stabilizers; i++){
    this->table[i].X.resize(this->num_qubits + 1);
    this->table[i].Z.resize(this->num_qubits + 1);
  }
  this->table.push_back(Pauli::Pauli(this->num_qubits + 1));
  this->table[this->num_stabilizers].Z.set1(this->num_qubits);
  this->phases.push_back(0);
  this->num_stabilizers += 1;
  this->num_qubits += 1;

  

  phase = fmod(phase , M_PI*2);
  if(phase < 0){
    phase += M_PI*2;
  }

  //now phase is in [0, 2*pi)
  while(phase > M_PI/2){
    phase -= M_PI/2;
    this->applyS(a);      
  }
  //now phase is in [0, M_PI/2.]

  if(fabs(phase) < AG_CHOP_THRESHOLD){
    //phase on gate is zero so it is an identity
  }else if(fabs(phase - M_PI/2.) < AG_CHOP_THRESHOLD){
    //phase on gate is pi/2 so it is an S
    this->applyS(a);
    this->applyCX(a, this->num_stabilizers-1);
  }else{
    //its actually non-Clifford
    //we want our phases to be in [0, pi/4]
    if(phase > T_ANGLE){
      phase -= M_PI/2 - phase;
      this->applyX(a);
      this->applyCX(a, this->num_stabilizers-1);
      this->applyX(a);
    }else{
      this->applyCX(a, this->num_stabilizers-1);
    }
  }  
  this->magic_phases.push_back(phase);  
}

void AGState::rowsum(size_t h, size_t i){
  unsigned char sum = 2*(this->phases[h] + this->phases[i]) + Pauli::Pauli::phase_exponent(this->table[i], this->table[h]);
  
  sum %= 4u;
  
  if(sum == 0){
    this->phases[h] = false;
  }
  if(sum == 2){
    this->phases[h] = true;
  }
  
  this->table[h].X += this->table[i].X;
  this->table[h].Z += this->table[i].Z;
    
}
bool AGState::rowsum2(Pauli::Pauli row, bool phase, size_t i){
  unsigned char sum = 2*(phase + this->phases[i])  + Pauli::Pauli::phase_exponent(this->table[i], row);
  sum %= 4u;
  row.X += this->table[i].X;
  row.Z += this->table[i].Z;

  if(sum == 0){
    return false;
  }
  if(sum == 2){
    return true;
  }
  
  //we should never reach here - maybe printing a warning or throwing an exception would be sensible
  return false;
}


void AGState::delete_identity_magic_qubits(){
  //indended for use when we've already restricted our table to only contain the magic qubits
  size_t qubits_deleted = 0;
  
  for(size_t q = 0; q < this->num_qubits; q++){
    size_t non_identity_paulis = 0;
    for(size_t s = 0; (s < this->num_stabilizers) && (non_identity_paulis == 0); s++){
      if(this->table[s].X[q] || this->table[s].Z[q]){
        non_identity_paulis += 1;
      }
    }
    if(non_identity_paulis == 0){
      //every stabiliser is identity on this qubit
      //so we can just delete this qubit
      qubits_deleted += 1;
    }else{
      if(qubits_deleted > 0){
        for(size_t s = 0; s < this->num_stabilizers; s++){
          this->table[s].X.setValue(this->table[s].X[q], q-qubits_deleted);
          this->table[s].Z.setValue(this->table[s].Z[q], q-qubits_deleted);
        }
        magic_phases[q - qubits_deleted] = magic_phases[q];
      }
    }
  }
  this->num_qubits = this->num_qubits - qubits_deleted;

  for(size_t s = 0; s < this->num_stabilizers; s++){
    this->table[s].X.resize(this->num_qubits);
    this->table[s].Z.resize(this->num_qubits);
  }
  magic_phases.resize(this->num_qubits);
}

bool AGState::tableau_element(size_t stabilizer, size_t column){
  if(column < this->num_qubits){
    return this->table[stabilizer].X[column];
  }else{
    return this->table[stabilizer].Z[column-num_qubits];
  }
}

std::pair<bool, size_t> AGState::first_non_zero_in_col(size_t col, size_t startpoint){
  for(size_t i = startpoint; i < this->num_stabilizers; i++){
    if(this->tableau_element(i, col)){
      return std::pair<bool, size_t>(true, i);
    }
  }
  return std::pair<bool, size_t>(false, 0);
}
std::pair<bool,size_t> AGState::first_non_zero_in_row(size_t row, size_t startpoint){
  for(size_t i = startpoint; i < 2*this->num_qubits; i++){
    if(this->tableau_element(row, i) != 0){
      return std::pair<bool, size_t>(true, i);
    }
  }
  
  return std::pair<bool, size_t>(false, 0);
}

void AGState::swap_rows(size_t i, size_t j){
  
  //this->table[i].X.swap(this->table[j].X);
  //this->table[i].Z.swap(this->table[j].Z);
  if(i != j){
    std::swap(this->table[i], this->table[j]);
    std::swap(this->phases[i], this->phases[j]);
  }
}

void AGState::delete_last_row(){
  this->num_stabilizers -= 1;
  this->table.resize(this->num_stabilizers);
  this->phases.resize(this->num_stabilizers);
}

Operations::Op make_H(uint_t a){
  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "h";
  op.qubits = {a};
  return op;
}

Operations::Op make_S(uint_t a){
  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "s";
  op.qubits = {a};
  return op;
}

Operations::Op make_CX(uint_t a, uint_t b){
  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "cx";
  op.qubits = {a, b};
  return op;
}

Operations::Op make_CZ(uint_t a, uint_t b){
  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "cz";
  op.qubits = {a, b};
  return op;
}


std::vector<Operations::Op> AGState::simplifying_unitary(){
  std::vector<Operations::Op> circuit;
  //first we do "augmented gaussian elimination" on the circuit
  //augmented here means that if we don't find a pivot in our column
  //we will hadamard that qubit and try again
  //in other words bring in a pivot from the z part if necessary
  
  size_t h = 0;
  size_t k = 0;
  while(h < this->num_stabilizers && k < this->num_qubits){
    std::pair<bool, size_t> poss_pivot = this->first_non_zero_in_col(k, h);
    if(!poss_pivot.first){
      this->applyH(k);
      circuit.push_back(make_H(k));
      poss_pivot = this->first_non_zero_in_col(k, h);
    }
    if(!poss_pivot.first){
      k += 1;
    }else{
      size_t pivot = poss_pivot.second; //now known to exist
      if(pivot != h){
        //swap rows h and pivot of the table
        this->swap_rows(h,pivot);
      }
      for(size_t j = 0; j < this->num_stabilizers; j++){
        if((j != h) && this->table[j].X[k]){
          this->rowsum(j,h);
        }
      }
      h += 1;
      k += 1;
    }
  }
  
  //so now we have a reduced row echelon form with the X part of the table having full rank
  
  //we swap columns (using CX) to make the X part into a kxk identity followed by a "junk" block
  
  for(int r = 0; r < this->num_stabilizers; r++){
    if(!this->table[r].X[r]){
      size_t col = this->first_non_zero_in_row(r, 0).second;
      
      this->applyCX(r, col);
      circuit.push_back(make_CX(r,col));
      this->applyCX(col, r);
      circuit.push_back(make_CX(col,r));
      this->applyCX(r, col);
      circuit.push_back(make_CX(r,col));
    }
  }


  //now we use CX to clear out the "junk" block at the end of the X table
  for(size_t r = 0; r < this->num_stabilizers; r++){
    for(size_t col = this->num_stabilizers; col < this->num_qubits; col++){
      if(this->table[r].X[col]){
        this->applyCX(r, col);
        circuit.push_back(make_CX(r,col));
      }
    }
  }
  
  //now we clear the leading diagonal of the z block
  for(size_t r = 0; r < this->num_stabilizers; r++){
    if(this->table[r].Z[r]){
      this->applyS(r);
      circuit.push_back(make_S(r));
    }
  }

  //clear out the last k x (n-k) block of the z matrix
  for(size_t col = this->num_stabilizers; col < this->num_qubits; col++){
    for(size_t r = 0; r < this->num_stabilizers; r++){
      if(this->table[r].Z[col]){
        this->applyCZ(r, col);
        circuit.push_back(make_CZ(r, col));
      }
    }
  }
  
  //clear out the first k x k block of the Z matrix, using that it is symmetric
  for(size_t col = 0; col < this->num_stabilizers; col++){
    for(size_t r = 0; r < col; r++){
      if(this->table[r].Z[col]){
        this->applyCZ(r, col);
        circuit.push_back(make_CZ(r, col));
      }
    }
  }

  //fix the phases
  for(size_t r = 0; r < this->num_stabilizers; r++){
    if(this->phases[r]){
      this->applyS(r);
      circuit.push_back(make_S(r));
      this->applyS(r);
      circuit.push_back(make_S(r));
    }
  }

  //swap the identity matrix to the z part
  for(size_t r = 0; r < this->num_stabilizers; r++){
    this->applyH(r);
    circuit.push_back(make_H(r));
  }
  
  return circuit;
}

std::pair<bool, size_t> AGState::find_x_on_q(size_t q, size_t a){
  for(size_t row = a; row < this->num_stabilizers-1; row++){
    if(this->table[row].X[q] && !this->table[row].Z[q]){
      return std::pair<bool, size_t>(true, row);
    }
  }
  return std::pair<bool, size_t>(false, 0);
}

std::pair<bool, size_t> AGState::find_y_on_q(size_t q, size_t a){
  for(size_t row = a; row < this->num_stabilizers-1; row++){
    if(this->table[row].X[q] && this->table[row].Z[q]){
      return std::pair<bool, size_t>(true, row);
    }
  }
  return std::pair<bool, size_t>(false, 0);
}

std::pair<bool, size_t> AGState::find_z_on_q(size_t q, size_t a){
  for(size_t row = a; row < this->num_stabilizers-1; row++){
    if(!this->table[row].X[q] && this->table[row].Z[q]){
      return std::pair<bool, size_t>(true, row);
    }
  }
  return std::pair<bool, size_t>(false, 0);
}

/*
 * we test that if you ignore the first q+1 qubits whether that last (n-q-1) part of the last stabiliser in the table can be generated (up to phase)
 * by the other stabilisers
 * for use in the apply_constraints code
 */
bool AGState::independence_test(int q){
  //we basically do gaussian elimination
  
  size_t a = 0;
  size_t b = q+1;
  while(a < this->num_stabilizers-1 && b < this->num_qubits){
    std::pair<bool,size_t> x = this->find_x_on_q(b, a);
    std::pair<bool,size_t> y = this->find_y_on_q(b, a);
    std::pair<bool,size_t> z = this->find_z_on_q(b, a);
    
    if(y.first && x.first){
      this->rowsum(y.second, x.second);
      z = y;
      y = std::pair<bool,size_t>(false,0);
    }
    if(y.first && z.first){
      this->rowsum(y.second, z.second);
      x = y;
      y = std::pair<bool,size_t>(false,0);
    }
    
    if(x.first){
      if(x.second != a){
        this->swap_rows(a, x.second);
      }
      if(z.first && (z.second == a)){
        z = x;
      }
      for(size_t j = 0; j < this->num_stabilizers; j++){
        if((j != a) && this->table[j].X[b]){
          this->rowsum(j,a);
        }
      }
      a += 1;
    }
    if(y.first){
      if(y.second != a){
        this->swap_rows(a,y.second);
      }
      for(size_t j = 0; j < this->num_stabilizers; j++){
        if((j != a) && this->table[j].X[b] && this->table[j].Z[b]){
          this->rowsum(j,a);
        }
      }
      a += 1;
    }
    if(z.first){
      if(z.second != a){
        this->swap_rows(a,z.second);
      }
      for(size_t j = 0; j < this->num_stabilizers; j++){
        if((j != a) && this->table[j].Z[b]){
          this->rowsum(j,a);
        }
      }
      a += 1;
    }
    b += 1;
  }
  
  for(size_t p = q+1; p < this->num_qubits; p++){
    if((this->table[this->num_stabilizers-1].X[p] == 1) || (this->table[this->num_stabilizers-1].Z[p] == 1)){
      return true;
    }
  } 
  return false;
}


/*
 * Apply constraints arising from the fact that we measure the first w qubits and project the last t onto T gates
 * In particular we kill stabilisers if qubits [0, w) get killed by taking the expectation value <0| P |0>
 * and we kill stabilisers if qubits in [w, table->n-t) aren't the identity
 * we do not remove any qubits from the table
 * note that it is possible for the region a constraints to be "inconsistent" - this means that the probability we were calculating is zero
 * in that case we return pair(false, 0)
 * in other cases we return pair(true, v) where v is as defined in the paper
 */
std::pair<bool, size_t> AGState::apply_constraints(size_t w, size_t t){
  size_t v = 0;
  
  //first apply region a constraints (measurement)
  for(size_t q=0; q < w; q++){ //iterate over all the measured qubits
    std::pair<bool,size_t> y_stab = std::pair<bool,size_t>(false, 0); //store the index of the first stab we come to with both x and z = 1 on this qubit
    std::pair<bool,size_t> x_stab = std::pair<bool,size_t>(false, 0); //store the index of the first stab we come to with x=1, z=0
    std::pair<bool,size_t> z_stab = std::pair<bool,size_t>(false, 0); //store the index of the first stab we come to with z=1, x=0
    
    for(size_t s=0; s < this->num_stabilizers && (((!y_stab.first) + (!x_stab.first) + (!z_stab.first)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
      
      if(this->table[s].X[q] && this->table[s].Z[q]){
        y_stab.first = true;
        y_stab.second = s;
      }
      if(this->table[s].X[q] && !this->table[s].Z[q]){
        x_stab.first = true;
        x_stab.second = s;
      }
      if(!this->table[s].X[q] && this->table[s].Z[q]){
        z_stab.first = true;
        z_stab.second = s;
      }
    }
    //there are several cases here
    //either a single z, a single x, a single y or we can generate the whole Pauli group on this qubit
    
    //case 1) we generate the whole group
    //put things in standard form (first stab is x then z)    
    if((y_stab.first + x_stab.first + z_stab.first) >= 2){ //we have at least two of the set
      if(!x_stab.first){//we don't have a generator for x alone, but we can make one
        this->rowsum(y_stab.second, z_stab.second);
        //now we have a z and an x but not a y
        x_stab = y_stab;
        y_stab = std::pair<bool, size_t>(false,0);
      }else if(!z_stab.first){//we don't have a generator for z alone, but we can make one
        this->rowsum(y_stab.second, x_stab.second);
        //now we have a z and an x but not a y
        z_stab = y_stab;
        y_stab = std::pair<bool, size_t>(false,0);
      }
    }
    
    if(y_stab.first && x_stab.first && z_stab.first){ //we have all 3
      //ignore the y one if we have all 3
      y_stab = std::pair<bool, size_t>(false,0);
    }

    //now the only possibilities are that we have an an x, y or z or both an x and a z
    //zero everything else on this qubit
    for(size_t s = 0; s < this->num_stabilizers; s++){
      if((!y_stab.first || s != y_stab.second) && (!x_stab.first || s != x_stab.second) && (!z_stab.first || s != z_stab.second)){
        if(this->table[s].X[q] && this->table[s].Z[q] && y_stab.first){
          this->rowsum(s, y_stab.second);
        }

        if(this->table[s].X[q]){
          this->rowsum(s, x_stab.second);
        }

        if(this->table[s].Z[q]){
          this->rowsum(s, z_stab.second);
        }
      }
    }

    //case 1 - there is a generator which does not commute with Z_q
    if(y_stab.first || x_stab.first){
      //we can't have both >= 0
      size_t non_commuting_generator = y_stab.first ? y_stab.second : x_stab.second;
      //we delete the non-commuting guy
      this->swap_rows(non_commuting_generator, this->num_stabilizers-1);
      this->delete_last_row();
    }else{
      //case 2 - all generators commute with Z_q
      //our generating set contains either Z_q or -Z_q
      //we need to work out which one it is
      
      //swap our Z_q guy to the end
      this->swap_rows(z_stab.second, this->num_stabilizers-1);
      bool independent = this->independence_test(q);
      
      if(!independent){
        if(this->phases[this->num_stabilizers-1] == 0){
          // +Z_q
          v += 1;
          this->delete_last_row();
        }else{
          //our chosen measurement outcome is impossible
          return std::pair<bool, size_t>(false,0);
        }
      }else{
        //if we get here there has been an error
        //TODO decide if we're going to throw an exception or print an error message here
      }
    }
  }
  
  //time to impose region b constraints  
  for(size_t q=w; q < this->num_qubits - t ; q++){ //iterate over all the non-measured non-magic qubits
    std::pair<bool, size_t> y_stab = std::pair<bool, size_t>(false,0); //store the index of the first stab we come to with both x and z = 1 on this qubit
    std::pair<bool, size_t> x_stab = std::pair<bool, size_t>(false,0); //store the index of the first stab we come to with x=1, z=0
    std::pair<bool, size_t> z_stab = std::pair<bool, size_t>(false,0); //store the index of the first stab we come to with z=1, x=0

    for(size_t s=0; s < this->num_stabilizers && (((!y_stab.first) + (!x_stab.first) + (!z_stab.first)) > 1); s++){//iterate over all stabilisers and find interesting stabilisers
      if(this->table[s].X[q] && this->table[s].Z[q]){
        y_stab.first = true;
        y_stab.second = s;
      }
      if(this->table[s].X[q] && !this->table[s].Z[q]){
        x_stab.first = true;
        x_stab.second = s;
      }
      if(!this->table[s].X[q] && this->table[s].Z[q]){
        z_stab.first = true;
        z_stab.second = s;
      }
    }

    //there are several cases here
    //either a single z, a single x, a single y or we can generate the whole Pauli group on this qubit
    
    //case 1) we generate the whole group
    //put things in standard form (first stab is x then z)    
    if((y_stab.first + x_stab.first + z_stab.first) >= 2){ //we have at least two of the set
      if(!x_stab.first){//we don't have a generator for x alone, but we can make one
        this->rowsum(y_stab.second, z_stab.second);
        //now we have a z and an x but not a y
        x_stab = y_stab;
        y_stab = std::pair<bool, size_t>(false,0);
      }else if(!z_stab.first){//we don't have a generator for z alone, but we can make one
        this->rowsum(y_stab.second, x_stab.second);
        //now we have a z and an x but not a y
        z_stab = y_stab;
        y_stab = std::pair<bool, size_t>(false,0);
      }
    }
    
    if(y_stab.first && x_stab.first && z_stab.first){ //we have all 3
      //ignore the y one if we have all 3
      y_stab = std::pair<bool, size_t>(false,0);
    }
    
    //now the only possibilities are that we have an x_and_z, an x a z or an x and a z
    //zero everything else on this qubit
    for(size_t s = 0; s < this->num_stabilizers; s++){
      if((!y_stab.first || s != y_stab.second) && (!x_stab.first || s != x_stab.second) && (!z_stab.first || s != z_stab.second)){
        if(this->table[s].X[q] && this->table[s].Z[q] && y_stab.first){
          this->rowsum(s, y_stab.second);
        }

        if(this->table[s].X[q]){
          this->rowsum(s, x_stab.second);
        }

        if(this->table[s].Z[q]){
          this->rowsum(s, z_stab.second);
        }
      }
    }
    
    //now we just delete the non-identity guys on this qubit    
    int num_to_delete = 0;
    if(y_stab.first){
      //if we have a Y stab we don't have either of the others
      this->swap_rows(y_stab.second, this->num_stabilizers-1);
      num_to_delete += 1;
    }else{
      if(x_stab.first){
        this->swap_rows(x_stab.second, this->num_stabilizers-1);
        if(z_stab.first && (this->num_stabilizers - 1 == z_stab.second)){
          z_stab = x_stab;
        }
        num_to_delete += 1;
      }
      if(z_stab.first){
        this->swap_rows(z_stab.second, this->num_stabilizers-1-num_to_delete);
        num_to_delete += 1;
      }
    }

    //delete the last num_to_delete rows
    //TODO should we implement an erase method that works like the std::vector one so you can pass a range?
    for(size_t deletes = 0; deletes < num_to_delete; deletes++){
      this->delete_last_row();
    }
  }
  
  return std::pair<bool, size_t>(true, v);
}


/*
 * Our magic states are equatorial
 * so <T|Z|T> = 0
 * here we delete any stabilisers with a Z in the magic region
 * which we assume now all the qubits
 * we return the number of qubits which have identities on them in every generator after this deletion
 */
size_t AGState::apply_T_constraints(){
  size_t starting_rows = this->num_stabilizers;
  size_t deleted_rows = 0;
  for(size_t reps = 0; reps < starting_rows; reps++){
    for(size_t q=0; q < this->num_qubits; q++){ //iterate over all the magic qubits
      std::pair<bool, size_t> y_stab = std::pair<bool, size_t>(false,0); //store the index of the first stab we come to with both x and z = 1 on this qubit
      std::pair<bool, size_t> x_stab = std::pair<bool, size_t>(false,0); //store the index of the first stab we come to with x=1, z=0
      std::pair<bool, size_t> z_stab = std::pair<bool, size_t>(false,0); //store the index of the first stab we come to with z=1, x=0
      
      for(size_t s=0; s < this->num_stabilizers && (((!y_stab.first) + (!x_stab.first) + (!z_stab.first)) > 1); s++){//iterate over all stabilisers and find interesting stabi        lisers
        if(this->table[s].X[q] && this->table[s].Z[q]){
          y_stab.first = true;
          y_stab.second = s;
      }
        if(this->table[s].X[q] && !this->table[s].Z[q]){
          x_stab.first = true;
          x_stab.second = s;
        }
        if(!this->table[s].X[q] && this->table[s].Z[q]){
          z_stab.first = true;
          z_stab.second = s;
        }
      }
      
      //there are several cases here
      //either a single z, a single x, a single y or we can generate the whole Pauli group on this qubit
      
      //case 1) we generate the whole group
      //put things in standard form (first stab is x then z)    
      if((y_stab.first + x_stab.first + z_stab.first) >= 2){ //we have at least two of the set
        if(!x_stab.first){//we don't have a generator for x alone, but we can make one
          this->rowsum(y_stab.second, z_stab.second);
          //now we have a z and an x but not a y
          x_stab = y_stab;
          y_stab = std::pair<bool, size_t>(false,0);
        }else if(!z_stab.first){//we don't have a generator for z alone, but we can make one
          this->rowsum(y_stab.second, x_stab.second);
          //now we have a z and an x but not a y
          z_stab = y_stab;
          y_stab = std::pair<bool, size_t>(false,0);
        }
      }
      
      if(y_stab.first && x_stab.first && z_stab.first){ //we have all 3
        //ignore the y one if we have all 3
        y_stab = std::pair<bool, size_t>(false,0);
      }
      
      if(z_stab.first && !x_stab.first){
        //kill all other z stuff on this qubit
        for(size_t s = 0; s < this->num_stabilizers; s++){
          if((s != z_stab.second) && this->table[s].Z[q]){
            this->rowsum(s, z_stab.second);
          }
        }
        //now delete the z guy

        if(z_stab.second != this->num_stabilizers-1){
          this->swap_rows(z_stab.second, this->num_stabilizers-1);
        }
        this->delete_last_row();
        deleted_rows += 1;
      }
    }
  }
  return deleted_rows;
}

}
}
#endif
