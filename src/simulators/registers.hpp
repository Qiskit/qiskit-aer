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

#ifndef _aer_base_register_hpp_
#define _aer_base_register_hpp_

#include "framework/types.hpp"
#include "framework/creg.hpp"

namespace AER {
namespace QuantumState {

using OpItr = std::vector<Operations::Op>::const_iterator;

class RegistersBase;

//base class of register storage 
class RegistersBase {
protected:
  ClassicalRegister creg_;

  //qubits map for chunk-distribution
  reg_t qubit_map_;

public:
  RegistersBase(){}
  virtual ~RegistersBase()
  {
  }

  // Return the state creg object
  auto &creg() { return creg_; }
  const auto &creg() const { return creg_; }

  virtual uint_t num_qregs() = 0;
  virtual void allocate(const uint_t nqregs = 1) = 0;


  void initialize_qubit_map(const uint_t num_qubits);
  void swap_qubit_map(const uint_t q0,const uint_t q1);
  uint_t get_mapped_index(const uint_t idx);
};

void RegistersBase::initialize_qubit_map(const uint_t num_qubits)
{
  qubit_map_.resize(num_qubits);
  for(int_t i=0;i<num_qubits;i++)
    qubit_map_[i] = i;
}

void RegistersBase::swap_qubit_map(const uint_t q0,const uint_t q1)
{
  std::swap(qubit_map_[q0], qubit_map_[q1]);
}

uint_t RegistersBase::get_mapped_index(const uint_t idx)
{
  uint_t i,ret = 0;
  uint_t t = idx;

  for(i=0;i<qubit_map_.size();i++){
    if(t & 1){
      ret |= (1ull << qubit_map_[i]);
    }
    t >>= 1;
  }
  return ret;
}

//object to store branch of shots
struct ShotBranch {
  //random generators for each shot
  std::vector<RngEngine> shots_;
  //additional opertions 
  std::vector<Operations::Op> additional_ops_;
  //creg to be stored to the state
  ClassicalRegister creg_;
};


//register storage for each state_t object
template <class state_t>
class Registers : public RegistersBase {
protected:
  std::vector<state_t> qregs_;

  //mark for control flow
  std::unordered_map<std::string, OpItr> flow_marks_;

  //iterator for restarting after branch
  OpItr next_iter_;

  //random generators for shots
  std::vector<RngEngine> shots_;
  //additional operations applied after shot branching
  std::vector<Operations::Op> additional_ops_;
  //array of branched shots
  std::vector<ShotBranch> branches_;
public:
  Registers()
  {

  }
  Registers(const Registers<state_t>& src);
  Registers<state_t> &operator=(const Registers<state_t>& src);

  virtual ~Registers()
  {
    qregs_.clear();

    shots_.clear();
    additional_ops_.clear();
    branches_.clear();
  }

  // Return the state qreg object
  auto& qregs(){return qregs_;}
  auto &qreg(const uint_t idx = 0) { return qregs_[idx]; }
  auto &qreg_non_const(const uint_t idx = 0) { return qregs_[idx]; }
  const auto &qreg(const uint_t idx = 0) const { return qregs_[idx]; }

  uint_t num_qregs()  override
  {
    return qregs_.size();
  }

  void allocate(const uint_t nqregs = 1) override
  {
    if(qregs_.size() > 0 && qregs_.size() != nqregs)
      qregs_.clear();
    qregs_.resize(nqregs);
  }

  void copy(const Registers<state_t>& src);

  std::unordered_map<std::string, OpItr>& marks(void)
  {
    return flow_marks_;
  }
  OpItr& next_iter(void)
  {
    return next_iter_;
  }

  uint_t num_shots(void)
  {
    uint_t nshots = shots_.size();
    if(nshots == 0)
      nshots = 1;
    return nshots;
  }
  RngEngine& rng_shots(uint_t ishot)
  {
    return shots_[ishot];
  }
  void set_shots(std::vector<RngEngine>& shots)
  {
    shots_ = shots;
  }
  void initialize_shots(const uint_t nshots, const uint_t seed);

  //functions for shot branching
  int_t num_branch(void)
  {
    return branches_.size();
  }
  ShotBranch& branch(int i)
  {
    return branches_[i];
  }

  //branch shots into nbranch states
  void branch_shots(reg_t& shots, int_t nbranch);
  void clear_branch(void)
  {
    branches_.clear();
  }

  void add_op_after_branch(const uint_t ibranch, Operations::Op& op)
  {
    branches_[ibranch].additional_ops_.push_back(op);
  }
  void copy_ops_after_branch(const uint_t ibranch, std::vector<Operations::Op>& ops)
  {
    branches_[ibranch].additional_ops_ = ops;
  }
  void clear_additional_ops(void)
  {
    additional_ops_.clear();
  }

  std::vector<Operations::Op>& additional_ops(void)
  {
    return additional_ops_;
  }
};

template <class state_t>
Registers<state_t>::Registers(const Registers<state_t>& src)
{
  copy(src);
}
template <class state_t>
Registers<state_t>& Registers<state_t>::operator=(const Registers<state_t>& src)
{
  copy(src);
  return *this;
}

template <class state_t>
void Registers<state_t>::copy(const Registers<state_t>& src)
{
  qregs_.resize(src.qregs_.size());
  for(int_t i=0;i<qregs_.size();i++)
    qregs_[i].initialize(src.qregs_[i]);  //make copy of qregs from src

  this->creg_ = src.creg_;
  this->qubit_map_ = src.qubit_map_;
  next_iter_ = src.next_iter_;
  flow_marks_ = src.flow_marks_;
}

template <class state_t>
void Registers<state_t>::initialize_shots(const uint_t nshots, const uint_t seed)
{
  shots_.resize(nshots);
  for(int_t i=0;i<nshots;i++){
    shots_[i].set_seed(seed + i);
  }
}

template <class state_t>
void Registers<state_t>::branch_shots(reg_t& shots, int_t nbranch)
{
  branches_.clear();
  branches_.resize(nbranch);

  for(int_t i=0;i<nbranch;i++){
    branches_[i].creg_ = creg_;
  }
  for(int_t i=0;i<shots.size();i++){
    branches_[shots[i]].shots_.push_back(shots_[i]);
  }
}

//-------------------------------------------------------------------------
} // end namespace QuantumState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif

