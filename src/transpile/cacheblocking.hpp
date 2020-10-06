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

#ifndef _aer_cache_blocking_hpp_
#define _aer_cache_blocking_hpp_

#include "transpile/circuitopt.hpp"
#include "framework/utils.hpp"


namespace AER {
namespace Transpile {

class CacheBlocking : public CircuitOptimization {
public:
  CacheBlocking(void);
  ~CacheBlocking();

  void optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const Operations::OpSet &opset,
                                ExperimentData &data) const;

  void set_config(const json_t &config);

protected:
  mutable int block_bits_;    //qubits less than this will be blocked
  mutable int qubits_;
  mutable reg_t qubitMap_;
  mutable reg_t qubitSwapped_;
  bool blocking_enabled_;

  uint_t add_ops(std::vector<Operations::Op>& ops,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue) const;

  bool isCrossQubitsOp(Operations::Op& op, reg_t& qubits) const;

  bool is_diagonal_op(Operations::Op& op) const;
};

CacheBlocking::CacheBlocking(void)
{
  block_bits_ = 22;
  blocking_enabled_ = false;
}

CacheBlocking::~CacheBlocking()
{
  
}

void CacheBlocking::set_config(const json_t &config)
{
  CircuitOptimization::set_config(config);

  if (JSON::check_key("blocking_enable", config_))
    JSON::get_value(blocking_enabled_, "blocking_enable", config_);

  if (JSON::check_key("blocking_qubits", config_))
    JSON::get_value(block_bits_, "blocking_qubits", config_);
}


void CacheBlocking::optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const Operations::OpSet &opset,
                                ExperimentData &data) const
{
  uint_t i,j,k,t;
  uint_t qt,n;

  if(!blocking_enabled_)
    return;

  std::vector<Operations::Op> optOps;
  std::vector<Operations::Op> queue1;
  std::vector<Operations::Op> queue2;

  qubits_ = circ.num_qubits;
  if(block_bits_ >= qubits_){
    return;
  }

  qubitMap_.resize(qubits_);
  qubitSwapped_.resize(qubits_);

  for(i=0;i<qubits_;i++){
    qubitMap_[i] = i;
    qubitSwapped_[i] = i;
  }

  n = add_ops(circ.ops,optOps,queue1);
  while(queue1.size() > 0 && n != 0){
    n = add_ops(queue1,optOps,queue2);
    queue1.clear();

    if(queue2.size() == 0 || n == 0){
      break;
    }
    n = add_ops(queue2,optOps,queue1);
    queue2.clear();
  }

  //insert swap gates to restore original qubit order
  int nInBlock = 0;
  for(i=0;i<block_bits_;i++){  //at first swap qubits in the blocking region
    if(qubitMap_[i] != i && qubitMap_[i] < block_bits_){
      if(nInBlock == 0){
        uint_t last = optOps.size() - 1;
        if(optOps[last].type == Operations::OpType::sim_op && optOps[last].name == "end_blocking"){
          optOps.pop_back();
          nInBlock = 1;
        }
        else{
          Operations::Op blk_begin;
          blk_begin.type = Operations::OpType::sim_op;
          blk_begin.name = "begin_blocking";
          optOps.push_back(blk_begin);
        }
      }
      Operations::Op sgate;
      sgate.type = Operations::OpType::gate;
      sgate.name = "swap";
      sgate.qubits = {i,qubitMap_[i]};
      sgate.string_params = {sgate.name};
      optOps.push_back(sgate);

      j = qubitMap_[i];
      qubitMap_[qubitSwapped_[i]] = j;
      qubitMap_[i] = i;

      qubitSwapped_[j] = qubitSwapped_[i];
      qubitSwapped_[i] = i;

      nInBlock++;
    }
  }

  for(i=0;i<block_bits_;i++){   //second round, swap 
    if(qubitMap_[i] != i){
      j = qubitMap_[qubitMap_[i]];
      if(j != i && j < block_bits_){
        Operations::Op sgate;
        sgate.type = Operations::OpType::gate;
        sgate.name = "swap";
        sgate.qubits = {i,j};
        sgate.string_params = {sgate.name};
        optOps.push_back(sgate);

        qubitMap_[qubitSwapped_[i]] = j;
        qubitMap_[qubitSwapped_[j]] = i;

        t = qubitSwapped_[j];
        qubitSwapped_[j] = qubitSwapped_[i];
        qubitSwapped_[i] = t;

        nInBlock++;
      }
    }
  }
  if(nInBlock > 0){
    Operations::Op blk_end;
    blk_end.type = Operations::OpType::sim_op;
    blk_end.name = "end_blocking";
    optOps.push_back(blk_end);
  }

  for(i=0;i<qubits_;i++){  //final round swap remaining qubits
    if(qubitMap_[i] != i){
      Operations::Op sgate;
      sgate.type = Operations::OpType::gate;
      sgate.name = "swap_chunk";
      sgate.qubits = {i,qubitMap_[i]};
      sgate.string_params = {sgate.name};
      optOps.push_back(sgate);

      j = qubitMap_[i];
      qubitMap_[qubitSwapped_[i]] = j;
      qubitMap_[i] = i;

      qubitSwapped_[j] = qubitSwapped_[i];
      qubitSwapped_[i] = i;
    }
  }
  //add non-gate operations
  for(i=0;i<queue1.size();i++){
    optOps.push_back(queue1[i]);
  }
  for(i=0;i<queue2.size();i++){
    optOps.push_back(queue2[i]);
  }

  circ.ops = optOps;
  circ.set_params();

}

uint_t CacheBlocking::add_ops(std::vector<Operations::Op>& ops,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue) const
{
  uint_t i,j,iq,jq;
  uint_t qt;

  int nqubitUsed = 0;
  reg_t blockedQubits;
  int nq,nb;
  bool exist;
  uint_t pos_begin,num_gates_added;

  //find qubits to be blocked
  if(out.size() == 0){
    for(i=0;i<block_bits_;i++){
      blockedQubits.push_back(i);
    }
  }
  else{
    for(i=0;i<ops.size();i++){
      reg_t cross_bits;
      if(isCrossQubitsOp(ops[i],cross_bits)){
        if(cross_bits.size() > 0){
          if(cross_bits.size() > block_bits_){
            std::string error = "CacheBlocking operation " + ops[i].name + " has " + 
                                std::to_string(cross_bits.size()) + " bits that can not be fit in cache block " +
                                std::to_string(block_bits_) + " bits";
            throw std::runtime_error(error);
            break;
          }
          for(iq=0;iq<cross_bits.size();iq++){
            exist = false;
            nq = blockedQubits.size();
            for(j=0;j<nq;j++){
              if(cross_bits[iq] == blockedQubits[j]){
                exist = true;
                break;
              }
            }
            if(!exist){
              blockedQubits.push_back(cross_bits[iq]);
            }
          }
          while(blockedQubits.size() > block_bits_){
            blockedQubits.pop_back();
          }
          if(blockedQubits.size() >= block_bits_){
            break;
          }
        }
      }
    }


    //not enough qubits are blocked
    if(blockedQubits.size() < block_bits_){
      for(i=0;i<ops.size();i++){
        if(ops[i].type == Operations::OpType::gate || ops[i].type == Operations::OpType::matrix){
          for(iq=0;iq<ops[i].qubits.size();iq++){
            exist = false;
            nq = blockedQubits.size();
            for(j=0;j<nq;j++){
              if(ops[i].qubits[iq] == blockedQubits[j]){
                exist = true;
                break;
              }
            }
            if(!exist){
              blockedQubits.push_back(ops[i].qubits[iq]);
              if(blockedQubits.size() >= block_bits_){
                break;
              }
            }
          }
          if(blockedQubits.size() >= block_bits_){
            break;
          }
        }
      }
    }
  }

  pos_begin = out.size();
  num_gates_added = 0;

  //insert swap gates to block operations
  reg_t swap(block_bits_);
  std::vector<bool> mapped(block_bits_,false);
  nq = blockedQubits.size();
  for(i=0;i<nq;i++){
    swap[i] = qubits_;  //not defined
    for(j=0;j<block_bits_;j++){
      if(blockedQubits[i] == qubitSwapped_[j]){
        swap[i] = j;
        mapped[j] = true;
        break;
      }
    }
  }
  for(i=0;i<nq;i++){
    if(swap[i] == qubits_){
      for(j=0;j<block_bits_;j++){
        if(!mapped[j]){
          swap[i] = j;
          mapped[j] = true;
          break;
        }
      }
    }
  }
  for(i=0;i<nq;i++){
    if(qubitSwapped_[swap[i]] != blockedQubits[i]){ //need swap gate
      if(out.size() > 0){   //swap gate is not required for initial state
        Operations::Op sgate;
        sgate.type = Operations::OpType::gate;
        sgate.name = "swap_chunk";
        sgate.qubits = {swap[i],qubitMap_[blockedQubits[i]]};
        sgate.string_params = {sgate.name};

        out.push_back(sgate);
      }

      //swap map
      j = qubitMap_[blockedQubits[i]];
      qubitMap_[qubitSwapped_[swap[i]]] = j;
      qubitMap_[blockedQubits[i]] = swap[i];

      qubitSwapped_[j] = qubitSwapped_[swap[i]];
      qubitSwapped_[swap[i]] = blockedQubits[i];
    }
  }

  num_gates_added = 0;

  //gather blocked gates
  Operations::Op blk_begin;
  blk_begin.type = Operations::OpType::sim_op;
  blk_begin.name = "begin_blocking";
  out.push_back(blk_begin);

  for(i=0;i<ops.size();i++){
    if(ops[i].type == Operations::OpType::gate || ops[i].type == Operations::OpType::matrix){
      nb = 0;
      if(is_diagonal_op(ops[i])){
        //diagonal gate can be applied anytime
        nb = ops[i].qubits.size();
      }
      else{
        nq = blockedQubits.size();
        for(j=0;j<nq;j++){
          for(iq=0;iq<ops[i].qubits.size();iq++){
            if(ops[i].qubits[iq] == blockedQubits[j]){
              nb++;
            }
          }
        }
      }
      if(nb == ops[i].qubits.size()){
        exist = false;
        for(j=0;j<queue.size();j++){
          for(iq=0;iq<ops[i].qubits.size();iq++){
            for(jq=0;jq<queue[j].qubits.size();jq++){
              if(ops[i].qubits[iq] == queue[j].qubits[jq]){
                exist = true;
                break;
              }
            }
          }
          if(exist)
            break;
        }
        if(exist){
          queue.push_back(ops[i]);
        }
        else{
          //mapping swapped qubits
          for(iq=0;iq<ops[i].qubits.size();iq++){
            ops[i].qubits[iq] = qubitMap_[ops[i].qubits[iq]];
          }
          out.push_back(ops[i]);
          num_gates_added++;
        }
      }
      else{
        queue.push_back(ops[i]);
      }
    }
    else{
      queue.push_back(ops[i]);
    }
  }

  if(num_gates_added > 0){
    Operations::Op blk_end;
    blk_end.type = Operations::OpType::sim_op;
    blk_end.name = "end_blocking";
    out.push_back(blk_end);
  }

  if(num_gates_added == 0){
    //pop unnecessary operations
    while(out.size() > pos_begin){
      out.pop_back();
    }
  }

  return num_gates_added;


}

bool CacheBlocking::isCrossQubitsOp(Operations::Op& op, reg_t& qubits) const
{
  bool is_cross_qubits = false;
  bool is_pauli_str = false;

  if(op.type == Operations::OpType::gate){
    if(op.name == "swap")
      is_cross_qubits = true;
    else if(op.name == "pauli"){
      is_cross_qubits = true;
      is_pauli_str = true;
    }
    else if(op.qubits.size() > 1){
      is_cross_qubits = true;
    }
  }
  else if(op.type == Operations::OpType::matrix){ //fusion
    if(op.qubits.size() > 1){
      if(!Utils::is_diagonal(op.mats[0], .0)){
        is_cross_qubits = true;
      }
    }
  }
  else if(op.type == Operations::OpType::snapshot){
    //block Pauli expectation
    if(op.name == "expectation_value_pauli" || op.name == "expectation_value_pauli_with_variance" || op.name == "expectation_value_pauli_single_shot"){
      is_cross_qubits = true;
      is_pauli_str = true;
    }
  }

  if(is_cross_qubits){
    int i;
    qubits.clear();
    if(is_pauli_str){
      for(i=0;i<op.qubits.size();i++){
        switch(op.string_params[0][op.qubits.size() - 1 - i]){
          case 'X':
          case 'Y':
            qubits.push_back(op.qubits[i]);
            break;
          default:
            break;
        }
      }
    }
    else{
      for(i=0;i<op.qubits.size();i++){
        qubits.push_back(op.qubits[i]);
      }
    }
  }

  return is_cross_qubits;
}

bool CacheBlocking::is_diagonal_op(Operations::Op& op) const
{
  bool is_pauli_str = false;
  if(op.type == Operations::OpType::gate){
    if(op.name == "u1")
      return true;
    else if(op.name == "z")
      return true;
    else if(op.name == "s" || op.name == "sdg")
      return true;
    else if(op.name == "t" || op.name == "tdg")
      return true;
    else if(op.name == "pauli"){
      is_pauli_str = true;
    }
  }
  else if(op.type == Operations::OpType::matrix){
    if (Utils::is_diagonal(op.mats[0], .0)){
      return true;
    }
  }
  else if(op.type == Operations::OpType::snapshot){
    //block Pauli expectation
    if(op.name == "expectation_value_pauli" || op.name == "expectation_value_pauli_with_variance" || op.name == "expectation_value_pauli_single_shot"){
      is_pauli_str = true;
    }
  }

  if(is_pauli_str){
    int i;
    bool is_diag = true;
    for(i=0;i<op.qubits.size();i++){
      switch(op.string_params[0][op.qubits.size() - 1 - i]){
        case 'X':
        case 'Y':
          is_diag = false;
          break;
        default:
          break;
      }
    }
    return is_diag;
  }

  return false;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
