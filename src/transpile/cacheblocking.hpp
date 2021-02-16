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


/*

The input 3 qubits circuit is transpiled with cache blocking 2 qbuits

0 --H--.--H-----------O--H--
       |              |   
1 -----O--H--.--H--O--.--H--
             |     |
2------------O--H--.--------

The output circuit, 2 (noiseless) swap gates are inserted and there is no gate on qubit 2

0 --H--.--H--x--O--H--.--x--O--H--
       |     |  |     |  |     |   
1 -----O--H--|--.--H--O--|--.--H--
             |           |
2------------x-----------x--------

*/


#ifndef _aer_cache_blocking_hpp_
#define _aer_cache_blocking_hpp_

#include "transpile/circuitopt.hpp"
#include "framework/utils.hpp"


namespace AER {
namespace Transpile {

class CacheBlocking : public CircuitOptimization {
public:
  CacheBlocking() : block_bits_(22), blocking_enabled_(false), gpu_blocking_bits_(0) {}
  ~CacheBlocking(){}

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        ExperimentResult &result) const override;

  void set_config(const json_t &config);
  bool enabled(void)
  {
    return blocking_enabled_;
  }

  //setting blocking parameters automatically
  void set_blocking(int bits, size_t min_memory, uint_t n_place, size_t complex_size = 16, bool is_matrix = false);

protected:
  mutable int block_bits_;    //qubits less than this will be blocked
  mutable int qubits_;
  mutable reg_t qubitMap_;
  mutable reg_t qubitSwapped_;
  bool blocking_enabled_;
  bool ignore_diagonal_ = false;
  int gpu_blocking_bits_;

  void block_circuit(Circuit& circ,bool doSwap) const;

  void put_nongate_ops(std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,std::vector<Operations::Op>& input,bool doSwap) const;

  uint_t add_ops(std::vector<Operations::Op>& ops,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,bool doSwap) const;

  void restore_qubits_order(std::vector<Operations::Op>& ops) const;

  bool is_cross_qubits_op(Operations::Op& op) const;
  void get_cross_qubits(Operations::Op& op, reg_t& qubits) const;

  bool is_diagonal_op(Operations::Op& op) const;

  bool is_single_qubit_gate(Operations::Op& op) const;

  void insert_swap(std::vector<Operations::Op>& ops,uint_t bit0,uint_t bit1,bool chunk) const;

  void insert_sim_op(std::vector<Operations::Op>& ops,char* name) const;

  void define_blocked_qubits(std::vector<Operations::Op>& ops,reg_t& blockedQubits,bool crossQubitOnly) const;

  bool can_block(Operations::Op& ops,reg_t& blockedQubits) const;
  bool can_reorder(Operations::Op& ops,std::vector<Operations::Op>& waiting_ops) const;

};

void CacheBlocking::set_config(const json_t &config)
{
  CircuitOptimization::set_config(config);

  if (JSON::check_key("blocking_enable", config_))
    JSON::get_value(blocking_enabled_, "blocking_enable", config_);

  if (JSON::check_key("blocking_qubits", config_))
    JSON::get_value(block_bits_, "blocking_qubits", config_);

  if (JSON::check_key("blocking_ignore_diagonal", config_))
    JSON::get_value(ignore_diagonal_, "blocking_ignore_diagonal", config_);

  if (JSON::check_key("gpu_blocking_bits", config_)){
    JSON::get_value(gpu_blocking_bits_, "gpu_blocking_bits", config_);
    if(gpu_blocking_bits_ >= 10){   //blocking qubit should be <=10
      gpu_blocking_bits_ = 10;
    }
  }
}


void CacheBlocking::set_blocking(int bits, size_t min_memory, uint_t n_place, size_t complex_size, bool is_matrix)
{
  int chunk_bits = bits;
  uint_t scale = is_matrix ? 2 : 1;
  size_t size;

  //get largest possible chunk bits
  while((complex_size << (scale*chunk_bits)) > min_memory){
    chunk_bits--;
    if(chunk_bits < 1){
      break;
    }
  }

  if(chunk_bits == 0){
    throw std::runtime_error("CacheBlocking : Auto blocking configure failed");
  }

  //divide chunks so that chunks can be distributed on all memory space
  while( (1ull << (bits - chunk_bits)) < n_place){
    chunk_bits--;
    if(chunk_bits < 1){
      break;
    }
  }

  if(chunk_bits == 0){
    throw std::runtime_error("CacheBlocking : Auto blocking configure failed");
  }

  blocking_enabled_ = true;
  block_bits_ = chunk_bits;
}

void CacheBlocking::insert_swap(std::vector<Operations::Op>& ops,uint_t bit0,uint_t bit1,bool chunk) const
{
  Operations::Op sgate;
  sgate.type = Operations::OpType::gate;
  if(chunk)
    sgate.name = "swap_chunk";
  else
    sgate.name = "swap";
  sgate.qubits = {bit0,bit1};
  sgate.string_params = {sgate.name};
  ops.push_back(sgate);
}

void CacheBlocking::insert_sim_op(std::vector<Operations::Op>& ops,char* name) const
{
  Operations::Op op;
  op.type = Operations::OpType::sim_op;
  op.name = name;
  op.string_params = {op.name};
  ops.push_back(op);
}


void CacheBlocking::optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const opset_t &allowed_opset,
                                ExperimentResult &result) const 
{
  if(!blocking_enabled_ && gpu_blocking_bits_ == 0){
    return;
  }

  if(blocking_enabled_){

    qubits_ = circ.num_qubits;
    if(block_bits_ >= qubits_){
      return;
    }

    result.metadata.add(true, "cacheblocking", "enabled");
    result.metadata.add(block_bits_, "cacheblocking", "block_bits");

    qubitMap_.resize(qubits_);
    qubitSwapped_.resize(qubits_);

    for(uint_t i=0;i<qubits_;i++){
      qubitMap_[i] = i;
      qubitSwapped_[i] = i;
    }

    block_circuit(circ,true);
  }

  if(gpu_blocking_bits_ > 0){
    block_circuit(circ,false);
  }

  circ.set_params();
}

void CacheBlocking::define_blocked_qubits(std::vector<Operations::Op>& ops,reg_t& blockedQubits,bool crossQubitOnly) const
{
  uint_t i,j,iq;
  int nq,nb;
  bool exist;
  for(i=0;i<ops.size();i++){
    reg_t cross_bits;
    if(is_cross_qubits_op(ops[i])){
      get_cross_qubits(ops[i],cross_bits);
      if(cross_bits.size() > block_bits_){
        std::string error = "CacheBlocking operation " + ops[i].name + " has " + 
                            std::to_string(cross_bits.size()) + " bits that can not be fit in cache block " +
                            std::to_string(block_bits_) + " bits";
        throw std::runtime_error(error);
      }
      reg_t blockedQubits_add;

      nq = blockedQubits.size();
      for(iq=0;iq<cross_bits.size();iq++){
        exist = false;
        for(j=0;j<nq;j++){
          if(cross_bits[iq] == blockedQubits[j]){
            exist = true;
            break;
          }
        }
        if(!exist)
          blockedQubits_add.push_back(cross_bits[iq]);
      }
      //only if all the qubits of gate can be added
      if(blockedQubits_add.size() + nq <= block_bits_){
        blockedQubits.insert(blockedQubits.end(),blockedQubits_add.begin(),blockedQubits_add.end());
      }
    }
    else if(!crossQubitOnly){
      if(is_single_qubit_gate(ops[i])){  //one qubit gate
        if(blockedQubits.size() + 1 <= block_bits_){
          blockedQubits.push_back(ops[i].qubits[0]);
        }
      }
    }

    if(blockedQubits.size() >= block_bits_){
      break;
    }
  }
}


bool CacheBlocking::can_block(Operations::Op& op,reg_t& blockedQubits) const
{
  //check if the operation can be blocked in cache
  uint_t j,iq,nq,nb;
  nq = blockedQubits.size();
  nb = 0;
  for(iq=0;iq<op.qubits.size();iq++){
    for(j=0;j<nq;j++){
      if(op.qubits[iq] == blockedQubits[j]){
        nb++;
        break;
      }
    }
  }
  if(nb == op.qubits.size())
    return true;
  return false;
}

bool CacheBlocking::can_reorder(Operations::Op& op,std::vector<Operations::Op>& waiting_ops) const
{
  //check if the operation can be reordered in front of waiting queue
  uint_t j,iq,jq;

  //only gate and matrix can reorder
  if(op.type != Operations::OpType::gate && op.type != Operations::OpType::matrix){
    return false;
  }

  for(j=0;j<waiting_ops.size();j++){
    for(iq=0;iq<op.qubits.size();iq++){
      for(jq=0;jq<waiting_ops[j].qubits.size();jq++){
        if(op.qubits[iq] == waiting_ops[j].qubits[jq]){
          return false;
        }
      }
    }
  }
  return true;
}

void CacheBlocking::block_circuit(Circuit& circ,bool doSwap) const
{
  uint_t i,n;
  std::vector<Operations::Op> out;
  std::vector<Operations::Op> queue;
  std::vector<Operations::Op> queue_next;

  n = add_ops(circ.ops,out,queue,doSwap);
  put_nongate_ops(out,queue_next,queue,doSwap);
  queue.clear();
  while(queue_next.size() > 0 && n != 0){
    n = add_ops(queue_next,out,queue,doSwap);
    queue_next.clear();
    put_nongate_ops(out,queue_next,queue,doSwap);
    queue.clear();
  }

  circ.ops = out;
}

void CacheBlocking::put_nongate_ops(std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,std::vector<Operations::Op>& input,bool doSwap) const
{
  uint_t i;
  for(i=0;i<input.size();i++){
    if(input[i].type == Operations::OpType::gate || input[i].type == Operations::OpType::matrix){
      for(uint_t j =i;j<input.size();j++){
        queue.push_back(input[j]);
      }
      return;   //there are still gates operations remaining in queue
    }

    if(doSwap){
      //insert swap to restore qubit ordering
      restore_qubits_order(out);

      doSwap = false;
    }
    //add operation to output
    out.push_back(input[i]);
  }
}

void CacheBlocking::restore_qubits_order(std::vector<Operations::Op>& ops) const
{
  uint_t i,j,t;

  //insert swap gates to restore original qubit order
  int nInBlock = 0;
  //at first, find pair of qubits can be swapped in the cache block
  for(i=0;i<block_bits_;i++){
    if(qubitMap_[i] != i && qubitMap_[i] < block_bits_){
      if(nInBlock == 0){
        uint_t last = ops.size() - 1;
        if(ops[last].type == Operations::OpType::sim_op && ops[last].name == "end_blocking"){
          ops.pop_back();
          nInBlock = 1;
        }
        else{
          insert_sim_op(ops,"begin_blocking");
        }
      }
      insert_swap(ops,i,qubitMap_[i],false);

      j = qubitMap_[i];
      qubitMap_[qubitSwapped_[i]] = j;
      qubitMap_[i] = i;

      qubitSwapped_[j] = qubitSwapped_[i];
      qubitSwapped_[i] = i;

      nInBlock++;
    }
  }

  //second, find pair of qubits inside cache block
  for(i=0;i<block_bits_;i++){
    if(qubitMap_[i] != i){
      j = qubitMap_[qubitMap_[i]];
      if(j != i && j < block_bits_){
        insert_swap(ops,i,j,false);

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
    insert_sim_op(ops,"end_blocking");
  }

  //finally find all pair of remaining qubits so that we can restore initial qubits
  uint_t count;
  do{
    count = 0;
    for(i=0;i<qubits_;i++){
      if(qubitMap_[i] != i){
        insert_swap(ops,i,qubitMap_[i],true);

        j = qubitMap_[i];
        qubitMap_[qubitSwapped_[i]] = j;
        qubitMap_[i] = i;

        qubitSwapped_[j] = qubitSwapped_[i];
        qubitSwapped_[i] = i;
        count++;
      }
    }
  }while(count != 0);
}

uint_t CacheBlocking::add_ops(std::vector<Operations::Op>& ops,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,bool doSwap) const
{
  uint_t i,j,iq;

  int nqubitUsed = 0;
  reg_t blockedQubits;
  int nq;
  bool exist;
  uint_t pos_begin,num_gates_added;

  pos_begin = out.size();
  num_gates_added = 0;

  if(doSwap){
    //find qubits to be blocked
    if(out.size() == 0){
      //use lower bits for initialization
      for(i=0;i<block_bits_;i++){
        blockedQubits.push_back(i);
      }
    }
    else{
      //add multi-qubits gate at first
      define_blocked_qubits(ops,blockedQubits,true);

      //not enough qubits are blocked, then add one qubit gate
      if(blockedQubits.size() < block_bits_)
        define_blocked_qubits(ops,blockedQubits,false);
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
          insert_swap(out,swap[i],qubitMap_[blockedQubits[i]],true);
        }

        //swap map
        j = qubitMap_[blockedQubits[i]];
        qubitMap_[qubitSwapped_[swap[i]]] = j;
        qubitMap_[blockedQubits[i]] = swap[i];

        qubitSwapped_[j] = qubitSwapped_[swap[i]];
        qubitSwapped_[swap[i]] = blockedQubits[i];
      }
    }

    insert_sim_op(out,"begin_blocking");

    //gather blocked gates
    for(i=0;i<ops.size();i++){
      if(ops[i].type == Operations::OpType::gate || ops[i].type == Operations::OpType::matrix){
        if((ignore_diagonal_ && is_diagonal_op(ops[i])) || can_block(ops[i],blockedQubits)){
          if(can_reorder(ops[i],queue)){
            //mapping swapped qubits
            for(iq=0;iq<ops[i].qubits.size();iq++){
              ops[i].qubits[iq] = qubitMap_[ops[i].qubits[iq]];
            }
            out.push_back(ops[i]);
            num_gates_added++;
            continue;
          }
        }
      }
      queue.push_back(ops[i]);
    }

    if(num_gates_added > 0){
      insert_sim_op(out,"end_blocking");
    }
    else{
      //pop unnecessary operations
      while(out.size() > pos_begin){
        out.pop_back();
      }
    }
  }
  else{
    i = 0;
    //add chunk swap and block ops (if blocking is enabled)
    if(blocking_enabled_){
      while(i <ops.size()){
        if(ops[i].type == Operations::OpType::sim_op){
          out.push_back(ops[i]);
        }
        else if(ops[i].type == Operations::OpType::gate && ops[i].name == "swap_chunk"){
          out.push_back(ops[i]);
        }
        else{
          break;
        }
        i++;
      }
    }

    insert_sim_op(out,"begin_register_blocking");
    //gather blocked gates
    while(i < ops.size()){
      if(ops[i].type == Operations::OpType::gate || ops[i].type == Operations::OpType::matrix){
        if((ops[i].qubits.size() > 1 && ops[i].type == Operations::OpType::matrix) || ops[i].name == "pauli"){
          queue.push_back(ops[i]);
        }
        else{
          if(can_reorder(ops[i],queue)){
            if(ignore_diagonal_ && is_diagonal_op(ops[i])){
              //diagonal gate can be applied
              out.push_back(ops[i]);
              num_gates_added++;
            }
            else{
              exist = false;
              iq = ops[i].qubits[ops[i].qubits.size()-1]; //block target bit
              nq = blockedQubits.size();
              for(j=0;j<nq;j++){
                if(iq == blockedQubits[j]){
                  exist = true;
                  break;
                }
              }
              if(exist){
                out.push_back(ops[i]);
                num_gates_added++;
              }
              else{
                if(nq == gpu_blocking_bits_){
                  queue.push_back(ops[i]);
                }
                else{
                  blockedQubits.push_back(iq);
                  out.push_back(ops[i]);
                  num_gates_added++;
                }
              }
            }
          }
          else{
            queue.push_back(ops[i]);
          }
        }
      }
      else{
        queue.push_back(ops[i]);
      }
      i++;
    }

    if(out.size() > pos_begin + 1){
      out[pos_begin].qubits = blockedQubits;  //store qubits to be blocked in the sim_op::begin_register_blocking
      insert_sim_op(out,"end_register_blocking");
    }
    else{
      out.pop_back();
    }
  }

  return num_gates_added;
}

bool CacheBlocking::is_single_qubit_gate(Operations::Op& op) const
{
  if(op.qubits.size() == 1){
    if((op.type == Operations::OpType::gate && op.name != "pauli") || op.type == Operations::OpType::matrix)
      return true;
  }
  return false;
}

bool CacheBlocking::is_cross_qubits_op(Operations::Op& op) const
{
  if(is_diagonal_op(op)){
    return false;
  }

  if(op.type == Operations::OpType::gate){
    if(op.name == "swap")
      return true;
    else if(op.name == "pauli")
      return true;
    else if(op.qubits.size() > 1)
      return true;
  }
  else if(op.type == Operations::OpType::matrix){ //fusion
    if(op.qubits.size() > 1)
      return true;
  }

  return false;
}

void CacheBlocking::get_cross_qubits(Operations::Op& op, reg_t& qubits) const
{
  int i;
  if(op.name == "pauli"){
    qubits.clear();
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
    qubits = op.qubits;
  }
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
    if (ignore_diagonal_ && Utils::is_diagonal(op.mats[0], .0)){
      return true;
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
