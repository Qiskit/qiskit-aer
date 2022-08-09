/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
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
  CacheBlocking() : block_bits_(0), blocking_enabled_(false), memory_blocking_bits_(0) {}
  ~CacheBlocking(){}

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        ExperimentResult &result) const override;

  void set_config(const json_t &config) override;
  bool enabled()
  {
    return blocking_enabled_;
  }
  int block_bits()
  {
    return block_bits_;
  }

  void set_sample_measure(bool enabled)
  {
    sample_measure_ = enabled;
  }

  void set_save_state(bool enabled)
  {
    save_state_ = enabled;
  }

  //setting blocking parameters automatically
  void set_blocking(int bits, size_t min_memory, uint_t n_place, const size_t complex_size, bool is_matrix = false);

  void set_num_processes(int np)
  {
    num_processes_ = np;
  }

protected:
  mutable int block_bits_;    //qubits less than this will be blocked
  mutable int qubits_;
  mutable reg_t qubitMap_;
  mutable reg_t qubitSwapped_;
  mutable bool blocking_enabled_;
  mutable bool sample_measure_ = false;
  mutable bool save_state_ = false;
  int memory_blocking_bits_ = 0;
  bool density_matrix_ = false;
  int num_processes_ = 1;

  bool block_circuit(Circuit& circ,bool doSwap) const;

  void put_nongate_ops(std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,std::vector<Operations::Op>& input,bool doSwap) const;

  uint_t add_ops(std::vector<Operations::Op>& ops,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,bool doSwap,bool first,bool crossQubitOnly) const;

  void restore_qubits_order(std::vector<Operations::Op>& ops) const;

  bool is_cross_qubits_op(Operations::Op& op) const;

  bool is_diagonal_op(Operations::Op& op) const;

  void insert_swap(std::vector<Operations::Op>& ops,uint_t bit0,uint_t bit1,bool chunk) const;
  void insert_sim_op(std::vector<Operations::Op>& ops,char* name,const reg_t& qubits) const;
  void insert_pauli(std::vector<Operations::Op>& ops,reg_t& qubits,std::string& pauli) const;

  void define_blocked_qubits(std::vector<Operations::Op>& ops,reg_t& blockedQubits,bool crossQubitOnly) const;

  bool can_block(Operations::Op& ops,reg_t& blockedQubits) const;
  bool can_reorder(Operations::Op& ops,std::vector<Operations::Op>& waiting_ops) const;

  bool split_pauli(const Operations::Op& op, const reg_t blockedQubits, std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue) const;

  bool split_op(const Operations::Op& op,const reg_t blockedQubits,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue) const;

  bool is_blockable_operation(Operations::Op& op) const;

  void target_qubits(Operations::Op& op, reg_t& targets) const;
};

void CacheBlocking::set_config(const json_t &config)
{
  CircuitOptimization::set_config(config);

  if (JSON::check_key("blocking_qubits", config_))
    JSON::get_value(block_bits_, "blocking_qubits", config_);

  if(block_bits_ >= 1){
    blocking_enabled_ = true;
  }

  if (JSON::check_key("memory_blocking_bits", config_)){
    JSON::get_value(memory_blocking_bits_, "memory_blocking_bits", config_);
    if(memory_blocking_bits_ >= 10){   //blocking qubit should be <=10
      memory_blocking_bits_ = 10;
    }
  }

  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if(method.find("density_matrix") != std::string::npos){
      density_matrix_ = true;
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

void CacheBlocking::insert_sim_op(std::vector<Operations::Op>& ops,char* name,const reg_t& qubits) const
{
  Operations::Op op;
  op.type = Operations::OpType::sim_op;
  op.name = name;
  op.string_params = {op.name};
  op.qubits = qubits;
  ops.push_back(op);
}


void CacheBlocking::optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const opset_t &allowed_opset,
                                ExperimentResult &result) const 
{
  if(!blocking_enabled_ && memory_blocking_bits_ == 0){
    return;
  }

  if(blocking_enabled_){
    qubits_ = circ.num_qubits;

    //loop over operations to find max number of parameters for cross-qubits operations
    int_t max_params = 1;
    for(uint_t i=0;i<circ.ops.size();i++){
      if(is_blockable_operation(circ.ops[i]) && is_cross_qubits_op(circ.ops[i])){
        reg_t targets;
        target_qubits(circ.ops[i],targets);
        if(targets.size() > max_params)
          max_params = targets.size();
      }
    }
    if(block_bits_ < max_params){
      block_bits_ = max_params;   //change blocking qubits so that we can put op with many params
    }

    if(num_processes_ > 1){
      if(block_bits_ >= qubits_){
        blocking_enabled_ = false;
        std::string error = "cache blocking : there are gates operation can not cache blocked in blocking_qubits = " + std::to_string(block_bits_);
        throw std::runtime_error(error);
        return;
      }
      if((1ull << (qubits_ - block_bits_)) < num_processes_){
        //not enough distribution
        blocking_enabled_ = false;
        std::string error = "cache blocking : blocking_qubits is to large to parallelize with " + std::to_string(num_processes_) +
                            " processes ";
        throw std::runtime_error(error);
        return;
      }
    }
    if(block_bits_ >= qubits_){
      blocking_enabled_ = false;
      return;
    }

    qubitMap_.resize(qubits_);
    qubitSwapped_.resize(qubits_);

    for(uint_t i=0;i<qubits_;i++){
      qubitMap_[i] = i;
      qubitSwapped_[i] = i;
    }

    blocking_enabled_ = block_circuit(circ,true);

    if(blocking_enabled_){
      result.metadata.add(true, "cacheblocking", "enabled");
      result.metadata.add(block_bits_, "cacheblocking", "block_bits");
    }
  }

  if(memory_blocking_bits_ > 0){
    if(memory_blocking_bits_ >= qubits_){
      return;
    }

    qubitMap_.resize(qubits_);
    qubitSwapped_.resize(qubits_);

    for(uint_t i=0;i<qubits_;i++){
      qubitMap_[i] = i;
      qubitSwapped_[i] = i;
    }

    uint_t bit_backup = block_bits_;
    block_bits_ = memory_blocking_bits_;

    block_circuit(circ,false);

    block_bits_ = bit_backup;

    result.metadata.add(true, "gpu_blocking", "enabled");
    result.metadata.add(memory_blocking_bits_, "gpu_blocking", "gpu_block_bits");
  }

  circ.set_params();
}

void CacheBlocking::define_blocked_qubits(std::vector<Operations::Op>& ops,reg_t& blockedQubits,bool crossQubitOnly) const
{
  uint_t i,j,iq;
  int nq,nb;
  bool exist;
  for(i=0;i<ops.size();i++){
    if(blockedQubits.size() >= block_bits_)
      break;

    reg_t targets;
    target_qubits(ops[i],targets);

    reg_t blockedQubits_add;

    nq = blockedQubits.size();
    for(iq=0;iq<targets.size();iq++){
      exist = false;
      for(j=0;j<nq;j++){
        if(targets[iq] == blockedQubits[j]){
          exist = true;
          break;
        }
      }
      if(!exist)
        blockedQubits_add.push_back(targets[iq]);
    }
    //only if all the qubits of gate can be added
    if(blockedQubits_add.size() + nq <= block_bits_){
      blockedQubits.insert(blockedQubits.end(),blockedQubits_add.begin(),blockedQubits_add.end());
    }
  }
}


bool CacheBlocking::can_block(Operations::Op& op,reg_t& blockedQubits) const
{
  //check if the operation can be blocked in cache
  reg_t targets;
  target_qubits(op,targets);

  if(targets.size() > block_bits_){
    return false;
  }

  uint_t j,iq,nq,nb;
  nq = blockedQubits.size();
  nb = 0;
  for(iq=0;iq<targets.size();iq++){
    for(j=0;j<nq;j++){
      if(targets[iq] == blockedQubits[j]){
        nb++;
        break;
      }
    }
  }
  if(nb == targets.size())
    return true;
  return false;
}

bool CacheBlocking::can_reorder(Operations::Op& op,std::vector<Operations::Op>& waiting_ops) const
{
  //check if the operation can be reordered in front of waiting queue
  uint_t j,iq,jq;

  //only blockable ops can be reordered
  if(!is_blockable_operation(op))
    return false;

  for(j=0;j<waiting_ops.size();j++){
    if(is_blockable_operation(waiting_ops[j])){
      for(iq=0;iq<op.qubits.size();iq++){
        for(jq=0;jq<waiting_ops[j].qubits.size();jq++){
          if(op.qubits[iq] == waiting_ops[j].qubits[jq]){
            return false;
          }
        }
      }
    }
    else{
      return false;
    }
  }
  return true;
}

bool CacheBlocking::block_circuit(Circuit& circ,bool doSwap) const
{
  uint_t i,n;
  std::vector<Operations::Op> out;
  std::vector<Operations::Op> queue;
  std::vector<Operations::Op> queue_next;
  bool crossQubits = false;

  n = add_ops(circ.ops,out,queue,doSwap,true,crossQubits);
  while(queue.size() > 0){
    n = add_ops(queue,out,queue_next,doSwap,false,crossQubits);

    queue = queue_next;
    queue_next.clear();
    if(n == 0){
      if(queue.size() > 0 && crossQubits == false){
        crossQubits = true;
        continue;
      }
      break;
    }
    crossQubits = false;
  }

  if(queue.size() > 0){
    return false;
  }

  if(doSwap && save_state_)
    restore_qubits_order(out);

  circ.ops = out;

  return true;
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
          insert_sim_op(ops,"begin_blocking",qubitMap_);
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
        if(nInBlock == 0){
          uint_t last = ops.size() - 1;
          if(ops[last].type == Operations::OpType::sim_op && ops[last].name == "end_blocking"){
            ops.pop_back();
            nInBlock = 1;
          }
          else{
            insert_sim_op(ops,"begin_blocking",qubitMap_);
          }
        }
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
    insert_sim_op(ops,"end_blocking",qubitMap_);
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

bool CacheBlocking::is_blockable_operation(Operations::Op& op) const
{
  if(op.type == Operations::OpType::gate || op.type == Operations::OpType::matrix || 
     op.type == Operations::OpType::diagonal_matrix || op.type == Operations::OpType::multiplexer ||
     op.type == Operations::OpType::superop){
    return true;
  }
  if(density_matrix_ && op.type == Operations::OpType::reset){
    return true;
  }

  return false;
}

uint_t CacheBlocking::add_ops(std::vector<Operations::Op>& ops,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue,bool doSwap,bool first,bool crossQubitOnly) const
{
  uint_t i,j,iq;

  int nqubitUsed = 0;
  reg_t blockedQubits;
  int nq;
  bool exist;
  uint_t pos_begin,num_gates_added;
  bool end_block_inserted;

  pos_begin = out.size();
  num_gates_added = 0;

  //find qubits to be blocked
  if(first && doSwap){
    //use lower bits for initialization
    for(i=0;i<block_bits_;i++){
      blockedQubits.push_back(i);
    }
  }
  else{
    if(crossQubitOnly){
      //add multi-qubits gate at first
      define_blocked_qubits(ops,blockedQubits,true);

      //not enough qubits are blocked, then add one qubit gate
      if(blockedQubits.size() < block_bits_)
        define_blocked_qubits(ops,blockedQubits,false);
    }
    else{
      define_blocked_qubits(ops,blockedQubits,false);
    }
  }

  pos_begin = out.size();
  num_gates_added = 0;

  if(doSwap){
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
        if(!first){   //swap gate is not required for initial state
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
  }

  if(doSwap)
    insert_sim_op(out,"begin_blocking",blockedQubits);
  else
    insert_sim_op(out,"begin_memory_blocking",blockedQubits);
  end_block_inserted = false;

  //gather blocked gates
  for(i=0;i<ops.size();i++){
    if(is_blockable_operation(ops[i])){
      if(!end_block_inserted){
        if(is_diagonal_op(ops[i]) || can_block(ops[i],blockedQubits)){
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
        else if(ops[i].name == "pauli"){
          if(can_reorder(ops[i],queue)){
            if(split_pauli(ops[i],blockedQubits,out,queue))
              num_gates_added++;
            continue;
          }
        }
        else if(ops[i].type == Operations::OpType::reset){    //reset for density matrix can be cache blocked
          if(can_reorder(ops[i],queue)){
            if(split_op(ops[i],blockedQubits,out,queue))
              num_gates_added++;
            continue;
          }
        }
      }
    }
    else{
      if(queue.size() == 0){          //if queue is empty, apply op here
        bool restore_qubits = false;
        if(ops[i].type == Operations::OpType::kraus){
          if(ops[i].qubits.size() > block_bits_){
            throw std::runtime_error("CacheBlocking : Kraus operator, number of qubits should be smaller than chunk qubit size");
            break;
          }
          if(!can_block(ops[i],blockedQubits)){  //if some qubits are out of chunk, queued for next step
            queue.push_back(ops[i]);
            continue;
          }
        }
        else if(ops[i].type == Operations::OpType::initialize){
          if(ops[i].qubits.size() <= block_bits_){
            if(!can_block(ops[i],blockedQubits)){  //if some qubits are out of chunk, queued for next step
              queue.push_back(ops[i]);
              continue;
            }
          }
          //otherwise StateChunk have to parallelize initialize operation
        }
        else if(sample_measure_ && ops[i].type == Operations::OpType::measure){
          //currently sampling should be done with original qubit mapping (TO DO : sampling without inserting swaps)
          restore_qubits = true;
        }
        else if(ops[i].type != Operations::OpType::measure && ops[i].type != Operations::OpType::reset && 
                ops[i].type != Operations::OpType::save_amps && ops[i].type != Operations::OpType::save_amps_sq &&
                ops[i].type != Operations::OpType::save_densmat && ops[i].type != Operations::OpType::bfunc){
          if(!(ops[i].type == Operations::OpType::snapshot && ops[i].name == "density_matrix")){
            restore_qubits = true;
          }
        }

        if(num_gates_added > 0 && !end_block_inserted){  //insert end of block to synchronize chunks
          if(doSwap)
            insert_sim_op(out,"end_blocking",blockedQubits);
          else
            insert_sim_op(out,"end_memory_blocking",blockedQubits);
        }
        else if(!end_block_inserted){
          out.pop_back();
        }
        if(restore_qubits && doSwap)
          restore_qubits_order(out);

        //mapping swapped qubits
        if(doSwap){
          for(iq=0;iq<ops[i].qubits.size();iq++){
            ops[i].qubits[iq] = qubitMap_[ops[i].qubits[iq]];
          }
        }

        out.push_back(ops[i]);
        num_gates_added++;

        end_block_inserted = true;
        continue;
      }
    }
    queue.push_back(ops[i]);
  }

  if(!end_block_inserted){
    if(num_gates_added > 0){
      if(doSwap)
        insert_sim_op(out,"end_blocking",blockedQubits);
      else
        insert_sim_op(out,"end_memory_blocking",blockedQubits);
    }
    else{
      //pop unnecessary operations
      while(out.size() > pos_begin){
        out.pop_back();
      }
    }
  }

  return num_gates_added;
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
      return false;   //pauli operation can be splited into non-cross-qubit ops
    else if(op.qubits.size() > 1)
      return true;
  }
  else if(op.type == Operations::OpType::matrix || op.type == Operations::OpType::multiplexer || op.type == Operations::OpType::superop){
    if(op.qubits.size() > 1)
      return true;
  }
  if(op.type == Operations::OpType::kraus){
    return true;
  }

  return false;
}

bool CacheBlocking::is_diagonal_op(Operations::Op& op) const
{
  if(op.type == Operations::OpType::gate){
    if(op.name == "u1")
      return true;
    else if(op.name == "z")
      return true;
    else if(op.name == "s" || op.name == "sdg")
      return true;
    else if(op.name == "t" || op.name == "tdg")
      return true;
  }
  else if(op.type == Operations::OpType::matrix){
    if (Utils::is_diagonal(op.mats[0], .0)){
      return true;
    }
  }
  else if(op.type == Operations::OpType::diagonal_matrix){
    return true;
  }
  return false;
}

void CacheBlocking::target_qubits(Operations::Op& op, reg_t& targets) const
{
  if(op.type == Operations::OpType::gate){
    bool swap = false;
    if(op.name.find("swap") != std::string::npos){
      swap = true;
    }

    if(op.name[0] == 'c' || op.name.find("mc") == 0){
      //multi control gates
      if(swap)
        targets.push_back(op.qubits[op.qubits.size() - 2]);
      targets.push_back(op.qubits[op.qubits.size() - 1]);
    }
    else{
      targets = op.qubits;
    }
  }
  else{
    targets = op.qubits;
  }
}

void CacheBlocking::insert_pauli(std::vector<Operations::Op>& ops,reg_t& qubits,std::string& pauli) const
{
  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "pauli";
  op.qubits = qubits;
  op.string_params = {pauli};
  ops.push_back(op);
}

//split 1 pauli gate to 2 pauli gates (inside and outside)
bool CacheBlocking::split_pauli(const Operations::Op& op,const reg_t blockedQubits,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue) const
{
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;
  std::string pauli_in_chunk;
  std::string pauli_out_chunk;
  int_t i,j,n;
  bool inside;

  //get inner/outer chunk pauli string
  n = op.qubits.size();
  for(i=0;i<n;i++){
    if(op.string_params[0][n - 1 - i] == 'I')
      continue;   //remove I

    inside = false;
    for(j=0;j<blockedQubits.size();j++){
      if(op.qubits[i] == blockedQubits[j]){
        inside = true;
        break;
      }
    }
    if(inside){
      qubits_in_chunk.push_back(op.qubits[i]);
      pauli_in_chunk.push_back(op.string_params[0][n-i-1]);
    }
    else{
      qubits_out_chunk.push_back(op.qubits[i]);
      pauli_out_chunk.push_back(op.string_params[0][n-i-1]);
    }
  }

  if(qubits_out_chunk.size() > 0){  //save in queue
    std::reverse(pauli_out_chunk.begin(),pauli_out_chunk.end());
    insert_pauli(queue,qubits_out_chunk,pauli_out_chunk);
  }

  if(qubits_in_chunk.size() > 0){
    std::reverse(pauli_in_chunk.begin(),pauli_in_chunk.end());
    //mapping swapped qubits
    for(i=0;i<qubits_in_chunk.size();i++){
      qubits_in_chunk[i] = qubitMap_[qubits_in_chunk[i]];
    }
    insert_pauli(out,qubits_in_chunk,pauli_in_chunk);
    return true;
  }

  return false;
}

//split op to inside op and outside op
bool CacheBlocking::split_op(const Operations::Op& op,const reg_t blockedQubits,std::vector<Operations::Op>& out,std::vector<Operations::Op>& queue) const
{
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;
  int_t i,j,n;
  bool inside;

  n = op.qubits.size();
  for(i=0;i<n;i++){
    inside = false;
    for(j=0;j<blockedQubits.size();j++){
      if(op.qubits[i] == blockedQubits[j]){
        inside = true;
        break;
      }
    }
    if(inside){
      qubits_in_chunk.push_back(op.qubits[i]);
    }
    else{
      qubits_out_chunk.push_back(op.qubits[i]);
    }
  }

  if(qubits_out_chunk.size() > 0){  //save in queue
    Operations::Op op_out = op;
    op_out.qubits = qubits_out_chunk;
    queue.push_back(op_out);
  }

  if(qubits_in_chunk.size() > 0){
    Operations::Op op_in = op;
    //mapping swapped qubits
    for(i=0;i<qubits_in_chunk.size();i++){
      qubits_in_chunk[i] = qubitMap_[qubits_in_chunk[i]];
    }
    op_in.qubits = qubits_in_chunk;
    out.push_back(op_in);
    return true;
  }

  return false;
}


//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
