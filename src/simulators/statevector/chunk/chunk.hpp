/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_chunk_hpp_
#define _qv_chunk_hpp_

#include "simulators/statevector/chunk/device_chunk_container.hpp"
#include "simulators/statevector/chunk/host_chunk_container.hpp"

namespace AER {
namespace QV {


//============================================================================
// chunk class
//============================================================================
template <typename data_t>
class Chunk 
{
protected:
  mutable std::weak_ptr<ChunkContainer<data_t>> chunk_container_;   //pointer to chunk container
  std::shared_ptr<Chunk<data_t>> cache_;                 //pointer to cache chunk on device
  uint_t chunk_pos_;                    //position in container
  int place_;                           //container ID
  uint_t num_qubits_;                   //total number of qubits
  uint_t chunk_index_;                  //global chunk index
  bool mapped_;                         //mapped to qubitvector
  mutable bool in_capture_;
  bool enable_capture_;
public:
  Chunk()
  {
    chunk_pos_ = 0;
    place_ = -1;
    num_qubits_ = 0;
    chunk_index_ = 0;
    mapped_ = false;
    in_capture_ = false;
    enable_capture_ = true;
  }

  Chunk(std::weak_ptr<ChunkContainer<data_t>> cc,uint_t pos)
  {
    chunk_container_ = cc;
    chunk_pos_ = pos;
    place_ = chunk_container_.lock()->place();
    num_qubits_ = 0;
    chunk_index_ = 0;
    mapped_ = false;
  }
  Chunk(Chunk<data_t>& chunk)   //map chunk from exisiting chunk (used fo cache chunk)
  {
    chunk_container_ = chunk.chunk_container_;
    chunk_pos_ = chunk.chunk_pos_;
    place_ = chunk.place_;
    num_qubits_ = chunk.num_qubits_;
    chunk_index_ = chunk.chunk_index_;
    mapped_ = true;
    in_capture_ = false;
  }
  ~Chunk()
  {
  }

  void set_device(void) const
  {
    chunk_container_.lock()->set_device();
  }
  int device(void)
  {
    return chunk_container_.lock()->device();
  }

  std::shared_ptr<ChunkContainer<data_t>> container()
  {
    return chunk_container_.lock();
  }

  void map(std::weak_ptr<ChunkContainer<data_t>> cc,uint_t pos)
  {
    chunk_container_ = cc;
    chunk_pos_ = pos;
    place_ = chunk_container_.lock()->place();
    mapped_ = true;
  }

  uint_t pos(void)
  {
    return chunk_pos_;
  }
  int place(void)
  {
    return place_;
  }
  void set_place(int ip)
  {
    place_ = ip;
  }
  void map_cache(Chunk<data_t>& chunk)
  {
    cache_ = std::make_shared<Chunk<data_t>>(chunk);
  }
  void unmap_cache(void)
  {
    cache_->end_capture();
    cache_->unmap();
    cache_.reset();
  }
  
  bool is_mapped(void)
  {
    return mapped_;
  }
  void unmap(void)
  {
    end_capture();
    mapped_ = false;
    if(cache_)
      unmap_cache();
  }

  void set_num_qubits(uint_t qubits)
  {
    num_qubits_ = qubits;
  }
  void set_chunk_index(uint_t id)
  {
    chunk_index_ = id;
  }
  void enable_omp(bool flg)
  {
    chunk_container_.lock()->enable_omp(flg);
  }

  void Set(uint_t i,const thrust::complex<data_t>& t)
  {
    end_capture();
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->Set(i + (chunk_pos_ << sel_chunk_container->chunk_bits()),t);
  }
  thrust::complex<data_t> Get(uint_t i) const
  {
    end_capture();
    auto sel_chunk_container = chunk_container_.lock();
    return sel_chunk_container->Get(i + (chunk_pos_ << sel_chunk_container->chunk_bits()));
  }

  thrust::complex<data_t>& operator[](uint_t i)
  {
    end_capture();
    auto sel_chunk_container = chunk_container_.lock();
    return (*sel_chunk_container)[i + (chunk_pos_ << sel_chunk_container->chunk_bits())];
  }

  thrust::complex<data_t>* pointer(void)
  {
    return chunk_container_.lock()->chunk_pointer(chunk_pos_);
  }

  void StoreMatrix(const std::vector<std::complex<double>>& mat)
  {
    if(cache_){
      cache_->StoreMatrix(mat);
    }
    else{
      begin_capture();
      chunk_container_.lock()->StoreMatrix(mat,chunk_pos_);
    }
  }
  void StoreBatchedMatrix(const std::vector<std::complex<double>>& mat)
  {
    begin_capture();
    chunk_container_.lock()->StoreBatchedMatrix(mat);
  }
  void StoreMatrix(const std::complex<double>* mat,uint_t size)
  {
    if(cache_){
      cache_->StoreMatrix(mat,size);
    }
    else{
      begin_capture();
      chunk_container_.lock()->StoreMatrix(mat,chunk_pos_,size);
    }
  }
  void StoreUintParams(const std::vector<uint_t>& prm)
  {
    if(cache_){
      cache_->StoreUintParams(prm);
    }
    else{
      begin_capture();
      chunk_container_.lock()->StoreUintParams(prm,chunk_pos_);
    }
  }
  void StoreBatchedParams(const std::vector<batched_matrix_params>& prm)
  {
    begin_capture();
    chunk_container_.lock()->StoreBatchedParams(prm);
  }

  void CopyIn(Chunk<data_t>& src)
  {
    end_capture();
    chunk_container_.lock()->CopyIn(src,chunk_pos_);
  }
  void CopyOut(Chunk<data_t>& dest)
  {
    end_capture();
    chunk_container_.lock()->CopyOut(dest,chunk_pos_);
  }
  void CopyIn(thrust::complex<data_t>* src, uint_t size)
  {
    end_capture();
    chunk_container_.lock()->CopyIn(src, chunk_pos_, size);
  }
  void CopyOut(thrust::complex<data_t>* dest, uint_t size)
  {
    end_capture();
    chunk_container_.lock()->CopyOut(dest, chunk_pos_, size);
  }
  void Swap(Chunk<data_t>& src)
  {
    end_capture();
    chunk_container_.lock()->Swap(src,chunk_pos_);
  }

  template <typename Function>
  void Execute(Function func,uint_t count)
  {
    if(cache_){
      cache_->Execute(func,count);
    }
    else{
      begin_capture();
      chunk_container_.lock()->Execute(func,chunk_pos_,count);
    }
  }

  template <typename Function>
  void ExecuteSum(double* pSum,Function func,uint_t count) const
  {
    if(cache_){
      cache_->ExecuteSum(pSum,func,count);
    }
    else{
      begin_capture();
      chunk_container_.lock()->ExecuteSum(pSum,func,chunk_pos_,count);
    }
  }

  template <typename Function>
  void ExecuteSum2(double* pSum,Function func,uint_t count) const
  {
    if(cache_){
      cache_->ExecuteSum2(pSum,func,count);
    }
    else{
      begin_capture();
      chunk_container_.lock()->ExecuteSum2(pSum,func,chunk_pos_,count);
    }
  }

  void ExecuteOnHost(HostFuncBase* func)
  {
    if(cache_){
      cache_->ExecuteOnHost(func);
    }
    else{
      chunk_container_.lock()->ExecuteOnHost(func,chunk_pos_);
    }
  }

  reg_t sample_measure(const std::vector<double> &rnds,uint_t stride = 1,bool dot = true,uint_t count = 1) const
  {
    end_capture();

    return chunk_container_.lock()->sample_measure(chunk_pos_,rnds,stride,dot,count);
  }

  thrust::complex<double> norm(uint_t stride = 1,bool dot = true) const
  {
    end_capture();

    return chunk_container_.lock()->norm(chunk_pos_,stride,dot);
  }

#ifdef AER_THRUST_CUDA
  cudaStream_t stream(void)
  {
    return std::static_pointer_cast<DeviceChunkContainer<data_t>>(chunk_container_.lock())->stream(chunk_pos_);
  }
#endif

  thrust::complex<double>* matrix_pointer(void)
  {
    return chunk_container_.lock()->matrix_pointer(chunk_pos_);
  }
  uint_t* param_pointer(void)
  {
    return chunk_container_.lock()->param_pointer(chunk_pos_);
  }
  double* reduce_buffer(void)
  {
    if(cache_){
      return cache_->reduce_buffer();
    }
    return chunk_container_.lock()->reduce_buffer(chunk_pos_);
  }
  uint_t reduce_buffer_size(void)
  {
    if(cache_){
      return cache_->reduce_buffer_size();
    }
    return chunk_container_.lock()->reduce_buffer_size();
  }
  double* condition_buffer(void)
  {
    if(cache_){
      return cache_->condition_buffer();
    }
    return chunk_container_.lock()->condition_buffer(chunk_pos_);
  }
  void init_condition(std::vector<double>& cond)
  {
    if(cache_){
      cache_->init_condition(cond);
    }
    else{
      chunk_container_.lock()->init_condition(chunk_pos_,cond);
    }
  }
  bool update_condition(uint_t count,bool async = true)
  {
    if(cache_){
      return cache_->update_condition(count,async);
    }
    else{
      return chunk_container_.lock()->update_condition(chunk_pos_,count,async);
    }
  }

  void synchronize(void) const
  {
    if(cache_){
      cache_->synchronize();
    }
    else{
      end_capture();
      chunk_container_.lock()->synchronize(chunk_pos_);
    }
  }

  //set qubits to be blocked
  void set_blocked_qubits(const reg_t& qubits)
  {
    chunk_container_.lock()->set_blocked_qubits(chunk_pos_,qubits);
  }

  //do all gates stored in queue
  void apply_blocked_gates(void)
  {
    chunk_container_.lock()->apply_blocked_gates(chunk_pos_);
  }

  //queue gate for blocked execution
  void queue_blocked_gate(char gate,uint_t qubit,uint_t mask,const std::complex<double>* pMat = nullptr)
  {
    chunk_container_.lock()->queue_blocked_gate(chunk_pos_,gate,qubit,mask,pMat);
  }

  //CUDA graph
  void begin_capture() const
  {
    if(!in_capture_ && enable_capture_){
      chunk_container_.lock()->begin_capture(chunk_pos_);
      in_capture_ = true;
    }
  }
  void end_capture() const
  {
    if(in_capture_){
      chunk_container_.lock()->end_capture(chunk_pos_);
      in_capture_ = false;
    }
  }
  void enable_capture(bool flg)
  {
    enable_capture_ = flg;
  }

  int measured_cbit(int qubit)
  {
    return chunk_container_.lock()->measured_cbit(chunk_pos_,qubit);
  }
  uint_t* measured_bits_buffer(void)
  {
    return chunk_container_.lock()->measured_bits_buffer(chunk_pos_);
  }
};

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
