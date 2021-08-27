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
  std::shared_ptr<Chunk<data_t>> cache_;                //pointer to cache chunk on device
  uint_t chunk_pos_;                    //position in container
  int place_;                           //container ID
  uint_t num_qubits_;                   //total number of qubits
  uint_t chunk_index_;                  //global chunk index
  bool mapped_;                         //mapped to qubitvector
public:
  Chunk(std::shared_ptr<ChunkContainer<data_t>> cc,uint_t pos)
  {
    chunk_container_ = cc;
    chunk_pos_ = pos;
    place_ = cc->place();
    num_qubits_ = 0;
    chunk_index_ = 0;
    mapped_ = false;
    cache_ = nullptr;
  }
  ~Chunk()
  {
    if(cache_)
      cache_.reset();
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
  void set_cache(const std::shared_ptr<Chunk<data_t>>& c)
  {
    cache_ = c;
  }
  bool is_mapped(void)
  {
    return mapped_;
  }
  void map(void)
  {
    mapped_ = true;
  }
  void unmap(void)
  {
    mapped_ = false;
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
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->synchronize(chunk_pos_);
    sel_chunk_container->Set(i + (chunk_pos_ << sel_chunk_container->chunk_bits()),t);
  }
  thrust::complex<data_t> Get(uint_t i) const
  {
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->synchronize(chunk_pos_);
    return sel_chunk_container->Get(i + (chunk_pos_ << sel_chunk_container->chunk_bits()));
  }

  thrust::complex<data_t>& operator[](uint_t i)
  {
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->synchronize(chunk_pos_);
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
      chunk_container_.lock()->StoreMatrix(mat,chunk_pos_);
    }
  }
  void StoreUintParams(const std::vector<uint_t>& prm)
  {
    if(cache_){
      cache_->StoreUintParams(prm);
    }
    else{
      chunk_container_.lock()->StoreUintParams(prm,chunk_pos_);
    }
  }

  void CopyIn(std::shared_ptr<Chunk<data_t>> src)
  {
    chunk_container_.lock()->CopyIn(src,chunk_pos_);
  }
  void CopyOut(std::shared_ptr<Chunk<data_t>> dest)
  {
    chunk_container_.lock()->CopyOut(dest,chunk_pos_);
  }
  void CopyIn(thrust::complex<data_t>* src, uint_t size)
  {
    chunk_container_.lock()->CopyIn(src, chunk_pos_, size);
  }
  void CopyOut(thrust::complex<data_t>* dest, uint_t size)
  {
    chunk_container_.lock()->CopyOut(dest, chunk_pos_, size);
  }
  void Swap(std::shared_ptr<Chunk<data_t>> src)
  {
    chunk_container_.lock()->Swap(src,chunk_pos_);
  }

  template <typename Function>
  void Execute(Function func,uint_t count)
  {
    if(cache_){
      cache_->Execute(func,count);
    }
    else{
      chunk_container_.lock()->Execute(func,chunk_pos_,count);
    }
  }

  template <typename Function>
  double ExecuteSum(Function func,uint_t count) const
  {
    if(cache_){
      return cache_->ExecuteSum(func,count);
    }
    else{
      return chunk_container_.lock()->ExecuteSum(func,chunk_pos_,count);
    }
  }

  void Zero(void)
  {
    auto sel_chunk_container = chunk_container_.lock();
    sel_chunk_container->Zero(chunk_pos_,sel_chunk_container->chunk_size());
  }

  reg_t sample_measure(const std::vector<double> &rnds,uint_t stride = 1,bool dot = true) const
  {
    return chunk_container_.lock()->sample_measure(chunk_pos_,rnds,stride,dot);
  }

  thrust::complex<double> norm(uint_t stride = 1,bool dot = true) const
  {
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

  void synchronize(void)
  {
    if(cache_){
      cache_->synchronize();
    }
    else{
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

};

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
