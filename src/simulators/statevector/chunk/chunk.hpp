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

namespace QV {




//============================================================================
// chunk class
//============================================================================
template <typename data_t>
class Chunk 
{
protected:
  mutable ChunkContainer<data_t>* chunk_container_;   //pointer to chunk container
  Chunk<data_t>* cache_;                //pointer to cache chunk on device
  uint_t chunk_pos_;                    //position in container
  int place_;                           //container ID
public:
  Chunk(ChunkContainer<data_t>* cc,uint_t pos)
  {
    chunk_container_ = cc;
    chunk_pos_ = pos;
    place_ = 0;
    cache_ = NULL;
  }
  ~Chunk()
  {
  }

  void set_device(void) const
  {
    chunk_container_->set_device();
  }
  int device(void)
  {
    return chunk_container_->device();
  }

  ChunkContainer<data_t>* container(void)
  {
    return chunk_container_;
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
  void set_cache(Chunk<data_t>* c)
  {
    cache_ = c;
  }

  void Set(uint_t i,const thrust::complex<data_t>& t)
  {
    chunk_container_->Set(i + (chunk_pos_ << chunk_container_->chunk_bits()),t);
  }
  thrust::complex<data_t> Get(uint_t i) const
  {
    return chunk_container_->Get(i + (chunk_pos_ << chunk_container_->chunk_bits()));
  }

  thrust::complex<data_t>& operator[](uint_t i)
  {
    return (*chunk_container_)[i + (chunk_pos_ << chunk_container_->chunk_bits())];
  }

  thrust::complex<data_t>* pointer(void)
  {
    return chunk_container_->chunk_pointer() + (chunk_pos_ << chunk_container_->chunk_bits());
  }

  void StoreMatrix(const std::vector<std::complex<double>>& mat)
  {
    if(cache_){
      cache_->StoreMatrix(mat);
    }
    else{
      chunk_container_->StoreMatrix(mat,chunk_pos_);
    }
  }
  void StoreUintParams(const std::vector<uint_t>& prm)
  {
    if(cache_){
      cache_->StoreUintParams(prm);
    }
    else{
      chunk_container_->StoreUintParams(prm,chunk_pos_);
    }
  }

  void CopyIn(Chunk<data_t>* src)
  {
    chunk_container_->CopyIn(src,chunk_pos_);
  }
  void CopyOut(Chunk<data_t>* dest)
  {
    chunk_container_->CopyOut(dest,chunk_pos_);
  }
  void CopyIn(thrust::complex<data_t>* src)
  {
    chunk_container_->CopyIn(src,chunk_pos_);
  }
  void CopyOut(thrust::complex<data_t>* dest)
  {
    chunk_container_->CopyOut(dest,chunk_pos_);
  }
  void Swap(Chunk<data_t>* src)
  {
    chunk_container_->Swap(src,chunk_pos_);
  }

  template <typename Function>
  void Execute(Function func,uint_t count)
  {
    if(cache_){
      cache_->Execute(func,count);
    }
    else{
      if(chunk_container_->device() >= 0)
        ((DeviceChunkContainer<data_t>*)chunk_container_)->Execute(func,chunk_pos_,count);
      else
        ((HostChunkContainer<data_t>*)chunk_container_)->Execute(func,chunk_pos_,count);
    }
  }
  template <typename Function>
  double ExecuteSum(Function func,uint_t count) const
  {
    if(cache_){
      return cache_->ExecuteSum(func,count);
    }
    else{
      if(chunk_container_->device() >= 0)
        return ((DeviceChunkContainer<data_t>*)chunk_container_)->ExecuteSum(func,chunk_pos_,count);
      else
        return ((HostChunkContainer<data_t>*)chunk_container_)->ExecuteSum(func,chunk_pos_,count);
    }
  }
  template <typename Function>
  thrust::complex<double> ExecuteComplexSum(Function func,uint_t count) const
  {
    if(cache_){
      return cache_->ExecuteComplexSum(func,count);
    }
    else{
      if(chunk_container_->device() >= 0)
        return ((DeviceChunkContainer<data_t>*)chunk_container_)->ExecuteComplexSum(func,chunk_pos_,count);
      else
        return ((HostChunkContainer<data_t>*)chunk_container_)->ExecuteComplexSum(func,chunk_pos_,count);
    }
  }
  void Zero(void)
  {
    chunk_container_->Zero(chunk_pos_,chunk_container_->chunk_size());
  }

  reg_t sample_measure(const std::vector<double> &rnds) const
  {
    return chunk_container_->sample_measure(chunk_pos_,rnds);
  }

#ifdef AER_THRUST_CUDA
  cudaStream_t stream(void)
  {
    return ((DeviceChunkContainer<data_t>*)chunk_container_)->stream(chunk_pos_);
  }
#endif

  thrust::complex<double>* matrix_pointer(void)
  {
    return chunk_container_->matrix_pointer(chunk_pos_);
  }
  uint_t* param_pointer(void)
  {
    return chunk_container_->param_pointer(chunk_pos_);
  }
  int matrix_bits(void)
  {
    return chunk_container_->matrix_bits();
  }
};


}

//------------------------------------------------------------------------------
#endif // end module
