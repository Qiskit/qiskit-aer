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


#ifndef _qv_chunk_container_hpp_
#define _qv_chunk_container_hpp_

#ifdef AER_THRUST_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <thrust/for_each.h>
#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/detail/vector_base.h>

#ifdef AER_THRUST_CUDA
#include <thrust/device_vector.h>
#endif
#include <thrust/host_vector.h>

#include <thrust/system/omp/execution_policy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define AER_DEFAULT_MATRIX_BITS   5

#define AER_CHUNK_BITS        21
#define AER_MAX_BUFFERS       4
#define AER_DUMMY_BUFFERS     4     //reserved storage for parameters

#ifdef AER_THRUST_CUDA
#define AERDeviceVector thrust::device_vector
#else
#define AERDeviceVector thrust::host_vector
#endif
#define AERHostVector thrust::host_vector

#include "framework/utils.hpp"

namespace AER {
namespace QV {

template <typename data_t> class Chunk;
template <typename data_t> class DeviceChunkContainer;
template <typename data_t> class HostChunkContainer;

//========================================
//  base class of gate functions
//========================================
template <typename data_t>
class GateFuncBase
{
protected:
  thrust::complex<data_t>* data_;   //pointer to state vector buffer
  thrust::complex<double>* matrix_;   //storage for matrix on device
  uint_t* params_;                    //storage for additional parameters on device
  uint_t base_index_;               //start index of state vector 
public:
  GateFuncBase(void)
  {
    data_ = NULL;
    base_index_ = 0;
  }

  virtual void set_data(thrust::complex<data_t>* p)
  {
    data_ = p;
  }
  void set_matrix(thrust::complex<double>* mat)
  {
    matrix_ = mat;
  }
  void set_params(uint_t* p)
  {
    params_ = p;
  }
  void set_base_index(uint_t i)
  {
    base_index_ = i;
  }

  virtual bool is_diagonal(void)
  {
    return false;
  }
  virtual int num_control_bits(void)
  {
    return 0;
  }
  virtual int control_mask(void)
  {
    return 1;
  }

  virtual const char* name(void)
  {
    return "base function";
  }
  virtual uint_t size(int num_qubits,int n)
  {
    if(is_diagonal()){
      return (1ull << num_qubits);
    }
    else{
      return (1ull << (num_qubits - (n-num_control_bits())));
    }
  }
};

//============================================================================
// chunk container base class
//============================================================================
template <typename data_t>
class ChunkContainer 
{
protected:
  int chunk_bits_;                    //number of qubits for a chunk
  int place_id_;                      //index of a container (device index + host)
  uint_t num_chunks_;                 //number of chunks in this container
  uint_t num_buffers_;                //number of buffers (buffer chunks) in this container
  uint_t num_checkpoint_;             //number of checkpoint buffers in this container
  uint_t num_chunk_mapped_;                 //number of chunks mapped
  std::vector<bool> chunk_mapped_;    //which chunk is mapped
  std::vector<bool> buffer_mapped_;   //which buffer is mapped
  std::vector<bool> checkpoint_mapped_;   //which checkpoint buffer is mapped
public:
  ChunkContainer()
  {
    chunk_bits_ = 0;
    place_id_ = 0;
    num_chunks_ = 0;
    num_buffers_ = 0;
    num_checkpoint_ = 0;
    num_chunk_mapped_ = 0;
  }
  virtual ~ChunkContainer()
  {
    chunk_mapped_.clear();
    buffer_mapped_.clear();
    checkpoint_mapped_.clear();
  }

  int chunk_bits(void)
  {
    return chunk_bits_;
  }
  int place(void)
  {
    return place_id_;
  }
  void set_place(int id)
  {
    place_id_ = id;
  }
  uint_t num_chunks(void)
  {
    return num_chunks_;
  }
  uint_t num_buffers(void)
  {
    return num_buffers_;
  }
  uint_t num_checkpoint(void)
  {
    return num_checkpoint_;
  }
  uint_t chunk_size(void)
  {
    return (1ull << chunk_bits_);
  }
  uint_t num_chunk_mapped(void)
  {
    return num_chunk_mapped_;
  }

  virtual void set_device(void) const
  {
  }

  virtual uint_t size(void) = 0;
  virtual int device(void)
  {
    return -1;
  }

  virtual bool peer_access(int i_dest)
  {
    return false;
  }

  virtual thrust::complex<data_t>& operator[](uint_t i) = 0;

  virtual uint_t Allocate(int idev,int bits,uint_t chunks,uint_t buffers = AER_MAX_BUFFERS,uint_t checkpoint = 0) = 0;
  virtual void Deallocate(void) = 0;
  virtual uint_t Resize(uint_t chunks,uint_t buffers = AER_MAX_BUFFERS,uint_t checkpoint = 0) = 0;

  virtual void Set(uint_t i,const thrust::complex<data_t>& t) = 0;
  virtual thrust::complex<data_t> Get(uint_t i) const = 0;

  virtual thrust::complex<data_t>* chunk_pointer(void) = 0;

  virtual void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk) = 0;
  virtual void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk) = 0;

  virtual void CopyIn(Chunk<data_t>* src,uint_t iChunk) = 0;
  virtual void CopyOut(Chunk<data_t>* dest,uint_t iChunk) = 0;
  virtual void CopyIn(thrust::complex<data_t>* src,uint_t iChunk) = 0;
  virtual void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk) = 0;
  virtual void Swap(Chunk<data_t>* src,uint_t iChunk) = 0;

  virtual void Zero(uint_t iChunk,uint_t count) = 0;

  virtual reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds) const = 0;

  size_t size_of_complex(void)
  {
    return sizeof(thrust::complex<data_t>);
  }

  Chunk<data_t>* MapChunk(void);
  Chunk<data_t>* MapBufferChunk(void);
  Chunk<data_t>* MapCheckpoint(int_t iChunk = -1);
  void UnmapChunk(Chunk<data_t>* chunk);
  void UnmapBuffer(Chunk<data_t>* buf);
  void UnmapCheckpoint(Chunk<data_t>* buf);

  virtual thrust::complex<double>* matrix_pointer(uint_t iChunk)
  {
    return NULL;
  }
  virtual uint_t* param_pointer(uint_t iChunk)
  {
    return NULL;
  }
  virtual int matrix_bits(void)
  {
    return 0;
  }

};

template <typename data_t>
Chunk<data_t>* ChunkContainer<data_t>::MapChunk(void)
{
  uint_t i,pos,idx;
  pos = num_chunks_;

#pragma omp critical
  {
    for(i=0;i<num_chunks_;i++){
      idx = (num_chunk_mapped_ + i) % num_chunks_;
      if(!chunk_mapped_[idx]){
        chunk_mapped_[idx] = true;
        pos = idx;
        num_chunk_mapped_++;
        break;
      }
    }
  }

  if(pos < num_chunks_)
    return new Chunk<data_t>(this,pos);
  return NULL;
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapChunk(Chunk<data_t>* chunk)
{
  chunk_mapped_[chunk->pos()] = false;
  num_chunk_mapped_--;
  delete chunk;
}

template <typename data_t>
Chunk<data_t>* ChunkContainer<data_t>::MapBufferChunk(void)
{
  uint_t i,pos;
  pos = num_buffers_;
#pragma omp critical
  {
    for(i=0;i<num_buffers_;i++){
      if(!buffer_mapped_[i]){
        buffer_mapped_[i] = true;
        pos = i;
        break;
      }
    }
  }

  if(pos < num_buffers_)
    return new Chunk<data_t>(this,num_chunks_+pos);
  return NULL;
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapBuffer(Chunk<data_t>* buf)
{
#pragma omp critical
  {
    buffer_mapped_[buf->pos()-num_chunks_] = false;
  }
  delete buf;
}

template <typename data_t>
Chunk<data_t>* ChunkContainer<data_t>::MapCheckpoint(int_t iChunk)
{
  if(iChunk >= 0 && num_checkpoint_ == num_chunks_){   //checkpoint buffers are reserved for all chunks
    if(iChunk < num_checkpoint_)
      return new Chunk<data_t>(this,num_chunks_+num_buffers_+iChunk);
    return NULL;
  }
  else{
    uint_t i,pos;
    pos = num_checkpoint_;
#pragma omp critical
    {
      for(i=0;i<num_checkpoint_;i++){
        if(!checkpoint_mapped_[i]){
          checkpoint_mapped_[i] = true;
          pos = i;
          break;
        }
      }
    }

    if(pos < num_checkpoint_)
      return new Chunk<data_t>(this,num_chunks_+num_buffers_+pos);
    return NULL;
  }
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapCheckpoint(Chunk<data_t>* buf)
{
  if(num_checkpoint_ != num_chunks_){
#pragma omp critical
    {
      checkpoint_mapped_[buf->pos()-num_chunks_-num_buffers_] = false;
    }
  }
  delete buf;
}


//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
