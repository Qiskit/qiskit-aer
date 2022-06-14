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

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#ifdef AER_THRUST_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
DISABLE_WARNING_POP

#include "misc/wrap_thrust.hpp"

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

#define AER_DEFAULT_MATRIX_BITS   6

#define AER_CHUNK_BITS        21
#define AER_MAX_BUFFERS       1
#define AER_DUMMY_BUFFERS     4     //reserved storage for parameters

#define QV_CUDA_NUM_THREADS 1024
#define QV_MAX_REGISTERS 10
#define QV_MAX_BLOCKED_GATES 64

#define QV_PROBABILITY_BUFFER_SIZE 4
#define QV_NUM_INTERNAL_REGS 4

#ifdef AER_THRUST_CUDA
#define AERDeviceVector thrust::device_vector
#else
#define AERDeviceVector thrust::host_vector
#endif
#define AERHostVector thrust::host_vector

#include "framework/utils.hpp"

#ifdef AER_THRUST_CUDA
#include "simulators/statevector/chunk/cuda_kernels.hpp"
#endif

#include "simulators/statevector/chunk/thrust_kernels.hpp"

namespace AER {
namespace QV {
namespace Chunk {

template <typename data_t> class Chunk;
template <typename data_t> class DeviceChunkContainer;
template <typename data_t> class HostChunkContainer;

struct BlockedGateParams
{
  uint_t mask_;
  char gate_;
  unsigned char qubit_;
};


//============================================================================
// chunk container base class
//============================================================================
template <typename data_t>
class ChunkContainer : public std::enable_shared_from_this<ChunkContainer<data_t>>
{
protected:
  int_t chunk_bits_;                  //number of qubits for a chunk
  int_t num_qubits_;                  //total qubits
  int_t place_id_;                    //index of a container (device index + host)
  int_t num_places_;
  uint_t num_chunks_;                 //number of chunks in this container
  uint_t chunk_index_;                //global chunk index for the first chunk in this container
  uint_t num_buffers_;                //number of buffers (buffer chunks) in this container
  uint_t num_chunk_mapped_;           //number of chunks mapped
  reg_t blocked_qubits_;
  std::vector<bool> chunks_map_;      //chunk mapper
  mutable reg_t reduced_queue_begin_;
  mutable reg_t reduced_queue_end_;
  uint_t matrix_bits_;                //max matrix bits
  uint_t num_creg_bits_;              //number of cregs
  uint_t num_cregisters_;
  uint_t num_cmemory_;
  mutable int_t conditional_bit_;
  bool keep_conditional_bit_;         //keep conditional bit alive
  int_t num_pow2_qubits_;             //largest number of qubits that meets num_chunks_ = m*(2^num_pow2_qubits_)
  bool density_matrix_;

  int_t omp_threads_;                 //number of threads can be used for parallelization on CPU
public:
  ChunkContainer()
  {
    chunk_bits_ = 0;
    place_id_ = 0;
    num_chunks_ = 0;
    chunk_index_ = 0;
    num_buffers_ = 0;
    num_chunk_mapped_ = 0;
    conditional_bit_ = -1;
    keep_conditional_bit_ = false;
    matrix_bits_ = AER_DEFAULT_MATRIX_BITS;
    density_matrix_ = false;
    omp_threads_ = 1;
  }
  virtual ~ChunkContainer(){}

  int_t chunk_bits(void)
  {
    return chunk_bits_;
  }
  int_t place(void)
  {
    return place_id_;
  }
  void set_place(int_t id,int_t n)
  {
    place_id_ = id;
    num_places_ = n;
  }
  uint_t num_chunks(void)
  {
    return num_chunks_;
  }
  uint_t num_buffers(void)
  {
    return num_buffers_;
  }
  uint_t chunk_size(void)
  {
    return (1ull << chunk_bits_);
  }
  uint_t num_chunk_mapped(void)
  {
    return num_chunk_mapped_;
  }
  uint_t matrix_bits(void)
  {
    return matrix_bits_;
  }

  virtual void set_device(void) const
  {
  }

#ifdef AER_THRUST_CUDA
  virtual cudaStream_t stream(uint_t iChunk) const
  {
    return nullptr;
  }
#endif

  virtual uint_t size(void) = 0;
  virtual int device(void)
  {
    return -1;
  }

  virtual bool peer_access(int i_dest)
  {
    return false;
  }

  void set_conditional(int_t reg)
  {
    conditional_bit_ = reg;
  }
  int_t get_conditional(void)
  {
    return conditional_bit_;
  }
  void keep_conditional(bool keep)
  {
    keep_conditional_bit_ = keep;
  }

  void set_chunk_index(uint_t chunk_index)
  {
    chunk_index_ = chunk_index;
  }

  void set_omp_threads(int_t nthreads)
  {
    omp_threads_ = nthreads;
  }

  virtual thrust::complex<data_t>& operator[](uint_t i) = 0;

  virtual uint_t Allocate(int idev,int chunk_bits,int num_qubits,uint_t chunks,uint_t buffers = AER_MAX_BUFFERS,bool multi_shots = false,int matrix_bit = AER_DEFAULT_MATRIX_BITS, bool density_matrix = false) = 0;
  virtual void Deallocate(void) = 0;

  virtual void Set(uint_t i,const thrust::complex<data_t>& t) = 0;
  virtual thrust::complex<data_t> Get(uint_t i) const = 0;

  virtual void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk) = 0;
  virtual void StoreMatrix(const std::complex<double>* mat,uint_t iChunk,uint_t size) = 0;
  virtual void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk) = 0;
  virtual void ResizeMatrixBuffers(int bits) = 0;

  virtual void CopyIn(Chunk<data_t>& src,uint_t iChunk) = 0;
  virtual void CopyOut(Chunk<data_t>& dest,uint_t iChunk) = 0;
  virtual void CopyIn(thrust::complex<data_t>* src,uint_t iChunk, uint_t size) = 0;
  virtual void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk, uint_t size) = 0;
  virtual void Swap(Chunk<data_t>& src,uint_t iChunk, uint_t dest_offset = 0, uint_t src_offset = 0, uint_t size = 0, bool write_back = true) = 0;

  virtual void Zero(uint_t iChunk,uint_t count) = 0;

  template <typename Function>
  void Execute(Function func,uint_t iChunk,const uint_t gid, const uint_t count);

  template <typename Function>
  void ExecuteSum(double* pSum,Function func,uint_t iChunk,uint_t count) const;

  template <typename Function>
  void ExecuteSum2(double* pSum,Function func,uint_t iChunk,uint_t count) const;

  virtual reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride = 1, bool dot = true,uint_t count = 1) const = 0;
  virtual double norm(uint_t iChunk,uint_t count) const;
  virtual double trace(uint_t iChunk,uint_t row,uint_t count) const;


  size_t size_of_complex(void)
  {
    return sizeof(thrust::complex<data_t>);
  }

  bool MapChunk(Chunk<data_t>& chunk);
  bool MapBufferChunk(Chunk<data_t>& chunk);
  void UnmapChunk(Chunk<data_t>& chunk);
  void UnmapBuffer(Chunk<data_t>& chunk);
  void unmap_all(void);

  virtual thrust::complex<data_t>* chunk_pointer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual thrust::complex<data_t>* buffer_pointer(void) const
  {
    return NULL;
  }
  virtual thrust::complex<double>* matrix_pointer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual uint_t* param_pointer(uint_t iChunk) const
  {
    return NULL;
  }


  virtual void synchronize(uint_t iChunk)
  {
    ;
  }

  //set qubits to be blocked
  virtual void set_blocked_qubits(uint_t iChunk,const reg_t& qubits)
  {
    ;
  }

  //do all gates stored in queue
  virtual void apply_blocked_gates(uint_t iChunk)
  {
    ;
  }

  //queue gate for blocked execution
  virtual void queue_blocked_gate(uint_t iChunk,char gate,uint_t qubit,uint_t mask,const std::complex<double>* pMat = NULL)
  {
    ;
  }

  virtual double* reduce_buffer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual uint_t reduce_buffer_size() const
  {
    return 1;
  }
  virtual double* condition_buffer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual double* probability_buffer(uint_t iChunk) const
  {
    return NULL;
  }

  virtual void copy_to_probability_buffer(std::vector<double>& buf,int pos){}

  //classical register to store measured bits/used for bfunc operations
  virtual void allocate_creg(uint_t num_mem,uint_t num_reg){}
  virtual int measured_cbit(uint_t iChunk,int qubit)
  {
    return 0;
  }

  virtual uint_t* creg_buffer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual void request_creg_update(void){}

  //apply matrix 
  virtual void apply_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t gid, const uint_t count);

  //apply diagonal matrix
  virtual void apply_diagonal_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t gid, const uint_t count);

  //apply (controlled) X
  virtual void apply_X(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count);

  //apply (controlled) Y
  virtual void apply_Y(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count);

  //apply (controlled) phase
  virtual void apply_phase(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t gid, const uint_t count);

  //apply (controlled) swap gate
  virtual void apply_swap(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const uint_t gid, const uint_t count);

  //apply multiple swap gates
  virtual void apply_multi_swaps(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count);

  //apply permutation
  virtual void apply_permutation(const uint_t iChunk,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t gid, const uint_t count);

  //apply rotation around axis
  virtual void apply_rotation(const uint_t iChunk,const reg_t &qubits, const Rotation r, const double theta, const uint_t gid, const uint_t count);

  //get probabilities of chunk
  virtual void probabilities(std::vector<double>& probs, const uint_t iChunk, const reg_t& qubits) const;

  //Pauli expectation values
  virtual double expval_pauli(const uint_t iChunk,const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const;

protected:
  int convert_blocked_qubit(int qubit)
  {
    int i;
    for(i=0;i<blocked_qubits_.size();i++){
      if(blocked_qubits_[i] == qubit){
        return i;
      }
    }
    return -1;
  }


  //allocate storage for chunk classes
  void allocate_chunks(void);
  void deallocate_chunks(void);
};

template <typename data_t>
bool ChunkContainer<data_t>::MapChunk(Chunk<data_t>& chunk)
{
  uint_t i,idx;

  for(i=0;i<num_chunks_;i++){
    idx = (num_chunk_mapped_ + i) % num_chunks_;
    if(!chunks_map_[idx]){
      chunks_map_[idx] = true;
      num_chunk_mapped_++;
      chunk.map(this->shared_from_this(),idx);
      break;
    }
  }
  return chunk.is_mapped();
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapChunk(Chunk<data_t>& chunk)
{
  chunks_map_[chunk.pos()] = false;
  chunk.unmap();
}

template <typename data_t>
bool ChunkContainer<data_t>::MapBufferChunk(Chunk<data_t>& chunk)
{
  chunk.map(this->shared_from_this(),num_chunks_);
  return chunk.is_mapped();
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapBuffer(Chunk<data_t>& buf)
{
  buf.unmap();
}

template <typename data_t>
void ChunkContainer<data_t>::unmap_all(void)
{
  int_t i;
  for(i=0;i<chunks_map_.size();i++)
    chunks_map_[i] = false;
  num_chunk_mapped_ = 0;
}

template <typename data_t>
template <typename Function>
void ChunkContainer<data_t>::Execute(Function func,uint_t iChunk,const uint_t gid, const uint_t count)
{
  set_device();

  func.set_base_index(gid << chunk_bits_);
  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_cregs_(creg_buffer(iChunk),num_creg_bits_);

  if(iChunk == 0 && conditional_bit_ >= 0){
    func.set_conditional(conditional_bit_);
    if(!keep_conditional_bit_)
      conditional_bit_ = -1;  //reset conditional
  }

#ifdef AER_THRUST_CUDA
  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t nt,nb;
    nb = 1;

    if(func.use_cache()){
      func.set_chunk_bits(chunk_bits_);

      nt = count << chunk_bits_;
      if(nt > 0){
        uint_t ntotal = nt;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dev_apply_function_with_cache<data_t,Function><<<nb,nt,0,strm>>>(func,ntotal);
      }
    }
    else{
      nt = count * func.size(chunk_bits_);

      if(nt > 0){
        uint_t ntotal = nt;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dev_apply_function<data_t,Function><<<nb,nt,0,strm>>>(func,ntotal);
      }
    }
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      std::stringstream str;
      str << "ChunkContainer::Execute in " << func.name() << " : " << cudaGetErrorName(err);
      throw std::runtime_error(str.str());
    }
  }
  else{ //if no stream returned, run on host
    uint_t size = count * func.size(chunk_bits_);
    auto ci = thrust::counting_iterator<uint_t>(0);
    thrust::for_each_n(thrust::seq, ci , size, func);
  }
#else
  uint_t size;
  if(func.use_cache())
    size = count << (chunk_bits_ - func.qubits_count());
  else
    size = count * func.size(chunk_bits_);
  auto ci = thrust::counting_iterator<uint_t>(0);
  if(omp_threads_ > 1)
    thrust::for_each_n(thrust::device, ci , size, func);
  else
    thrust::for_each_n(thrust::seq, ci , size, func);
#endif

}

template <typename data_t>
template <typename Function>
void ChunkContainer<data_t>::ExecuteSum(double* pSum,Function func,uint_t iChunk,uint_t count) const
{

#ifdef AER_THRUST_CUDA
  uint_t size = count * func.size(chunk_bits_);

  set_device();

  func.set_base_index((chunk_index_ + iChunk) << chunk_bits_);
  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_cregs_(creg_buffer(iChunk),num_creg_bits_);

  auto ci = thrust::counting_iterator<uint_t>(0);

  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t buf_size;
    double* buf = reduce_buffer(iChunk);

    if(pSum){   //sum for all chunks are gathered and stored to pSum
      buf_size = 0;
      uint_t n,nt,nb;
      nb = 1;

      if(func.use_cache()){
        nt = count << chunk_bits_;
        if(nt > 0){
          uint_t ntotal = nt;
          if(nt > QV_CUDA_NUM_THREADS){
            nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
            nt = QV_CUDA_NUM_THREADS;
          }
          dev_apply_function_sum_with_cache<data_t,Function><<<nb,nt,0,strm>>>(buf,func,buf_size,ntotal);
        }
      }
      else{
        nt = size;
        if(nt > 0){
          uint_t ntotal = nt;
          if(nt > QV_CUDA_NUM_THREADS){
            nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
            nt = QV_CUDA_NUM_THREADS;
          }
          dev_apply_function_sum<data_t,Function><<<nb,nt,0,strm>>>(buf,func,buf_size,ntotal);
        }
      }
      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess){
        std::stringstream str;
        str << "ChunkContainer::ExecuteSum in " << func.name() << " : " << cudaGetErrorName(err);
        throw std::runtime_error(str.str());
      }

      while(nb > 1){
        n = nb;
        nt = nb;
        nb = 1;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dev_reduce_sum<<<nb,nt,0,strm>>>(buf,n,buf_size);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
          std::stringstream str;
          str << "ChunkContainer::ExecuteSum in " << func.name() << " :: " << cudaGetErrorName(err);
          throw std::runtime_error(str.str());
        }
      }
      cudaMemcpyAsync(pSum,buf,sizeof(double),cudaMemcpyDeviceToHost,strm);
    }
    else{
      buf_size = reduce_buffer_size();

      uint_t n,nt,nb;
      nb = 1;

      if(func.use_cache()){
        nt = 1ull << chunk_bits_;
        if(nt > 0){
          uint_t ntotal = nt*count;
          if(nt > QV_CUDA_NUM_THREADS){
            nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
            nt = QV_CUDA_NUM_THREADS;
          }
          dim3 grid(nb,count,1);
          dev_apply_function_sum_with_cache<data_t,Function><<<grid,nt,0,strm>>>(buf,func,buf_size,ntotal);
        }
      }
      else{
        nt = func.size(chunk_bits_);
        if(nt > 0){
          uint_t ntotal = nt*count;
          if(nt > QV_CUDA_NUM_THREADS){
            nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
            nt = QV_CUDA_NUM_THREADS;
          }
          dim3 grid(nb,count,1);
          dev_apply_function_sum<data_t,Function><<<grid,nt,0,strm>>>(buf,func,buf_size,ntotal);
        }
      }
      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess){
        std::stringstream str;
        str << "ChunkContainer::ExecuteSum in " << func.name() << " : " << cudaGetErrorName(err);
        throw std::runtime_error(str.str());
      }

      while(nb > 1){
        n = nb;
        nt = nb;
        nb = 1;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dim3 grid(nb,count,1);
        dev_reduce_sum<<<grid,nt,0,strm>>>(buf,n,buf_size);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
          std::stringstream str;
          str << "ChunkContainer::ExecuteSum in " << func.name() << " :: " << cudaGetErrorName(err);
          throw std::runtime_error(str.str());
        }
      }
    }
  }
  else{ //if no stream returned, run on host
    *pSum = thrust::transform_reduce(thrust::seq, ci, ci + size, func,0.0,thrust::plus<double>());
  }
#else
  uint_t size = func.size(chunk_bits_);

  func.set_base_index((chunk_index_ + iChunk) << chunk_bits_);
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );

  uint_t i;
  for(i=0;i<count;i++){

    func.set_data( chunk_pointer(iChunk + i) );
    func.set_index_offset(iChunk << chunk_bits_);

    auto ci = thrust::counting_iterator<uint_t>(0);

    double sum;
    if(omp_threads_ > 1)
      sum = thrust::transform_reduce(thrust::device, ci, ci + size, func,0.0,thrust::plus<double>());
    else
      sum = thrust::transform_reduce(thrust::seq, ci, ci + size, func,0.0,thrust::plus<double>());
    if(count == 1 && pSum){
      *pSum = sum;
    }
    else{
      *(reduce_buffer(iChunk + i)) = sum;
    }
  }
#endif
}

struct complex_sum
{
  __host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& a,const thrust::complex<double>& b)
  {
    return a+b;
  }
};

template <typename data_t>
template <typename Function>
void ChunkContainer<data_t>::ExecuteSum2(double* pSum,Function func,uint_t iChunk,uint_t count) const
{

#ifdef AER_THRUST_CUDA
  uint_t size = count * func.size(chunk_bits_);

  set_device();

  func.set_base_index((chunk_index_ + iChunk) << chunk_bits_);
  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_cregs_(creg_buffer(iChunk),num_creg_bits_);

  auto ci = thrust::counting_iterator<uint_t>(0);

  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t buf_size;
    uint_t n,nt,nb;
    nb = 1;
    nt = func.size(chunk_bits_);
    thrust::complex<double>* buf = (thrust::complex<double>*)reduce_buffer(iChunk);

    if(pSum){   //sum for all chunks are gathered and stored to pSum
      buf_size = 0;
      nt = size;
      if(nt > 0){
        uint_t ntotal = nt;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dev_apply_function_sum_complex<data_t,Function><<<nb,nt,0,strm>>>(buf,func,buf_size,ntotal);
      }
      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess){
        std::stringstream str;
        str << "ChunkContainer::ExecuteSum2 in " << func.name() << " : " << cudaGetErrorName(err);
        throw std::runtime_error(str.str());
      }

      while(nb > 1){
        n = nb;
        nt = nb;
        nb = 1;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dev_reduce_sum_complex<<<nb,nt,0,strm>>>(buf,n,buf_size);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
          std::stringstream str;
          str << "ChunkContainer::ExecuteSum2 in " << func.name() << " :: " << cudaGetErrorName(err);
          throw std::runtime_error(str.str());
        }
      }
      cudaMemcpyAsync(pSum,buf,sizeof(double)*2,cudaMemcpyDeviceToHost,strm);
    }
    else{
      buf_size = reduce_buffer_size()/2;

      if(nt > 0){
        uint_t ntotal = nt*count;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dim3 grid(nb,count,1);
        dev_apply_function_sum_complex<data_t,Function><<<grid,nt,0,strm>>>(buf,func,buf_size,ntotal);
      }
      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess){
        std::stringstream str;
        str << "ChunkContainer::ExecuteSum2 in " << func.name() << " : " << cudaGetErrorName(err);
        throw std::runtime_error(str.str());
      }

      while(nb > 1){
        n = nb;
        nt = nb;
        nb = 1;
        if(nt > QV_CUDA_NUM_THREADS){
          nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
          nt = QV_CUDA_NUM_THREADS;
        }
        dim3 grid(nb,count,1);
        dev_reduce_sum_complex<<<grid,nt,0,strm>>>(buf,n,buf_size);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
          std::stringstream str;
          str << "ChunkContainer::ExecuteSum2 in " << func.name() << " :: " << cudaGetErrorName(err);
          throw std::runtime_error(str.str());
        }
      }
    }
  }
  else{ //if no stream returned, run on host
    thrust::complex<double> ret,zero = 0.0;
    ret = thrust::transform_reduce(thrust::seq, ci, ci + size, func,zero,complex_sum());
    *((thrust::complex<double>*)pSum) = ret;
  }
#else
  uint_t size = func.size(chunk_bits_);

  func.set_base_index((chunk_index_ + iChunk) << chunk_bits_);
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
 
  uint_t i;
  for(i=0;i<count;i++){
    thrust::complex<double> ret,zero = 0.0;
    func.set_data( chunk_pointer(iChunk + i) );
    func.set_index_offset(iChunk << chunk_bits_);

    auto ci = thrust::counting_iterator<uint_t>(0);

    if(omp_threads_ > 1)
      ret = thrust::transform_reduce(thrust::device, ci, ci + size, func,zero,complex_sum());
    else
      ret = thrust::transform_reduce(thrust::seq, ci, ci + size, func,zero,complex_sum());

    if(count == 1 && pSum){
      *((thrust::complex<double>*)pSum) = ret;
    }
    else{
      *((thrust::complex<double>*)reduce_buffer(iChunk + i)) = ret;
    }
  }
#endif
}


void host_func_launcher(void* pParam)
{
  HostFuncBase* func = reinterpret_cast<HostFuncBase*>(pParam);
  func->execute();
}

template <typename data_t>
void ChunkContainer<data_t>::allocate_chunks(void)
{
  uint_t i;
  chunks_map_.resize(num_chunks_,false);

  reduced_queue_begin_.resize(num_chunks_,0);
  reduced_queue_end_.resize(num_chunks_,0);
}

template <typename data_t>
void ChunkContainer<data_t>::deallocate_chunks(void)
{
  chunks_map_.clear();

  reduced_queue_begin_.clear();
  reduced_queue_end_.clear();
}

template <typename data_t>
void ChunkContainer<data_t>::apply_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t gid, const uint_t count)
{
  const size_t N = qubits.size() - control_bits;

  if(N == 1){
    if(control_bits == 0)
      Execute(MatrixMult2x2<data_t>(mat,qubits[0]), iChunk, gid, count);
    else  //2x2 matrix with control bits
      Execute(MatrixMult2x2Controlled<data_t>(mat,qubits), iChunk, gid, count);
  }
  else if(N == 2){
    Execute(MatrixMult4x4<data_t>(mat,qubits[0],qubits[1]), iChunk, gid, count);
  }
  else{
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());
#ifndef AER_THRUST_CUDA
    if(N == 3){
      StoreMatrix(mat, iChunk);
      Execute(MatrixMult8x8<data_t>(qubits,qubits_sorted), iChunk, gid, count);
    }
    else if(N == 4){
      StoreMatrix(mat, iChunk);
      Execute(MatrixMult16x16<data_t>(qubits,qubits_sorted), iChunk, gid, count);
    }
    else if(N <= 10){
#else
    if(N <= 10){
#endif
      int i;
      for(i=0;i<N;i++){
        qubits_sorted.push_back(qubits[i]);
      }
      StoreMatrix(mat, iChunk);
      StoreUintParams(qubits_sorted, iChunk);

      Execute(MatrixMultNxN<data_t>(N), iChunk, gid, count);
    }
    else{
      cvector_t<double> matLU;
      reg_t params;
      MatrixMultNxN_LU<data_t> f(mat,qubits_sorted,matLU,params);

      StoreMatrix(matLU, iChunk);
      StoreUintParams(params, iChunk);

      Execute(f, iChunk, gid, count);
    }
  }
}

template <typename data_t>
void ChunkContainer<data_t>::apply_diagonal_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t gid, const uint_t count)
{
  const size_t N = qubits.size() - control_bits;

  if(N == 1){
    if(control_bits == 0)
      Execute(DiagonalMult2x2<data_t>(diag,qubits[0]), iChunk, gid, count);
    else
      Execute(DiagonalMult2x2Controlled<data_t>(diag,qubits), iChunk, gid, count);
  }
  else if(N == 2){
    Execute(DiagonalMult4x4<data_t>(diag,qubits[0],qubits[1]), iChunk, gid, count);
  }
  else{
    StoreMatrix(diag, iChunk);
    StoreUintParams(qubits, iChunk);

    Execute(DiagonalMultNxN<data_t>(qubits), iChunk, gid, count);
  }
}

template <typename data_t>
void ChunkContainer<data_t>::apply_X(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count)
{
  Execute(CX_func<data_t>(qubits), iChunk, gid, count);
}

template <typename data_t>
void ChunkContainer<data_t>::apply_Y(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count)
{
  Execute(CY_func<data_t>(qubits), iChunk, gid, count);
}

template <typename data_t>
void ChunkContainer<data_t>::apply_phase(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t gid, const uint_t count)
{
  Execute(phase_func<data_t>(qubits,*(thrust::complex<double>*)&phase), iChunk, gid, count );
}

template <typename data_t>
void ChunkContainer<data_t>::apply_swap(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const uint_t gid, const uint_t count)
{
  Execute(CSwap_func<data_t>(qubits), iChunk, gid, count);
}


template <typename data_t>
void ChunkContainer<data_t>::apply_multi_swaps(const uint_t iChunk,const reg_t& qubits,const uint_t gid,const uint_t count)
{
  //max 5 swaps can be applied at once using GPU's shared memory
  for(int_t i=0;i<qubits.size();i+=10){
    int_t n = 10;
    if(i + n > qubits.size())
      n = qubits.size() - i;

    reg_t qubits_swap(qubits.begin() + i,qubits.begin() + i + n);
    std::sort(qubits_swap.begin(), qubits_swap.end());
    qubits_swap.insert(qubits_swap.end(), qubits.begin() + i,qubits.begin() + i + n);

    StoreUintParams(qubits_swap, iChunk);
    Execute(MultiSwap_func<data_t>(n), iChunk, gid, count);
  }
}

template <typename data_t>
void ChunkContainer<data_t>::apply_permutation(const uint_t iChunk,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t gid, const uint_t count)
{
  const size_t N = qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  reg_t params;
  Permutation<data_t> f(qubits_sorted,qubits,pairs,params);

  StoreUintParams(params, iChunk);

  Execute(f, iChunk, gid, count);
}

template <typename data_t>
void ChunkContainer<data_t>::apply_rotation(const uint_t iChunk,const reg_t &qubits, const Rotation r, const double theta, const uint_t gid, const uint_t count)
{
  int control_bits = qubits.size() - 1;
  switch(r){
    case Rotation::x:
      apply_matrix(iChunk, qubits, control_bits, Linalg::VMatrix::rx(theta), gid, count);
      break;
    case Rotation::y:
      apply_matrix(iChunk, qubits, control_bits, Linalg::VMatrix::ry(theta), gid, count);
      break;
    case Rotation::z:
      apply_diagonal_matrix(iChunk, qubits, control_bits, Linalg::VMatrix::rz_diag(theta), gid, count);
      break;
    case Rotation::xx:
      apply_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::rxx(theta), gid, count);
      break;
    case Rotation::yy:
      apply_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::ryy(theta), gid, count);
      break;
    case Rotation::zz:
      apply_diagonal_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::rzz_diag(theta), gid, count);
      break;
    case Rotation::zx:
      apply_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::rzx(theta), gid, count);
      break;
    default:
      throw std::invalid_argument(
          "QubitVectorThrust::invalid rotation axis.");
  }
}

template <typename data_t>
void ChunkContainer<data_t>::probabilities(std::vector<double>& probs, const uint_t iChunk, const reg_t& qubits) const
{
  const size_t N = qubits.size();
  const int_t DIM = 1 << N;
  probs.resize(DIM);

  if(N == 1){ //special case for 1 qubit (optimized for measure)
    ExecuteSum2(&probs[0],probability_1qubit_func<data_t>(qubits[0]), iChunk, 1);
  }
  else{
    for(int_t i=0;i<DIM;i++){
      ExecuteSum(&probs[i],probability_func<data_t>(qubits,i), iChunk, 1);
    }
  }
}

template <typename data_t>
double ChunkContainer<data_t>::norm(uint_t iChunk,uint_t count) const
{
  double ret;
  ExecuteSum(&ret,norm_func<data_t>(), iChunk, count);

  return ret;
}

template <typename data_t>
double ChunkContainer<data_t>::trace(uint_t iChunk,uint_t row,uint_t count) const
{
  double ret;
  ExecuteSum(&ret,trace_func<data_t>(row), iChunk, count);

  return ret;
}

template <typename data_t>
double ChunkContainer<data_t>::expval_pauli(const uint_t iChunk,const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const
{
  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    thrust::complex<double> ret = norm(iChunk, 1);
    return ret.real() + ret.imag();
  }
  double ret;
  // specialize x_max == 0
  if(x_mask == 0) {
    ExecuteSum(&ret, expval_pauli_Z_func<data_t>(z_mask), iChunk,  1 );
    return ret;
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);
  ExecuteSum(&ret, expval_pauli_XYZ_func<data_t>(x_mask, z_mask, x_max, phase), iChunk, 1 );
  return ret;
}




//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
