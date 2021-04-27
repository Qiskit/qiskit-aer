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
#define AER_MAX_BUFFERS       4
#define AER_DUMMY_BUFFERS     4     //reserved storage for parameters

#define QV_CUDA_NUM_THREADS 512
#define QV_MAX_REGISTERS 10
#define QV_MAX_BLOCKED_GATES 64


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

struct BlockedGateParams
{
  uint_t mask_;
  char gate_;
  unsigned char qubit_;
};

//========================================
//  base class of gate functions
//========================================
template <typename data_t>
class GateFuncBase
{
protected:
  thrust::complex<data_t>* data_;   //pointer to state vector buffer
  thrust::complex<double>* matrix_; //storage for matrix on device
  uint_t* params_;                  //storage for additional parameters on device
  uint_t base_index_;               //start index of state vector 
public:
  GateFuncBase()
  {
    data_ = NULL;
    base_index_ = 0;
  }
  virtual __host__ __device__ ~GateFuncBase(){}

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
  __host__ __device__ thrust::complex<data_t>* data(void)
  {
    return data_;
  }

  virtual bool is_diagonal(void)
  {
    return false;
  }
  virtual int qubits_count(void)
  {
    return 1;
  }
  virtual int num_control_bits(void)
  {
    return 0;
  }
  virtual int control_mask(void)
  {
    return 1;
  }
  virtual bool use_cache(void)
  {
    return false;
  }
  virtual bool batch_enable(void)
  {
    return true;
  }

  virtual const char* name(void)
  {
    return "base function";
  }
  virtual uint_t size(int num_qubits)
  {
    if(is_diagonal()){
      return (1ull << num_qubits);
    }
    else{
      return (1ull << (num_qubits - (qubits_count() - num_control_bits())));
    }
  }

  virtual __host__ __device__ uint_t thread_to_index(uint_t _tid) const
  {
    return _tid;
  }
  virtual __host__ __device__ void run_with_cache(uint_t _tid,uint_t _idx,thrust::complex<data_t>* _cache) const
  {
    //implemente this in the kernel class
  }
};

//========================================
  //  gate functions with cache
//========================================
template <typename data_t>
class GateFuncWithCache : public GateFuncBase<data_t>
{
protected:
  int nqubits_;
public:
  GateFuncWithCache(uint_t nq)
  {
    nqubits_ = nq;
  }

  bool use_cache(void)
  {
    return true;
  }

  __host__ __device__ uint_t thread_to_index(uint_t _tid) const
  {
    uint_t idx,ii,t,j;
    uint_t* qubits;
    uint_t* qubits_sorted;

    qubits_sorted = this->params_;
    qubits = qubits_sorted + nqubits_;

    idx = 0;
    ii = _tid >> nqubits_;
    for(j=0;j<nqubits_;j++){
      t = ii & ((1ull << qubits_sorted[j]) - 1);
      idx += t;
      ii = (ii - t) << 1;

      if(((_tid >> j) & 1) != 0){
        idx += (1ull << qubits[j]);
      }
    }
    idx += ii;
    return idx;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> cache[1024];
    uint_t j,idx;
    uint_t matSize = 1ull << nqubits_;

    //load data to cache
    for(j=0;j<matSize;j++){
      idx = thread_to_index((i << nqubits_) + j);
      cache[j] = this->data_[idx];
    }

    //execute using cache
    for(j=0;j<matSize;j++){
      idx = thread_to_index((i << nqubits_) + j);
      this->run_with_cache(j,idx,cache);
    }
  }

  virtual int qubits_count(void)
  {
    return nqubits_;
  }
};

//stridded iterator to access diagonal probabilities
template <typename Iterator>
class strided_range
{
  public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor : public thrust::unary_function<difference_type,difference_type>
  {
    difference_type stride;

    stride_functor(difference_type stride)
        : stride(stride) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const
    {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first), last(last), stride(stride) {}
 
  iterator begin(void) const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end(void) const
  {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }
  
  protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

template <typename data_t>
struct complex_dot_scan : public thrust::unary_function<thrust::complex<data_t>,thrust::complex<data_t>>
{
  __host__ __device__
  thrust::complex<data_t> operator()(thrust::complex<data_t> x) { return thrust::complex<data_t>(x.real()*x.real()+x.imag()*x.imag(),0); }
};

template <typename data_t>
struct complex_norm : public thrust::unary_function<thrust::complex<data_t>,thrust::complex<data_t>>
{
  __host__ __device__
  thrust::complex<double> operator()(thrust::complex<data_t> x) { return thrust::complex<double>((double)x.real()*(double)x.real(),(double)x.imag()*(double)x.imag()); }
};

template<typename data_t>
struct complex_less
{
  typedef thrust::complex<data_t> first_argument_type;
  typedef thrust::complex<data_t> second_argument_type;
  typedef bool result_type;
  __thrust_exec_check_disable__
    __host__ __device__ bool operator()(const thrust::complex<data_t> &lhs, const thrust::complex<data_t> &rhs) const {return lhs.real() < rhs.real();}
}; // end less


//============================================================================
// chunk container base class
//============================================================================
template <typename data_t>
class ChunkContainer : public std::enable_shared_from_this<ChunkContainer<data_t>>
{
protected:
  int chunk_bits_;                    //number of qubits for a chunk
  int place_id_;                      //index of a container (device index + host)
  uint_t num_chunks_;                 //number of chunks in this container
  uint_t num_buffers_;                //number of buffers (buffer chunks) in this container
  uint_t num_checkpoint_;             //number of checkpoint buffers in this container
  uint_t num_chunk_mapped_;           //number of chunks mapped
  reg_t blocked_qubits_;
  std::vector<std::shared_ptr<Chunk<data_t>>> chunks_;         //chunk storage
  std::vector<std::shared_ptr<Chunk<data_t>>> buffers_;        //buffer storage
  std::vector<std::shared_ptr<Chunk<data_t>>> checkpoints_;    //checkpoint storage
  bool enable_omp_;                 //disable this when shots are parallelized outside
public:
  ChunkContainer()
  {
    chunk_bits_ = 0;
    place_id_ = 0;
    num_chunks_ = 0;
    num_buffers_ = 0;
    num_checkpoint_ = 0;
    num_chunk_mapped_ = 0;
    enable_omp_ = true;
  }
  virtual ~ChunkContainer(){}

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
  void enable_omp(bool flg)
  {
#pragma omp critical
    {
      enable_omp_ = flg;
    }
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

  virtual thrust::complex<data_t>& operator[](uint_t i) = 0;

  virtual uint_t Allocate(int idev,int bits,uint_t chunks,uint_t buffers = AER_MAX_BUFFERS,uint_t checkpoint = 0) = 0;
  virtual void Deallocate(void) = 0;
  virtual uint_t Resize(uint_t chunks,uint_t buffers = AER_MAX_BUFFERS,uint_t checkpoint = 0) = 0;

  virtual void Set(uint_t i,const thrust::complex<data_t>& t) = 0;
  virtual thrust::complex<data_t> Get(uint_t i) const = 0;

  virtual void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk) = 0;
  virtual void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk) = 0;

  virtual void CopyIn(std::shared_ptr<Chunk<data_t>> src,uint_t iChunk) = 0;
  virtual void CopyOut(std::shared_ptr<Chunk<data_t>> dest,uint_t iChunk) = 0;
  virtual void CopyIn(thrust::complex<data_t>* src,uint_t iChunk, uint_t size) = 0;
  virtual void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk, uint_t size) = 0;
  virtual void Swap(std::shared_ptr<Chunk<data_t>> src,uint_t iChunk) = 0;

  virtual void Zero(uint_t iChunk,uint_t count) = 0;

  template <typename Function>
  void Execute(Function func,uint_t iChunk,uint_t count);

  template <typename Function>
  double ExecuteSum(Function func,uint_t iChunk,uint_t count) const;

  virtual reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride = 1, bool dot = true) const = 0;
  virtual thrust::complex<double> norm(uint_t iChunk,uint_t stride = 1,bool dot = true) const = 0;


  size_t size_of_complex(void)
  {
    return sizeof(thrust::complex<data_t>);
  }

  std::shared_ptr<Chunk<data_t>> MapChunk(void);
  std::shared_ptr<Chunk<data_t>> MapBufferChunk(void);
  std::shared_ptr<Chunk<data_t>> MapCheckpoint(int_t iChunk = -1);
  void UnmapChunk(std::shared_ptr<Chunk<data_t>> chunk);
  void UnmapBuffer(std::shared_ptr<Chunk<data_t>> buf);
  void UnmapCheckpoint(std::shared_ptr<Chunk<data_t>> buf);

  virtual thrust::complex<data_t>* chunk_pointer(uint_t iChunk) const
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
std::shared_ptr<Chunk<data_t>> ChunkContainer<data_t>::MapChunk(void)
{
  uint_t i,pos,idx;
  pos = num_chunks_;

#pragma omp critical
  {
    for(i=0;i<num_chunks_;i++){
      idx = (num_chunk_mapped_ + i) % num_chunks_;
      if(!chunks_[idx]->is_mapped()){
        chunks_[idx]->map();
        pos = idx;
        num_chunk_mapped_++;
        break;
      }
    }
  }

  if(pos < num_chunks_)
    return chunks_[pos];
  return nullptr;
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapChunk(std::shared_ptr<Chunk<data_t>> chunk)
{
  chunk->unmap();
}

template <typename data_t>
std::shared_ptr<Chunk<data_t>> ChunkContainer<data_t>::MapBufferChunk(void)
{
  uint_t i,pos;
  std::shared_ptr<Chunk<data_t>> ret = nullptr;

#pragma omp critical
  {
    for(i=0;i<num_buffers_;i++){
      if(!buffers_[i]->is_mapped()){
        buffers_[i]->map();
        ret = buffers_[i];
        break;
      }
    }
  }

  return ret;
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapBuffer(std::shared_ptr<Chunk<data_t>> buf)
{
#pragma omp critical
  {
    buf->unmap();
  }
}

template <typename data_t>
std::shared_ptr<Chunk<data_t>> ChunkContainer<data_t>::MapCheckpoint(int_t iChunk)
{
  if(iChunk >= 0 && num_checkpoint_ == num_chunks_){   //checkpoint buffers are reserved for all chunks
    if(iChunk < num_checkpoint_)
      return checkpoints_[iChunk];
    return nullptr;
  }
  else{
    uint_t i,pos;
    pos = num_checkpoint_;
#pragma omp critical
    {
      for(i=0;i<num_checkpoint_;i++){
        if(!checkpoints_[i]->is_mapped()){
          checkpoints_[i]->map();
          pos = i;
          break;
        }
      }
    }

    if(pos < num_checkpoint_)
      return checkpoints_[pos];
    return nullptr;
  }
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapCheckpoint(std::shared_ptr<Chunk<data_t>> buf)
{
  if(num_checkpoint_ != num_chunks_){
#pragma omp critical
    {
      buf->unmap();
    }
  }
}

#ifdef AER_THRUST_CUDA

template <typename data_t,typename kernel_t> __global__
void dev_apply_function(kernel_t func)
{
  uint_t i;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  func(i);
}

template <typename data_t,typename kernel_t> __global__
void dev_apply_function_with_cache(kernel_t func)
{
  __shared__ thrust::complex<data_t> cache[1024];
  uint_t i,idx;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  idx = func.thread_to_index(i);

  cache[threadIdx.x] = func.data()[idx];
  __syncthreads();

  func.run_with_cache(threadIdx.x,idx,cache);
}

#endif

template <typename data_t>
template <typename Function>
void ChunkContainer<data_t>::Execute(Function func,uint_t iChunk,uint_t count)
{
  set_device();

  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );

#ifdef AER_THRUST_CUDA
  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t nt,nb;
    nb = 1;

    if(func.use_cache()){
      nt = count << chunk_bits_;

      if(nt > 1024){
        nb = (nt + 1024 - 1) / 1024;
        nt = 1024;
      }
      dev_apply_function_with_cache<data_t,Function><<<nb,nt,0,strm>>>(func);
    }
    else{
      nt = count * func.size(chunk_bits_);
      if(nt > QV_CUDA_NUM_THREADS){
        nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
        nt = QV_CUDA_NUM_THREADS;
      }
      dev_apply_function<data_t,Function><<<nb,nt,0,strm>>>(func);
    }
  }
  else{ //if no stream returned, run on host
    uint_t size = count * func.size(chunk_bits_);
    auto ci = thrust::counting_iterator<uint_t>(0);
    thrust::for_each_n(thrust::seq, ci , size, func);
  }
#else
  uint_t size = count * func.size(chunk_bits_);
  auto ci = thrust::counting_iterator<uint_t>(0);
  if(enable_omp_)
    thrust::for_each_n(thrust::device, ci , size, func);
  else
    thrust::for_each_n(thrust::seq, ci , size, func);  //disable nested OMP parallelization when shots are parallelized
#endif

}

template <typename data_t>
template <typename Function>
double ChunkContainer<data_t>::ExecuteSum(Function func,uint_t iChunk,uint_t count) const
{
  double ret;
  uint_t size = count * func.size(chunk_bits_);

  set_device();

  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );

  auto ci = thrust::counting_iterator<uint_t>(0);

#ifdef AER_THRUST_CUDA
  cudaStream_t strm = stream(iChunk);
  if(strm){
    ret = thrust::transform_reduce(thrust::cuda::par.on(strm), ci, ci + size, func,0.0,thrust::plus<double>());
  }
  else{ //if no stream returned, run on host
    ret = thrust::transform_reduce(thrust::seq, ci, ci + size, func,0.0,thrust::plus<double>());
  }
#else
  if(enable_omp_)
    ret = thrust::transform_reduce(thrust::device, ci, ci + size, func,0.0,thrust::plus<double>());
  else
    ret = thrust::transform_reduce(thrust::seq, ci, ci + size, func,0.0,thrust::plus<double>());  //disable nested OMP parallelization when shots are parallelized
#endif

  return ret;
}


template <typename data_t>
void ChunkContainer<data_t>::allocate_chunks(void)
{
  uint_t i;
  chunks_.resize(num_chunks_);
  buffers_.resize(num_buffers_);
  checkpoints_.resize(num_checkpoint_);

  if(num_chunks_ > 0){
    chunks_.resize(num_chunks_);
    for(i=0;i<num_chunks_;i++){
      chunks_[i] = std::make_shared<Chunk<data_t>>(this->shared_from_this(),i);
    }
  }
  if(num_buffers_ > 0){
    buffers_.resize(num_buffers_);
    for(i=0;i<num_buffers_;i++){
      buffers_[i] = std::make_shared<Chunk<data_t>>(this->shared_from_this(),num_chunks_+i);
    }
  }
  if(num_checkpoint_ > 0){
    checkpoints_.resize(num_checkpoint_);
    for(i=0;i<num_checkpoint_;i++){
      checkpoints_[i] = std::make_shared<Chunk<data_t>>(this->shared_from_this(),num_chunks_+num_buffers_+i);
    }
  }
}

template <typename data_t>
void ChunkContainer<data_t>::deallocate_chunks(void)
{
  uint_t i;

  if(num_chunks_ > 0){
    for(i=0;i<num_chunks_;i++){
      chunks_[i].reset();
    }
    chunks_.clear();
  }
  if(num_buffers_ > 0){
    for(i=0;i<num_buffers_;i++){
      buffers_[i].reset();
    }
    buffers_.clear();
  }
  if(num_checkpoint_ > 0){
    for(i=0;i<num_checkpoint_;i++){
      checkpoints_[i].reset();
    }
    checkpoints_.clear();
  }
}

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
