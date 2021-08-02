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

#define QV_CUDA_NUM_THREADS 1024
#define QV_MAX_REGISTERS 10
#define QV_MAX_BLOCKED_GATES 64


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

#include "simulators/statevector/batched_matrix.hpp"

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
  batched_matrix_params* batched_params_; //storage for parameters for batched matrix multiplier on device
  uint_t base_index_;               //start index of state vector 
  uint_t chunk_bits_;
  uint_t bfunc_mask_;
  uint_t bfunc_target_;
  Operations::RegComparison bfunc_;
  uint_t* creg_bits_;
#ifndef AER_THRUST_CUDA
  uint_t index_offset_;
#endif
public:
  GateFuncBase()
  {
    data_ = NULL;
    base_index_ = 0;
    bfunc_ = Operations::RegComparison::Nop;
    creg_bits_ = NULL;
#ifndef AER_THRUST_CUDA
    index_offset_ = 0;
#endif
  }
//  virtual __host__ __device__ ~GateFuncBase(){}

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
  void set_batched_params(batched_matrix_params* p)
  {
    batched_params_ = p;
  }
  void set_chunk_bits(uint_t bits)
  {
    chunk_bits_ = bits;
  }

  void set_base_index(uint_t i)
  {
    base_index_ = i;
  }
  void set_creg_bits(uint_t* cbits)
  {
    creg_bits_ = cbits;
  }
  void set_conditional(uint_t mask,uint_t target,Operations::RegComparison bfunc)
  {
    bfunc_mask_ = mask;
    bfunc_target_ = target;
    bfunc_ = bfunc;
  }
#ifndef AER_THRUST_CUDA
  void set_index_offset(uint_t i)
  {
    index_offset_ = i;
  }
#endif

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
      chunk_bits_ = num_qubits;
      return (1ull << num_qubits);
    }
    else{
      chunk_bits_ = num_qubits - (qubits_count() - num_control_bits());
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
  virtual __host__ __device__ double run_with_cache_sum(uint_t _tid,uint_t _idx,thrust::complex<data_t>* _cache) const
  {
    //implemente this in the kernel class
    return 0.0;
  }

  virtual __host__ __device__  bool check_condition(uint_t i) const
  {
    return true;
  }

  virtual __host__ __device__ bool check_branch_condition(uint_t i) const
  {
    if(bfunc_ == Operations::RegComparison::Nop)
      return true;
    uint_t iChunk = i >> chunk_bits_;
    uint_t reg = creg_bits_[iChunk];
    int_t compared = (reg & bfunc_mask_) - bfunc_target_;
    bool outcome = true;

    switch(bfunc_) {
      case Operations::RegComparison::Equal:
        outcome = (compared == 0);
        break;
      case Operations::RegComparison::NotEqual:
        outcome = (compared != 0);
        break;
      case Operations::RegComparison::Less:
        outcome = (compared < 0);
        break;
      case Operations::RegComparison::LessEqual:
        outcome = (compared <= 0);
        break;
      case Operations::RegComparison::Greater:
        outcome = (compared > 0);
        break;
      case Operations::RegComparison::GreaterEqual:
        outcome = (compared >= 0);
        break;
      default:
        break;
    }
    return outcome;
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

    __host__ __device__ virtual uint_t thread_to_index(uint_t _tid) const
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

  __host__ __device__ void sync_threads() const
  {
#ifdef CUDA_ARCH
    __syncthreads();
#endif
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    if(!check_condition(i << nqubits_))
      return;

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

  virtual __host__ __device__  bool check_condition(uint_t i) const
  {
    return true;
  }
};

template <typename data_t>
class GateFuncSumWithCache : public GateFuncBase<data_t>
{
protected:
  int nqubits_;
public:
  GateFuncSumWithCache(uint_t nq)
  {
    nqubits_ = nq;
  }

  bool use_cache(void)
  {
    return true;
  }


  __host__ __device__ virtual uint_t thread_to_index(uint_t _tid) const
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

  __host__ __device__ double operator()(const uint_t &i) const
  {
#ifndef AER_THRUST_CUDA
    if(!check_condition((i << nqubits_) + this->index_offset_))
      return 0.0;
#endif

    thrust::complex<data_t> cache[1024];
    uint_t j,idx;
    uint_t matSize = 1ull << nqubits_;
    double sum = 0.0;

    //load data to cache
    for(j=0;j<matSize;j++){
      idx = thread_to_index((i << nqubits_) + j);
      cache[j] = this->data_[idx];
    }

    //execute using cache
    for(j=0;j<matSize;j++){
      idx = thread_to_index((i << nqubits_) + j);
      sum += this->run_with_cache_sum(j,idx,cache);
    }
    return sum;
  }

  virtual int qubits_count(void)
  {
    return nqubits_;
  }

  virtual __host__ __device__  bool check_condition(uint_t i) const
  {
    return true;
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


class HostFuncBase
{
protected:
public:
  HostFuncBase(){}

  virtual void execute(){}
};

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
  uint_t num_chunk_mapped_;           //number of chunks mapped
  reg_t blocked_qubits_;
  std::vector<bool> chunks_map_;      //chunk mapper
  std::vector<bool> buffers_map_;     //buffer mapper
  bool enable_omp_;                 //disable this when shots are parallelized outside
  mutable reg_t reduced_queue_begin_;
  mutable reg_t reduced_queue_end_;
  uint_t bfunc_mask_;
  uint_t bfunc_target_;
  Operations::RegComparison bfunc_;
public:
  ChunkContainer()
  {
    chunk_bits_ = 0;
    place_id_ = 0;
    num_chunks_ = 0;
    num_buffers_ = 0;
    num_chunk_mapped_ = 0;
    enable_omp_ = false;
    bfunc_ = Operations::RegComparison::Nop;
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
    enable_omp_ = flg;
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

  virtual void set_conditional(uint_t mask,uint_t target,Operations::RegComparison bfunc)
  {
    bfunc_mask_ = mask;
    bfunc_target_ = target;
    bfunc_ = bfunc;
  }

  virtual thrust::complex<data_t>& operator[](uint_t i) = 0;

  virtual uint_t Allocate(int idev,int bits,uint_t chunks,uint_t buffers = AER_MAX_BUFFERS,bool multi_shots = false) = 0;
  virtual void Deallocate(void) = 0;
  virtual uint_t Resize(uint_t chunks,uint_t buffers = AER_MAX_BUFFERS) = 0;

  virtual void Set(uint_t i,const thrust::complex<data_t>& t) = 0;
  virtual thrust::complex<data_t> Get(uint_t i) const = 0;

  virtual void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk) = 0;
  virtual void StoreBatchedMatrix(const std::vector<std::complex<double>>& mat) = 0;
  virtual void StoreMatrix(const std::complex<double>* mat,uint_t iChunk,uint_t size) = 0;
  virtual void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk) = 0;
  virtual void StoreBatchedParams(const std::vector<batched_matrix_params>& params) = 0;

  virtual void CopyIn(Chunk<data_t>& src,uint_t iChunk) = 0;
  virtual void CopyOut(Chunk<data_t>& dest,uint_t iChunk) = 0;
  virtual void CopyIn(thrust::complex<data_t>* src,uint_t iChunk, uint_t size) = 0;
  virtual void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk, uint_t size) = 0;
  virtual void Swap(Chunk<data_t>& src,uint_t iChunk) = 0;

  virtual void Zero(uint_t iChunk,uint_t count) = 0;

  template <typename Function>
  void Execute(Function func,uint_t iChunk,uint_t count);

  template <typename Function>
  void ExecuteSum(double* pSum,Function func,uint_t iChunk,uint_t count) const;

  template <typename Function>
  void ExecuteSum2(double* pSum,Function func,uint_t iChunk,uint_t count) const;

  virtual reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride = 1, bool dot = true,uint_t count = 1) const = 0;
  virtual thrust::complex<double> norm(uint_t iChunk,uint_t count,uint_t stride = 1,bool dot = true) const = 0;


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
  virtual thrust::complex<double>* matrix_pointer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual uint_t* param_pointer(uint_t iChunk) const
  {
    return NULL;
  }
  virtual batched_matrix_params* batched_param_pointer(void) const
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

  //CUDA graph
  virtual void begin_capture(uint_t iChunk) const
  {
    ;
  }
  virtual void end_capture(uint_t iChunk) const
  {
    ;
  }
  virtual void begin_add_graph_node(uint_t iChunk){}
  virtual void end_add_graph_node(uint_t iChunk){}

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
  virtual void init_condition(uint_t iChunk,std::vector<double>& cond){}
  virtual bool update_condition(uint_t iChunk,uint_t count,bool async){return true;}

  virtual int measured_cbit(uint_t iChunk,int qubit)
  {
    return 0;
  }
  virtual uint_t* measured_bits_buffer(uint_t iChunk)
  {
    return NULL;
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
  uint_t i,pos;

  for(i=0;i<num_buffers_;i++){
    if(!buffers_map_[i]){
      buffers_map_[i] = true;
      chunk.map(this->shared_from_this(),num_chunks_+i);
      break;
    }
  }
  return chunk.is_mapped();
}

template <typename data_t>
void ChunkContainer<data_t>::UnmapBuffer(Chunk<data_t>& buf)
{
  buffers_map_[buf.pos()-num_chunks_] = false;
  buf.unmap();
}

template <typename data_t>
void ChunkContainer<data_t>::unmap_all(void)
{
  int_t i;
  for(i=0;i<chunks_map_.size();i++)
    chunks_map_[i] = false;
  for(i=0;i<buffers_map_.size();i++)
    buffers_map_[i] = false;
  num_chunk_mapped_ = 0;
}

template <typename data_t>
template <typename Function>
void ChunkContainer<data_t>::Execute(Function func,uint_t iChunk,uint_t count)
{
  set_device();

  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_batched_params(batched_param_pointer() );
  func.set_creg_bits(measured_bits_buffer(iChunk));
  if(iChunk == 0 && bfunc_ != Operations::RegComparison::Nop){
    func.set_conditional(bfunc_mask_,bfunc_target_,bfunc_);

    bfunc_ = Operations::RegComparison::Nop;  //reset conditional
  }

#ifdef AER_THRUST_CUDA
  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t nt,nb;
    nb = 1;

    if(func.use_cache()){
      func.set_chunk_bits(chunk_bits_);

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
void ChunkContainer<data_t>::ExecuteSum(double* pSum,Function func,uint_t iChunk,uint_t count) const
{

#ifdef AER_THRUST_CUDA
  uint_t size = count * func.size(chunk_bits_);

  set_device();

  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_batched_params(batched_param_pointer() );

  auto ci = thrust::counting_iterator<uint_t>(0);

  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t buf_size = reduce_buffer_size();
    double* buf = reduce_buffer(iChunk);

    uint_t n,nt,nb;
    nb = 1;

    if(func.use_cache()){
      nt = 1ull << chunk_bits_;
      if(nt > 1024){
        nb = (nt + 1024 - 1) / 1024;
        nt = 1024;
      }
      dim3 grid(nb,count,1);
      dev_apply_function_sum_with_cache<data_t,Function><<<grid,nt,0,strm>>>(buf,func,buf_size);
    }
    else{
      nt = func.size(chunk_bits_);
      if(nt > 1024){
        nb = (nt + 1024 - 1) / 1024;
        nt = 1024;
      }
      dim3 grid(nb,count,1);
      dev_apply_function_sum<data_t,Function><<<grid,nt,0,strm>>>(buf,func,buf_size);
    }

    while(nb > 1){
      n = nb;
      nt = nb;
      nb = 1;
      if(nt > 1024){
        nb = (nt + 1024 - 1) / 1024;
        nt = 1024;
      }
      dim3 grid(nb,count,1);
      dev_reduce_sum<<<grid,nt,0,strm>>>(buf,n,buf_size);
    }
    if(pSum)
      cudaMemcpyAsync(pSum,buf,sizeof(double),cudaMemcpyDeviceToHost,strm);
  }
  else{ //if no stream returned, run on host
    *pSum = thrust::transform_reduce(thrust::seq, ci, ci + size, func,0.0,thrust::plus<double>());
  }
#else
  uint_t size = func.size(chunk_bits_);

  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_batched_params(batched_param_pointer() );

  uint_t i;
  for(i=0;i<count;i++){

    func.set_data( chunk_pointer(iChunk + i) );
    func.set_index_offset(iChunk << chunk_bits_);

    auto ci = thrust::counting_iterator<uint_t>(0);

    double sum;
    if(enable_omp_)
      sum = thrust::transform_reduce(thrust::device, ci, ci + size, func,0.0,thrust::plus<double>());
    else
      sum = thrust::transform_reduce(thrust::seq, ci, ci + size, func,0.0,thrust::plus<double>());  //disable nested OMP parallelization when shots are parallelized
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

  func.set_data( chunk_pointer(iChunk) );
  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_batched_params(batched_param_pointer() );

  auto ci = thrust::counting_iterator<uint_t>(0);

  cudaStream_t strm = stream(iChunk);
  if(strm){
    uint_t buf_size = reduce_buffer_size() / 2;
    uint_t n,nt,nb;
    nb = 1;
    nt = func.size(chunk_bits_);

    if(nt > 1024){
      nb = (nt + 1024 - 1) / 1024;
      nt = 1024;
    }
    thrust::complex<double>* buf = (thrust::complex<double>*)reduce_buffer(iChunk);
    dim3 grid(nb,count,1);
    dev_apply_function_sum_complex<data_t,Function><<<grid,nt,0,strm>>>(buf,func,buf_size);

    while(nb > 1){
      n = nb;
      nt = nb;
      nb = 1;
      if(nt > 1024){
        nb = (nt + 1024 - 1) / 1024;
        nt = 1024;
      }
      dim3 grid(nb,count,1);
      dev_reduce_sum_complex<<<grid,nt,0,strm>>>(buf,n,buf_size);
    }
    if(pSum)
      cudaMemcpyAsync(pSum,buf,sizeof(double)*2,cudaMemcpyDeviceToHost,strm);
  }
  else{ //if no stream returned, run on host
    thrust::complex<double> ret,zero = 0.0;
    ret = thrust::transform_reduce(thrust::seq, ci, ci + size, func,zero,complex_sum());
    *((thrust::complex<double>*)pSum) = ret;
  }
#else
  uint_t size = func.size(chunk_bits_);

  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );
  func.set_batched_params(batched_param_pointer() );

  uint_t i;
  for(i=0;i<count;i++){
    thrust::complex<double> ret,zero = 0.0;
    func.set_data( chunk_pointer(iChunk + i) );
    func.set_index_offset(iChunk << chunk_bits_);

    auto ci = thrust::counting_iterator<uint_t>(0);

    if(enable_omp_)
      ret = thrust::transform_reduce(thrust::device, ci, ci + size, func,zero,complex_sum());
    else
      ret = thrust::transform_reduce(thrust::seq, ci, ci + size, func,zero,complex_sum());  //disable nested OMP parallelization when shots are parallelized

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
  buffers_map_.resize(num_buffers_,false);

  reduced_queue_begin_.resize(num_chunks_,0);
  reduced_queue_end_.resize(num_chunks_,0);
}

template <typename data_t>
void ChunkContainer<data_t>::deallocate_chunks(void)
{
  chunks_map_.clear();
  buffers_map_.clear();

  reduced_queue_begin_.clear();
  reduced_queue_end_.clear();
}

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
