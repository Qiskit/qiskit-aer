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


#ifndef _qv_qubit_vector_thrust_hpp_
#define _qv_qubit_vector_thrust_hpp_

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

#include "framework/json.hpp"


#ifdef AER_TIMING

#include <sys/time.h>
double mysecond()
{
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#define QS_NUM_GATES          5
#define QS_GATE_INIT          0
#define QS_GATE_MULT          1
#define QS_GATE_CX            2
#define QS_GATE_DIAG          3
#define QS_GATE_MEASURE         4

#endif

#define AER_DEFAULT_MATRIX_BITS   5

#define AER_CHUNK_BITS        21
#define AER_MAX_BUFFERS       4
#define AER_DUMMY_BUFFERS     4     //reserved storage for parameters

namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;

#ifdef AER_THRUST_CUDA
#define AERDeviceVector thrust::device_vector
#else
#define AERDeviceVector thrust::host_vector
#endif
#define AERHostVector thrust::host_vector

#define ExtractIndexFromTuple(itp)        thrust::get<0>(itp)
#define ExtractParamsFromTuple(itp)       thrust::get<1>(itp)

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
template <typename data_t> class Chunk;
template <typename data_t> class HostChunkContainer;

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


//============================================================================
// device chunk container class
//============================================================================
template <typename data_t>
class DeviceChunkContainer : public ChunkContainer<data_t>
{
protected:
  mutable AERDeviceVector<thrust::complex<data_t>>  data_;    //device vector to chunks and buffers
  mutable AERDeviceVector<thrust::complex<double>>          matrix_;  //storage for large matrix
  mutable AERDeviceVector<uint_t>                           params_;  //storage for additional parameters
  int device_id_;                     //device index
  std::vector<bool> peer_access_;     //to which device accepts peer access 
  int matrix_bits_;                   //number of bits (fusion bits) for matrix buffer
  uint_t num_matrices_;               //number of matrices for chunks (1 shared matrix for multi-chunk execution)
#ifdef AER_THRUST_CUDA
  std::vector<cudaStream_t> stream_; 
#endif
public:
  DeviceChunkContainer()
  {
    device_id_ = 0;
    matrix_bits_ = 0;
    num_matrices_ = 1;
  }
  ~DeviceChunkContainer();

  DeviceChunkContainer(const DeviceChunkContainer& obj){}
  DeviceChunkContainer &operator=(const DeviceChunkContainer& obj){}

  uint_t size(void)
  {
    return data_.size();
  }
  int device(void)
  {
    return device_id_;
  }

  AERDeviceVector<thrust::complex<data_t>>& vector(void)
  {
    return data_;
  }

  bool peer_access(int i_dest)
  {
    if(i_dest < 0){
#ifdef AER_ATS
      //for IBM AC922
      return true;
#else
      return false;
#endif
    }
    return peer_access_[i_dest];
  }

  thrust::complex<data_t>& operator[](uint_t i)
  {
    return raw_reference_cast(data_[i]);
  }

  uint_t Allocate(int idev,int bits,uint_t chunks,uint_t buffers,uint_t checkpoint);
  void Deallocate(void);
  uint_t Resize(uint_t chunks,uint_t buffers,uint_t checkpoint);

  void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk);
  void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk);
  void ResizeMatrixBuffers(int bits);

  void set_device(void) const
  {
#ifdef AER_THRUST_CUDA
    cudaSetDevice(device_id_);
#endif
  }

#ifdef AER_THRUST_CUDA
  cudaStream_t stream(uint_t iChunk)
  {
    return stream_[iChunk];
  }
#endif

  void Set(uint_t i,const thrust::complex<data_t>& t)
  {
    data_[i] = t;
  }
  thrust::complex<data_t> Get(uint_t i) const
  {
    return data_[i];
  }

  thrust::complex<data_t>* chunk_pointer(void)
  {
    return (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data());
  }

  void CopyIn(Chunk<data_t>* src,uint_t iChunk);
  void CopyOut(Chunk<data_t>* src,uint_t iChunk);
  void CopyIn(thrust::complex<data_t>* src,uint_t iChunk);
  void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk);
  void Swap(Chunk<data_t>* src,uint_t iChunk);

  void Zero(uint_t iChunk,uint_t count);

  reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds) const;

  template <typename Function>
  void Execute(Function func,uint_t iChunk,uint_t count);

  template <typename Function>
  double ExecuteSum(Function func,uint_t iChunk,uint_t count) const;

  template <typename Function>
  thrust::complex<double> ExecuteComplexSum(Function func,uint_t iChunk,uint_t count) const;

  thrust::complex<double>* matrix_pointer(uint_t iChunk) const
  {
    if(iChunk >= this->num_chunks_){  //for buffer chunks
      return ((thrust::complex<double>*)thrust::raw_pointer_cast(matrix_.data())) + ((num_matrices_ + iChunk - this->num_chunks_) << (matrix_bits_*2));
    }
    else{
      return ((thrust::complex<double>*)thrust::raw_pointer_cast(matrix_.data())) + (iChunk << (matrix_bits_*2));
    }
  }
  uint_t* param_pointer(uint_t iChunk) const
  {
    if(iChunk >= this->num_chunks_){  //for buffer chunks
      return ((uint_t*)thrust::raw_pointer_cast(params_.data())) + ((num_matrices_ + iChunk - this->num_chunks_) << (matrix_bits_+2));
    }
    else{
      return ((uint_t*)thrust::raw_pointer_cast(params_.data())) + (iChunk << (matrix_bits_+2));
    }
  }
  int matrix_bits(void)
  {
    return matrix_bits_;
  }
};

template <typename data_t>
DeviceChunkContainer<data_t>::~DeviceChunkContainer(void)
{
  Deallocate();
}

template <typename data_t>
uint_t DeviceChunkContainer<data_t>::Allocate(int idev,int bits,uint_t chunks,uint_t buffers,uint_t checkpoint)
{
  uint_t nc = chunks;
  uint_t i;
  int mat_bits;

  this->chunk_bits_ = bits;

  device_id_ = idev;
  set_device();

#ifdef AER_THRUST_CUDA
  int ip,nd;
  cudaGetDeviceCount(&nd);
  peer_access_.resize(nd);
  for(i=0;i<nd;i++){
    ip = 1;
    if(i != device_id_){
      cudaDeviceCanAccessPeer(&ip,device_id_,i);
    }
    if(ip)
      peer_access_[i] = true;
    else
      peer_access_[i] = false;
  }
#else
  peer_access_.resize(1);
  peer_access_[0] = true;
#endif

  this->num_buffers_ = buffers;

  if(omp_get_num_threads() > 1){    //mult-shot parallelization
    mat_bits = bits;
    this->num_checkpoint_ = checkpoint;
    nc = chunks;
    num_matrices_ = chunks;
  }
  else{
    mat_bits = AER_DEFAULT_MATRIX_BITS;
    num_matrices_ = 1;

    nc = chunks;
#ifdef AER_THRUST_CUDA
    uint_t param_size;
    param_size = (sizeof(thrust::complex<double>) << (matrix_bits_*2)) + (sizeof(uint_t) << (matrix_bits_+2));

    size_t freeMem,totalMem;
    cudaMemGetInfo(&freeMem,&totalMem);
    while(freeMem < ((((nc+buffers+checkpoint + (uint_t)AER_DUMMY_BUFFERS)*(uint_t)sizeof(thrust::complex<data_t>)) << bits) + param_size* (num_matrices_ + buffers)) ){
      if(checkpoint > 0){
        checkpoint--;
      }
      else{
        nc--;
        if(checkpoint > nc){
          checkpoint = nc;
        }
      }
      if(nc == 0){
        break;
      }
    }
#endif
    this->num_checkpoint_ = checkpoint;
  }

  ResizeMatrixBuffers(mat_bits);

  this->num_chunks_ = nc;
  data_.resize((nc+buffers+checkpoint) << bits);

  if(nc > 0){
    this->chunk_mapped_.resize(nc);
    for(i=0;i<nc;i++){
      this->chunk_mapped_[i] = false;
    }
  }

  this->buffer_mapped_.resize(buffers);
  for(i=0;i<buffers;i++){
    this->buffer_mapped_[i] = false;
  }

  this->checkpoint_mapped_.resize(checkpoint);
  for(i=0;i<checkpoint;i++){
    this->checkpoint_mapped_[i] = false;
  }

#ifdef AER_THRUST_CUDA
  stream_.resize(nc + buffers);
  for(i=0;i<nc + buffers;i++){
    cudaStreamCreateWithFlags(&stream_[i], cudaStreamNonBlocking);
  }
#endif

  return nc;
}

template <typename data_t>
uint_t DeviceChunkContainer<data_t>::Resize(uint_t chunks,uint_t buffers,uint_t checkpoint)
{
  uint_t i;

  if(chunks + buffers + checkpoint > this->num_chunks_ + this->num_buffers_ + this->num_checkpoint_){
    set_device();
    data_.resize((chunks + buffers + checkpoint) << this->chunk_bits_);
  }

  if(chunks > this->num_chunks_){
    this->chunk_mapped_.resize(chunks);
    for(i=this->num_chunks_;i<chunks;i++){
      this->chunk_mapped_[i] = false;
    }
  }
  this->num_chunks_ = chunks;

  if(buffers > this->num_buffers_){
    this->buffer_mapped_.resize(buffers);
    for(i=this->num_buffers_;i<buffers;i++){
      this->buffer_mapped_[i] = false;
    }
  }
  this->num_buffers_ = buffers;

  if(checkpoint > this->num_checkpoint_){
    this->checkpoint_mapped_.resize(checkpoint);
    for(i=this->num_checkpoint_;i<checkpoint;i++){
      this->checkpoint_mapped_[i] = false;
    }
  }
  this->num_checkpoint_ = checkpoint;

#ifdef AER_THRUST_CUDA
  if(stream_.size() < chunks + buffers){
    uint_t size = stream_.size();
    stream_.resize(chunks + buffers);
    for(i=size;i<chunks + buffers;i++){
      cudaStreamCreateWithFlags(&stream_[i], cudaStreamNonBlocking);
    }
  }
#endif

  return chunks + buffers + checkpoint;
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Deallocate(void)
{
  set_device();
  data_.clear();

  peer_access_.clear();
  matrix_.clear();
  params_.clear();

#ifdef AER_THRUST_CUDA
  uint_t i;
  for(i=0;i<stream_.size();i++){
    cudaStreamDestroy(stream_[i]);
  }
  stream_.clear();
#endif

}

template <typename data_t>
void DeviceChunkContainer<data_t>::ResizeMatrixBuffers(int bits)
{
  if(bits > matrix_bits_){
    uint_t n = num_matrices_ + this->num_buffers_;

    matrix_bits_ = bits;
    matrix_.resize(n << (matrix_bits_ * 2));
    params_.resize(n << (matrix_bits_ + 2));
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can store (multi-chunk mode)
    return;
  }
  if((1ull << (matrix_bits_ * 2)) < mat.size()){
    int bits;
    bits = matrix_bits_;
    while((1 << (bits*2)) < mat.size()){
      bits++;
    }
    ResizeMatrixBuffers(bits);
  }

  set_device();
#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(matrix_pointer(iChunk),&mat[0],mat.size()*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[iChunk]);
#else
  uint_t offset;
  if(iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunk_) << (matrix_bits_*2);
  else
    offset = iChunk << (matrix_bits_*2);
  thrust::copy_n(mat.begin(),mat.size(),matrix_.begin() + offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can store (multi-chunk mode)
    return;
  }
  set_device();

  if((1ull << (matrix_bits_ + 2)) < prm.size()){
    int bits;
    bits = matrix_bits_;
    while((1 << (bits+2)) < prm.size()){
      bits++;
    }
    ResizeMatrixBuffers(bits);
  }

#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(param_pointer(iChunk),&prm[0],prm.size()*sizeof(uint_t),cudaMemcpyHostToDevice,stream_[iChunk]);
#else
  uint_t offset;
  if(iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunk_) << (matrix_bits_ + 2);
  else
    offset = iChunk << (matrix_bits_ + 2);
  thrust::copy_n(prm.begin(),prm.size(),params_.begin() + offset);
#endif
}

template <typename data_t>
template <typename Function>
void DeviceChunkContainer<data_t>::Execute(Function func,uint_t iChunk,uint_t count)
{
  set_device();
  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data()) + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );

  auto ci = thrust::counting_iterator<uint_t>(0);

#ifdef AER_THRUST_CUDA
  thrust::for_each_n(thrust::cuda::par.on(stream_[iChunk]), ci , count, func);
#else
  thrust::for_each_n(thrust::device, ci , count, func);
#endif
}

template <typename data_t>
template <typename Function>
double DeviceChunkContainer<data_t>::ExecuteSum(Function func,uint_t iChunk,uint_t count) const
{
  double ret;

  set_device();
  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data())  + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );

  auto ci = thrust::counting_iterator<uint_t>(0);

#ifdef AER_THRUST_CUDA
  ret = thrust::transform_reduce(thrust::cuda::par.on(stream_[iChunk]), ci, ci + count, func,0.0,thrust::plus<double>());
#else
  ret = thrust::transform_reduce(thrust::device, ci, ci + count, func,0.0,thrust::plus<double>());
#endif
  return ret;
}

template <typename data_t>
template <typename Function>
thrust::complex<double> DeviceChunkContainer<data_t>::ExecuteComplexSum(Function func,uint_t iChunk,uint_t count) const
{
  thrust::complex<double> ret;
  thrust::complex<double> zero = 0.0;

  set_device();
  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data())  + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_pointer(iChunk) );
  func.set_params( param_pointer(iChunk) );

  auto ci = thrust::counting_iterator<uint_t>(0);

#ifdef AER_THRUST_CUDA
  ret = thrust::transform_reduce(thrust::cuda::par.on(stream_[iChunk]), ci, ci + count, func,zero,thrust::plus<thrust::complex<double>>());
#else
  ret = thrust::transform_reduce(thrust::device, ci, ci + count, func,zero,thrust::plus<thrust::complex<double>>());
#endif
  return ret;
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyIn(Chunk<data_t>* src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  if(src->device() >= 0){
    DeviceChunkContainer<data_t>* src_cont = (DeviceChunkContainer<data_t>*)src->container();
    if(peer_access(src->device())){
      thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
    }
    else{
      AERHostVector<thrust::complex<data_t>> tmp(size);
      thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,tmp.begin());
      thrust::copy_n(tmp.begin(),size,data_.begin() + (iChunk << this->chunk_bits_));
    }
  }
  else{
    HostChunkContainer<data_t>* src_cont = (HostChunkContainer<data_t>*)src->container();
    thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyOut(Chunk<data_t>* dest,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  if(dest->device() >= 0){
    DeviceChunkContainer<data_t>* dest_cont = (DeviceChunkContainer<data_t>*)dest->container();
    if(peer_access(dest->device())){
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest_cont->vector().begin() + (dest->pos() << this->chunk_bits_));
    }
    else{
      AERHostVector<thrust::complex<data_t>> tmp(size);
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,tmp.begin());
      thrust::copy_n(tmp.begin(),size,dest_cont->vector().begin() + (dest->pos() << this->chunk_bits_));
    }
  }
  else{
    HostChunkContainer<data_t>* dest_cont = (HostChunkContainer<data_t>*)dest->container();
    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest_cont->vector().begin() + (dest->pos() << this->chunk_bits_));
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyIn(thrust::complex<data_t>* src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();

  thrust::copy_n(src,size,data_.begin() + (iChunk << this->chunk_bits_));
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyOut(thrust::complex<data_t>* dest,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest);
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Swap(Chunk<data_t>* src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  if(src->device() >= 0){
    DeviceChunkContainer<data_t>* src_cont = (DeviceChunkContainer<data_t>*)src->container();
    if(peer_access(src->device())){
      thrust::swap_ranges(thrust::device,data_.begin() + (iChunk << this->chunk_bits_),data_.begin() + (iChunk << this->chunk_bits_) + size,src_cont->vector().begin() + (src->pos() << this->chunk_bits_));
    }
    else{
      //using temporary buffer on host
      AERHostVector<thrust::complex<data_t>> tmp1(size);
      AERHostVector<thrust::complex<data_t>> tmp2(size);

      thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,tmp1.begin());
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,tmp2.begin());
      thrust::copy_n(tmp1.begin(),size,data_.begin() + (iChunk << this->chunk_bits_));
      thrust::copy_n(tmp2.begin(),size,src_cont->vector().begin() + (src->pos() << this->chunk_bits_));
    }
  }
  else{
    //using temporary buffer on host
    AERHostVector<thrust::complex<data_t>> tmp1(size);
    HostChunkContainer<data_t>* src_cont = (HostChunkContainer<data_t>*)src->container();

#ifdef AER_ATS
    //for IBM AC922
    thrust::swap_ranges(thrust::device,data_.begin() + (iChunk << this->chunk_bits_),data_.begin() + (iChunk << this->chunk_bits_) + size,src_cont->vector().begin() + (src->pos() << this->chunk_bits_));
#else
    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,tmp1.begin());
    thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
    thrust::copy_n(tmp1.begin(),size,src_cont->vector().begin() + (src->pos() << this->chunk_bits_));
#endif
  }
}



template <typename data_t>
void DeviceChunkContainer<data_t>::Zero(uint_t iChunk,uint_t count)
{
  set_device();
#ifdef AER_THRUST_CUDA
  thrust::fill_n(thrust::cuda::par.on(stream_[iChunk]),data_.begin() + (iChunk << this->chunk_bits_),count,0.0);
#else
  thrust::fill_n(thrust::device,data_.begin() + (iChunk << this->chunk_bits_),count,0.0);
#endif
}

template <typename data_t>
class probability_scan : public GateFuncBase<data_t>
{
protected:
  int qubit_begin;
  int qubit_end;
  uint_t size_;
public:
  probability_scan(int qubit0,int qubit1)
  {
    qubit_begin = qubit0;
    qubit_end = qubit1;
    size_ = 1ull << (qubit_end);
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t mask,offset;
    int j,k,begin;
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec;

    vec = this->data_ + (i << qubit_end);

    begin = qubit_begin;
    if(qubit_begin == 0){
      data_t t0,t1,t2,t3;
      for(j=0;j<size_;j+=2){
        q0 = vec[j];
        q1 = vec[j+1];

        t0 = q0.real();
        t1 = q0.imag();
        t2 = q1.real();
        t3 = q1.imag();

        t0 = t0*t0;
        t1 = t0 + t1*t1;
        t2 = t1 + t2*t2;
        t3 = t2 + t3*t3;

        q0 = (t0,t1);
        q1 = (t2,t3);
        vec[j  ] = q0;
        vec[j+1] = q1;
      }

      begin++;
    }

    for(j=begin;j<qubit_end;j++){
      mask = (1ull << j) - 1;
      offset = 1ull << j;
      for(k=0;k<size_/2;k++){
        i1 = k & mask;
        i0 = (k - i1) << 1;
        i0 += i1;

        i1 = i0 + offset;
        i0 |= mask;

        q0 = vec[i0];
        q1 = vec[i1];

        q1 += q0;
        vec[i1] = q1;
      }
    }
  }
  const char* name(void)
  {
    return "probability_scan";
  }

  uint_t size(int num_qubits,int n)
  {
    return (1ull << (num_qubits - qubit_end));
  }

};

template <typename data_t>
reg_t DeviceChunkContainer<data_t>::sample_measure(uint_t iChunk,const std::vector<double> &rnds) const
{
  const int_t SHOTS = rnds.size();
  reg_t samples(SHOTS,0);
  thrust::host_vector<uint_t> vSmp(SHOTS);
  data_t* pVec;
  int i;
  uint_t size = (2 << this->chunk_bits_);

  set_device();
  pVec = (data_t*)((thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data()) + (iChunk << this->chunk_bits_));

#ifdef AER_THRUST_CUDA
  
  if(omp_get_num_threads() > 1 && this->chunk_bits_ < 15){
    //for multi-shot parallelization, use custom kernel to avoid cudaMalloc used in inclusive_scan
    int i_next;
    uint_t count;

    i = 0;
    while(i < this->chunk_bits_){
      i_next = i + 5;
      if(i_next > this->chunk_bits_){
        i_next = this->chunk_bits_;
      }
      count = 1ull << (this->chunk_bits_ - i_next);

      auto ci = thrust::counting_iterator<uint_t>(0);
      probability_scan<data_t> scan(i,i_next);

      scan.set_data( (thrust::complex<data_t>*)pVec );
      thrust::for_each_n(thrust::cuda::par.on(stream_[iChunk]), ci , count, scan);

      i = i_next;
    }
  }
  else{
    thrust::transform_inclusive_scan(thrust::cuda::par.on(stream_[iChunk]),pVec,pVec+size,pVec,thrust::square<double>(),thrust::plus<double>());
  }

  if(SHOTS < (1 << (matrix_bits_ + 2))){
    //matrix and parameter buffers can be used
    double* pRnd = (double*)matrix_pointer(iChunk);
    uint_t* pSmp = param_pointer(iChunk);

    cudaMemcpyAsync(pRnd,&rnds[0],SHOTS*sizeof(double),cudaMemcpyHostToDevice,stream_[iChunk]);

    thrust::lower_bound(thrust::cuda::par.on(stream_[iChunk]), pVec, pVec + size, pRnd, pRnd + SHOTS, params_.begin() + (iChunk << (matrix_bits_+2)) );

    cudaMemcpyAsync(thrust::raw_pointer_cast(vSmp.data()),pSmp,SHOTS*sizeof(uint_t),cudaMemcpyDeviceToHost,stream_[iChunk]);
    cudaStreamSynchronize(stream_[iChunk]);
  }
  else{
    thrust::device_vector<double> vRnd_dev(SHOTS);
    thrust::device_vector<uint_t> vSmp_dev(SHOTS);

    cudaMemcpyAsync(thrust::raw_pointer_cast(vRnd_dev.data()),&rnds[0],SHOTS*sizeof(double),cudaMemcpyHostToDevice,stream_[iChunk]);

    thrust::lower_bound(thrust::cuda::par.on(stream_[iChunk]), pVec, pVec + size, vRnd_dev.begin(), vRnd_dev.begin() + SHOTS, vSmp_dev.begin());

    cudaMemcpyAsync(thrust::raw_pointer_cast(vSmp.data()),thrust::raw_pointer_cast(vSmp_dev.data()),SHOTS*sizeof(uint_t),cudaMemcpyDeviceToHost,stream_[iChunk]);
    cudaStreamSynchronize(stream_[iChunk]);

    vRnd_dev.clear();
    vSmp_dev.clear();
  }
#else
  thrust::transform_inclusive_scan(thrust::device,pVec,pVec+size,pVec,thrust::square<double>(),thrust::plus<double>());
  thrust::lower_bound(thrust::device, pVec, pVec + size, rnds.begin(), rnds.begin() + SHOTS, vSmp.begin());
#endif

  for(i=0;i<SHOTS;i++){
    samples[i] = vSmp[i]/2;
  }
  vSmp.clear();

  return samples;
}


//============================================================================
// host chunk container class
//============================================================================
template <typename data_t>
class HostChunkContainer : public ChunkContainer<data_t>
{
protected:
  AERHostVector<thrust::complex<data_t>>  data_;     //host vector for chunks + buffers
  thrust::complex<double>* matrix_;                 //pointer to matrix
  uint_t* params_;                                  //pointer to additional parameters
public:
  HostChunkContainer()
  {
  }
  ~HostChunkContainer();

  HostChunkContainer(const HostChunkContainer& obj){}
  HostChunkContainer &operator=(const HostChunkContainer& obj){}

  uint_t size(void)
  {
    return data_.size();
  }

  AERHostVector<thrust::complex<data_t>>& vector(void)
  {
    return data_;
  }

  thrust::complex<data_t>& operator[](uint_t i)
  {
    return data_[i];
  }

  uint_t Allocate(int idev,int bits,uint_t chunks,uint_t buffers,uint_t checkpoint);
  void Deallocate(void);
  uint_t Resize(uint_t chunks,uint_t buffers,uint_t checkpoint);

  void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk)
  {
    matrix_ = (thrust::complex<double>*)&mat[0];
  }
  void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk)
  {
    params_ = (uint_t*)&prm[0];
  }

  void Set(uint_t i,const thrust::complex<data_t>& t)
  {
    data_[i] = t;
  }
  thrust::complex<data_t> Get(uint_t i) const
  {
    return data_[i];
  }

  thrust::complex<data_t>* chunk_pointer(void)
  {
    return (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data());
  }

  bool peer_access(int i_dest)
  {
#ifdef AER_ATS
    //for IBM AC922
    return true;
#else
    return false;
#endif
  }

  void CopyIn(Chunk<data_t>* src,uint_t iChunk);
  void CopyOut(Chunk<data_t>* src,uint_t iChunk);
  void CopyIn(thrust::complex<data_t>* src,uint_t iChunk);
  void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk);
  void Swap(Chunk<data_t>* src,uint_t iChunk);

  void Zero(uint_t iChunk,uint_t count);

  reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds) const;

  template <typename Function>
  void Execute(Function func,uint_t iChunk,uint_t count);

  template <typename Function>
  double ExecuteSum(Function func,uint_t iChunk,uint_t count) const;

  template <typename Function>
  thrust::complex<double> ExecuteComplexSum(Function func,uint_t iChunk,uint_t count) const;


};

template <typename data_t>
HostChunkContainer<data_t>::~HostChunkContainer(void)
{
  data_.clear();
}

template <typename data_t>
uint_t HostChunkContainer<data_t>::Allocate(int idev,int bits,uint_t chunks,uint_t buffers,uint_t checkpoint)
{
  uint_t nc = chunks;
  uint_t i;

  ChunkContainer<data_t>::chunk_bits_ = bits;

  ChunkContainer<data_t>::num_buffers_ = buffers;
  ChunkContainer<data_t>::num_checkpoint_ = checkpoint;
  ChunkContainer<data_t>::num_chunks_ = nc;
  data_.resize((nc + buffers + checkpoint) << bits);

  this->chunk_mapped_.resize(nc);
  for(i=0;i<nc;i++){
    this->chunk_mapped_[i] = false;
  }
  this->buffer_mapped_.resize(buffers);
  for(i=0;i<buffers;i++){
    this->buffer_mapped_[i] = false;
  }
  this->checkpoint_mapped_.resize(checkpoint);
  for(i=0;i<checkpoint;i++){
    this->checkpoint_mapped_[i] = false;
  }
  return nc;
}

template <typename data_t>
uint_t HostChunkContainer<data_t>::Resize(uint_t chunks,uint_t buffers,uint_t checkpoint)
{
  uint_t i;

  if(chunks + buffers + checkpoint > this->num_chunks_ + this->num_buffers_ + this->num_checkpoint_){
    data_.resize((chunks + buffers + checkpoint) << this->chunk_bits_);
  }

  if(chunks > this->num_chunks_){
    this->chunk_mapped_.resize(chunks);
    for(i=this->num_chunks_;i<chunks;i++){
      this->chunk_mapped_[i] = false;
    }
  }
  this->num_chunks_ = chunks;

  if(buffers > this->num_buffers_){
    this->buffer_mapped_.resize(buffers);
    for(i=this->num_buffers_;i<buffers;i++){
      this->buffer_mapped_[i] = false;
    }
  }
  this->num_buffers_ = buffers;

  if(checkpoint > this->num_checkpoint_){
    this->checkpoint_mapped_.resize(checkpoint);
    for(i=this->num_checkpoint_;i<checkpoint;i++){
      this->checkpoint_mapped_[i] = false;
    }
  }
  this->num_checkpoint_ = checkpoint;

  return chunks + buffers + checkpoint;
}

template <typename data_t>
void HostChunkContainer<data_t>::Deallocate(void)
{
  data_.clear();
}

template <typename data_t>
template <typename Function>
void HostChunkContainer<data_t>::Execute(Function func,uint_t iChunk,uint_t count)
{
  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data()) + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_);
  func.set_params( params_);

  if(omp_get_num_threads() > 1){  //in parallel region
    auto ci = thrust::counting_iterator<uint_t>(0);

    thrust::for_each_n(thrust::host, ci, count, func);
  }
  else{
#pragma omp parallel 
    {
      int nid = omp_get_num_threads();
      int tid = omp_get_thread_num();
      uint_t is,ie;

      auto ci = thrust::counting_iterator<uint_t>(0);

      is = (uint_t)tid * count / (uint_t)nid;
      ie = (uint_t)(tid + 1) * count / (uint_t)nid;

      thrust::for_each_n(thrust::host, ci + is, ie-is, func);
    }
  }
}

template <typename data_t>
template <typename Function>
double HostChunkContainer<data_t>::ExecuteSum(Function func,uint_t iChunk,uint_t count) const
{
  double ret = 0.0;

  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data())  + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_);
  func.set_params( params_);

  if(omp_get_num_threads() > 1){  //in parallel region
    auto ci = thrust::counting_iterator<uint_t>(0);

    ret = thrust::transform_reduce(thrust::host, ci, ci + count, func,0.0,thrust::plus<double>());
  }
  else{
#pragma omp parallel reduction(+:ret)
    {
      int nid = omp_get_num_threads();
      int tid = omp_get_thread_num();
      uint_t is,ie;

      auto ci = thrust::counting_iterator<uint_t>(0);

      is = (uint_t)tid * count / (uint_t)nid;
      ie = (uint_t)(tid + 1) * count / (uint_t)nid;

      ret += thrust::transform_reduce(thrust::host, ci + is, ci + ie, func,0.0,thrust::plus<double>());
    }
  }

  return ret;
}

template <typename data_t>
template <typename Function>
thrust::complex<double> HostChunkContainer<data_t>::ExecuteComplexSum(Function func,uint_t iChunk,uint_t count) const
{
  thrust::complex<double> ret = 0.0;
  thrust::complex<double> zero = 0.0;

  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data())  + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_);
  func.set_params( params_);

  if(omp_get_num_threads() > 1){  //in parallel region
    auto ci = thrust::counting_iterator<uint_t>(0);

    ret = thrust::transform_reduce(thrust::host, ci, ci + count, func,zero,thrust::plus<thrust::complex<double>>());
  }
  else{
    double re = 0.0,im = 0.0;
#pragma omp parallel reduction(+:re,im)
    {
      int nid = omp_get_num_threads();
      int tid = omp_get_thread_num();
      uint_t is,ie;
      thrust::complex<double> sum;

      auto ci = thrust::counting_iterator<uint_t>(0);

      is = (uint_t)tid * count / (uint_t)nid;
      ie = (uint_t)(tid + 1) * count / (uint_t)nid;

      sum = thrust::transform_reduce(thrust::host, ci + is, ci + ie, func,zero,thrust::plus<thrust::complex<double>>());
      re += sum.real();
      im += sum.imag();
    }
    ret = thrust::complex<double>(re,im);
  }

  return ret;
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyIn(Chunk<data_t>* src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;

  if(src->device() >= 0){
    src->set_device();
    DeviceChunkContainer<data_t>* src_cont = (DeviceChunkContainer<data_t>*)src->container();
    thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
  }
  else{
    HostChunkContainer<data_t>* src_cont = (HostChunkContainer<data_t>*)src->container();

    if(omp_get_num_threads() > 1){  //in parallel region
      thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
    }
    else{
#pragma omp parallel
      {
        int nid = omp_get_num_threads();
        int tid = omp_get_thread_num();
        uint_t is,ie;

        is = (uint_t)(tid) * size / (uint_t)(nid);
        ie = (uint_t)(tid + 1) * size / (uint_t)(nid);

        thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_) + is,ie - is,data_.begin() + (iChunk << this->chunk_bits_) + is);
      }
    }
  }
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyOut(Chunk<data_t>* dest,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  if(dest->device() >= 0){
    dest->set_device();
    DeviceChunkContainer<data_t>* dest_cont = (DeviceChunkContainer<data_t>*)dest->container();
    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest_cont->vector().begin() + (dest->pos() << this->chunk_bits_));
  }
  else{
    HostChunkContainer<data_t>* dest_cont = (HostChunkContainer<data_t>*)dest->container();

    if(omp_get_num_threads() > 1){  //in parallel region
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest_cont->vector().begin() + (dest->pos() << this->chunk_bits_));
    }
    else{
#pragma omp parallel
      {
        int nid = omp_get_num_threads();
        int tid = omp_get_thread_num();
        uint_t is,ie;

        is = (uint_t)(tid) * size / (uint_t)(nid);
        ie = (uint_t)(tid + 1) * size / (uint_t)(nid);
        thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_)+is,ie-is,dest_cont->vector().begin() + (dest->pos() << this->chunk_bits_)+is);
      }
    }
  }
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyIn(thrust::complex<data_t>* src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;

  thrust::copy_n(src,size,data_.begin() + (iChunk << this->chunk_bits_));
}

template <typename data_t>
void HostChunkContainer<data_t>::CopyOut(thrust::complex<data_t>* dest,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest);
}

template <typename data_t>
void HostChunkContainer<data_t>::Swap(Chunk<data_t>* src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  if(src->device() >= 0){
    src->set_device();

    AERHostVector<thrust::complex<data_t>> tmp1(size);
    DeviceChunkContainer<data_t>* src_cont = (DeviceChunkContainer<data_t>*)src->container();

    thrust::copy_n(thrust::omp::par,data_.begin() + (iChunk << this->chunk_bits_),size,tmp1.begin());

    thrust::copy_n(src_cont->vector().begin() + (src->pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
    thrust::copy_n(tmp1.begin(),size,src_cont->vector().begin() + (src->pos() << this->chunk_bits_));
  }
  else{
    HostChunkContainer<data_t>* src_cont = (HostChunkContainer<data_t>*)src->container();

    if(omp_get_num_threads() > 1){  //in parallel region
      thrust::swap_ranges(thrust::host,data_.begin() + (iChunk << this->chunk_bits_),data_.begin() + (iChunk << this->chunk_bits_) + size,src_cont->vector().begin() + (src->pos() << this->chunk_bits_));
    }
    else{
#pragma omp parallel
      {
        int nid = omp_get_num_threads();
        int tid = omp_get_thread_num();
        uint_t is,ie;

        is = (uint_t)(tid) * size / (uint_t)(nid);
        ie = (uint_t)(tid + 1) * size / (uint_t)(nid);
        thrust::swap_ranges(thrust::host,data_.begin() + (iChunk << this->chunk_bits_) + is,data_.begin() + (iChunk << this->chunk_bits_) + ie,src_cont->vector().begin() + (src->pos() << this->chunk_bits_) + is);
      }
    }
  }
}


template <typename data_t>
void HostChunkContainer<data_t>::Zero(uint_t iChunk,uint_t count)
{
  if(omp_get_num_threads() > 1){  //in parallel region
    thrust::fill_n(thrust::host,data_.begin() + (iChunk << this->chunk_bits_),count,0.0);
  }
  else{
#pragma omp parallel
    {
      int nid = omp_get_num_threads();
      int tid = omp_get_thread_num();
      uint_t is,ie;

      is = (uint_t)(tid) * count / (uint_t)(nid);
      ie = (uint_t)(tid + 1) * count / (uint_t)(nid);
      thrust::fill_n(thrust::host,data_.begin() + (iChunk << this->chunk_bits_) + is,ie-is,0.0);
    }
  }
}

template <typename data_t>
reg_t HostChunkContainer<data_t>::sample_measure(uint_t iChunk,const std::vector<double> &rnds) const
{
  const int_t SHOTS = rnds.size();
  reg_t samples(SHOTS,0);
  thrust::host_vector<uint_t> vSmp(SHOTS);
  data_t* pVec;
  int i;
  uint_t size = (2 << this->chunk_bits_);

  pVec = (data_t*)(thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data()) + (iChunk << this->chunk_bits_);

  if(omp_get_num_threads() == 1){
    thrust::transform_inclusive_scan(thrust::omp::par,pVec,pVec + size,pVec,thrust::square<double>(),thrust::plus<double>());
    thrust::lower_bound(thrust::omp::par, pVec, pVec + size, rnds.begin(), rnds.begin() + SHOTS, vSmp.begin());
  }
  else{
    thrust::transform_inclusive_scan(thrust::host,pVec,pVec + size,pVec,thrust::square<double>(),thrust::plus<double>());
    thrust::lower_bound(thrust::host, pVec, pVec + size, rnds.begin(), rnds.begin() + SHOTS, vSmp.begin());
  }

  for(i=0;i<SHOTS;i++){
    samples[i] = vSmp[i]/2;
  }
  vSmp.clear();

  return samples;
}

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



//============================================================================
// chunk manager class
// this is static class, there is only 1 manager class
//============================================================================
template <typename data_t>
class ChunkManager 
{
protected:
  std::vector<ChunkContainer<data_t>*> chunks_;         //chunk containers for each device and host

  int num_devices_;            //number of devices
  int num_places_;             //number of places (devices + host)

  int chunk_bits_;             //number of qubits of chunk
  int num_qubits_;             //number of global qubits

  uint_t num_chunks_;          //number of chunks on this process

  int i_dev_map_;              //device index chunk to be mapped
  int idev_buffer_map_;        //device index buffer to be mapped

  int iplace_host_;            //chunk container for host memory
public:
  ChunkManager();

  ~ChunkManager();

  ChunkContainer<data_t>* container(uint_t i)
  {
    if(i < chunks_.size())
      return chunks_[i];
    return NULL;
  }
  uint_t num_containers(void)
  {
    return chunks_.size();
  }

  uint_t Allocate(int chunk_bits,int nqubits,uint_t nchunks);
  void Free(void);

  int num_devices(void)
  {
    return num_devices_;
  }
  int num_places(void)
  {
    return num_places_;
  }
  int place_host(void)
  {
    return iplace_host_;
  }
  uint_t num_chunks(void)
  {
    return num_chunks_;
  }
  int chunk_bits(void)
  {
    return chunk_bits_;
  }
  int num_qubits(void)
  {
    return num_qubits_;
  }

  Chunk<data_t>* MapChunk(int iplace = -1);
  Chunk<data_t>* MapBufferChunk(int idev);
  Chunk<data_t>* MapCheckpoint(Chunk<data_t>* chunk);
  void UnmapChunk(Chunk<data_t>* chunk);
  void UnmapBufferChunk(Chunk<data_t>* buffer);
  void UnmapCheckpoint(Chunk<data_t>* buffer);

};

template <typename data_t>
ChunkManager<data_t>::ChunkManager()
{
  int i,j;

  num_places_ = 1;
  chunk_bits_ = 0;
  num_chunks_ = 0;
  num_qubits_ = 0;

  num_devices_ = 1;

#ifdef AER_THRUST_CUDA
  cudaGetDeviceCount(&num_devices_);
#endif

  chunks_.resize(num_devices_*2 + 1);
  num_places_ = num_devices_;

  iplace_host_ = num_places_ ;

}

template <typename data_t>
ChunkManager<data_t>::~ChunkManager()
{
  Free();

  chunks_.clear();
}

template <typename data_t>
uint_t ChunkManager<data_t>::Allocate(int chunk_bits,int nqubits,uint_t nchunks)
{
  int tid,nid;
  uint_t num_buffers;
  int iDev;
  uint_t is,ie,nc;
  int i;
  char* str;
  bool multi_gpu = false;
  bool hybrid = false;
  uint_t num_checkpoint,total_checkpoint = 0;

  //free previous allocation
  Free();

  num_qubits_ = nqubits;
  chunk_bits_ = chunk_bits;

  i_dev_map_ = 0;

  idev_buffer_map_ = 0;

  str = getenv("AER_MULTI_GPU");
  if(str){
    multi_gpu = true;
    num_places_ = num_devices_;
  }
  str = getenv("AER_HYBRID");
  if(str){
    hybrid = true;
  }

  nid = omp_get_num_threads();
  if(nid > 1){
    //multi-shot parallelization
    multi_gpu = true;
    num_buffers = 0;
    num_places_ = num_devices_;
  }
  else{
    if(chunk_bits == nqubits){    //single chunk
      num_buffers = 0;
      multi_gpu = false;
      num_places_ = 1;
    }
    else{   //multiple-chunks
      num_buffers = AER_MAX_BUFFERS;

      num_places_ = num_devices_;
      if(!multi_gpu){
#ifdef AER_THRUST_CUDA
        size_t freeMem,totalMem;
        cudaSetDevice(0);
        cudaMemGetInfo(&freeMem,&totalMem);
        if(freeMem > ( ((uint_t)sizeof(thrust::complex<data_t>) * (nchunks + num_buffers + AER_DUMMY_BUFFERS)) << chunk_bits_)){
          num_places_ = 1;
        }
#endif
      }
    }
  }

  num_chunks_ = 0;
  for(iDev=0;iDev<num_places_;iDev++){
    is = nchunks * (uint_t)iDev / (uint_t)num_places_;
    ie = nchunks * (uint_t)(iDev + 1) / (uint_t)num_places_;
    nc = ie - is;
    if(hybrid){
      nc /= 2;
    }

    chunks_[iDev] = new DeviceChunkContainer<data_t>;

    num_checkpoint = nc;
#ifdef AER_THRUST_CUDA
    size_t freeMem,totalMem;
    cudaSetDevice(iDev);
    cudaMemGetInfo(&freeMem,&totalMem);
    if(freeMem <= ( ((uint_t)sizeof(thrust::complex<data_t>) * (nc + num_buffers + num_checkpoint + AER_DUMMY_BUFFERS)) << chunk_bits_)){
      num_checkpoint = 0;
    }
#endif
    total_checkpoint += num_checkpoint;
    num_chunks_ += chunks_[iDev]->Allocate(iDev,chunk_bits,nc,num_buffers,num_checkpoint);
  }
  if(num_chunks_ < nchunks){
    for(iDev=0;iDev<num_places_;iDev++){
      chunks_[num_places_ + iDev] = new HostChunkContainer<data_t>;
      is = (nchunks-num_chunks_) * (uint_t)iDev / (uint_t)num_places_;
      ie = (nchunks-num_chunks_) * (uint_t)(iDev + 1) / (uint_t)num_places_;

      chunks_[num_places_ + iDev]->Allocate(-1,chunk_bits,ie-is,AER_MAX_BUFFERS);
    }
    num_places_ *= 2;
    num_chunks_ = nchunks;

    omp_set_nested(1);
  }

  //additional host buffer
  iplace_host_ = num_places_;
  chunks_[iplace_host_] = new HostChunkContainer<data_t>;
  chunks_[iplace_host_]->Allocate(-1,chunk_bits,0,AER_MAX_BUFFERS);

  return num_chunks_;
}

template <typename data_t>
void ChunkManager<data_t>::Free(void)
{
  int i;

  for(i=0;i<chunks_.size();i++){
    if(chunks_[i])
      delete chunks_[i];
    chunks_[i] = NULL;
  }

  chunk_bits_ = 0;
  num_qubits_ = 0;
  num_chunks_ = 0;
}

template <typename data_t>
Chunk<data_t>* ChunkManager<data_t>::MapChunk(int iplace)
{
  Chunk<data_t>* pChunk;
  int i;

  pChunk = NULL;
  while(iplace < num_places_){
    pChunk = chunks_[iplace]->MapChunk();
    if(pChunk){
      pChunk->set_place(iplace);
      break;
    }
    iplace++;
  }

  return pChunk;
}

template <typename data_t>
Chunk<data_t>* ChunkManager<data_t>::MapBufferChunk(int idev)
{
  Chunk<data_t>* pChunk = NULL;

  if(idev < 0){
    int i,iplace;
    for(i=0;i<num_devices_;i++){
      iplace = idev_buffer_map_;

      pChunk = chunks_[idev_buffer_map_++]->MapBufferChunk();
      if(idev_buffer_map_ >= num_devices_)
        idev_buffer_map_ = 0;

      if(pChunk != NULL){
        pChunk->set_place(iplace);
        break;
      }
    }
    return pChunk;
  }

  pChunk = chunks_[idev]->MapBufferChunk();
  if(pChunk != NULL){
    pChunk->set_place(idev);
  }

  return pChunk;
}

template <typename data_t>
Chunk<data_t>* ChunkManager<data_t>::MapCheckpoint(Chunk<data_t>* chunk)
{
  Chunk<data_t>* checkpoint = NULL;
  int iplace = chunk->place();

  if(chunks_[iplace]->num_checkpoint() > 0){
    checkpoint = chunks_[iplace]->MapCheckpoint(chunk->pos());
    if(checkpoint != NULL){
      checkpoint->set_place(iplace);
    }
  }

  if(checkpoint == NULL){
#pragma omp critical
    {
      //map checkpoint on host
      if(chunks_[iplace_host_]->num_checkpoint() == 0){
        chunks_[iplace_host_]->Resize(chunks_[iplace_host_]->num_chunks(),chunks_[iplace_host_]->num_buffers(),num_chunks_);
      }
    }
    checkpoint = chunks_[iplace_host_]->MapCheckpoint(-1);
    if(checkpoint != NULL){
      checkpoint->set_place(iplace_host_);
    }
  }

  return checkpoint;
}


template <typename data_t>
void ChunkManager<data_t>::UnmapChunk(Chunk<data_t>* chunk)
{
  int iPlace = chunk->place();

#pragma omp critical
  {
    chunks_[iPlace]->UnmapChunk(chunk);
    if(chunks_[iPlace]->num_chunk_mapped() == 0){   //last one
      delete chunks_[iPlace];
      chunks_[iPlace] = NULL;
    }
  }
}


template <typename data_t>
void ChunkManager<data_t>::UnmapBufferChunk(Chunk<data_t>* buffer)
{
  chunks_[buffer->place()]->UnmapBuffer(buffer);
}

template <typename data_t>
void ChunkManager<data_t>::UnmapCheckpoint(Chunk<data_t>* buffer)
{
  chunks_[buffer->place()]->UnmapCheckpoint(buffer);
}

//============================================================================
// QubitVectorThrust class
//============================================================================

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize_from_vector(cvector_t<data_t>)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <typename data_t = double>
class QubitVectorThrust {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVectorThrust();
  explicit QubitVectorThrust(size_t num_qubits);
  virtual ~QubitVectorThrust();
  QubitVectorThrust(const QubitVectorThrust& obj){}
  QubitVectorThrust &operator=(const QubitVectorThrust& obj){}

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
  thrust::complex<data_t> &operator[](uint_t element);
  thrust::complex<data_t> operator[](uint_t element) const;

  void set_state(uint_t pos,std::complex<double>& c);
  std::complex<data_t> get_state(uint_t pos) const;

  // Returns a reference to the underlying data_t data class
//  std::complex<data_t>* &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  std::complex<data_t>* data() const {return (std::complex<data_t>*)chunk_->pointer();}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the QubitVector class
#ifdef AER_THRUST_CUDA
  static std::string name() {return "statevector_gpu";}
#else
  static std::string name() {return "statevector_thrust";}
#endif

  // Set the size of the vector in terms of qubit number
  virtual void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const {return num_qubits_;}

  // Returns the size of the underlying n-qubit vector
  uint_t size() const {return data_size_;}

  // Returns required memory
  size_t required_memory_mb(uint_t num_qubits) const;

  // Returns a copy of the underlying data_t data as a complex vector
  cvector_t<data_t> vector() const;

  // Return JSON serialization of QubitVectorThrust;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();


  // State initialization of a component
  // Initialize the specified qubits to a desired statevector
  // (leaving the other qubits in their current state)
  // assuming the qubits being initialized have already been reset to the zero state
  // (using apply_reset)
  void initialize_component(const reg_t &qubits, const cvector_t<double> &state);

  //chunk setup
  void chunk_setup(int chunk_bits,int num_qubits,uint_t chunk_index,uint_t num_local_chunks);

  //cache control for chunks on host
  void fetch_chunk(void) const;
  void release_chunk(bool write_back = true) const;

  //prepare buffer for MPI send/recv
  void* send_buffer(uint_t& size_in_byte);
  void* recv_buffer(uint_t& size_in_byte);

  //-----------------------------------------------------------------------
  // Check point operations
  //-----------------------------------------------------------------------

  // Create a checkpoint of the current state
  void checkpoint();

  // Revert to the checkpoint
  void revert(bool keep);

  // Compute the inner product of current state with checkpoint state
  std::complex<double> inner_product() const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the data vector does not match the number of qubits
  // an exception is raised.
  void initialize_from_vector(const cvector_t<double> &data);

  // Initializes the vector to a custom initial state.
  // If num_states does not match the number of qubits an exception is raised.
  void initialize_from_data(const std::complex<data_t>* data, const size_t num_states);

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cvector_t<double> &mat);

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t<double> &mat);
  
  // Swap pairs of indicies in the underlying vector
  void apply_permutation_matrix(const reg_t &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a general N-qubit multi-controlled X-gate
  // If N=1 this implements an optimized X gate
  // If N=2 this implements an optimized CX gate
  // If N=3 this implements an optimized Toffoli gate
  void apply_mcx(const reg_t &qubits);

  // Apply a general multi-controlled Y-gate
  // If N=1 this implements an optimized Y gate
  // If N=2 this implements an optimized CY gate
  // If N=3 this implements an optimized CCY gate
  void apply_mcy(const reg_t &qubits);
  
  // Apply a general multi-controlled single-qubit phase gate
  // with diagonal [1, ..., 1, phase]
  // If N=1 this implements an optimized single-qubit phase gate
  // If N=2 this implements an optimized CPhase gate
  // If N=3 this implements an optimized CCPhase gate
  // if phase = -1 this is a Z, CZ, CCZ gate
  void apply_mcphase(const reg_t &qubits, const std::complex<double> phase);

  // Apply a general multi-controlled single-qubit unitary gate
  // If N=1 this implements an optimized single-qubit U gate
  // If N=2 this implements an optimized CU gate
  // If N=3 this implements an optimized CCU gate
  void apply_mcu(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a general multi-controlled SWAP gate
  // If N=2 this implements an optimized SWAP  gate
  // If N=3 this implements an optimized Fredkin gate
  void apply_mcswap(const reg_t &qubits);

  //swap between chunk
  void apply_chunk_swap(const reg_t &qubits, QubitVectorThrust<data_t> &chunk, bool write_back = true);
  void apply_chunk_swap(const reg_t &qubits, uint_t remote_chunk_index);

  //-----------------------------------------------------------------------
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  virtual double probability(const uint_t outcome) const;

  // Return the probabilities for all measurement outcomes in the current vector
  // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  virtual std::vector<double> probabilities() const;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  virtual std::vector<double> probabilities(const reg_t &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  virtual reg_t sample_measure(const std::vector<double> &rnds) const;

  //-----------------------------------------------------------------------
  // Norms
  //-----------------------------------------------------------------------
  
  // Returns the norm of the current vector
  double norm() const;

  // These functions return the norm <psi|A^dagger.A|psi> obtained by
  // applying a matrix A to the vector. It is equivalent to returning the
  // expectation value of A^\dagger A, and could probably be removed because
  // of this->

  // Return the norm for of the vector obtained after apply the 1-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  double norm(const uint_t qubit, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  double norm(const reg_t &qubits, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the 1-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const;

  //-----------------------------------------------------------------------
  // Expectation Value
  //-----------------------------------------------------------------------

  // These functions return the expectation value <psi|A|psi> for a matrix A.
  // If A is hermitian these will return real values, if A is non-Hermitian
  // they in general will return complex values.

  // Return the expectation value of an N-qubit Pauli matrix.
  // The Pauli is input as a length N string of I,X,Y,Z characters.
  double expval_pauli(const reg_t &qubits, const std::string &pauli) const;


  //-----------------------------------------------------------------------
  // JSON configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() {return json_chop_threshold_;}

  //-----------------------------------------------------------------------
  // OpenMP configuration settings
  //-----------------------------------------------------------------------

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint_t get_omp_threads() {return omp_threads_;}

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint_t get_omp_threshold() {return omp_threshold_;}

  //-----------------------------------------------------------------------
  // Optimization configuration settings
  //-----------------------------------------------------------------------

  // Set the sample_measure index size
  void set_sample_measure_index_size(int n) {sample_measure_index_size_ = n;}

  // Get the sample_measure index size
  int get_sample_measure_index_size() {return sample_measure_index_size_;}

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t data_size_;

  mutable Chunk<data_t>* chunk_;
  mutable Chunk<data_t>* buffer_chunk_;
  Chunk<data_t>* checkpoint_;
  Chunk<data_t>* send_chunk_;
  Chunk<data_t>* recv_chunk_;
  static ChunkManager<data_t> chunk_manager_;

  uint_t chunk_index_;
  bool multi_chunk_distribution_;

  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 1;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 1; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------

  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t<data_t> &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t<data_t> &mat, uint_t nqubits) const;
  void check_dimension(const QubitVectorThrust &qv) const;
  void check_checkpoint() const;

  //-----------------------------------------------------------------------
  // Statevector update with Lambda function
  //-----------------------------------------------------------------------
  template <typename Function>
  void apply_function(Function func,const reg_t &qubits) const;

  template <typename Function>
  double apply_function_sum(Function func,const reg_t &qubits) const;

  template <typename Function>
  std::complex<double> apply_function_complex_sum(Function func,const reg_t &qubits) const;


#ifdef AER_DEBUG
  //for debugging
  mutable FILE* debug_fp;
  mutable uint_t debug_count;

  void DebugMsg(const char* str,const reg_t &qubits) const;
  void DebugMsg(const char* str,const int qubit) const;
  void DebugMsg(const char* str) const;
  void DebugMsg(const char* str,const std::complex<double> c) const;
  void DebugMsg(const char* str,const double d) const;
  void DebugMsg(const char* str,const std::vector<double>& v) const;
  void DebugDump(void) const;
#endif
};

template <typename data_t>
ChunkManager<data_t> QubitVectorThrust<data_t>::chunk_manager_;


/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <typename data_t>
inline void to_json(json_t &js, const QubitVectorThrust<data_t> &qv) {
  js = qv.json();
}

template <typename data_t>
json_t QubitVectorThrust<data_t>::json() const 
{
  thrust::complex<data_t> t;
  uint_t i;

  const json_t ZERO = std::complex<data_t>(0.0, 0.0);
  json_t js = json_t(data_size_, ZERO);

#ifdef AER_DEBUG
  DebugMsg("json()");
#endif

  for(i=0;i<data_size_;i++){
    t = chunk_->Get(i);
    js[i][0] = t.real();
    js[i][1] = t.imag();
  }
  return js;
}

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::string error = "QubitVectorThrust: qubit index " + std::to_string(qubit) +
                        " > " + std::to_string(num_qubits_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_matrix(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = 1ull << nqubits;
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVectorThrust: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_vector(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = 1ull << nqubits;
  const auto SIZE = vec.size();
  if (SIZE != DIM) {
    std::string error = "QubitVectorThrust: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_dimension(const QubitVectorThrust &qv) const {
  if (data_size_ != qv.size_) {
    std::string error = "QubitVectorThrust: vectors are different shape " +
                         std::to_string(data_size_) + " != " +
                         std::to_string(qv.num_states_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_checkpoint() const {
  if (checkpoint_ == NULL) {
    throw std::runtime_error("QubitVectorThrust: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust(size_t num_qubits) : num_qubits_(0)
{
  chunk_ = NULL;
  chunk_index_ = 0;
  multi_chunk_distribution_ = false;
  checkpoint_ = NULL;

#ifdef AER_DEBUG
  debug_fp = NULL;
  debug_count = 0;
#endif

  if(num_qubits != 0){
    set_num_qubits(num_qubits);
  }
}

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust() : QubitVectorThrust(0)
{

}

template <typename data_t>
QubitVectorThrust<data_t>::~QubitVectorThrust() 
{
  if(checkpoint_ != NULL){
    chunk_manager_.UnmapCheckpoint(checkpoint_);
    checkpoint_ = NULL;
  }

  if(chunk_ != NULL){
    chunk_manager_.UnmapChunk(chunk_);
    chunk_ = NULL;
  }

#ifdef AER_DEBUG
  if(debug_fp != NULL){
    fflush(debug_fp);
    if(debug_fp != stdout)
      fclose(debug_fp);
    debug_fp = NULL;
  }
#endif

}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
thrust::complex<data_t> &QubitVectorThrust<data_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif

  return (*chunk_)[element];
}


template <typename data_t>
thrust::complex<data_t> QubitVectorThrust<data_t>::operator[](uint_t element) const
{
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
#ifdef AER_DEBUG
    DebugMsg(" calling []");
#endif

  return (*chunk_)[element];
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_state(uint_t pos, std::complex<double>& c)
{
  if(pos < data_size_){
    thrust::complex<data_t> t = c;
    chunk_->Set(pos,t);
  }
}

template <typename data_t>
std::complex<data_t> QubitVectorThrust<data_t>::get_state(uint_t pos) const
{
  std::complex<data_t> ret = 0.0;

  if(pos < data_size_){
    ret = chunk_->Get(pos);
  }
  return ret;
}


template <typename data_t>
cvector_t<data_t> QubitVectorThrust<data_t>::vector() const 
{
  cvector_t<data_t> ret(data_size_, 0.);

  chunk_->CopyOut((thrust::complex<data_t>*)&ret[0]);

  return ret;
}

//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
class initialize_component_1qubit_func : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> s0,s1;
  uint_t mask;
  uint_t offset;
public:
  initialize_component_1qubit_func(int qubit,thrust::complex<double> state0,thrust::complex<double> state1)
  {
    s0 = state0;
    s1 = state1;

    mask = (1ull << qubit) - 1;
    offset = 1ull << qubit;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];

    vec0[i0] = s0*q0;
    vec1[i0] = s1*q0;
  }

  const char* name(void)
  {
    return "initialize_component 1 qubit";
  }
};

template <typename data_t>
class initialize_component_func : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  uint_t matSize;
public:
  initialize_component_func(const cvector_t<double>& mat,const reg_t &qb)
  {
    nqubits = qb.size();
    matSize = 1ull << nqubits;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<double> q0;
    thrust::complex<double> q;
    thrust::complex<double>* state;
    uint_t* qubits;
    uint_t* qubits_sorted;
    uint_t j,k;
    uint_t ii,idx,t;
    uint_t mask;

    //get parameters from iterator
    vec = this->data_;
    state = this->matrix_;
    qubits = this->params_;
    qubits_sorted = qubits + nqubits;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits_sorted[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    q0 = vec[idx];
    for(k=0;k<matSize;k++){
      ii = idx;
      for(j=0;j<nqubits;j++){
        if(((k >> j) & 1) != 0)
          ii += (1ull << qubits[j]);
      }
      q = q0 * state[k];
      vec[ii] = q;
    }
  }

  const char* name(void)
  {
    return "initialize_component";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0) 
{
  if(qubits.size() == 1){
    apply_function(initialize_component_1qubit_func<data_t>(qubits[0],state0[0],state0[1]), qubits);
  }
  else{
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    auto qubits_param = qubits;
    int i;
    for(i=0;i<qubits.size();i++)
      qubits_param.push_back(qubits_sorted[i]);

    chunk_->StoreMatrix(state0);
    chunk_->StoreUintParams(qubits_param);

    apply_function(initialize_component_func<data_t>(state0,qubits_sorted), qubits);
  }
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
  chunk_->Zero();
}


template <typename data_t>
void QubitVectorThrust<data_t>::chunk_setup(int chunk_bits,int num_qubits,uint_t chunk_index,uint_t num_local_chunks)
{
  //only first chunk call allocation function
  if(chunk_manager_.chunk_bits() != chunk_bits || chunk_manager_.num_qubits() != num_qubits || chunk_manager_.num_chunks() != num_local_chunks){
    chunk_manager_.Allocate(chunk_bits,num_qubits,num_local_chunks);
  }

  //set global chunk ID
  chunk_index_ = chunk_index;

  multi_chunk_distribution_ = true;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_num_qubits(size_t num_qubits)
{
  data_size_ = 1ull << num_qubits;
  char* str;
  int i;

#ifdef AER_TIMING
  TimeReset();
  TimeStart(QS_GATE_INIT);
#endif

  int nid = omp_get_num_threads();

  if(checkpoint_){
//    chunk_manager_.UnmapCheckpoint(checkpoint_);
    delete checkpoint_;
    checkpoint_ = NULL;
  }

/*  if(num_qubits_ != num_qubits){
    if(chunk_){
      chunk_manager_.UnmapChunk(chunk_);
    }*/

  if(num_qubits_ != num_qubits || chunk_ == NULL){
    if(!multi_chunk_distribution_){
#pragma omp barrier
#pragma omp single
      {
        chunk_manager_.Allocate(num_qubits,num_qubits,nid);
      }
#pragma omp barrier
    }

    if(chunk_){
      delete chunk_;
    }

    num_qubits_ = num_qubits;
    chunk_ = chunk_manager_.MapChunk(0);
  }


#ifdef AER_DEBUG
  //TODO Migrate to SpdLog
  if(debug_fp == NULL){
//    char filename[1024];
//    sprintf(filename,"logs/debug_%d.txt",getpid());
//    debug_fp = fopen(filename,"a");
    debug_fp = stdout;

    fprintf(debug_fp," ==== Thrust qubit vector initialization %d qubits ====\n",num_qubits_);
    if(chunk_->device() >= 0)
      fprintf(debug_fp,"    TEST : device = %d, pos = %d, place = %d / %d\n",chunk_->device(),chunk_->pos(),chunk_->place(),chunk_manager_.num_places());
    else
      fprintf(debug_fp,"    TEST : allocated on host (place = %d)\n",chunk_->place());
  }
#endif

}

template <typename data_t>
size_t QubitVectorThrust<data_t>::required_memory_mb(uint_t num_qubits) const {

  size_t unit = std::log2(sizeof(std::complex<data_t>));
  size_t shift_mb = std::max<int_t>(0, num_qubits + unit - 20);
  size_t mem_mb = 1ULL << shift_mb;

  int np = 1;
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD,&np);
#endif

  mem_mb /= np;

  return mem_mb;
}


template <typename data_t>
void QubitVectorThrust<data_t>::checkpoint()
{
#ifdef AER_DEBUG
  DebugMsg("calling checkpoint");
//  DebugDump();
#endif

  checkpoint_ = chunk_manager_.MapCheckpoint(chunk_);
  if(checkpoint_){
    chunk_->CopyOut(checkpoint_);
  }

#ifdef AER_DEBUG
  DebugMsg("checkpoint done");
#endif
}


template <typename data_t>
void QubitVectorThrust<data_t>::revert(bool keep) {

#ifdef DEBUG
check_checkpoint();
#endif

#ifdef AER_DEBUG
  DebugMsg("calling revert");
#endif
  if(checkpoint_){
    chunk_->CopyIn(checkpoint_);
    chunk_manager_.UnmapCheckpoint(checkpoint_);
    checkpoint_ = NULL;
  }

#ifdef AER_DEBUG
  DebugMsg("revert");
//  DebugDump();
#endif

}

template <typename data_t>
std::complex<double> QubitVectorThrust<data_t>::inner_product() const
{

#ifdef AER_DEBUG
  DebugMsg("calling inner_product");
#endif

  double dot;
  data_t* vec0;
  data_t* vec1;

  if(checkpoint_ == NULL){
    return std::complex<double>(0.0,0.0);
  }

  chunk_->set_device();

  vec0 = (data_t*)chunk_->pointer();
  if(chunk_->device() >= 0){
    if(chunk_->device() == checkpoint_->device()){
      vec1 = (data_t*)checkpoint_->pointer();

      dot = thrust::inner_product(thrust::device,vec0,vec0 + data_size_*2,vec1,0.0);
    }
    else{
      Chunk<data_t>* pBuffer = chunk_manager_.MapBufferChunk(chunk_->place());
      pBuffer->CopyIn(checkpoint_);
      vec1 = (data_t*)pBuffer->pointer();

      dot = thrust::inner_product(thrust::device,vec0,vec0 + data_size_*2,vec1,0.0);
      chunk_manager_.UnmapBufferChunk(pBuffer);
    }
  }
  else{
    vec1 = (data_t*)checkpoint_->pointer();

    dot = thrust::inner_product(thrust::omp::par,vec0,vec0 + data_size_*2,vec1,0.0);
  }

#ifdef AER_DEBUG
  DebugMsg("inner_product",std::complex<double>(dot,0.0));
#endif

  return std::complex<double>(dot,0.0);
}

template <typename data_t>
void QubitVectorThrust<data_t>::fetch_chunk(void) const
{
  if(chunk_->device() < 0){
    do{
      buffer_chunk_ = chunk_manager_.MapBufferChunk(chunk_->place() - chunk_manager_.num_devices());
    }while(buffer_chunk_ == NULL);
    chunk_->set_cache(buffer_chunk_);
    buffer_chunk_->CopyIn(chunk_);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_chunk(bool write_back) const
{
  if(chunk_->device() < 0){
    buffer_chunk_->CopyOut(chunk_);
    chunk_manager_.UnmapBufferChunk(buffer_chunk_);
    chunk_->set_cache(NULL);
    buffer_chunk_ = NULL;
  }
}

template <typename data_t>
void* QubitVectorThrust<data_t>::send_buffer(uint_t& size_in_byte)
{
  void* pRet;

  send_chunk_ = NULL;
#ifdef AER_DISABLE_GDR
  if(chunk_->device() < 0){
    pRet = chunk_->pointer();
  }
  else{   //if there is no GPUDirectRDMA support, copy chunk on CPU before using MPI
    send_chunk_ = chunk_manager_.MapBufferChunk(chunk_manager_.num_devices_());
    chunk_->CopyOut(send_chunk_);
    pRet = send_chunk_->pointer();
  }
#else
    pRet = chunk_->pointer();
#endif

  size_in_byte = (uint_t)sizeof(thrust::complex<data_t>) << num_qubits_;
  return pRet;
}

template <typename data_t>
void* QubitVectorThrust<data_t>::recv_buffer(uint_t& size_in_byte)
{

#ifdef AER_DISABLE_GDR
  if(chunk_->device() < 0){
    recv_chunk_ = chunk_manager_.MapBufferChunk(chunk_->place());
  }
  else{   //if there is no GPUDirectRDMA support, receive in CPU memory
    recv_chunk_ = chunk_manager_.MapBufferChunk(chunk_manager_.num_devices_());
  }
#else
    recv_chunk_ = chunk_manager_.MapBufferChunk(chunk_->place());
#endif
  if(recv_chunk_ == NULL){
    throw std::runtime_error("QubitVectorThrust: receive buffer can not be allocated");
  }

  size_in_byte = (uint_t)sizeof(thrust::complex<data_t>) << num_qubits_;
  return recv_chunk_->pointer();
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::initialize()
{
  zero();

  thrust::complex<data_t> t;
  t = 1.0;

  if(chunk_index_ == 0){
    chunk_->Set(0,t);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_vector(const cvector_t<double> &statevec) 
{
  uint_t offset = chunk_index_ << num_qubits_;
  if(data_size_ < statevec.size() - offset) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" + 
                        std::to_string(data_size_) + "!=" +
                        std::to_string(statevec.size() - offset) + ")";
    throw std::runtime_error(error);
  }
#ifdef AER_DEBUG
  DebugMsg("calling initialize_from_vector");
#endif

  cvector_t<data_t> tmp(data_size_);
  int_t i;

#pragma omp parallel for
  for(i=0;i<data_size_;i++){
    tmp[i] = statevec[offset + i];
  }

  chunk_->CopyIn((thrust::complex<data_t>*)&tmp[0]);

#ifdef AER_DEBUG
  DebugMsg("initialize_from_vector");
  DebugDump();
#endif

}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_data(const std::complex<data_t>* statevec, const size_t num_states) {
  if (data_size_ != num_states) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" +
                        std::to_string(data_size_) + "!=" + std::to_string(num_states) + ")";
    throw std::runtime_error(error);
  }

#ifdef AER_DEBUG
  DebugMsg("calling initialize_from_data");
#endif

  chunk_->CopyIn((thrust::complex<data_t>*)statevec);

#ifdef AER_DEBUG
  DebugMsg("initialize_from_data");
  DebugDump();
#endif

}

//--------------------------------------------------------------------------------------
//  gate kernel execution
//--------------------------------------------------------------------------------------

template <typename data_t>
template <typename Function>
void QubitVectorThrust<data_t>::apply_function(Function func,const reg_t &qubits) const
{
  const size_t N = qubits.size();
  uint_t size;

#ifdef AER_DEBUG
  DebugMsg(func.name(),qubits);
#endif


  func.set_base_index(chunk_index_ << num_qubits_);
  if(multi_chunk_distribution_ && chunk_->device() >= 0){
    if(chunk_->pos() == 0){   //only first chunk on device calculates all the chunks
      size = func.size(num_qubits_,N) * chunk_->container()->num_chunks();

      chunk_->Execute(func,size);
    }
  }
  else{
    size = func.size(num_qubits_,N);
    chunk_->Execute(func,size);
  }

#ifdef AER_DEBUG
  DebugDump();
#endif
}

template <typename data_t>
template <typename Function>
double QubitVectorThrust<data_t>::apply_function_sum(Function func,const reg_t &qubits) const
{
  const size_t N = qubits.size();
  uint_t size;
  double ret = 0.0;

#ifdef AER_DEBUG
  DebugMsg(func.name(),qubits);
#endif

  func.set_base_index(chunk_index_ << num_qubits_);
  size = func.size(num_qubits_,N);
  ret = chunk_->ExecuteSum(func,size);

  return ret;
}

template <typename data_t>
template <typename Function>
std::complex<double> QubitVectorThrust<data_t>::apply_function_complex_sum(Function func,const reg_t &qubits) const
{
  const size_t N = qubits.size();
  uint_t size;
  thrust::complex<double> ret;

#ifdef AER_DEBUG
  DebugMsg(func.name(),qubits);
#endif

  func.set_base_index(chunk_index_ << num_qubits_);
  size = func.size(num_qubits_,N);
  ret = chunk_->ExecuteComplexSum(func,size);

  return *(std::complex<double>*)&ret;
}

/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <typename data_t>
void QubitVectorThrust<data_t>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template <typename data_t>
class MatrixMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit;
  uint_t mask;
  uint_t offset0;

public:
  MatrixMult2x2(const cvector_t<double>& mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    mask = (1ull << qubit) - 1;

    offset0 = 1ull << qubit;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset0;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];

    vec0[i0] = m0 * q0 + m2 * q1;
    vec1[i0] = m1 * q0 + m3 * q1;
  }
  const char* name(void)
  {
    return "mult2x2";
  }
};


template <typename data_t>
class MatrixMult4x4 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m00,m10,m20,m30;
  thrust::complex<double> m01,m11,m21,m31;
  thrust::complex<double> m02,m12,m22,m32;
  thrust::complex<double> m03,m13,m23,m33;
  uint_t mask0;
  uint_t mask1;
  uint_t offset0;
  uint_t offset1;

public:
  MatrixMult4x4(const cvector_t<double>& mat,int qubit0,int qubit1)
  {
    m00 = mat[0];
    m01 = mat[1];
    m02 = mat[2];
    m03 = mat[3];

    m10 = mat[4];
    m11 = mat[5];
    m12 = mat[6];
    m13 = mat[7];

    m20 = mat[8];
    m21 = mat[9];
    m22 = mat[10];
    m23 = mat[11];

    m30 = mat[12];
    m31 = mat[13];
    m32 = mat[14];
    m33 = mat[15];

    offset0 = 1ull << qubit0;
    offset1 = 1ull << qubit1;
    if(qubit0 < qubit1){
      mask0 = offset0 - 1;
      mask1 = offset1 - 1;
    }
    else{
      mask0 = offset1 - 1;
      mask1 = offset0 - 1;
    }
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;
    thrust::complex<data_t>* vec2;
    thrust::complex<data_t>* vec3;
    thrust::complex<data_t> q0,q1,q2,q3;

    vec0 = this->data_;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    vec1 = vec0 + offset0;
    vec2 = vec0 + offset1;
    vec3 = vec2 + offset0;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    vec0[i0] = m00 * q0 + m10 * q1 + m20 * q2 + m30 * q3;
    vec1[i0] = m01 * q0 + m11 * q1 + m21 * q2 + m31 * q3;
    vec2[i0] = m02 * q0 + m12 * q1 + m22 * q2 + m32 * q3;
    vec3[i0] = m03 * q0 + m13 * q1 + m23 * q2 + m33 * q3;
  }
  const char* name(void)
  {
    return "mult4x4";
  }
};

template <typename data_t>
class MatrixMult8x8 : public GateFuncBase<data_t>
{
protected:
  uint_t offset0;
  uint_t offset1;
  uint_t offset2;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;

public:
  MatrixMult8x8(const reg_t &qubit,const reg_t &qubit_ordered)
  {
    offset0 = (1ull << qubit[0]);
    offset1 = (1ull << qubit[1]);
    offset2 = (1ull << qubit[2]);

    mask0 = (1ull << qubit_ordered[0]) - 1;
    mask1 = (1ull << qubit_ordered[1]) - 1;
    mask2 = (1ull << qubit_ordered[2]) - 1;
  }


  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2,i3;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
    thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
    thrust::complex<double>* pMat;

    vec = this->data_;
    pMat = this->matrix_;

    i0 = i & mask0;
    i3 = (i - i0) << 1;
    i1 = i3 & mask1;
    i3 = (i3 - i1) << 1;
    i2 = i3 & mask2;
    i3 = (i3 - i2) << 1;

    i0 = i0 + i1 + i2 + i3;

    q0 = vec[i0];
    q1 = vec[i0 + offset0];
    q2 = vec[i0 + offset1];
    q3 = vec[i0 + offset1 + offset0];
    q4 = vec[i0 + offset2];
    q5 = vec[i0 + offset2 + offset0];
    q6 = vec[i0 + offset2 + offset1];
    q7 = vec[i0 + offset2 + offset1 + offset0];

    m0 = pMat[0];
    m1 = pMat[8];
    m2 = pMat[16];
    m3 = pMat[24];
    m4 = pMat[32];
    m5 = pMat[40];
    m6 = pMat[48];
    m7 = pMat[56];

    vec[i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[1];
    m1 = pMat[9];
    m2 = pMat[17];
    m3 = pMat[25];
    m4 = pMat[33];
    m5 = pMat[41];
    m6 = pMat[49];
    m7 = pMat[57];

    vec[i0 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[2];
    m1 = pMat[10];
    m2 = pMat[18];
    m3 = pMat[26];
    m4 = pMat[34];
    m5 = pMat[42];
    m6 = pMat[50];
    m7 = pMat[58];

    vec[i0 + offset1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[3];
    m1 = pMat[11];
    m2 = pMat[19];
    m3 = pMat[27];
    m4 = pMat[35];
    m5 = pMat[43];
    m6 = pMat[51];
    m7 = pMat[59];

    vec[i0 + offset1 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[4];
    m1 = pMat[12];
    m2 = pMat[20];
    m3 = pMat[28];
    m4 = pMat[36];
    m5 = pMat[44];
    m6 = pMat[52];
    m7 = pMat[60];

    vec[i0 + offset2] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[5];
    m1 = pMat[13];
    m2 = pMat[21];
    m3 = pMat[29];
    m4 = pMat[37];
    m5 = pMat[45];
    m6 = pMat[53];
    m7 = pMat[61];

    vec[i0 + offset2 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[6];
    m1 = pMat[14];
    m2 = pMat[22];
    m3 = pMat[30];
    m4 = pMat[38];
    m5 = pMat[46];
    m6 = pMat[54];
    m7 = pMat[62];

    vec[i0 + offset2 + offset1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[7];
    m1 = pMat[15];
    m2 = pMat[23];
    m3 = pMat[31];
    m4 = pMat[39];
    m5 = pMat[47];
    m6 = pMat[55];
    m7 = pMat[63];

    vec[i0 + offset2 + offset1 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
  }
  const char* name(void)
  {
    return "mult8x8";
  }
};

template <typename data_t>
class MatrixMult16x16 : public GateFuncBase<data_t>
{
protected:
  uint_t offset0;
  uint_t offset1;
  uint_t offset2;
  uint_t offset3;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;
  uint_t mask3;
public:
  MatrixMult16x16(const reg_t &qubit,const reg_t &qubit_ordered)
  {
    offset0 = (1ull << qubit[0]);
    offset1 = (1ull << qubit[1]);
    offset2 = (1ull << qubit[2]);
    offset3 = (1ull << qubit[3]);

    mask0 = (1ull << qubit_ordered[0]) - 1;
    mask1 = (1ull << qubit_ordered[1]) - 1;
    mask2 = (1ull << qubit_ordered[2]) - 1;
    mask3 = (1ull << qubit_ordered[3]) - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2,i3,i4,offset,f0,f1,f2;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
    thrust::complex<data_t> q8,q9,q10,q11,q12,q13,q14,q15;
    thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
    thrust::complex<double> m8,m9,m10,m11,m12,m13,m14,m15;
    thrust::complex<double>* pMat;
    int j;

    vec = this->data_;
    pMat = this->matrix_;

    i0 = i & mask0;
    i4 = (i - i0) << 1;
    i1 = i4 & mask1;
    i4 = (i4 - i1) << 1;
    i2 = i4 & mask2;
    i4 = (i4 - i2) << 1;
    i3 = i4 & mask3;
    i4 = (i4 - i3) << 1;

    i0 = i0 + i1 + i2 + i3 + i4;

    q0 = vec[i0];
    q1 = vec[i0 + offset0];
    q2 = vec[i0 + offset1];
    q3 = vec[i0 + offset1 + offset0];
    q4 = vec[i0 + offset2];
    q5 = vec[i0 + offset2 + offset0];
    q6 = vec[i0 + offset2 + offset1];
    q7 = vec[i0 + offset2 + offset1 + offset0];
    q8 = vec[i0 + offset3];
    q9 = vec[i0 + offset3 + offset0];
    q10 = vec[i0 + offset3 + offset1];
    q11 = vec[i0 + offset3 + offset1 + offset0];
    q12 = vec[i0 + offset3 + offset2];
    q13 = vec[i0 + offset3 + offset2 + offset0];
    q14 = vec[i0 + offset3 + offset2 + offset1];
    q15 = vec[i0 + offset3 + offset2 + offset1 + offset0];

    offset = 0;
    f0 = 0;
    f1 = 0;
    f2 = 0;
    for(j=0;j<16;j++){
      m0 = pMat[0+j];
      m1 = pMat[16+j];
      m2 = pMat[32+j];
      m3 = pMat[48+j];
      m4 = pMat[64+j];
      m5 = pMat[80+j];
      m6 = pMat[96+j];
      m7 = pMat[112+j];
      m8 = pMat[128+j];
      m9 = pMat[144+j];
      m10= pMat[160+j];
      m11= pMat[176+j];
      m12= pMat[192+j];
      m13= pMat[208+j];
      m14= pMat[224+j];
      m15= pMat[240+j];

      offset = offset3 * (((uint_t)j >> 3) & 1) + 
               offset2 * (((uint_t)j >> 2) & 1) + 
               offset1 * (((uint_t)j >> 1) & 1) + 
               offset0 *  ((uint_t)j & 1);

      vec[i0 + offset] =   m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
              m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;
    }
  }
  const char* name(void)
  {
    return "mult16x16";
  }
};


//in-place NxN matrix multiplication using LU factorization
template <typename data_t>
class MatrixMultNxN_LU : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  uint_t matSize;
  int nswap;
public:
  MatrixMultNxN_LU(const cvector_t<double>& mat,const reg_t &qb,cvector_t<double>& matLU,reg_t& params)
  {
    uint_t i,j,k,imax;
    std::complex<double> c0,c1;
    double d,dmax;
    uint_t* pSwap;

    nqubits = qb.size();
    matSize = 1ull << nqubits;

    matLU = mat;
    params.resize(nqubits + matSize*2);

    for(k=0;k<nqubits;k++){
      params[k] = qb[k];
    }

    //LU factorization of input matrix
    for(i=0;i<matSize;i++){
      params[nqubits + i] = i;  //init pivot
    }
    for(i=0;i<matSize;i++){
      imax = i;
      dmax = std::abs(matLU[(i << nqubits) + params[nqubits + i]]);
      for(j=i+1;j<matSize;j++){
        d = std::abs(matLU[(i << nqubits) + params[nqubits + j]]);
        if(d > dmax){
          dmax = d;
          imax = j;
        }
      }
      if(imax != i){
        j = params[nqubits + imax];
        params[nqubits + imax] = params[nqubits + i];
        params[nqubits + i] = j;
      }

      if(dmax != 0){
        c0 = matLU[(i << nqubits) + params[nqubits + i]];

        for(j=i+1;j<matSize;j++){
          c1 = matLU[(i << nqubits) + params[nqubits + j]]/c0;

          for(k=i+1;k<matSize;k++){
            matLU[(k << nqubits) + params[nqubits + j]] -= c1*matLU[(k << nqubits) + params[nqubits + i]];
          }
          matLU[(i << nqubits) + params[nqubits + j]] = c1;
        }
      }
    }

    //making table for swapping pivotted result
    pSwap = new uint_t[matSize];
    nswap = 0;
    for(i=0;i<matSize;i++){
      pSwap[i] = params[nqubits + i];
    }
    i = 0;
    while(i<matSize){
      if(pSwap[i] != i){
        params[nqubits + matSize + nswap++] = i;
        j = pSwap[i];
        params[nqubits + matSize + nswap++] = j;
        k = pSwap[j];
        pSwap[j] = j;
        while(i != k){
          j = k;
          params[nqubits + matSize + nswap++] = k;
          k = pSwap[j];
          pSwap[j] = j;
        }
        pSwap[i] = i;
      }
      i++;
    }
    delete[] pSwap;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q,qt;
    thrust::complex<double> m;
    thrust::complex<double> r;
    uint_t j,k,l,iq;
    uint_t ii,idx,t;
    uint_t mask,offset_j,offset_k;
    thrust::complex<data_t>* vec;
    thrust::complex<double>* pMat;
    uint_t* qubits;
    uint_t* pivot;
    uint_t* table;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;

    pivot = qubits + nqubits;
    table = pivot + matSize;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    //mult U
    for(j=0;j<matSize;j++){
      r = 0.0;
      for(k=j;k<matSize;k++){
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];

        offset_k = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        q = vec[offset_k+idx];

        r += m*q;
      }
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      vec[offset_j+idx] = r;
    }

    //mult L
    for(j=matSize-1;j>0;j--){
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      r = vec[offset_j+idx];

      for(k=0;k<j;k++){
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];

        offset_k = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        q = vec[offset_k+idx];

        r += m*q;
      }
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      vec[offset_j+idx] = r;
    }

    //swap results
    if(nswap > 0){
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((table[0] >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      q = vec[offset_j+idx];
      k = pivot[table[0]];
      for(j=1;j<nswap;j++){
        offset_j = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((table[j] >> iq) & 1) != 0)
            offset_j += (1ull << qubits[iq]);
        }
        qt = vec[offset_j+idx];

        offset_k = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        vec[offset_k+idx] = q;
        q = qt;
        k = pivot[table[j]];
      }
      offset_k = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((k >> iq) & 1) != 0)
          offset_k += (1ull << qubits[iq]);
      }
      vec[offset_k+idx] = q;
    }
  }
  const char* name(void)
  {
    return "multNxN";
  }
};



template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &mat)
{
  const size_t N = qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

#ifdef AER_TIMING
  TimeStart(QS_GATE_MULT);
#endif

  if(N == 1){
    apply_function(MatrixMult2x2<data_t>(mat,qubits[0]), qubits);
  }
  else if(N == 2){
    apply_function(MatrixMult4x4<data_t>(mat,qubits[0],qubits[1]), qubits);
  }
  else if(N == 3){
    chunk_->StoreMatrix(mat);
    apply_function(MatrixMult8x8<data_t>(qubits,qubits_sorted), qubits);
  }
  else if(N == 4){
    chunk_->StoreMatrix(mat);
    apply_function(MatrixMult16x16<data_t>(qubits,qubits_sorted), qubits);
  }
  else{
    cvector_t<double> matLU;
    reg_t params;
    ///TODO : swap matrix to fit sorted qubits
    MatrixMultNxN_LU<data_t> f(mat,qubits_sorted,matLU,params);

    chunk_->StoreMatrix(matLU);
    chunk_->StoreUintParams(params);

    apply_function(f, qubits);
  }

#ifdef AER_TIMING
  TimeEnd(QS_GATE_MULT);
#endif

}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_multiplexer(const reg_t &control_qubits,
                                            const reg_t &target_qubits,
                                            const cvector_t<double>  &mat)
{
  const size_t control_count = control_qubits.size();
  const size_t target_count  = target_qubits.size();
  const uint_t DIM = 1ull << (target_count+control_count);
  const uint_t columns = 1ull << target_count;
  const uint_t blocks = 1ull << control_count;

  auto qubits = target_qubits;
  for (const auto &q : control_qubits) {qubits.push_back(q);}
  size_t N = qubits.size();

  cvector_t<double> matMP(DIM*DIM,0.0);
  uint_t b,i,j;

  //make DIMxDIM matrix
  for(b = 0; b < blocks; b++){
    for(i = 0; i < columns; i++){
      for(j = 0; j < columns; j++){
        matMP[(i+b*columns) + DIM*(b*columns+j)] += mat[i+b*columns + DIM * j];
      }
    }
  }

#ifdef AER_DEBUG
  DebugMsg("apply_multiplexer",control_qubits);
  DebugMsg("                 ",target_qubits);
#endif

  apply_matrix(qubits,matMP);
}

template <typename data_t>
class DiagonalMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1;
  int qubit;
public:

  DiagonalMult2x2(const cvector_t<double>& mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    thrust::complex<double> m;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if((((i + gid) >> qubit) & 1) == 0){
      m = m0;
    }
    else{
      m = m1;
    }

    vec[i] = m * q;
  }
  const char* name(void)
  {
    return "diagonal_mult2x2";
  }
};

template <typename data_t>
class DiagonalMult4x4 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit0;
  int qubit1;
public:

  DiagonalMult4x4(const cvector_t<double>& mat,int q0,int q1)
  {
    qubit0 = q0;
    qubit1 = q1;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    thrust::complex<double> m;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if((((i+gid) >> qubit1) & 1) == 0){
      if((((i+gid) >> qubit0) & 1) == 0){
        m = m0;
      }
      else{
        m = m1;
      }
    }
    else{
      if((((i+gid) >> qubit0) & 1) == 0){
        m = m2;
      }
      else{
        m = m3;
      }
    }

    vec[i] = m * q;
  }
  const char* name(void)
  {
    return "diagonal_mult4x4";
  }
};

template <typename data_t>
class DiagonalMultNxN : public GateFuncBase<data_t>
{
protected:
  int nqubits;
public:
  DiagonalMultNxN(const reg_t &qb)
  {
    nqubits = qb.size();
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t j,im;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double>* pMat;
    uint_t* qubits;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    pMat = this->matrix_;
    qubits = this->params_;

    im = 0;
    for(j=0;j<nqubits;j++){
      if((((i + gid) >> qubits[j]) & 1) != 0){
        im += (1 << j);
      }
    }

    q = vec[i];
    m = pMat[im];

    vec[i] = m * q;
  }
  const char* name(void)
  {
    return "diagonal_multNxN";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag)
{
  const int_t N = qubits.size();

#ifdef AER_TIMING
  TimeStart(QS_GATE_DIAG);
#endif

  if(N == 1){
    apply_function(DiagonalMult2x2<data_t>(diag,qubits[0]), qubits);
  }
  else if(N == 2){
    apply_function(DiagonalMult4x4<data_t>(diag,qubits[0],qubits[1]), qubits);
  }
  else{
    chunk_->StoreMatrix(diag);
    chunk_->StoreUintParams(qubits);

    apply_function(DiagonalMultNxN<data_t>(qubits), qubits);
  }

#ifdef AER_TIMING
  TimeEnd(QS_GATE_DIAG);
#endif
}


template <typename data_t>
class Permutation : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  int npairs;

public:
  Permutation(const reg_t& qubits_sorted,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs,reg_t& params)
  {
    uint_t j,k;
    uint_t offset0,offset1;

    nqubits = qubits.size();
    npairs = pairs.size();

    params.resize(nqubits + npairs*2);

    for(j=0;j<nqubits;j++){ //save masks
      params[j] = (1ull << qubits_sorted[j]) - 1;
    }
    //make offset for pairs
    for(j=0;j<npairs;j++){
      offset0 = 0;
      offset1 = 0;
      for(k=0;k<nqubits;k++){
        if(((pairs[j].first >> k) & 1) != 0){
          offset0 += (1ull << qubits[k]);
        }
        if(((pairs[j].second >> k) & 1) != 0){
          offset1 += (1ull << qubits[k]);
        }
      }
      params[nqubits + j*2  ] = offset0;
      params[nqubits + j*2+1] = offset1;
    }
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    uint_t j;
    uint_t ii,idx,t;
    uint_t* mask;
    uint_t* pairs;

    vec = this->data_;
    mask = this->params_;
    pairs = mask + nqubits;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      t = ii & mask[j];
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    for(j=0;j<npairs;j++){
      q0 = vec[idx + pairs[j*2]];
      q1 = vec[idx + pairs[j*2+1]];

      vec[idx + pairs[j*2]]   = q1;
      vec[idx + pairs[j*2+1]] = q0;
    }
  }
  const char* name(void)
  {
    return "Permutation";
  }
};


template <typename data_t>
void QubitVectorThrust<data_t>::apply_permutation_matrix(const reg_t& qubits,
             const std::vector<std::pair<uint_t, uint_t>> &pairs)
{
  const size_t N = qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  reg_t params;
  Permutation<data_t> f(qubits_sorted,qubits,pairs,params);
  chunk_->StoreUintParams(params);

  apply_function(f, qubits);
}


/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Multi-controlled gates
//------------------------------------------------------------------------------

template <typename data_t>
class CX_func : public GateFuncBase<data_t>
{
protected:
  uint_t offset;
  uint_t mask;
  uint_t cmask;
  int nqubits;
  int qubit_t;
public:

  CX_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits-1];
    offset = 1ull << qubit_t;
    mask = offset - 1;

    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if((i0 & cmask) == cmask){
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = q1;
      vec1[i0] = q0;
    }
  }
  const char* name(void)
  {
    return "CX";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcx(const reg_t &qubits) 
{
#ifdef AER_TIMING
    TimeStart(QS_GATE_CX);
#endif

  apply_function(CX_func<data_t>(qubits), qubits);

#ifdef AER_TIMING
    TimeEnd(QS_GATE_CX);
#endif

}


template <typename data_t>
class CY_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask;
  uint_t cmask;
  uint_t offset;
  int nqubits;
  int qubit_t;
public:
  CY_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits-1];
    offset = (1ull << qubit_t);
    mask = (1ull << qubit_t) - 1;

    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if((i0 & cmask) == cmask){
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = thrust::complex<data_t>(q1.imag(),-q1.real());
      vec1[i0] = thrust::complex<data_t>(-q0.imag(),q0.real());
    }
  }
  const char* name(void)
  {
    return "CY";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcy(const reg_t &qubits) 
{
  apply_function(CY_func<data_t>(qubits), qubits);
}

template <typename data_t>
class CSwap_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask0;
  uint_t mask1;
  uint_t cmask;
  int nqubits;
  int qubit_t0;
  int qubit_t1;
  uint_t offset1;
  uint_t offset2;
public:

  CSwap_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    if(qubits[nqubits-2] < qubits[nqubits-1]){
      qubit_t0 = qubits[nqubits-2];
      qubit_t1 = qubits[nqubits-1];
    }
    else{
      qubit_t1 = qubits[nqubits-2];
      qubit_t0 = qubits[nqubits-1];
    }
    mask0 = (1ull << qubit_t0) - 1;
    mask1 = (1ull << qubit_t1) - 1;

    offset1 = 1ull << qubit_t0;
    offset2 = 1ull << qubit_t1;

    cmask = 0;
    for(i=0;i<nqubits-2;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int num_control_bits(void)
  {
    return nqubits - 2;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2;
    thrust::complex<data_t> q1,q2;
    thrust::complex<data_t>* vec1;
    thrust::complex<data_t>* vec2;

    vec1 = this->data_;

    vec2 = vec1 + offset2;
    vec1 = vec1 + offset1;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    if((i0 & cmask) == cmask){
      q1 = vec1[i0];
      q2 = vec2[i0];
      vec1[i0] = q2;
      vec2[i0] = q1;
    }
  }
  const char* name(void)
  {
    return "CSWAP";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits)
{
  apply_function(CSwap_func<data_t>(qubits), qubits);
}


//swap operator between chunks
template <typename data_t>
class CSwapChunk_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask;
  int qubit_t;
  bool write_back_;
  thrust::complex<data_t>* vec0;
  thrust::complex<data_t>* vec1;
public:

  CSwapChunk_func(const reg_t &qubits,thrust::complex<data_t>* pVec0,thrust::complex<data_t>* pVec1,bool wb)
  {
    int i;
    int nqubits;
    nqubits = qubits.size();

    if(qubits[nqubits-2] < qubits[nqubits-1]){
      qubit_t = qubits[nqubits-2];
    }
    else{
      qubit_t = qubits[nqubits-1];
    }
    mask = (1ull << qubit_t) - 1;

    vec0 = pVec0;
    vec1 = pVec1;

    write_back_ = wb;
  }

  uint_t size(int num_qubits,int n)
  {
    return (1ull << (num_qubits - 1));
  }
  int num_control_bits(void)
  {
    //return 1 to claculate "size = 1ull << (num_qubits_ -1)" in apply_function
    return 1;
  }

  __host__ __device__  void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;

    i0 = i & mask;
    i1 = (i - i0) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];
    vec0[i0] = q1;
    if(write_back_)
      vec1[i0] = q0;
  }
  const char* name(void)
  {
    return "Chunk SWAP";
  }
};


template <typename data_t>
void QubitVectorThrust<data_t>::apply_chunk_swap(const reg_t &qubits, QubitVectorThrust<data_t> &src, bool write_back)
{
  int q0,q1,t;


  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(q0 > q1){
    t = q0;
    q0 = q1;
    q1 = t;
  }

  if(q0 >= num_qubits_){  //exchange whole of chunk each other
#ifdef AER_DEBUG
    DebugMsg("SWAP chunks",qubits);
#endif
    if(write_back){
      chunk_->Swap(src.chunk_);
    }
    else{
      chunk_->CopyIn(src.chunk_);
    }
  }
  else{
    thrust::complex<data_t>* pChunk0;
    thrust::complex<data_t>* pChunk1;
    Chunk<data_t>* pBuffer0 = NULL;
    Chunk<data_t>* pExec;

    if(chunk_->device() >= 0){
      pExec = chunk_;
      if(chunk_->container()->peer_access(src.chunk_->device())){
        pChunk1 = src.chunk_->pointer();
      }
      else{
        do{
          pBuffer0 = chunk_manager_.MapBufferChunk(chunk_->place());
        }while(pBuffer0 == NULL);
        pBuffer0->CopyIn(src.chunk_);
        pChunk1 = pBuffer0->pointer();
      }
      pChunk0 = chunk_->pointer();
    }
    else{
      if(src.chunk_->device() >= 0){
        do{
          pBuffer0 = chunk_manager_.MapBufferChunk(src.chunk_->place());
        }while(pBuffer0 == NULL);
        pBuffer0->CopyIn(chunk_);
        pChunk0 = pBuffer0->pointer();
        pChunk1 = src.chunk_->pointer();
        pExec = src.chunk_;
      }
      else{
        pChunk1 = src.chunk_->pointer();
        pChunk0 = chunk_->pointer();
        pExec = chunk_;
      }
    }

    if(chunk_index_ < src.chunk_index_)
      pChunk0 += (1ull << q0);
    else
      pChunk1 += (1ull << q0);

#ifdef AER_DEBUG
    DebugMsg("chunk swap",qubits);
#endif

    pExec->Execute(CSwapChunk_func<data_t>(qubits,pChunk0,pChunk1,true),data_size_/2);

    if(pBuffer0 != NULL){
      if(pExec == chunk_)
        pBuffer0->CopyOut(src.chunk_);
      else
        pBuffer0->CopyOut(chunk_);
      chunk_manager_.UnmapBufferChunk(pBuffer0);
    }
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_chunk_swap(const reg_t &qubits, uint_t remote_chunk_index)
{
  int q0,q1,t;


  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(q0 > q1){
    t = q0;
    q0 = q1;
    q1 = t;
  }

  if(q0 >= num_qubits_){  //exchange whole of chunk each other
#ifdef AER_DEBUG
    DebugMsg("SWAP chunks between process",qubits);
#endif
    chunk_->CopyIn(recv_chunk_);
  }
  else{
    thrust::complex<data_t>* pLocal;
    thrust::complex<data_t>* pRemote;
    Chunk<data_t>* pBuffer = NULL;

#ifdef AER_DISABLE_GDR
    if(chunk_->device() >= 0){    //if there is no GPUDirectRDMA support, copy chunk from CPU
      pBuffer = chunk_manager_.MapBufferChunk(chunk_->place());
      pBuffer->CopyIn(recv_chunk_);
      pRemote = pBuffer->pointer();
    }
    else{
      pRemote = recv_chunk_->pointer();
    }
#else
    pRemote = recv_chunk_->pointer();
#endif
    pLocal = chunk_->pointer();

    if(chunk_index_ < remote_chunk_index)
      pLocal += (1ull << q0);
    else
      pRemote += (1ull << q0);

#ifdef AER_DEBUG
    DebugMsg("chunk swap (process)",qubits);
#endif

    chunk_->Execute(CSwapChunk_func<data_t>(qubits,pLocal,pRemote,false),data_size_/2);

    if(pBuffer){
      chunk_manager_.UnmapBufferChunk(pBuffer);
    }
  }

  chunk_manager_.UnmapBufferChunk(recv_chunk_);
  recv_chunk_ = NULL;

#ifdef AER_DISABLE_GDR
  if(send_chunk_ != NULL){
    chunk_manager_.UnmapBufferChunk(send_chunk_);
    send_chunk_ = NULL;
  }
#endif
}

template <typename data_t>
class phase_func : public GateFuncBase<data_t> 
{
protected:
  thrust::complex<double> phase;
  uint_t mask;
  int nqubits;
public:
  phase_func(const reg_t &qubits,thrust::complex<double> p)
  {
    int i;
    nqubits = qubits.size();
    phase = p;

    mask = 0;
    for(i=0;i<nqubits;i++){
      mask |= (1ull << qubits[i]);
    }
  }
  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t gid;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;

    vec = this->data_;
    gid = this->base_index_;

    if(((i+gid) & mask) == mask){
      q0 = vec[i];
      vec[i] = q0 * phase;
    }
  }
  const char* name(void)
  {
    return "phase";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcphase(const reg_t &qubits, const std::complex<double> phase)
{
  apply_function(phase_func<data_t>(qubits,*(thrust::complex<double>*)&phase), qubits );
}

template <typename data_t>
class DiagonalMult2x2Controlled : public GateFuncBase<data_t> 
{
protected:
  thrust::complex<double> m0,m1;
  uint_t mask;
  uint_t cmask;
  int nqubits;
public:
  DiagonalMult2x2Controlled(const cvector_t<double>& mat,const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    m0 = mat[0];
    m1 = mat[1];

    mask = (1ull << qubits[nqubits-1]) - 1;
    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t gid;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<double> m;

    vec = this->data_;
    gid = this->base_index_;

    if(((i + gid) & cmask) == cmask){
      if((i + gid) & mask){
        m = m1;
      }
      else{
        m = m0;
      }

      q0 = vec[i];
      vec[i] = m*q0;
    }
  }
  const char* name(void)
  {
    return "diagonal_Cmult2x2";
  }
};

template <typename data_t>
class MatrixMult2x2Controlled : public GateFuncBase<data_t> 
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  uint_t mask;
  uint_t cmask;
  uint_t offset;
  int nqubits;
public:
  MatrixMult2x2Controlled(const cvector_t<double>& mat,const reg_t &qubits)
  {
    int i;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];
    nqubits = qubits.size();

    offset = 1ull << qubits[nqubits-1];
    mask = (1ull << qubits[nqubits-1]) - 1;
    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if((i0 & cmask) == cmask){
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = m0 * q0 + m2 * q1;
      vec1[i0] = m1 * q0 + m3 * q1;
    }
  }
  const char* name(void)
  {
    return "diagonal_CmultNxN";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcu(const reg_t &qubits,
                                    const cvector_t<double> &mat) 
{
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();

  // Check if matrix is actually diagonal and if so use
  // diagonal matrix lambda function
  // TODO: this should be changed to not check doubles with ==
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    // Check if actually a phase gate
    if (mat[0] == 1.0) {
      apply_mcphase(qubits, mat[3]);
      return;
    }
    // Otherwise apply general diagonal gate
    const cvector_t<double> diag = {{mat[0], mat[3]}};

    if(N == 1){
      // If N=1 this is just a single-qubit matrix
      apply_diagonal_matrix(qubits[0], diag);
      return;
    }
    else{
      apply_function(DiagonalMult2x2Controlled<data_t>(diag,qubits), qubits );
    }
  }
  else{
    if(N == 1){
      // If N=1 this is just a single-qubit matrix
      apply_matrix(qubits[0], mat);
      return;
    }
    else{
      apply_function(MatrixMult2x2Controlled<data_t>(mat,qubits), qubits );
    }
  }
}


//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t<double>& mat)
{
  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
#ifdef AER_TIMING
  TimeStart(QS_GATE_DIAG);
#endif
    const std::vector<std::complex<double>> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);

#ifdef AER_TIMING
  TimeEnd(QS_GATE_DIAG);
#endif
    return;
  }
#ifdef AER_TIMING
  TimeStart(QS_GATE_MULT);
#endif

  reg_t qubits = {qubit};
  apply_function(MatrixMult2x2<data_t>(mat,qubit), qubits);

#ifdef AER_TIMING
  TimeEnd(QS_GATE_MULT);
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag) 
{
#ifdef AER_TIMING
  TimeStart(QS_GATE_DIAG);
#endif
  reg_t qubits = {qubit};
  apply_function(DiagonalMult2x2<data_t>(diag,qubits[0]), qubits);

#ifdef AER_TIMING
  TimeEnd(QS_GATE_DIAG);
#endif

}
/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
class Norm : public GateFuncBase<data_t>
{
protected:

public:
  Norm()
  {

  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    double ret;

    vec = this->data_;

    ret = 0.0;

    q0 = vec[i];
    ret = q0.real()*q0.real() + q0.imag()*q0.imag();

    return ret;
  }
  const char* name(void)
  {
    return "Norm";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm() const
{
  double ret;
  reg_t qubits(1,0);
  ret = chunk_->ExecuteSum(Norm<data_t>(),data_size_);
  
//  ret = apply_function_sum(Norm<data_t>(), qubits);

#ifdef AER_DEBUG
  DebugMsg("norm",ret);
#endif

  return ret;
}

template <typename data_t>
class NormMatrixMultNxN : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  uint_t matSize;
public:
  NormMatrixMultNxN(const cvector_t<double>& mat,const reg_t &qb)
  {
    nqubits = qb.size();
    matSize = 1ull << nqubits;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    uint_t offset;
    thrust::complex<double>* pMat;

    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double> r;
    double sum = 0.0;
    uint_t j,k,l,iq;
    uint_t ii,idx,t;
    uint_t mask;
    uint_t* qubits;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    for(j=0;j<matSize;j++){
      r = 0.0;
      for(k=0;k<matSize;k++){
        l = (j + (k << nqubits));
        m = pMat[l];

        offset = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset += (1ull << qubits[iq]);
        }
        q = vec[offset+idx];
        r += m*q;
      }
      sum += (r.real()*r.real() + r.imag()*r.imag());
    }
    return sum;
  }
  const char* name(void)
  {
    return "Norm_multNxN";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const reg_t &qubits, const cvector_t<double> &mat) const 
{
  const size_t N = qubits.size();

  if(N == 1){
    return norm(qubits[0], mat);
  }
  else{

    chunk_->StoreMatrix(mat);
    chunk_->StoreUintParams(qubits);

    double ret = apply_function_sum(NormMatrixMultNxN<data_t>(mat,qubits), qubits);
    return ret;
  }
}

template <typename data_t>
class NormDiagonalMultNxN : public GateFuncBase<data_t>
{
protected:
  int nqubits;
public:
  NormDiagonalMultNxN(const reg_t &qb)
  {
    nqubits = qb.size();
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    uint_t im,j,gid;
    thrust::complex<data_t> q;
    thrust::complex<double> m,r;
    thrust::complex<double>* pMat;
    thrust::complex<data_t>* vec;
    uint_t* qubits;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;
    gid = this->base_index_;

    im = 0;
    for(j=0;j<nqubits;j++){
      if(((i+gid) & (1ull << qubits[j])) != 0){
        im += (1 << j);
      }
    }

    q = vec[i];
    m = pMat[im];

    r = m * q;
    return (r.real()*r.real() + r.imag()*r.imag());
  }
  const char* name(void)
  {
    return "Norm_diagonal_multNxN";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const {

  const uint_t N = qubits.size();

  if(N == 1){
    return norm_diagonal(qubits[0], mat);
  }
  else{
    chunk_->StoreMatrix(mat);
    chunk_->StoreUintParams(qubits);

    double ret = apply_function_sum(NormDiagonalMultNxN<data_t>(qubits), qubits );
    return ret;
  }
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
class NormMatrixMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit;
  uint_t mask;
  uint_t offset;
public:
  NormMatrixMult2x2(const cvector_t<double> &mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    offset = 1ull << qubit;
    mask = (1ull << qubit) - 1;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0,q1;
    thrust::complex<double> r0,r1;
    double sum = 0.0;

    vec = this->data_;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec[i0];
    q1 = vec[offset+i0];

    r0 = m0 * q0 + m2 * q1;
    sum += r0.real()*r0.real() + r0.imag()*r0.imag();
    r1 = m1 * q0 + m3 * q1;
    sum += r1.real()*r1.real() + r1.imag()*r1.imag();
    return sum;
  }
  const char* name(void)
  {
    return "Norm_mult2x2";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const
{
  reg_t qubits = {qubit};

  double ret = apply_function_sum(NormMatrixMult2x2<data_t>(mat,qubit), qubits);

  return ret;
}


template <typename data_t>
class NormDiagonalMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1;
  int qubit;
public:
  NormDiagonalMult2x2(cvector_t<double> &mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    uint_t gid;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m,r;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if((((i+gid) >> qubit) & 1) == 0){
      m = m0;
    }
    else{
      m = m1;
    }

    r = m * q;

    return (r.real()*r.real() + r.imag()*r.imag());
  }
  const char* name(void)
  {
    return "Norm_diagonal_mult2x2";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const
{
  reg_t qubits = {qubit};
  double ret = apply_function_sum(NormDiagonalMult2x2<data_t>(mat,qubit), qubits);

  return ret;
}



/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/
template <typename data_t>
double QubitVectorThrust<data_t>::probability(const uint_t outcome) const 
{

  std::complex<data_t> ret;
  ret = (std::complex<data_t>)chunk_->Get(outcome);

  return std::real(ret)*std::real(ret) + std::imag(ret) * std::imag(ret);
}

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities() const {
  const int_t END = 1LL << num_qubits();
  std::vector<double> probs(END, 0.);
#ifdef AER_DEBUG
  DebugMsg("calling probabilities");
#endif

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    probs[j] = probability(j);
  }

#ifdef AER_DEBUG
  DebugMsg("probabilities",probs);
#endif
  return probs;
}


template <typename data_t>
class probability_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask;
  uint_t cmask;
public:
  probability_func(const reg_t &qubits,int i)
  {
    int k;
    int nq = qubits.size();

    mask = 0;
    cmask = 0;
    for(k=0;k<nq;k++){
      mask |= (1ull << qubits[k]);

      if(((i >> k) & 1) != 0){
        cmask |= (1ull << qubits[k]);
      }
    }
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    double ret;

    vec = this->data_;

    ret = 0.0;

    if((i & mask) == cmask){
      q = vec[i];
      ret = q.real()*q.real() + q.imag()*q.imag();
    }
    return ret;
  }

  const char* name(void)
  {
    return "probabilities";
  }
};

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities(const reg_t &qubits) const 
{
  const size_t N = qubits.size();
  const int_t DIM = 1 << N;

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits_) && (qubits == qubits_sorted))
    return probabilities();

  std::vector<double> probs(DIM, 0.);

  int i;
  for(i=0;i<DIM;i++){
    probs[i] = apply_function_sum(probability_func<data_t>(qubits,i), qubits);
  }

#ifdef AER_DEBUG
  DebugMsg("probabilities",probs);
#endif

  return probs;
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorThrust<data_t>::sample_measure(const std::vector<double> &rnds) const
{
/*  const int_t SHOTS = rnds.size();
  reg_t samples;
  data_t* pVec;
  int i;
*/
#ifdef AER_TIMING
  TimeStart(QS_GATE_MEASURE);
#endif

#ifdef AER_DEBUG
  reg_t samples;

  samples = chunk_->sample_measure(rnds);
  DebugMsg("sample_measure",samples);
  return samples;
#else
  return chunk_->sample_measure(rnds);
#endif
}



/*******************************************************************************
 *
 * EXPECTATION VALUES
 *
 ******************************************************************************/

template <typename data_t>
class expval_pauli_Z_func : public GateFuncBase<data_t>
{
protected:
  uint_t z_mask_;
  thrust::complex<data_t> phase_;
public:
  expval_pauli_Z_func(uint_t z,thrust::complex<data_t> p)
  {
    z_mask_ = z;
    phase_ = p;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    double ret = 0.0;

    vec = this->data_;

    q0 = vec[i];
    q0 = phase_ * q0;
    ret = q0.real()*q0.real() + q0.imag()*q0.imag();

    if(z_mask_ != 0){
      uint_t count;
      //count bits (__builtin_popcountll can not be used on GPU)
      count = i & z_mask_;
      count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
      count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
      count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
      count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
      count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
      count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
      if(count & 1)
        ret = -ret;
    }
    return ret;
  }
  const char* name(void)
  {
    return "expval_pauli_Z";
  }
};

template <typename data_t>
class expval_pauli_XYZ_func : public GateFuncBase<data_t>
{
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;
  data_t sign_;
public:
  expval_pauli_XYZ_func(uint_t x,uint_t z,uint_t x_max,uint_t x_count,thrust::complex<data_t> p)
  {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max+1)) - 1);
    mask_l_ = (1ull << x_max) - 1;

    if(z == 0){
      sign_ = 1.0;
    }
    else{
      if(x_count & 1){
        sign_ = -1.0;
      }
      else{
        sign_ = 1.0;
      }
    }
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    thrust::complex<data_t> q0p;
    thrust::complex<data_t> q1p;
    double ret = 0.0;
    uint_t idx;

    vec = this->data_;

    idx = ((i << 1) & mask_u_) | (i & mask_l_);

    q0 = vec[idx];
    q1 = vec[idx ^ x_mask_];
    q0p = q1 * phase_;
    q1p = q0 * phase_;
    ret =           q0.real()*q0p.real() + q0.imag()*q0p.imag() + 
          sign_ * q1.real()*q1p.real() + q1.imag()*q1p.imag();

    if(z_mask_ != 0){
      uint_t count;
      //count bits (__builtin_popcountll can not be used on GPU)
      count = idx & z_mask_;
      count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
      count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
      count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
      count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
      count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
      count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
      if(count & 1)
        ret = -ret;
    }
    return ret;
  }
  const char* name(void)
  {
    return "expval_pauli_XYZ";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::expval_pauli(const reg_t &qubits,
                                               const std::string &pauli) const 
{
  // Break string up into Z and X
  // With Y being both Z and X (plus a phase)
  const size_t N = qubits.size();
  uint_t x_mask = 0;
  uint_t z_mask = 0;
  uint_t num_y = 0;
  uint_t num_x = 0;
  uint_t x_max = 0;
  for (size_t i = 0; i < N; ++i) {
    if(qubits[i] >= num_qubits_){  //only accepts bits inside chunk
      continue;
    }
    const auto bit = 1ull << qubits[i];
    switch (pauli[N - 1 - i]) {
      case 'I':
        break;
      case 'X': {
        x_mask += bit;
        x_max = std::max(x_max, (qubits[i]));
        num_x++;
        break;
      }
      case 'Z': {
        z_mask += bit;
        break;
      }
      case 'Y': {
        x_mask += bit;
        x_max = std::max(x_max, (qubits[i]));
        z_mask += bit;
        num_y++;
        num_x++;
        break;
      }
      default:
        throw std::invalid_argument("Invalid Pauli \"" + std::to_string(pauli[N - 1 - i]) + "\".");
    }
  }

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    return norm();
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  thrust::complex<data_t> phase(1,0);
  switch (num_y & 3) {
    case 0:
      // phase = 1
      break;
    case 1:
      // phase = -1j
      phase = thrust::complex<data_t>(0, -1);
      break;
    case 2:
      // phase = -1
      phase = thrust::complex<data_t>(-1, 0);
      break;
    case 3:
      // phase = 1j
      phase = thrust::complex<data_t>(0, 1);
      break;
  }

  // specialize x_max == 0
  if(x_mask == 0) {
    return chunk_->ExecuteSum(expval_pauli_Z_func<data_t>(z_mask, phase),data_size_);
  }

  return chunk_->ExecuteSum(expval_pauli_XYZ_func<data_t>(x_mask, z_mask, x_max, num_x, phase),(data_size_ >> 1) );
}



#ifdef AER_DEBUG

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const reg_t &qubits) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : %d (",debug_count,str,qubits.size());
    int iq;
    for(iq=0;iq<qubits.size();iq++){
      fprintf(debug_fp," %d ",qubits[iq]);
    }
    fprintf(debug_fp," )\n");
    fflush(debug_fp);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const int qubit) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : (%d) \n",debug_count,str,qubit);
    fflush(debug_fp);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s \n",debug_count,str);
    fflush(debug_fp);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::complex<double> c) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : %e, %e \n",debug_count,str,std::real(c),imag(c));
    fflush(debug_fp);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const double d) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : %e \n",debug_count,str,d);
    fflush(debug_fp);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::vector<double>& v) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : <",debug_count,str);
    int i,n;
    n = v.size();
    for(i=0;i<n;i++){
      fprintf(debug_fp," %e ",v[i]);
    }
    fprintf(debug_fp," >\n");
    fflush(debug_fp);
  }
  debug_count++;

}


template <typename data_t>
void QubitVectorThrust<data_t>::DebugDump(void) const
{
  if(debug_fp != NULL){
    if(num_qubits_ < 10){
      thrust::complex<data_t> t;
      uint_t i,j;
      char bin[64];

      bin[num_qubits_] = 0;
      for(i=0;i<data_size_;i++){
        t = chunk_->Get(i);
        for(j=0;j<num_qubits_;j++){
          bin[num_qubits_-j-1] = '0' + (char)((i >> j) & 1);
        }

        fprintf(debug_fp,"   %s | %e, %e\n",bin,t.real(),t.imag());
      }
    }
    fflush(debug_fp);
  }
}


#endif

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::QubitVectorThrust<data_t>&qv) {

  out << "[";
  size_t last = qv.size() - 1;
  for (size_t i = 0; i < qv.size(); ++i) {
    out << qv[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

//------------------------------------------------------------------------------
#endif // end module



