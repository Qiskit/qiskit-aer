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


#ifndef _qv_device_chunk_container_hpp_
#define _qv_device_chunk_container_hpp_

#include "simulators/statevector/chunk/chunk_container.hpp"

namespace QV {

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
    offset = (num_matrices_ + iChunk - this->num_chunks_) << (matrix_bits_*2);
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
    offset = (num_matrices_ + iChunk - this->num_chunks_) << (matrix_bits_ + 2);
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



}

//------------------------------------------------------------------------------
#endif // end module
