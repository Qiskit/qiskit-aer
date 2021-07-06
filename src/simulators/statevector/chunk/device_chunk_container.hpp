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



namespace AER {
namespace QV {


//============================================================================
// device chunk container class
//============================================================================
template <typename data_t>
class DeviceChunkContainer : public ChunkContainer<data_t>
{
protected:
  AERDeviceVector<thrust::complex<data_t>>  data_;    //device vector to chunks and buffers
  AERDeviceVector<thrust::complex<double>>  matrix_;  //storage for large matrix
  mutable AERDeviceVector<uint_t>           params_;  //storage for additional parameters
  AERDeviceVector<batched_matrix_params>    batched_params_;  //storage for parameters for batched matrix multiplication
  AERDeviceVector<double>                   reduce_buffer_; //buffer for reduction
  AERDeviceVector<double>                   condition_buffer_; //buffer to store condition
  AERDeviceVector<uint_t>                   condition_count_buffer_;     //buffer to count condition
  AERDeviceVector<uint_t>                   measured_bits_;     //measured classical bits (0-63)
  AERHostVector<uint_t>                     measured_bits_on_host_;
  int device_id_;                     //device index
  std::vector<bool> peer_access_;     //to which device accepts peer access 
  uint_t matrix_buffer_size_;         //matrix buffer size per chunk
  uint_t params_buffer_size_;         //params buffer size per chunk
  uint_t num_matrices_;               //number of matrices for chunks (1 shared matrix for multi-chunk execution)
  uint_t reduce_buffer_size_;

  bool multi_shots_;                  //multi-shot parallelization

  bool measured_bits_update_;

  //for register blocking
  thrust::host_vector<uint_t>               blocked_qubits_holder_;
  uint_t max_blocked_gates_;
  reg_t num_blocked_gates_;
  reg_t num_blocked_matrix_;
  reg_t num_blocked_qubits_;

#ifdef AER_THRUST_CUDA
  std::vector<cudaStream_t> stream_;    //asynchronous execution
#endif
public:
  DeviceChunkContainer()
  {
    device_id_ = 0;
    matrix_buffer_size_ = 0;
    params_buffer_size_ = 0;
    num_matrices_ = 1;
    multi_shots_ = false;
  }
  ~DeviceChunkContainer();

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

  uint_t Allocate(int idev,int bits,uint_t chunks,uint_t buffers,bool multi_shots);
  void Deallocate(void);
  uint_t Resize(uint_t chunks,uint_t buffers);

  void StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk);
  void StoreBatchedMatrix(const std::vector<std::complex<double>>& mat);
  void StoreMatrix(const std::complex<double>* mat,uint_t iChunk,uint_t size);
  void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk);
  void StoreBatchedParams(const std::vector<batched_matrix_params>& params);
  void ResizeMatrixBuffers(int bits);

  void set_device(void) const
  {
#ifdef AER_THRUST_CUDA
    cudaSetDevice(device_id_);
#endif
  }

#ifdef AER_THRUST_CUDA
  cudaStream_t stream(uint_t iChunk) const
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

  void CopyIn(Chunk<data_t>& src,uint_t iChunk);
  void CopyOut(Chunk<data_t>& src,uint_t iChunk);
  void CopyIn(thrust::complex<data_t>* src,uint_t iChunk, uint_t size);
  void CopyOut(thrust::complex<data_t>* dest,uint_t iChunk, uint_t size);
  void Swap(Chunk<data_t>& src,uint_t iChunk);

  void Zero(uint_t iChunk,uint_t count);

  reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride = 1, bool dot = true,uint_t count = 1) const;
  thrust::complex<double> norm(uint_t iChunk,uint_t stride = 1,bool dot = true) const;

  thrust::complex<data_t>* chunk_pointer(uint_t iChunk) const
  {
    return (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data()) + (iChunk << this->chunk_bits_);
  }

  thrust::complex<double>* matrix_pointer(uint_t iChunk) const
  {
    if(iChunk >= this->num_chunks_){  //for buffer chunks
      return ((thrust::complex<double>*)thrust::raw_pointer_cast(matrix_.data())) + ((num_matrices_ + iChunk - this->num_chunks_) * matrix_buffer_size_);
    }
    else{
      return ((thrust::complex<double>*)thrust::raw_pointer_cast(matrix_.data())) + (iChunk * matrix_buffer_size_);
    }
  }

  uint_t* param_pointer(uint_t iChunk) const
  {
    if(iChunk >= this->num_chunks_){  //for buffer chunks
      return ((uint_t*)thrust::raw_pointer_cast(params_.data())) + ((num_matrices_ + iChunk - this->num_chunks_) * params_buffer_size_);
    }
    else{
      return ((uint_t*)thrust::raw_pointer_cast(params_.data())) + (iChunk * params_buffer_size_);
    }
  }
  batched_matrix_params* batched_param_pointer(void) const
  {
    return (batched_matrix_params*)thrust::raw_pointer_cast(batched_params_.data());
  }

  double* reduce_buffer(uint_t iChunk) const
  {
    return ((double*)thrust::raw_pointer_cast(reduce_buffer_.data()) + iChunk * reduce_buffer_size_);
  }
  uint_t reduce_buffer_size() const
  {
    return reduce_buffer_size_;
  }
  double* condition_buffer(uint_t iChunk) const
  {
    return ((double*)thrust::raw_pointer_cast(condition_buffer_.data()) + iChunk);
  }
  void init_condition(uint_t iChunk,std::vector<double>& cond);
  bool update_condition(uint_t iChunk,uint_t count,bool async);

  int measured_cbit(uint_t iChunk,int qubit)
  {
    if(measured_bits_update_){
      measured_bits_update_ = false;
#ifdef AER_THRUST_CUDA
      cudaMemcpyAsync(thrust::raw_pointer_cast(measured_bits_on_host_.data()),thrust::raw_pointer_cast(measured_bits_.data()),sizeof(uint_t)*this->num_chunks_,cudaMemcpyDeviceToHost,stream_[0]);
      cudaStreamSynchronize(stream_[0]);
#else
      thrust::copy_n(measured_bits_.begin(),this->num_chunks_,measured_bits_on_host_.begin());
#endif
    }
    return (measured_bits_on_host_[iChunk] >> qubit) & 1;
  }
  uint_t* measured_bits_buffer(uint_t iChunk)
  {
    measured_bits_update_ = true;
    return ((uint_t*)thrust::raw_pointer_cast(measured_bits_.data()) + iChunk);
  }

  void synchronize(uint_t iChunk)
  {
#ifdef AER_THRUST_CUDA
    set_device();
    cudaStreamSynchronize(stream_[iChunk]);
#endif
  }

  //set qubits to be blocked
  void set_blocked_qubits(uint_t iChunk,const reg_t& qubits);

  //do all gates stored in queue
  void apply_blocked_gates(uint_t iChunk);

  //queue gate for blocked execution
  void queue_blocked_gate(uint_t iChunk,char gate,uint_t qubit,uint_t mask,const std::complex<double>* pMat = NULL);

  //CUDA graph
  void begin_capture(uint_t iChunk) const;
  void end_capture(uint_t iChunk) const;
};

template <typename data_t>
DeviceChunkContainer<data_t>::~DeviceChunkContainer(void)
{
  Deallocate();
}

template <typename data_t>
uint_t DeviceChunkContainer<data_t>::Allocate(int idev,int bits,uint_t chunks,uint_t buffers,bool multi_shots)
{
  uint_t nc = chunks;
  uint_t i;
  int mat_bits;

  this->chunk_bits_ = bits;

  device_id_ = idev;
  set_device();

#ifdef AER_THRUST_CUDA
  if(!multi_shots){
    int ip,nd;
    cudaGetDeviceCount(&nd);
    peer_access_.resize(nd);
    for(i=0;i<nd;i++){
      ip = 1;
      if(i != device_id_){
        cudaDeviceCanAccessPeer(&ip,device_id_,i);
      }
      if(ip){
        if(cudaDeviceEnablePeerAccess(i,0) != cudaSuccess)
          cudaGetLastError();
        peer_access_[i] = true;
      }
      else
        peer_access_[i] = false;
    }
  }
  else{
#endif
    peer_access_.resize(1);
    peer_access_[0] = true;
#ifdef AER_THRUST_CUDA
  }
#endif

  this->num_buffers_ = buffers;

  if(multi_shots){    //mult-shot parallelization for small qubits
    multi_shots_ = true;
    mat_bits = AER_DEFAULT_MATRIX_BITS;
    nc = chunks;
    num_matrices_ = chunks;
  }
  else{
    multi_shots_ = false;

    mat_bits = AER_DEFAULT_MATRIX_BITS;
    num_matrices_ = 1;
    nc = chunks;
  }

#ifdef AER_THRUST_CUDA
  uint_t param_size;
  param_size = (sizeof(thrust::complex<double>) << (mat_bits*2)) + (sizeof(uint_t) << (mat_bits+2));

  size_t freeMem,totalMem;
  cudaMemGetInfo(&freeMem,&totalMem);
  while(freeMem < ((((nc+buffers)*(uint_t)sizeof(thrust::complex<data_t>)) << bits) + param_size* (num_matrices_ + buffers)) ){
    nc--;
    if(nc == 0){
      break;
    }
  }
#endif

  max_blocked_gates_ = QV_MAX_BLOCKED_GATES;

  ResizeMatrixBuffers(mat_bits);

  this->num_chunks_ = nc;
  data_.resize((nc+buffers) << bits);

#ifdef AER_THRUST_CUDA
  stream_.resize(nc + buffers);
  for(i=0;i<nc + buffers;i++){
    cudaStreamCreateWithFlags(&stream_[i], cudaStreamNonBlocking);
  }
#endif

  if(bits < 10){
    reduce_buffer_size_ = 1;
  }
  else{
    reduce_buffer_size_ = (1ull << (bits - 10));
  }
  reduce_buffer_size_ *= 2;
  reduce_buffer_.resize(reduce_buffer_size_*nc);
  condition_buffer_.resize(nc);
  condition_count_buffer_.resize(nc);

  measured_bits_.resize(nc);
  measured_bits_on_host_.resize(nc);
  measured_bits_update_ = false;

  if(multi_shots)
    batched_params_.resize(nc);

  uint_t size = num_matrices_ + this->num_buffers_;
  num_blocked_gates_.resize(size);
  num_blocked_matrix_.resize(size);
  num_blocked_qubits_.resize(size);
  for(i=0;i<size;i++){
    num_blocked_gates_[i] = 0;
    num_blocked_matrix_[i] = 0;
  }
  blocked_qubits_holder_.resize(QV_MAX_REGISTERS*size);

  //allocate chunk classes
  ChunkContainer<data_t>::allocate_chunks();

  return nc;
}

template <typename data_t>
uint_t DeviceChunkContainer<data_t>::Resize(uint_t chunks,uint_t buffers)
{
  uint_t i;

  if(chunks + buffers > this->num_chunks_ + this->num_buffers_){
    set_device();
    data_.resize((chunks + buffers) << this->chunk_bits_);

    reduce_buffer_.resize(reduce_buffer_size_*chunks);
    condition_buffer_.resize(chunks);
    condition_count_buffer_.resize(chunks);
    measured_bits_.resize(chunks);
    measured_bits_on_host_.resize(chunks);

    if(multi_shots_)
      batched_params_.resize(chunks);
  }

  this->num_chunks_ = chunks;
  this->num_buffers_ = buffers;

  if(multi_shots_){
    num_matrices_ = chunks;
    ResizeMatrixBuffers(-1);
  }

#ifdef AER_THRUST_CUDA
  if(stream_.size() < chunks + buffers){
    uint_t size = stream_.size();
    stream_.resize(chunks + buffers);
    for(i=size;i<chunks + buffers;i++){
      cudaStreamCreateWithFlags(&stream_[i], cudaStreamNonBlocking);
    }
  }
#endif

  uint_t size = num_matrices_ + this->num_buffers_;
  num_blocked_gates_.resize(size);
  num_blocked_matrix_.resize(size);
  num_blocked_qubits_.resize(size);
  for(i=0;i<size;i++){
    num_blocked_gates_[i] = 0;
    num_blocked_matrix_[i] = 0;
  }
  blocked_qubits_holder_.resize(QV_MAX_REGISTERS*size);

  //allocate chunk classes
  ChunkContainer<data_t>::allocate_chunks();

  return chunks + buffers;
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Deallocate(void)
{
  set_device();

  data_.clear();
  data_.shrink_to_fit();
  matrix_.clear();
  matrix_.shrink_to_fit();
  params_.clear();
  params_.shrink_to_fit();
  batched_params_.clear();
  batched_params_.shrink_to_fit();
  reduce_buffer_.clear();
  reduce_buffer_.shrink_to_fit();
  condition_buffer_.clear();
  condition_buffer_.shrink_to_fit();
  condition_count_buffer_.clear();
  condition_count_buffer_.shrink_to_fit();
  measured_bits_.clear();
  measured_bits_.shrink_to_fit();
  measured_bits_on_host_.clear();
  measured_bits_on_host_.shrink_to_fit();

  peer_access_.clear();
  num_blocked_gates_.clear();
  num_blocked_matrix_.clear();
  num_blocked_qubits_.clear();
  blocked_qubits_holder_.clear();

#ifdef AER_THRUST_CUDA
  uint_t i;
  for(i=0;i<stream_.size();i++){
    cudaStreamDestroy(stream_[i]);
  }
  stream_.clear();
#endif
  ChunkContainer<data_t>::deallocate_chunks();
}

template <typename data_t>
void DeviceChunkContainer<data_t>::ResizeMatrixBuffers(int bits)
{
  uint_t size;
  uint_t n = num_matrices_ + this->num_buffers_;

  if(bits < 0){
    matrix_.resize(n * matrix_buffer_size_);
  }
  else{
    size = 1ull << (bits*2);

    if(max_blocked_gates_*4 > size){
      size = max_blocked_gates_*4;
    }
    if(size > matrix_buffer_size_){
      matrix_buffer_size_ = size;
      matrix_.resize(n * size);
    }
  }

  if(bits < 0){
    params_.resize(n * params_buffer_size_);
  }
  else{
    size = 1ull << (bits + 2);
    if(QV_MAX_REGISTERS + max_blocked_gates_*4 > size){
      size = QV_MAX_REGISTERS + max_blocked_gates_*4;
    }
    if(num_matrices_*size < (1ull << (bits*2))){
      size = (1ull << (bits*2))/num_matrices_;
    }
    if(size > params_buffer_size_){
      params_buffer_size_ = size;
      params_.resize(n * size);
    }
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreMatrix(const std::vector<std::complex<double>>& mat,uint_t iChunk)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can store (multi-chunk mode)
    return;
  }
  set_device();

  if(matrix_buffer_size_ < mat.size()){
    int bits;
    bits = 1;
    while((1 << (bits*2)) < mat.size()){
      bits++;
    }
    ResizeMatrixBuffers(bits);
  }

#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(matrix_pointer(iChunk),&mat[0],mat.size()*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[iChunk]);
#else
  uint_t offset;
  if(iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunks_) * matrix_buffer_size_;
  else
    offset = iChunk * matrix_buffer_size_;
  thrust::copy_n(mat.begin(),mat.size(),matrix_.begin() + offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreBatchedMatrix(const std::vector<std::complex<double>>& mat)
{
  set_device();

#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(matrix_pointer(0),&mat[0],mat.size()*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[0]);
#else
  thrust::copy_n(mat.begin(),mat.size(),matrix_.begin());
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreMatrix(const std::complex<double>* mat,uint_t iChunk,uint_t size)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can store (multi-chunk mode)
    return;
  }
  set_device();

  if(matrix_buffer_size_ < size){
    int bits;
    bits = 1;
    while((1 << (bits*2)) < size){
      bits++;
    }
    ResizeMatrixBuffers(bits);
  }

#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(matrix_pointer(iChunk),mat,size*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[iChunk]);
#else
  uint_t offset;
  if(iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunks_) * matrix_buffer_size_;
  else
    offset = iChunk * matrix_buffer_size_;
  thrust::copy_n(mat,mat+size,matrix_.begin() + offset);
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

#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(param_pointer(iChunk),&prm[0],prm.size()*sizeof(uint_t),cudaMemcpyHostToDevice,stream_[iChunk]);
#else
  uint_t offset;
  if(iChunk >= this->num_chunks_)
    offset = (num_matrices_ + iChunk - this->num_chunks_) * params_buffer_size_;
  else
    offset = iChunk * params_buffer_size_;
  thrust::copy_n(prm.begin(),prm.size(),params_.begin() + offset);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::StoreBatchedParams(const std::vector<batched_matrix_params>& params)
{
  set_device();

#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(batched_param_pointer(),&params[0],params.size()*sizeof(batched_matrix_params),cudaMemcpyHostToDevice,stream_[0]);
#else
  thrust::copy_n(params.begin(),params.size(),batched_params_.begin());
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyIn(Chunk<data_t>& src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  if(src.device() >= 0){
    if(peer_access(src.device())){
      thrust::copy_n(src.pointer(),size,data_.begin() + (iChunk << this->chunk_bits_));
    }
    else{
      AERHostVector<thrust::complex<data_t>> tmp(size);
      thrust::copy_n(src.pointer(),size,tmp.begin());
      thrust::copy_n(tmp.begin(),size,data_.begin() + (iChunk << this->chunk_bits_));
    }
  }
  else{
    thrust::copy_n(src.pointer(),size,data_.begin() + (iChunk << this->chunk_bits_));
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyOut(Chunk<data_t>& dest,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  if(dest.device() >= 0){
    if(peer_access(dest.device())){
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest.pointer());
    }
    else{
      AERHostVector<thrust::complex<data_t>> tmp(size);
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,tmp.begin());
      thrust::copy_n(tmp.begin(),size,dest.pointer());
    }
  }
  else{
    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest.pointer());
  }
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyIn(thrust::complex<data_t>* src,uint_t iChunk, uint_t size)
{
  uint_t this_size = 1ull << this->chunk_bits_;
  if(this_size < size) throw std::runtime_error("CopyIn chunk size is less than provided size");
  
  set_device();
  thrust::copy_n(src,size,data_.begin() + (iChunk << this->chunk_bits_));
}

template <typename data_t>
void DeviceChunkContainer<data_t>::CopyOut(thrust::complex<data_t>* dest,uint_t iChunk, uint_t size)
{
  uint_t this_size = 1ull << this->chunk_bits_;
  if(this_size < size) throw std::runtime_error("CopyOut chunk size is less than provided size");
  
  set_device();
  thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,dest);
}

template <typename data_t>
void DeviceChunkContainer<data_t>::Swap(Chunk<data_t>& src,uint_t iChunk)
{
  uint_t size = 1ull << this->chunk_bits_;
  set_device();
  if(src.device() >= 0){
    auto src_cont = std::static_pointer_cast<DeviceChunkContainer<data_t>>(src.container());
    if(peer_access(src.device())){
#ifdef AER_THRUST_CUDA
      thrust::swap_ranges(thrust::cuda::par.on(stream_[iChunk]),chunk_pointer(iChunk),chunk_pointer(iChunk + 1),src.pointer());
#else
      thrust::swap_ranges(thrust::device,chunk_pointer(iChunk),chunk_pointer(iChunk + 1),src.pointer());
#endif
    }
    else{
      //using temporary buffer on host
      AERHostVector<thrust::complex<data_t>> tmp1(size);
      AERHostVector<thrust::complex<data_t>> tmp2(size);

      thrust::copy_n(src_cont->vector().begin() + (src.pos() << this->chunk_bits_),size,tmp1.begin());
      thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,tmp2.begin());
      thrust::copy_n(tmp1.begin(),size,data_.begin() + (iChunk << this->chunk_bits_));
      thrust::copy_n(tmp2.begin(),size,src_cont->vector().begin() + (src.pos() << this->chunk_bits_));
    }
  }
  else{
    //using temporary buffer on host
    AERHostVector<thrust::complex<data_t>> tmp1(size);
    auto src_cont = std::static_pointer_cast<HostChunkContainer<data_t>>(src.container());

#ifdef AER_ATS
    //for IBM AC922
    thrust::swap_ranges(thrust::device,chunk_pointer(iChunk),chunk_pointer(iChunk + 1),src.pointer());
#else
    thrust::copy_n(data_.begin() + (iChunk << this->chunk_bits_),size,tmp1.begin());
    thrust::copy_n(src_cont->vector().begin() + (src.pos() << this->chunk_bits_),size,data_.begin() + (iChunk << this->chunk_bits_));
    thrust::copy_n(tmp1.begin(),size,src_cont->vector().begin() + (src.pos() << this->chunk_bits_));
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
  if(ChunkContainer<data_t>::enable_omp_)
    thrust::fill_n(thrust::device,data_.begin() + (iChunk << this->chunk_bits_),count,0.0);
  else{
    //disable nested OMP parallelization when shots are parallelized
    thrust::fill_n(thrust::seq,data_.begin() + (iChunk << this->chunk_bits_),count,0.0);
  }
#endif
}


template <typename data_t>
reg_t DeviceChunkContainer<data_t>::sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride, bool dot,uint_t count) const
{
  const int_t SHOTS = rnds.size();
  reg_t samples(SHOTS,0);

  set_device();

  strided_range<thrust::complex<data_t>*> iter(chunk_pointer(iChunk), chunk_pointer(iChunk+count), stride);

#ifdef AER_THRUST_CUDA
  if(dot)
    thrust::transform_inclusive_scan(thrust::cuda::par.on(stream_[iChunk]),iter.begin(),iter.end(),iter.begin(),complex_dot_scan<data_t>(),thrust::plus<thrust::complex<data_t>>());
  else
    thrust::inclusive_scan(thrust::cuda::par.on(stream_[iChunk]),iter.begin(),iter.end(),iter.begin(),thrust::plus<thrust::complex<data_t>>());

  if(multi_shots_ || this->num_chunks_ == 1){
    //matrix and parameter buffers can be used
    double* pRnd = (double*)matrix_pointer(iChunk);
    uint_t* pSmp = param_pointer(iChunk);
    thrust::device_ptr<double> rnd_dev_ptr = thrust::device_pointer_cast(pRnd);
    uint_t i,nshots,size = matrix_.size()*2;
    if(size > params_.size())
      size = params_.size();

    for(i=0;i<SHOTS;i+=size){
      nshots = size;
      if(i + nshots > SHOTS)
        nshots = SHOTS - i;

      cudaMemcpyAsync(pRnd,&rnds[i],nshots*sizeof(double),cudaMemcpyHostToDevice,stream_[iChunk]);

      thrust::lower_bound(thrust::cuda::par.on(stream_[iChunk]), iter.begin(), iter.end(), rnd_dev_ptr, rnd_dev_ptr + nshots, params_.begin() + (iChunk * params_buffer_size_),complex_less<data_t>());

      cudaMemcpyAsync(&samples[i],pSmp,nshots*sizeof(uint_t),cudaMemcpyDeviceToHost,stream_[iChunk]);
    }
    cudaStreamSynchronize(stream_[iChunk]);
  }
  else{
    thrust::device_vector<double> vRnd_dev(SHOTS);
    thrust::device_vector<uint_t> vSmp_dev(SHOTS);

    cudaMemcpyAsync(thrust::raw_pointer_cast(vRnd_dev.data()),&rnds[0],SHOTS*sizeof(double),cudaMemcpyHostToDevice,stream_[iChunk]);

    thrust::lower_bound(thrust::cuda::par.on(stream_[iChunk]), iter.begin(), iter.end(), vRnd_dev.begin(), vRnd_dev.begin() + SHOTS, vSmp_dev.begin() ,complex_less<data_t>());

    cudaMemcpyAsync(&samples[0],thrust::raw_pointer_cast(vSmp_dev.data()),SHOTS*sizeof(uint_t),cudaMemcpyDeviceToHost,stream_[iChunk]);
    cudaStreamSynchronize(stream_[iChunk]);

    vRnd_dev.clear();
    vRnd_dev.shrink_to_fit();
    vSmp_dev.clear();
    vSmp_dev.shrink_to_fit();
  }
#else

  if(ChunkContainer<data_t>::enable_omp_){
    if(dot)
      thrust::transform_inclusive_scan(thrust::device,iter.begin(),iter.end(),iter.begin(),complex_dot_scan<data_t>(),thrust::plus<thrust::complex<data_t>>());
    else
      thrust::inclusive_scan(thrust::device,iter.begin(),iter.end(),iter.begin(),thrust::plus<thrust::complex<data_t>>());

    thrust::lower_bound(thrust::device, iter.begin(), iter.end(), rnds.begin(), rnds.begin() + SHOTS, samples.begin() ,complex_less<data_t>());
  }
  else{
    //disable nested OMP parallelization when shots are parallelized
    if(dot)
      thrust::transform_inclusive_scan(thrust::seq,iter.begin(),iter.end(),iter.begin(),complex_dot_scan<data_t>(),thrust::plus<thrust::complex<data_t>>());
    else
      thrust::inclusive_scan(thrust::seq,iter.begin(),iter.end(),iter.begin(),thrust::plus<thrust::complex<data_t>>());

    thrust::lower_bound(thrust::seq, iter.begin(), iter.end(), rnds.begin(), rnds.begin() + SHOTS, samples.begin() ,complex_less<data_t>());
  }
#endif

  return samples;
}

template <typename data_t>
thrust::complex<double> DeviceChunkContainer<data_t>::norm(uint_t iChunk, uint_t stride, bool dot) const
{
  thrust::complex<double> sum,zero(0.0,0.0);
  set_device();

  strided_range<thrust::complex<data_t>*> iter(chunk_pointer(iChunk), chunk_pointer(iChunk+1), stride);

#ifdef AER_THRUST_CUDA
  if(dot)
    sum = thrust::transform_reduce(thrust::cuda::par.on(stream_[iChunk]), iter.begin(),iter.end(),complex_norm<data_t>() ,zero,thrust::plus<thrust::complex<double>>());
  else
    sum = thrust::reduce(thrust::cuda::par.on(stream_[iChunk]), iter.begin(),iter.end(),zero,thrust::plus<thrust::complex<double>>());
#else
  if(ChunkContainer<data_t>::enable_omp_){
    if(dot)
      sum = thrust::transform_reduce(thrust::device, iter.begin(),iter.end(),complex_norm<data_t>() ,zero,thrust::plus<thrust::complex<double>>());
    else
      sum = thrust::reduce(thrust::device, iter.begin(),iter.end(),zero,thrust::plus<thrust::complex<double>>());
  }
  else{
    //disable nested OMP parallelization when shots are parallelized
    if(dot)
      sum = thrust::transform_reduce(thrust::seq, iter.begin(),iter.end(),complex_norm<data_t>() ,zero,thrust::plus<thrust::complex<double>>());
    else
      sum = thrust::reduce(thrust::seq, iter.begin(),iter.end(),zero,thrust::plus<thrust::complex<double>>());
  }

#endif

  return sum;
}


//set qubits to be blocked
template <typename data_t>
void DeviceChunkContainer<data_t>::set_blocked_qubits(uint_t iChunk,const reg_t& qubits)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can store (multi-chunk mode)
    return;
  }
  uint_t iBlock;
  if(iChunk >= this->num_chunks_){  //for buffer chunks
    iBlock = num_matrices_ + iChunk - this->num_chunks_;
  }
  else{
    iBlock = iChunk;
  }

  if(num_blocked_gates_[iBlock] > 0){
    apply_blocked_gates(iChunk);
  }

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  int i;
  for(i=0;i<qubits.size();i++){
    blocked_qubits_holder_[iBlock*QV_MAX_REGISTERS + i] = qubits_sorted[i];
  }
#ifdef AER_THRUST_CUDA
  set_device();
  cudaMemcpyAsync(param_pointer(iChunk),
                  (uint_t*)&qubits_sorted[0],
                  qubits.size()*sizeof(uint_t),cudaMemcpyHostToDevice,stream_[iChunk]);
#endif

  num_blocked_gates_[iBlock] = 0;
  num_blocked_matrix_[iBlock] = 0;
  num_blocked_qubits_[iBlock] = qubits.size();
}

template <typename data_t>
class GeneralMatrixMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0_,m1_,m2_,m3_;
  uint_t mask_;
  uint_t offset_;

public:
  GeneralMatrixMult2x2(const cvector_t<double>& mat,int q,uint_t mask)
  {
    m0_ = mat[0];
    m1_ = mat[1];
    m2_ = mat[2];
    m3_ = mat[3];

    mask_ = mask;

    offset_ = 1ull << q;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec;

    vec = this->data_;

    i1 = i & (offset_ - 1);
    i0 = (i - i1) << 1;
    i0 += i1;
    i1 = i0 + offset_;

    q0 = vec[i0];
    q1 = vec[i1];

    if((i0 & mask_) == mask_)
      vec[i0] = m0_ * q0 + m2_ * q1;
    if((i1 & mask_) == mask_)
      vec[i1] = m1_ * q0 + m3_ * q1;
  }
  const char* name(void)
  {
    return "general_mult2x2";
  }
};


//queue gate for blocked execution
template <typename data_t>
void DeviceChunkContainer<data_t>::queue_blocked_gate(uint_t iChunk,char gate,uint_t qubit,uint_t mask,const std::complex<double>* pMat)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can store (multi-chunk mode)
    return;
  }

  cvector_t<double> mat(4,0.0);
  int i;
  uint_t idx,idxParam,iBlock;
  if(iChunk >= this->num_chunks_){  //for buffer chunks
    iBlock = num_matrices_ + iChunk - this->num_chunks_;
  }
  else{
    iBlock = iChunk;
  }

  if(num_blocked_gates_[iBlock] >= max_blocked_gates_){
    apply_blocked_gates(iChunk);
  }

#ifdef AER_THRUST_CUDA
  BlockedGateParams params;

  params.mask_ = mask;
  params.gate_ = gate;
  params.qubit_ = 0;
  for(i=0;i<num_blocked_qubits_[iBlock];i++){
    if(blocked_qubits_holder_[iBlock*QV_MAX_REGISTERS + i] == qubit){
      params.qubit_ = i;
      break;
    }
  }
  set_device();
  cudaMemcpyAsync((BlockedGateParams*)(param_pointer(iChunk) + num_blocked_qubits_[iBlock]) + num_blocked_gates_[iBlock],
                  &params,
                  sizeof(BlockedGateParams),cudaMemcpyHostToDevice,stream_[iChunk]);

  if(pMat != NULL){
    if(gate == 'd'){  //diagonal matrix
      mat[0] = pMat[0];
      mat[1] = pMat[1];
      cudaMemcpyAsync(matrix_pointer(iChunk) + num_blocked_matrix_[iBlock],
                      (thrust::complex<double>*)&mat[0],
                      2*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[iChunk]);
      num_blocked_matrix_[iBlock] += 2;
    }
    else if(gate == 'p'){ //phase
      mat[0]   = pMat[0];
      cudaMemcpyAsync(matrix_pointer(iChunk) + num_blocked_matrix_[iBlock],
                      (thrust::complex<double>*)&mat[0],
                      1*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[iChunk]);
      num_blocked_matrix_[iBlock] += 1;
    }
    else{ //otherwise, 2x2 matrix
      mat[0] = pMat[0];
      mat[1] = pMat[2];
      mat[2] = pMat[3];
      mat[3] = pMat[1];
      cudaMemcpyAsync(matrix_pointer(iChunk) + num_blocked_matrix_[iBlock],
                      (thrust::complex<double>*)&mat[0],
                      4*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice,stream_[iChunk]);
      num_blocked_matrix_[iBlock] += 4;
    }
  }
  num_blocked_gates_[iBlock]++;

#else
  //for statevector_cpu, apply now

  switch(gate){
  case 'x':
    mat[1] = 1.0;
    mat[2] = 1.0;
    break;
  case 'y':
    mat[1] = std::complex<double>(0.0,-1.0);
    mat[2] = std::complex<double>(0.0,1.0);
    break;
  case 'p':
    if(pMat){
      mat[0] = pMat[0];
      mat[3] = pMat[0];
    }
    break;
  case 'd':
    if(pMat){
      mat[0] = pMat[0];
      mat[3] = pMat[1];
    }
    break;
  default:
    if(pMat){
      mat[0] = pMat[0];
      mat[1] = pMat[1];
      mat[2] = pMat[2];
      mat[3] = pMat[3];
    }
    break;
  }
  uint_t count = 1;
  if(num_matrices_ == 1)
    count = this->chunk_bits_;
  this->Execute(GeneralMatrixMult2x2<data_t>(mat,qubit,mask),iChunk,count);

#endif

}

#ifdef AER_THRUST_CUDA

template <typename data_t> __global__
void dev_apply_register_blocked_gates(thrust::complex<data_t>* data,int num_gates,int num_qubits,int num_matrix,uint_t* qubits,BlockedGateParams* params,thrust::complex<double>* matrix)
{
  uint_t i,idx,ii,t,offset;
  uint_t j,laneID,iPair;
  thrust::complex<data_t> q,qp,qt;
  thrust::complex<double> m0,m1;
  data_t qr,qi;
  int nElem;
  thrust::complex<double>* matrix_load;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  laneID = i & 31;

  //index for this thread
  idx = 0;
  ii = i >> num_qubits;
  for(j=0;j<num_qubits;j++){
    offset = (1ull << qubits[j]);
    t = ii & (offset - 1);
    idx += t;
    ii = (ii - t) << 1;

    if(((laneID >> j) & 1) != 0){
      idx += offset;
    }
  }
  idx += ii;

  q = data[idx];

  //prefetch
  if(threadIdx.x < num_matrix)
    m0 = matrix[threadIdx.x];

  for(j=0;j<num_gates;j++){
    iPair = laneID ^ (1ull << params[j].qubit_);

    matrix_load = matrix;
    nElem = 0;

    switch(params[j].gate_){
    case 'x':
      m0 = 0.0;
      m1 = 1.0;
      break;
    case 'y':
      m0 = 0.0;
      if(iPair > laneID)
        m1 = thrust::complex<double>(0.0,-1.0);
      else
        m1 = thrust::complex<double>(0.0,1.0);
      break;
    case 'p':
      nElem = 1;
      matrix += 1;
      m1 = 0.0;
      break;
    case 'd':
      nElem = 2;
      matrix += 2;
      m1 = 0.0;
      break;
    default:
      nElem = 4;
      matrix += 4;
      break;
    }

    if(iPair < laneID){
      matrix_load += (nElem >> 1);
    }
    if(nElem > 0)
      m0 = *(matrix_load);
    if(nElem > 2)
      m1 = *(matrix_load + 1);

    //warp shuffle to get pair amplitude
    qr = __shfl_sync(0xffffffff,q.real(),iPair,32);
    qi = __shfl_sync(0xffffffff,q.imag(),iPair,32);
    qp = thrust::complex<data_t>(qr,qi);
    qt = m0*q + m1* qp;

    if((idx & params[j].mask_) == params[j].mask_){   //handling control bits
      q = qt;
    }
  }

  data[idx] = q;
}


template <typename data_t> __global__
void dev_apply_shared_memory_blocked_gates(thrust::complex<data_t>* data,int num_gates,int num_qubits,uint_t* qubits,BlockedGateParams* params,thrust::complex<double>* matrix)
{
  __shared__ thrust::complex<data_t> buf[1024];
  uint_t i,idx,ii,t,offset;
  uint_t j,laneID,iPair;
  thrust::complex<data_t> q,qp;
  thrust::complex<double> m0,m1;
  data_t qr,qi;
  int nElem;
  thrust::complex<double>* matrix_load;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  laneID = threadIdx.x;

  //index for this thread
  idx = 0;
  ii = i >> num_qubits;
  for(j=0;j<num_qubits;j++){
    offset = (1ull << qubits[j]);
    t = ii & (offset - 1);
    idx += t;
    ii = (ii - t) << 1;

    if(((laneID >> j) & 1) != 0){
      idx += offset;
    }
  }
  idx += ii;

  q = data[idx];

  for(j=0;j<num_gates;j++){
    iPair = laneID ^ (1ull << params[j].qubit_);

    if(params[j].qubit_ < 5){
      //warp shuffle to get pair amplitude
      qr = q.real();
      qi = q.imag();
      qr = __shfl_sync(0xffffffff,qr,iPair & 31,32);
      qi = __shfl_sync(0xffffffff,qi,iPair & 31,32);
      qp = thrust::complex<data_t>(qr,qi);
    }
    else{
      __syncthreads();
      buf[laneID] = q;
      __syncthreads();
      qp = buf[iPair];
    }

    matrix_load = matrix;
    nElem = 0;

    switch(params[j].gate_){
    case 'x':
      m0 = 0.0;
      m1 = 1.0;
      break;
    case 'y':
      m0 = 0.0;
      if(iPair > laneID)
        m1 = thrust::complex<double>(0.0,-1.0);
      else
        m1 = thrust::complex<double>(0.0,1.0);
      break;
    case 'p':
      nElem = 1;
      matrix += 1;
      m1 = 0.0;
      break;
    case 'd':
      nElem = 2;
      matrix += 2;
      m1 = 0.0;
      break;
    default:
      nElem = 4;
      matrix += 4;
      break;
    }

    if(iPair < laneID){
      matrix_load += (nElem >> 1);
    }
    if(nElem > 0)
      m0 = *(matrix_load);
    if(nElem > 2)
      m1 = *(matrix_load + 1);

    if((idx & params[j].mask_) == params[j].mask_){   //handling control bits
      q = m0*q + m1* qp;
    }
  }

  data[idx] = q;
}

#endif

//do all gates stored in queue
template <typename data_t>
void DeviceChunkContainer<data_t>::apply_blocked_gates(uint_t iChunk)
{
  if(num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can apply
    return;
  }
  uint_t iBlock;
  if(iChunk >= this->num_chunks_){  //for buffer chunks
    iBlock = num_matrices_ + iChunk - this->num_chunks_;
  }
  else{
    iBlock = iChunk;
  }

  if(num_blocked_gates_[iBlock] == 0)
    return;

#ifdef AER_THRUST_CUDA

  uint_t size;
  uint_t* pQubits;
  BlockedGateParams* pParams;
  thrust::complex<double>* pMatrix;

  set_device();

  pQubits = param_pointer(iChunk);
  pParams = (BlockedGateParams*)(param_pointer(iChunk) + num_blocked_qubits_[iBlock]);
  pMatrix = matrix_pointer(iChunk);

  if(num_matrices_ == 1){
    size = this->num_chunks_ << this->chunk_bits_;
  }
  else{
    size = 1ull << this->chunk_bits_;
  }
  uint_t nt,nb;
  nt = size;
  nb = 1;
  if(nt > 1024){
    nb = (nt + 1024 - 1) / 1024;
    nt = 1024;
  }

  if(num_blocked_qubits_[iBlock] < 6){
    //using register blocking (<=5 qubits)
    dev_apply_register_blocked_gates<data_t><<<nb,nt,num_blocked_matrix_[iChunk]*sizeof(thrust::complex<double>),stream_[iChunk]>>>(
                                                                          chunk_pointer(iChunk),
                                                                          num_blocked_gates_[iBlock],
                                                                          num_blocked_qubits_[iBlock],
                                                                          num_blocked_matrix_[iBlock],
                                                                          pQubits,pParams,pMatrix);
  }
  else{
    //using shared memory blocking (<=10 qubits)
    dev_apply_shared_memory_blocked_gates<data_t><<<nb,nt,1024*sizeof(thrust::complex<data_t>),stream_[iChunk]>>>(
                                                                          chunk_pointer(iChunk),
                                                                          num_blocked_gates_[iBlock],
                                                                          num_blocked_qubits_[iBlock],
                                                                          pQubits,pParams,pMatrix);
  }

#endif

  num_blocked_gates_[iBlock] = 0;
  num_blocked_matrix_[iBlock] = 0;

}

template <typename data_t>
void DeviceChunkContainer<data_t>::begin_capture(uint_t iChunk) const
{
#ifdef AER_THRUST_CUDA_
  cudaStreamBeginCapture(stream_[iChunk], cudaStreamCaptureModeThreadLocal);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::end_capture(uint_t iChunk) const
{
#ifdef AER_THRUST_CUDA_
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  cudaStreamEndCapture(stream_[iChunk], &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

  cudaGraphLaunch(instance, stream_[iChunk]);
  cudaStreamSynchronize(stream_[iChunk]);

  cudaGraphDestroy(graph);
#endif
}

template <typename data_t>
void DeviceChunkContainer<data_t>::init_condition(uint_t iChunk,std::vector<double>& cond)
{
#ifdef AER_THRUST_CUDA
  cudaMemcpyAsync(condition_buffer(iChunk),&cond[0],sizeof(double)*cond.size(),cudaMemcpyHostToDevice,stream_[iChunk]);
#else
  condition_buffer_[iChunk] = 0.0;
#endif
}

template <typename data_t>
bool DeviceChunkContainer<data_t>::update_condition(uint_t iChunk,uint_t count,bool async)
{
#ifdef AER_THRUST_CUDA
  uint_t* count_buf = (uint_t*)thrust::raw_pointer_cast(condition_count_buffer_.data());
  uint_t nt,nb;
  nt = count;
  nb = 1;
  if(nt > 1024){
    nb = (nt + 1024 - 1) / 1024;
    nt = 1024;
  }

  dev_update_condition<<<nb,nt,0,stream_[iChunk]>>>(condition_buffer(iChunk),reduce_buffer(iChunk),count_buf,reduce_buffer_size_,count);

  if(async){
    return false;
  }

  while(nb > 1){
    uint_t n = nb;
    nt = nb;
    nb = 1;
    if(nt > 1024){
      nb = (nt + 1024 - 1) / 1024;
      nt = 1024;
    }

    dev_reduce_sum_uint<<<nb,nt,0,stream_[iChunk]>>>(count_buf,n,0);
  }
  cudaMemcpyAsync(&nt,count_buf,sizeof(uint_t),cudaMemcpyDeviceToHost,stream_[iChunk]);
  cudaStreamSynchronize(stream_[iChunk]);

  if(nt >= count){
    return true;
  }
#else
  condition_buffer_[iChunk] -= reduce_buffer[iChunk*reduce_buffer_size_];
#endif

  return false;
}


//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
