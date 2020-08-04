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


#ifndef _qv_host_chunk_container_hpp_
#define _qv_host_chunk_container_hpp_

#include "simulators/statevector/chunk/chunk_container.hpp"


namespace AER {
namespace QV {


//============================================================================
// host chunk container class
//============================================================================
template <typename data_t>
class HostChunkContainer : public ChunkContainer<data_t>
{
protected:
  AERHostVector<thrust::complex<data_t>>  data_;     //host vector for chunks + buffers
  std::vector<thrust::complex<double>*> matrix_;     //pointer to matrix
  std::vector<uint_t*> params_;                      //pointer to additional parameters
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
    matrix_[iChunk] = (thrust::complex<double>*)&mat[0];
  }
  void StoreUintParams(const std::vector<uint_t>& prm,uint_t iChunk)
  {
    params_[iChunk] = (uint_t*)&prm[0];
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
  Deallocate();
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
  matrix_.resize(nc + buffers);
  params_.resize(nc + buffers);

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
    matrix_.resize(chunks + buffers);
    params_.resize(chunks + buffers);
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
  matrix_.clear();
  params_.clear();
}

template <typename data_t>
template <typename Function>
void HostChunkContainer<data_t>::Execute(Function func,uint_t iChunk,uint_t count)
{
  func.set_data( (thrust::complex<data_t>*)thrust::raw_pointer_cast(data_.data()) + (iChunk << ChunkContainer<data_t>::chunk_bits_));

  func.set_matrix( matrix_[iChunk]);
  func.set_params( params_[iChunk]);

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

  func.set_matrix( matrix_[iChunk]);
  func.set_params( params_[iChunk]);

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

  func.set_matrix( matrix_[iChunk]);
  func.set_params( params_[iChunk]);

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


//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
