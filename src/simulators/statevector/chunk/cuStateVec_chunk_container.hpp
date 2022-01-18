/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */


#ifndef _qv_cuStateVec_chunk_container_hpp_
#define _qv_cuStateVec_chunk_container_hpp_

#include "simulators/statevector/chunk/device_chunk_container.hpp"

#include "custatevec.h"

namespace AER {
namespace QV {


//============================================================================
// cuStateVec chunk container class
//============================================================================
template <typename data_t>
class cuStateVecChunkContainer : public DeviceChunkContainer<data_t>
{
protected:
  std::vector<custatevecHandle_t> custatevec_handle_;                       //cuStatevec handle for this chunk container
  AERDeviceVector<unsigned char>            custatevec_work_;  //work buffer for cuStatevec
  uint_t                                    custatevec_work_size_;    //buffer size
  uint_t                                    custatevec_chunk_total_qubits_;   //total qubits of statevector passed to ApplyMatrix
  uint_t                                    custatevec_chunk_count_;          //number of counts for all chunks

public:
  using BaseContainer = DeviceChunkContainer<data_t>;

  cuStateVecChunkContainer()
  {
  }
  ~cuStateVecChunkContainer();

  uint_t Allocate(int idev,int chunk_bits,int num_qubits,uint_t chunks,uint_t buffers,bool multi_shots,int matrix_bit) override;
  void Deallocate(void) override;

  unsigned char* custatevec_work_pointer(uint_t iChunk) const
  {
    if(custatevec_work_size_ == 0)
      return nullptr;
    if(iChunk >= this->num_chunks_){  //for buffer chunks
      return ((unsigned char*)thrust::raw_pointer_cast(custatevec_work_.data())) + ((BaseContainer::num_matrices_ + iChunk - this->num_chunks_) * custatevec_work_size_);
    }
    else{
      return ((unsigned char*)thrust::raw_pointer_cast(custatevec_work_.data())) + ((iChunk % BaseContainer::num_matrices_) * custatevec_work_size_);
    }
  }

  reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride = 1, bool dot = true,uint_t count = 1) const override;
  double norm(uint_t iChunk,uint_t count) const override;

  //apply matrix
  void apply_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t count) override;

  //apply diagonal matrix
  void apply_diagonal_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t count) override;

  //apply (controlled) X
  void apply_X(const uint_t iChunk,const reg_t& qubits,const uint_t count) override;

  //apply (controlled) Y
  void apply_Y(const uint_t iChunk,const reg_t& qubits,const uint_t count) override;

  //apply (controlled) phase
  virtual void apply_phase(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t count) override;

  //apply (controlled) swap gate
  void apply_swap(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const uint_t count) override;

  //apply permutation
  void apply_permutation(const uint_t iChunk,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t count) override;

  //get probabilities of chunk
  void probabilities(std::vector<double>& probs, const uint_t iChunk, const reg_t& qubits) const override;

  //Pauli expectation values
  double expval_pauli(const uint_t iChunk,const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const override;
};

template <typename data_t>
cuStateVecChunkContainer<data_t>::~cuStateVecChunkContainer(void)
{
  Deallocate();
}

template <typename data_t>
uint_t cuStateVecChunkContainer<data_t>::Allocate(int idev,int chunk_bits,int num_qubits,uint_t chunks,uint_t buffers,bool multi_shots,int matrix_bit)
{
  uint_t nc;
  nc = BaseContainer::Allocate(idev,chunk_bits,num_qubits,chunks,buffers,multi_shots,matrix_bit);

  //initialize custatevevtor handle
  custatevecStatus_t err;

  custatevec_handle_.resize(nc + buffers);
  for(uint_t i=0;i<nc + buffers;i++){
    err = custatevecCreate(&custatevec_handle_[i]);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::allocate : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }

    //set stream to custatevec handle
    err = custatevecSetStream(custatevec_handle_[i],BaseContainer::stream_[i]);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::allocate : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }

  //allocate extra workspace for custatevec
  std::vector<std::complex<double>> mat(1ull << (matrix_bit*2));

  //count bits for multi-chunks
  custatevec_chunk_total_qubits_ = this->num_pow2_qubits_;
  custatevec_chunk_count_ = this->num_chunks_ >> (this->num_pow2_qubits_ - this->chunk_bits_);

  //matrix
  err = custatevecApplyMatrix_bufferSize(
                  custatevec_handle_[0], CUDA_C_64F, custatevec_chunk_total_qubits_ , &mat[0], CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL,
                  0, matrix_bit, 0, CUSTATEVEC_COMPUTE_64F, &custatevec_work_size_);
  if(err != CUSTATEVEC_STATUS_SUCCESS){
    std::stringstream str;
    str << "cuStateVecChunkContainer::ResizeMatrixBuffers : " << custatevecGetErrorString(err);
    throw std::runtime_error(str.str());
  }

  //diagonal matrix
  size_t diag_size;
  std::vector<custatevecIndex_t> perm(matrix_bit);
  std::vector<int32_t> basis(matrix_bit);
  for(int_t i=0;i<matrix_bit;i++){
    perm[i] = i;
    basis[i] = i;
  }
  err = custatevecApplyGeneralizedPermutationMatrix_bufferSize(
                  custatevec_handle_[0], CUDA_C_64F, custatevec_chunk_total_qubits_ , &perm[0], &mat[0], CUDA_C_64F,
                  &basis[0], matrix_bit, 0, &diag_size);
  if(err != CUSTATEVEC_STATUS_SUCCESS){
    std::stringstream str;
    str << "cuStateVecChunkContainer::ResizeMatrixBuffers : " << custatevecGetErrorString(err);
    throw std::runtime_error(str.str());
  }
  if(custatevec_work_size_ < diag_size)
    custatevec_work_size_ = diag_size;
  if(custatevec_work_size_ > 0)
    custatevec_work_.resize(custatevec_work_size_*BaseContainer::num_matrices_);

  return nc;
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::Deallocate(void)
{
  BaseContainer::Deallocate();

  custatevec_work_.clear();
  custatevec_work_.shrink_to_fit();
  for(int_t i=0;i<custatevec_handle_.size();i++){
    custatevecDestroy(custatevec_handle_[i]);
  }
  custatevec_handle_.clear();
}

template <typename data_t>
reg_t cuStateVecChunkContainer<data_t>::sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride, bool dot,uint_t count) const
{
  if(count == (1ull << (this->num_qubits_ - this->chunk_bits_))){
    //custatevecSampler_sample only can be applied to whole statevector
    const int_t SHOTS = rnds.size();
    reg_t samples(SHOTS,0);

    BaseContainer::set_device();

    custatevecStatus_t err;
    custatevecSamplerDescriptor_t sampler;
    size_t extSize;

    cudaStreamSynchronize(BaseContainer::stream_[iChunk]);

    cudaDataType_t state_type;
    if(sizeof(data_t) == sizeof(double))
      state_type = CUDA_C_64F;
    else
      state_type = CUDA_C_32F;

    err = custatevecSampler_create(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk), state_type, this->num_qubits_, &sampler, SHOTS, &extSize);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::sample_measure : custatevecSampler_create " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }

    AERDeviceVector<unsigned char> extBuf;
    void* pExtBuf = nullptr;
    if(extSize > 0){
      extBuf.resize(extSize);
      pExtBuf = thrust::raw_pointer_cast(extBuf.data());
    }

    err = custatevecSampler_preprocess(custatevec_handle_[iChunk],&sampler,pExtBuf,extSize);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::sample_measure : custatevecSampler_preprocess " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }

    std::vector<custatevecIndex_t> bitStr(SHOTS);
    std::vector<int> bitOrdering(this->num_qubits_);
    for(int_t i=0;i<this->num_qubits_;i++){
      bitOrdering[i] = i;
    }

    err = custatevecSampler_sample(custatevec_handle_[iChunk], &sampler, &bitStr[0], &bitOrdering[0], this->num_qubits_, &rnds[0], SHOTS,
                    CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER ) ;
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::sample_measure : custatevecSampler_sample " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }

    for(int_t i=0;i<SHOTS;i++){
      samples[i] = bitStr[i];
    }

    if(extSize > 0){
      extBuf.clear();
      extBuf.shrink_to_fit();
    }
    return samples;
  }
  else{
    return BaseContainer::sample_measure(iChunk, rnds, stride, dot, count);
  }
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t count)
{
  thrust::complex<double>* pMat;
  int_t num_qubits = qubits.size()-control_bits;

  if((BaseContainer::matrix_buffer_size_  >= (1ull << (num_qubits*2))) && ((count == this->num_chunks_ && iChunk == 0) || BaseContainer::num_matrices_ > 1)){
    BaseContainer::StoreMatrix(mat,iChunk);
    pMat = BaseContainer::matrix_pointer(iChunk);
  }
  else{
    //if operation is not batchable, use host memory
    pMat = (thrust::complex<double>*)&mat[0];
    BaseContainer::set_device();
  }

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];

  int32_t* pQubits = &qubits32[control_bits];
  int32_t* pControl = nullptr;
  if(control_bits > 0)
    pControl = &qubits32[0];

  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }
  cudaDataType_t state_type;
  custatevecComputeType_t comp_type;
  if(sizeof(data_t) == sizeof(double)){
    state_type = CUDA_C_64F;
    comp_type = CUSTATEVEC_COMPUTE_64F;
  }
  else{
    state_type = CUDA_C_32F;
    comp_type = CUSTATEVEC_COMPUTE_32F;
  }

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    err = custatevecApplyMatrix(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk) + (i << bits), state_type, bits, pMat, CUDA_C_64F,
                          CUSTATEVEC_MATRIX_LAYOUT_COL, 0, pQubits, num_qubits, pControl, control_bits, 
                          nullptr, comp_type, custatevec_work_pointer(iChunk), custatevec_work_size_);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::apply_matrix : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_diagonal_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t count)
{
  thrust::complex<double>* pMat;
  int_t num_qubits = qubits.size();

  if(control_bits > 0){
    uint_t size = 1ull << num_qubits;
    cvector_t<double> diag_ctrl(size);    //make diagonal matrix with controls

    for(int_t i=0;i<size;i++)
      diag_ctrl[i] = 1.0;
    uint_t offset = (1ull << control_bits) - 1;
    for(int_t i=0;i<diag.size();i++)
      diag_ctrl[(i << control_bits)+offset] = diag[i];

    return apply_diagonal_matrix(iChunk, qubits, 0, diag_ctrl, count);
  }

  if((BaseContainer::matrix_buffer_size_  >= (1ull << num_qubits)) && ((count == this->num_chunks_ && iChunk == 0) || BaseContainer::num_matrices_ > 1)){
    BaseContainer::StoreMatrix(diag,iChunk);
    pMat = BaseContainer::matrix_pointer(iChunk);
  }
  else{
    //if operation is not batchable, use host memory
    pMat = (thrust::complex<double>*)&diag[0];
    BaseContainer::set_device();
  }

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];

  int32_t* pQubits = &qubits32[control_bits];
  int32_t* pControl = nullptr;
  if(control_bits > 0)
    pControl = &qubits32[0];

  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    err = custatevecApplyGeneralizedPermutationMatrix(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk) + (i << bits), state_type, bits, 
                          nullptr, pMat, CUDA_C_64F, 0, pQubits, num_qubits, nullptr, nullptr, 0, 
                          custatevec_work_pointer(iChunk), custatevec_work_size_);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::apply_diagonal_matrix : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_X(const uint_t iChunk,const reg_t& qubits,const uint_t count)
{
  int_t num_qubits = qubits.size();

  BaseContainer::set_device();

  uint_t perm_size = 1ull << num_qubits;
  std::vector<custatevecIndex_t> perm(perm_size);
  for(int_t i=0;i<perm_size;i++)
    perm[i] = i;

  //set permutation
  uint_t ctrl_offset = (1ull << (num_qubits - 1)) - 1;
  uint_t t_offset = (1ull << (num_qubits - 1)) + ctrl_offset;
  perm[ctrl_offset] = t_offset;
  perm[t_offset] = ctrl_offset;

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];
  int32_t* pQubits = &qubits32[0];

  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    err = custatevecApplyGeneralizedPermutationMatrix(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk) + (i << bits), state_type, bits, 
                          &perm[0], nullptr, CUDA_C_64F, 0, pQubits, num_qubits, nullptr, nullptr, 0, 
                          custatevec_work_pointer(iChunk), custatevec_work_size_);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::apply_X : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_Y(const uint_t iChunk,const reg_t& qubits,const uint_t count)
{
  int_t num_qubits = qubits.size();

  BaseContainer::set_device();

  uint_t perm_size = 1ull << num_qubits;
  cvector_t<double> diag(perm_size);
  std::vector<custatevecIndex_t> perm(perm_size);
  for(int_t i=0;i<perm_size;i++){
    perm[i] = i;
    diag[i] = 1.0;
  }

  //set diagonal matrix and permutation matrix
  uint_t ctrl_offset = (1ull << (num_qubits - 1)) - 1;
  uint_t t_offset = (1ull << (num_qubits - 1)) + ctrl_offset;
  perm[ctrl_offset] = t_offset;
  perm[t_offset] = ctrl_offset;
  diag[ctrl_offset] = {0.0, -1.0};
  diag[t_offset] = {0.0, 1.0};

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];
  int32_t* pQubits = &qubits32[0];

  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    err = custatevecApplyGeneralizedPermutationMatrix(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk) + (i << bits), state_type, bits, 
                          &perm[0], &diag[0], CUDA_C_64F, 0, pQubits, num_qubits, nullptr, nullptr, 0, 
                          custatevec_work_pointer(iChunk), custatevec_work_size_);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::apply_Y : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_phase(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t count)
{
  uint_t size = 1ull << qubits.size();
  cvector_t<double> diag(size);
  for(int_t i=0;i<size-1;i++)
    diag[i] = 1.0;
  diag[size-1] = phase;

  apply_diagonal_matrix(iChunk, qubits, 0, diag, count);
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_swap(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const uint_t count)
{
  int_t num_qubits = qubits.size();

  BaseContainer::set_device();

  uint_t perm_size = 1ull << num_qubits;
  std::vector<custatevecIndex_t> swap(perm_size);
  for(int_t i=0;i<perm_size;i++)
    swap[i] = i;

  //set permutation
  uint_t ctrl_offset = (1ull << control_bits) - 1;
  uint_t t1_offset = (1ull << (num_qubits - 2)) + ctrl_offset;
  uint_t t2_offset = (1ull << (num_qubits - 1)) + ctrl_offset;
  swap[t1_offset] = t2_offset;
  swap[t2_offset] = t1_offset;

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];
  int32_t* pQubits = &qubits32[0];

  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    err = custatevecApplyGeneralizedPermutationMatrix(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk) + (i << bits), state_type, bits, 
                          &swap[0], nullptr, CUDA_C_64F, 0, pQubits, num_qubits, nullptr, nullptr, 0, 
                          custatevec_work_pointer(iChunk), custatevec_work_size_);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::apply_swap : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::apply_permutation(const uint_t iChunk,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t count)
{
  BaseContainer::set_device();

  int_t size = 1ull << qubits.size();
  custatevecIndex_t perm[size];
  for(int_t i=0;i<size;i++)
    perm[i] = i;
  for(int_t i=0;i<pairs.size();i++)
    std::swap(perm[pairs[i].first],perm[pairs[i].second]);

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];

  int32_t* pQubits = &qubits32[0];

  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    err = custatevecApplyGeneralizedPermutationMatrix(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk) + (i << bits), state_type, bits, 
                          perm, nullptr, CUDA_C_64F, 0, pQubits, qubits.size(), nullptr, nullptr, 0, 
                          custatevec_work_pointer(iChunk), custatevec_work_size_);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::apply_permutation : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
  }
}

template <typename data_t>
double cuStateVecChunkContainer<data_t>::norm(uint_t iChunk,uint_t count) const 
{
  double ret = 0.0;
  uint_t bits;
  uint_t nc;
  if(count == this->num_chunks_){
    bits = custatevec_chunk_total_qubits_;
    nc = custatevec_chunk_count_;
  }
  else{
    nc = count;
    bits = this->chunk_bits_;
    if(nc > 0){
      while((nc & 1) == 0){
        nc >>= 1;
        bits++;
      }
    }
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecStatus_t err;
  for(int_t i=0;i<nc;i++){
    double d;
    err = custatevecAbs2SumArray(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk), state_type, bits, 
                           &d, nullptr, 0, nullptr,nullptr,0);
    if(err != CUSTATEVEC_STATUS_SUCCESS){
      std::stringstream str;
      str << "cuStateVecChunkContainer::norm : " << custatevecGetErrorString(err);
      throw std::runtime_error(str.str());
    }
    ret += d;
  }

  return ret;
}

template <typename data_t>
void cuStateVecChunkContainer<data_t>::probabilities(std::vector<double>& probs, const uint_t iChunk, const reg_t& qubits) const
{
  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  std::vector<int32_t> qubits32(qubits.size());
  for(int_t i=0;i<qubits.size();i++)
    qubits32[i] = qubits[i];

  custatevecStatus_t err;
  if(qubits.size() == 1){
    double p0,p1;
    err = custatevecAbs2SumOnZBasis(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk), state_type, this->chunk_bits_, 
                              &p0, &p1, &qubits32[0], 1);
    probs.resize(2);
    probs[0] = p0;
    probs[1] = p1;
  }
  else{
    probs.resize(1ull << qubits.size());
    err = custatevecAbs2SumArray(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk), state_type, this->chunk_bits_, 
                           &probs[0], &qubits32[0], qubits.size(), nullptr,nullptr,0);
  }

  if(err != CUSTATEVEC_STATUS_SUCCESS){
    std::stringstream str;
    str << "cuStateVecChunkContainer::probabilities : " << custatevecGetErrorString(err);
    throw std::runtime_error(str.str());
  }
}

template <typename data_t>
double cuStateVecChunkContainer<data_t>::expval_pauli(const uint_t iChunk,const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const
{
  if(initial_phase != 1.0){
    return BaseContainer::expval_pauli(iChunk, qubits, pauli, initial_phase);
  }

  cudaDataType_t state_type;
  if(sizeof(data_t) == sizeof(double))
    state_type = CUDA_C_64F;
  else
    state_type = CUDA_C_32F;

  custatevecPauli_t pauliOps[pauli.size()];
  int32_t qubits32[qubits.size()];
  for(int_t i=0;i<qubits.size();i++){
    qubits32[i] = qubits[i];
    if(pauli[pauli.size()-1-i] == 'X')
      pauliOps[i] = CUSTATEVEC_PAULI_X;
    else if(pauli[pauli.size()-1-i] == 'Y')
      pauliOps[i] = CUSTATEVEC_PAULI_Y;
    else if(pauli[pauli.size()-1-i] == 'Z')
      pauliOps[i] = CUSTATEVEC_PAULI_Z;
    else
      pauliOps[i] = CUSTATEVEC_PAULI_I;
  }

  const custatevecPauli_t* pauliOperatorsArray[] = {pauliOps};
  const int32_t *basisBitsArray[] = { qubits32 };
  double ret[1];
  const uint32_t nBasisBitsArray[] = {qubits.size()};

  custatevecStatus_t err;
  err = custatevecExpectationsOnPauliBasis(custatevec_handle_[iChunk], BaseContainer::chunk_pointer(iChunk), state_type, this->chunk_bits_, 
                                           ret, pauliOperatorsArray, basisBitsArray, nBasisBitsArray, 1);

  if(err != CUSTATEVEC_STATUS_SUCCESS){
    std::stringstream str;
    str << "cuStateVecChunkContainer::expval_pauli : " << custatevecGetErrorString(err);
    throw std::runtime_error(str.str());
  }

  return ret[0];
}



//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
