/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
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

#define AER_DEFAULT_MATRIX_BITS   8

#define AER_CHUNK_BITS        21
#define AER_MAX_BUFFERS       2

namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;

//==================================
// parameters for gate kernels
//==================================
template <typename data_t>
struct GateParams {
  thrust::complex<data_t>* buf_;
  uint_t* offsets_;
  thrust::complex<double>* matrix_;
  uint_t* params_;
  uint_t gid_;
  uint_t lmask_;
};

//========================================
//  base class of gate functions
//========================================
class GateFuncBase
{
public:
  GateFuncBase(void)
  {
  }

  virtual bool IsDiagonal(void)
  {
    return false;
  }
  virtual int NumControlBits(void)
  {
    return 0;
  }
  virtual int ControlMask(void)
  {
    return 1;
  }
  virtual bool Reduction(void)
  {
    return false;
  }

  virtual const char* Name(void)
  {
    return "base function";
  }
};


//=============================================================
//    virtual buffer class
//=============================================================
template <typename data_t>
class QubitVectorBuffer
{
protected:
  uint_t m_size;
public:
  QubitVectorBuffer(uint_t size = 0)
  {
    m_size = size;
  }
  virtual ~QubitVectorBuffer(){};

  uint_t Size(void)
  {
    return m_size;
  }

  virtual data_t* BufferPtr(void)
  {
    return NULL;
  }
  virtual void Set(uint_t i,const data_t& t) = 0;
  virtual data_t Get(uint_t i) const = 0;

  virtual void Resize(uint_t size) = 0;

  virtual void Copy(const std::vector<data_t>& v) = 0;

  virtual void Copy(uint_t pos,QubitVectorBuffer<data_t>* pSrc,uint_t srcPos,uint_t size,int isDevice = 0) = 0;

  virtual void CopyIn(uint_t pos,const data_t* pSrc,uint_t size) = 0;
  virtual void CopyOut(uint_t pos,data_t* pDest,uint_t size) = 0;

};

#ifdef AER_THRUST_CUDA
#define AERDeviceVector thrust::device_vector
#else
#define AERDeviceVector thrust::host_vector
#endif

template <typename data_t>
class QubitVectorDeviceBuffer : public QubitVectorBuffer<data_t>
{
protected:
  AERDeviceVector<data_t> m_Buffer;
public:
  QubitVectorDeviceBuffer(uint_t size) : m_Buffer(size)
  {
    ;
  }

  AERDeviceVector<data_t>& Buffer(void)
  {
    return m_Buffer;
  }

  data_t* BufferPtr(void)
  {
    return (data_t*)thrust::raw_pointer_cast(m_Buffer.data());
  }
  void Set(uint_t i,const data_t& t)
  {
    m_Buffer[i] = t;
  }
  data_t Get(uint_t i) const
  {
    return m_Buffer[i];
  }

  void Resize(uint_t size)
  {
    if(QubitVectorBuffer<data_t>::m_size != size){
      m_Buffer.resize(size);
      QubitVectorBuffer<data_t>::m_size = size;
    }
  }
  void Copy(const std::vector<data_t>& v)
  {
    m_Buffer = v;
  }

  void Copy(uint_t pos,QubitVectorBuffer<data_t>* pSrc,uint_t srcPos,uint_t size,int isDevice = 1);

  void CopyIn(uint_t pos,const data_t* pSrc,uint_t size);
  void CopyOut(uint_t pos,data_t* pDest,uint_t size);
};


template <typename data_t>
class QubitVectorHostBuffer : public QubitVectorBuffer<data_t>
{
protected:
  thrust::host_vector<data_t> m_Buffer;
public:
  QubitVectorHostBuffer(uint_t size) : m_Buffer(size)
  {
    ;
  }

  thrust::host_vector<data_t>& Buffer(void)
  {
    return m_Buffer;
  }

  data_t* BufferPtr(void)
  {
    return (data_t*)thrust::raw_pointer_cast(&m_Buffer[0]);
  }
  void Set(uint_t i,const data_t& t)
  {
    m_Buffer[i] = t;
  }
  data_t Get(uint_t i) const
  {
    return m_Buffer[i];
  }

  void Resize(uint_t size)
  {
    if(QubitVectorBuffer<data_t>::m_size != size){
      m_Buffer.resize(size);
      QubitVectorBuffer<data_t>::m_size = size;
    }
  }
  void Copy(const std::vector<data_t>& v)
  {
    m_Buffer = v;
  }

  void Copy(uint_t pos,QubitVectorBuffer<data_t>* pSrc,uint_t srcPos,uint_t size,int isDevice = 0);

  void CopyIn(uint_t pos,const data_t* pSrc,uint_t size);
  void CopyOut(uint_t pos,data_t* pDest,uint_t size);
};


template <typename data_t>
void QubitVectorDeviceBuffer<data_t>::Copy(uint_t pos,QubitVectorBuffer<data_t>* pSrc,uint_t srcPos,uint_t size,int isDevice)
{
  if(isDevice){
    QubitVectorDeviceBuffer<data_t>* pSrcDev = (QubitVectorDeviceBuffer<data_t>*)pSrc;
    thrust::copy_n(pSrcDev->Buffer().begin() + srcPos,size,m_Buffer.begin() + pos);
  }
  else{
    QubitVectorHostBuffer<data_t>* pSrcHost = (QubitVectorHostBuffer<data_t>*)pSrc;
    thrust::copy_n(pSrcHost->Buffer().begin() + srcPos,size,m_Buffer.begin() + pos);
  }
}

template <typename data_t>
void QubitVectorDeviceBuffer<data_t>::CopyIn(uint_t pos,const data_t* pSrc,uint_t size)
{
  thrust::copy_n(pSrc,size,m_Buffer.begin() + pos);
}

template <typename data_t>
void QubitVectorDeviceBuffer<data_t>::CopyOut(uint_t pos,data_t* pDest,uint_t size)
{
  thrust::copy_n(m_Buffer.begin() + pos,size,pDest);
}

template <typename data_t>
void QubitVectorHostBuffer<data_t>::Copy(uint_t pos,QubitVectorBuffer<data_t>* pSrc,uint_t srcPos,uint_t size,int isDevice)
{
  if(isDevice){
    QubitVectorDeviceBuffer<data_t>* pSrcDev = (QubitVectorDeviceBuffer<data_t>*)pSrc;
    thrust::copy_n(pSrcDev->Buffer().begin() + srcPos,size,m_Buffer.begin() + pos);
  }
  else{
    QubitVectorHostBuffer<data_t>* pSrcHost = (QubitVectorHostBuffer<data_t>*)pSrc;
    thrust::copy_n(pSrcHost->Buffer().begin() + srcPos,size,m_Buffer.begin() + pos);
  }
}

template <typename data_t>
void QubitVectorHostBuffer<data_t>::CopyIn(uint_t pos,const data_t* pSrc,uint_t size)
{
  thrust::copy_n(pSrc,size,m_Buffer.begin() + pos);
}
template <typename data_t>
void QubitVectorHostBuffer<data_t>::CopyOut(uint_t pos,data_t* pDest,uint_t size)
{
  thrust::copy_n(m_Buffer.begin() + pos,size,pDest);
}

//=============================================================
// chunk container class
//=============================================================
template <typename data_t>
class QubitVectorChunkContainer 
{
protected:
  QubitVectorBuffer<thrust::complex<data_t>>* m_pChunks;
  QubitVectorBuffer<thrust::complex<double>>* m_pMatrix;
  QubitVectorBuffer<uint_t>* m_pOffsets;
  QubitVectorBuffer<uint_t>* m_pParams;

  uint_t m_size;
  uint_t m_bufferSize;

  uint_t m_globalID;
  int m_iDevice;      //device ID : if device ID < 0, allocate chunks on host memory

  int m_matrixBits;

  std::vector<int> m_p2pEnable;
public:
  QubitVectorChunkContainer(void)
  {
    m_pChunks = NULL;
    m_pMatrix = NULL;
    m_pOffsets = NULL;
    m_pParams = NULL;

    m_size = 0;
    m_bufferSize = 0;

    m_matrixBits = 0;
    m_iDevice = -1;
    m_globalID = 0;
  }

  ~QubitVectorChunkContainer(void);

  void SetDevice(int iDev)
  {
    m_iDevice = iDev;
  }
  void SetGlobalIndex(uint_t idx)
  {
    m_globalID = idx;
  }
  uint_t GlobalIndex(void)
  {
    return m_globalID;
  }
  uint_t NumChunks(int chunkBits) const
  {
    return (m_size >> chunkBits);
  }
  uint_t ChunkID(uint_t id,int chunkBits) const
  {
    return (m_globalID >> chunkBits) + id;
  }
  uint_t LocalChunkID(uint_t id,int chunkBits) const
  {
    return id - (m_globalID >> chunkBits);
  }
  int DeviceID(void) const
  {
    return m_iDevice;
  }
  uint_t Size(void)
  {
    return m_size;
  }

  int Allocate(uint_t size,uint_t bufferSize = 0);
  int AllocateParameters(int bits);

  int Get(const QubitVectorChunkContainer& chunks,uint_t src,uint_t bufDest,int chunkBits);
  int Put(QubitVectorChunkContainer& chunks,uint_t dest,uint_t bufSrc,int chunkBits);

  int CopyIn(const thrust::complex<data_t>* pVec,uint_t offset,uint_t chunkID,int chunkBits);
  int CopyOut(thrust::complex<data_t>* pVec,uint_t offset,uint_t chunkID,int chunkBits);

  int SetState(uint_t chunkID,uint_t pos,thrust::complex<data_t> t,int chunkBits);
  int SetState(uint_t lid,thrust::complex<data_t> t);

  thrust::complex<data_t> GetState(uint_t chunkID,uint_t pos,int chunkBits) const;
  thrust::complex<data_t> GetState(uint_t lid) const;

  thrust::complex<data_t>* ChunkPtr(uint_t chunkID,int chunkBits) const;
  thrust::complex<data_t>* BufferPtr(uint_t ibuf,int chunkBits);

  void StoreMatrix(const std::vector<thrust::complex<double>>& mat);
  void StoreUintParams(const std::vector<uint_t>& prm);
  void StoreOffsets(const std::vector<uint_t>& ptr);

  template <typename Function>
  int Execute(std::vector<uint_t>& offsets,Function func,uint_t size,uint_t gid,uint_t localMask, bool omp_parallel);

  template <typename Function>
  double ExecuteSum(std::vector<uint_t>& offsets,Function func,uint_t size,uint_t gid, uint_t localMask, bool omp_parallel);

  void SetParams(struct GateParams<data_t>& params);


  void SetupP2P(int nDev);
};

template <typename data_t>
QubitVectorChunkContainer<data_t>::~QubitVectorChunkContainer(void)
{
  if(m_pChunks){
    delete m_pChunks;
  }
  if(m_pMatrix){
    delete m_pMatrix;
  }
  if(m_pOffsets){
    delete m_pOffsets;
  }
  if(m_pParams){
    delete m_pParams;
  }
}

//allocate buffer for chunks
template <typename data_t>
int QubitVectorChunkContainer<data_t>::Allocate(uint_t size_in,uint_t bufferSize)
{
  uint_t size = size_in + bufferSize;
  m_size = size_in;
  m_bufferSize = bufferSize;

  if(m_pChunks == NULL){
#ifdef AER_THRUST_CUDA
    if(m_iDevice >= 0){
      cudaSetDevice(m_iDevice);
      m_pChunks = new QubitVectorDeviceBuffer<thrust::complex<data_t>>(size);
    }
    else{
#endif
      m_pChunks = new QubitVectorHostBuffer<thrust::complex<data_t>>(size);
#ifdef AER_THRUST_CUDA
    }
#endif
  }
  else if(m_pChunks->Size() != size){
    if(m_iDevice >= 0){
#ifdef AER_THRUST_CUDA
      cudaSetDevice(m_iDevice);
#endif
    }
    m_pChunks->Resize(size);
  }
  return 0;
}

//allocate buffers for parameters
template <typename data_t>
int QubitVectorChunkContainer<data_t>::AllocateParameters(int bits)
{
  uint_t size;
  if(bits > m_matrixBits){
    size = 1ull << bits;

    if(m_iDevice >= 0){
#ifdef AER_THRUST_CUDA
      cudaSetDevice(m_iDevice);
#endif
    }

    if(m_pMatrix == NULL){
#ifdef AER_THRUST_CUDA
      if(m_iDevice >= 0){
        m_pMatrix = new QubitVectorDeviceBuffer<thrust::complex<double>>(size*size);
      }
      else{
#endif
        m_pMatrix = new QubitVectorHostBuffer<thrust::complex<double>>(size*size);
#ifdef AER_THRUST_CUDA
      }
#endif
    }
    else{
      m_pMatrix->Resize(size*size);
    }

    if(m_pOffsets == NULL){
#ifdef AER_THRUST_CUDA
      if(m_iDevice >= 0){
        m_pOffsets = new QubitVectorDeviceBuffer<uint_t>(size);
      }
      else{
#endif
        m_pOffsets = new QubitVectorHostBuffer<uint_t>(size);
#ifdef AER_THRUST_CUDA
      }
#endif
    }
    else{
      m_pOffsets->Resize(size);
    }

    if(m_pParams == NULL){
#ifdef AER_THRUST_CUDA
      if(m_iDevice >= 0){
        m_pParams = new QubitVectorDeviceBuffer<uint_t>(size*4);
      }
      else{
#endif
        m_pParams = new QubitVectorHostBuffer<uint_t>(size*4);
#ifdef AER_THRUST_CUDA
      }
#endif
    }
    else{
      m_pParams->Resize(size*4);
    }

    m_matrixBits = bits;
  }
  return 0;
}

//copy chunk from other container to buffer
template <typename data_t>
int QubitVectorChunkContainer<data_t>::Get(const QubitVectorChunkContainer& chunks,uint_t src,uint_t bufDest,int chunkBits)
{
  uint_t srcPos,destPos,size;
  srcPos = src << chunkBits;
  destPos = m_size + (bufDest << chunkBits);
  size = 1ull << chunkBits;

  if(m_iDevice >=0 && chunks.DeviceID() >= 0){
    if(m_p2pEnable[chunks.DeviceID()]){
      m_pChunks->Copy(destPos,chunks.m_pChunks,srcPos,size,1);
    }
    else{
      QubitVectorHostBuffer<thrust::complex<data_t>> tmp(size);
      tmp.Copy(0,chunks.m_pChunks,srcPos,size,1);   //D to H
      m_pChunks->Copy(destPos,&tmp,0,size,0);     //H to D
    }
  }
  else{
    m_pChunks->Copy(destPos,chunks.m_pChunks,srcPos,size,(chunks.DeviceID() >= 0));
  }

  return 0;
}

//copy chunk to other container from buffer
template <typename data_t>
int QubitVectorChunkContainer<data_t>::Put(QubitVectorChunkContainer& chunks,uint_t dest,uint_t bufSrc,int chunkBits)
{
  uint_t srcPos,destPos,size;
  destPos = dest << chunkBits;
  srcPos = m_size + (bufSrc << chunkBits);
  size = 1ull << chunkBits;

  if(m_iDevice >=0 && chunks.DeviceID() >= 0){
    if(m_p2pEnable[chunks.DeviceID()]){
      chunks.m_pChunks->Copy(destPos,m_pChunks,srcPos,size,1);
    }
    else{
      QubitVectorHostBuffer<thrust::complex<data_t>> tmp(size);
      tmp.Copy(0,m_pChunks,srcPos,size,1);      //D to H
      chunks.m_pChunks->Copy(destPos,&tmp,0,size,0);  //H to D
    }
  }
  else{
    chunks.m_pChunks->Copy(destPos,m_pChunks,srcPos,size,(DeviceID() >= 0));
  }

  return 0;
}

//copy chunk from std::vector
template <typename data_t>
int QubitVectorChunkContainer<data_t>::CopyIn(const thrust::complex<data_t>* pVec,uint_t offset,uint_t chunkID,int chunkBits)
{
  uint_t size,destPos;

  size = 1ull << chunkBits;
  destPos = chunkID << chunkBits;

  m_pChunks->CopyIn(destPos,pVec + offset,size);

  return 0;
}

//copy chunk to std::vector
template <typename data_t>
int QubitVectorChunkContainer<data_t>::CopyOut(thrust::complex<data_t>* pVec,uint_t offset,uint_t chunkID,int chunkBits)
{
  uint_t size,srcPos;

  size = 1ull << chunkBits;
  srcPos = chunkID << chunkBits;

  m_pChunks->CopyOut(srcPos,pVec + offset,size);

  return 0;
}


template <typename data_t>
thrust::complex<data_t>* QubitVectorChunkContainer<data_t>::ChunkPtr(uint_t chunkID,int chunkBits) const
{
  return m_pChunks->BufferPtr() + (chunkID << chunkBits);
}

template <typename data_t>
thrust::complex<data_t>* QubitVectorChunkContainer<data_t>::BufferPtr(uint_t ibuf,int chunkBits)
{
  return m_pChunks->BufferPtr() + m_size + (ibuf << chunkBits);
}


template <typename data_t>
int QubitVectorChunkContainer<data_t>::SetState(uint_t chunkID,uint_t pos,thrust::complex<data_t> t,int chunkBits)
{
  m_pChunks->Set((chunkID << chunkBits) + pos,t);
  return 0;
}

template <typename data_t>
int QubitVectorChunkContainer<data_t>::SetState(uint_t lid,thrust::complex<data_t> t)
{
  m_pChunks->Set(lid,t);
  return 0;
}

template <typename data_t>
thrust::complex<data_t> QubitVectorChunkContainer<data_t>::GetState(uint_t chunkID,uint_t pos,int chunkBits) const
{
  return m_pChunks->Get((chunkID << chunkBits) + pos);
}

template <typename data_t>
thrust::complex<data_t> QubitVectorChunkContainer<data_t>::GetState(uint_t lid) const
{
  return m_pChunks->Get(lid);
}

template <typename data_t>
void QubitVectorChunkContainer<data_t>::StoreMatrix(const std::vector<thrust::complex<double>>& mat)
{
  m_pMatrix->Copy(mat);
}

template <typename data_t>
void QubitVectorChunkContainer<data_t>::StoreUintParams(const std::vector<uint_t>& prm)
{
  m_pParams->Copy(prm);
}

template <typename data_t>
void QubitVectorChunkContainer<data_t>::StoreOffsets(const std::vector<uint_t>& offsets)
{
  m_pOffsets->Copy(offsets);
}

#define ExtractIndexFromTuple(itp)        thrust::get<0>(itp)
#define ExtractParamsFromTuple(itp)       thrust::get<1>(itp)

template <typename data_t>
template <typename Function>
int QubitVectorChunkContainer<data_t>::Execute(std::vector<uint_t>& offsets,Function func,uint_t size,uint_t gid,uint_t localMask, bool omp_parallel)
{
  struct GateParams<data_t> params;

  params.buf_ = m_pChunks->BufferPtr();
  if(m_iDevice >= 0){
    StoreOffsets(offsets);
    params.offsets_ = m_pOffsets->BufferPtr();
  }
  else{
    params.offsets_ = &offsets[0];
  }
  params.matrix_ = m_pMatrix->BufferPtr();
  params.params_ = m_pParams->BufferPtr();
  params.gid_ = gid;
  params.lmask_ = localMask;

  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::constant_iterator<struct GateParams<data_t>> cp(params);

  auto chunkTuple = thrust::make_tuple(ci,cp);
  auto chunkIter = thrust::make_zip_iterator(chunkTuple);

  if(m_iDevice >= 0){
#ifdef AER_THRUST_CUDA
    cudaSetDevice(m_iDevice);
#endif
    thrust::for_each_n(thrust::device, chunkIter, size, func);
  }
  else{
    if (omp_parallel) {
      thrust::for_each_n(thrust::omp::par, chunkIter, size, func);
    } else {
      thrust::for_each_n(thrust::seq, chunkIter, size, func);
    }
  }

  return 0;
}

template <typename data_t>
template <typename Function>
double QubitVectorChunkContainer<data_t>::ExecuteSum(std::vector<uint_t>& offsets,Function func,uint_t size,uint_t gid,uint_t localMask, bool omp_parallel)
{
  struct GateParams<data_t> params;
  double ret;

  params.buf_ = m_pChunks->BufferPtr();
  if(m_iDevice >= 0){
    StoreOffsets(offsets);
    params.offsets_ = m_pOffsets->BufferPtr();
  }
  else{
    params.offsets_ = &offsets[0];
  }
  params.matrix_ = m_pMatrix->BufferPtr();
  params.params_ = m_pParams->BufferPtr();
  params.gid_ = gid;
  params.lmask_ = localMask;

  auto ci = thrust::counting_iterator<uint_t>(0);
  thrust::constant_iterator<struct GateParams<data_t>> cp(params);

  auto chunkTuple = thrust::make_tuple(ci,cp);
  auto chunkIter = thrust::make_zip_iterator(chunkTuple);

  if(m_iDevice >= 0){
#ifdef AER_THRUST_CUDA
    cudaSetDevice(m_iDevice);
#endif
    ret = thrust::transform_reduce(thrust::device, chunkIter, chunkIter + size, func,0.0,thrust::plus<double>());
  }
  else{
    if (omp_parallel) {
      ret = thrust::transform_reduce(thrust::omp::par, chunkIter, chunkIter + size, func,0.0,thrust::plus<double>());
    } else {
      ret = thrust::transform_reduce(thrust::seq, chunkIter, chunkIter + size, func,0.0,thrust::plus<double>());
    }
  }
  return ret;
}

template <typename data_t>
void QubitVectorChunkContainer<data_t>::SetParams(struct GateParams<data_t>& params)
{
  params.buf_ = m_pChunks->BufferPtr();
  params.offsets_ = m_pOffsets->BufferPtr();
  params.matrix_ = m_pMatrix->BufferPtr();
  params.params_ = m_pParams->BufferPtr();
}

template <typename data_t>
void QubitVectorChunkContainer<data_t>::SetupP2P(int nDev)
{
  int i;
  if(nDev > 0){
    m_p2pEnable.resize(nDev);

    if(m_iDevice >= 0){
      for(i=0;i<nDev;i++){
        m_p2pEnable[i] = 0;
      }

#ifdef AER_THRUST_CUDA
      cudaSetDevice(m_iDevice);
      for(i=0;i<nDev;i++){
        if(i != m_iDevice){
          cudaDeviceCanAccessPeer(&m_p2pEnable[i],m_iDevice,i);
          if(m_p2pEnable[i])
            cudaDeviceEnablePeerAccess(i,0);
        }
        else{
          m_p2pEnable[i] = 1;
        }
      }
#endif
    }
    else{
      //H to D is always p2p
      for(i=0;i<nDev;i++){
        m_p2pEnable[i] = 1;
      }
    }
  }
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
  QubitVectorThrust(const QubitVectorThrust& obj) = delete;
  QubitVectorThrust &operator=(const QubitVectorThrust& obj) = delete;

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
  std::complex<data_t> &operator[](uint_t element);
  std::complex<data_t> operator[](uint_t element) const;

  void set_state(uint_t pos,std::complex<data_t>& c);
  std::complex<data_t> get_state(uint_t pos) const;

  // Returns a reference to the underlying data_t data class
  std::complex<data_t>* &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  std::complex<data_t>* data() const {return data_;}

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
  // of this.

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
  std::complex<data_t>* data_;    //this is allocated on host for reference
  std::complex<data_t>* checkpoint_;

  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
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
  double apply_function(Function func,const reg_t &qubits) const;

  void set_matrix(const cvector_t<double>& mat) const;
  void set_params(const reg_t& prm) const;

  void allocate_buffers(int qubit);

  int m_iDev;
  int m_nDev;
  int m_nDevParallel;
  int m_iPlaceHost;
  int m_nPlaces;

  mutable std::vector<QubitVectorChunkContainer<data_t>> m_Chunks;

  int m_maxChunkBits;           //bits per chunk
  uint_t m_globalSize;          //number of total states
  uint_t m_localSize;           //number of states in this process
  uint_t m_globalIndex;         //starting state ID for this process
  int m_maxNumBuffers;          //max number of buffer chunks

  mutable uint_t m_refPosition;         //position for reference (if >= data_size_ data_ is empty)


  int FindPlace(uint_t chunkID,int chunkBits) const;
  int GlobalToLocal(uint_t& lcid,uint_t& lid,uint_t gid,int chunkBits) const;
  uint_t GetBaseChunkID(const uint_t gid,const reg_t& qubits,const int chunkBits) const;

  void UpdateReferencedValue(void) const;

#ifdef AER_TIMING
  mutable uint_t m_gateCounts[QS_NUM_GATES];
  mutable double m_gateTime[QS_NUM_GATES];
  mutable double m_gateStartTime[QS_NUM_GATES];

  void TimeStart(int i) const;
  void TimeEnd(int i) const;
  void TimeReset(void);
  void TimePrint(void);
#endif

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
  int iPlace;
  uint_t i,ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;
  cvector_t<data_t> tmp(csize);

  const json_t ZERO = std::complex<data_t>(0.0, 0.0);
  json_t js = json_t(data_size_, ZERO);

#ifdef AER_DEBUG
  DebugMsg("json()");
#endif

  UpdateReferencedValue();

  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);

    for(ic=0;ic<nc;ic++){
      m_Chunks[iPlace].CopyOut((thrust::complex<data_t>*)&tmp[0],0,ic,m_maxChunkBits);

      if (json_chop_threshold_ > 0) {
        for(i=0;i<csize;i++){
          if (std::abs(tmp[i].real()) > json_chop_threshold_)
            js[pos+i][0] = tmp[i].real();
          if (std::abs(tmp[i].imag()) > json_chop_threshold_)
                js[pos+i][1] = tmp[i].imag();
        }
      }
      else{
        for(i=0;i<csize;i++){
          js[pos+i][0] = tmp[i].real();
              js[pos+i][1] = tmp[i].imag();
        }
      }
      pos += csize;
    }
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
  if (!checkpoint_) {
    throw std::runtime_error("QubitVectorThrust: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust(size_t num_qubits) : num_qubits_(0), data_(nullptr), checkpoint_(0)
{
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
QubitVectorThrust<data_t>::~QubitVectorThrust() {
#ifdef AER_TIMING
  TimePrint();
#endif

//  m_DeviceChunks.erase(m_DeviceChunks.begin(),m_DeviceChunks.end());

  if(data_)
    free(data_);

  if (checkpoint_)
    free(checkpoint_);

#ifdef AER_DEBUG
  if(debug_fp != NULL){
    fflush(debug_fp);
    if(debug_fp != stdout)
      fclose(debug_fp);
  }
#endif

}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
std::complex<data_t> &QubitVectorThrust<data_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif

  uint_t lcid,lid;
  int iPlace = GlobalToLocal(lcid,lid,element,m_maxChunkBits);

#ifdef AER_DEBUG
  DebugMsg(" calling ref []");
#endif

  UpdateReferencedValue();

  if(iPlace >= 0){
    data_[0] = (std::complex<data_t>)m_Chunks[iPlace].GetState(lcid,lid,m_maxChunkBits);
    m_refPosition = element;
  }
  else{
    data_[0] = 0.0;
    m_refPosition = data_size_;
  }
#ifdef AER_DEBUG
  DebugMsg("ref operator[]",(int)element);
  DebugMsg("          ",data_[0]);
  DebugDump();
#endif

  return data_[0];
}

template <typename data_t>
void QubitVectorThrust<data_t>::UpdateReferencedValue(void) const
{
  if(m_refPosition < data_size_){
    uint_t lcid,lid;
    int iPlace = GlobalToLocal(lcid,lid,m_refPosition,m_maxChunkBits);

    if(iPlace >= 0){
      m_Chunks[iPlace].SetState(lcid,lid,(thrust::complex<data_t>)data_[0],m_maxChunkBits);
    }
    m_refPosition = data_size_;
  }
}

template <typename data_t>
std::complex<data_t> QubitVectorThrust<data_t>::operator[](uint_t element) const
{
  uint_t lcid,lid;
  int iPlace = GlobalToLocal(lcid,lid,element,m_maxChunkBits);

#ifdef AER_DEBUG
    DebugMsg(" calling []");
#endif

  UpdateReferencedValue();

  if(iPlace >= 0){
    std::complex<data_t> ret;
    ret = (std::complex<data_t>)m_Chunks[iPlace].GetState(lcid,lid,m_maxChunkBits);

#ifdef AER_DEBUG
    DebugMsg("operator[]",(int)element);
    DebugMsg("          ",ret);
    DebugDump();
#endif
    return ret;
  }
  else{
    return data_[0];
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_state(uint_t pos, std::complex<data_t>& c)
{
  uint_t lcid,lid;
  int iPlace = GlobalToLocal(lcid,lid,pos,m_maxChunkBits);

  if(iPlace >= 0){
    m_Chunks[iPlace].SetState(lcid,lid,(thrust::complex<data_t>)c,m_maxChunkBits);
  }
}

template <typename data_t>
std::complex<data_t> QubitVectorThrust<data_t>::get_state(uint_t pos) const
{
  uint_t lcid,lid;
  std::complex<data_t> ret = 0.0;;
  int iPlace = GlobalToLocal(lcid,lid,pos,m_maxChunkBits);

  UpdateReferencedValue();

  if(iPlace >= 0){
    ret = (std::complex<data_t>)m_Chunks[iPlace].GetState(lcid,lid,m_maxChunkBits);
  }
  return ret;
}


template <typename data_t>
cvector_t<data_t> QubitVectorThrust<data_t>::vector() const 
{
  cvector_t<data_t> ret(data_size_, 0.);

  int iPlace;
  uint_t ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;

#ifdef AER_DEBUG
  DebugMsg("vector");
#endif

  UpdateReferencedValue();

  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
    for(ic=0;ic<nc;ic++){
      m_Chunks[iPlace].CopyOut((thrust::complex<data_t>*)&ret[0],pos,ic,m_maxChunkBits);
      pos += csize;
    }
  }

  return ret;
}

//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
class initialize_component_func : public GateFuncBase
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

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<double> q0;
    thrust::complex<double> q;
    thrust::complex<double>* state;
    uint_t* qubits;
    uint_t i,j,k;
    uint_t ii,idx,t;
    uint_t mask;
    struct GateParams<data_t> params;

    //get parameters from iterator
    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    state = params.matrix_;
    qubits = params.params_;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    q0 = pV[offsets[0]+idx];
    for(k=0;k<matSize;k++){
      q = q0 * state[k];
      pV[offsets[k] + idx] = q;
    }
    return 0.0;
  }

  const char* Name(void)
  {
    return "initialize_component";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0) 
{
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  set_matrix(state0);
  set_params(qubits_sorted);
  apply_function(initialize_component_func<data_t>(state0,qubits_sorted), qubits);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
template <typename data_t>
class fill_func : public GateFuncBase
{
protected:
  thrust::complex<data_t> val;
public:

  fill_func(thrust::complex<data_t>& v)
  {
    val = v;
  }

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i;
    thrust::complex<data_t>* pV;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;

    pV[i] = val;
    return 0.0;
  }
  const char* Name(void)
  {
    return "fill";
  }
};


template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
  thrust::complex<data_t> z = 0.0;

  reg_t qubits = {0};

  apply_function(fill_func<data_t>(z), qubits);
}


template <typename data_t>
void QubitVectorThrust<data_t>::set_num_qubits(size_t num_qubits)
{
  size_t prev_num_qubits = num_qubits_;
  num_qubits_ = num_qubits;
  data_size_ = 1ull << num_qubits;
  char* str;
  int i;

#ifdef AER_TIMING
  TimeReset();
  TimeStart(QS_GATE_INIT);
#endif
  if (checkpoint_) {
    free(checkpoint_);
    checkpoint_ = nullptr;
  }

  m_refPosition = data_size_;
  if (!data_)
    data_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>)));  //data_ is only allocated to store reference value

  // Free any currently assigned memory
  if(m_Chunks.size() > 0){
    if (prev_num_qubits != num_qubits_) {
      m_Chunks.erase(m_Chunks.begin(),m_Chunks.end());
    }
  }

  int tid,nid;
  nid = omp_get_num_threads();
  tid = omp_get_thread_num();
  m_nDev = 1;
#ifdef AER_THRUST_CUDA
  cudaGetDeviceCount(&m_nDev);
#endif

  m_iDev = 0;
  if(nid > 1 && m_nDev > 0){
    m_iDev = tid % m_nDev;
#ifdef AER_THRUST_CUDA
    cudaSetDevice(m_iDev);
#endif
    m_nDevParallel = 1;
  }
  else{
    m_nDevParallel = 1;
    str = getenv("AER_MULTI_GPU");
    if(str != NULL){
      m_nDevParallel = m_nDev;
    }
  }

  str = getenv("AER_HOST_ONLY");
  if(str || m_nDev == 0){
    m_nDevParallel = 0;
  }

  //chunk setting
  int numBuffers = AER_MAX_BUFFERS;
  //uint_t numBuffers = 1;
  int numDummy = 2;
  m_maxChunkBits = AER_CHUNK_BITS;
  str = getenv("AER_CHUNK_BITS");
  if(str){
    m_maxChunkBits = atol(str);
  }
//  else if(m_nDevParallel <= 1){
//    m_maxChunkBits = num_qubits_;
//  }

  if(m_maxChunkBits > num_qubits_){
    m_maxChunkBits = num_qubits_;
    i = m_nDevParallel;
    while(i > 1){
      m_maxChunkBits--;
      i >>= 1;
    }
  }

  //currently using only one process
  m_globalSize = 1ull << num_qubits_;
  m_localSize = m_globalSize;
  m_globalIndex = 0;
  //--

  int fitOneGPU = 0;

#ifdef AER_THRUST_CUDA
  if(m_nDevParallel == 1){
    size_t freeMem,totalMem;
    cudaMemGetInfo(&freeMem,&totalMem);

    if(m_localSize < (freeMem / ((uint_t)sizeof(thrust::complex<data_t>) )) ){
      fitOneGPU = 1;
    }
    else{
      //need multiple GPUs
      if(nid <= 1){
        m_nDevParallel = m_nDev;
      }
    }
  }
#else
  //for OMP, TBB use 1 device
  fitOneGPU = 1;

  omp_set_nested(1);
#endif

  m_Chunks.resize(m_nDevParallel+1);
  m_nPlaces = m_nDevParallel;
  m_iPlaceHost = -1;

  uint_t is,ie,chunksOnDevice = m_localSize >> m_maxChunkBits;
  str = getenv("AER_TEST_HYBRID");  //use only for debugging
  if(str != NULL){
    chunksOnDevice = chunksOnDevice/2;
  }
  else if(fitOneGPU){
    m_maxChunkBits = num_qubits_;
  }

  if(m_maxChunkBits == num_qubits_){
    //no buffer needed for chunk exchange
    numBuffers = 0;
    numDummy = 0;
    chunksOnDevice = 1;
  }

  is = 0;
  for(i=0;i<m_nDevParallel;i++){
    ie = is + ((i + 1) * chunksOnDevice / m_nDevParallel) - (i * chunksOnDevice / m_nDevParallel);

#ifdef AER_THRUST_CUDA
    cudaSetDevice((m_iDev + i) % m_nDev);

    //check we can store chunks or not
    size_t freeMem,totalMem;
    cudaMemGetInfo(&freeMem,&totalMem);
    if(ie - is + numBuffers + numDummy >= (freeMem / ((uint_t)sizeof(thrust::complex<data_t>) << m_maxChunkBits)) ){
      ie = is + (freeMem / ((uint_t)sizeof(thrust::complex<data_t>) << m_maxChunkBits)) - numBuffers - numDummy;
    }
#endif

    m_Chunks[i].SetGlobalIndex(m_globalIndex + (is << m_maxChunkBits));
#ifdef AER_THRUST_CUDA
    m_Chunks[i].SetDevice((m_iDev + i) % m_nDev);
#else
    m_Chunks[i].SetDevice(-1);
#endif
    m_Chunks[i].Allocate((ie - is) << m_maxChunkBits,numBuffers << m_maxChunkBits);
    m_Chunks[i].AllocateParameters(AER_DEFAULT_MATRIX_BITS);
    m_Chunks[i].SetupP2P(m_nDevParallel);

    is = ie;
  }

  if(is < (m_localSize >> m_maxChunkBits)){ //rest of chunks are stored on host memory
    m_iPlaceHost = m_nDevParallel;
    m_nPlaces = m_nDevParallel + 1;

    m_Chunks[m_iPlaceHost].SetGlobalIndex(m_globalIndex + (is << m_maxChunkBits));
    m_Chunks[m_iPlaceHost].SetDevice(-1);
    m_Chunks[m_iPlaceHost].Allocate(m_localSize - (is << m_maxChunkBits),numBuffers << m_maxChunkBits);
    m_Chunks[m_iPlaceHost].AllocateParameters(AER_DEFAULT_MATRIX_BITS);
    m_Chunks[m_iPlaceHost].SetupP2P(m_nDevParallel);

    //Host execution uses nested parallelizm
    omp_set_nested(1);
  }

#ifdef AER_TIMING
  TimeEnd(QS_GATE_INIT);
#endif


#ifdef AER_DEBUG
  //TODO Migrate to SpdLog
  if(debug_fp == NULL && tid == 0){
//    char filename[1024];
//    sprintf(filename,"logs/debug_%d.txt",getpid());
//    debug_fp = fopen(filename,"a");
    debug_fp = stdout;

    fprintf(debug_fp," ==== Thrust qubit vector initialization %d qubits ====\n",num_qubits_);
    fprintf(debug_fp,"    TEST : threads %d/%d , dev %d/%d, using %d devices\n",tid,nid,m_iDev,m_nDev,m_nDevParallel);
    fprintf(debug_fp,"    TEST : max chunk bit = %d, %d/%d states, gid = %d , numBuffer = %d\n",m_maxChunkBits,m_localSize,m_globalSize,m_globalIndex,numBuffers);
    fprintf(debug_fp,"    TEST : ");
    for(i=0;i<m_nPlaces;i++){
      if(m_Chunks[i].DeviceID() >= 0){
        fprintf(debug_fp," [%d] %d ",m_Chunks[i].DeviceID(),m_Chunks[i].NumChunks(m_maxChunkBits));
      }
      else{
        fprintf(debug_fp," [Host] %d ",m_Chunks[i].NumChunks(m_maxChunkBits));
      }
    }
    fprintf(debug_fp,"\n");
  }
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::allocate_buffers(int nq)
{
  //delete this
}

template <typename data_t>
size_t QubitVectorThrust<data_t>::required_memory_mb(uint_t num_qubits) const {

  size_t unit = std::log2(sizeof(std::complex<data_t>));
  size_t shift_mb = std::max<int_t>(0, num_qubits + unit - 20);
  size_t mem_mb = 1ULL << shift_mb;
  return mem_mb;
}


template <typename data_t>
void QubitVectorThrust<data_t>::checkpoint()
{
#ifdef AER_DEBUG
  DebugMsg("checkpoint");
  DebugDump();
#endif

  if (!checkpoint_)
    checkpoint_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>) * data_size_));

  int iPlace;
  uint_t ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;

  UpdateReferencedValue();

  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
    for(ic=0;ic<nc;ic++){
      m_Chunks[iPlace].CopyOut((thrust::complex<data_t>*)checkpoint_,pos,ic,m_maxChunkBits);
      pos += csize;
    }
  }
}


template <typename data_t>
void QubitVectorThrust<data_t>::revert(bool keep) {

  #ifdef DEBUG
  check_checkpoint();
  #endif

#ifdef AER_DEBUG
  DebugMsg("calling revert");
#endif

  int iPlace;
  uint_t ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;
  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
    for(ic=0;ic<nc;ic++){
      m_Chunks[iPlace].CopyIn((thrust::complex<data_t>*)checkpoint_,pos,ic,m_maxChunkBits);
      pos += csize;
    }
  }
#ifdef AER_DEBUG
  DebugMsg("revert");
  DebugDump();
#endif

}

template <typename data_t>
std::complex<double> QubitVectorThrust<data_t>::inner_product() const
{
  double d = 0.0;
  int iPlace;
  uint_t i,ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;
  cvector_t<data_t> tmp(csize);

#ifdef AER_DEBUG
  DebugMsg("calling inner_product");
#endif

  UpdateReferencedValue();

  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
    for(ic=0;ic<nc;ic++){
      m_Chunks[iPlace].CopyOut((thrust::complex<data_t>*)&tmp[0],0,ic,m_maxChunkBits);

      for(i=0;i<csize;i++){
        d += std::real(tmp[i]) * std::real(checkpoint_[pos + i]) + std::imag(tmp[i]) * std::imag(checkpoint_[pos + i]);
      }

      pos += csize;
    }
  }
#ifdef AER_DEBUG
  DebugMsg("inner_product",std::complex<double>(d,0.0));
#endif

  return std::complex<double>(d,0.0);
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

  if(m_globalIndex == 0){
    m_Chunks[0].SetState(0,0,t,m_maxChunkBits);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_vector(const cvector_t<double> &statevec) {
  if (data_size_ != statevec.size()) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" + 
                        std::to_string(data_size_) + "!=" +
                        std::to_string(statevec.size()) + ")";
    throw std::runtime_error(error);
  }
#ifdef AER_DEBUG
  DebugMsg("calling initialize_from_vector");
#endif

  int iPlace;
  uint_t i,ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;

  cvector_t<data_t> tmp(csize);

  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
    for(ic=0;ic<nc;ic++){
      for(i=0;i<csize;i++){
        std::complex<data_t> t((data_t)std::real(statevec[pos + i]), (data_t)std::imag(statevec[pos + i]));
        tmp[i] = t;
      }

      m_Chunks[iPlace].CopyIn((thrust::complex<data_t>*)&tmp[0],0,ic,m_maxChunkBits);
      pos += csize;
    }
  }
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
  int iPlace;
  uint_t ic,nc;
  uint_t pos = 0;
  uint_t csize = 1ull << m_maxChunkBits;
  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
    for(ic=0;ic<nc;ic++){
      m_Chunks[iPlace].CopyIn((thrust::complex<data_t>*)statevec,pos,ic,m_maxChunkBits);
      pos += csize;
    }
  }

#ifdef AER_DEBUG
  DebugMsg("initialize_from_data");
  DebugDump();
#endif

}

//--------------------------------------------------------------------------------------
//  gate kernel execution
//--------------------------------------------------------------------------------------

template <typename data_t>
int QubitVectorThrust<data_t>::FindPlace(uint_t chunkID,int chunkBits) const
{
  int i;
  uint_t ids,baseGID,endGID;

  baseGID = m_globalIndex;
  endGID = m_globalIndex + m_localSize;
  if(chunkID < baseGID || chunkID >= endGID){
    return -1;    //not in this process
  }

  for(i=0;i<m_nPlaces;i++){
    ids = m_Chunks[i].ChunkID(0,chunkBits);
    if(chunkID >= ids && chunkID < ids + m_Chunks[i].NumChunks(chunkBits)){
      return i;
    }
  }

  return -1;
}

template <typename data_t>
int QubitVectorThrust<data_t>::GlobalToLocal(uint_t& lcid,uint_t& lid,uint_t gid,int chunkBits) const
{
  uint_t gcid = (gid >> chunkBits);
  int iPlace = FindPlace(gcid,chunkBits);
  if(iPlace >= 0){
    lcid = gcid - m_Chunks[iPlace].ChunkID(0,chunkBits);
    lid = gid - (gcid << chunkBits);
  }
  return iPlace;
}

template <typename data_t>
uint_t QubitVectorThrust<data_t>::GetBaseChunkID(const uint_t gid,const reg_t& qubits,const int chunkBits) const
{
  int i,n;
  uint_t base = gid;
  n = qubits.size();
  for(i=0;i<n;i++){
    base &= ~(1ull << (qubits[i] - chunkBits));
  }
  return base;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_matrix(const cvector_t<double>& mat) const
{
  uint_t i,size = mat.size();
  std::vector<thrust::complex<double>> matTh(size);

  for(i=0;i<size;i++){
    matTh[i] = (thrust::complex<double>)mat[i];
  }

  //store matrix to devices
  for(i=0;i<m_nPlaces;i++){
    m_Chunks[i].StoreMatrix(matTh);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_params(const reg_t& prm) const
{
  int i;
  //store params to devices
  for(i=0;i<m_nPlaces;i++){
    m_Chunks[i].StoreUintParams(prm);
  }
}

template <typename data_t>
template <typename Function>
double QubitVectorThrust<data_t>::apply_function(Function func,const reg_t &qubits) const
{
  const size_t N = qubits.size();
  const int numCBits = func.NumControlBits();
  uint_t size,iChunk,nChunk,controlMask,controlFlag;
  int i,ib,nBuf;
  int nSmall,nLarge = 0;
  reg_t large_qubits;
  int chunkBits;
  double ret = 0.0;
  int noDataExchange = 0;

#ifdef AER_DEBUG
  DebugMsg(func.Name(),qubits);
#endif

  UpdateReferencedValue();

  //decreasing chunk-bits for fusion
  chunkBits = m_maxChunkBits - (N - 1);

  //If no data exchange required execute along with all the state vectors
  if(m_nPlaces == 1 || func.IsDiagonal()){    //note: for multi-process m_nPlaces == 1 is not valid
    noDataExchange = 1;
    chunkBits = num_qubits_;
  }

  //count number of qubits which is larger than chunk size
  for(ib=numCBits;ib<N;ib++){
    if(qubits[ib] >= chunkBits){
      large_qubits.push_back(qubits[ib]);
      nLarge++;
    }
  }
  nSmall = N - nLarge - numCBits;

  if(nLarge == 0){
    noDataExchange = 1;
    chunkBits = num_qubits_;
  }

  if(func.IsDiagonal()){
    size = 1ull << chunkBits;
    nBuf = 1;
    nChunk = 1;
  }
  else{
    size = 1ull << (chunkBits - nSmall);
    nBuf = 1ull << (N - numCBits);
    nChunk = 1ull << nLarge;
  }

  //setup buffer configuration
  reg_t offsetBase(nBuf,0);
  reg_t buf2chunk(nBuf,0);

  iChunk = 1;
  for(ib=numCBits;ib<N;ib++){
    if(qubits[ib] >= chunkBits){
      for(i=0;i<nBuf;i++){
        if((i >> (ib-numCBits)) & 1){
          buf2chunk[i] += iChunk;
        }
      }
      iChunk <<= 1;
    }
    else{
      for(i=0;i<nBuf;i++){
        if((i >> (ib-numCBits)) & 1){
          offsetBase[i] += (1ull << qubits[ib]);
        }
      }
    }
  }

  if(noDataExchange){
#pragma omp parallel if (num_qubits_ > omp_threshold_ && m_nPlaces > 1) private(size,iChunk,i,ib) num_threads(m_nPlaces)
    {
      int iPlace = omp_get_thread_num();
      uint_t localMask = 0xffffffff;

      size = m_Chunks[iPlace].Size();
      if(!func.IsDiagonal()){
        size >>= (N - numCBits);
      }

      //execute kernel
      bool enable_omp = (num_qubits_ > omp_threshold_ && omp_threads_ > 1);
      if(func.Reduction())
        ret += m_Chunks[iPlace].ExecuteSum(offsetBase,func,size,m_Chunks[iPlace].ChunkID(0,0),localMask, enable_omp);
      else
        m_Chunks[iPlace].Execute(offsetBase,func,size,m_Chunks[iPlace].ChunkID(0,0),localMask, enable_omp);
    }
  }
  else{
    controlMask = 0;
    controlFlag = 0;
    for(ib=0;ib<numCBits;ib++){
      if(qubits[ib] >= chunkBits)
        controlMask |= (1ull << (qubits[ib] - chunkBits));
    }
    if(func.ControlMask() != 0){
      controlFlag = controlMask;
    }

#pragma omp parallel if (num_qubits_ > omp_threshold_ && m_nPlaces > 1) private(iChunk,i,ib) num_threads(m_nPlaces)
    {
      int iPlace = omp_get_thread_num();
      int iPlaceSrc;
      uint_t localMask,baseChunk;
      reg_t offsets(nBuf);
      reg_t chunkOffsets(nChunk);
      reg_t chunkIDs(nChunk);
      std::vector<int> places(nChunk);

      for(iChunk=0;iChunk<m_Chunks[iPlace].NumChunks(chunkBits);iChunk++){
        baseChunk = GetBaseChunkID(m_Chunks[iPlace].ChunkID(iChunk,chunkBits),large_qubits,chunkBits);
        if(baseChunk != m_Chunks[iPlace].ChunkID(iChunk,chunkBits)){  //already calculated
          continue;
        }

        //control mask
        if((baseChunk & controlMask) != controlFlag){
          continue;
        }

        for(i=0;i<nChunk;i++){
          chunkIDs[i] = baseChunk;
          for(ib=0;ib<nLarge;ib++){
            if((i >> ib) & 1){
              chunkIDs[i] += (1ull << (large_qubits[ib] - chunkBits));
            }
          }
          iPlaceSrc = FindPlace(chunkIDs[i],chunkBits);
          places[i] = iPlaceSrc;
          if(iPlaceSrc == iPlace){
            chunkOffsets[i] = m_Chunks[iPlace].LocalChunkID(chunkIDs[i],chunkBits) << chunkBits;
          }
          else{
            m_Chunks[iPlace].Get(m_Chunks[iPlaceSrc],m_Chunks[iPlaceSrc].LocalChunkID(chunkIDs[i],chunkBits),i,chunkBits);  //copy chunk from other place
            chunkOffsets[i] = m_Chunks[iPlace].Size() + (i << chunkBits); //offset to buffer
          }
        }

        //setting buffers
        localMask = 0;
        for(i=0;i<nBuf;i++){
          offsets[i] = chunkOffsets[buf2chunk[i]] + offsetBase[i];
          localMask |= (1ull << i); //currently all buffers are local
        }

        //execute kernel
        bool enable_omp = (num_qubits_ > omp_threshold_ && omp_threads_ > 1);
        if(func.Reduction())
          ret += m_Chunks[iPlace].ExecuteSum(offsets,func,size,(baseChunk << chunkBits),localMask, enable_omp);
        else
          m_Chunks[iPlace].Execute(offsets,func,size,(baseChunk << chunkBits),localMask, enable_omp);

        //copy back
        for(i=0;i<nChunk;i++){
          if(places[i] != iPlace){
            m_Chunks[iPlace].Put(m_Chunks[places[i]],m_Chunks[places[i]].LocalChunkID(chunkIDs[i],chunkBits),i,chunkBits);
          }
        }
      }
    }
  }

#ifdef AER_DEBUG
  if(func.Reduction())
    DebugMsg("   ret = ",ret);
  else
    DebugDump();
#endif

  return ret;
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
class MatrixMult2x2 : public GateFuncBase
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit;
  uint_t mask;

public:
  MatrixMult2x2(const cvector_t<double>& mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    mask = (1ull << qubit) - 1;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* pV0;
    thrust::complex<data_t>* pV1;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    localMask = params.lmask_;

    pV0 = pV + offsets[0];
    pV1 = pV + offsets[1];

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = pV0[i0];
    q1 = pV1[i0];

    if((localMask & 1) == 1)
      pV0[i0] = m0 * q0 + m2 * q1;
    if((localMask & 2) == 2)
      pV1[i0] = m1 * q0 + m3 * q1;
    return 0.0;
  }
  const char* Name(void)
  {
    return "mult2x2";
  }
};


template <typename data_t>
class MatrixMult4x4 : public GateFuncBase
{
protected:
  thrust::complex<double> m00,m10,m20,m30;
  thrust::complex<double> m01,m11,m21,m31;
  thrust::complex<double> m02,m12,m22,m32;
  thrust::complex<double> m03,m13,m23,m33;
  int qubit0;
  int qubit1;
  uint_t mask0;
  uint_t mask1;

public:
  MatrixMult4x4(const cvector_t<double>& mat,int q0,int q1)
  {
    qubit0 = q0;
    qubit1 = q1;

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

    mask0 = (1ull << qubit0) - 1;
    mask1 = (1ull << qubit1) - 1;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1,i2,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1,q2,q3;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    localMask = params.lmask_;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q0 = pV[offsets[0] + i0];
    q1 = pV[offsets[1] + i0];
    q2 = pV[offsets[2] + i0];
    q3 = pV[offsets[3] + i0];

    if(localMask & 1)
      pV[offsets[0]+i0] = m00 * q0 + m10 * q1 + m20 * q2 + m30 * q3;

    if(localMask & 2)
      pV[offsets[1]+i0] = m01 * q0 + m11 * q1 + m21 * q2 + m31 * q3;

    if(localMask & 4)
      pV[offsets[2]+i0] = m02 * q0 + m12 * q1 + m22 * q2 + m32 * q3;

    if(localMask & 8)
      pV[offsets[3]+i0] = m03 * q0 + m13 * q1 + m23 * q2 + m33 * q3;
    return 0.0;
  }
  const char* Name(void)
  {
    return "mult4x4";
  }
};

template <typename data_t>
class MatrixMult8x8 : public GateFuncBase
{
protected:
  int qubit0;
  int qubit1;
  int qubit2;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;

public:
  MatrixMult8x8(int q0,int q1,int q2)
  {
    qubit0 = q0;
    qubit1 = q1;
    qubit2 = q2;

    mask0 = (1ull << qubit0) - 1;
    mask1 = (1ull << qubit1) - 1;
    mask2 = (1ull << qubit2) - 1;
  }


  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1,i2,i3,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
    thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
    thrust::complex<double>* pMat;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    pMat = params.matrix_;
    localMask = params.lmask_;

    i0 = i & mask0;
    i3 = (i - i0) << 1;
    i1 = i3 & mask1;
    i3 = (i3 - i1) << 1;
    i2 = i3 & mask2;
    i3 = (i3 - i2) << 1;

    i0 = i0 + i1 + i2 + i3;

    q0 = pV[offsets[0]+i0];
    q1 = pV[offsets[1]+i0];
    q2 = pV[offsets[2]+i0];
    q3 = pV[offsets[3]+i0];
    q4 = pV[offsets[4]+i0];
    q5 = pV[offsets[5]+i0];
    q6 = pV[offsets[6]+i0];
    q7 = pV[offsets[7]+i0];

    if(localMask & 1){
      m0 = pMat[0];
      m1 = pMat[8];
      m2 = pMat[16];
      m3 = pMat[24];
      m4 = pMat[32];
      m5 = pMat[40];
      m6 = pMat[48];
      m7 = pMat[56];

      pV[offsets[0]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 2){
      m0 = pMat[1];
      m1 = pMat[9];
      m2 = pMat[17];
      m3 = pMat[25];
      m4 = pMat[33];
      m5 = pMat[41];
      m6 = pMat[49];
      m7 = pMat[57];

      pV[offsets[1]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 4){
      m0 = pMat[2];
      m1 = pMat[10];
      m2 = pMat[18];
      m3 = pMat[26];
      m4 = pMat[34];
      m5 = pMat[42];
      m6 = pMat[50];
      m7 = pMat[58];

      pV[offsets[2]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 8){
      m0 = pMat[3];
      m1 = pMat[11];
      m2 = pMat[19];
      m3 = pMat[27];
      m4 = pMat[35];
      m5 = pMat[43];
      m6 = pMat[51];
      m7 = pMat[59];

      pV[offsets[3]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 16){
      m0 = pMat[4];
      m1 = pMat[12];
      m2 = pMat[20];
      m3 = pMat[28];
      m4 = pMat[36];
      m5 = pMat[44];
      m6 = pMat[52];
      m7 = pMat[60];

      pV[offsets[4]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 32){
      m0 = pMat[5];
      m1 = pMat[13];
      m2 = pMat[21];
      m3 = pMat[29];
      m4 = pMat[37];
      m5 = pMat[45];
      m6 = pMat[53];
      m7 = pMat[61];

      pV[offsets[5]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 64){
      m0 = pMat[6];
      m1 = pMat[14];
      m2 = pMat[22];
      m3 = pMat[30];
      m4 = pMat[38];
      m5 = pMat[46];
      m6 = pMat[54];
      m7 = pMat[62];

      pV[offsets[6]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }

    if(localMask & 128){
      m0 = pMat[7];
      m1 = pMat[15];
      m2 = pMat[23];
      m3 = pMat[31];
      m4 = pMat[39];
      m5 = pMat[47];
      m6 = pMat[55];
      m7 = pMat[63];

      pV[offsets[7]+i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
    }
    return 0.0;
  }
  const char* Name(void)
  {
    return "mult8x8";
  }
};

template <typename data_t>
class MatrixMult16x16 : public GateFuncBase
{
protected:
  int qubit0;
  int qubit1;
  int qubit2;
  int qubit3;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;
  uint_t mask3;
public:
  MatrixMult16x16(int q0,int q1,int q2,int q3)
  {
    qubit0 = q0;
    qubit1 = q1;
    qubit2 = q2;
    qubit3 = q3;

    mask0 = (1ull << qubit0) - 1;
    mask1 = (1ull << qubit1) - 1;
    mask2 = (1ull << qubit2) - 1;
    mask3 = (1ull << qubit3) - 1;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1,i2,i3,i4,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
    thrust::complex<data_t> q8,q9,q10,q11,q12,q13,q14,q15;
    thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
    thrust::complex<double> m8,m9,m10,m11,m12,m13,m14,m15;
    thrust::complex<double>* pMat;
    int j;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    pMat = params.matrix_;
    localMask = params.lmask_;

    i0 = i & mask0;
    i4 = (i - i0) << 1;
    i1 = i4 & mask1;
    i4 = (i4 - i1) << 1;
    i2 = i4 & mask2;
    i4 = (i4 - i2) << 1;
    i3 = i4 & mask3;
    i4 = (i4 - i3) << 1;

    i0 = i0 + i1 + i2 + i3 + i4;

    q0 = pV[offsets[0]+i0];
    q1 = pV[offsets[1]+i0];
    q2 = pV[offsets[2]+i0];
    q3 = pV[offsets[3]+i0];
    q4 = pV[offsets[4]+i0];
    q5 = pV[offsets[5]+i0];
    q6 = pV[offsets[6]+i0];
    q7 = pV[offsets[7]+i0];
    q8 = pV[offsets[8]+i0];
    q9 = pV[offsets[9]+i0];
    q10 = pV[offsets[10]+i0];
    q11 = pV[offsets[11]+i0];
    q12 = pV[offsets[12]+i0];
    q13 = pV[offsets[13]+i0];
    q14 = pV[offsets[14]+i0];
    q15 = pV[offsets[15]+i0];

    for(j=0;j<16;j++){
      if(((localMask >> j) & 1) == 0){
        continue;
      }
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

      pV[offsets[j]+i0] =   m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
              m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;
    }
    return 0.0;
  }
  const char* Name(void)
  {
    return "mult16x16";
  }
};


//in-place NxN matrix multiplication using LU factorization
template <typename data_t>
class MatrixMultNxN_LU : public GateFuncBase
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

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    thrust::complex<data_t> q,qt;
    thrust::complex<double> m;
    thrust::complex<double> r;
    uint_t i,j,k,l;
    uint_t ii,idx,t;
    uint_t mask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<double>* pMat;
    uint_t* qubits;
    uint_t* pivot;
    uint_t* table;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    pMat = params.matrix_;
    qubits = params.params_;

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
        q = pV[offsets[k]+idx];

        r += m*q;
      }
      pV[offsets[j]+idx] = r;
    }

    //mult L
    for(j=matSize-1;j>0;j--){
      r = pV[offsets[j]+idx];

      for(k=0;k<j;k++){
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];
        q = pV[offsets[k]+idx];

        r += m*q;
      }
      pV[offsets[j]+idx] = r;
    }

    //swap results
    if(nswap > 0){
      q = pV[offsets[table[0]]+idx];
      k = pivot[table[0]];
      for(j=1;j<nswap;j++){
        qt = pV[offsets[table[j]]+idx];
        pV[offsets[k]+idx] = q;
        q = qt;
        k = pivot[table[j]];
      }
      pV[offsets[k]+idx] = q;
    }
    return 0.0;
  }
  const char* Name(void)
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
    apply_function(MatrixMult2x2<data_t>(mat,qubits_sorted[0]), qubits);
  }
  else if(N == 2){
    apply_function(MatrixMult4x4<data_t>(mat,qubits_sorted[0],qubits_sorted[1]), qubits);
  }
  else if(N == 3){
    set_matrix(mat);
    apply_function(MatrixMult8x8<data_t>(qubits_sorted[0],qubits_sorted[1],qubits_sorted[2]), qubits);
  }
  else if(N == 4){
    set_matrix(mat);
    apply_function(MatrixMult16x16<data_t>(qubits_sorted[0],qubits_sorted[1],qubits_sorted[2],qubits_sorted[3]), qubits);
  }
  else{
    cvector_t<double> matLU;
    reg_t params;
    MatrixMultNxN_LU<data_t> f(mat,qubits_sorted,matLU,params);
    set_matrix(matLU);
    set_params(params);
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
class DiagonalMult2x2 : public GateFuncBase
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

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,gid;
    thrust::complex<data_t> q;
    thrust::complex<data_t>* pV;
    thrust::complex<double> m;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    gid = params.gid_;

    q = pV[i];
    if((((i + gid) >> qubit) & 1) == 0){
      m = m0;
    }
    else{
      m = m1;
    }

    pV[i] = m * q;
    return 0.0;
  }
  const char* Name(void)
  {
    return "diagonal_mult2x2";
  }
};

template <typename data_t>
class DiagonalMultNxN : public GateFuncBase
{
protected:
  int nqubits;

public:
  DiagonalMultNxN(const reg_t &qb)
  {
    nqubits = qb.size();
  }

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,j,im,gid;
    thrust::complex<data_t>* pV;
    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double>* pMat;
    uint_t* qubits;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    pMat = params.matrix_;
    qubits = params.params_;
    gid = params.gid_;

    im = 0;
    for(j=0;j<nqubits;j++){
      if(((i + gid) & (1ull << qubits[j])) != 0){
        im += (1 << j);
      }
    }

    q = pV[i];
    m = pMat[im];

    pV[i] = m * q;
    return 0.0;
  }
  const char* Name(void)
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
  else{
    set_matrix(diag);
    set_params(qubits);
    apply_function(DiagonalMultNxN<data_t>(qubits), qubits);
  }

#ifdef AER_TIMING
  TimeEnd(QS_GATE_DIAG);
#endif
}


template <typename data_t>
class Permutation : public GateFuncBase
{
protected:
  uint_t matSize;
  int nqubits;
  int npairs;

public:
  Permutation(const reg_t& qb,const std::vector<std::pair<uint_t, uint_t>> &pairs_in,reg_t& params)
  {
    uint_t j;

    nqubits = qb.size();
    matSize = 1ull << nqubits;
    npairs = pairs_in.size();

    params.resize(nqubits + npairs*2);

    for(j=0;j<nqubits;j++){
      params[j] = qb[j];
    }
    for(j=0;j<npairs;j++){
      params[nqubits + j*2  ] = pairs_in[j].first;
      params[nqubits + j*2+1] = pairs_in[j].second;
    }
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q;
    uint_t j,ip0,ip1;
    uint_t ii,idx,t;
    uint_t mask;
    uint_t* pairs;
    uint_t* qubits;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    qubits = params.params_;
    pairs = qubits + nqubits;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    for(j=0;j<npairs;j++){
      ip0 = pairs[j*2];
      ip1 = pairs[j*2+1];
      q = pV[offsets[ip0]+idx];

      pV[offsets[ip0]+idx] = pV[offsets[ip1]+idx];
      pV[offsets[ip1]+idx] = q;
    }
    return 0.0;
  }
  const char* Name(void)
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
  Permutation<data_t> f(qubits_sorted,pairs,params);
  set_params(params);
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
class CX_func : public GateFuncBase
{
protected:
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
    mask = (1ull << qubit_t) - 1;

    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int NumControlBits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1;
    uint_t gid,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* pV0;
    thrust::complex<data_t>* pV1;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    gid = params.gid_;
    localMask = params.lmask_;

    pV0 = pV + offsets[0];
    pV1 = pV + offsets[1];

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if(((i0 + gid) & cmask) == cmask){
      q0 = pV0[i0];
      q1 = pV1[i0];

      if((localMask & 1) == 1)
        pV0[i0] = q1;
      if((localMask & 2) == 2)
        pV1[i0] = q0;
    }
    return 0.0;
  }
  const char* Name(void)
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
class CY_func : public GateFuncBase
{
protected:
  uint_t mask;
  uint_t cmask;
  int nqubits;
  int qubit_t;
public:
  CY_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits-1];
    mask = (1ull << qubit_t) - 1;

    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int NumControlBits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1;
    uint_t gid,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* pV0;
    thrust::complex<data_t>* pV1;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    gid = params.gid_;
    localMask = params.lmask_;

    pV0 = pV + offsets[0];
    pV1 = pV + offsets[1];

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if(((i0 + gid) & cmask) == cmask){
      q0 = pV0[i0];
      q1 = pV1[i0];

      if((localMask & 1) == 1)
        pV0[i0] = thrust::complex<data_t>(q1.imag(),-q1.real());
      if((localMask & 2) == 2)
        pV1[i0] = thrust::complex<data_t>(-q0.imag(),q0.real());
    }
    return 0.0;
  }
  const char* Name(void)
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
class CSwap_func : public GateFuncBase
{
protected:
  uint_t mask0;
  uint_t mask1;
  uint_t cmask;
  int nqubits;
  int qubit_t0;
  int qubit_t1;
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

    cmask = 0;
    for(i=0;i<nqubits-2;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int NumControlBits(void)
  {
    return nqubits - 2;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1,i2;
    uint_t gid,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q1,q2;
    thrust::complex<data_t>* pV1;
    thrust::complex<data_t>* pV2;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    gid = params.gid_;
    localMask = params.lmask_;

    pV1 = pV + offsets[1];
    pV2 = pV + offsets[2];

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    if(((i0+gid) & cmask) == cmask){
      q1 = pV1[i0];
      q2 = pV2[i0];
      if(localMask & 2)
        pV1[i0] = q2;
      if(localMask & 4)
        pV2[i0] = q1;
    }
    return 0.0;
  }
  const char* Name(void)
  {
    return "CSWAP";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits)
{
  apply_function(CSwap_func<data_t>(qubits), qubits);
}

template <typename data_t>
class phase_func : public GateFuncBase 
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
  int NumControlBits(void)
  {
    return nqubits - 1;
  }

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,gid;
    thrust::complex<data_t>* pV;
    thrust::complex<data_t> q0;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    gid = params.gid_;

    if(((i+gid) & mask) == mask){
      q0 = pV[i];
      pV[i] = q0 * phase;
    }
    return 0.0;
  }
  const char* Name(void)
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
class DiagonalMult2x2Controlled : public GateFuncBase 
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

  int NumControlBits(void)
  {
    return nqubits - 1;
  }

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,gid;
    thrust::complex<data_t>* pV;
    thrust::complex<data_t> q0;
    thrust::complex<double> m;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    gid = params.gid_;

    if(((i + gid) & cmask) == cmask){
      if((i + gid) & mask){
        m = m1;
      }
      else{
        m = m0;
      }

      q0 = pV[i];
      pV[i] = m*q0;
    }
    return 0.0;
  }
  const char* Name(void)
  {
    return "diagonal_Cmult2x2";
  }
};

template <typename data_t>
class MatrixMult2x2Controlled : public GateFuncBase 
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  uint_t mask;
  uint_t cmask;
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

    mask = (1ull << qubits[nqubits-1]) - 1;
    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int NumControlBits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1;
    uint_t gid,localMask;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* pV0;
    thrust::complex<data_t>* pV1;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    gid = params.gid_;
    localMask = params.lmask_;

    pV0 = pV + offsets[0];
    pV1 = pV + offsets[1];

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if(((i0+gid) & cmask) == cmask){
      q0 = pV0[i0];
      q1 = pV1[i0];

      if(localMask & 1)
        pV0[i0]  = m0 * q0 + m2 * q1;
      if(localMask & 2)
        pV1[i0] = m1 * q0 + m3 * q1;
    }
    return 0.0;
  }
  const char* Name(void)
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
class Norm : public GateFuncBase
{
protected:

public:
  Norm()
  {

  }

  bool IsDiagonal(void)
  {
    return true;
  }
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i;
    thrust::complex<data_t>* pV;
    thrust::complex<data_t> q0;
    double ret;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;

    ret = 0.0;

    q0 = pV[i];
    ret = q0.real()*q0.real() + q0.imag()*q0.imag();

    return ret;
  }
  const char* Name(void)
  {
    return "Norm";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm() const
{
  reg_t qubits = {0};
  double ret = apply_function(Norm<data_t>(),qubits);
  return ret;
}

template <typename data_t>
class NormMatrixMultNxN : public GateFuncBase
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
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<double>* pMat;

    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double> r;
    double sum = 0.0;
    uint_t j,k,l;
    uint_t ii,idx,t;
    uint_t mask;
    uint_t* qubits;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;
    pMat = params.matrix_;
    qubits = params.params_;

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
        q = pV[offsets[k]+idx];
        r += m*q;
      }
      sum += (r.real()*r.real() + r.imag()*r.imag());
    }
    return sum;
  }
  const char* Name(void)
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
    set_matrix(mat);
    set_params(qubits);
    double ret = apply_function(NormMatrixMultNxN<data_t>(mat,qubits), qubits);
    return ret;
  }
}

template <typename data_t>
class NormDiagonalMultNxN : public GateFuncBase
{
protected:
  int nqubits;
public:
  NormDiagonalMultNxN(const reg_t &qb)
  {
    nqubits = qb.size();
  }

  bool IsDiagonal(void)
  {
    return true;
  }
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,im,j,gid;
    thrust::complex<data_t> q;
    thrust::complex<double> m,r;
    thrust::complex<double>* pMat;
    thrust::complex<data_t>* pV;
    uint_t* qubits;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    pMat = params.matrix_;
    qubits = params.params_;
    gid = params.gid_;

    im = 0;
    for(j=0;j<nqubits;j++){
      if(((i+gid) & (1ull << qubits[j])) != 0){
        im += (1 << j);
      }
    }

    q = pV[i];
    m = pMat[im];

    r = m * q;
    return (r.real()*r.real() + r.imag()*r.imag());
  }
  const char* Name(void)
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
    set_matrix(mat);
    set_params(qubits);
    double ret = apply_function(NormDiagonalMultNxN<data_t>(qubits), qubits );
    return ret;
  }
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
class NormMatrixMult2x2 : public GateFuncBase
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit;
  uint_t mask;
public:
  NormMatrixMult2x2(const cvector_t<double> &mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    mask = (1ull << qubit) - 1;
  }
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q0,q1;
    thrust::complex<double> r0,r1;
    double sum = 0.0;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    offsets = params.offsets_;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = pV[offsets[0]+i0];
    q1 = pV[offsets[1]+i0];

    r0 = m0 * q0 + m2 * q1;
    sum += r0.real()*r0.real() + r0.imag()*r0.imag();
    r1 = m1 * q0 + m3 * q1;
    sum += r1.real()*r1.real() + r1.imag()*r1.imag();
    return sum;
  }
  const char* Name(void)
  {
    return "Norm_mult2x2";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const
{
  reg_t qubits = {qubit};

  double ret = apply_function(NormMatrixMult2x2<data_t>(mat,qubit), qubits);

  return ret;
}


template <typename data_t>
class NormDiagonalMult2x2 : public GateFuncBase
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

  bool IsDiagonal(void)
  {
    return true;
  }
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,gid;
    thrust::complex<data_t>* pV;
    uint_t* offsets;
    thrust::complex<data_t> q;
    thrust::complex<double> m,r;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    gid = params.gid_;

    q = pV[i];
    if((((i+gid) >> qubit) & 1) == 0){
      m = m0;
    }
    else{
      m = m1;
    }

    r = m * q;

    return (r.real()*r.real() + r.imag()*r.imag());
  }
  const char* Name(void)
  {
    return "Norm_diagonal_mult2x2";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const
{
  reg_t qubits = {qubit};
  double ret = apply_function(NormDiagonalMult2x2<data_t>(mat,qubit), qubits);

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
  uint_t lcid,lid;
  int iPlace = GlobalToLocal(lcid,lid,outcome,m_maxChunkBits);

  UpdateReferencedValue();

  if(iPlace >= 0){
    std::complex<data_t> ret;
    ret = (std::complex<data_t>)m_Chunks[iPlace].GetState(lcid,lid,m_maxChunkBits);

    return std::real(ret)*std::real(ret) + std::imag(ret) * std::imag(ret);
  }
  else{
    return 0.0;
  }
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
class dot_func : public GateFuncBase
{
protected:
  uint64_t mask;
  uint64_t cmask;
public:
  dot_func(const reg_t &qubits,const reg_t &qubits_sorted,int i)
  {
    int k;
    int nq = qubits.size();

    mask = 0;
    cmask = 0;
    for(k=0;k<nq;k++){
      mask |= (1ull << qubits_sorted[k]);

      if(((i >> k) & 1) != 0){
        cmask |= (1ull << qubits[k]);
      }
    }
  }

  bool IsDiagonal(void)
  {
    return true;
  }
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,gid;
    thrust::complex<data_t> q;
    thrust::complex<data_t>* pV;
    double ret;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;
    gid = params.gid_;

    ret = 0.0;

    if(((i + gid) & mask) == cmask){
      q = pV[i];
      ret = q.real()*q.real() + q.imag()*q.imag();
    }
    return ret;
  }

  const char* Name(void)
  {
    return "dot";
  }
};

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities(const reg_t &qubits) const {

  const size_t N = qubits.size();
  const int_t DIM = 1 << N;

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits_) && (qubits == qubits_sorted))
    return probabilities();

  std::vector<double> probs((1ull << N), 0.);

  int i;
  for(i=0;i<DIM;i++){
    probs[i] = apply_function(dot_func<data_t>(qubits,qubits_sorted,i), qubits_sorted);
  }

  return probs;
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorThrust<data_t>::sample_measure(const std::vector<double> &rnds) const
{
  const int_t SHOTS = rnds.size();
  reg_t samples,localSamples;
  std::vector<double> placeSum;
  data_t* pVec;
  uint_t i;
  uint_t size;
  double sum,localSum,globalSum;
  int iPlace;

#ifdef AER_TIMING
  TimeStart(QS_GATE_MEASURE);
#endif

  UpdateReferencedValue();

  samples.assign(SHOTS, 0);

  localSamples.assign(SHOTS, 0);

  placeSum.assign(m_nPlaces+1, 0.0);

  //calculate sum of each place
#pragma omp parallel if (num_qubits_ > omp_threshold_ && m_nPlaces > 1) num_threads(m_nPlaces) private(pVec,size,iPlace)
  {
    int iDev;

    iPlace = omp_get_thread_num();
    iDev = m_Chunks[iPlace].DeviceID();
#ifdef AER_THRUST_CUDA
    if(iDev >= 0){
      cudaSetDevice(iDev);
    }
#endif

    pVec = (data_t*)m_Chunks[iPlace].ChunkPtr(0,m_maxChunkBits);
    size = m_Chunks[iPlace].Size() * 2;

    if(iPlace < m_nDevParallel){
      thrust::transform_inclusive_scan(thrust::device,pVec,pVec+size,pVec,thrust::square<double>(),thrust::plus<double>());
    }
    else{
      auto policy = (num_qubits_ > omp_threshold_ && omp_threads_ > 1)
          ? thrust::omp::par
          : thrust::seq;
      thrust::transform_inclusive_scan(policy,pVec,pVec+size,pVec,thrust::square<double>(),thrust::plus<double>());
    }
    placeSum[iPlace] = m_Chunks[iPlace].GetState(m_Chunks[iPlace].Size()-1).imag();
  }

  localSum = 0.0;
  for(iPlace=0;iPlace<m_nPlaces;iPlace++){
    sum = localSum;
    localSum += placeSum[iPlace];
    placeSum[iPlace] = sum;
  }
  placeSum[m_nPlaces] = localSum;

  globalSum = 0.0;
#ifdef AER_MPI
  if(m_nprocs > 1){
    pProcTotal = new double[m_nprocs];

    for(i=0;i<m_nprocs;i++){
      pProcTotal[i] = localSum;
    }

    MPI_Alltoall(pProcTotal,1,MPI_DOUBLE_PRECISION,pProcTotal,1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD);

    for(i=0;i<m_myrank;i++){
      globalSum += pProcTotal[i];
    }
    delete[] pProcTotal;
  }
#endif


  //now search for the position
#pragma omp parallel if (num_qubits_ > omp_threshold_ && m_nPlaces > 1) num_threads(m_nPlaces) private(pVec,iPlace,i,size)
  {
    thrust::host_vector<uint_t> vIdx(SHOTS);
    thrust::host_vector<double> vRnd(SHOTS);
    thrust::host_vector<uint_t> vSmp(SHOTS);
    uint_t nIn;
    int iDev;

    iPlace = omp_get_thread_num();
    iDev = m_Chunks[iPlace].DeviceID();
#ifdef AER_THRUST_CUDA
    if(iDev >= 0){
      cudaSetDevice(iDev);
    }
#endif

    pVec = (data_t*)m_Chunks[iPlace].ChunkPtr(0,m_maxChunkBits);
    size = m_Chunks[iPlace].Size() * 2;

    nIn = 0;
    for(i=0;i<SHOTS;i++){
      if(rnds[i] >= globalSum + placeSum[iPlace] && rnds[i] < globalSum + placeSum[iPlace+1]){
        vRnd[nIn] = rnds[i] - (globalSum + placeSum[iPlace]);
        vIdx[nIn] = i;
        nIn++;
      }
    }

    if(nIn > 0){
#ifdef AER_THRUST_CUDA
      if(iPlace < m_nDevParallel){
        thrust::device_vector<double> vRnd_dev(SHOTS);
        thrust::device_vector<uint_t> vSmp_dev(SHOTS);

        vRnd_dev = vRnd;
        thrust::lower_bound(thrust::device, pVec, pVec + size, vRnd_dev.begin(), vRnd_dev.begin() + nIn, vSmp_dev.begin());
        vSmp = vSmp_dev;
      }
      else{
#endif
        auto policy = (num_qubits_ > omp_threshold_ && omp_threads_ > 1)
          ? thrust::omp::par
          : thrust::seq;
        thrust::lower_bound(policy, pVec, pVec + size, vRnd.begin(), vRnd.begin() + nIn, vSmp.begin());
#ifdef AER_THRUST_CUDA
      }
#endif

      for(i=0;i<nIn;i++){
        localSamples[vIdx[i]] = m_Chunks[iPlace].GlobalIndex() + vSmp[i]/2;
      }
    }
  }

#ifdef QASM_MPI
  MPI_Allreduce(&localSamples[0],&samples[0],SHOTS,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
#else
  samples = localSamples;
#endif


#ifdef QASM_TIMING
  TimeEnd(QS_GATE_MEASURE);
#endif

#ifdef AER_DEBUG
  DebugMsg("sample_measure",samples);
#endif

  return samples;

}

#ifdef AER_TIMING

template <typename data_t>
void QubitVectorThrust<data_t>::TimeReset(void)
{
  int i;
  for(i=0;i<QS_NUM_GATES;i++){
    m_gateCounts[i] = 0;
    m_gateTime[i] = 0.0;
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::TimeStart(int i) const
{
  m_gateStartTime[i] = mysecond();
}

template <typename data_t>
void QubitVectorThrust<data_t>::TimeEnd(int i) const
{
  double t = mysecond();
  m_gateTime[i] += t - m_gateStartTime[i];
  m_gateCounts[i]++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::TimePrint(void)
{
  int i;
  double total;

  total = 0;
  for(i=0;i<QS_NUM_GATES;i++){
    total += m_gateTime[i];
  }

  printf("   ==================== Timing Summary =================== \n");
  if(m_gateCounts[QS_GATE_INIT] > 0)
    printf("  Initialization : %f \n",m_gateTime[QS_GATE_INIT]);
  if(m_gateCounts[QS_GATE_MULT] > 0)
    printf("    Matrix mult. : %f (%d)\n",m_gateTime[QS_GATE_MULT],m_gateCounts[QS_GATE_MULT]);
  if(m_gateCounts[QS_GATE_CX] > 0)
    printf("    CX           : %f (%d)\n",m_gateTime[QS_GATE_CX],m_gateCounts[QS_GATE_CX]);
  if(m_gateCounts[QS_GATE_DIAG] > 0)
    printf("    Diagonal     : %f (%d)\n",m_gateTime[QS_GATE_DIAG],m_gateCounts[QS_GATE_DIAG]);
  if(m_gateCounts[QS_GATE_MEASURE] > 0)
    printf("    Measure     : %f  (%d)\n",m_gateTime[QS_GATE_MEASURE],m_gateCounts[QS_GATE_MEASURE]);
  printf("    Total Kernel time : %f sec\n",total);
  printf("   ======================================================= \n");

}
#endif


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
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const int qubit) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : (%d) \n",debug_count,str,qubit);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s \n",debug_count,str);
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::complex<double> c) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : %e, %e \n",debug_count,str,std::real(c),imag(c));
  }
  debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const double d) const
{
  if(debug_fp != NULL){
    fprintf(debug_fp," [%d] %s : %e \n",debug_count,str,d);
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
  }
  debug_count++;

}


template <typename data_t>
void QubitVectorThrust<data_t>::DebugDump(void) const
{
  if(debug_fp != NULL){
    if(num_qubits_ < 10){
      char bin[64];
      int iPlace;
      uint_t i,j,ic,nc;
      uint_t pos = 0;
      uint_t csize = 1ull << m_maxChunkBits;
      cvector_t<data_t> tmp(csize);

      UpdateReferencedValue();

      for(iPlace=0;iPlace<m_nPlaces;iPlace++){
        nc = m_Chunks[iPlace].NumChunks(m_maxChunkBits);
        for(ic=0;ic<nc;ic++){
          m_Chunks[iPlace].CopyOut((thrust::complex<data_t>*)&tmp[0],0,ic,m_maxChunkBits);

          for(i=0;i<csize;i++){
            for(j=0;j<num_qubits_;j++){
              bin[num_qubits_-j-1] = '0' + (char)(((pos+i) >> j) & 1);
            }
            bin[num_qubits_] = 0;
            fprintf(debug_fp,"   %s | %e, %e\n",bin,std::real(tmp[i]),imag(tmp[i]));
          }

          pos += csize;
        }
      }
    }
  }
}


#endif

/*******************************************************************************
 *
 * EXPECTATION VALUES
 *
 ******************************************************************************/

template <typename data_t>
class expval_pauli_func : public GateFuncBase
{
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  thrust::complex<data_t> phase_;
public:
  expval_pauli_func(uint_t x,uint_t z,thrust::complex<data_t> p)
  {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;
  }

  bool IsDiagonal(void)
  {
    return true;
  }
  bool Reduction(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i;
    thrust::complex<data_t>* pV;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    double ret = 0.0;
    struct GateParams<data_t> params;

    i = ExtractIndexFromTuple(iter);
    params = ExtractParamsFromTuple(iter);
    pV = params.buf_;

    q0 = pV[i];
    q1 = pV[i ^ x_mask_];
    q1 = q1 * phase_;
    ret = q0.real()*q1.real() + q0.imag()*q1.imag();

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
  const char* Name(void)
  {
    return "expval_pauli";
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
  for (size_t i = 0; i < N; ++i) {
    const auto bit = BITS[qubits[i]];
    switch (pauli[N - 1 - i]) {
      case 'I':
        break;
      case 'X': {
        x_mask += bit;
        break;
      }
      case 'Z': {
        z_mask += bit;
        break;
      }
      case 'Y': {
        x_mask += bit;
        z_mask += bit;
        num_y++;
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
  return apply_function(expval_pauli_func<data_t>(x_mask, z_mask, phase),qubits);
}

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



