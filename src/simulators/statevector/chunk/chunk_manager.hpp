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


#ifndef _qv_chunk_manager_hpp_
#define _qv_chunk_manager_hpp_

#include "simulators/statevector/chunk/chunk.hpp"

namespace QV {

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

#ifdef AER_THRUST_CPU
  num_devices_ = 0;
  num_places_ = 1;
#else

#ifdef AER_THRUST_CUDA
  cudaGetDeviceCount(&num_devices_);
  num_places_ = num_devices_;
#else
  num_devices_ = 1;
  num_places_ = 1;
#endif

#endif

  chunks_.resize(num_places_*2 + 1);

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
#ifdef AER_THRUST_CPU
    multi_gpu = false;
    num_buffers = 0;
    num_places_ = 1;
#else
    multi_gpu = true;
    num_buffers = 0;
    num_places_ = num_devices_;
#endif
    omp_set_nested(1);
  }
  else{
    if(chunk_bits == nqubits){    //single chunk
      num_buffers = 0;
      multi_gpu = false;
      num_places_ = 1;
    }
    else{   //multiple-chunks
      num_buffers = AER_MAX_BUFFERS;

#ifdef AER_THRUST_CUDA
      num_places_ = num_devices_;
      if(!multi_gpu){
        size_t freeMem,totalMem;
        cudaSetDevice(0);
        cudaMemGetInfo(&freeMem,&totalMem);
        if(freeMem > ( ((uint_t)sizeof(thrust::complex<data_t>) * (nchunks + num_buffers + AER_DUMMY_BUFFERS)) << chunk_bits_)){
          num_places_ = 1;
        }
      }
#else
      num_places_ = 1;
#endif
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

    num_checkpoint = nc;
#ifdef AER_THRUST_CPU
    if(nid > 1){
      //allocate as host mode for serial execution
      chunks_[iDev] = new HostChunkContainer<data_t>;
    }
    else{
      chunks_[iDev] = new DeviceChunkContainer<data_t>;
    }
#else
    chunks_[iDev] = new DeviceChunkContainer<data_t>;
#endif

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

#pragma omp barrier

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



}

//------------------------------------------------------------------------------
#endif // end module
