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

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>


namespace AER {
namespace QV {

//============================================================================
// chunk manager class
// this is static class, there is only 1 manager class
//============================================================================
template <typename data_t>
class ChunkManager 
{
protected:
  std::vector<std::shared_ptr<ChunkContainer<data_t>>> chunks_;         //chunk containers for each device and host

  int num_devices_;            //number of devices
  int num_places_;             //number of places (devices + host)

  int chunk_bits_;             //number of qubits of chunk
  int num_qubits_;             //number of global qubits

  uint_t num_chunks_;          //number of chunks on this process

  int i_dev_map_;              //device index chunk to be mapped
  int idev_buffer_map_;        //device index buffer to be mapped

  int iplace_host_;            //chunk container for host memory
  bool multi_shots_;
public:
  ChunkManager();

  ~ChunkManager();

  std::shared_ptr<ChunkContainer<data_t>> container(uint_t i)
  {
    if(i < chunks_.size())
      return chunks_[i];
    return nullptr;
  }
  uint_t num_containers(void)
  {
    return chunks_.size();
  }

  uint_t Allocate(int chunk_bits,int nqubits,uint_t nchunks,int matrix_bit);
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

  bool MapChunk(Chunk<data_t>& chunk,int iplace = -1);
  bool MapBufferChunk(Chunk<data_t>& out,int idev);
  bool MapBufferChunkOnHost(Chunk<data_t>& out);

  void UnmapChunk(Chunk<data_t>& chunk);
  void UnmapBufferChunk(Chunk<data_t>& buffer);
};

template <typename data_t>
ChunkManager<data_t>::ChunkManager()
{
  int i,j;
  num_places_ = 1;
  chunk_bits_ = 0;
  num_chunks_ = 0;
  num_qubits_ = 0;
  multi_shots_ = false;

  idev_buffer_map_ = 0;

#ifdef AER_THRUST_CPU
  num_devices_ = 0;
  num_places_ = 1;
#else

#ifdef AER_THRUST_CUDA
  if(cudaGetDeviceCount(&num_devices_) == cudaSuccess){
    num_places_ = num_devices_;
  }
  else{
    cudaGetLastError();
    num_devices_ = 1;
    num_places_ = 1;
  }
#else
  num_devices_ = 1;
  num_places_ = 1;
#endif

#endif

  iplace_host_ = num_places_ ;

#ifdef AER_DEBUG
  //spdlog for Thrust implementation debugging
  auto logger = spdlog::get("qv_thrust_logger");
  if(!logger){  //for the first call of this process
    char filename[512];
    sprintf(filename,"logs/qubitvector_thrust_%d.txt",getpid());
    auto file_logger = spdlog::basic_logger_mt<spdlog::async_factory>("qv_thrust_logger", filename);
    file_logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(file_logger);
  }
#endif
}

template <typename data_t>
ChunkManager<data_t>::~ChunkManager()
{
  Free();
}

template <typename data_t>
uint_t ChunkManager<data_t>::Allocate(int chunk_bits,int nqubits,uint_t nchunks,int matrix_bit)
{
  uint_t num_buffers;
  int iDev;
  uint_t is,ie,nc;
  int i;
  char* str;
  bool multi_gpu = false;
  bool hybrid = false;

  //--- for test
  str = getenv("AER_MULTI_GPU");
  if(str){
    multi_gpu = true;
    num_places_ = num_devices_;
  }
  str = getenv("AER_HYBRID");
  if(str){
    hybrid = true;
  }
  //---

  if(num_qubits_ != nqubits || chunk_bits_ != chunk_bits || nchunks > num_chunks_){
    //free previous allocation
    Free();

    num_qubits_ = nqubits;
    chunk_bits_ = chunk_bits;

    num_chunks_ = 0;

    if(chunk_bits == nqubits){
      if(nchunks > 1){  //multi-shot parallelization
        //accumulate number of chunks
        num_chunks_ = nchunks;

        num_buffers = 0;
        multi_shots_ = true;

#ifdef AER_THRUST_CPU
        multi_gpu = false;
        num_places_ = 1;
#else
        multi_gpu = true;
        num_places_ = num_devices_;
#endif
      }
      else{    //single chunk
        num_buffers = 0;
        multi_gpu = false;
        num_places_ = 1;
        num_chunks_ = nchunks;
        multi_shots_ = false;
      }
    }
    else{   //multiple-chunk parallelization
      multi_shots_ = false;

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
      num_chunks_ = nchunks;
    }
    if(num_chunks_ < num_places_){
      num_places_ = num_chunks_;
    }

    nchunks = num_chunks_;
    num_chunks_ = 0;
    for(iDev=0;iDev<num_places_;iDev++){
      is = nchunks * (uint_t)iDev / (uint_t)num_places_;
      ie = nchunks * (uint_t)(iDev + 1) / (uint_t)num_places_;
      nc = ie - is;
      if(hybrid){
        nc /= 2;
      }

      chunks_.push_back(std::make_shared<DeviceChunkContainer<data_t>>());
      num_chunks_ += chunks_[iDev]->Allocate(iDev,chunk_bits,nqubits,nc,num_buffers,multi_shots_,matrix_bit);
    }
    if(num_chunks_ < nchunks){
      //rest of chunks are stored on host
      chunks_.push_back(std::make_shared<HostChunkContainer<data_t>>());
      chunks_[num_places_]->Allocate(-1,chunk_bits,nqubits,nchunks-num_chunks_,num_buffers,multi_shots_,matrix_bit);
      num_places_ += 1;
      num_chunks_ = nchunks;
    }

#ifdef AER_DISABLE_GDR
    //additional host buffer
    iplace_host_ = chunks_.size();
    chunks_.push_back(std::make_shared<HostChunkContainer<data_t>>());
    chunks_[iplace_host_]->Allocate(-1,chunk_bits,nqubits,0,AER_MAX_BUFFERS,multi_shots_,matrix_bit);
#endif
  }
  else{
    for(iDev=0;iDev<chunks_.size();iDev++){
      chunks_[iDev]->unmap_all();
    }
  }

  return num_chunks_;
}

template <typename data_t>
void ChunkManager<data_t>::Free(void)
{
  int i;

  for(i=0;i<chunks_.size();i++){
    chunks_[i]->Deallocate();
    chunks_[i].reset();
  }
  chunks_.clear();

  chunk_bits_ = 0;
  num_qubits_ = 0;
  num_chunks_ = 0;

  idev_buffer_map_ = 0;
}

template <typename data_t>
bool ChunkManager<data_t>::MapChunk(Chunk<data_t>& chunk,int iplace)
{
  int i;

  for(i=0;i<num_places_;i++){
    if(chunks_[(iplace + i) % num_places_]->MapChunk(chunk)){
      chunk.set_place((iplace + i) % num_places_);
      break;
    }
  }
  return chunk.is_mapped();
}

template <typename data_t>
bool ChunkManager<data_t>::MapBufferChunk(Chunk<data_t>& out,int idev)
{
  if(idev < 0){
    int i;
    for(i=0;i<num_devices_;i++){
      if(chunks_[i]->MapBufferChunk(out))
        break;
    }
  }
  else{
    chunks_[(idev % num_devices_)]->MapBufferChunk(out);
  }
  return out.is_mapped();
}

template <typename data_t>
bool ChunkManager<data_t>::MapBufferChunkOnHost(Chunk<data_t>& out)
{
  return chunks_[iplace_host_]->MapBufferChunk(out);
}

template <typename data_t>
void ChunkManager<data_t>::UnmapChunk(Chunk<data_t>& chunk)
{
  int iPlace = chunk.place();

  chunks_[iPlace]->UnmapChunk(chunk);
}


template <typename data_t>
void ChunkManager<data_t>::UnmapBufferChunk(Chunk<data_t>& buffer)
{
  chunks_[buffer.place()]->UnmapBuffer(buffer);
}


//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
