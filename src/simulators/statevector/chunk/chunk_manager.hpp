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

#ifndef _qv_chunk_manager_hpp_
#define _qv_chunk_manager_hpp_

#include "simulators/statevector/chunk/chunk.hpp"

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace AER {
namespace QV {
namespace Chunk {

//============================================================================
// chunk manager class
// this is static class, there is only 1 manager class
//============================================================================
template <typename data_t>
class ChunkManager {
protected:
  std::vector<std::shared_ptr<ChunkContainer<data_t>>>
      chunks_; // chunk containers for each device and host

  uint_t num_devices_; // number of devices
  uint_t num_places_;  // number of places (devices + host)

  uint_t chunk_bits_; // number of qubits of chunk
  uint_t num_qubits_; // number of global qubits

  uint_t num_chunks_;  // number of chunks on this process
  uint_t chunk_index_; // global chunk index for the first chunk

  int i_dev_map_;       // device index chunk to be mapped
  int idev_buffer_map_; // device index buffer to be mapped

  int iplace_host_; // chunk container for host memory
  bool multi_shots_;

  bool density_matrix_;

  bool enable_cuStatevec_;

  int num_threads_per_group_;
  uint_t num_creg_bits_ = 0;

  bool chunk_distribution_enable_ = true; // enable distribution over GPUs
  reg_t target_gpus_;

public:
  ChunkManager();

  ~ChunkManager();

  std::shared_ptr<ChunkContainer<data_t>> container(uint_t i) {
    if (i < chunks_.size())
      return chunks_[i];
    return nullptr;
  }
  uint_t num_containers(void) { return chunks_.size(); }

  uint_t Allocate(int chunk_bits, int nqubits, uint_t nchunks,
                  uint_t chunk_index, int matrix_bit, int max_shots,
                  bool density_mat, reg_t &gpus, bool enable_cuStatevec);
  void Free(void);

  int num_devices(void) { return num_devices_; }
  int num_places(void) { return num_places_; }
  int place_host(void) { return iplace_host_; }
  uint_t num_chunks(void) { return num_chunks_; }
  int chunk_bits(void) { return chunk_bits_; }
  int num_qubits(void) { return num_qubits_; }

  void set_num_threads_per_group(int n) { num_threads_per_group_ = n; }
  void set_num_creg_bits(uint_t bits) { num_creg_bits_ = bits; }

  bool MapChunk(Chunk<data_t> &chunk, int iplace = -1);
  bool MapBufferChunk(Chunk<data_t> &out, int idev);
  bool MapBufferChunkOnHost(Chunk<data_t> &out);

  void UnmapChunk(Chunk<data_t> &chunk);
  void UnmapBufferChunk(Chunk<data_t> &buffer);

  // execute a kernel on all the chunks
  template <typename Function>
  void execute_on_device(Function func,
                         const std::vector<std::complex<double>> &mat,
                         const std::vector<uint_t> &prm);

  void enable_chunk_distribution(bool flg) { chunk_distribution_enable_ = flg; }
};

template <typename data_t>
ChunkManager<data_t>::ChunkManager() {
  num_places_ = 1;
  chunk_bits_ = 0;
  num_chunks_ = 0;
  chunk_index_ = 0;
  num_qubits_ = 0;
  multi_shots_ = false;
  num_threads_per_group_ = 1;

  density_matrix_ = false;

  enable_cuStatevec_ = false;

  idev_buffer_map_ = 0;

#ifdef AER_THRUST_CPU
  num_devices_ = 0;
  num_places_ = 1;
#else

#ifdef AER_THRUST_GPU
  int ndev;
  if (cudaGetDeviceCount(&ndev) == cudaSuccess) {
    num_devices_ = ndev;
    num_places_ = num_devices_;
  } else {
    cudaGetLastError();
    num_devices_ = 1;
    num_places_ = 1;
  }
#else
  num_devices_ = 1;
  num_places_ = 1;
#endif

#endif

  iplace_host_ = num_places_;

#ifdef AER_DEBUG
  // spdlog for Thrust implementation debugging
  auto logger = spdlog::get("qv_thrust_logger");
  if (!logger) { // for the first call of this process
    char filename[512];
    sprintf(filename, "logs/qubitvector_thrust_%d.txt", getpid());
    auto file_logger = spdlog::basic_logger_mt<spdlog::async_factory>(
        "qv_thrust_logger", filename);
    file_logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(file_logger);
  }
#endif
}

template <typename data_t>
ChunkManager<data_t>::~ChunkManager() {
  Free();
}

template <typename data_t>
uint_t ChunkManager<data_t>::Allocate(int chunk_bits, int nqubits,
                                      uint_t nchunks, uint_t chunk_index,
                                      int matrix_bit, int max_shots,
                                      bool density_mat, reg_t &gpus,
                                      bool enable_cuStatevec) {
  uint_t num_buffers;
  uint_t iDev;
  uint_t is, ie, nc;
  uint_t i;
  char *str;

  bool hybrid = false;
#ifdef AER_THRUST_GPU
  bool multi_gpu = false;
  //--- for test
  str = getenv("AER_MULTI_GPU");
  if (str) {
    multi_gpu = true;
    num_places_ = num_devices_;
  }
#endif
  str = getenv("AER_HYBRID");
  if (str) {
    hybrid = true;
  }
  //---
  density_matrix_ = density_mat;

  enable_cuStatevec_ = enable_cuStatevec;
  target_gpus_ = gpus;
  if (target_gpus_.size() > 0) {
    num_devices_ = target_gpus_.size();
#ifdef AER_THRUST_GPU
    if (num_devices_ > 1)
      multi_gpu = true;
#endif
  } else {
    target_gpus_.resize(num_devices_);
    for (iDev = 0; iDev < num_devices_; iDev++) {
      target_gpus_[iDev] = iDev;
    }
  }

  chunk_index_ = chunk_index;

  if (num_qubits_ != (uint_t)nqubits || chunk_bits_ != (uint_t)chunk_bits ||
      nchunks > num_chunks_) {
    // free previous allocation
    Free();

    num_qubits_ = nqubits;
    chunk_bits_ = chunk_bits;

    num_chunks_ = 0;

    uint_t idev_start = 0;

    if (chunk_bits == nqubits) {
      if (nchunks > 1) { // multi-shot parallelization
        // accumulate number of chunks
        num_chunks_ = nchunks;

        num_buffers = 0;
        multi_shots_ = true;

#ifdef AER_THRUST_CPU
        num_places_ = 1;
#else
        if (chunk_distribution_enable_) {
          multi_gpu = true;
          num_places_ = num_devices_;
          if (num_threads_per_group_ > 1)
            num_places_ *= num_threads_per_group_;

          if (num_places_ > omp_get_max_threads()) {
            num_places_ = num_devices_;
          }
        } else {
          multi_gpu = false;
          num_places_ = 1;
          idev_start = 0;

          // define device to be allocated
          if (num_devices_ > 1) {
            size_t freeMem, totalMem, maxMem;
            cudaSetDevice(0);
            cudaMemGetInfo(&freeMem, &totalMem);
            maxMem = freeMem;
            for (i = 1; i < num_devices_; i++) {
              cudaSetDevice(i);
              cudaMemGetInfo(&freeMem, &totalMem);
              if (freeMem > maxMem) {
                maxMem = freeMem;
                idev_start = i;
              }
            }
          }
        }
#endif
      } else { // single chunk
        num_buffers = 0;
#ifdef AER_THRUST_GPU
        multi_gpu = false;
#endif
        num_places_ = 1;
        num_chunks_ = nchunks;
        multi_shots_ = false;

        // distribute chunk on multiple GPUs when OMP parallel shots are used
        if (num_devices_ > 0)
          idev_start = omp_get_thread_num() % num_devices_;
      }
    } else { // multiple-chunk parallelization
      multi_shots_ = false;

      num_buffers = AER_MAX_BUFFERS;

#ifdef AER_THRUST_GPU
      num_places_ = num_devices_;
      if (num_threads_per_group_ > 1)
        num_places_ *= num_threads_per_group_;

      if (num_places_ > omp_get_max_threads()) {
        num_places_ = num_devices_;
      }

      if (!multi_gpu) {
        size_t freeMem, totalMem;
        cudaSetDevice(target_gpus_[0]);
        cudaMemGetInfo(&freeMem, &totalMem);
        if (freeMem > (((uint_t)sizeof(thrust::complex<data_t>) *
                        (nchunks + num_buffers + AER_DUMMY_BUFFERS))
                       << chunk_bits_)) {
          num_places_ = 1;
        }
      }

#else
      num_places_ = 1;
#endif
      num_chunks_ = nchunks;
    }
    if (num_chunks_ < num_places_) {
      num_places_ = num_chunks_;
    }

    // allocate chunk container before parallel loop using push_back to store
    // shared pointer
    for (i = 0; i < num_places_; i++) {
#ifdef AER_CUSTATEVEC
      if (enable_cuStatevec_) { // allocate cuStateVec chunk
        chunks_.push_back(std::make_shared<cuStateVecChunkContainer<data_t>>());
        continue;
      } else {
#endif
        chunks_.push_back(std::make_shared<DeviceChunkContainer<data_t>>());
#ifdef AER_CUSTATEVEC
      }
#endif
    }

    uint_t chunks_allocated = 0;
    // #pragma omp parallel for if(num_places_ == num_devices_)
    // private(is,ie,nc) reduction(+:chunks_allocated)
    for (iDev = 0; iDev < num_places_; iDev++) {
      is = num_chunks_ * (uint_t)iDev / (uint_t)num_places_;
      ie = num_chunks_ * (uint_t)(iDev + 1) / (uint_t)num_places_;
      nc = ie - is;
      if (hybrid) {
        nc /= 2;
        if (nc < 1)
          nc = 1;
      }
      chunks_[iDev]->set_chunk_index(
          chunk_index_ +
          chunks_allocated); // set first chunk index for the container
      chunks_[iDev]->set_num_creg_bits(num_creg_bits_);
      if (num_devices_ > 0) {
        int id = target_gpus_[(iDev + idev_start) % num_devices_];
        chunks_allocated += chunks_[iDev]->Allocate(
            id, chunk_bits, nqubits, nc, num_buffers, multi_shots_, matrix_bit,
            max_shots, density_matrix_);
      } else {
        chunks_allocated += chunks_[iDev]->Allocate(
            iDev, chunk_bits, nqubits, nc, num_buffers, multi_shots_,
            matrix_bit, max_shots, density_matrix_);
      }
    }
    if (chunks_allocated < num_chunks_) {
      uint_t nplaces_add = num_places_;
      if ((num_chunks_ - chunks_allocated) < nplaces_add)
        nplaces_add = (num_chunks_ - chunks_allocated);
      // rest of chunks are stored on host
      for (iDev = 0; iDev < nplaces_add; iDev++) {
        is = (num_chunks_ - chunks_allocated) * (uint_t)iDev /
             (uint_t)nplaces_add;
        ie = (num_chunks_ - chunks_allocated) * (uint_t)(iDev + 1) /
             (uint_t)nplaces_add;
        nc = ie - is;
        if (nc > 0) {
          chunks_.push_back(std::make_shared<HostChunkContainer<data_t>>());
          chunks_[chunks_.size() - 1]->set_chunk_index(
              chunk_index_ + chunks_allocated +
              is); // set first chunk index for the container
          chunks_[chunks_.size() - 1]->Allocate(
              -1, chunk_bits, nqubits, nc, num_buffers, multi_shots_,
              matrix_bit, max_shots, density_matrix_);
        }
      }
      num_places_ += nplaces_add;
    }

    for (iDev = 0; iDev < num_places_; iDev++)
      chunks_[iDev]->set_place(iDev, num_places_);

#ifdef AER_DISABLE_GDR
    // additional host buffer
    iplace_host_ = chunks_.size();
    chunks_.push_back(std::make_shared<HostChunkContainer<data_t>>());
    chunks_[iplace_host_]->Allocate(-1, chunk_bits, nqubits, 0, AER_MAX_BUFFERS,
                                    multi_shots_, matrix_bit, max_shots,
                                    density_matrix_);
#endif
  } else {
    for (iDev = 0; iDev < chunks_.size(); iDev++) {
      chunks_[iDev]->unmap_all();
    }
  }

  return num_chunks_;
}

template <typename data_t>
void ChunkManager<data_t>::Free(void) {
  uint_t i;

  for (i = 0; i < chunks_.size(); i++) {
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
bool ChunkManager<data_t>::MapChunk(Chunk<data_t> &chunk, int iplace) {
  uint_t i;

  for (i = 0; i < num_places_; i++) {
    if (chunks_[(iplace + i) % num_places_]->MapChunk(chunk)) {
      chunk.set_place((iplace + i) % num_places_);
      break;
    }
  }
  return chunk.is_mapped();
}

template <typename data_t>
bool ChunkManager<data_t>::MapBufferChunk(Chunk<data_t> &out, int idev) {
  if (idev < 0) {
    uint_t i;
    for (i = 0; i < num_devices_; i++) {
      if (chunks_[i]->MapBufferChunk(out))
        break;
    }
  } else {
    chunks_[(idev % num_devices_)]->MapBufferChunk(out);
  }
  return out.is_mapped();
}

template <typename data_t>
bool ChunkManager<data_t>::MapBufferChunkOnHost(Chunk<data_t> &out) {
  return chunks_[iplace_host_]->MapBufferChunk(out);
}

template <typename data_t>
void ChunkManager<data_t>::UnmapChunk(Chunk<data_t> &chunk) {
  int iPlace = chunk.place();

  chunks_[iPlace]->UnmapChunk(chunk);
}

template <typename data_t>
void ChunkManager<data_t>::UnmapBufferChunk(Chunk<data_t> &buffer) {
  chunks_[buffer.place()]->UnmapBuffer(buffer);
}

template <typename data_t>
template <typename Function>
void ChunkManager<data_t>::execute_on_device(
    Function func, const std::vector<std::complex<double>> &mat,
    const std::vector<uint_t> &prm) {
#pragma omp parallel num_threads(num_places_)
  {
    int_t place = omp_get_thread_num();

    // store matrix and params if exist
    if (mat.size() > 0)
      chunks_[place]->StoreMatrix(mat, 0);
    if (prm.size() > 0)
      chunks_[place]->StoreUintParams(prm, 0);

    // execute a kernel
    chunks_[place]->Execute(func, 0);
  }
}

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
