
    set(QSIM_PAR_DEFS "${QSIM_PAR_DEFS} -DQSIM_COL_MAJOR")

if(AER_CUDA)
find_package(CUDA REQUIRED)

    set(QSIM_PAR_DEFS "${QSIM_PAR_DEFS} -DQSIM_CUDA")
    set(CMAKE_CXX_FLAGS "-O3 -fopenmp ${QSIM_PAR_DEFS} -I/usr/local/cuda/include")
    set(CUDA_NVCC_FLAGS -gencode; arch=compute_70,code=sm_70; -use_fast_math; -ccbin; /usr/bin/g++;)

add_library(qsim_par
  STATIC
    QSUnitManager.cpp
    QSUnitStorage.cpp
    QSUnitStorageHost.cpp
    QSUnitStorageGPU.cpp
    QSGate_CX.cpp
    QSGate_X.cpp
    QSGate_Y.cpp
    QSGate_Dot.cpp
    QSGate_MatMult.cpp
    QSGate_MultiShot.cpp
    QSGate_DiagMult.cpp
  )

cuda_add_library(qsim_par_cuda
  STATIC
    QSGate_CX_CUDA.cu
    QSGate_X_CUDA.cu
    QSGate_Y_CUDA.cu
    QSGate_Dot_CUDA.cu
    QSGate_MatMult_CUDA.cu
    QSGate_MultiShot_CUDA.cu
    QSGate_DiagMult_CUDA.cu
  )

target_link_libraries(qsim_par_cuda 
     qsim_par)


else(AER_CUDA)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${QSIM_PAR_DEFS}")

add_library(qsim_par
  STATIC
    QSUnitManager.cpp
    QSUnitStorage.cpp
    QSUnitStorageHost.cpp
    QSGate_CX.cpp
    QSGate_X.cpp
    QSGate_Y.cpp
    QSGate_Dot.cpp
    QSGate_MatMult.cpp
    QSGate_MultiShot.cpp
  )


endif(AER_CUDA)

if(AER_MPI)
target_include_directories(qsim_par
  PUBLIC ${PROJECT_SOURCE_DIR}/src/simulators/statevector/qsim_par
  ${MPI_CXX_INCLUDE_PATH}
  )
else(AER_MPI)
target_include_directories(qsim_par
  PUBLIC ${PROJECT_SOURCE_DIR}/src/simulators/statevector/qsim_par
  )
endif(AER_MPI)




