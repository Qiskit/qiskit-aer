
    set(QSIM_PAR_DEFS "${QSIM_PAR_DEFS} -DQSIM_CUDA")
	if(AER_DEBUG)
		set(QSIM_PAR_DEFS "${QSIM_PAR_DEFS} -DQSIM_DEBUG")
	endif(AER_DEBUG)

    set(CMAKE_CXX_FLAGS "-O3 -fopenmp ${QSIM_PAR_DEFS} -I/usr/local/cuda/include")
#    set(CUDA_NVCC_FLAGS -gencode; arch=compute_70,code=sm_70; -use_fast_math; -ccbin /usr/bin/g++;)
    set(CUDA_NVCC_FLAGS -use_fast_math; -ccbin /usr/bin/g++; -Xcompiler -fopenmp;)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 6.0 7.0)
    LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

add_library(qsim_par
  STATIC
    QSUnitManager.cpp
    QSUnitStorage.cpp
    QSUnitStorageHost.cpp
    QSUnitStorageFile.cpp
    QSUnitStorageGPU.cpp
    QSGate_CX.cpp
    QSGate_X.cpp
    QSGate_Y.cpp
    QSGate_Z.cpp
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
    QSGate_Z_CUDA.cu
    QSGate_Dot_CUDA.cu
    QSGate_MatMult_CUDA.cu
    QSGate_MultiShot_CUDA.cu
    QSGate_DiagMult_CUDA.cu
  )

target_link_libraries(qsim_par_cuda 
     qsim_par)


#target_include_directories(qsim_par
#  PUBLIC ${PROJECT_SOURCE_DIR}/src/simulators/statevector/qsim_par
#  )




