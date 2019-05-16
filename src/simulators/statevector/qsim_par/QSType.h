/*------------------------------------------------------------------------------------
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

	IBM Q Simulator

	type definitions

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_TYPES_H_
#define _IBM_Q_SIMULATOR_TYPES_H_

#include <stdio.h>
#include <stdint.h>


#ifdef QSIM_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#endif


typedef double QSDouble;
typedef double _Complex QSDoubleComplex;


#ifdef QSIM_DOUBLE

//data type for storage
typedef double QSReal;
typedef double _Complex QSComplex;

//data type for computation
typedef double QSRealC;
typedef double _Complex QSComplexC;

#ifdef QSIM_CUDA

//data type on GPU
typedef double QSRealDev;
typedef double2 QSVec2;
//data type for computation on GPU
typedef double2 QSVec2C;

#ifdef QSIM_MPI
#define QS_MPI_REAL_TYPE			MPI_DOUBLE_PRECISION
#endif

#endif

#endif	//QSIM_DOUBLE

#ifdef QSIM_FLOAT

typedef float QSReal;
typedef float _Complex QSComplex;

typedef double QSRealC;
typedef double _Complex QSComplexC;


#ifdef QSIM_CUDA

typedef float QSRealDev;
typedef float2 QSVec2;
typedef double2 QSVec2C;

#endif


#ifdef QSIM_MPI
#define QS_MPI_REAL_TYPE			MPI_FLOAT
#endif


#endif	//QSIM_FLOAT

#ifdef QSIM_HALF

typedef uint16_t QSReal;
typedef uint32_t QSComplex;

typedef float QSRealC;
typedef float _Complex QSComplexC;


#ifdef QSIM_CUDA

typedef half QSRealDev;
typedef half2 QSVec2;
typedef float2 QSVec2C;

#endif


#ifdef QSIM_MPI
#define QS_MPI_REAL_TYPE			MPI_SHORT
#endif


#endif	//QSIM_HALF





typedef uint64_t 	QSUint;
typedef int			QSInt;
typedef unsigned char QSByte;


#endif	//_IBM_Q_SIMULATOR_TYPES_H_

