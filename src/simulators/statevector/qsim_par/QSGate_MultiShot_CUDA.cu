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

	Multi-shot measure

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_MultiShot.h"
#include "QSUnitStorageGPU.h"

#include "reduction.h"


__global__ void QSGate_MultiShot_Dot_deviceReduceSum(double *pT,uint64_t n)
{
	uint64_t i;
	double sum;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n)
		sum = pT[i];
	else
		sum = 0.0;

	sum = blockReduceSum(sum);

	if(threadIdx.x==0)
		pT[blockIdx.x]=sum;
}

__global__ void QSGate_MultiShot_Dot_cuda_kernel(double* pSum,QSVec2* qreg,int bits_per_row,uint64_t rows)
{
	uint64_t add,i,ir;
	double sum = 0.0;
	QSVec2 q0;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	add = blockDim.x << bits_per_row;

	for(ir=0;ir<rows;ir++){
		q0 = qreg[i];
		sum += (double)q0.x*(double)q0.x + (double)q0.y*(double)q0.y;
		i += add;
	}

	sum = blockReduceSum(sum);

	if(threadIdx.x == 0){
		pSum[blockIdx.x] = sum;
	}
}


void QSGate_MultiShot::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint nt,ng,ngg,nrows,iu,offset;
	cudaStream_t strm;
	double* pDot;
	int qubit = qubits[0];
	double dot[16];

	pUnit->SetDevice();

	strm = (cudaStream_t)pUnit->GetStreamPipe();

	if(m_Key < 0.0){	//reduction mode
		pDot = pUnit->GetNormPointer();

		if(nTrans > 0){
			pUnit->SynchronizeInput();
		}

		CUDA_FitThreads(nt,ng,1ull << pUnit->UnitBits());
		nrows = 1;
		if(ng > NORM_BUF_SIZE){
			nrows = (ng + NORM_BUF_SIZE - 1) >> NORM_BUF_BITS;
			ng = NORM_BUF_SIZE;
		}

		QSGate_MultiShot_Dot_cuda_kernel<<<ng,nt,sizeof(double)*32,strm>>>(pDot,(QSVec2*)(ppBuf[0]),NORM_BUF_BITS,nrows);

		while(ng > 1){
			ngg = ng;
			CUDA_FitThreads(nt,ng,ngg);
			if(nt < 32){
				nt = 32;
			}
			QSGate_MultiShot_Dot_deviceReduceSum<<<ng,nt,sizeof(double)*32,strm>>>(pDot,ngg);
		}
		cudaMemcpyAsync(dot,pDot,sizeof(double),cudaMemcpyDeviceToHost,strm);

		cudaStreamSynchronize(strm);

		m_pDotPerUnits[pGuid[0] - pUnit->GetGlobalUnitIndexBase()] = dot[0];
		m_Total += dot[0];
	}
	else{	//search mode
		//currently search on host
	}
}



