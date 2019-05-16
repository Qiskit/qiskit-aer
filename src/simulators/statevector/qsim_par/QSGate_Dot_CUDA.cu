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

	U1 gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_Dot.h"
#include "QSUnitStorageGPU.h"

__inline__ __device__ double QSGate_Dot_warpReduceSum(double val)
{
	int i;
	for(i=16;i>0;i/=2){
		val += __shfl_xor_sync(0xffffffff,val,i,32);
	}
	return val;
}

__inline__ __device__ double QSGate_Dot_blockReduceSum(double val) 
{
	__shared__ double buf[32];
	int lid = threadIdx.x & 0x1f;
	int wid = threadIdx.x >> 5;

	val = QSGate_Dot_warpReduceSum(val);

	if(lid == 0)
		buf[wid] = val;

	__syncthreads();

	val = (threadIdx.x < (blockDim.x >> 5)) ? buf[lid] : 0;
	if(wid == 0){
		val = QSGate_Dot_warpReduceSum(val);
	}

	return val;
}


__global__ void QSGate_Dot_deviceReduceSum(double *pT,uint64_t n)
{
	uint64_t i;
	double sum;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n)
		sum = pT[i];
	else
		sum = 0.0;

	sum = QSGate_Dot_blockReduceSum(sum);

	if(threadIdx.x==0)
		pT[blockIdx.x]=sum;
}

__global__ void QSGate_Dot_Init(double* pSum,int size)
{
	uint64_t i;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < size){
		pSum[i] = 0.0;
	}
}


__global__ void QSGate_Dot_InUnit_cuda_kernel(double* pSum,QSVec2* qreg,uint64_t offset,int qubit,int bits_per_row,uint64_t rows)
{
	uint64_t add,k,i,k1,k2,mask,ir;
	double sum = 0.0;
	QSVec2 q0;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	add = blockDim.x << bits_per_row;

	mask = ((1ull << qubit) - 1);

	for(ir=0;ir<rows;ir++){
		k2 = i & mask;
		k1 = (i - k2) << 1;
		k = offset + k1 + k2;

		q0 = qreg[k];
		sum += (double)q0.x*(double)q0.x + (double)q0.y*(double)q0.y;
		i += add;
	}

	sum = QSGate_Dot_blockReduceSum(sum);

	if(threadIdx.x == 0){
		pSum[blockIdx.x] += sum;
	}
}

__global__ void QSGate_Dot_cuda_kernel(double* pSum,QSVec2* qreg,uint64_t offset,int bits_per_row,uint64_t rows)
{
	uint64_t add,i,ir;
	double sum = 0.0;
	QSVec2 q0;

	i = offset + threadIdx.x + blockIdx.x * blockDim.x;
	add = blockDim.x << bits_per_row;

	for(ir=0;ir<rows;ir++){
		q0 = qreg[i];
		sum += (double)q0.x*(double)q0.x + (double)q0.y*(double)q0.y;
		i += add;
	}

	sum = QSGate_Dot_blockReduceSum(sum);

	if(threadIdx.x == 0){
		pSum[blockIdx.x] += sum;
	}
}

void QSGate_Dot::InitBuffer(QSUnitStorage* pUnit)
{
	QSUint nt,ng;
	cudaStream_t strm;
	double* pDot;

	pUnit->SetDevice();

	strm = (cudaStream_t)pUnit->GetStream();

	pDot = pUnit->GetNormPointer();

	CUDA_FitThreads(nt,ng,NORM_BUF_SIZE);

	QSGate_Dot_Init<<<ng,nt,0,strm>>>(pDot,NORM_BUF_SIZE);

	cudaStreamSynchronize(strm);
}

double QSGate_Dot::ReduceAll(QSUnitStorage* pUnit)
{
	QSUint nt,ng,ngg;
	cudaStream_t strm;
	double* pDot;
	double dot[16];

	pUnit->SetDevice();

	strm = (cudaStream_t)pUnit->GetStream();

	pDot = pUnit->GetNormPointer();

	ng = NORM_BUF_SIZE;

	while(ng > 1){
		ngg = ng;
		CUDA_FitThreads(nt,ng,ngg);
		if(nt < 32){
			nt = 32;
		}
		QSGate_Dot_deviceReduceSum<<<ng,nt,sizeof(double)*32,strm>>>(pDot,ngg);
	}
	cudaMemcpyAsync(dot,pDot,sizeof(double),cudaMemcpyDeviceToHost,strm);

	cudaStreamSynchronize(strm);

	return dot[0];
}

void QSGate_Dot::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint nt,ng,ngg,nrows,iu,offset;
	cudaStream_t strm;
	double* pDot;
	int qubit = qubits[0];

	pUnit->SetDevice();

	if(nTrans > 0){
		strm = (cudaStream_t)pUnit->GetStreamPipe();
	}
	else{
		strm = (cudaStream_t)pUnit->GetStream();
	}

	pDot = pUnit->GetNormPointer();

	if(nTrans > 0){
		pUnit->SynchronizeInput();
	}

	if(qubit < pUnit->UnitBits()){
		CUDA_FitThreads(nt,ng,1ull << (pUnit->UnitBits()-1));
		nrows = 1;
		if(ng > NORM_BUF_SIZE){
			nrows = (ng + NORM_BUF_SIZE - 1) >> NORM_BUF_BITS;
			ng = NORM_BUF_SIZE;
		}

		QSGate_Dot_InUnit_cuda_kernel<<<ng,nt,sizeof(double)*32*32,strm>>>(pDot,(QSVec2*)(ppBuf[0]),0,qubit,NORM_BUF_BITS,nrows);
	}
	else{
		CUDA_FitThreads(nt,ng,1ull << pUnit->UnitBits());
		nrows = 1;
		if(ng > NORM_BUF_SIZE){
			nrows = (ng + NORM_BUF_SIZE - 1) >> NORM_BUF_BITS;
			ng = NORM_BUF_SIZE;
		}

		QSGate_Dot_cuda_kernel<<<ng,nt,sizeof(double)*32,strm>>>(pDot,(QSVec2*)(ppBuf[0]),0,NORM_BUF_BITS,nrows);
	}

	if(nTrans > 0){
		pUnit->SynchronizeOutput();
	}

	cudaStreamSynchronize(strm);
}



