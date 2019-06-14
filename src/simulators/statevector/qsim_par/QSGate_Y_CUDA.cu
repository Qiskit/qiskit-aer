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

	Y gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_Y.h"
#include "QSUnitStorageGPU.h"


__global__ void QSGate_Y_InUnit_cuda_kernel(QSVec2* pAmp,int qubit)
{
	QSUint i,k,k1,k2,kb,mask,add;
	QSVec2 vs0,vs1,vd0,vd1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	add = 1ull << qubit;
	mask = add - 1;
	k2 = i & mask;
	k1 = (i - k2) << 1;
	k = k1 + k2;
	kb = k + add;

	vs0 = pAmp[k];
	vs1 = pAmp[kb];

	vd0.x =  vs1.y;
	vd0.y = -vs1.x;
	vd1.x = -vs0.y;
	vd1.y =  vs0.x;

	pAmp[k] = vd0;
	pAmp[kb] = vd1;
}


__global__ void QSGate_Y_InUnit_shfl_cuda_kernel(QSVec2* pAmp,int qubit)
{
	QSUint i,iPair,lmask;
	QSVec2 vs0,vd;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	lmask = 1ull << qubit;
	iPair = i ^ lmask;

	vs0 = pAmp[i];

	vd.y = __shfl_xor_sync(0xffffffff,vs0.x,lmask,32);
	vd.x = __shfl_xor_sync(0xffffffff,vs0.y,lmask,32);

	if((i & lmask) != 0){
		vd.x = -vd.x;
	}
	else{
		vd.y = -vd.y;
	}

	pAmp[i] = vd;
}

__global__ void QSGate_Y_InUnit_shm_cuda_kernel(QSVec2* pAmp,int qubit)
{
	extern __shared__ QSVec2 buf[];
	QSUint i,iPair,lmask;
	QSVec2 vs0,vd;

	i = threadIdx.x;

	lmask = 1ull << qubit;
	iPair = i ^ lmask;

	i +=  + blockIdx.x * blockDim.x;

	vs0 = pAmp[i];
	buf[threadIdx.x] = vs0;
	__syncthreads();

	vs0 = buf[iPair];

	if((i & lmask) != 0){
		vd.x = -vs0.y;
		vd.y =  vs0.x;
	}
	else{
		vd.x =  vs0.y;
		vd.y = -vs0.x;
	}

	pAmp[i] = vd;
}


__global__ void QSGate_Y_cuda_kernel(QSVec2* pBuf0,QSVec2* pBuf1,QSUint localMask)
{
	QSUint i;
	QSVec2 vs0,vs1,vd0,vd1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	vs0 = pBuf0[i];
	vs1 = pBuf1[i];

	if(localMask & 1){
		vd0.x =  vs1.y;
		vd0.y = -vs1.x;
		pBuf0[i] = vd0;
	}
	if(localMask & 2){
		vd1.x = -vs0.y;
		vd1.y =  vs0.x;
		pBuf1[i] = vd1;
	}
}


void QSGate_Y::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	int i,matSize;
	QSComplex** pBuf_dev;
	QSUint na,nt,ng;
	cudaStream_t strm;

	matSize = 1 << nqubits;
	na = 1ull << (pUnit->UnitBits() - (nqubits - nqubitsLarge));

	CUDA_FitThreads(nt,ng,na);

	pUnit->SetDevice();

	pBuf_dev = pUnit->GetBufferPointer(matSize);
	strm = (cudaStream_t)pUnit->GetStreamPipe();

	if(nTrans > 0){
		pUnit->SynchronizeInput();
	}

	if(nqubits == 1){
		if(nqubitsLarge == 0){		//inside unit
			if(qubits[0] < 5){
				CUDA_FitThreads(nt,ng,(na << 1));
				QSGate_Y_InUnit_shfl_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
			}
			else if(qubits[0] < 10){
				CUDA_FitThreads(nt,ng,(na << 1));
				QSGate_Y_InUnit_shm_cuda_kernel<<<ng,nt,sizeof(QSVec2)*1024,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
			}
			else{
				QSGate_Y_InUnit_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
			}
		}
		else{
			QSGate_Y_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],(QSVec2*)ppBuf[1],localMask);
		}
	}

	if(nTrans > 0){
		pUnit->SynchronizeOutput();
	}

}


