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

	Z gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_Z.h"
#include "QSUnitStorageGPU.h"


__global__ void QSGate_Z_InUnit_cuda_kernel(QSVec2* pAmp,int qubit)
{
	QSUint i,k,k1,k2,kb,mask,add;
	QSVec2 vs1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	add = 1ull << qubit;
	mask = add - 1;
	k2 = i & mask;
	k1 = (i - k2) << 1;
	k = k1 + k2;
	kb = k + add;

	vs1 = pAmp[kb];
	vs1.x = -vs1.x;
	vs1.y = -vs1.y;
	pAmp[kb] = vs1;
}


__global__ void QSGate_Z_cuda_kernel(QSVec2* pBuf)
{
	QSUint i;
	QSVec2 vs1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	vs1 = pBuf[i];
	vs1.x = -vs1.x;
	vs1.y = -vs1.y;
	pBuf[i] = vs1;
}


void QSGate_Z::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
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
			QSGate_Z_InUnit_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
		}
		else{
			QSGate_Z_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[1]);
		}
	}

	if(nTrans > 0){
		pUnit->SynchronizeOutput();
	}

}


