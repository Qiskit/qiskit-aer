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

	Controlled not gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_CX.h"
#include "QSUnitStorageGPU.h"

__global__ void QSGate_CX_InUnit_cuda_kernel(QSVec2* pAmp,int qubit_t,int qubit_c,int bIn,int bOut)
{
	QSUint i,ind0,ind1,iIn,iOut;
	QSUint inMask,outMask;
	QSVec2 psi0,psi1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	inMask = (1ull << bIn) - 1;
	outMask = (1ull << (bOut - 1)) - 1;

	iIn = i & inMask;
	iOut = i & outMask;

	ind0 = (1ull << qubit_c) + ((i >> (bOut - 1)) << (bOut + 1)) + ((iOut >> bIn) << (bIn + 1)) + iIn;
	ind1 = ind0 + (1ull << qubit_t);

	psi0 = pAmp[ind0];
	psi1 = pAmp[ind1];

	pAmp[ind0] = psi1;
	pAmp[ind1] = psi0;
}


__global__ void QSGate_CX_t_cuda_kernel(QSVec2* pAmp,int qubit_t)
{
	QSUint i,ind0,ind1,inMask,iIn;
	QSVec2 psi0,psi1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	inMask = (1ull << qubit_t) - 1;

	iIn = i & inMask;

	ind0 = ((i >> qubit_t) << (qubit_t + 1)) + iIn;
	ind1 = ind0 + (1ull << qubit_t);

	psi0 = pAmp[ind0];
	psi1 = pAmp[ind1];

	pAmp[ind0] = psi1;
	pAmp[ind1] = psi0;
}


__global__ void QSGate_CX_cuda_kernel(QSVec2* pBuf0,QSVec2* pBuf1,int qubit_c,QSUint localMask)
{
	QSUint i,ind,inMask;
	QSVec2 psi0,psi1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	inMask = (1ull << qubit_c) - 1;

	ind = (1ull << qubit_c) + ((i >> qubit_c) << (qubit_c + 1)) + (i & inMask);

	psi0 = pBuf0[ind];
	psi1 = pBuf1[ind];
	if(localMask & 1){
		pBuf0[ind] = psi1;
	}
	if(localMask & 2){
		pBuf1[ind] = psi0;
	}
}

__global__ void QSGate_CX_swap_cuda_kernel(QSVec2* pBuf0,QSVec2* pBuf1,QSUint localMask)
{
	QSUint i;
	QSVec2 psi0,psi1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	psi0 = pBuf0[i];
	psi1 = pBuf1[i];
	if(localMask & 1){
		pBuf0[i] = psi1;
	}
	if(localMask & 2){
		pBuf1[i] = psi0;
	}
}



void QSGate_CX::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	int i,nBuf,matSize;
	QSUint na,nt,ng;
	cudaStream_t strm;
	int qubit_t = qubits[0];
	int qubit_c = qubits_c[0];


	pUnit->SetDevice();

	strm = (cudaStream_t)pUnit->GetStreamPipe();

	if(nTrans > 0){
		pUnit->SynchronizeInput();
	}

	if(nqubitsLarge == 0){		//inside unit
		if(qubit_c < pUnit->UnitBits()){
			int bIn,bOut;

			na = 1ull << (pUnit->UnitBits() - 2);
			CUDA_FitThreads(nt,ng,na);

			if(qubit_c < qubit_t){
				bIn = qubit_c;
				bOut = qubit_t;
			}
			else{
				bIn = qubit_t;
				bOut = qubit_c;
			}

			QSGate_CX_InUnit_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubit_t,qubit_c,bIn,bOut);
		}
		else{
			na = 1ull << (pUnit->UnitBits() - 1);
			CUDA_FitThreads(nt,ng,na);

			QSGate_CX_t_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubit_t);
		}
	}
	else{
		if(qubit_c < pUnit->UnitBits()){
			na = 1ull << (pUnit->UnitBits() - 1);
			CUDA_FitThreads(nt,ng,na);

			QSGate_CX_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)(ppBuf[0]),(QSVec2*)(ppBuf[1]),qubit_c,localMask);
		}
		else{
			na = 1ull << (pUnit->UnitBits());
			CUDA_FitThreads(nt,ng,na);

			QSGate_CX_swap_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)(ppBuf[0]),(QSVec2*)(ppBuf[1]),localMask);
		}
	}

	if(nTrans > 0){
		pUnit->SynchronizeOutput();
	}

}



