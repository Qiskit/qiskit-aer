/*------------------------------------------------------------------------------------
	IBM Q Simulator

	Diagonal multiply gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_DiagMult.h"
#include "QSUnitStorageGPU.h"


#define LoadMatrix(r,m) \
	r.x = (QSRealC)m.x; \
	r.y = (QSRealC)m.y

/*
__global__ void QSGate_DiagMult_InUnit_2x2_cuda_kernel(QSVec2* pAmp,int qubit,double2 mat0,double2 mat1)
{
	QSUint i,k,k1,k2,kb,mask,add;
	QSVec2 vs0,vs1,vd0,vd1;
	QSVec2C m00,m11;
	double2 md;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	m00.x = (QSRealC)mat0.x;
	m00.y = (QSRealC)mat0.y;
	m11.x = (QSRealC)mat1.x;
	m11.y = (QSRealC)mat1.y;

	add = 1ull << qubit;
	mask = add - 1;
	k2 = i & mask;
	k1 = (i - k2) << 1;
	k = k1 + k2;
	kb = k + add;

	vs0 = pAmp[k];
	vs1 = pAmp[kb];

	vd0.x = (QSRealDev)(m00.x * (QSRealC)vs0.x - m00.y * (QSRealC)vs0.y);
	vd0.y = (QSRealDev)(m00.x * (QSRealC)vs0.y + m00.y * (QSRealC)vs0.x);
	vd1.x = (QSRealDev)(m11.x * (QSRealC)vs1.x - m11.y * (QSRealC)vs1.y);
	vd1.y = (QSRealDev)(m11.x * (QSRealC)vs1.y + m11.y * (QSRealC)vs1.x);

	pAmp[k] = vd0;
	pAmp[kb] = vd1;
}
*/

__global__ void QSGate_DiagMult_InUnit_2x2_cuda_kernel(QSVec2* pAmp,int qubit,double2 mat0,double2 mat1)
{
	QSUint i,lmask;
	QSVec2 vs,vd;
	QSVec2C m0;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	if(((i >> qubit) & 1ull) == 0){
		m0.x = (QSRealC)mat0.x;
		m0.y = (QSRealC)mat0.y;
	}
	else{
		m0.x = (QSRealC)mat1.x;
		m0.y = (QSRealC)mat1.y;
	}

	vs = pAmp[i];

	vd.x = (QSRealDev)(m0.x * (QSRealC)vs.x - m0.y * (QSRealC)vs.y);
	vd.y = (QSRealDev)(m0.x * (QSRealC)vs.y + m0.y * (QSRealC)vs.x);

	pAmp[i] = vd;

}


__global__ void QSGate_DiagMult_2x2_cuda_kernel(QSVec2* pBuf,double2 mat)
{
	QSUint i;
	QSVec2 vs,vd;
	QSVec2C m;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	vs = pBuf[i];
	m.x = (QSRealC)mat.x;
	m.y = (QSRealC)mat.y;

	vd.x = (QSRealDev)(m.x * (QSRealC)vs.x - m.y * (QSRealC)vs.y);
	vd.y = (QSRealDev)(m.x * (QSRealC)vs.y + m.y * (QSRealC)vs.x);

	pBuf[i] = vd;
}



void QSGate_DiagMult::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	int i,matSize;
	QSComplex** pBuf_dev;
	QSUint na,nt,ng;
	cudaStream_t strm;

	matSize = 1 << nqubits;
//	na = 1ull << (pUnit->UnitBits() - (nqubits - nqubitsLarge));

	pUnit->SetDevice();

	pBuf_dev = pUnit->GetBufferPointer(matSize);
	strm = (cudaStream_t)pUnit->GetStreamPipe();

	if(nTrans > 0){
		pUnit->SynchronizeInput();
	}


	//multiply matrix
	if(nqubits == 1){
		double2 mat0,mat1;
		mat0.x = ((double*)m_pMat)[0];
		mat0.y = ((double*)m_pMat)[1];
		mat1.x = ((double*)m_pMat)[2];
		mat1.y = ((double*)m_pMat)[3];

		na = 1ull << pUnit->UnitBits();
		CUDA_FitThreads(nt,ng,na);

		if(qubits[0] < pUnit->UnitBits()){		//inside unit
			QSGate_DiagMult_InUnit_2x2_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0],mat0,mat1);
		}
		else{
			if(((pGuid[0] >> (qubits[0] - pUnit->UnitBits())) & 1ull) == 0){
				QSGate_DiagMult_2x2_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],mat0);
			}
			else{
				QSGate_DiagMult_2x2_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],mat1);
			}
		}
	}

	if(nTrans > 0){
		pUnit->SynchronizeOutput();
	}

}

void QSGate_DiagMult::CopyMatrix(QSUnitStorage* pUnit,int* qubits,int nqubits)
{
	cudaStream_t strm;
	int matSize;

	matSize = 1 << nqubits;

	pUnit->SetDevice();
	strm = (cudaStream_t)pUnit->GetStream();

	cudaMemcpyAsync(pUnit->GetMatrixPointer(),m_pMat,sizeof(double2)*matSize*matSize,cudaMemcpyHostToDevice,strm);
	cudaMemcpyAsync(pUnit->GetQubitsPointer(),qubits,sizeof(int)*nqubits,cudaMemcpyHostToDevice,strm);

	cudaStreamSynchronize(strm);
}




