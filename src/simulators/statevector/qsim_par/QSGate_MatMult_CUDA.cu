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

	matrix multiply gate
	2x2 : U3 gate
	4- : fusion gates

	2018-2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

#include "QSGate_MatMult.h"
#include "QSUnitStorageGPU.h"


__constant__ double2 gMat[QS_MAX_MATRIX_SIZE*QS_MAX_MATRIX_SIZE];
__constant__ int qsm_qubits[QS_MAX_FUSION];


#ifdef QSIM_COL_MAJOR	//for Aer

#define MatrixLoad(r,t,m,i,j,size) \
	t = m[j + i*size]; \
	r.x = (QSRealC)t.x; \
	r.y = (QSRealC)t.y

#else

#define MatrixLoad(r,t,m,i,j,size) \
	t = m[i + j*size]; \
	r.x = (QSRealC)t.x; \
	r.y = (QSRealC)t.y

#endif




__global__ void QS_Set_Value_cuda_kernel(QSVec2* pBuf,double2 c,int pos)
{
	int i;
	QSVec2 v;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i == pos){
		v.x = (QSRealDev)c.x;
		v.y = (QSRealDev)c.y;

		pBuf[i] = v;
	}
}

void QS_Set_Value(QSUnitStorage* pUnit,QSComplex* pBuf,QSDoubleComplex c,int pos)
{
	QSUint na,nt,ng;
	cudaStream_t strm;
	double2 c2;

	na = 1ull << (pUnit->UnitBits());

	CUDA_FitThreads(nt,ng,na);

	pUnit->SetDevice();

	strm = (cudaStream_t)pUnit->GetStream();

	c2.x = ((double*)&c)[0];
	c2.y = ((double*)&c)[1];

	QS_Set_Value_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)pBuf,c2,pos);

	cudaStreamSynchronize(strm);
}



__global__ void QSGate_MatMult_InUnit_2x2_cuda_kernel(QSVec2* pAmp,int qubit)
{
	QSUint i,k,k1,k2,kb,mask,add;
	QSVec2 vs0,vs1,vd0,vd1;
	QSVec2C m00,m01,m10,m11;
	double2 md;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	MatrixLoad(m00,md,gMat,0,0,2);
	MatrixLoad(m01,md,gMat,1,0,2);
	MatrixLoad(m10,md,gMat,0,1,2);
	MatrixLoad(m11,md,gMat,1,1,2);

	add = 1ull << qubit;
	mask = add - 1;
	k2 = i & mask;
	k1 = (i - k2) << 1;
	k = k1 + k2;
	kb = k + add;

	vs0 = pAmp[k];
	vs1 = pAmp[kb];

	vd0.x = (QSRealDev)(m00.x * (QSRealC)vs0.x - m00.y * (QSRealC)vs0.y + m01.x * (QSRealC)vs1.x - m01.y * (QSRealC)vs1.y);
	vd0.y = (QSRealDev)(m00.x * (QSRealC)vs0.y + m00.y * (QSRealC)vs0.x + m01.x * (QSRealC)vs1.y + m01.y * (QSRealC)vs1.x);
	vd1.x = (QSRealDev)(m10.x * (QSRealC)vs0.x - m10.y * (QSRealC)vs0.y + m11.x * (QSRealC)vs1.x - m11.y * (QSRealC)vs1.y);
	vd1.y = (QSRealDev)(m10.x * (QSRealC)vs0.y + m10.y * (QSRealC)vs0.x + m11.x * (QSRealC)vs1.y + m11.y * (QSRealC)vs1.x);

	pAmp[k] = vd0;
	pAmp[kb] = vd1;
}



__global__ void QSGate_MatMult_InUnit_2x2_shfl_cuda_kernel(QSVec2* pAmp,int qubit)
{
	QSUint i,iPair,lmask;
	QSVec2 vs0,vs1,vd;
	QSVec2C m0,m1;
	double2 md;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	lmask = 1ull << qubit;
	iPair = i ^ lmask;

	if(i < iPair){
		MatrixLoad(m0,md,gMat,0,0,2);
		MatrixLoad(m1,md,gMat,1,0,2);
	}
	else{
		MatrixLoad(m1,md,gMat,0,1,2);
		MatrixLoad(m0,md,gMat,1,1,2);
	}

	vs0 = pAmp[i];

	vs1.x = __shfl_xor_sync(0xffffffff,vs0.x,lmask,32);
	vs1.y = __shfl_xor_sync(0xffffffff,vs0.y,lmask,32);

	vd.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y);
	vd.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x);

	pAmp[i] = vd;
}

__global__ void QSGate_MatMult_InUnit_2x2_shm_cuda_kernel(QSVec2* pAmp,int qubit)
{
	extern __shared__ QSVec2 buf[];
	QSUint i,iPair,lmask;
	QSVec2 vs0,vs1,vd;
	QSVec2C m0,m1;
	double2 md;

	i = threadIdx.x;

	lmask = 1ull << qubit;
	iPair = i ^ lmask;

	if(i < iPair){
		MatrixLoad(m0,md,gMat,0,0,2);
		MatrixLoad(m1,md,gMat,1,0,2);
	}
	else{
		MatrixLoad(m1,md,gMat,0,1,2);
		MatrixLoad(m0,md,gMat,1,1,2);
	}

	i +=  + blockIdx.x * blockDim.x;

	vs0 = pAmp[i];
	buf[threadIdx.x] = vs0;
	__syncthreads();

	vs1 = buf[iPair];

	vd.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y);
	vd.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x);

	pAmp[i] = vd;
}



__global__ void QSGate_MatMult_InUnit_4x4_cuda_kernel(QSVec2* pAmp,int qubit_0,int qubit_1)
{
	QSUint mask0,mask1,i,j0,j1,j2,i0,i1,ip0,ip1,add0,add1;
	QSVec2 vs0,vs1,vs2,vs3,vd0,vd1,vd2,vd3;
	QSVec2C m0,m1,m2,m3;
	double2 md;

	add0 = (1ull << qubit_0);
	add1 = (1ull << qubit_1);
	mask0 = add0 - 1;
	mask1 = add1 - 1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	j0 = i & mask0;
	i = (i - j0) << 1;
	j1 = i & mask1;
	i = (i - j1) << 1;
	j2 = i;

	i0 = j0 + j1 + j2;
	ip0 = i0 + add0;
	i1 = i0 + add1;
	ip1 = i1 + add0;

	vs0 = pAmp[i0];
	vs1 = pAmp[ip0];
	vs2 = pAmp[i1];
	vs3 = pAmp[ip1];

	/*
	md = gMat[0];
	m0.x = (QSRealC)md.x;
	m0.y = (QSRealC)md.y;
	md = gMat[1];
	m1.x = (QSRealC)md.x;
	m1.y = (QSRealC)md.y;
	md = gMat[2];
	m2.x = (QSRealC)md.x;
	m2.y = (QSRealC)md.y;
	md = gMat[3];
	m3.x = (QSRealC)md.x;
	m3.y = (QSRealC)md.y;
	*/

	MatrixLoad(m0,md,gMat,0,0,4);
	MatrixLoad(m1,md,gMat,1,0,4);
	MatrixLoad(m2,md,gMat,2,0,4);
	MatrixLoad(m3,md,gMat,3,0,4);
	
	vd0.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
	vd0.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

	/*
	md = gMat[4];
	m0.x = (QSRealC)md.x;
	m0.y = (QSRealC)md.y;
	md = gMat[5];
	m1.x = (QSRealC)md.x;
	m1.y = (QSRealC)md.y;
	md = gMat[6];
	m2.x = (QSRealC)md.x;
	m2.y = (QSRealC)md.y;
	md = gMat[7];
	m3.x = (QSRealC)md.x;
	m3.y = (QSRealC)md.y;
	*/
	MatrixLoad(m0,md,gMat,0,1,4);
	MatrixLoad(m1,md,gMat,1,1,4);
	MatrixLoad(m2,md,gMat,2,1,4);
	MatrixLoad(m3,md,gMat,3,1,4);

	pAmp[i0] = vd0;

	vd1.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
	vd1.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

	/*
	md = gMat[8];
	m0.x = (QSRealC)md.x;
	m0.y = (QSRealC)md.y;
	md = gMat[9];
	m1.x = (QSRealC)md.x;
	m1.y = (QSRealC)md.y;
	md = gMat[10];
	m2.x = (QSRealC)md.x;
	m2.y = (QSRealC)md.y;
	md = gMat[11];
	m3.x = (QSRealC)md.x;
	m3.y = (QSRealC)md.y;
	*/
	MatrixLoad(m0,md,gMat,0,2,4);
	MatrixLoad(m1,md,gMat,1,2,4);
	MatrixLoad(m2,md,gMat,2,2,4);
	MatrixLoad(m3,md,gMat,3,2,4);

	pAmp[ip0] = vd1;

	vd2.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
	vd2.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

	/*
	md = gMat[12];
	m0.x = (QSRealC)md.x;
	m0.y = (QSRealC)md.y;
	md = gMat[13];
	m1.x = (QSRealC)md.x;
	m1.y = (QSRealC)md.y;
	md = gMat[14];
	m2.x = (QSRealC)md.x;
	m2.y = (QSRealC)md.y;
	md = gMat[15];
	m3.x = (QSRealC)md.x;
	m3.y = (QSRealC)md.y;
	*/
	MatrixLoad(m0,md,gMat,0,3,4);
	MatrixLoad(m1,md,gMat,1,3,4);
	MatrixLoad(m2,md,gMat,2,3,4);
	MatrixLoad(m3,md,gMat,3,3,4);

	pAmp[i1] = vd2;

	vd3.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
	vd3.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

	pAmp[ip1] = vd3;
}

__global__ void QSGate_MatMult_InUnit_8x8_cuda_kernel(QSVec2* pAmp,int qubit_0,int qubit_1,int qubit_2)
{
	QSUint mask0,mask1,mask2,add0,add1,add2;
	QSUint i,j,j0,j1,j2,j3;
	QSUint pos[8];
	QSVec2 vs0,vs1,vs2,vs3,vs4,vs5,vs6,vs7;
	QSVec2 vd;
	QSVec2C m0,m1,m2,m3,m4,m5,m6,m7;
	double2 md;

	add0 = (1ull << qubit_0);
	add1 = (1ull << qubit_1);
	add2 = (1ull << qubit_2);
	mask0 = add0 - 1;
	mask1 = add1 - 1;
	mask2 = add2 - 1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	j0 = i & mask0;
	i = (i - j0) << 1;
	j1 = i & mask1;
	i = (i - j1) << 1;
	j2 = i & mask2;
	i = (i - j2) << 1;
	j3 = i;

	pos[0] = j0 + j1 + j2 + j3;
	pos[1] = pos[0] + add0;
	pos[2] = pos[0] + add1;
	pos[3] = pos[2] + add0;
	pos[4] = pos[0] + add2;
	pos[5] = pos[4] + add0;
	pos[6] = pos[4] + add1;
	pos[7] = pos[6] + add0;

	vs0 = pAmp[pos[0]];
	vs1 = pAmp[pos[1]];
	vs2 = pAmp[pos[2]];
	vs3 = pAmp[pos[3]];
	vs4 = pAmp[pos[4]];
	vs5 = pAmp[pos[5]];
	vs6 = pAmp[pos[6]];
	vs7 = pAmp[pos[7]];

	for(j=0;j<8;j++){
		MatrixLoad(m0,md,gMat,0,j,8);
		MatrixLoad(m1,md,gMat,1,j,8);
		MatrixLoad(m2,md,gMat,2,j,8);
		MatrixLoad(m3,md,gMat,3,j,8);
		MatrixLoad(m0,md,gMat,4,j,8);
		MatrixLoad(m1,md,gMat,5,j,8);
		MatrixLoad(m2,md,gMat,6,j,8);
		MatrixLoad(m3,md,gMat,7,j,8);

		vd.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y
							+ m4.x * (QSRealC)vs4.x - m4.y * (QSRealC)vs4.y + m5.x * (QSRealC)vs5.x - m5.y * (QSRealC)vs5.y + m6.x * (QSRealC)vs6.x - m6.y * (QSRealC)vs6.y + m7.x * (QSRealC)vs7.x - m7.y * (QSRealC)vs7.y);
		vd.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x
							+ m4.x * (QSRealC)vs4.y + m4.y * (QSRealC)vs4.x + m5.x * (QSRealC)vs5.y + m5.y * (QSRealC)vs5.x + m6.x * (QSRealC)vs6.y + m6.y * (QSRealC)vs6.x + m7.x * (QSRealC)vs7.y + m7.y * (QSRealC)vs7.x);
		pAmp[pos[j]] = vd;
	}
}


__global__ void QSGate_MatMult_InUnit_NxN_cuda_kernel(QSVec2* pAmp,int n)
{
	extern __shared__ QSVec2 vs_buf[];
	QSUint mask,add;
	QSUint i,idx,t;
	int j,matSize,iv;
	QSVec2* vs;
	QSVec2 vd;
	QSVec2C vt;
	QSVec2C m;
	double2 md;

	i = threadIdx.y + blockIdx.x * blockDim.y;
	iv = threadIdx.x;

	matSize = 1 << n;

	idx = 0;
	for(j=0;j<n;j++){
		add = (1ull << qsm_qubits[j]);
		mask = add - 1;

		t = i & mask;
		idx += t;
		i = (i - t) << 1;

		if((iv >> j) & 1){
			idx += add;
		}
	}
	idx += i;

	vs = vs_buf + threadIdx.y*matSize;
	//load amplitudes into shared memory
	vs[iv] = pAmp[idx];
	__syncthreads();

	vt.x = 0.0;
	vt.y = 0.0;
	for(j=0;j<matSize;j++){
		/*
		md = gMat[j+iv*matSize];
		m.x = (QSRealC)md.x;
		m.y = (QSRealC)md.y;
		*/
		MatrixLoad(m,md,gMat,j,iv,matSize);

		vt.x += m.x * (QSRealC)vs[j].x - m.y * (QSRealC)vs[j].y;
		vt.y += m.x * (QSRealC)vs[j].y + m.y * (QSRealC)vs[j].x;
	}

	vd.x = (QSRealDev)vt.x;
	vd.y = (QSRealDev)vt.y;

	pAmp[idx] = vd;

}


__global__ void QSGate_MatMult_InUnit_NxN_shfl_cuda_kernel(QSVec2* pAmp,int n)
{
	QSUint mask,add;
	QSUint i,idx,t;
	int j,matSize,iv,ic,sid;
	QSVec2 vs;
	QSVec2 vd;
	QSVec2C vt;
	QSVec2C m;
	double2 md;

	i = threadIdx.y + blockIdx.x * blockDim.y;
	iv = threadIdx.x;

	matSize = 1 << n;

	sid = (iv + 1) & (matSize - 1);

	idx = 0;
	for(j=0;j<n;j++){
		add = (1ull << qsm_qubits[j]);
		mask = add - 1;

		t = i & mask;
		idx += t;
		i = (i - t) << 1;

		if((iv >> j) & 1){
			idx += add;
		}
	}
	idx += i;

	vs = pAmp[idx];

	ic = iv;
	vt.x = 0.0;
	vt.y = 0.0;
	for(j=0;j<matSize;j++){
		/*
		md = gMat[ic+iv*matSize];
		m.x = (QSRealC)md.x;
		m.y = (QSRealC)md.y;
		*/
		MatrixLoad(m,md,gMat,ic,iv,matSize);

		vt.x += m.x * (QSRealC)vs.x - m.y * (QSRealC)vs.y;
		vt.y += m.x * (QSRealC)vs.y + m.y * (QSRealC)vs.x;

		vs.x = __shfl_sync(0xffffffff,vs.x,sid,matSize);
		vs.y = __shfl_sync(0xffffffff,vs.y,sid,matSize);
		ic = (ic + 1) & (matSize - 1);
	}

	vd.x = (QSRealDev)vt.x;
	vd.y = (QSRealDev)vt.y;

	pAmp[idx] = vd;

}




__global__ void QSGate_MatMult_2x2_cuda_kernel(QSVec2* pBuf0,QSVec2* pBuf1,QSUint localMask)
{
	QSUint i;
	QSVec2 vs0,vs1,vd0,vd1;
	QSVec2C m00,m01,m10,m11;
	double2 md;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	vs0 = pBuf0[i];
	vs1 = pBuf1[i];

	if(localMask & 1){
		MatrixLoad(m00,md,gMat,0,0,2);
		MatrixLoad(m01,md,gMat,1,0,2);

		vd0.x = (QSRealDev)(m00.x * (QSRealC)vs0.x - m00.y * (QSRealC)vs0.y + m01.x * (QSRealC)vs1.x - m01.y * (QSRealC)vs1.y);
		vd0.y = (QSRealDev)(m00.x * (QSRealC)vs0.y + m00.y * (QSRealC)vs0.x + m01.x * (QSRealC)vs1.y + m01.y * (QSRealC)vs1.x);
		pBuf0[i] = vd0;
	}
	if(localMask & 2){
		MatrixLoad(m10,md,gMat,0,1,2);
		MatrixLoad(m11,md,gMat,1,1,2);

		vd1.x = (QSRealDev)(m10.x * (QSRealC)vs0.x - m10.y * (QSRealC)vs0.y + m11.x * (QSRealC)vs1.x - m11.y * (QSRealC)vs1.y);
		vd1.y = (QSRealDev)(m10.x * (QSRealC)vs0.y + m10.y * (QSRealC)vs0.x + m11.x * (QSRealC)vs1.y + m11.y * (QSRealC)vs1.x);
		pBuf1[i] = vd1;
	}
}


__global__ void QSGate_MatMult_4x4_cuda_kernel(QSVec2* pBuf0,QSVec2* pBuf1,QSVec2* pBuf2,QSVec2* pBuf3,QSUint localMask,int qubit_0,int nLarge)
{
	QSUint i,i0,i1,mask0,add0,j0,j1;
	QSVec2 vs0,vs1,vs2,vs3,vd0,vd1,vd2,vd3;
	QSVec2C m0,m1,m2,m3;
	double2 md;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	if(nLarge == 1){
		add0 = (1ull << qubit_0);
		mask0 = add0 - 1;

		j0 = (i & mask0);
		j1 = (i - j0) << 1;

		i0 = j0 + j1;
		i1 = i0 + add0;
	}
	else{	//nLarge = 2, qubit_0 >= unitBits
		i0 = i;
		i1 = i;
	}

	vs0 = pBuf0[i0];
	vs1 = pBuf1[i1];
	vs2 = pBuf2[i0];
	vs3 = pBuf3[i1];

	if(localMask & 1){
		/*
		md = gMat[0];
		m0.x = (QSRealC)md.x;
		m0.y = (QSRealC)md.y;
		md = gMat[1];
		m1.x = (QSRealC)md.x;
		m1.y = (QSRealC)md.y;
		md = gMat[2];
		m2.x = (QSRealC)md.x;
		m2.y = (QSRealC)md.y;
		md = gMat[3];
		m3.x = (QSRealC)md.x;
		m3.y = (QSRealC)md.y;
		*/
		MatrixLoad(m0,md,gMat,0,0,4);
		MatrixLoad(m1,md,gMat,1,0,4);
		MatrixLoad(m2,md,gMat,2,0,4);
		MatrixLoad(m3,md,gMat,3,0,4);

		vd0.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
		vd0.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

		pBuf0[i0] = vd0;
	}
	if(localMask & 2){
		/*
		md = gMat[4];
		m0.x = (QSRealC)md.x;
		m0.y = (QSRealC)md.y;
		md = gMat[5];
		m1.x = (QSRealC)md.x;
		m1.y = (QSRealC)md.y;
		md = gMat[6];
		m2.x = (QSRealC)md.x;
		m2.y = (QSRealC)md.y;
		md = gMat[7];
		m3.x = (QSRealC)md.x;
		m3.y = (QSRealC)md.y;
		*/
		MatrixLoad(m0,md,gMat,0,1,4);
		MatrixLoad(m1,md,gMat,1,1,4);
		MatrixLoad(m2,md,gMat,2,1,4);
		MatrixLoad(m3,md,gMat,3,1,4);

		vd1.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
		vd1.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

		pBuf1[i1] = vd1;
	}
	if(localMask & 4){
		/*
		md = gMat[8];
		m0.x = (QSRealC)md.x;
		m0.y = (QSRealC)md.y;
		md = gMat[9];
		m1.x = (QSRealC)md.x;
		m1.y = (QSRealC)md.y;
		md = gMat[10];
		m2.x = (QSRealC)md.x;
		m2.y = (QSRealC)md.y;
		md = gMat[11];
		m3.x = (QSRealC)md.x;
		m3.y = (QSRealC)md.y;
		*/
		MatrixLoad(m0,md,gMat,0,2,4);
		MatrixLoad(m1,md,gMat,1,2,4);
		MatrixLoad(m2,md,gMat,2,2,4);
		MatrixLoad(m3,md,gMat,3,2,4);

		vd2.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
		vd2.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

		pBuf2[i0] = vd2;
	}
	if(localMask & 8){
		/*
		md = gMat[12];
		m0.x = (QSRealC)md.x;
		m0.y = (QSRealC)md.y;
		md = gMat[13];
		m1.x = (QSRealC)md.x;
		m1.y = (QSRealC)md.y;
		md = gMat[14];
		m2.x = (QSRealC)md.x;
		m2.y = (QSRealC)md.y;
		md = gMat[15];
		m3.x = (QSRealC)md.x;
		m3.y = (QSRealC)md.y;
		*/
		MatrixLoad(m0,md,gMat,0,3,4);
		MatrixLoad(m1,md,gMat,1,3,4);
		MatrixLoad(m2,md,gMat,2,3,4);
		MatrixLoad(m3,md,gMat,3,3,4);

		vd3.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y);
		vd3.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x);

		pBuf3[i1] = vd3;
	}
}


__global__ void QSGate_MatMult_8x8_cuda_kernel(QSVec2** ppBuf,QSUint localMask,int qubit_0,int qubit_1,int nLarge)
{
	QSUint mask0,mask1,add0,add1;
	QSUint i,j,j0,j1,j2,pos;
	QSVec2 vs0,vs1,vs2,vs3,vs4,vs5,vs6,vs7;
	QSVec2 vd;
	QSVec2C m0,m1,m2,m3,m4,m5,m6,m7;
	double2 md;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	if(nLarge == 1){
		add0 = (1ull << qubit_0);
		add1 = (1ull << qubit_1);
		mask0 = add0 - 1;
		mask1 = add1 - 1;

		j0 = i & mask0;
		i = (i - j0) << 1;
		j1 = i & mask1;
		i = (i - j1) << 1;
		j2 = i;

		pos = j0 + j1 + j2;
	}
	else if(nLarge == 2){	// qubit[1] >= unitBits
		add0 = (1ull << qubit_0);
		mask0 = add0 - 1;

		j0 = i & mask0;
		j1 = (i - j0) << 1;

		pos = j0 + j1;
	}
	else{	//nLarge = 3, qubit[0] >= unitBits
		pos = i;
	}

	vs0 = *(ppBuf[0] + pos);
	vs1 = *(ppBuf[1] + pos);
	vs2 = *(ppBuf[2] + pos);
	vs3 = *(ppBuf[3] + pos);
	vs4 = *(ppBuf[4] + pos);
	vs5 = *(ppBuf[5] + pos);
	vs6 = *(ppBuf[6] + pos);
	vs7 = *(ppBuf[7] + pos);

	for(j=0;j<8;j++){
		if(localMask & (1ull << j)){
			MatrixLoad(m0,md,gMat,0,j,8);
			MatrixLoad(m1,md,gMat,1,j,8);
			MatrixLoad(m2,md,gMat,2,j,8);
			MatrixLoad(m3,md,gMat,3,j,8);
			MatrixLoad(m0,md,gMat,4,j,8);
			MatrixLoad(m1,md,gMat,5,j,8);
			MatrixLoad(m2,md,gMat,6,j,8);
			MatrixLoad(m3,md,gMat,7,j,8);

			vd.x = (QSRealDev)(m0.x * (QSRealC)vs0.x - m0.y * (QSRealC)vs0.y + m1.x * (QSRealC)vs1.x - m1.y * (QSRealC)vs1.y + m2.x * (QSRealC)vs2.x - m2.y * (QSRealC)vs2.y + m3.x * (QSRealC)vs3.x - m3.y * (QSRealC)vs3.y
							+ m4.x * (QSRealC)vs4.x - m4.y * (QSRealC)vs4.y + m5.x * (QSRealC)vs5.x - m5.y * (QSRealC)vs5.y + m6.x * (QSRealC)vs6.x - m6.y * (QSRealC)vs6.y + m7.x * (QSRealC)vs7.x - m7.y * (QSRealC)vs7.y);
			vd.y = (QSRealDev)(m0.x * (QSRealC)vs0.y + m0.y * (QSRealC)vs0.x + m1.x * (QSRealC)vs1.y + m1.y * (QSRealC)vs1.x + m2.x * (QSRealC)vs2.y + m2.y * (QSRealC)vs2.x + m3.x * (QSRealC)vs3.y + m3.y * (QSRealC)vs3.x
							+ m4.x * (QSRealC)vs4.y + m4.y * (QSRealC)vs4.x + m5.x * (QSRealC)vs5.y + m5.y * (QSRealC)vs5.x + m6.x * (QSRealC)vs6.y + m6.y * (QSRealC)vs6.x + m7.x * (QSRealC)vs7.y + m7.y * (QSRealC)vs7.x);

			*(ppBuf[j] + pos) = vd;
		}
	}
}

__global__ void QSGate_MatMult_NxN_cuda_kernel(QSVec2** ppBuf,QSUint localMask,int n,int nLarge)
{
	extern __shared__ QSVec2 vs_buf[];
	QSUint mask,add;
	QSUint i,idx,t;
	int j,matSize,iv;
	QSVec2* vs;
	QSVec2 vd;
	QSVec2C vt;
	QSVec2C m;
	double2 md;

	i = threadIdx.y + blockIdx.x * blockDim.y;
	iv = threadIdx.x;

	matSize = 1 << n;

	idx = 0;
	for(j=0;j<n-nLarge;j++){
		add = (1ull << qsm_qubits[j]);
		mask = add - 1;

		t = i & mask;
		idx += t;
		i = (i - t) << 1;
	}
	idx += i;

	vs = vs_buf + threadIdx.y*matSize;

	//load amplitudes into shared memory
	vs[iv] = *(ppBuf[iv] + idx);
	__syncthreads();

	if(localMask & (1ull << iv)){
		vt.x = 0.0;
		vt.y = 0.0;
		for(j=0;j<matSize;j++){
			/*
			md = gMat[j+iv*matSize];
			m.x = (QSRealC)md.x;
			m.y = (QSRealC)md.y;
			*/
			MatrixLoad(m,md,gMat,j,iv,matSize);

			vt.x += m.x * (QSRealC)vs[j].x - m.y * (QSRealC)vs[j].y;
			vt.y += m.x * (QSRealC)vs[j].y + m.y * (QSRealC)vs[j].x;
		}

		vd.x = (QSRealDev)vt.x;
		vd.y = (QSRealDev)vt.y;
		*(ppBuf[iv] + idx) = vd;
	}
}

__global__ void QSGate_MatMult_NxN_shfl_cuda_kernel(QSVec2** ppBuf,QSUint localMask,int n,int nLarge)
{
	QSUint mask,add;
	QSUint i,idx,t;
	int j,matSize,iv,ic,sid;
	QSVec2 vs;
	QSVec2 vd;
	QSVec2C vt;
	QSVec2C m;
	double2 md;

	i = threadIdx.y + blockIdx.x * blockDim.y;
	iv = threadIdx.x;

	matSize = 1 << n;

	sid = (iv + 1) & (matSize - 1);

	idx = 0;
	for(j=0;j<n-nLarge;j++){
		add = (1ull << qsm_qubits[j]);
		mask = add - 1;

		t = i & mask;
		idx += t;
		i = (i - t) << 1;
	}
	idx += i;

	vs = *(ppBuf[iv] + idx);

	vt.x = 0.0;
	vt.y = 0.0;
	ic = iv;
	for(j=0;j<matSize;j++){
		/*
		md = gMat[ic+iv*matSize];
		m.x = (QSRealC)md.x;
		m.y = (QSRealC)md.y;
		*/
		MatrixLoad(m,md,gMat,ic,iv,matSize);

		vt.x += m.x * (QSRealC)vs.x - m.y * (QSRealC)vs.y;
		vt.y += m.x * (QSRealC)vs.y + m.y * (QSRealC)vs.x;

		vs.x = __shfl_sync(0xffffffff,vs.x,sid,matSize);
		vs.y = __shfl_sync(0xffffffff,vs.y,sid,matSize);
		ic = (ic + 1) & (matSize - 1);
	}


	if(localMask & (1ull << iv)){
		vd.x = (QSRealDev)vt.x;
		vd.y = (QSRealDev)vt.y;
		*(ppBuf[iv] + idx) = vd;
	}
}


void QSGate_MatMult::ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
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


	//multiply matrix
	if(nqubits == 1){
		if(nqubitsLarge == 0){		//inside unit
			if(qubits[0] < 5){
				CUDA_FitThreads(nt,ng,(na << 1));
				QSGate_MatMult_InUnit_2x2_shfl_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
			}
			else if(qubits[0] < 10){
				CUDA_FitThreads(nt,ng,(na << 1));
				QSGate_MatMult_InUnit_2x2_shm_cuda_kernel<<<ng,nt,sizeof(QSVec2)*1024,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
			}
			else{
				QSGate_MatMult_InUnit_2x2_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0]);
			}
		}
		else{
			QSGate_MatMult_2x2_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],(QSVec2*)ppBuf[1],localMask);
		}
	}
	else if(nqubits == 2){
		if(nqubitsLarge == 0){		//inside unit
			QSGate_MatMult_InUnit_4x4_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0],qubits[1]);
		}
		else if(nqubitsLarge == 1){
			localMask = ((localMask & 1) << 1) | (localMask & 1) | ((localMask & 2) << 1) | ((localMask & 2) << 2);
			QSGate_MatMult_4x4_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],(QSVec2*)ppBuf[0],(QSVec2*)ppBuf[1],(QSVec2*)ppBuf[1],localMask,qubits[0],nqubitsLarge);
		}
		else{
			QSGate_MatMult_4x4_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],(QSVec2*)ppBuf[1],(QSVec2*)ppBuf[2],(QSVec2*)ppBuf[3],localMask,qubits[0],nqubitsLarge);
		}
	}
#if 0
	else if(nqubits == 3){
		if(nqubitsLarge == 0){		//inside unit
			QSGate_MatMult_InUnit_8x8_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2*)ppBuf[0],qubits[0],qubits[1],qubits[2]);
		}
		else{
			if(nqubitsLarge == 1){
				QSUint add0,add1;
				localMask = (localMask & 1) | ((localMask & 1) << 1) | ((localMask & 1) << 2) | ((localMask & 1) << 3) | ((localMask & 2) << 3) | ((localMask & 2) << 4) | ((localMask & 2) << 5) | ((localMask & 2) << 6);

				add0 = (1ull << qubits[0]);
				add1 = (1ull << qubits[1]);

				pBuf_dev[0] = ppBuf[0];
				pBuf_dev[1] = ppBuf[0] + add0;
				pBuf_dev[2] = ppBuf[0] + add1;
				pBuf_dev[3] = ppBuf[0] + add1 + add0;
				pBuf_dev[4] = ppBuf[1];
				pBuf_dev[5] = ppBuf[1] + add0;
				pBuf_dev[6] = ppBuf[1] + add1;
				pBuf_dev[7] = ppBuf[1] + add1 + add0;
			}
			else if(nqubitsLarge == 2){
				QSUint add0;

				localMask = (localMask & 1) | ((localMask & 1) << 1) | ((localMask & 2) << 1) | ((localMask & 2) << 2) | ((localMask & 4) << 2) | ((localMask & 4) << 3) | ((localMask & 8) << 3) | ((localMask & 8) << 4);

				add0 = (1ull << qubits[0]);

				pBuf_dev[0] = ppBuf[0];
				pBuf_dev[1] = ppBuf[0] + add0;
				pBuf_dev[2] = ppBuf[1];
				pBuf_dev[3] = ppBuf[1] + add0;
				pBuf_dev[4] = ppBuf[2];
				pBuf_dev[5] = ppBuf[2] + add0;
				pBuf_dev[6] = ppBuf[3];
				pBuf_dev[7] = ppBuf[3] + add0;
			}
			else{	//3
				pBuf_dev[0] = ppBuf[0];
				pBuf_dev[1] = ppBuf[1];
				pBuf_dev[2] = ppBuf[2];
				pBuf_dev[3] = ppBuf[3];
				pBuf_dev[4] = ppBuf[4];
				pBuf_dev[5] = ppBuf[5];
				pBuf_dev[6] = ppBuf[6];
				pBuf_dev[7] = ppBuf[7];
			}
			QSGate_MatMult_8x8_cuda_kernel<<<ng,nt,0,strm>>>((QSVec2**)pBuf_dev,localMask,qubits[0],qubits[1],nqubitsLarge);
		}
	}
#endif
	else{	//NxN
		if(nqubitsLarge == 0){		//inside unit
			na = na << nqubits;
			CUDA_FitThreads(nt,ng,na);
			if(nqubits > 5){	//larger than warp size
				QSGate_MatMult_InUnit_NxN_cuda_kernel<<<ng,dim3(matSize,(nt>>nqubits),1),sizeof(QSVec2C)*nt,strm>>>((QSVec2*)ppBuf[0],nqubits);
			}
			else{
				QSGate_MatMult_InUnit_NxN_shfl_cuda_kernel<<<ng,dim3(matSize,(nt>>nqubits),1),0,strm>>>((QSVec2*)ppBuf[0],nqubits);
			}
		}
		else{
			QSUint add,addf;
			QSUint mask;
			int j,k;

			mask = 0;
			for(j=0;j<matSize;j++){
				k = (j >> (nqubits - nqubitsLarge));
				pBuf_dev[j] = ppBuf[k];
				mask |= ( ((localMask >> k) & 1ull) << j );
			}

			for(i=0;i<nqubits - nqubitsLarge;i++){
				add = (1ull << qubits[i]);
				addf = 1 << i;
				for(j=0;j<matSize;j++){
					if(j & addf)
						pBuf_dev[j] += add;
				}
			}

			na = na << nqubits;
			CUDA_FitThreads(nt,ng,na);
			if(nqubits > 5){	//larger than warp size
				QSGate_MatMult_NxN_cuda_kernel<<<ng,dim3(matSize,(nt>>nqubits),1),sizeof(QSVec2C)*nt,strm>>>((QSVec2**)pBuf_dev,mask,nqubits,nqubitsLarge);
			}
			else{
				QSGate_MatMult_NxN_shfl_cuda_kernel<<<ng,dim3(matSize,(nt>>nqubits),1),0,strm>>>((QSVec2**)pBuf_dev,mask,nqubits,nqubitsLarge);
			}
		}
	}

	if(nTrans > 0){
		pUnit->SynchronizeOutput();
	}

}

void QSGate_MatMult::CopyMatrix(QSUnitStorage* pUnit,int* qubits,int nqubits)
{
	cudaStream_t strm;
	int matSize;

	matSize = 1 << nqubits;

	pUnit->SetDevice();
	strm = (cudaStream_t)pUnit->GetStream();

	cudaMemcpyToSymbolAsync(gMat,m_pMat,sizeof(double2)*matSize*matSize,0,cudaMemcpyHostToDevice,strm);
	cudaMemcpyToSymbolAsync(qsm_qubits,qubits,sizeof(int)*nqubits,0,cudaMemcpyHostToDevice,strm);

	cudaStreamSynchronize(strm);
}




