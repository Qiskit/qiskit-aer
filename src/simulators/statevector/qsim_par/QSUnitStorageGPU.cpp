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

	Unit storage on GPU

	2018-2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <sys/sysinfo.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef QSIM_MPI
#include <mpi.h>
#endif

#include <omp.h>


#include "QSUnitStorageGPU.h"


int QSUnitStorageGPU::Allocate(QSUint numUnits,int nPipe,int numBuffers)
{
	size_t freeMem,totalMem;
	int i;
	int ndev;
	long long nu;

	cudaGetDeviceCount(&ndev);

	cudaSetDevice(m_devID);

	//allocate buffer for reduction
	cudaMalloc(&m_pNormBuf,sizeof(double)*NORM_BUF_SIZE);

	//initialize stream and event
	cudaStreamCreateWithFlags(&m_strm, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_strmToHost, cudaStreamNonBlocking);

	if(nPipe > 0){
		m_flagTrans = new QSUint[nPipe];
		for(i=0;i<nPipe;i++){
			m_flagTrans[i] = 0;
		}
		m_pStrmPipe = new cudaStream_t[nPipe];
		m_pStrmTrans = new cudaStream_t[nPipe];
		m_pEventIn = new cudaEvent_t[nPipe];
		m_pEventOut = new cudaEvent_t[nPipe];
		for(i=0;i<nPipe;i++){
			cudaStreamCreateWithFlags(&m_pStrmPipe[i], cudaStreamNonBlocking);
			cudaStreamCreateWithFlags(&m_pStrmTrans[i], cudaStreamNonBlocking);
			cudaEventCreateWithFlags(&m_pEventIn[i],cudaEventDisableTiming);
			cudaEventCreateWithFlags(&m_pEventOut[i],cudaEventDisableTiming);
		}
	}

	//Set p2p
	for(i=0;i<ndev;i++){
		if(i != m_devID)
			cudaDeviceEnablePeerAccess(i,0);
	}

	//allocate storage for matrix
	int nPipeForMaxMat = (nPipe)/QS_MAX_MATRIX_SIZE;
	cudaMalloc(&m_pMat,sizeof(QSDoubleComplex)*QS_MAX_MATRIX_SIZE*QS_MAX_MATRIX_SIZE);
//	cudaMalloc(&m_pBufPtr,sizeof(QSComplex*)*QS_MAX_MATRIX_SIZE*(nPipeForMaxMat+1));
	cudaMalloc(&m_pBitsPtr,sizeof(int)*QS_MAX_MATRIX_SIZE);

	//use Unified Memory to point buffers
	cudaMallocManaged(&m_pBufPtr,sizeof(QSComplex*)*QS_MAX_MATRIX_SIZE*(nPipeForMaxMat+1));

	//allocate units + buffers
	nu = numUnits;
	cudaMemGetInfo(&freeMem,&totalMem);
	if( ((uint64_t)(numUnits + numBuffers) << m_unitBits)*sizeof(QSComplex) >= freeMem ){
		nu = ((long)(freeMem/sizeof(QSComplex)) >> m_unitBits) - (long)numBuffers - 1;
	}

#ifdef QSIM_DEBUG
	printf(" [%d] units = %lld, mem = %lld/%lld\n",m_devID,nu,freeMem,totalMem);
#endif

	if(nu <= 0){
		return 0;
	}
	numUnits = nu;

	while(cudaMalloc(&m_pAmp,(uint64_t)(sizeof(QSComplex)*(numUnits + numBuffers)) << m_unitBits) != cudaSuccess){
		numUnits -= 1;
		if(numUnits <= 0){
			printf(" Allocating buffer on GPU fails\n");
			break;
		}
	}
	m_numUnits = numUnits;
	m_numBuffer = numBuffers;
	m_nMaxPipe = nPipe;
	m_numStorage = numUnits + numBuffers;

//	cudaMemsetAsync(m_pAmp,0,sizeof(QSComplex)*(m_numStorage)*m_unitSize,m_strm);
//	cudaStreamSynchronize(m_strm);


	return m_numUnits;
}



void QSUnitStorageGPU::Release(void)
{
	int i;

	if(m_pAmp){
		cudaFree(m_pAmp);
		m_pAmp = NULL;
	}

	if(m_pNormBuf){
		cudaFree(m_pNormBuf);
		m_pNormBuf = NULL;
	}

	if(m_nMaxPipe > 0){
		if(m_pStrmPipe){
			for(i=0;i<m_nMaxPipe;i++){
				cudaStreamDestroy(m_pStrmPipe[i]);
				cudaStreamDestroy(m_pStrmTrans[i]);
				cudaEventDestroy(m_pEventIn[i]);
				cudaEventDestroy(m_pEventOut[i]);
			}
			delete[] m_pStrmPipe;
			delete[] m_pStrmTrans;
			delete[] m_pEventIn;
			delete[] m_pEventOut;
			m_pStrmPipe = NULL;
		}
	}

	if(m_pMat){
		cudaFree(m_pMat);
		m_pMat = NULL;
	}
	if(m_pBufPtr){
		cudaFree(m_pBufPtr);
		m_pBufPtr = NULL;
	}
	if(m_pBitsPtr){
		cudaFree(m_pBitsPtr);
		m_pBitsPtr = NULL;
	}

	if(m_flagTrans){
		delete[] m_flagTrans;
	}

	cudaStreamDestroy(m_strm);
	cudaStreamDestroy(m_strmToHost);
}

void QSUnitStorageGPU::SetDevice(void)
{
	cudaSetDevice(m_devID);
}

extern void QS_Set_Value(QSUnitStorage* pUnit,QSComplex* pBuf,QSDoubleComplex c,int pos);

void QSUnitStorageGPU::SetValue(QSDoubleComplex c,QSUint uid,int pos)
{
	/*
	cudaSetDevice(m_devID);
	cudaMemcpyAsync(m_pAmp + uid*m_unitSize + pos,&c,sizeof(QSComplex),cudaMemcpyHostToDevice,m_pStrmTrans[0]);
	cudaStreamSynchronize(m_pStrmTrans[0]);
	*/
	QS_Set_Value(this,m_pAmp + uid*m_unitSize,c,pos);
}

void QSUnitStorageGPU::Clear(void)
{
	cudaSetDevice(m_devID);
	cudaMemsetAsync(m_pAmp,0,sizeof(QSComplex)*m_unitSize*(QSUint)m_numUnits,m_pStrmTrans[0]);
	cudaStreamSynchronize(m_pStrmTrans[0]);
}


void QSUnitStorageGPU::ClearUnit(QSUint iUnit)
{
	cudaSetDevice(m_devID);
	cudaMemsetAsync(m_pAmp + iUnit*m_unitSize,0,sizeof(QSComplex)*m_unitSize,m_pStrmTrans[0]);
	cudaStreamSynchronize(m_pStrmTrans[0]);
}

void QSUnitStorageGPU::Copy(QSComplex* pV,QSUint iUnit)
{
	cudaSetDevice(m_devID);
	cudaMemcpyAsync(m_pAmp + iUnit*m_unitSize,pV,sizeof(QSComplex)*m_unitSize,cudaMemcpyHostToDevice,m_strm);
}

void QSUnitStorageGPU::ToHost(QSComplex* pDest,QSComplex* pSrc)
{
	cudaSetDevice(m_devID);
	cudaMemcpyAsync(pDest,pSrc,sizeof(QSComplex)*m_unitSize,cudaMemcpyDeviceToHost,m_strmToHost);
}

void QSUnitStorageGPU::WaitToHost(void)
{
	cudaSetDevice(m_devID);
	cudaStreamSynchronize(m_strmToHost);
}

int QSUnitStorageGPU::TestToHost(void)
{
	cudaSetDevice(m_devID);
	return (cudaStreamQuery(m_strmToHost) == cudaSuccess);
}


void QSUnitStorageGPU::Put(int iBuf,QSComplex* pSrc,int iPlace)
{
	if(m_pPlaces[iPlace]->IsGPU() == 0){	//from host
		cudaSetDevice(m_devID);
		cudaMemcpyAsync(Buffer(iBuf),pSrc,sizeof(QSComplex)*m_unitSize,cudaMemcpyHostToDevice,m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
	}
	else{	//from other GPUs
		cudaMemcpyPeerAsync(Buffer(iBuf),m_devID,pSrc,m_pPlaces[iPlace]->GetDeviceID(),sizeof(QSComplex)*m_unitSize,m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
	}
	m_flagTrans[m_pipeCount] |= (1ull << iBuf);
}

void QSUnitStorageGPU::Get(int iBuf,QSComplex* pDest,int iPlace)
{
	if(m_pPlaces[iPlace]->IsGPU() == 0){	//from host
		cudaSetDevice(m_devID);
		cudaMemcpyAsync(pDest,Buffer(iBuf),sizeof(QSComplex)*m_unitSize,cudaMemcpyDeviceToHost,m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
	}
	else{	//from other GPUs
		cudaMemcpyPeerAsync(pDest,m_pPlaces[iPlace]->GetDeviceID(),Buffer(iBuf),m_devID,sizeof(QSComplex)*m_unitSize,m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
	}
	m_flagTrans[m_pipeCount] |= (1ull << iBuf);
}

void QSUnitStorageGPU::GetAck(int iBuf,QSComplex* pDest,int iPlace)
{
	if(m_pPlaces[iPlace]->IsGPU() == 0){	//from host
		cudaSetDevice(m_devID);
		cudaMemcpyAsync(pDest,Buffer(iBuf),sizeof(QSComplex),cudaMemcpyDeviceToHost,m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
	}
	else{	//from other GPUs
		cudaMemcpyPeerAsync(pDest,m_pPlaces[iPlace]->GetDeviceID(),Buffer(iBuf),m_devID,sizeof(QSComplex),m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
	}
	m_flagTrans[m_pipeCount] |= (1ull << iBuf);
}


void QSUnitStorageGPU::PutOnPipe(int iBuf,QSComplex* pSrc,int iPlace)
{
	if(m_pPlaces[iPlace]->IsGPU() == 0){	//from host
		cudaSetDevice(m_devID);
		cudaMemcpyAsync(Buffer(iBuf),pSrc,sizeof(QSComplex)*m_unitSize,cudaMemcpyHostToDevice,m_strm);
	}
	else{	//from other GPUs
		cudaMemcpyPeerAsync(Buffer(iBuf),m_devID,pSrc,m_pPlaces[iPlace]->GetDeviceID(),sizeof(QSComplex)*m_unitSize,m_strm);
	}
}

void QSUnitStorageGPU::GetOnPipe(int iBuf,QSComplex* pDest,int iPlace)
{
	if(m_pPlaces[iPlace]->IsGPU() == 0){	//from host
		cudaSetDevice(m_devID);
		cudaMemcpyAsync(pDest,Buffer(iBuf),sizeof(QSComplex)*m_unitSize,cudaMemcpyDeviceToHost,m_strm);
	}
	else{	//from other GPUs
		cudaMemcpyPeerAsync(pDest,m_pPlaces[iPlace]->GetDeviceID(),Buffer(iBuf),m_devID,sizeof(QSComplex)*m_unitSize,m_strm);
	}
}


void QSUnitStorageGPU::WaitPut(int iBuf)
{
//	cudaSetDevice(m_devID);
	cudaStreamSynchronize(m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);
}

void QSUnitStorageGPU::WaitGet(int iBuf)
{
//	cudaSetDevice(m_devID);
	cudaStreamSynchronize(m_pStrmTrans[iBuf + m_pipeCount*m_unitPerPipe]);

	m_flagTrans[m_pipeCount] ^= (1ull << iBuf);
}

int QSUnitStorageGPU::TestPut(int iPlace)
{
	cudaSetDevice(m_devID);
	return (cudaStreamQuery(m_pStrmTrans[iPlace + m_pipeCount*m_numPlaces]) == cudaSuccess);
}

int QSUnitStorageGPU::TestGet(int iPlace)
{
	cudaSetDevice(m_devID);
	return (cudaStreamQuery(m_pStrmTrans[iPlace + m_pipeCount*m_numPlaces]) == cudaSuccess);

}

void QSUnitStorageGPU::WaitPipe(int iPipe)
{
	int i;
	QSUint flag = 0;

//	cudaSetDevice(m_devID);

	cudaStreamSynchronize(m_pStrmPipe[iPipe]);
	for(i=0;i<m_unitPerPipe;i++){
		if((m_flagTrans[iPipe] >> i) & 1){
			cudaStreamSynchronize(m_pStrmTrans[i + iPipe*m_unitPerPipe]);
		}
	}
	m_flagTrans[iPipe] = 0;
}

void QSUnitStorageGPU::WaitAll(void)
{
	int i,j;
//	cudaSetDevice(m_devID);

	for(i=0;i<m_nPipe;i++){
		cudaStreamSynchronize(m_pStrmPipe[i]);
		if(m_flagTrans[i] != 0){
			for(j=0;j<m_unitPerPipe;j++){
				if((m_flagTrans[i] >> j) & 1){
					cudaStreamSynchronize(m_pStrmTrans[j + i*m_unitPerPipe]);
				}
			}
			m_flagTrans[i] = 0;
		}
	}

}

void QSUnitStorageGPU::SynchronizeInput(void)
{
	int i;

	if(m_flagTrans[m_pipeCount] != 0){
		for(i=0;i<m_unitPerPipe;i++){
			if((m_flagTrans[m_pipeCount] >> i) & 1){
				cudaEventRecord(m_pEventIn[i + m_pipeCount*m_unitPerPipe],m_pStrmTrans[i + m_pipeCount*m_unitPerPipe]);
				cudaStreamWaitEvent(m_pStrmPipe[m_pipeCount],m_pEventIn[i + m_pipeCount*m_unitPerPipe],0);
			}
		}
	}
}

void QSUnitStorageGPU::SynchronizeOutput(void)
{
	int i;

	if(m_flagTrans[m_pipeCount] != 0){
		for(i=0;i<m_unitPerPipe;i++){
			if((m_flagTrans[m_pipeCount] >> i) & 1){
				cudaEventRecord(m_pEventOut[i + m_pipeCount*m_unitPerPipe],m_pStrmPipe[m_pipeCount]);
				cudaStreamWaitEvent(m_pStrmTrans[i + m_pipeCount*m_unitPerPipe],m_pEventOut[i + m_pipeCount*m_unitPerPipe],0);
			}
		}
	}
}




