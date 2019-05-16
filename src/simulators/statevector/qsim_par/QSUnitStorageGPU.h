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

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_UNIT_STORAGE_GPU_H_
#define _IBM_Q_SIMULATOR_UNIT_STORAGE_GPU_H_


#include "QSUnitStorage.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define NORM_BUF_SIZE		65536
#define NORM_BUF_BITS		16


#define CUDA_MAX_THREAD_BITS		10
#define CUDA_MAX_Y_GRID_BITS		16

#define CUDA_FitThreads(nt,ng,n) \
	ng = 1;\
	nt = (n);\
	if((n) > (1 << CUDA_MAX_THREAD_BITS)){ \
		ng = ((n) + ((1 << CUDA_MAX_THREAD_BITS) - 1)) >> CUDA_MAX_THREAD_BITS;\
		nt = (1 << CUDA_MAX_THREAD_BITS);\
	}\

#define CUDA_FitGridYZ(ny,nz,n) \
	nz = 1;\
	ny = (n);\
	if((n) > (1 << CUDA_MAX_Y_GRID_BITS)){ \
		nz = ((n) + ((1 << CUDA_MAX_Y_GRID_BITS) - 1)) >> CUDA_MAX_Y_GRID_BITS;\
		ny = (1 << CUDA_MAX_Y_GRID_BITS);\
	}\





class QSUnitStorageGPU : public QSUnitStorage
{
protected:
	int m_devID;
	int m_devIDoffset;
	QSDouble* m_pNormBuf;
	cudaStream_t m_strm;
	cudaStream_t* m_pStrmPipe;
	cudaStream_t* m_pStrmTrans;
	cudaStream_t m_strmToHost;
	cudaEvent_t* m_pEventIn;
	cudaEvent_t* m_pEventOut;
//	QSDoubleComplex* m_pMat;
	QSComplex** m_pBufPtr;
//	int* m_pBitsPtr;
	QSUint* m_flagTrans;
public:
	QSUnitStorageGPU()
	{
		m_pNormBuf = NULL;
		m_pStrmPipe = NULL;
		m_pStrmTrans = NULL;
		m_pEventIn = NULL;
		m_pEventOut = NULL;
//		m_pMat = NULL;
		m_pBufPtr = NULL;
//		m_pBitsPtr = NULL;
		m_devIDoffset = 0;
		m_devID = 0;
		m_flagTrans = NULL;
	}
	QSUnitStorageGPU(int iPlace,int bits,int iOffset) : QSUnitStorage(iPlace,bits)
	{
		m_pNormBuf = NULL;
		m_pStrmPipe = NULL;
		m_pStrmTrans = NULL;
		m_pEventIn = NULL;
		m_pEventOut = NULL;
//		m_pMat = NULL;
		m_pBufPtr = NULL;
//		m_pBitsPtr = NULL;
		m_devIDoffset = iOffset;
		m_devID = iOffset + iPlace;
		m_flagTrans = NULL;
	}

	virtual ~QSUnitStorageGPU()
	{
		Release();
	}

	int Allocate(QSUint numUnits,int numBuffers);

	void Release(void);

	void SetValue(QSDoubleComplex c,QSUint uid,int pos);
	void Clear(void);
	void ClearUnit(QSUint iUnit);

	void Wait(void)
	{
		cudaSetDevice(m_devID);
		cudaStreamSynchronize(m_strm);
	}
	void WaitPipe(int iPipe);
	void WaitAll(void);

	void ToHost(QSComplex* pDest,QSComplex* pSrc);
	void WaitToHost(void);
	int TestToHost(void);

	void Put(int iBuf,QSComplex* pSrc,int iPlace);
	void Get(int iBuf,QSComplex* pDest,int iPlace);
	void GetAck(int iBuf,QSComplex* pDest,int iPlace);
	void PutOnPipe(int iBuf,QSComplex* pSrc,int iPlace);
	void GetOnPipe(int iBuf,QSComplex* pDest,int iPlace);

	void WaitPut(int iPlace);
	void WaitGet(int iPlace);
	int TestPut(int iPlace);
	int TestGet(int iPlace);

	void SetDevice(void);
	int GetDeviceID(void)
	{
		return m_devID;
	}

	int IsGPU(void)
	{
		return 1;
	}
	int canCompute(void)
	{
		return 1;
	}


//	QSDoubleComplex* GetMatrixPointer(void)
//	{
//		return m_pMat;
//	}
	QSComplex** GetBufferPointer(int matSize)
	{
		return (m_pBufPtr + m_pipeCount*matSize);
	}
//	int* GetQubitsPointer(void)
//	{
//		return m_pBitsPtr;
//	}
	void* GetStream(void)
	{
		return m_strm;
	}
	void* GetStreamPipe(void)
	{
		return m_pStrmPipe[m_pipeCount];
	}
	QSDouble* GetNormPointer(void)
	{
		return m_pNormBuf;
	}

	void SynchronizeInput(void);
	void SynchronizeOutput(void);


};


#endif	//_IBM_Q_SIMULATOR_UNIT_STORAGE_GPU_H_

