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

	Unit storage on Host

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_UNIT_STORAGE_HOST_H_
#define _IBM_Q_SIMULATOR_UNIT_STORAGE_HOST_H_

#include <stdio.h>
#include <stdlib.h>


#include "QSUnitStorage.h"


class QSUnitStorageHost : public QSUnitStorage
{
protected:

public:
	QSUnitStorageHost()
	{
		m_placeID = -1;
	}
	QSUnitStorageHost(int iPlace,int bits) : QSUnitStorage(iPlace,bits)
	{
	}

	virtual ~QSUnitStorageHost()
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
		;
	}
	void WaitPipe(int iPipe)
	{
		;
	}
	void WaitAll(void)
	{
		;
	}

	void ToHost(QSComplex* pDest,QSComplex* pSrc)
	{
		;
	}
	void WaitToHost(void)
	{
		;
	}
	int TestToHost(void)
	{
		;
	}


	void Put(int iBuf,QSComplex* pSrc,int iPlace)
	{
		;
	}
	void Get(int iBuf,QSComplex* pDest,int iPlace)
	{
		;
	}

	void GetAck(int iBuf,QSComplex* pDest,int iPlace)
	{
		;
	}
	void PutOnPipe(int iBuf,QSComplex* pSrc,int iPlace)
	{
		;
	}
	void GetOnPipe(int iBuf,QSComplex* pDest,int iPlace)
	{
		;
	}

	void WaitPut(int iPlace)
	{
		;
	}
	void WaitGet(int iPlace)
	{
		;
	}
	int TestPut(int iPlace)
	{
		;
	}
	int TestGet(int iPlace)
	{
		;
	}

	void SetDevice(void)
	{
		;
	}
	int GetDeviceID(void)
	{
		return -1;
	}

	int IsGPU(void)
	{
		return 0;
	}
	int canCompute(void)
	{
		return 1;
	}


//	QSDoubleComplex* GetMatrixPointer(void)
//	{
//		return NULL;
//	}
	QSComplex** GetBufferPointer(int matSize)
	{
		return NULL;
	}
//	int* GetQubitsPointer(void)
//	{
//		return NULL;
//	}
	void* GetStream(void)
	{
		return NULL;
	}
	void* GetStreamPipe(void)
	{
		return NULL;
	}
	double* GetNormPointer(void)
	{
		return NULL;
	}

	void SynchronizeInput(void)
	{
		;
	}
	void SynchronizeOutput(void)
	{
		;
	}

};


#endif	//_IBM_Q_SIMULATOR_UNIT_STORAGE_HOST_H_
