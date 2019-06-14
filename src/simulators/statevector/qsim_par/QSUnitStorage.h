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

	Unit storage base class

	2018-2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_UNIT_STORAGE_H_
#define _IBM_Q_SIMULATOR_UNIT_STORAGE_H_

#include "QSType.h"

#define QS_MAX_FUSION				5
#define QS_MAX_MATRIX_SIZE			(1 << QS_MAX_FUSION)
#define QS_MAX_PIPE					32

class QSUnitStorage;


class QSUnitStorage
{
protected:
	QSComplex* m_pAmp;
	int m_unitBits;
	QSUint m_unitSize;
	QSUint m_numUnits;
	int m_numBuffer;
	QSUint m_numStorage;
	int* m_pIndex;
	int m_placeID;
	QSUint m_globalUnitIndex;
	int m_numPlaces;
	int m_nMaxPipe;
	int m_nPipe;
	int m_unitPerPipe;
	int m_pipeCount;
	QSUnitStorage** m_pPlaces;
	int m_procPerNode;
public:
	QSUnitStorage()
	{
		m_pAmp = NULL;
		m_pIndex = NULL;
		m_pPlaces = NULL;
		m_procPerNode = 1;
	}
	QSUnitStorage(int iPlace,int bits)
	{
		m_pAmp = NULL;
		m_pIndex = NULL;

		m_unitBits = bits;
		m_unitSize = 1ull << bits;

		m_placeID = iPlace;

		m_unitPerPipe = 1;
		m_nPipe = QS_MAX_PIPE;
		m_pipeCount = 0;

		m_pPlaces = NULL;

		m_procPerNode = 1;
	}

	virtual ~QSUnitStorage()
	{
		if(m_pIndex){
			delete[] m_pIndex;
		}
	}

	void Init(int is);

	void SetProcPerNode(int n)
	{
		m_procPerNode = n;
	}

	virtual int Allocate(QSUint numUnits,int nPipe,int numBuffers) = 0;
	virtual void Release(void) = 0;

	virtual QSComplex* Unit(QSUint i)
	{
		return (m_pAmp + i*m_unitSize);
	}
	virtual QSComplex* Buffer(int i)
	{
		return (m_pAmp + ((QSUint)i + (QSUint)m_pipeCount*(QSUint)m_unitPerPipe + m_numUnits)*m_unitSize);
	}

	int& Index(int i)
	{
		return m_pIndex[i];
	}

	int UnitBits(void)
	{
		return m_unitBits;
	}

	QSUint NumUnits(void)
	{
		return m_numUnits;
	}

	int Place(void)
	{
		return m_placeID;
	}

	void SetGlobalUnitIndex(QSUint i)
	{
		m_globalUnitIndex = i;
	}
	void SetNumPlaces(int n)
	{
		m_numPlaces = n;
	}
	void SetPlaces(QSUnitStorage** pP)
	{
		m_pPlaces = pP;
	}

	void InitPipe(int nu)
	{
		m_unitPerPipe = nu;
		m_nPipe = m_nMaxPipe / nu;
		m_pipeCount = 0;
	}
	int Pipe(void)
	{
		return m_pipeCount;
	}
	int PipeLength(void)
	{
		return m_nPipe;
	}
	void AddPipe(void)
	{
		m_pipeCount = ((m_pipeCount + 1) % m_nPipe);
	}

	QSUint GetGlobalUnitIndex(int iUnit)
	{
		return m_globalUnitIndex + (QSUint)m_pIndex[iUnit];
	}
	QSUint GetGlobalUnitIndexBase(void)
	{
		return m_globalUnitIndex;
	}

	
	virtual void SetValue(QSDoubleComplex c,QSUint uid,int pos) = 0;
	virtual void Clear(void) = 0;
	virtual void ClearUnit(QSUint iUnit) = 0;
	virtual void Copy(QSComplex* pV,QSUint iUnit) = 0;


	virtual void Wait(void) = 0;
	virtual void WaitAll(void) = 0;
	virtual void WaitPipe(int iPipe) = 0;


	virtual void ToHost(QSComplex* pDest,QSComplex* pSrc) = 0;
	virtual void WaitToHost(void) = 0;
	virtual int TestToHost(void) = 0;

	virtual void Put(int iBuf,QSComplex* pSrc,int iPlace) = 0;
	virtual void Get(int iBuf,QSComplex* pDest,int iPlace) = 0;
	virtual void GetAck(int iBuf,QSComplex* pDest,int iPlace) = 0;

	virtual void PutOnPipe(int iBuf,QSComplex* pSrc,int iPlace) = 0;
	virtual void GetOnPipe(int iBuf,QSComplex* pDest,int iPlace) = 0;

	virtual void WaitPut(int iPlace) = 0;
	virtual void WaitGet(int iPlace) = 0;
	virtual int TestPut(int iPlace) = 0;
	virtual int TestGet(int iPlace) = 0;

	virtual void SetDevice(void) = 0;
	virtual int GetDeviceID(void) = 0;

	virtual int IsGPU(void) = 0;
	virtual int canCompute(void) = 0;

	//buffers to put parameters for gate operations (used for parameters for GPU)
	virtual QSDoubleComplex* GetMatrixPointer(void)
	{
		return NULL;
	}
	virtual QSComplex** GetBufferPointer(int matSize) = 0;
	virtual int* GetQubitsPointer(void)
	{
		return NULL;
	}
	virtual void* GetStream(void) = 0;
	virtual void* GetStreamPipe(void) = 0;
	virtual double* GetNormPointer(void) = 0;

	//for GPU, synchronize multiple streams for input/output
	virtual void SynchronizeInput(void) = 0;
	virtual void SynchronizeOutput(void) = 0;

};


#endif	//_IBM_Q_SIMULATOR_UNIT_STORAGE_H_

