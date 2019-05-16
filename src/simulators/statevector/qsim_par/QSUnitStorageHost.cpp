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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "QSUnitStorageHost.h"

#ifdef QSIM_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif



int QSUnitStorageHost::Allocate(QSUint numUnits,int numBuffers)
{
	m_numUnits = (QSUint)numUnits;
	m_numBuffer = numBuffers;
	m_numStorage = (QSUint)numUnits + (QSUint)numBuffers;

#ifdef QSIM_CUDA
	if(cudaMallocHost(&m_pAmp,(sizeof(QSComplex)*m_numStorage) << m_unitBits) != cudaSuccess){
		m_pAmp = (QSComplex*)malloc((sizeof(QSComplex)*m_numStorage) << m_unitBits);
	}
#else
	m_pAmp = (QSComplex*)malloc((sizeof(QSComplex)*m_numStorage) << m_unitBits);
#endif

#ifdef QSIM_DEBUG
	printf(" allocating host memory, num storage = %d, %X - %X\n", m_numStorage,m_pAmp,m_pAmp + (m_numStorage << m_unitBits));
#endif

	return numUnits;
}

void QSUnitStorageHost::Release(void)
{
	if(m_pAmp){
#ifdef QSIM_CUDA
		cudaFreeHost(m_pAmp);
#else
		free(m_pAmp);
#endif
		m_pAmp = NULL;
	}
}


void QSUnitStorageHost::SetValue(QSDoubleComplex c,QSUint uid,int pos)
{
	QSReal* pR;
	double* pC = (double*)&c;

	pR = (QSReal*)(m_pAmp + uid*m_unitSize + pos);
	pR[0] = (QSReal)pC[0];
	pR[1] = (QSReal)pC[1];
}


void QSUnitStorageHost::Clear(void)
{
	memset(m_pAmp,0,sizeof(QSComplex)*m_unitSize*(QSUint)m_numUnits);
}


void QSUnitStorageHost::ClearUnit(QSUint iUnit)
{
	memset(m_pAmp + iUnit*m_unitSize,0,sizeof(QSComplex)*m_unitSize);
}

