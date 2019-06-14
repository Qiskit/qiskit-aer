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

	Unit storage on host + file using mmap

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_UNIT_STORAGE_FILE_H_
#define _IBM_Q_SIMULATOR_UNIT_STORAGE_FILE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/mman.h>


#include "QSUnitStorageHost.h"


class QSUnitStorageFile : public QSUnitStorageHost
{
protected:
	QSComplex* m_pAmpFile;
	int m_fd;
	QSUint m_fileSize;
	char* m_filename;
	QSUint m_numUnitsOnMem;
	QSUint m_numUnitsOnFile;
public:
	QSUnitStorageFile()
	{
		m_pAmpFile = NULL;
		m_placeID = -1;
		m_filename = NULL;
		m_numUnitsOnMem = 0;
		m_numUnitsOnFile = 0;
	}
	QSUnitStorageFile(int iPlace,int bits) : QSUnitStorageHost(iPlace,bits)
	{
		m_pAmpFile = NULL;
		m_filename = NULL;
		m_numUnitsOnMem = 0;
		m_numUnitsOnFile = 0;
	}

	virtual ~QSUnitStorageFile()
	{
		Release();
	}

	void SetFilename(char* filename);

	int Allocate(QSUint numUnits,int nPipe,int numBuffers);

	void Release(void);

	void SetValue(QSDoubleComplex c,QSUint uid,int pos);
	void Clear(void);
	void ClearUnit(QSUint iUnit);

	virtual QSComplex* Unit(QSUint i)
	{
		if(i < m_numUnitsOnMem){
			return (m_pAmp + i*m_unitSize);
		}
		else{
			return (m_pAmpFile + (i - m_numUnitsOnMem)*m_unitSize);
		}
	}
	virtual QSComplex* Buffer(int i)
	{
		return (m_pAmp + ((QSUint)i + (QSUint)m_pipeCount*(QSUint)m_unitPerPipe + m_numUnitsOnMem)*m_unitSize);
	}

};


#endif	//_IBM_Q_SIMULATOR_UNIT_STORAGE_FILE_H_
