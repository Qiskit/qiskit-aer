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

	Unit storage on Host + file using mmap

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <sys/time.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "QSUnitStorageFile.h"

#ifdef QSIM_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

void QSUnitStorageFile::SetFilename(char* filename)
{
	if(m_filename != NULL){
		delete[] m_filename;
	}
	m_filename = new char[strlen(filename)+1];
	strcpy(m_filename,filename);
}



int QSUnitStorageFile::Allocate(QSUint numUnits,int nPipe,int numBuffers)
{
	struct sysinfo sinfo;
	QSUint availableMem;

	m_numUnits = (QSUint)numUnits;
	m_numBuffer = numBuffers;
	m_nMaxPipe = nPipe;
	m_numStorage = (QSUint)numUnits + (QSUint)numBuffers;

	m_numUnitsOnMem = 0;
	m_numUnitsOnFile = 0;
	m_pAmp = NULL;
	m_pAmpFile = NULL;

	if(m_numStorage > 0){
		sysinfo(&sinfo);

		//65% of total memory
		availableMem = sinfo.totalram*sinfo.mem_unit/m_procPerNode;
		availableMem = availableMem*65/100;

		if((((QSUint)(numUnits + numBuffers) << m_unitBits) * sizeof(QSComplex)) >= availableMem){

			m_numUnitsOnFile = numUnits - ( (availableMem/sizeof(QSComplex)) >> m_unitBits ) + numBuffers;
		}
		m_numUnitsOnMem = m_numUnits - m_numUnitsOnFile;

#ifdef QSIM_DEBUG
		printf(" available = %ld, %ld units on host memory, %ld units on file , %s\n",availableMem,m_numUnitsOnMem,m_numUnitsOnFile,m_filename);
#endif

#ifdef QSIM_CUDA_
		if(cudaMallocHost(&m_pAmp,(sizeof(QSComplex)*(m_numUnitsOnMem + m_numBuffer)) << m_unitBits) != cudaSuccess){
			m_pAmp = (QSComplex*)malloc((sizeof(QSComplex)*(m_numUnitsOnMem + m_numBuffer)) << m_unitBits);
		}
#else
		m_pAmp = (QSComplex*)malloc((sizeof(QSComplex)*(m_numUnitsOnMem + m_numBuffer)) << m_unitBits);
#endif
	}

	if(m_numUnitsOnFile > 0){
		//allocate rest of units on file
		QSUint size,pagesize;
		char c;

		m_fd = open(m_filename,O_CREAT|O_RDWR,0666);
		if(m_fd == -1){
			printf(" ERROR : Unable to open unit storage on file : %s\n",m_filename);
			return m_numUnitsOnMem;
		}

		pagesize = sysconf(_SC_PAGESIZE);
		m_fileSize = ((sizeof(QSComplex)*(m_numUnitsOnFile << m_unitBits) + pagesize - 1)/pagesize)*pagesize;

		lseek(m_fd,m_fileSize,SEEK_SET);
		read(m_fd,&c,sizeof(char));
		write(m_fd,&c,sizeof(char));
		lseek(m_fd,0,SEEK_SET);

		m_pAmpFile = (QSComplex*)mmap(NULL,m_fileSize,PROT_READ|PROT_WRITE,MAP_SHARED,m_fd,0);
	}

	return numUnits;
}

void QSUnitStorageFile::Release(void)
{
	if(m_pAmp){
#ifdef QSIM_CUDA_
		cudaFreeHost(m_pAmp);
#else
		free(m_pAmp);
#endif
		m_pAmp = NULL;
	}

	if(m_pAmpFile){
		munmap(m_pAmpFile,m_fileSize);
		close(m_fd);
	}
}


void QSUnitStorageFile::SetValue(QSDoubleComplex c,QSUint uid,int pos)
{
	QSReal* pR;
	double* pC = (double*)&c;

	if(m_numStorage > 0){
		if(uid < m_numUnitsOnMem){
			pR = (QSReal*)(m_pAmp + uid*m_unitSize + pos);
		}
		else{
			pR = (QSReal*)(m_pAmpFile + (uid - m_numUnitsOnMem)*m_unitSize + pos);
		}
		pR[0] = (QSReal)pC[0];
		pR[1] = (QSReal)pC[1];
	}
}


void QSUnitStorageFile::Clear(void)
{
	if(m_numStorage > 0){
		memset(m_pAmp,0,sizeof(QSComplex)*m_unitSize*(QSUint)m_numUnitsOnMem);
		if(m_pAmpFile){
			memset(m_pAmpFile,0,sizeof(QSComplex)*m_unitSize*(QSUint)m_numUnitsOnFile);
		}
	}
}


void QSUnitStorageFile::ClearUnit(QSUint iUnit)
{
	if(m_numStorage > 0){
		if(iUnit < m_numUnitsOnMem){
			memset(m_pAmp + iUnit*m_unitSize,0,sizeof(QSComplex)*m_unitSize);
		}
		else{
			memset(m_pAmpFile + (iUnit - m_numUnitsOnMem)*m_unitSize,0,sizeof(QSComplex)*m_unitSize);
		}
	}
}

