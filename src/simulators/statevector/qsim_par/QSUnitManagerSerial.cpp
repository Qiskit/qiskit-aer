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

	Unit manager for serial execution

	2018-2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <sys/sysinfo.h>

#include "QSUnitManagerSerial.h"

#include "QSUnitStorageHost.h"
#include "QSUnitStorageFile.h"

#ifdef QSIM_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include "QSUnitStorageGPU.h"

#endif

#include <omp.h>

#include "QSGate_MatMult.h"
#include "QSGate_DiagMult.h"
#include "QSGate_CX.h"
#include "QSGate_X.h"
#include "QSGate_Y.h"
#include "QSGate_Z.h"
#include "QSGate_Dot.h"
#include "QSGate_MultiShot.h"


void QSUnitManagerSerial::Init(void)
{
	QSUint i,n,ret,is,nu,offset;
	int iDev,isDev,ieDev;
	int nDev = 0;
	char* buf;
	int useGPU = 1;
	QSUint freeMemSize,size;
	QSUint guidBase;
	int testHybrid = 0,testMmap = 0;
	double tStart,tEnd;
	int nPipe;

	struct sysinfo sinfo;

	m_procBits = 0;
	m_isPowerOf2 = 1;
	m_nprocs_per_node = 1;
	m_iproc_per_node = 0;

	m_unitBits = m_globalBits;
	m_unitSize = 1ull << m_unitBits;
	m_numGlobalUnits = m_numUnits = 1;

	m_globalUnitIndex = 0;

	nPipe = m_numBuffers;
	m_numBuffers = 0;

	m_numPlaces = 0;
	n = 1;
	is = 0;

	m_pUnits = new QSUnitStorage*[2];
	m_pCountPlace = new int[2];
	m_pNormBuf = new QSReal[2];

#ifdef QSIM_CUDA
	buf = getenv("QSIM_EXEC_ON_GPU");
	if(buf){
		m_executeAllOnGPU = 1;
	}

	cudaGetDeviceCount(&nDev);

	if(nDev > 0){
		size_t freeMem,totalMem;
		QSUint localSize;

		m_numGPU = 1;

		//allocate on GPUs
		int nid = omp_get_thread_num();
		iDev = 0;
		ret = 0;
		while(iDev < nDev){
			QSUnitStorageGPU* pQSG;
			pQSG = new QSUnitStorageGPU( ((iDev + nid)%nDev),m_unitBits,0);
			pQSG->SetProcPerNode(1);
			pQSG->SetNumPlaces(1);
			pQSG->SetGlobalUnitIndex(0);

			ret = pQSG->Allocate(m_numUnits,nPipe,m_numBuffers);
			if(ret < m_numUnits){
				delete pQSG;
			}
			else{
				m_pUnits[m_numPlaces] = pQSG;
				break;
			}
			iDev++;
		}
		n -= ret;
		is += ret;

		if(ret == 0){
			m_numGPU = 0;
		}
		else{
			m_numPlaces++;
		}
	}
#endif

	//allocate units on Host
	QSUnitStorageHost* pHost;
	m_iPlaceHost = m_numPlaces;

	buf = getenv("QSIM_USE_FILE");
	if(buf != NULL){
		QSUnitStorageFile* pHostFile;
		char* filename = new char[strlen(buf) + 32];

		pHost = pHostFile = new QSUnitStorageFile(m_iPlaceHost,m_unitBits);
		sprintf(filename,"%s/qsunits_%d",buf,m_myrank);
		pHostFile->SetFilename(filename);
		delete[] filename;
	}
	else{
		pHost = new QSUnitStorageHost(m_iPlaceHost,m_unitBits);
	}
	pHost->SetProcPerNode(1);

	m_pUnits[m_iPlaceHost] = pHost;
	pHost->SetNumPlaces(m_numPlaces+1);
	pHost->SetGlobalUnitIndex(0);
	pHost->Allocate(n,nPipe,m_numBuffers);
	pHost->Init(is);
	m_numPlaces++;

	for(i=0;i<m_numPlaces;i++){
		m_pUnits[i]->SetPlaces(m_pUnits);
		m_pUnits[i]->SetNumPlaces(m_numPlaces);		//correct number of places

		m_pUnits[i]->InitPipe(1);
	}

#ifdef QSIM_DEBUG
	//test
	printf(" ================================= \n");
	printf("  Serial execution\n" );
	printf("  Places = %d, host = %d\n",m_numPlaces,m_iPlaceHost);
	for(i=0;i<m_numPlaces;i++){
		printf("     place[%d] : %d units ",i,m_pUnits[i]->NumUnits());
		if(m_pUnits[i]->IsGPU()){
			printf(" on GPU\n");
		}
		else{
			printf(" on Host\n");
		}
	}
	if(m_executeAllOnGPU){
		printf("  gate calculation on GPU only\n");
	}
	printf(" ================================= \n");
#endif

}



void QSUnitManagerSerial::SetValue(QSDoubleComplex c,QSUint gid)
{
	m_pUnits[0]->SetValue(c,0,gid);
}

void QSUnitManagerSerial::Clear(void)
{
	int i;
	for(i=0;i<m_numPlaces;i++){
		m_pUnits[i]->Clear();
	}
}

void QSUnitManagerSerial::Copy(QSComplex* pV)
{
	m_pUnits[0]->Copy(pV,0);
	m_pUnits[0]->Wait();
}


QSDouble QSUnitManagerSerial::Dot(int qubit)
{
	QSGate_Dot dotGate;
	int i;
	QSDouble ret = 0.0;


#ifdef QSIM_CUDA
	if(m_pUnits[0]->IsGPU()){
		dotGate.InitBuffer(m_pUnits[0]);
	}
#endif

	ExecuteGate(&dotGate,&qubit,&qubit,1);

	for(i=0;i<m_numPlaces;i++){
		m_pUnits[i]->WaitPipe(0);
	}

	ret = dotGate.Result();

#ifdef QSIM_CUDA
	//reduce buffers
	if(m_pUnits[0]->IsGPU()){
		ret += dotGate.ReduceAll(m_pUnits[0]);
	}
#endif

#ifdef QSIM_DEBUG
	if(m_myrank == 0){
		printf("Dot : %d  - %e\n",qubit,ret);
	}
#endif

	return ret;
}


void QSUnitManagerSerial::Measure(int qubit,int flg,QSDouble norm)
{
	double mat[4];
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i;
	QSGate_DiagMult matMultGate;

	if(flg == 0){
		mat[0] = norm;
		mat[1] = 0.0;
		mat[2] = 0.0;
		mat[3] = 0.0;
	}
	else{
		mat[0] = 0.0;
		mat[1] = 0.0;
		mat[2] = norm;
		mat[3] = 0.0;
	}

	qubits_c[0] = -1;

	matMultGate.SetMatrix((QSDoubleComplex*)mat,2);

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
	if(m_pUnits[0]->IsGPU()){
		matMultGate.CopyMatrix(m_pUnits[0],&qubit,1,0);
	}
#endif

	ExecuteGate(&matMultGate,&qubit,qubits_c,1);
}



/*-------------------------------------------------------------
	multiplication of NxN matrix and N amplitudes vector
--------------------------------------------------------------*/
void QSUnitManagerSerial::MatMult(QSDoubleComplex* pM,int* qubits,int nqubits)
{
	QSGate_MatMult matMultGate;
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i,matSize;

	qubits_c[0] = -1;

	matSize = 1 << nqubits;

	matMultGate.SetMatrix(pM,matSize);

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
	if(m_pUnits[0]->IsGPU()){
		matMultGate.CopyMatrix(m_pUnits[0],qubits,nqubits,0);
	}
#endif

	ExecuteGate(&matMultGate,qubits,qubits_c,nqubits);

}


/*-------------------------------------------------------------
	multiplication of Diagonal NxN matrix and N amplitudes vector
--------------------------------------------------------------*/
void QSUnitManagerSerial::MatMultDiagonal(QSDoubleComplex* pM,int* qubits,int nqubits)
{
	QSGate_DiagMult matMultGate;
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i,matSize;

	qubits_c[0] = -1;

	matMultGate.SetMatrix(pM,(1 << nqubits));

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
	if(m_pUnits[0]->IsGPU()){
		matMultGate.CopyMatrix(m_pUnits[0],qubits,nqubits,0);
	}
#endif

	ExecuteGate(&matMultGate,qubits,qubits_c,nqubits);
}

/*-------------------------------------------------------------
	Controlled X gate
--------------------------------------------------------------*/
void QSUnitManagerSerial::CX(int qubit_t,int qubit_c)
{
	QSGate_CX cxGate;

	ExecuteGate(&cxGate,&qubit_t,&qubit_c,1);
}

/*-------------------------------------------------------------
	U1 gate
--------------------------------------------------------------*/
void QSUnitManagerSerial::U1(int qubit,QSDouble* pPhase)
{
	QSDouble mat[4];
	mat[0] = 1.0;
	mat[1] = 0.0;
	mat[2] = pPhase[0];
	mat[3] = pPhase[1];
	QSGate_DiagMult matMultGate;

	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i;

	qubits_c[0] = -1;

	matMultGate.SetMatrix((QSDoubleComplex*)mat,2);

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
	if(m_pUnits[0]->IsGPU()){
		matMultGate.CopyMatrix(m_pUnits[0],&qubit,1,0);
	}
#endif

	ExecuteGate(&matMultGate,&qubit,qubits_c,1);
}

/*-------------------------------------------------------------
	gate handler
--------------------------------------------------------------*/
void QSUnitManagerSerial::ExecuteGate(QSGate* pGate,int* qubits,int* qubits_c,int nqubits)
{
	QSUint guid;
	QSComplex* pBuf;
	int iPlaceExec;

#ifdef QSIM_CUDA
	//on GPU
	if(m_pUnits[0]->IsGPU()){
		guid = 0;
		pBuf = m_pUnits[0]->Unit(0);

		pGate->ExecuteOnGPU(m_pUnits[0],&guid,&pBuf,qubits,qubits_c,nqubits,0,1,0);
	}
#endif

	//on host
	if(m_pUnits[m_iPlaceHost]->NumUnits() > 0){
		iPlaceExec = m_iPlaceHost;
		guid = 0;
		pBuf = m_pUnits[iPlaceExec]->Unit(0);
		pGate->ExecuteOnHost(m_pUnits[iPlaceExec],&guid,&pBuf,qubits,qubits_c,nqubits,0,1,0);
	}
}



/*-------------------------------------------------------------
	Multi-shot optimization
--------------------------------------------------------------*/
void QSUnitManagerSerial::Measure_FindPos(QSDouble* rs,QSUint* ret,int ns)
{
	QSGate_MultiShot msGate;
	QSDouble* pProcTotal;
	int* ranks;
	int i,is,qubit = 0;
	QSDouble t;
	int iPlace;
	QSUint iUnit;

	QSUint guid = 0;
	QSComplex* pBuf;

	for(is=0;is<ns;is++){
		msGate.SetKey(rs[is]);

		pBuf = m_pUnits[0]->Unit(0);
		msGate.Execute(m_pUnits[0],&guid,&pBuf,&qubit,&qubit,1,0,1,0);

		ret[is] = msGate.Pos();
	}
}



