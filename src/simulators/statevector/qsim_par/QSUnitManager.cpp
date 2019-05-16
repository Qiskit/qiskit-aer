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

	Unit manager

	2018-2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <sys/sysinfo.h>

#include "QSUnitManager.h"

#include "QSUnitStorageHost.h"

#ifdef QSIM_MPI
#include <mpi.h>
#endif

#ifdef QSIM_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include "QSUnitStorageGPU.h"

#endif

#include <sys/time.h>


#include "QSGate_MatMult.h"
#include "QSGate_DiagMult.h"
#include "QSGate_CX.h"
#include "QSGate_X.h"
#include "QSGate_Y.h"
#include "QSGate_Dot.h"
#include "QSGate_MultiShot.h"


static double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


void QSUnitManager::Init(void)
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
#ifdef QSIM_MPI
	MPI_Status st;
#endif

	struct sysinfo sinfo;

	tStart = mysecond();

#ifdef QSIM_MPI
	MPI_Comm_size(MPI_COMM_WORLD,&m_nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&m_myrank);

	m_pReqSend = new MPI_Request[QS_MAX_MATRIX_SIZE*m_numBuffers];
	m_pReqRecv = new MPI_Request[m_numBuffers];
#else
	m_nprocs = 1;
	m_myrank = 0;
#endif

	m_procBits = 0;
	n = m_nprocs;
	while(n > 1){
		n = n >> 1;
		m_procBits++;
	}
	m_isPowerOf2 = 0;
	if((1 << m_procBits) == m_nprocs){
		m_isPowerOf2 = 1;
		m_procBits = m_globalBits - m_procBits;
	}


	m_nprocs_per_node = 1;
	buf = getenv("QSIM_PROC_PER_NODE");
	if(buf){
		m_nprocs_per_node = atoi(buf);
		if(m_nprocs_per_node > m_nprocs){
			m_nprocs_per_node = m_nprocs;
		}
	}
	m_iproc_per_node = m_myrank % m_nprocs_per_node;

	//test
	buf = getenv("QSIM_TEST_HYBRID");
	if(buf){
		testHybrid = 1;
	}

	//--- adjust unit bits ---
	buf = getenv("QSIM_UNIT_BITS");
	if(buf){
		i = atoi(buf);
		if(i > 0){
			m_unitBits = i;
		}
	}
	if(m_unitBits > 25){
		m_unitBits = 25;
	}
	if(m_unitBits > m_globalBits){
		m_unitBits = m_globalBits;
	}
	while((1ull << (m_globalBits - m_unitBits)) < m_nprocs && m_unitBits > 0){
		m_unitBits--;
	}
	m_unitSize = 1ull << m_unitBits;
	m_numGlobalUnits = 1ull << (m_globalBits - m_unitBits);
	//------


	m_globalUnitIndex = m_myrank*m_numGlobalUnits/m_nprocs;
	m_numUnits = ((m_myrank+1)*m_numGlobalUnits/m_nprocs) - m_globalUnitIndex;

	m_pUStart = new QSUint[m_nprocs];
	m_pUEnd = new QSUint[m_nprocs];
	m_pUFile = new QSUint[m_nprocs];
	m_pOffsetFile = new QSUint[m_nprocs];

	m_pProcIndex = new QSUint[m_nprocs];
	m_pProcMap = new QSUint[m_nprocs];

#pragma omp for
	for(i=0;i<m_nprocs;i++){
		m_pUStart[i] = i*m_numGlobalUnits/m_nprocs;
		m_pUEnd[i] = (i+1)*m_numGlobalUnits/m_nprocs;
		m_pProcIndex[i] = i;
		m_pProcMap[i] = i;
	}

	m_pUnitTable = new QSUint[m_numUnits];
	m_pPlaceTable = new int8_t[m_numUnits];

	m_pGuid_Pipe = new QSUint[m_numBuffers];
	m_pLocalMask_Pipe = new QSUint[m_numBuffers];
	m_pPlaceExec_Pipe = new int[m_numBuffers];
	m_pPlace_Pipe = new int[m_numBuffers];
	m_nTrans_Pipe = new int[m_numBuffers];
	m_nDest_Pipe = new int[m_numBuffers];
	m_pBuf_Pipe = new QSComplex*[m_numBuffers];
	m_pSrc_Pipe = new QSComplex*[m_numBuffers];
	m_flgSend_Pipe = new QSUint[m_numBuffers];
	m_flgRecv_Pipe = new QSUint[m_numBuffers];

	m_numPlaces = 0;
	n = m_numUnits;
	is = 0;

#ifdef QSIM_CUDA
	buf = getenv("QSIM_HOST_ONLY");
	if(buf){
		useGPU = 0;
	}

	if(useGPU){
		buf = getenv("QSIM_EXEC_ON_GPU");
		if(buf){
			m_executeAllOnGPU = 1;
		}

		cudaGetDeviceCount(&nDev);

		isDev = m_iproc_per_node *nDev / m_nprocs_per_node;
		ieDev = (m_iproc_per_node+1) *nDev / m_nprocs_per_node;

		m_numGPU = ieDev - isDev;

//		for(iDev=isDev;iDev<ieDev;iDev++){
//			size_t freeMem,totalMem;
//			cudaSetDevice(iDev);
//			cudaMemGetInfo(&freeMem,&totalMem);		//why this is needed?
//		}

		m_pUnits = new QSUnitStorage*[m_numGPU+2];
		m_pCountPlace = new int[m_numGPU+2];
		m_pNormBuf = new QSReal[m_numGPU+2];

		//allocate units on GPUs
#pragma omp parallel for private(iDev,nu)
		for(iDev=0;iDev<m_numGPU;iDev++){
			nu = ((iDev+1)*m_numUnits/m_numGPU) - (iDev*m_numUnits/m_numGPU);
			m_pUnits[m_numPlaces + iDev] = new QSUnitStorageGPU(iDev,m_unitBits,isDev);

			m_pUnits[m_numPlaces + iDev]->SetNumPlaces(m_numGPU + 2);	//temporary set numplaces + 2 for host + file
			m_pUnits[m_numPlaces + iDev]->SetGlobalUnitIndex(m_globalUnitIndex);

			//to debug hybrid mode use below
			if(testHybrid){
				nu/=2;
			}

			m_pUnits[m_numPlaces + iDev]->Allocate(nu,m_numBuffers);
			m_pUnits[m_numPlaces + iDev]->Clear();
		}

		for(iDev=0;iDev<m_numGPU;iDev++){
			m_pUnits[m_numPlaces]->Init(is);
			ret = m_pUnits[m_numPlaces]->NumUnits();

			n -= ret;

			//set tables
#pragma omp parallel for
			for(i=0;i<ret;i++){
				m_pUnitTable[is+i] = i;
				m_pPlaceTable[is+i] = m_numPlaces;
			}
			is += ret;
			m_numPlaces++;
		}

		if(n == 0){
			m_executeAllOnGPU = 1;
		}
	}
	else{
#endif
		m_pUnits = new QSUnitStorage*[2];
		m_pCountPlace = new int[2];
		m_pNormBuf = new QSReal[2];
#ifdef QSIM_CUDA
	}
#endif


	sysinfo(&sinfo);

	size = ((QSUint)(n + m_numBuffers) << m_unitBits) * sizeof(QSComplex);	//size to allocate units on host memory

	m_numUnitsOnFile = 0;

	//allocate units on Host
	QSUnitStorageHost* pHost;
	m_iPlaceHost = m_numPlaces;

	pHost = new QSUnitStorageHost(m_iPlaceHost,m_unitBits);
	m_pUnits[m_iPlaceHost] = pHost;
	pHost->SetNumPlaces(m_numPlaces+2);
	pHost->SetGlobalUnitIndex(m_globalUnitIndex);
	pHost->Allocate(n - m_numUnitsOnFile,m_numBuffers);
	pHost->Init(is);
	if(n - m_numUnitsOnFile > 0){
		pHost->Clear();
#pragma omp parallel for
		for(i=0;i<n - m_numUnitsOnFile;i++){
			m_pUnitTable[is + i] = i;
			m_pPlaceTable[is + i] = m_iPlaceHost;
		}
		is += n - m_numUnitsOnFile;
	}
	m_numPlaces++;

	m_numGlobalFile = 0;
	m_pOffsetFile[0] = 0;
	m_numGlobalFile = 0;

	for(i=0;i<m_numPlaces;i++){
		m_pUnits[i]->SetPlaces(m_pUnits);
		m_pUnits[i]->SetNumPlaces(m_numPlaces);		//correct number of places
	}

#ifdef QSIM_DEBUG
	//test
	printf(" ================================= \n");
	printf("  [%d] unitBits = %d, localUnits = %d, global Units = %d \n",m_myrank,m_unitBits,m_numUnits,m_numGlobalUnits);
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
	if(m_numUnitsOnFile > 0){
		printf("     On File   : %d units\n",m_numUnitsOnFile);
	}
	if(m_executeAllOnGPU){
		printf("  gate calculation on GPU only\n");
	}
	printf(" ================================= \n");
#endif

	TimeReset();

	tEnd = mysecond();
#ifdef QSIM_DEBUG
	printf("    Initialization time = %f sec\n",tEnd - tStart);
#endif
}



void QSUnitManager::SetValue(QSDoubleComplex c,QSUint gid)
{
	QSUint guid,luid,lid;
	int ip,iu;

	guid = gid >> m_unitBits;
	if(guid >= m_globalUnitIndex && guid < m_globalUnitIndex + m_numUnits){
		luid = guid - m_globalUnitIndex;
		lid = gid - (guid << m_unitBits);

		ip = m_pPlaceTable[luid];
		iu = m_pUnitTable[luid];
		m_pUnits[ip]->SetValue(c,iu,lid);
	}
}

void QSUnitManager::Clear(void)
{
	int i;
	for(i=0;i<m_numPlaces;i++){
		m_pUnits[i]->Clear();
	}
}



int QSUnitManager::GetProcess(QSUint ui)
{
	int i;

	for(i=0;i<m_nprocs;i++){
		if(m_pUStart[i] <= ui && ui < m_pUEnd[i]){
			return m_pProcMap[i];
		}
	}

	return -1;
}

void QSUnitManager::SortProcs(int* pProcs,int n)
{
	int i,j,t;

	for(i=0;i<n;i++){
		for(j=n-1;j>i;j--){
			if(pProcs[j] < pProcs[j-1]){
				t = pProcs[j-1];
				pProcs[j-1] = pProcs[j];
				pProcs[j] = t;
			}
		}
	}
}


QSDouble QSUnitManager::Dot(int qubit)
{
	QSGate_Dot dotGate;
	int i;
	QSDouble ret = 0.0;

#ifdef QSIM_CUDA

	//Init buffers
#pragma omp parallel for
	for(i=0;i<m_numPlaces;i++){
		if(m_pUnits[i]->IsGPU()){
			dotGate.InitBuffer(m_pUnits[i]);
		}
	}
#endif

	ExecuteGate(&dotGate,&qubit,&qubit,1);

	ret = dotGate.Result();

#ifdef QSIM_CUDA

	//reduce buffers
#pragma omp parallel for reduction(+:ret)
	for(i=0;i<m_numPlaces;i++){
		if(m_pUnits[i]->IsGPU()){
			ret += dotGate.ReduceAll(m_pUnits[i]);
		}
	}
#endif

#ifdef QSIM_MPI
	double sum = ret;
	MPI_Allreduce(&sum,&ret,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
#endif

#ifdef QSIM_DEBUG
	if(m_myrank == 0){
		printf("Dot : %d  - %e\n",qubit,ret);
	}
#endif
	
	return ret;
}


void QSUnitManager::Measure(int qubit,int flg,QSDouble norm)
{
	double mat[4];
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i;
	QSGate_DiagMult matMultGate((QSDoubleComplex*)mat);

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

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
#pragma omp parallel for
	for(i=0;i<m_numPlaces;i++){
		if(m_pUnits[i]->IsGPU()){
			matMultGate.CopyMatrix(m_pUnits[i],&qubit,1);
		}
	}
#endif

	ExecuteGate(&matMultGate,&qubit,qubits_c,1);
}



/*-------------------------------------------------------------
	multiplication of NxN matrix and N amplitudes vector
--------------------------------------------------------------*/
void QSUnitManager::MatMult(QSDoubleComplex* pM,int* qubits,int nqubits)
{
	QSGate_MatMult matMultGate(pM);
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i,matSize;
	QSDoubleComplex mt[32*32];

	qubits_c[0] = -1;

	matSize = 1 << nqubits;


#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
#pragma omp parallel for
	for(i=0;i<m_numPlaces;i++){
		if(m_pUnits[i]->IsGPU()){
			matMultGate.CopyMatrix(m_pUnits[i],qubits,nqubits);
		}
	}
#endif

	ExecuteGate(&matMultGate,qubits,qubits_c,nqubits);

}


/*-------------------------------------------------------------
	multiplication of Diagonal NxN matrix and N amplitudes vector
--------------------------------------------------------------*/
void QSUnitManager::MatMultDiagonal(QSDoubleComplex* pM,int* qubits,int nqubits)
{
	QSGate_DiagMult matMultGate(pM);
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i,matSize;

	qubits_c[0] = -1;

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
#pragma omp parallel for
	for(i=0;i<m_numPlaces;i++){
		if(m_pUnits[i]->IsGPU()){
			matMultGate.CopyMatrix(m_pUnits[i],qubits,nqubits);
		}
	}
#endif

	ExecuteGate(&matMultGate,qubits,qubits_c,nqubits);
}

/*-------------------------------------------------------------
	Controlled X gate
--------------------------------------------------------------*/
void QSUnitManager::CX(int qubit_t,int qubit_c)
{
	QSGate_CX cxGate;
	QSUint i,i0,i1,add,mask,t,myID,pairID;
	int8_t tp;
	

	if(m_isPowerOf2){
		if(qubit_t >= m_unitBits && qubit_c >= m_unitBits){
			if(qubit_t < m_procBits){
				add = 1ull << (qubit_t - m_unitBits);
				mask = 1ull << (qubit_c - m_unitBits);

#pragma omp parallel for private(i,i0,i1,t,tp)
				for(i=0;i<m_numUnits/2;i++){
					i0 = i & (add - 1);
					i0 += ((i - i0) << 1);
					i1 = i0 + add;

					if(((i0 + m_globalUnitIndex) & mask) != 0){
						//swap table

						if(m_pPlaceTable[i0] < m_numPlaces)
							m_pUnits[m_pPlaceTable[i0]]->Index(m_pUnitTable[i0]) = i1;
						if(m_pPlaceTable[i1] < m_numPlaces)
							m_pUnits[m_pPlaceTable[i1]]->Index(m_pUnitTable[i1]) = i0;

						t = m_pUnitTable[i0];
						m_pUnitTable[i0] = m_pUnitTable[i1];
						m_pUnitTable[i1] = t;

						tp = m_pPlaceTable[i0];
						m_pPlaceTable[i0] = m_pPlaceTable[i1];
						m_pPlaceTable[i1] = tp;
					}
				}
				return;
			}
			else if(qubit_c >= m_procBits){
				add = 1ull << (qubit_t - m_procBits);
				mask = 1ull << (qubit_c - m_procBits);
#pragma omp parallel for private(i,i0,i1,myID,pairID)
				for(i=0;i<m_nprocs;i++){
					i0 = i;
					myID = m_pProcIndex[i];
					if((myID & mask) != 0){
						pairID = myID ^ add;
						i1 = m_pProcMap[pairID];

						if(i0 < i1){
							//exchange table
							m_pProcIndex[i0] = pairID;
							m_pProcIndex[i1] = myID;
							m_pProcMap[myID] = i1;
							m_pProcMap[pairID] = i0;
						}
					}
				}

				//update global index
				m_globalUnitIndex = m_pUStart[m_pProcIndex[m_myrank]];
				for(i=0;i<m_numPlaces;i++){
					m_pUnits[i]->SetGlobalUnitIndex(m_globalUnitIndex);
				}
				return;
			}
		}
	}


	ExecuteGate(&cxGate,&qubit_t,&qubit_c,1);
}

/*-------------------------------------------------------------
	U1 gate
--------------------------------------------------------------*/
void QSUnitManager::U1(int qubit,QSDouble* pPhase)
{
	QSDouble mat[4];
	mat[0] = 1.0;
	mat[1] = 0.0;
	mat[2] = pPhase[0];
	mat[3] = pPhase[1];
	QSGate_DiagMult matMultGate((QSDoubleComplex*)mat);

	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy
	int i;

	qubits_c[0] = -1;

#ifdef QSIM_CUDA

	//copy matrix and qubits on GPUs
#pragma omp parallel for
	for(i=0;i<m_numPlaces;i++){
		if(m_pUnits[i]->IsGPU()){
			matMultGate.CopyMatrix(m_pUnits[i],&qubit,1);
		}
	}
#endif

	ExecuteGate(&matMultGate,&qubit,qubits_c,1);
}


/*-------------------------------------------------------------
	X gate
--------------------------------------------------------------*/
void QSUnitManager::X(int qubit)
{
	QSGate_X xGate;
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy

	qubits_c[0] = -1;

	ExecuteGate(&xGate,&qubit,qubits_c,1);
}

/*-------------------------------------------------------------
	Y gate
--------------------------------------------------------------*/
void QSUnitManager::Y(int qubit)
{
	QSGate_Y yGate;
	int qubits_c[QS_MAX_MATRIX_SIZE];	//dummy

	qubits_c[0] = -1;

	ExecuteGate(&yGate,&qubit,qubits_c,1);
}


/*-------------------------------------------------------------
	gate handler
--------------------------------------------------------------*/
void QSUnitManager::ExecuteGate(QSGate* pGate,int* qubits,int* qubits_c,int nqubits)
{
	QSUint guid[QS_MAX_MATRIX_SIZE];
	QSUint luid;
	QSUint iu[QS_MAX_FUSION+1];
	QSUint nu[QS_MAX_FUSION+1];
	QSUint ub[QS_MAX_FUSION+1];
	QSUint iAdd[QS_MAX_FUSION];
	int iPlace[QS_MAX_MATRIX_SIZE];
	int iproc[QS_MAX_MATRIX_SIZE];
	int destProcs[QS_MAX_MATRIX_SIZE];
	int fileProcs[QS_MAX_MATRIX_SIZE];
	QSComplex* pBuf[QS_MAX_MATRIX_SIZE];
	QSComplex* pSrcBuf[QS_MAX_MATRIX_SIZE];
	QSComplex* pRemoteBuf[QS_MAX_MATRIX_SIZE];
	QSUint iPair,guidBase;
	int j,k,l,iUnit;
	int nLarge,nTrans,nDest,add,iPipe,nPipe,iPipeWait,flg;
	QSUint nPair,nUnit,localMask;
	int* pUnitsPerPlace;
	int iPlaceExec,ip,iDev;
	QSUint i,isu,ieu;
	int countAdditional = 0;
	int nUnitOnFile,nProcFile;
	int nLocal;
	QSUint offset;
	int nFile,wbFile;
	QSUint myGuid;
	QSUint uMask;
	QSUint nuPlace;
	int iFirstUnitGPU;
#ifdef QSIM_MPI
	MPI_Status st;
#endif


	nLarge = 0;
	for(i=0;i<nqubits;i++){
		if(qubits[i] >= m_unitBits){
			iAdd[nLarge] = 1ull << (qubits[i] - m_unitBits);
			nLarge++;
		}
	}
	iAdd[nLarge] = 0;

	nPair = 1ull << (m_globalBits - m_unitBits - nLarge);	//number of pairs of units
	nUnit = 1ull << nLarge;									//number of units per pair

	//initialize pipeline counter
	for(i=0;i<m_numPlaces;i++){
		m_pUnits[i]->InitPipe(nUnit);
	}
	nPipe = m_pUnits[m_iPlaceHost]->PipeLength();

#ifdef QSIM_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if(nLarge == 0 || pGate->ExchangeNeeded() == 0){		//local calculations inside units, no data exchange needed
#ifdef QSIM_CUDA

		//on GPU
#pragma omp parallel for private(nUnit,i,j,pBuf,guid)
		for(iPlaceExec=0;iPlaceExec<m_numPlaces;iPlaceExec++){
			if(m_pUnits[iPlaceExec]->IsGPU()){
				nUnit = m_pUnits[iPlaceExec]->NumUnits();
				for(i=0;i<nUnit;i++){
					guid[0] = m_pUnits[iPlaceExec]->GetGlobalUnitIndex(i);
					if(qubits_c[0] >= m_unitBits){		//currently only top control bit is used, set qubits_c[0] to -1 if not used
						if(((guid[0] >> (qubits_c[0] - m_unitBits)) & 1) != pGate->ControlMask()){
							continue;
						}
					}

					pBuf[0] = m_pUnits[iPlaceExec]->Unit(i);
					pGate->ExecuteOnGPU(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,0,(1 << (nqubits+1))-1,0);
				}
			}
		}
#endif

		//on host
		if(m_executeAllOnGPU){
			nUnit = m_pUnits[m_iPlaceHost]->NumUnits();

#pragma omp parallel for private(isu,ieu,i,j,pBuf,pSrcBuf,guid,iPipe,iDev,iPlaceExec)
			for(iDev=0;iDev<m_numGPU;iDev++){
				isu = (iDev)*nUnit/(m_numGPU);
				ieu = (iDev+1)*nUnit/(m_numGPU);
				j = 0;
				for(i=0;i<m_numPlaces;i++){
					if(m_pUnits[i]->IsGPU()){
						if(j == iDev){
							iPlaceExec = i;
							break;
						}
						j++;
					}
				}

				for(i=isu;i<ieu;i++){
					guid[0] = m_pUnits[m_iPlaceHost]->GetGlobalUnitIndex(i);
					if(qubits_c[0] >= m_unitBits){		//currently only top control bit is used, set qubits_c[0] to -1 if not used
						if(((guid[0] >> (qubits_c[0] - m_unitBits)) & 1) != pGate->ControlMask()){
							continue;
						}
					}
					pBuf[0] = m_pUnits[iPlaceExec]->Buffer(0);
					pSrcBuf[0] = m_pUnits[m_iPlaceHost]->Unit(i);

					iPipe = m_pUnits[iPlaceExec]->Pipe();	//to keep current pipe

					//synchronize pipeline in case previous stream is still running
					m_pUnits[iPlaceExec]->WaitPipe(iPipe);

					//copy to GPU
					m_pUnits[iPlaceExec]->Put(0,pSrcBuf[0],m_iPlaceHost);
					//execute on GPU
					pGate->ExecuteOnGPU(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,0,(1 << (nqubits+1))-1,1);
					//copy back to host
					m_pUnits[iPlaceExec]->Get(0,pSrcBuf[0],m_iPlaceHost);

					m_pUnits[iPlaceExec]->AddPipe();
				}
			}
		}
		else{
			iPlaceExec = m_iPlaceHost;
			nUnit = m_pUnits[iPlaceExec]->NumUnits();
			for(i=0;i<nUnit;i++){
				guid[0] = m_pUnits[iPlaceExec]->GetGlobalUnitIndex(i);
				if(qubits_c[0] >= m_unitBits){		//currently only top control bit is used, set qubits_c[0] to -1 if not used
					if(((guid[0] >> (qubits_c[0] - m_unitBits)) & 1) != pGate->ControlMask()){
						continue;
					}
				}

				pBuf[0] = m_pUnits[iPlaceExec]->Unit(i);
				pGate->ExecuteOnHost(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,0,(1 << (nqubits+1))-1,0);
			}
		}

		if(m_executeAllOnGPU && m_pUnits[m_iPlaceHost]->NumUnits() > 0){
			for(j=0;j<m_numPlaces;j++){
				m_pUnits[j]->WaitAll();
			}
		}
		else{
			for(j=0;j<m_numPlaces;j++){
				m_pUnits[j]->WaitPipe(m_pUnits[j]->Pipe());
			}
		}
	}
	else if(m_numGlobalFile == 0 && m_isPowerOf2 && qubits[nqubits-1] < m_procBits){
		//on GPU
#pragma omp parallel for private(myGuid,uMask,iPlaceExec,nuPlace,iFirstUnitGPU,i,k,iUnit,localMask,nLocal,guidBase,guid,luid,iPlace,pSrcBuf,pBuf,nTrans,iPipe)
		for(iPlaceExec=0;iPlaceExec<m_numPlaces;iPlaceExec++){
			if(m_pUnits[iPlaceExec]->IsGPU()){
				nuPlace = m_pUnits[iPlaceExec]->NumUnits();
				for(i=0;i<nuPlace;i++){
					myGuid = m_pUnits[iPlaceExec]->GetGlobalUnitIndex(i);
					guidBase = myGuid;
					for(k=0;k<nLarge;k++){
						guidBase &= (~iAdd[k]);
					}

					if(qubits_c[0] >= m_unitBits){		//currently only CX gate uses top control bit, for other gate set qubits_c[0] to -1
						if(((guidBase >> (qubits_c[0] - m_unitBits)) & 1) != pGate->ControlMask()){
							continue;	//not a controled unit, skip
						}
					}
					localMask = 0;
					iFirstUnitGPU = -1;
					for(iUnit=0;iUnit<nUnit;iUnit++){
						//get unit index
						guid[iUnit] = guidBase;
						for(k=0;k<nLarge;k++){
							if((iUnit >> k) & 1){
								guid[iUnit] += iAdd[k];
							}
						}
						luid = guid[iUnit] - m_globalUnitIndex;
						iPlace[iUnit] = (int)m_pPlaceTable[luid];
						if(m_pUnits[iPlace[iUnit]]->IsGPU()){
							if(iFirstUnitGPU < 0){
								iFirstUnitGPU = iUnit;
							}
						}
						localMask |= (1ull << iUnit);
						pSrcBuf[iUnit] = GetUnitPtr(luid);
					}

					if(iFirstUnitGPU >= 0){
						if(guid[iFirstUnitGPU] != myGuid){	//already calculated
							continue;
						}
					}

					m_pUnits[iPlaceExec]->WaitPipe(m_pUnits[iPlaceExec]->Pipe());

					//copy to this place
					nTrans = 0;
					for(iUnit=0;iUnit<nUnit;iUnit++){
						if(iPlace[iUnit] == iPlaceExec){
							pBuf[iUnit] = pSrcBuf[iUnit];
						}
						else{
							m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],iPlace[iUnit]);

							pBuf[iUnit] = m_pUnits[iPlaceExec]->Buffer(iUnit);
							nTrans++;
						}
					}
					pGate->Execute(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,nLarge,localMask,nTrans);

					//copy back results to other places on local process
					for(iUnit=0;iUnit<nUnit;iUnit++){
						if(iPlace[iUnit] != iPlaceExec){
							m_pUnits[iPlaceExec]->Get(iUnit,pSrcBuf[iUnit],iPlace[iUnit]);
						}
					}

					m_pUnits[iPlaceExec]->AddPipe();
				}
			}
		}

		//on Host
		iPlaceExec = 0;
		nuPlace = m_pUnits[m_iPlaceHost]->NumUnits();
		for(i=0;i<nuPlace;i++){
			myGuid = m_pUnits[m_iPlaceHost]->GetGlobalUnitIndex(i);
			guidBase = myGuid;
			for(k=0;k<nLarge;k++){
				guidBase &= (~iAdd[k]);
			}
			if(qubits_c[0] >= m_unitBits){		//currently only CX gate uses top control bit, for other gate set qubits_c[0] to -1
				if(((guidBase >> (qubits_c[0] - m_unitBits)) & 1) != pGate->ControlMask()){
					continue;	//not a controled unit, skip
				}
			}
			if(guidBase != myGuid){
				continue;
			}

			iFirstUnitGPU = -1;
			for(iUnit=0;iUnit<nUnit;iUnit++){
				//get unit index
				guid[iUnit] = guidBase;
				for(k=0;k<nLarge;k++){
					if((iUnit >> k) & 1){
						guid[iUnit] += iAdd[k];
					}
				}
				luid = guid[iUnit] - m_globalUnitIndex;
				iPlace[iUnit] = (int)m_pPlaceTable[luid];
				if(m_pUnits[iPlace[iUnit]]->IsGPU()){
					iFirstUnitGPU = iUnit;
					break;
				}
				localMask |= (1ull << iUnit);
				pSrcBuf[iUnit] = GetUnitPtr(luid);
			}
			if(iFirstUnitGPU >= 0){	//a GPU calculates this pairs
				continue;
			}

			//now all units should be on host
			if(m_executeAllOnGPU){
				if(!m_pUnits[iPlaceExec]->IsGPU()){
					iPlaceExec = (iPlaceExec + 1) % m_numPlaces;
				}

				nTrans = 0;
				for(iUnit=0;iUnit<nUnit;iUnit++){
					m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],m_iPlaceHost);
					pBuf[iUnit] = m_pUnits[iPlaceExec]->Buffer(iUnit);
					nTrans++;
				}
				pGate->Execute(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,nLarge,(1 << (nqubits+1))-1,nTrans);

				//copy back results to other places on local process
				for(iUnit=0;iUnit<nUnit;iUnit++){
					m_pUnits[iPlaceExec]->Get(iUnit,pSrcBuf[iUnit],m_iPlaceHost);
				}

				m_pUnits[iPlaceExec]->AddPipe();
				iPlaceExec = (iPlaceExec + 1) % m_numPlaces;
			}
			else{
				pGate->Execute(m_pUnits[m_iPlaceHost],guid,pSrcBuf,qubits,qubits_c,nqubits,nLarge,localMask,0);
			}
		}

		for(j=0;j<m_numPlaces;j++){
			m_pUnits[j]->WaitAll();
		}
	}
	else{		//interaction between units
		pUnitsPerPlace = m_pCountPlace;

		iPipe = 0;		//pipe counter for MPI send/recv
		iPipeWait = 0;

		for(k=0;k<nPipe;k++){
			m_flgSend_Pipe[k] = 0;
			m_flgRecv_Pipe[k] = 0;
		}

		for(iPair=0;iPair<nPair;iPair++){
			//calculate index
			guidBase = 0;
			i = iPair;
			for(j=0;j<nLarge;j++){
				guidBase += (i & (iAdd[j]-1));
				i = (i - (i & (iAdd[j]-1))) << 1;
			}
			guidBase += i;

			if(qubits_c[0] >= m_unitBits){		//currently only CX gate uses top control bit, for other gate set qubits_c[0] to -1
				if(((guidBase >> (qubits_c[0] - m_unitBits)) & 1) != pGate->ControlMask()){
					continue;	//not a controled unit, skip
				}
			}

			m_flgSend_Pipe[iPipe] = 0;
			m_flgRecv_Pipe[iPipe] = 0;

			for(j=0;j<m_numPlaces;j++){
				pUnitsPerPlace[j] = 0;
			}

			localMask = 0;
			nDest = 0;
			nFile = 0;
			nLocal = 0;
			nUnitOnFile = 0;
			nProcFile = 0;
			for(iUnit=0;iUnit<nUnit;iUnit++){
				//get unit index
				guid[iUnit] = guidBase;
				for(k=0;k<nLarge;k++){
					if((iUnit >> k) & 1){
						guid[iUnit] += iAdd[k];
					}
				}

				pRemoteBuf[iUnit] = m_pUnits[m_iPlaceHost]->Buffer(iUnit);

				iproc[iUnit] = GetProcess(guid[iUnit]);
				if(iproc[iUnit] == m_myrank){	//in this process
					luid = guid[iUnit] - m_globalUnitIndex;
					iPlace[iUnit] = (int)m_pPlaceTable[luid];

					nLocal++;

					localMask |= (1ull << iUnit);		//only units on this process is calculated

					pUnitsPerPlace[iPlace[iUnit]]++;
					pSrcBuf[iUnit] = GetUnitPtr(luid);
					if(!m_pUnits[iPlace[iUnit]]->IsGPU()){		//use raw data for host
						pRemoteBuf[iUnit] = pSrcBuf[iUnit];
					}
#ifdef QSIM_GDR
					pRemoteBuf[iUnit] = pSrcBuf[iUnit];
#endif
				}
				else{	//on other process
					pSrcBuf[iUnit] = pRemoteBuf[iUnit];

					iPlace[iUnit] = -1;

					flg = 0;
					for(k=0;k<nDest;k++){
						if(iproc[iUnit] == destProcs[k]){
							flg = 1;
							break;
						}
					}
					if(flg == 0){
						destProcs[nDest] = iproc[iUnit];
						nDest++;
					}
				}
			}

			if(localMask == 0){	//there is no unit to be calculated on this process
				continue;
			}

			iPlaceExec = m_iPlaceHost;
#ifdef QSIM_CUDA
			//find place which has more units
			for(j=0;j<m_numPlaces;j++){
				if(pUnitsPerPlace[j] > pUnitsPerPlace[iPlaceExec]){
					iPlaceExec = j;
				}
			}

			if(m_pUnits[iPlaceExec]->IsGPU() == 0 && (m_executeAllOnGPU || pUnitsPerPlace[iPlaceExec] < nUnit)){	//execute on host only when all the units are on host
				k = 0;
				for(j=0;j<m_numPlaces;j++){
					if(m_pUnits[j]->IsGPU()){
						if(k == (iPair % m_numGPU)){
							iPlaceExec = j;
							break;
						}
						k++;
					}
				}
			}
#endif

			nTrans = 0;
			for(iUnit=0;iUnit<nUnit;iUnit++){
				pBuf[iUnit] = pSrcBuf[iUnit];
				if(iPlace[iUnit] != iPlaceExec){
					pBuf[iUnit] = m_pUnits[iPlaceExec]->Buffer(iUnit);
					nTrans++;
				}
			}

			if(nTrans == 0){		//without data transfer
				pGate->Execute(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,nLarge,localMask,0);

				if(iPlaceExec != m_iPlaceHost){
					m_pUnits[iPlaceExec]->AddPipe();
				}
			}
			else if(nDest == 0){	//without communiction between processes
				//sync
				//m_pUnits[iPlaceExec]->WaitPipe(m_pUnits[iPlaceExec]->Pipe());

				for(iUnit=0;iUnit<nUnit;iUnit++){
					if(iPlace[iUnit] != iPlaceExec){
						m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],iPlace[iUnit]);
					}
				}

				pGate->Execute(m_pUnits[iPlaceExec],guid,pBuf,qubits,qubits_c,nqubits,nLarge,localMask,nTrans);

				//copy back results to other places on local process
				for(iUnit=0;iUnit<nUnit;iUnit++){
					if(iPlace[iUnit] != iPlaceExec){
						m_pUnits[iPlaceExec]->Get(iUnit,pSrcBuf[iUnit],iPlace[iUnit]);
					}
				}

				m_pUnits[iPlaceExec]->AddPipe();
			}
			else{	//need data communications

#ifdef QSIM_MPI
				//receive from other process
				for(iUnit=0;iUnit<nUnit;iUnit++){
					if(iproc[iUnit] != m_myrank){
						MPI_Irecv(pRemoteBuf[iUnit],m_unitSize*2,QS_MPI_REAL_TYPE,iproc[iUnit],guid[iUnit],MPI_COMM_WORLD,&m_pReqRecv[iUnit + iPipe*nUnit]);

						m_flgRecv_Pipe[iPipe] |= (1ull << iUnit);
					}
#ifndef QSIM_GDR
					else{
						if(m_pUnits[iPlace[iUnit]]->IsGPU()){	//copy unit for other process
							m_pUnits[iPlace[iUnit]]->ToHost(pRemoteBuf[iUnit],pSrcBuf[iUnit]);
						}
					}
#endif
				}

				//send units to other processes
				for(iUnit=0;iUnit<nUnit;iUnit++){
					if(iPlace[iUnit] >= 0){
#ifndef QSIM_GDR
						if(m_pUnits[iPlace[iUnit]]->IsGPU()){
							m_pUnits[iPlace[iUnit]]->WaitToHost();
						}
#endif
						for(k=0;k<nDest;k++){
							MPI_Isend(pRemoteBuf[iUnit],m_unitSize*2,QS_MPI_REAL_TYPE,destProcs[k],guid[iUnit],MPI_COMM_WORLD,&m_pReqSend[iUnit*nUnit+k + iPipe*nUnit*nUnit]);
						}

						m_flgSend_Pipe[iPipe] |= (1ull << iUnit);
					}
				}
#endif

				//save parameters
				for(iUnit=0;iUnit<nUnit;iUnit++){
					m_pGuid_Pipe[iPipe*nUnit + iUnit] = guid[iUnit];
					m_pPlace_Pipe[iPipe*nUnit + iUnit] = iPlace[iUnit];
					m_pSrc_Pipe[iPipe*nUnit + iUnit] = pSrcBuf[iUnit];
				}
				m_pLocalMask_Pipe[iPipe] = localMask;
				m_pPlaceExec_Pipe[iPipe] = iPlaceExec;
				m_nDest_Pipe[iPipe] = nDest;

				m_pUnits[m_iPlaceHost]->AddPipe();
				iPipe = m_pUnits[m_iPlaceHost]->Pipe();
			}

			if(m_flgSend_Pipe[iPipeWait] != 0 || m_flgRecv_Pipe[iPipeWait] != 0){
				int endComm;
#ifdef QSIM_MPI
				//wait for recv
				for(iUnit=0;iUnit<nUnit;iUnit++){
					uMask = 1ull << iUnit;
					if(m_flgRecv_Pipe[iPipeWait] & uMask){
						if(iPipe == iPipeWait){
							MPI_Wait(&m_pReqRecv[iUnit + iPipeWait*nUnit],&st);
							endComm = 1;
						}
						else{
							MPI_Test(&m_pReqRecv[iUnit + iPipeWait*nUnit],&endComm,&st);
						}

						if(endComm){
							m_flgRecv_Pipe[iPipeWait] ^= uMask;
						}
					}
				}
				//wait for send
				for(iUnit=0;iUnit<nUnit;iUnit++){
					uMask = 1ull << iUnit;
					if(m_flgSend_Pipe[iPipeWait] & uMask){
						endComm = 0;
						for(k=0;k<m_nDest_Pipe[iPipeWait];k++){
							if(iPipe == iPipeWait){
								MPI_Wait(&m_pReqSend[iUnit*nUnit+k + iPipeWait*nUnit*nUnit],&st);
								endComm++;
							}
							else{
								MPI_Test(&m_pReqSend[iUnit*nUnit+k+iPipeWait*nUnit*nUnit],&j,&st);
								if(j){
									endComm++;
								}
							}
						}
						if(endComm >= m_nDest_Pipe[iPipeWait]){
							m_flgSend_Pipe[iPipeWait] ^= uMask;
						}
					}
				}
#endif
				if(m_flgSend_Pipe[iPipeWait] == 0 && m_flgRecv_Pipe[iPipeWait] == 0){
					iPlaceExec = m_pPlaceExec_Pipe[iPipeWait];

					//sync
					//m_pUnits[iPlaceExec]->WaitPipe(m_pUnits[iPlaceExec]->Pipe());

					nTrans = 0;
					for(iUnit=0;iUnit<nUnit;iUnit++){
						pSrcBuf[iUnit] = m_pSrc_Pipe[iUnit + iPipeWait*nUnit];
						pBuf[iUnit] = pSrcBuf[iUnit];
						if(m_pPlace_Pipe[iUnit + iPipeWait*nUnit] != iPlaceExec && iPlaceExec != m_iPlaceHost){
							pBuf[iUnit] = m_pUnits[iPlaceExec]->Buffer(iUnit);

							nTrans++;
							if(m_pPlace_Pipe[iUnit + iPipeWait*nUnit] >= 0){
								m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],m_pPlace_Pipe[iUnit + iPipeWait*nUnit]);
							}
							else if(iPlaceExec != m_iPlaceHost){
								m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],m_iPlaceHost);
							}
						}
					}

					//execution
					pGate->Execute(m_pUnits[iPlaceExec],m_pGuid_Pipe + iPipeWait*nUnit,pBuf,qubits,qubits_c,nqubits,nLarge,m_pLocalMask_Pipe[iPipeWait],nTrans);

					//copy back results to other places on local process
					if(iPlaceExec != m_iPlaceHost){
						for(iUnit=0;iUnit<nUnit;iUnit++){
							if(m_pPlace_Pipe[iUnit + iPipeWait*nUnit] != iPlaceExec && m_pPlace_Pipe[iUnit + iPipeWait*nUnit] >= 0){
								m_pUnits[iPlaceExec]->Get(iUnit,pSrcBuf[iUnit],m_pPlace_Pipe[iUnit + iPipeWait*nUnit]);
							}
						}
					}

					if(iPipe == iPipeWait){		//synchronize needed for end of ring buffer
						//synchronize pipeline in case previous stream is still running
						m_pUnits[iPlaceExec]->WaitPipe(m_pUnits[iPlaceExec]->Pipe());
					}

					if(iPlaceExec != m_iPlaceHost){
						m_pUnits[iPlaceExec]->AddPipe();
					}
					iPipeWait = (iPipeWait + 1) % nPipe;
				}
			}
		}

		for(iPair=0;iPair<nPipe;iPair++){
			if(m_flgSend_Pipe[iPipeWait] != 0 || m_flgRecv_Pipe[iPipeWait] != 0){
#ifdef QSIM_MPI
				//wait for recv
				for(iUnit=0;iUnit<nUnit;iUnit++){
					uMask = 1ull << iUnit;
					if(m_flgRecv_Pipe[iPipeWait] & uMask){
						MPI_Wait(&m_pReqRecv[iUnit + iPipeWait*nUnit],&st);
						m_flgRecv_Pipe[iPipeWait] ^= uMask;
					}
				}
				//wait for send
				for(iUnit=0;iUnit<nUnit;iUnit++){
					uMask = 1ull << iUnit;
					if(m_flgSend_Pipe[iPipeWait] & uMask){
						for(k=0;k<m_nDest_Pipe[iPipeWait];k++){
							MPI_Wait(&m_pReqSend[iUnit*nUnit+k + iPipeWait*nUnit*nUnit],&st);
						}
						m_flgSend_Pipe[iPipeWait] ^= uMask;
					}
				}
#endif
				iPlaceExec = m_pPlaceExec_Pipe[iPipeWait];

				//sync
				//m_pUnits[iPlaceExec]->WaitPipe(m_pUnits[iPlaceExec]->Pipe());

				nTrans = 0;
				for(iUnit=0;iUnit<nUnit;iUnit++){
					pSrcBuf[iUnit] = m_pSrc_Pipe[iUnit + iPipeWait*nUnit];
					pBuf[iUnit] = pSrcBuf[iUnit];
					if(m_pPlace_Pipe[iUnit + iPipeWait*nUnit] != iPlaceExec && iPlaceExec != m_iPlaceHost){
						pBuf[iUnit] = m_pUnits[iPlaceExec]->Buffer(iUnit);
						nTrans++;
						if(m_pPlace_Pipe[iUnit + iPipeWait*nUnit] >= 0){
							m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],m_pPlace_Pipe[iUnit + iPipeWait*nUnit]);
						}
						else{
							m_pUnits[iPlaceExec]->Put(iUnit,pSrcBuf[iUnit],m_iPlaceHost);
						}
					}
				}

				//execution
				pGate->Execute(m_pUnits[iPlaceExec],m_pGuid_Pipe + iPipeWait*nUnit,pBuf,qubits,qubits_c,nqubits,nLarge,m_pLocalMask_Pipe[iPipeWait],nTrans);

				//copy back results to other places on local process
				if(iPlaceExec != m_iPlaceHost){
					for(iUnit=0;iUnit<nUnit;iUnit++){
						if(m_pPlace_Pipe[iUnit + iPipeWait*nUnit] != iPlaceExec && m_pPlace_Pipe[iUnit + iPipeWait*nUnit] >= 0){
							m_pUnits[iPlaceExec]->Get(iUnit,pSrcBuf[iUnit],m_pPlace_Pipe[iUnit + iPipeWait*nUnit]);
						}
					}
				}
				if(iPlaceExec != m_iPlaceHost){
					m_pUnits[iPlaceExec]->AddPipe();
				}
			}
			iPipeWait = (iPipeWait + 1) % nPipe;
		}

//#pragma omp parallel for private(j)
		for(j=0;j<m_numPlaces;j++){
			m_pUnits[j]->WaitAll();
		}
	}


}



/*-------------------------------------------------------------
	Multi-shot optimization
--------------------------------------------------------------*/
void QSUnitManager::Measure_FindPos(QSDouble* rs,QSUint* ret,int ns)
{
	QSGate_MultiShot msGate;
	QSDouble* pProcTotal;
	int* ranks;
	int i,is,qubit = 0;
	QSDouble t;
	int iPlace;
	QSUint iUnit;

	QSUint guid;
	QSComplex* pBuf;

	msGate.SetNumUnits(m_numUnits);

	ExecuteGate(&msGate,&qubit,&qubit,1);

#ifdef QSIM_DEBUG
//	for(iUnit=0;iUnit<m_numUnits;iUnit++){
//		printf(" [%d] unit %d total = %e\n",m_myrank,iUnit,msGate.UnitTotal(iUnit));
//	}
#endif

	pProcTotal = new QSDouble[m_nprocs];
	ranks = new int[ns];

	pProcTotal[m_myrank] = msGate.Total();


#ifdef QSIM_MPI
	for(i=0;i<m_nprocs;i++){
		MPI_Bcast(pProcTotal+i,1,MPI_UINT64_T,i,MPI_COMM_WORLD);
	}

	//copy random key value on process 0 to all processes
//	MPI_Bcast(rs,64,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD);

#endif


	for(is=0;is<ns;is++){
		t = 0.0;
		//find process
		for(i=0;i<m_nprocs;i++){
			t += pProcTotal[m_pProcIndex[i]];
			if(t > rs[is]){
				t -= pProcTotal[m_pProcIndex[i]];
				break;
			}
		}
		if(i >= m_nprocs){
			ret[is] = m_nState - 1;
			ranks[is] = m_pProcIndex[m_nprocs-1];
			continue;
		}
		ranks[is] = m_pProcIndex[i];

//		printf(" [%d] multishot : rs[%d] = %f , t = %f, rank = %d\n",m_myrank,is,rs[is],t,ranks[is]);

		if(ranks[is] == m_myrank){	//on this process
			//find unit
			for(iUnit=0;iUnit<m_numUnits;iUnit++){
				t += msGate.UnitTotal(iUnit);
				if(t > rs[is]){
					t -= msGate.UnitTotal(iUnit);
					break;
				}
			}

			iPlace = (int)m_pPlaceTable[iUnit];
			msGate.SetKey(rs[is] - t);

			if(m_pUnits[iPlace]->IsGPU()){	//search on Host, copy unit from GPU
				pBuf = m_pUnits[m_iPlaceHost]->Buffer(0);
				m_pUnits[iPlace]->ToHost(pBuf,GetUnitPtr(iUnit));
				m_pUnits[iPlace]->WaitToHost();
			}
			else{
				pBuf = GetUnitPtr(iUnit);
			}

			msGate.ExecuteOnHost(m_pUnits[m_iPlaceHost],&guid,&pBuf,&qubit,&qubit,1,0,1,0);

			ret[is] = ((m_pUnits[iPlace]->GetGlobalUnitIndexBase() + iUnit) << m_unitBits) + msGate.Pos();
		}
	}


#ifdef QSIM_MPI
	for(is=0;is<ns;is++){
		MPI_Bcast(ret + is,1,MPI_UINT64_T,ranks[is],MPI_COMM_WORLD);
	}
#endif

	delete[] pProcTotal;
	delete[] ranks;
}


void QSUnitManager::TimeReset(void)
{
	int i;
	for(i=0;i<QS_NUM_GATES;i++){
		m_gateCounts[i] = 0;
		m_gateTime[i] = 0.0;
	}
}

void QSUnitManager::TimeStart(int i)
{
	m_gateStartTime[i] = mysecond();
}

void QSUnitManager::TimeEnd(int i)
{
	double t = mysecond();
	m_gateTime[i] += t - m_gateStartTime[i];
	m_gateCounts[i]++;
}

void QSUnitManager::TimePrint(void)
{
	int i;
	double total;
#ifdef QSIM_MPI
	double t[QS_NUM_GATES];
	QSUint c[QS_NUM_GATES];
	MPI_Allreduce(m_gateTime,t,QS_NUM_GATES,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(m_gateCounts,c,QS_NUM_GATES,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);

	for(i=0;i<QS_NUM_GATES;i++){
		m_gateTime[i] = t[i] / (double)m_nprocs;
		m_gateCounts[i] = c[i] / m_nprocs;
	}
#endif

	total = 0;
	for(i=0;i<QS_NUM_GATES;i++){
		total += m_gateTime[i];
	}

	if(m_myrank == 0){
		printf("   ==================== Timing Summary =================== \n");
		if(m_gateCounts[QS_GATE_MULT] > 0)
			printf("    Matrix mult. : %f  (%d)\n",m_gateTime[QS_GATE_MULT],m_gateCounts[QS_GATE_MULT]);
		if(m_gateCounts[QS_GATE_CX] > 0)
			printf("    CX           : %f  (%d)\n",m_gateTime[QS_GATE_CX],m_gateCounts[QS_GATE_CX]);
		if(m_gateCounts[QS_GATE_U1] > 0)
			printf("    U1           : %f  (%d)\n",m_gateTime[QS_GATE_U1],m_gateCounts[QS_GATE_U1]);
		if(m_gateCounts[QS_GATE_MEASURE] > 0)
			printf("    Measure      : %f  (%d)\n",m_gateTime[QS_GATE_MEASURE],m_gateCounts[QS_GATE_MEASURE]);
		printf("    Total Kernel time : %f sec\n",total);
	}

}



