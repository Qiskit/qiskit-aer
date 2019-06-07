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

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_UNIT_MANAGER_H_
#define _IBM_Q_SIMULATOR_UNIT_MANAGER_H_



#include "QSUnitStorage.h"
#include "QSGate.h"

#define QS_COMM_STATE_FINISH			0
#define QS_COMM_STATE_WAIT_RECV			1
#define QS_COMM_STATE_WAIT_PUT			2
#define QS_COMM_STATE_WAIT_SEND			3
#define QS_COMM_STATE_WAIT_GET			4


#define QS_NUM_GATES					4
#define QS_GATE_MULT					0
#define QS_GATE_CX						1
#define QS_GATE_U1						2
#define QS_GATE_MEASURE					3


class QSUnitManager
{
protected:
	QSUnitStorage** m_pUnits;
	int m_numPlaces;
	int m_iPlaceHost;
	int m_numGPU;
	int m_unitBits;
	int m_procBits;			//only set this when number of processes = 2^n
	int m_globalBits;
	QSUint m_unitSize;
	QSUint m_numUnits;
	QSUint m_numGlobalUnits;
	QSUint m_nState;
	QSUint* m_pUnitTable;
	int8_t* m_pPlaceTable;
	int m_myrank;
	int m_nprocs;
	int m_nprocs_per_node;
	int m_iproc_per_node;
	QSUint m_globalUnitIndex;
	int m_numBuffers;
	QSReal* m_pNormBuf;

	QSUint* m_pUStart;
	QSUint* m_pUEnd;
	QSUint* m_pUFile;
	QSUint* m_pOffsetFile;

	int* m_pCountPlace;

	int m_executeAllOnGPU;

	QSUint m_numUnitsOnFile;
	QSUint m_numGlobalFile;

	int m_isPowerOf2;
	QSUint* m_pProcIndex;
	QSUint* m_pProcMap;

	QSUint* m_pGuid_Pipe;
	QSUint* m_pLocalMask_Pipe;
	int* m_pPlaceExec_Pipe;
	int* m_pPlace_Pipe;
	int* m_nTrans_Pipe;
	int* m_nDest_Pipe;
	QSComplex** m_pBuf_Pipe;
	QSComplex** m_pSrc_Pipe;
	QSUint* m_flgSend_Pipe;
	QSUint* m_flgRecv_Pipe;


	QSUint m_gateCounts[QS_NUM_GATES];
	double m_gateTime[QS_NUM_GATES];
	double m_gateStartTime[QS_NUM_GATES];

public:
	QSUnitManager(int globalBits)
	{
		m_unitBits = 20;
		m_globalBits = globalBits;

		m_nState = 1ull << globalBits;

		m_numGPU = 0;
		m_iPlaceHost = 0;

		m_pUnits = NULL;
		m_pUnitTable = NULL;
		m_pPlaceTable = NULL;
		m_pNormBuf = NULL;

		m_pUStart = NULL;
		m_pUEnd = NULL;
		m_pUFile = NULL;
		m_pOffsetFile = NULL;

		m_numBuffers = QS_MAX_PIPE*2;

		m_numUnitsOnFile = 0;
		m_numGlobalFile = 0;

		m_isPowerOf2 = 0;

		m_pCountPlace = NULL;

		m_pProcIndex = NULL;
		m_pProcMap = NULL;

		m_pGuid_Pipe = NULL;
		m_pLocalMask_Pipe = NULL;
		m_pPlaceExec_Pipe = NULL;
		m_pPlace_Pipe = NULL;
		m_nTrans_Pipe = NULL;
		m_nDest_Pipe = NULL;
		m_pBuf_Pipe = NULL;
		m_pSrc_Pipe = NULL;
		m_flgSend_Pipe = NULL;
		m_flgRecv_Pipe = NULL;

#ifdef QSIM_HALF
		m_executeAllOnGPU = 1;
#else
		m_executeAllOnGPU = 0;
#endif
	}

	~QSUnitManager(void)
	{
		if(m_pUnits){
			int i;
			for(i=0;i<m_numPlaces;i++){
				delete m_pUnits[i];
			}
			delete[] m_pUnits;
		}

		if(m_pUnitTable){
			delete[] m_pUnitTable;
		}
		if(m_pPlaceTable){
			delete[] m_pPlaceTable;
		}

		if(m_pNormBuf){
			delete[] m_pNormBuf;
		}
		if(m_pUStart){
			delete[] m_pUStart;
		}
		if(m_pUEnd){
			delete[] m_pUEnd;
		}
		if(m_pUFile){
			delete[] m_pUFile;
		}
		if(m_pOffsetFile){
			delete[] m_pOffsetFile;
		}

		if(m_pCountPlace){
			delete[] m_pCountPlace;
		}

		if(m_pProcIndex){
			delete[] m_pProcIndex;
		}
		if(m_pProcMap){
			delete[] m_pProcMap;
		}
		

		if(m_pGuid_Pipe){
			delete[] m_pGuid_Pipe;
			delete[] m_pLocalMask_Pipe;
			delete[] m_pPlaceExec_Pipe;
			delete[] m_pPlace_Pipe;
			delete[] m_nTrans_Pipe;
			delete[] m_nDest_Pipe;
			delete[] m_pBuf_Pipe;
			delete[] m_pSrc_Pipe;
			delete[] m_flgSend_Pipe;
			delete[] m_flgRecv_Pipe;
		}
	}

	void SetNumBuffers(int n)
	{
		m_numBuffers = n;
	}

	void Init(void);


	QSUnitStorage* Storage(int i)
	{
		return m_pUnits[i];
	}

	QSComplex* GetUnitPtr(QSUint i)
	{
		return m_pUnits[m_pPlaceTable[i]]->Unit(m_pUnitTable[i]);
	}

	void SetValue(QSDoubleComplex c,QSUint gid);
	void Clear(void);

	int GetProcess(QSUint ui);

	void ExecuteAllOnGPU(int t)
	{
		m_executeAllOnGPU = t;
	}

	void TimeReset(void);
	void TimeStart(int i);
	void TimeEnd(int i);
	void TimePrint(void);

	//-------------
	//operations
	//-------------

	QSDouble Dot(int qubit);

	void Measure(int qubit,int flg,QSDouble norm);

	void MatMult(QSDoubleComplex* pM,int* qubits,int n);

	void MatMultDiagonal(QSDoubleComplex* pM,int* qubits,int n);

	void CX(int qubit_t,int qubit_c);

	void U1(int qubit,QSDouble* pPhase);

	void X(int qubit);
	void Y(int qubit);

	void Measure_FindPos(QSDouble* rs,QSUint* ret,int ns);

protected:
	void ExecuteGate(QSGate* pGate,int* qubits,int* qubits_c,int nqubits);


	void SortProcs(int* pProcs,int n);



};















#endif	//_IBM_Q_SIMULATOR_UNIT_STORAGE_H_


