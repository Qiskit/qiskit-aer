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

	Multi-shot measure

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_GATE_MULTI_SHOT_H_
#define _IBM_Q_SIMULATOR_GATE_MULTI_SHOT_H_

#include "QSGate.h"



class QSGate_MultiShot : public QSGate
{
protected:
	QSDouble* m_pDotPerUnits;
	QSDouble m_Total;
	QSDouble m_Key;
	QSUint m_Pos;
public:
	QSGate_MultiShot(void)
	{
		m_pDotPerUnits = NULL;
		m_Total = 0.0;
		m_Key = -1.0;
	}

	virtual ~QSGate_MultiShot()
	{
		if(m_pDotPerUnits){
			delete[] m_pDotPerUnits;
		}
	}

	void SetNumUnits(QSUint n)
	{
		m_pDotPerUnits = new QSDouble[n];
		m_Total = 0.0;
	}

	QSDouble Total(void)
	{
		return m_Total;
	}
	QSDouble UnitTotal(QSUint i)
	{
		return m_pDotPerUnits[i];
	}

	void SetKey(QSDouble key)
	{
		m_Key = key;
	}
	QSUint Pos(void)
	{
		return m_Pos;
	}

	//return 1 if data exchange is required for the gate operation
	int ExchangeNeeded(void)
	{
		return 0;	//no data exchange needed
	}

	//implementation of kernel for gate operation
	void ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);
	void ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);

};


#endif	//_IBM_Q_SIMULATOR_GATE_MULTI_SHOT_H_

