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

	Diagonal matrix multiplication 

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_GATE_DIAG_MULT_H_
#define _IBM_Q_SIMULATOR_GATE_DIAG_MULT_H_

#include "QSGate.h"



class QSGate_DiagMult : public QSGate
{
protected:
	QSDoubleComplex m_Mat[QS_MAX_MATRIX_SIZE];
public:
	QSGate_DiagMult()
	{
		;
	}

	virtual ~QSGate_DiagMult()
	{
	}

	void SetMatrix(QSDoubleComplex* pM,int size)
	{
		::memcpy(m_Mat,pM,sizeof(QSDoubleComplex)*size);
	}

	//return 1 if data exchange is required for the gate operation
	int ExchangeNeeded(void)
	{
		return 0;	//no data exchange needed
	}

	//implementation of kernel for gate operation
	void ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);
	void ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);

	void CopyMatrix(QSUnitStorage* pUnit,int* qubits,int nqubits,int wait);

};


#endif	//_IBM_Q_SIMULATOR_GATE_DIAG_MULT_H_

