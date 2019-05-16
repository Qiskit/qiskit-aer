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

	Y gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_GATE_Y_H_
#define _IBM_Q_SIMULATOR_GATE_Y_H_

#include "QSGate.h"



class QSGate_Y : public QSGate
{
protected:

public:
	QSGate_Y()
	{

	}

	virtual ~QSGate_Y()
	{
	}

	//return 1 if data exchange is required for the gate operation
	int ExchangeNeeded(void)
	{
		return 1;
	}

	//implementation of kernel for gate operation
	void ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);
	void ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);

};


#endif	//_IBM_Q_SIMULATOR_GATE_Y_H_

