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

	virtual class for gate operators

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_GATE_H_
#define _IBM_Q_SIMULATOR_GATE_H_

#include "QSType.h"
#include "QSUnitStorage.h"



class QSGate
{
protected:

public:
	QSGate()
	{
	}

	virtual ~QSGate()
	{
	}


	//return 1 if data exchange is required for the gate operation
	virtual int ExchangeNeeded(void) = 0;

	//control bit mask
	virtual int ControlMask(void)
	{
		return -1;
	}

	//implementation of kernel for gate operation
	void Execute(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans)
	{
		if(pUnit->IsGPU()){
			ExecuteOnGPU(pUnit,pGuid,ppBuf,qubits,qubits_c,nqubits,nqubitsLarge,localmask,nTrans);
		}
		else{
			ExecuteOnHost(pUnit,pGuid,ppBuf,qubits,qubits_c,nqubits,nqubitsLarge,localmask,nTrans);
		}
	}

	virtual void ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans) = 0;
	virtual void ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans) = 0;
};




#endif	//_IBM_Q_SIMULATOR_GATE_H_

