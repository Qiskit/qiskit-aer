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

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_UNIT_MANAGER_SERIAL_H_
#define _IBM_Q_SIMULATOR_UNIT_MANAGER_SERIAL_H_



#include "QSUnitStorage.h"
#include "QSGate.h"

#include "QSUnitManager.h"


class QSUnitManagerSerial : public QSUnitManager
{
protected:

public:
	QSUnitManagerSerial(int globalBits) : QSUnitManager(globalBits)
	{

	}

	~QSUnitManagerSerial(void)
	{
	}

	void Init(void);

	virtual QSComplex* GetUnitPtr(QSUint i)
	{
		return m_pUnits[0]->Unit(0);
	}

	virtual void SetValue(QSDoubleComplex c,QSUint gid);
	virtual void Clear(void);

	virtual void Copy(QSComplex* pV);

	//-------------
	//operations
	//-------------

	virtual QSDouble Dot(int qubit);

	virtual void Measure(int qubit,int flg,QSDouble norm);

	virtual void MatMult(QSDoubleComplex* pM,int* qubits,int n);

	virtual void MatMultDiagonal(QSDoubleComplex* pM,int* qubits,int n);

	virtual void CX(int qubit_t,int qubit_c);

	virtual void U1(int qubit,QSDouble* pPhase);


	virtual void Measure_FindPos(QSDouble* rs,QSUint* ret,int ns);

protected:
	virtual void ExecuteGate(QSGate* pGate,int* qubits,int* qubits_c,int nqubits);

};


#endif	//_IBM_Q_SIMULATOR_UNIT_STORAGE_SERIAL_H_


