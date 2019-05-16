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

#include "QSGate_MultiShot.h"

void QSGate_MultiShot::ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint n,i;
	double ret = 0.0;
	QSReal* pD = (QSReal*)ppBuf[0];

	n = 1ull << (pUnit->UnitBits());

	if(m_Key < 0.0){	//reduction mode
#pragma omp parallel for reduction(+:ret) private(i)
		for(i=0;i<n;i++){
			ret += (double)pD[i*2]*(double)pD[i*2] + (double)pD[i*2+1]*(double)pD[i*2+1];
		}

		m_pDotPerUnits[pGuid[0] - pUnit->GetGlobalUnitIndexBase()] = ret;
		m_Total += ret;
	}
	else{	//search mode
		for(i=0;i<n;i++){
			ret += (double)pD[i*2]*(double)pD[i*2] + (double)pD[i*2+1]*(double)pD[i*2+1];
			if(ret > m_Key){
				break;
			}
		}
		m_Pos = i;
	}
}



