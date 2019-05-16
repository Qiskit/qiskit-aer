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

	X gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include "QSGate_X.h"

void QSGate_X::ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint i,k,k1,k2,kb,mask,kbadd,n;
	QSRealC psr0,psi0,psr1,psi1;

	if(qubits[0] < pUnit->UnitBits()){
		QSReal* pBuf0 = (QSReal*)ppBuf[0];
		n = 1ull << (pUnit->UnitBits() - 1);

		mask = (1ull << qubits[0]) - 1;
		kbadd = (1ull << qubits[0]);

#pragma omp parallel for private(k,k1,k2,kb,psr0,psi0,psr1,psi1)
		for(i=0;i<n;i++){
			k2 = i & mask;
			k1 = (i - k2) << 1;
			k = k1 + k2;
			kb = k + kbadd;

			psr0 = (QSRealC)pBuf0[k*2];
			psi0 = (QSRealC)pBuf0[k*2+1];
			psr1 = (QSRealC)pBuf0[kb*2];
			psi1 = (QSRealC)pBuf0[kb*2+1];

			pBuf0[k*2  ] = psr1;
			pBuf0[k*2+1] = psi1;
			pBuf0[kb*2  ] = psr0;
			pBuf0[kb*2+1] = psi0;
		}
	}
	else{
		QSReal* pBuf0 = (QSReal*)ppBuf[0];
		QSReal* pBuf1 = (QSReal*)ppBuf[1];

		n = 1ull << (pUnit->UnitBits());
#pragma omp parallel for private(psr0,psi0,psr1,psi1)
		for(i=0;i<n;i++){
			psr0 = (QSRealC)pBuf0[i*2];
			psi0 = (QSRealC)pBuf0[i*2+1];
			psr1 = (QSRealC)pBuf1[i*2];
			psi1 = (QSRealC)pBuf1[i*2+1];

			pBuf0[i*2  ] = psr1;
			pBuf0[i*2+1] = psi1;
			pBuf1[i*2  ] = psr0;
			pBuf1[i*2+1] = psi0;
		}
	}
}



