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

	controlled not gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include "QSGate_CX.h"

void QSGate_CX::ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	int bIn,bOut;
	QSUint inMask,outMask;
	QSUint iIn,iOut,ind0,ind1,iadd,i,iu,n,offset;
	QSComplex psi0,psi1;
	QSComplex* pBuf0 = ppBuf[0];
	QSComplex* pBuf1 = ppBuf[1];
	int qubit_t = qubits[0];
	int qubit_c = qubits_c[0];

	if(nqubitsLarge == 0){	//local calculation inside unit
		if(qubit_c < pUnit->UnitBits()){
			if(qubit_c < qubit_t){
				bIn = qubit_c;
				bOut = qubit_t;
			}
			else{
				bIn = qubit_t;
				bOut = qubit_c;
			}
			inMask = (1ull << bIn) - 1;
			outMask = (1ull << (bOut - 1)) - 1;
			iadd = (1ull << qubit_t);

			n = 1ull << (pUnit->UnitBits() - 2);

#pragma omp parallel for private(i,iIn,iOut,ind0,ind1,psi0,psi1)
			for(i=0;i<n;i++){
				iIn = i & inMask;
				iOut = i & outMask;

				ind0 = (1ull << qubit_c) + ((i >> (bOut - 1)) << (bOut + 1)) + ((iOut >> bIn) << (bIn + 1)) + iIn;
				ind1 = ind0 + iadd;

				psi0 = pBuf0[ind0];
				psi1 = pBuf0[ind1];

				pBuf0[ind0] = psi1;
				pBuf0[ind1] = psi0;
			}
		}
		else{
			n = 1ull << (pUnit->UnitBits() - 1);
			inMask = (1ull << qubit_t) - 1;
			iadd = (1ull << qubit_t);

#pragma omp parallel for private(i,ind0,ind1,psi0,psi1)
			for(i=0;i<n;i++){
				ind0 = ((i >> qubit_t) << (qubit_t + 1)) + (i & inMask);
				ind1 = ind0 + iadd;

				psi0 = pBuf0[ind0];
				psi1 = pBuf0[ind1];

				pBuf0[ind0] = psi1;
				pBuf0[ind1] = psi0;
			}
		}
	}
	else{
		if(qubit_c < pUnit->UnitBits()){
			n = 1ull << (pUnit->UnitBits() - 1);

			inMask = (1ull << qubit_c) - 1;
#pragma omp parallel for private(i,ind0,psi0,psi1)
			for(i=0;i<n;i++){
				ind0 = (1ull << qubit_c) + ((i >> qubit_c) << (qubit_c + 1)) + (i & inMask);

				psi0 = pBuf0[ind0];
				psi1 = pBuf1[ind0];

				if(localMask & 1){
					pBuf0[ind0] = psi1;
				}
				if(localMask & 2){
					pBuf1[ind0] = psi0;
				}
			}
		}
		else{
			n = 1ull << (pUnit->UnitBits());
#pragma omp parallel for private(i,psi0,psi1)
			for(i=0;i<n;i++){
				psi0 = pBuf0[i];
				psi1 = pBuf1[i];

				if(localMask & 1){
					pBuf0[i] = psi1;
				}
				if(localMask & 2){
					pBuf1[i] = psi0;
				}
			}
		}
	}

}



