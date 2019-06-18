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

	Diagonal matrix multiply gate

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include "QSGate_DiagMult.h"


static void QSGate_DiagMult_2x2_InUnit(QSDouble* pM,QSReal* pBuf,int qubit,QSUint ncols)
{
	QSUint i;
	QSRealC psr0,psi0,psr1,psi1,qr0,qi0,qr1,qi1;

#pragma omp parallel for private(psr0,psi0,qr0,qi0)
	for(i=0;i<ncols;i++){
		psr0 = (QSRealC)pBuf[i*2];
		psi0 = (QSRealC)pBuf[i*2+1];

		if(((i >> qubit) & 1ull) == 0){
			qr0 = pM[0] * psr0 - pM[1] * psi0;
			qi0 = pM[0] * psi0 + pM[1] * psr0;
		}
		else{
			qr0 = pM[2] * psr0 - pM[3] * psi0;
			qi0 = pM[2] * psi0 + pM[3] * psr0;
		}

		pBuf[i*2  ] = qr0;
		pBuf[i*2+1] = qi0;
	}
}

static void QSGate_DiagMult_2x2(QSDouble* pM,QSReal* pBuf,QSUint ncols)
{
	QSUint i;
	QSRealC psr0,psi0,psr1,psi1,qr0,qi0,qr1,qi1;

#pragma omp parallel for private(psr0,psi0,qr0,qi0)
	for(i=0;i<ncols;i++){
		psr0 = (QSRealC)pBuf[i*2];
		psi0 = (QSRealC)pBuf[i*2+1];

		qr0 = pM[0] * psr0 - pM[1] * psi0;
		qi0 = pM[0] * psi0 + pM[1] * psr0;

		pBuf[i*2  ] = (QSReal)qr0;
		pBuf[i*2+1] = (QSReal)qi0;
	}
}

static void QSGate_DiagMult_NxN(QSDouble* pM,QSComplex** ppBuf_in,int* qubits,int nqubits,QSUint localMask,QSUint ncols,int nLarge)
{
	QSUint mask,add;
	QSUint i,ii,idx,t;
	int j,k,l,matSize;
	QSRealC pr;
	QSRealC pi;
	QSRealC mr,mi;
	QSRealC qr,qi;
	QSReal* ppBuf[QS_MAX_MATRIX_SIZE];

	matSize = 1 << nqubits;

	//offset calculation
	for(k=0;k<matSize;k++){
		j = (k >> (nqubits - nLarge));
		ppBuf[k] = (QSReal*)ppBuf_in[j];
	}
	for(j=0;j<nqubits - nLarge;j++){
		add = (1ull << qubits[j]);
		for(k=0;k<matSize;k++){
			if((k >> j) & 1ull){
				ppBuf[k] += add*2;
			}
		}
	}

#pragma omp parallel for private(i,j,k,l,idx,ii,add,mask,t,pr,pi,mr,mi,qr,qi)
	for(i=0;i<ncols;i++){
		idx = 0;
		ii = i;
		for(j=0;j<nqubits - nLarge;j++){
			add = (1ull << qubits[j]);
			mask = add - 1;

			t = ii & mask;
			idx += t;
			ii = (ii - t) << 1;
		}
		idx += ii;
		idx <<= 1;

		for(j=0;j<matSize;j++){
			if((localMask >> j) & 1ull){
				pr = (QSRealC)*(ppBuf[j] + idx  );
				pi = (QSRealC)*(ppBuf[j] + idx+1);

				mr = pM[j*2];
				mi = pM[j*2+1];

				qr = mr*pr - mi*pi;
				qi = mr*pi + mi*pr;

				*(ppBuf[j] + idx  ) = (QSReal)qr;
				*(ppBuf[j] + idx+1) = (QSReal)qi;
			}
		}
	}
}



void QSGate_DiagMult::ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint i,ncols;

//	ncols = 1ull << (pUnit->UnitBits() - (nqubits - nqubitsLarge));
	ncols = 1ull << pUnit->UnitBits();

	if(nqubits == 1){
		if(qubits[0] < pUnit->UnitBits()){		//inside unit
			QSGate_DiagMult_2x2_InUnit((QSDouble*)m_Mat,(QSReal*)ppBuf[0],qubits[0],ncols);
		}
		else{
			if(((pGuid[0] >> (qubits[0] - pUnit->UnitBits())) & 1ull) == 0){
				QSGate_DiagMult_2x2((QSDouble*)(m_Mat),(QSReal*)ppBuf[0],ncols);
			}
			else{
				QSGate_DiagMult_2x2((QSDouble*)(m_Mat+1),(QSReal*)ppBuf[0],ncols);
			}
		}
	}
	else{
		QSUint add,addf;
		QSUint mask;
		int j,k,matSize;

		matSize = 1 << nqubits;
		mask = 0;
		for(j=0;j<matSize;j++){
			k = (j >> (nqubits - nqubitsLarge));
			mask |= ( ((localMask >> k) & 1ull) << j );
		}
		QSGate_DiagMult_NxN((QSDouble*)m_Mat,ppBuf,qubits,nqubits,mask,ncols,nqubitsLarge);
	}
}



