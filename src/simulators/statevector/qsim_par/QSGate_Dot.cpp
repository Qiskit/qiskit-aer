/*------------------------------------------------------------------------------------
	IBM Q Simulator

	calculate dot product of qubit

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include "QSGate_Dot.h"

void QSGate_Dot::ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint n, mask,i,k,k1,k2;
	double ret = 0.0;
	QSReal* pD = (QSReal*)ppBuf[0];

	if(qubits[0] < pUnit->UnitBits()){
		n = 1ull << (pUnit->UnitBits() - 1);
		mask = ((1ull << qubits[0]) - 1);

#pragma omp parallel for reduction(+:ret) private(k,k1,k2)
		for(i=0;i<n;i++){
			k2 = i & mask;
			k1 = (i - k2) << 1;
			k = k1 + k2;
			ret += (double)pD[k*2]*(double)pD[k*2] + (double)pD[k*2+1]*(double)pD[k*2+1];
		}
	}
	else{
		n = 1ull << (pUnit->UnitBits());
#pragma omp parallel for reduction(+:ret) private(k)
		for(i=0;i<n;i++){
			k = i;
			ret += (double)pD[k*2]*(double)pD[k*2] + (double)pD[k*2+1]*(double)pD[k*2+1];
		}
	}

	m_Dot += ret;
}



