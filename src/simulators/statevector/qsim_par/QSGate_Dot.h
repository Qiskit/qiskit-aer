/*------------------------------------------------------------------------------------
	IBM Q Simulator

	calculate dot product of qubit

	2019 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#ifndef _IBM_Q_SIMULATOR_GATE_DOT_H_
#define _IBM_Q_SIMULATOR_GATE_DOT_H_

#include "QSGate.h"



class QSGate_Dot : public QSGate
{
protected:
	double m_Dot;
public:
	QSGate_Dot(void)
	{
		m_Dot = 0.0;
	}

	virtual ~QSGate_Dot()
	{
	}

	double Result(void)
	{
		return m_Dot;
	}


	//return 1 if data exchange is required for the gate operation
	int ExchangeNeeded(void)
	{
		return 0;	//no data exchange needed
	}

	//control bit mask
	int ControlMask(void)
	{
		return 0;
	}

	//implementation of kernel for gate operation
	void ExecuteOnGPU(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);
	void ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localmask,int nTrans);


	//for GPU
	void InitBuffer(QSUnitStorage* pUnit);
	double ReduceAll(QSUnitStorage* pUnit);
};


#endif	//_IBM_Q_SIMULATOR_GATE_DOT_H_

