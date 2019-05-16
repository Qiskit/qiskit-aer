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

	Unit storage base class

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include "QSUnitStorage.h"


void QSUnitStorage::Init(int is)
{
	int i;

	m_pIndex = new int[m_numStorage];

#pragma omp parallel for
	for(i=0;i<m_numUnits;i++){
		m_pIndex[i] = is + i;
	}
	for(i=m_numUnits;i<m_numStorage;i++){
		m_pIndex[i] = -1;
	}

}














