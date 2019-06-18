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

	matrix multiply gate
	2x2 : U3 gate
	4- : fusion gates

	2018 IBM Research - Tokyo
--------------------------------------------------------------------------------------*/

#include "QSGate_MatMult.h"


static void QSGate_MatMult_2x2(QSDouble* pM,QSReal* pBuf0,QSReal* pBuf1,int qubit,QSUint localMask,QSUint ncols)
{
	QSUint i,k,k1,k2,kb,mask,kbadd;
	QSRealC psr0,psi0,psr1,psi1,qr0,qi0,qr1,qi1;
	QSRealC mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3;

	mr0 = pM[0];
	mi0 = pM[1];
#ifdef QSIM_COL_MAJOR	//for Aer
	mr1 = pM[4];
	mi1 = pM[5];
	mr2 = pM[2];
	mi2 = pM[3];
#else
	mr1 = pM[2];
	mi1 = pM[3];
	mr2 = pM[4];
	mi2 = pM[5];
#endif
	mr3 = pM[6];
	mi3 = pM[7];

	if(pBuf1 == pBuf0){
		mask = (1ull << qubit) - 1;
		kbadd = (1ull << qubit);

#pragma omp parallel for private(k,k1,k2,kb,psr0,psi0,psr1,psi1,qr0,qi0,qr1,qi1)
		for(i=0;i<ncols;i++){
			k2 = i & mask;
			k1 = (i - k2) << 1;
			k = k1 + k2;
			kb = k + kbadd;

			psr0 = (QSRealC)pBuf0[k*2];
			psi0 = (QSRealC)pBuf0[k*2+1];
			psr1 = (QSRealC)pBuf0[kb*2];
			psi1 = (QSRealC)pBuf0[kb*2+1];

			qr0 = mr0 * psr0 - mi0 * psi0 + mr1 * psr1 - mi1 * psi1;
			qi0 = mr0 * psi0 + mi0 * psr0 + mr1 * psi1 + mi1 * psr1;
			qr1 = mr2 * psr0 - mi2 * psi0 + mr3 * psr1 - mi3 * psi1;
			qi1 = mr2 * psi0 + mi2 * psr0 + mr3 * psi1 + mi3 * psr1;

			pBuf0[k*2  ] = qr0;
			pBuf0[k*2+1] = qi0;
			pBuf0[kb*2  ] = qr1;
			pBuf0[kb*2+1] = qi1;
		}
	}
	else{

#pragma omp parallel for private(psr0,psi0,psr1,psi1,qr0,qi0,qr1,qi1)
		for(i=0;i<ncols;i++){

			psr0 = (QSRealC)pBuf0[i*2];
			psi0 = (QSRealC)pBuf0[i*2+1];
			psr1 = (QSRealC)pBuf1[i*2];
			psi1 = (QSRealC)pBuf1[i*2+1];

			if(localMask & 1){
				qr0 = mr0 * psr0 - mi0 * psi0 + mr1 * psr1 - mi1 * psi1;
				qi0 = mr0 * psi0 + mi0 * psr0 + mr1 * psi1 + mi1 * psr1;
				pBuf0[i*2  ] = (QSReal)qr0;
				pBuf0[i*2+1] = (QSReal)qi0;
			}
			if(localMask & 2){
				qr1 = mr2 * psr0 - mi2 * psi0 + mr3 * psr1 - mi3 * psi1;
				qi1 = mr2 * psi0 + mi2 * psr0 + mr3 * psi1 + mi3 * psr1;

				pBuf1[i*2  ] = (QSReal)qr1;
				pBuf1[i*2+1] = (QSReal)qi1;
			}
		}
	}
}

static void QSGate_MatMult_4x4(QSDouble* pM,QSReal* pBuf0,QSReal* pBuf1,QSReal* pBuf2,QSReal* pBuf3,int qubit_0,int qubit_1,QSUint localMask,QSUint ncols,int nLarge)
{
	QSUint mask0,mask1,i,j0,j1,j2,i0,i1,ip0,ip1,add0,add1;
	QSRealC pr0,pi0,pr1,pi1,pr2,pi2,pr3,pi3,qr0,qi0,qr1,qi1,qr2,qi2,qr3,qi3;
	QSRealC mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3;
	int shft_0;

	if(nLarge == 0){
		add0 = (1ull << qubit_0);
		add1 = (1ull << qubit_1);
		mask0 = add0 - 1;
		mask1 = add1 - 1;

#pragma omp parallel for private(i,j0,j1,j2,i0,i1,ip0,ip1,pr0,pi0,pr1,pi1,pr2,pi2,pr3,pi3,qr0,qi0,qr1,qi1,qr2,qi2,qr3,qi3,mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3)
		for(i=0;i<ncols;i++){
			j0 = (i & mask0);
			j1 = (((i - j0) << 1) & mask1);
			j2 = ((i >> (qubit_1 - 1)) << (qubit_1 + 1));

			i0 = j0 + j1 + j2;
			ip0 = i0 + add0;
			i1 = i0 + add1;
			ip1 = i1 + add0;

			pr0 = (QSRealC)pBuf0[i0*2];
			pi0 = (QSRealC)pBuf0[i0*2+1];
			pr1 = (QSRealC)pBuf0[ip0*2];
			pi1 = (QSRealC)pBuf0[ip0*2+1];
			pr2 = (QSRealC)pBuf0[i1*2];
			pi2 = (QSRealC)pBuf0[i1*2+1];
			pr3 = (QSRealC)pBuf0[ip1*2];
			pi3 = (QSRealC)pBuf0[ip1*2+1];

#ifdef QSIM_COL_MAJOR	//for Aer
			mr0 = pM[0];
			mi0 = pM[1];
			mr1 = pM[8];
			mi1 = pM[9];
			mr2 = pM[16];
			mi2 = pM[17];
			mr3 = pM[24];
			mi3 = pM[25];
#else
			mr0 = pM[0];
			mi0 = pM[1];
			mr1 = pM[2];
			mi1 = pM[3];
			mr2 = pM[4];
			mi2 = pM[5];
			mr3 = pM[6];
			mi3 = pM[7];
#endif

			qr0 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
			qi0 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

			pBuf0[i0*2  ] = (QSReal)qr0;
			pBuf0[i0*2+1] = (QSReal)qi0;

#ifdef QSIM_COL_MAJOR	//for Aer
			mr0 = pM[2];
			mi0 = pM[3];
			mr1 = pM[10];
			mi1 = pM[11];
			mr2 = pM[18];
			mi2 = pM[19];
			mr3 = pM[26];
			mi3 = pM[27];
#else
			mr0 = pM[8];
			mi0 = pM[9];
			mr1 = pM[10];
			mi1 = pM[11];
			mr2 = pM[12];
			mi2 = pM[13];
			mr3 = pM[14];
			mi3 = pM[15];
#endif

			qr1 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
			qi1 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

			pBuf0[ip0*2  ] = (QSReal)qr1;
			pBuf0[ip0*2+1] = (QSReal)qi1;

#ifdef QSIM_COL_MAJOR	//for Aer
			mr0 = pM[4];
			mi0 = pM[5];
			mr1 = pM[12];
			mi1 = pM[13];
			mr2 = pM[20];
			mi2 = pM[21];
			mr3 = pM[28];
			mi3 = pM[29];
#else
			mr0 = pM[16];
			mi0 = pM[17];
			mr1 = pM[18];
			mi1 = pM[19];
			mr2 = pM[20];
			mi2 = pM[21];
			mr3 = pM[22];
			mi3 = pM[23];
#endif

			qr2 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
			qi2 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

			pBuf0[i1*2  ] = (QSReal)qr2;
			pBuf0[i1*2+1] = (QSReal)qi2;

#ifdef QSIM_COL_MAJOR	//for Aer
			mr0 = pM[6];
			mi0 = pM[7];
			mr1 = pM[14];
			mi1 = pM[15];
			mr2 = pM[22];
			mi2 = pM[23];
			mr3 = pM[30];
			mi3 = pM[31];
#else
			mr0 = pM[24];
			mi0 = pM[25];
			mr1 = pM[26];
			mi1 = pM[27];
			mr2 = pM[28];
			mi2 = pM[29];
			mr3 = pM[30];
			mi3 = pM[31];
#endif

			qr3 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
			qi3 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

			pBuf0[ip1*2  ] = (QSReal)qr3;
			pBuf0[ip1*2+1] = (QSReal)qi3;
		}
	}
	else{
		if(nLarge == 1){
			add0 = (1ull << qubit_0);
			mask0 = add0 - 1;
			shft_0 = 1;
		}
		else{
			add0 = 0;
			mask0 = 0;
			shft_0 = 0;
		}

#pragma omp parallel for private(i,j0,j1,i0,i1,pr0,pi0,pr1,pi1,pr2,pi2,pr3,pi3,qr0,qi0,qr1,qi1,qr2,qi2,qr3,qi3,mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3)
		for(i=0;i<ncols;i++){
			j0 = (i & mask0);
			j1 = (i - j0) << shft_0;

			i0 = j0 + j1;
			i1 = i0 + add0;

			pr0 = (QSRealC)pBuf0[i0*2];
			pi0 = (QSRealC)pBuf0[i0*2+1];
			pr1 = (QSRealC)pBuf1[i1*2];
			pi1 = (QSRealC)pBuf1[i1*2+1];
			pr2 = (QSRealC)pBuf2[i0*2];
			pi2 = (QSRealC)pBuf2[i0*2+1];
			pr3 = (QSRealC)pBuf3[i1*2];
			pi3 = (QSRealC)pBuf3[i1*2+1];

			if(localMask & 1){
#ifdef QSIM_COL_MAJOR	//for Aer
				mr0 = pM[0];
				mi0 = pM[1];
				mr1 = pM[8];
				mi1 = pM[9];
				mr2 = pM[16];
				mi2 = pM[17];
				mr3 = pM[24];
				mi3 = pM[25];
#else
				mr0 = pM[0];
				mi0 = pM[1];
				mr1 = pM[2];
				mi1 = pM[3];
				mr2 = pM[4];
				mi2 = pM[5];
				mr3 = pM[6];
				mi3 = pM[7];
#endif

				qr0 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
				qi0 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

				pBuf0[i0*2  ] = (QSReal)qr0;
				pBuf0[i0*2+1] = (QSReal)qi0;
			}

			if(localMask & 2){
#ifdef QSIM_COL_MAJOR	//for Aer
				mr0 = pM[2];
				mi0 = pM[3];
				mr1 = pM[10];
				mi1 = pM[11];
				mr2 = pM[18];
				mi2 = pM[19];
				mr3 = pM[26];
				mi3 = pM[27];
#else
				mr0 = pM[8];
				mi0 = pM[9];
				mr1 = pM[10];
				mi1 = pM[11];
				mr2 = pM[12];
				mi2 = pM[13];
				mr3 = pM[14];
				mi3 = pM[15];
#endif

				qr1 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
				qi1 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

				pBuf1[i1*2  ] = (QSReal)qr1;
				pBuf1[i1*2+1] = (QSReal)qi1;
			}

			if(localMask & 4){
#ifdef QSIM_COL_MAJOR	//for Aer
				mr0 = pM[4];
				mi0 = pM[5];
				mr1 = pM[12];
				mi1 = pM[13];
				mr2 = pM[20];
				mi2 = pM[21];
				mr3 = pM[28];
				mi3 = pM[29];
#else
				mr0 = pM[16];
				mi0 = pM[17];
				mr1 = pM[18];
				mi1 = pM[19];
				mr2 = pM[20];
				mi2 = pM[21];
				mr3 = pM[22];
				mi3 = pM[23];
#endif

				qr2 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
				qi2 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

				pBuf2[i0*2  ] = (QSReal)qr2;
				pBuf2[i0*2+1] = (QSReal)qi2;
			}

			if(localMask & 8){
#ifdef QSIM_COL_MAJOR	//for Aer
				mr0 = pM[6];
				mi0 = pM[7];
				mr1 = pM[14];
				mi1 = pM[15];
				mr2 = pM[22];
				mi2 = pM[23];
				mr3 = pM[30];
				mi3 = pM[31];
#else
				mr0 = pM[24];
				mi0 = pM[25];
				mr1 = pM[26];
				mi1 = pM[27];
				mr2 = pM[28];
				mi2 = pM[29];
				mr3 = pM[30];
				mi3 = pM[31];
#endif

				qr3 = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3;
				qi3 = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3;

				pBuf3[i1*2  ] = (QSReal)qr3;
				pBuf3[i1*2+1] = (QSReal)qi3;
			}
		}
	}
}


#if 0
static void QSGate_MatMult_8x8(QSDouble* pMin,QSComplex** ppBuf_in,int qubit_0,int qubit_1,int qubit_2,QSUint localMask,QSUint ncols,int nLarge)
{
	QSUint mask0,mask1,mask2,add0,add1,add2;
	QSUint i,j,j0,j1,j2,j3;
	QSUint pos[8];
	QSRealC pr0,pi0,pr1,pi1,pr2,pi2,pr3,pi3;
	QSRealC pr4,pi4,pr5,pi5,pr6,pi6,pr7,pi7;
	QSRealC mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3;
	QSRealC mr4,mi4,mr5,mi5,mr6,mi6,mr7,mi7;
	QSRealC qr,qi;
	QSComplex* ppBuf[8];
	QSDouble* pM;

	add0 = (1ull << qubit_0);
	add1 = (1ull << qubit_1);
	add2 = (1ull << qubit_2);
	mask0 = add0 - 1;
	mask1 = add1 - 1;
	mask2 = add2 - 1;

	if(nLarge == 0){
		ppBuf[0] = ppBuf_in[0];


#pragma omp parallel for private(i,j,pos,j0,j1,j2,j3, pr0,pi0,pr1,pi1,pr2,pi2,pr3,pi3,pr4,pi4,pr5,pi5,pr6,pi6,pr7,pi7,qr,qi, mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3,mr4,mi4,mr5,mi5,mr6,mi6,mr7,mi7,pM)
		for(i=0;i<ncols;i++){
			pM = pMin;

			j = i;
			j0 = j & mask0;
			j = (j - j0) << 1;
			j1 = j & mask1;
			j = (j - j1) << 1;
			j2 = j & mask2;
			j = (j - j2) << 1;
			j3 = j;

			pos[0] = j0 + j1 + j2 + j3;
			pos[1] = pos[0] + add0;
			pos[2] = pos[0] + add1;
			pos[3] = pos[2] + add0;
			pos[4] = pos[0] + add2;
			pos[5] = pos[4] + add0;
			pos[6] = pos[4] + add1;
			pos[7] = pos[6] + add0;

			pr0 = (QSRealC)*((QSReal*)ppBuf[0] + pos[0]*2  );
			pi0 = (QSRealC)*((QSReal*)ppBuf[0] + pos[0]*2+1);
			pr1 = (QSRealC)*((QSReal*)ppBuf[0] + pos[1]*2  );
			pi1 = (QSRealC)*((QSReal*)ppBuf[0] + pos[1]*2+1);
			pr2 = (QSRealC)*((QSReal*)ppBuf[0] + pos[2]*2  );
			pi2 = (QSRealC)*((QSReal*)ppBuf[0] + pos[2]*2+1);
			pr3 = (QSRealC)*((QSReal*)ppBuf[0] + pos[3]*2  );
			pi3 = (QSRealC)*((QSReal*)ppBuf[0] + pos[3]*2+1);
			pr4 = (QSRealC)*((QSReal*)ppBuf[0] + pos[4]*2  );
			pi4 = (QSRealC)*((QSReal*)ppBuf[0] + pos[4]*2+1);
			pr5 = (QSRealC)*((QSReal*)ppBuf[0] + pos[5]*2  );
			pi5 = (QSRealC)*((QSReal*)ppBuf[0] + pos[5]*2+1);
			pr6 = (QSRealC)*((QSReal*)ppBuf[0] + pos[6]*2  );
			pi6 = (QSRealC)*((QSReal*)ppBuf[0] + pos[6]*2+1);
			pr7 = (QSRealC)*((QSReal*)ppBuf[0] + pos[7]*2  );
			pi7 = (QSRealC)*((QSReal*)ppBuf[0] + pos[7]*2+1);

			for(j=0;j<8;j++){
				mr0 = pM[0];
				mi0 = pM[1];
				mr1 = pM[2];
				mi1 = pM[3];
				mr2 = pM[4];
				mi2 = pM[5];
				mr3 = pM[6];
				mi3 = pM[7];
				mr4 = pM[8];
				mi4 = pM[9];
				mr5 = pM[10];
				mi5 = pM[11];
				mr6 = pM[12];
				mi6 = pM[13];
				mr7 = pM[14];
				mi7 = pM[15];
				pM += 16;

				qr = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3
				   + mr4 * pr4 - mi4 * pi4 + mr5 * pr5 - mi5 * pi5 + mr6 * pr6 - mi6 * pi6 + mr7 * pr7 - mi7 * pi7;
				qi = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3
				   + mr4 * pi4 + mi4 * pr4 + mr5 * pi5 + mi5 * pr5 + mr6 * pi6 + mi6 * pr6 + mr7 * pi7 + mi7 * pr7;

				*((QSReal*)ppBuf[0] + pos[j]*2  ) = (QSReal)qr;
				*((QSReal*)ppBuf[0] + pos[j]*2+1) = (QSReal)qi;
			}
		}
	}
	else{
		if(nLarge == 1){
			localMask = (localMask & 1) | ((localMask & 1) << 1) | ((localMask & 1) << 2) | ((localMask & 1) << 3) | ((localMask & 2) << 3) | ((localMask & 2) << 4) | ((localMask & 2) << 5) | ((localMask & 2) << 6);

			add0 = (1ull << qubit_0);
			add1 = (1ull << qubit_1);
			mask0 = add0 - 1;
			mask1 = add1 - 1;

			ppBuf[0] = ppBuf_in[0];
			ppBuf[1] = ppBuf_in[0];
			ppBuf[2] = ppBuf_in[0];
			ppBuf[3] = ppBuf_in[0];
			ppBuf[4] = ppBuf_in[1];
			ppBuf[5] = ppBuf_in[1];
			ppBuf[6] = ppBuf_in[1];
			ppBuf[7] = ppBuf_in[1];
		}
		else if(nLarge == 2){	// qubit[1] >= unitBits
			localMask = (localMask & 1) | ((localMask & 1) << 1) | ((localMask & 2) << 1) | ((localMask & 2) << 2) | ((localMask & 4) << 2) | ((localMask & 4) << 3) | ((localMask & 8) << 3) | ((localMask & 8) << 4);

			add0 = (1ull << qubit_0);
			mask0 = add0 - 1;

			ppBuf[0] = ppBuf_in[0];
			ppBuf[1] = ppBuf_in[0];
			ppBuf[2] = ppBuf_in[1];
			ppBuf[3] = ppBuf_in[1];
			ppBuf[4] = ppBuf_in[2];
			ppBuf[5] = ppBuf_in[2];
			ppBuf[6] = ppBuf_in[3];
			ppBuf[7] = ppBuf_in[3];
		}
		else{	//nLarge = 3, qubit[0] >= unitBits
			ppBuf[0] = ppBuf_in[0];
			ppBuf[1] = ppBuf_in[1];
			ppBuf[2] = ppBuf_in[2];
			ppBuf[3] = ppBuf_in[3];
			ppBuf[4] = ppBuf_in[4];
			ppBuf[5] = ppBuf_in[5];
			ppBuf[6] = ppBuf_in[6];
			ppBuf[7] = ppBuf_in[7];
		}

#pragma omp parallel for private(i,j,pos,j0,j1,j2,j3, pr0,pi0,pr1,pi1,pr2,pi2,pr3,pi3,pr4,pi4,pr5,pi5,pr6,pi6,pr7,pi7,qr,qi, mr0,mi0,mr1,mi1,mr2,mi2,mr3,mi3,mr4,mi4,mr5,mi5,mr6,mi6,mr7,mi7,pM)
		for(i=0;i<ncols;i++){
			pM = pMin;

			if(nLarge == 1){
				j = i;
				j0 = j & mask0;
				j = (j - j0) << 1;
				j1 = j & mask1;
				j = (j - j1) << 1;
				j2 = j;

				pos[0] = j0 + j1 + j2;
				pos[1] = pos[0] + add0;
				pos[2] = pos[0] + add1;
				pos[3] = pos[2] + add0;
				pos[4] = pos[0];
				pos[5] = pos[1];
				pos[6] = pos[2];
				pos[7] = pos[3];
			}
			else if(nLarge == 2){	// qubit[1] >= unitBits
				j0 = i & mask0;
				j1 = (i - j0) << 1;

				pos[0] = j0 + j1;
				pos[1] = pos[0] + add0;
				pos[2] = pos[0];
				pos[3] = pos[1];
				pos[4] = pos[0];
				pos[5] = pos[1];
				pos[6] = pos[0];
				pos[7] = pos[1];
			}
			else{	//nLarge = 3, qubit[0] >= unitBits
				pos[0] = i;
				pos[1] = i;
				pos[2] = i;
				pos[3] = i;
				pos[4] = i;
				pos[5] = i;
				pos[6] = i;
				pos[7] = i;
			}

			pr0 = (QSRealC)*((QSReal*)ppBuf[0] + pos[0]*2  );
			pi0 = (QSRealC)*((QSReal*)ppBuf[0] + pos[0]*2+1);
			pr1 = (QSRealC)*((QSReal*)ppBuf[1] + pos[1]*2  );
			pi1 = (QSRealC)*((QSReal*)ppBuf[1] + pos[1]*2+1);
			pr2 = (QSRealC)*((QSReal*)ppBuf[2] + pos[2]*2  );
			pi2 = (QSRealC)*((QSReal*)ppBuf[2] + pos[2]*2+1);
			pr3 = (QSRealC)*((QSReal*)ppBuf[3] + pos[3]*2  );
			pi3 = (QSRealC)*((QSReal*)ppBuf[3] + pos[3]*2+1);
			pr4 = (QSRealC)*((QSReal*)ppBuf[4] + pos[4]*2  );
			pi4 = (QSRealC)*((QSReal*)ppBuf[4] + pos[4]*2+1);
			pr5 = (QSRealC)*((QSReal*)ppBuf[5] + pos[5]*2  );
			pi5 = (QSRealC)*((QSReal*)ppBuf[5] + pos[5]*2+1);
			pr6 = (QSRealC)*((QSReal*)ppBuf[6] + pos[6]*2  );
			pi6 = (QSRealC)*((QSReal*)ppBuf[6] + pos[6]*2+1);
			pr7 = (QSRealC)*((QSReal*)ppBuf[7] + pos[7]*2  );
			pi7 = (QSRealC)*((QSReal*)ppBuf[7] + pos[7]*2+1);

			for(j=0;j<8;j++){
				mr0 = pM[0];
				mi0 = pM[1];
				mr1 = pM[2];
				mi1 = pM[3];
				mr2 = pM[4];
				mi2 = pM[5];
				mr3 = pM[6];
				mi3 = pM[7];
				mr4 = pM[8];
				mi4 = pM[9];
				mr5 = pM[10];
				mi5 = pM[11];
				mr6 = pM[12];
				mi6 = pM[13];
				mr7 = pM[14];
				mi7 = pM[15];
				pM += 16;

				qr = mr0 * pr0 - mi0 * pi0 + mr1 * pr1 - mi1 * pi1 + mr2 * pr2 - mi2 * pi2 + mr3 * pr3 - mi3 * pi3
				   + mr4 * pr4 - mi4 * pi4 + mr5 * pr5 - mi5 * pi5 + mr6 * pr6 - mi6 * pi6 + mr7 * pr7 - mi7 * pi7;
				qi = mr0 * pi0 + mi0 * pr0 + mr1 * pi1 + mi1 * pr1 + mr2 * pi2 + mi2 * pr2 + mr3 * pi3 + mi3 * pr3
				   + mr4 * pi4 + mi4 * pr4 + mr5 * pi5 + mi5 * pr5 + mr6 * pi6 + mi6 * pr6 + mr7 * pi7 + mi7 * pr7;

				*((QSReal*)ppBuf[j] + pos[j]*2  ) = (QSReal)qr;
				*((QSReal*)ppBuf[j] + pos[j]*2+1) = (QSReal)qi;
			}
		}
	}
}

#endif

static void QSGate_MatMult_NxN(QSDouble* pM,QSComplex** ppBuf_in,int* qubits,int nqubits,QSUint localMask,QSUint ncols,int nLarge)
{
	QSUint mask,add;
	QSUint i,ii,idx,t;
	int j,k,l,matSize;
	QSRealC pr[QS_MAX_MATRIX_SIZE];
	QSRealC pi[QS_MAX_MATRIX_SIZE];
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
			if((k >> j) & 1){
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

		for(k=0;k<matSize;k++){
			pr[k] = (QSRealC)*(ppBuf[k] + idx  );
			pi[k] = (QSRealC)*(ppBuf[k] + idx+1);
		}

		for(j=0;j<matSize;j++){
			if((localMask >> j) & 1){
				qr = 0.0;
				qi = 0.0;
				for(k=0;k<matSize;k++){
#ifdef QSIM_COL_MAJOR	//for Aer
					l = (j + (k << nqubits)) << 1;
#else
					l = (k + (j << nqubits)) << 1;
#endif
					mr = pM[l];
					mi = pM[l+1];

					qr += mr*pr[k] - mi*pi[k];
					qi += mr*pi[k] + mi*pr[k];
				}

				*(ppBuf[j] + idx  ) = (QSReal)qr;
				*(ppBuf[j] + idx+1) = (QSReal)qi;
			}
		}
	}
}



void QSGate_MatMult::ExecuteOnHost(QSUnitStorage* pUnit,QSUint* pGuid,QSComplex** ppBuf,int* qubits,int* qubits_c,int nqubits,int nqubitsLarge,QSUint localMask,int nTrans)
{
	QSUint i,ncols;

	ncols = 1ull << (pUnit->UnitBits() - (nqubits - nqubitsLarge));
	if(nqubits == 1){
		if(nqubitsLarge == 0){
			QSGate_MatMult_2x2((QSDouble*)m_Mat,(QSReal*)ppBuf[0],(QSReal*)ppBuf[0],qubits[0],3,ncols);
		}
		else{
			QSGate_MatMult_2x2((QSDouble*)m_Mat,(QSReal*)ppBuf[0],(QSReal*)ppBuf[1],qubits[0],localMask,ncols);
		}
	}
	else if(nqubits == 2){
		if(nqubitsLarge == 0){
			QSGate_MatMult_4x4((QSDouble*)m_Mat,(QSReal*)ppBuf[0],(QSReal*)ppBuf[0],(QSReal*)ppBuf[0],(QSReal*)ppBuf[0],qubits[0],qubits[1],15,ncols,nqubitsLarge);
		}
		else if(nqubitsLarge == 1){
			localMask = ((localMask & 1) << 1) | (localMask & 1) | ((localMask & 2) << 1) | ((localMask & 2) << 2);
			QSGate_MatMult_4x4((QSDouble*)m_Mat,(QSReal*)ppBuf[0],(QSReal*)ppBuf[0],(QSReal*)ppBuf[1],(QSReal*)ppBuf[1],qubits[0],qubits[1],localMask,ncols,nqubitsLarge);
		}
		else{
			QSGate_MatMult_4x4((QSDouble*)m_Mat,(QSReal*)ppBuf[0],(QSReal*)ppBuf[1],(QSReal*)ppBuf[2],(QSReal*)ppBuf[3],qubits[0],qubits[1],localMask,ncols,nqubitsLarge);
		}
	}
//	else if(nqubits == 3){
//		QSGate_MatMult_8x8((QSDouble*)m_pMat,ppBuf,qubits[0],qubits[1],qubits[2],localMask,ncols,nqubitsLarge);
//	}
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
		QSGate_MatMult_NxN((QSDouble*)m_Mat,ppBuf,qubits,nqubits,mask,ncols,nqubitsLarge);
	}

}



