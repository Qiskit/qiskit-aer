/*
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
*/

#ifndef __CUDA_REDUCTION_H__
#define __CUDA_REDUCTION_H__


__inline__ __device__ double warpReduceSum(double val)
{
	int i;
	for(i=16;i>0;i/=2){
		val += __shfl_xor_sync(0xffffffff,val,i,32);
	}
	return val;
}

__inline__ __device__ double blockReduceSum(double val) 
{
	__shared__ double buf[32];
	int lid = threadIdx.x & 0x1f;
	int wid = threadIdx.x >> 5;

	val = warpReduceSum(val);

	if(lid == 0)
		buf[wid] = val;

	__syncthreads();

	val = (threadIdx.x < (blockDim.x >> 5)) ? buf[lid] : 0;
	if(wid == 0){
		val = warpReduceSum(val);
	}

	return val;
}







#endif	//__CUDA_REDUCTION_H__
