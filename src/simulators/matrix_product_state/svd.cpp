/**
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

/*
 * Adapted from: P. A. Businger and G. H. Golub, Comm. ACM 12, 564 (1969)
*/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <complex>
#include <cassert>
#include "svd.hpp"
#include "framework/utils.hpp"
#include "framework/linalg/almost_equal.hpp"

#define mul_factor 1e2
#define tiny_factor 1e30
#define THRESHOLD 1e-9
#define NUM_SVD_TRIES 15

namespace AER {

cmatrix_t diag(rvector_t S, uint_t m, uint_t n);

cmatrix_t diag(rvector_t S, uint_t m, uint_t n)
{
	cmatrix_t Res = cmatrix_t(m, n);
	for(uint_t i = 0; i < m; i++)
	{
		for(uint_t j = 0; j < n; j++)
		{
			Res(i,j) = (i==j ? complex_t(S[i]) : 0);
		}
	}
	return Res;
}

cmatrix_t reshape_before_SVD(std::vector<cmatrix_t> data)
{
//	Turns 4 matrices A0,A1,A2,A3 to big matrix:
//	A0 A1
//	A2 A3
	cmatrix_t temp1 = AER::Utils::concatenate(data[0], data[1], 1),
		  temp2 = AER::Utils::concatenate(data[2], data[3], 1);
	return AER::Utils::concatenate(temp1, temp2, 0);
}
std::vector<cmatrix_t> reshape_U_after_SVD(const cmatrix_t U)
{
  std::vector<cmatrix_t> Res(2);
  AER::Utils::split(U, Res[0], Res[1], 0);
  return Res;
}
std::vector<cmatrix_t> reshape_V_after_SVD(const cmatrix_t V)
{
  std::vector<cmatrix_t> Res(2);
  AER::Utils::split(AER::Utils::dagger(V), Res[0], Res[1] ,1);
  return Res;
}


//-------------------------------------------------------------
// function name: num_of_SV
// Description: Computes the number of none-zero singular values
//				in S
// Parameters: rvector_t S - vector of singular values from the
//			   SVD decomposition
// Returns: number of elements in S that are greater than 0
//			(actually greater than threshold)
//-------------------------------------------------------------
uint_t num_of_SV(rvector_t S, double threshold)
{
	uint_t sum = 0;
	for(uint_t i = 0; i < S.size(); ++i)
	{
	  if(std::norm(S[i]) > threshold)
		sum++;
	}
	if (sum == 0)
	  std::cout << "SV_Num == 0"<< '\n';
	return sum;
}

void reduce_zeros(cmatrix_t &U, rvector_t &S, cmatrix_t &V) {
  uint_t SV_num = num_of_SV(S, 1e-16);
  U.resize(U.GetRows(), SV_num);
  S.resize(SV_num);
  V.resize(V.GetRows(), SV_num);
}

// added cut-off at the end
status csvd(cmatrix_t &A, cmatrix_t &U,rvector_t &S,cmatrix_t &V)
{
  int m = A.GetRows(), n = A.GetColumns(), size = std::max(m,n);
  rvector_t b(size,0.0), c(size,0.0), t(size,0.0);
  double cs = 0.0, eps = 0.0, f = 0.0 ,g = 0.0, h = 0.0, sn = 0.0 , w = 0.0, x = 0.0, y = 0.0, z = 0.0;
  double eta = 1e-10, tol = 1.5e-34;
  // using int and not uint_t because uint_t caused bugs in loops with condition of >= 0
  int i = 0, j = 0, k = 0, k1 = 0, l = 0, l1 = 0;
  complex_t q = 0;
  // Transpose when m < n
  bool transposed = false;
  if (m < n)
    {
    	transposed = true;
    	A = AER::Utils::dagger(A);
	std::swap(m,n);
    }
	cmatrix_t temp_A = A;
	c[0] = 0;
	while(true)
	{
		k1 = k + 1;
		z = 0.0;
		for( i = k; i < m; i++){
			z = z + norm(A(i,k));
		}
		b[k] = 0.0;
		if ( tol < z )
		{
			z = std::sqrt( z );
			b[k] = z;
			w = std::abs( A(k,k) );
			if (Linalg::almost_equal(static_cast<long double>(w), 
						 static_cast<long double>(0.0) )) {
				q = complex_t( 1.0, 0.0 );
			}
			else {
				q = A(k,k) / w;
			}
			A(k,k) = q * ( z + w );

			if ( k != n - 1 )
			{
				for( j = k1; j < n ; j++)
				{

					q = complex_t( 0.0, 0.0 );
					for( i = k; i < m; i++){
						q = q + std::conj( A(i,k) ) * A(i,j);
					}
					q = q / ( z * ( z + w ) );

					for( i = k; i < m; i++){
					  A(i,j) = A(i,j) - q * A(i,k);
					}

				}
//
// Phase transformation.
//
				q = -std::conj(A(k,k))/std::abs(A(k,k));

				for( j = k1; j < n; j++){
					A(k,j) = q * A(k,j);
				}
			}
        }
		if ( k == n - 1 ) break;

		z = 0.0;
		for( j = k1; j < n; j++){
			z = z + norm(A(k,j));
		}
		c[k1] = 0.0;

		if ( tol < z )
		{
			z = std::sqrt( z );
			c[k1] = z;
			w = std::abs( A(k,k1) );

			if (Linalg::almost_equal(static_cast<long double>(w), 
						 static_cast<long double>(0.0) )){
				q = complex_t( 1.0, 0.0 );
			}
			else{
				q = A(k,k1) / w;
			}
			A(k,k1) = q * ( z + w );

			for( i = k1; i < m; i++)
			{
				q = complex_t( 0.0, 0.0 );

				for( j = k1; j < n; j++){
					q = q + std::conj( A(k,j) ) * A(i,j);
				}
				q = q / ( z * ( z + w ) );

				for( j = k1; j < n; j++){
					A(i,j) = A(i,j) - q * A(k,j);
				}
			}
//
// Phase transformation.
//
			q = -std::conj(A(k,k1) )/std::abs(A(k,k1));
			for( i = k1; i < m; i++){
				A(i,k1) = A(i,k1) * q;
			}
		}
		k = k1;
    }

    eps = 0.0;
	for( k = 0; k < n; k++)
	{
		S[k] = b[k];
		t[k] = c[k];
		eps = std::max( eps, S[k] + t[k] );
	}
	eps = eps * eta;

//
// Initialization of U and V.
//
	U.initialize(m, m);
	V.initialize(n, n);
	for( j = 0; j < m; j++)
	{
		for( i = 0; i < m; i++){
			U(i,j) = complex_t( 0.0, 0.0 );
		}
		U(j,j) = complex_t( 1.0, 0.0 );
	}

	for( j = 0; j < n; j++)
	{
		for( i = 0; i < n; i++){
			V(i,j) = complex_t( 0.0, 0.0 );
		}
		V(j,j) = complex_t( 1.0, 0.0 );
	}



	for( k = n-1; k >= 0; k--)
	{
		while(true)
		{
			bool jump = false;
			for( l = k; l >= 0; l--)
			{

				if ( std::abs( t[l] ) < eps )
				{
					jump = true;
					break;
				}
				else if ( std::abs( S[l-1] ) < eps ) {
					break;
				}
			}
			if(!jump)
			{
				cs = 0.0;
				sn = 1.0;
				l1 = l - 1;

				for( i = l; i <= k; i++)
				{
					f = sn * t[i];
					t[i] = cs * t[i];

					if ( std::abs(f) < eps ) {
						break;
					}
					h = S[i];
					w = std::sqrt( f * f + h * h );
					S[i] = w;
					cs = h / w;
					sn = - f / w;

					for( j = 0; j < n; j++)
					{
						x = std::real( U(j,l1) );
						y = std::real( U(j,i) );
						U(j,l1) = complex_t( x * cs + y * sn, 0.0 );
						U(j,i)  = complex_t( y * cs - x * sn, 0.0 );
					}
				}
			}
			w = S[k];
			if ( l == k ){
				break;
			}
			x = S[l];
			y = S[k-1];
			g = t[k-1];
			h = t[k];
			f = ( ( y - w ) * ( y + w ) + ( g - h ) * ( g + h ) )/ ( 2.0 * h * y );
			g = std::sqrt( f * f + 1.0 );
			if ( f < -1.0e-13){ //if ( f < 0.0){ //didn't work when f was negative very close to 0 (because of numerical reasons)
				g = -g;
			}
			f = ( ( x - w ) * ( x + w ) + ( y / ( f + g ) - h ) * h ) / x;
			cs = 1.0;
			sn = 1.0;
			l1 = l + 1;
			for( i = l1; i <= k; i++)
			{
				g = t[i];
				y = S[i];
				h = sn * g;
				g = cs * g;
				w = std::sqrt( h * h + f * f );
				if (Linalg::almost_equal(static_cast<long double>(w), 
							 static_cast<long double>(0.0) )) {
#ifdef DEBUG
				  std::cout << "ERROR 1: w is exactly 0: h = " << h << " , f = " << f << std::endl;
				  std::cout << " w = " << w << std::endl;
#endif
				}
				t[i-1] = w;
				cs = f / w;
				sn = h / w;
				f = x * cs + g * sn; // might be 0

				long double large_f = 0;
				if (Linalg::almost_equal(static_cast<long double>(f), 
							 static_cast<long double>(0.0) )) {
#ifdef DEBUG
				  std::cout << "f == 0 because " << "x = " << x << ", cs = " << cs << ", g = " << g << ", sn = " << sn  <<std::endl;
#endif
				  long double large_x =   x * tiny_factor;
				  long double large_g =   g * tiny_factor;
				  long double large_cs = cs * tiny_factor;
				  long double large_sn = sn * tiny_factor;
				  large_f = large_x * large_cs + large_g * large_sn;

#ifdef DEBUG
				  std::cout << large_x * large_cs <<std::endl;;
				  std::cout << large_g * large_sn <<std::endl;
				  std::cout << "new f = " << large_f << std::endl;

#endif
				}
				g = g * cs - x * sn;
				h = y * sn; // h == 0 because y==0
				y = y * cs;

				for( j = 0; j < n; j++)
				{
					x = std::real( V(j,i-1) );
					w = std::real( V(j,i) );
					V(j,i-1) = complex_t( x * cs + w * sn, 0.0 );
					V(j,i)   = complex_t( w * cs - x * sn, 0.0 );
				}

				bool tiny_w = false;
#ifdef DEBUG
				std::cout << " h = " << h << " f = " << f << " large_f = " << large_f << std::endl;
#endif
				if (std::abs(h) < 1e-13 && std::abs(f) < 1e-13 && 
				    !Linalg::almost_equal(large_f, 
							  static_cast<long double>(0.0))) {
				  tiny_w = true;
				} else {
				  w = std::sqrt( h * h + f * f );
				}
				w = std::sqrt( h * h + f * f );
				if (Linalg::almost_equal(static_cast<long double>(w), 
							 static_cast<long double>(0.0)) && !tiny_w) {

#ifdef DEBUG
				  std::cout << "ERROR: w is exactly 0: h = " << h << " , f = " << f << std::endl;
				  std::cout << " w = " << w << std::endl;
#endif
				  return FAILURE;
				}

				S[i-1] = w;
				if (tiny_w) {
				  cs = 1.0; // because h==0, so w = f
				  sn = 0;
				} else {
				  cs = f / w;
				  sn = h / w;
				}

				f = cs * g + sn * y;
				x = cs * y - sn * g;
				for( j = 0; j < n; j++)
				{
					y = std::real( U(j,i-1) );
					w = std::real( U(j,i) );
					U(j,i-1) = complex_t( y * cs + w * sn, 0.0 );
					U(j,i)   = complex_t( w * cs - y * sn, 0.0 );
				}
			}
			t[l] = 0.0;
			t[k] = f;
			S[k] = x;
		}


        if ( w < -1e-13 ) //
		{
			S[k] = - w;
			for( j = 0; j < n; j++){
				V(j,k) = - V(j,k);
			}
		}
	}

//
//  Sort the singular values.
//
	for( k = 0; k < n; k++)
	{
		g = - 1.0;
		j = k;
        for( i = k; i < n; i++)
		{
			if ( g < S[i] )
			{
				g = S[i];
				j = i;
			}
        }

        if ( j != k )
		{
			S[j] = S[k];
			S[k] = g;

			for( i = 0; i < n; i++)
			{
				q      = V(i,j);
				V(i,j) = V(i,k);
				V(i,k) = q;
			}

			for( i = 0; i < n; i++)
			{
				q      = U(i,j);
				U(i,j) = U(i,k);
				U(i,k) = q;
			}
        }
    }

    for( k = n-1 ; k >= 0; k--)
	{
	  if (!Linalg::almost_equal(static_cast<long double>(b[k]), 
				    static_cast<long double>(0.0)) )
		{
			q = -A(k,k) / std::abs( A(k,k) );
			for( j = 0; j < m; j++){
				U(k,j) = q * U(k,j);
			}
			for( j = 0; j < m; j++)
			{
				q = complex_t( 0.0, 0.0 );
				for( i = k; i < m; i++){
					q = q + std::conj( A(i,k) ) * U(i,j);
				}
				q = q / ( std::abs( A(k,k) ) * b[k] );
				for( i = k; i < m; i++){
					U(i,j) = U(i,j) - q * A(i,k);
				}
			}
		}
	}

	for( k = n-1 -1; k >= 0; k--)
	{
		k1 = k + 1;
		if ( !Linalg::almost_equal(static_cast<long double>(c[k1]), 
					   static_cast<long double>(0.0) ))
		{
			q = -std::conj( A(k,k1) ) / std::abs( A(k,k1) );

			for( j = 0; j < n; j++){
				V(k1,j) = q * V(k1,j);
			}

			for( j = 0; j < n; j++)
			{
				q = complex_t( 0.0, 0.0 );
				for( i = k1 ; i < n; i++){
					q = q + A(k,i) * V(i,j);
				}
				q = q / ( std::abs( A(k,k1) ) * c[k1] );
				for( i = k1; i < n; i++){
					V(i,j) = V(i,j) - q * std::conj( A(k,i) );
				}
			}
		}
	}

	// Check if SVD output is wrong
	cmatrix_t diag_S = diag(S,m,n);
	cmatrix_t temp = U*diag_S;
	temp = temp * AER::Utils::dagger(V);
	const auto nrows = temp_A.GetRows();
	const auto ncols = temp_A.GetColumns();
	bool equal = true;

	for (uint_t ii=0; ii < nrows; ii++)
	    for (uint_t jj=0; jj < ncols; jj++)
	      if (std::real(std::abs(temp_A(ii, jj) - temp(ii, jj))) > THRESHOLD)
	      {
	    	  equal = false;
	      }
	if( ! equal )
	{
	  std::stringstream ss;
	  ss << "error: wrong SVD calc: A != USV*";
	  throw std::runtime_error(ss.str());
	}

	// Transpose again if m < n
	if(transposed)
	  std::swap(U,V);

	return SUCCESS;
}


void csvd_wrapper (cmatrix_t &A, cmatrix_t &U,rvector_t &S,cmatrix_t &V)
{
  cmatrix_t copied_A = A;
  int times = 0;
#ifdef DEBUG
  std::cout << "1st try" << std::endl;
#endif
  status current_status = csvd(A, U, S, V);
  if (current_status == SUCCESS) {
      return;
  }

  while(times <= NUM_SVD_TRIES && current_status == FAILURE)
    {
      times++;
      copied_A = copied_A*mul_factor;
      A = copied_A;

#ifdef DEBUG
      std::cout << "SVD trial #" << times << std::endl;
#endif

      current_status = csvd(A, U, S, V);
    }
  if(times == NUM_SVD_TRIES) {
    std::stringstream ss;
    ss << "SVD failed";
    throw std::runtime_error(ss.str());
  }

  //Divide by mul_factor every singular value after we multiplied matrix a
  for(uint_t k = 0; k < S.size(); k++)
    S[k] /= pow(mul_factor, times);

}

} // namespace AER

