/*
 * CSVD.cpp
 *
 * Adapted from: P. A. Businger and G. H. Golub, Comm. ACM 12, 564 (1969)
 *
 *
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <complex>
#include "defs_and_types.hpp"
#include "SVD.hpp"
#include "Qreg.hpp"

#define mul_factor 1e2
#define tiny_factor 1e30

using namespace std;

template <class T>
void cswap(T &a, T &b)
{
	T temp = a;
	a = b;
	b = temp;
}

template <class T>
void printMatrix(T **matrix, uint rows, uint columns){
  for(uint i = 0; i < rows; ++i)
    {
      for(uint j = 0; j < columns; ++j)
	{
//	  if( norm(matrix[i][j]) > 0.0000000000001)
	    cout << matrix[i][j] << ' ';
//	  else
//	    cout << 0 << ' ';
	}
      cout << endl;
    }
}

void cDagger(complex_t a[NN][NN], int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		a[i][i] = std::conj(a[i][i]);
		for (int j = i+1; j < n; j++)
		{
			a[i][j] = std::conj(a[i][j]);
			a[j][i] = std::conj(a[j][i]);
			cswap(a[i][j],a[j][i]);
		}
	}
}

template <class T>
void copyMatrix(T **matrix, T **coppied_matrix,uint rows,uint columns)
{
	for(uint i = 0; i < rows; ++i)
	{
	  for(uint j = 0; j < columns; ++j)
	  {
		matrix[i][j] = coppied_matrix[i][j];
	  }
	}
}

void inside_csvd (complex_t** a, int m, int n, double* s, complex_t** u, complex_t** v)
{
	cout.precision(16);
//	complex_t a[m][n], u[m][m] = {{0}}, v[n][n] = {{0}} ;
//	long double s[max(m,n)];
//	for(int i = 0; i < m; i++)
//		for(int j = 0; j < n; j++)
//		{
//			a[i][j].real((long double)(A[i][j].real()) * mul_factor);
//			a[i][j].imag((long double)(A[i][j].imag()) * mul_factor);
//		}

//	for(int i = 0; i < m; i++)
//		for(int j = 0; j < n; j++)
//		{
//			a[i][j] *= mul_factor;
//		}

	double b[NN] = {0}, c[NN] = {0}, t[NN] = {0};
//	double b[max(m,n)] = {0}, c[max(m,n)] = {0}, t[max(m,n)] = {0};
	double cs = 0, eps = 0, f = 0 ,g = 0, h = 0, sn = 0 , w = 0, x = 0, y = 0, z = 0;
	double eta = 1.1920929e-20, tol = 1.5e-34;
	int i = 0, j = 0, k = 0, k1 = 0, l = 0, l1 = 0;
	complex_t q = 0;

	// Transpose when m < n
    bool transposed = false;
    if (m < n)
    {
    	transposed = true;

    	//    	cDagger(a,m,n);
		for (int i = 0; i < m; i++)
		{
			a[i][i] = std::conj(a[i][i]);
			for (int j = i+1; j < n; j++)
			{
				a[i][j] = std::conj(a[i][j]);
				a[j][i] = std::conj(a[j][i]);
				cswap(a[i][j],a[j][i]);
			}
		}

    	swap(m,n);
    }



	c[0] = 0;
	while(true)
	{
		k1 = k + 1;
		z = 0.0;
		for( i = k; i < m; i++){
			z = z + norm(a[i][k]);
		}
		b[k] = 0.0;
		if ( tol < z )
		{
			z = sqrt( z );
			b[k] = z;
			w = abs( a[k][k] );
			if ( w == 0.0 ) {
				q = complex_t( 1.0, 0.0 );
			}
			else {
				q = a[k][k] / w;
			}
			a[k][k] = q * ( z + w );

			if ( k != n - 1 )
			{
				for( j = k1; j < n ; j++)
				{

					q = complex_t( 0.0, 0.0 );
					for( i = k; i < m; i++){
						q = q + conj( a[i][k] ) * a[i][j];
					}
					q = q / ( z * ( z + w ) );

					for( i = k; i < m; i++){
					  a[i][j] = a[i][j] - q * a[i][k];
					}

				}
//
// Phase transformation.
//
				q = -conj(a[k][k])/abs(a[k][k]);

				for( j = k1; j < n; j++){
					a[k][j] = q * a[k][j];
				}
			}
        }
		if ( k == n - 1 ) break;

		z = 0.0;
		for( j = k1; j < n; j++){
			z = z + norm(a[k][j]);
		}
		c[k1] = 0.0;

		if ( tol < z )
		{
			z = sqrt( z );
			c[k1] = z;
			w = abs( a[k][k1] );

			if ( w == 0.0 ){
				q = complex_t( 1.0, 0.0 );
			}
			else{
				q = a[k][k1] / w;
			}
			a[k][k1] = q * ( z + w );

			for( i = k1; i < m; i++)
			{
				q = complex_t( 0.0, 0.0 );

				for( j = k1; j < n; j++){
					q = q + conj( a[k][j] ) * a[i][j];
				}
				q = q / ( z * ( z + w ) );

				for( j = k1; j < n; j++){
					a[i][j] = a[i][j] - q * a[k][j];
				}
			}
//
// Phase transformation.
//
			q = -conj(a[k][k1] )/abs(a[k][k1]);
			for( i = k1; i < m; i++){
				a[i][k1] = a[i][k1] * q;
			}
		}
		k = k1;
    }

	//Print Matrix
//	for(uint i = 0; i < m; ++i)
//	{
//	  for(uint j = 0; j < n; ++j)
//	{
//		cout << a[i][j] << ' ';
//	}
//	  cout << endl;
//	}

    eps = 0.0;
	for( k = 0; k < n; k++)
	{
		s[k] = b[k];
		t[k] = c[k];
//		cout << "s[" << k << "] = " << s[k] << endl;
//		cout << "t[" << k << "] = " << t[k] << endl;
		eps = max( eps, s[k] + t[k] );
	}
	eps = eps * eta;

//	cout << eps << endl;
//
// Initialization of U and V.
//
	for( j = 0; j < m; j++)
	{
		for( i = 0; i < m; i++){
			u[i][j] = complex_t( 0.0, 0.0 );
		}
		u[j][j] = complex_t( 1.0, 0.0 );
	}

	for( j = 0; j < n; j++)
	{
		for( i = 0; i < n; i++){
			v[i][j] = complex_t( 0.0, 0.0 );
		}
		v[j][j] = complex_t( 1.0, 0.0 );
	}

	for( k = n-1; k >= 0; k--)
	{
//		int while_runs = 0;
		while(true)
		{
			bool jump = false;
			for( l = k; l >= 0; l--)
			{

				if ( abs( t[l] ) < eps )
				{
					jump = true;
					break;
				}
				else if ( abs( s[l-1] ) < eps ) {
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

					if ( abs(f) < eps ) {
						break;
					}
					h = s[i];
					w = sqrt( f * f + h * h );
					s[i] = w;
					cs = h / w;
					sn = - f / w;

					for( j = 0; j < n; j++)
					{
						x = real( u[j][l1] );
						y = real( u[j][i] );
						u[j][l1] = complex_t( x * cs + y * sn, 0.0 );
						u[j][i]  = complex_t( y * cs - x * sn, 0.0 );
					}
				}
			}
			w = s[k];
			if ( l == k ){
				break;
			}
			x = s[l];
			y = s[k-1];
			g = t[k-1];
			h = t[k];
			f = ( ( y - w ) * ( y + w ) + ( g - h ) * ( g + h ) )/ ( 2.0 * h * y );
			g = sqrt( f * f + 1.0 );
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
				y = s[i];
				h = sn * g;
				g = cs * g;
				w = sqrt( h * h + f * f );
//				if (w == 0) {
//				  cout << "ERROR 2: w is exactly 0: h = " << h << " , f = " << f << endl;
//				  cout << " w = " << w << endl;
//				  assert(false);
//				}
				t[i-1] = w;
				cs = f / w;
				sn = h / w;
				f = x * cs + g * sn; // might be 0
				long double large_f = 0;
				if (f==0) {
				  if (DEBUG) cout << "f == 0 because " << "x = " << x << ", cs = " << cs << ", g = " << g << ", sn = " << sn  <<endl;
				  long double large_x =   x * tiny_factor;
				  long double large_g =   g * tiny_factor;
				  long double large_cs = cs * tiny_factor;
				  long double large_sn = sn * tiny_factor;
				  if (DEBUG) cout << large_x * large_cs <<endl;;
				  if (DEBUG) cout << large_g * large_sn <<endl;
				  large_f = large_x * large_cs + large_g * large_sn;
				  if (DEBUG) cout << "new f = " << large_f << endl;
				}

				g = g * cs - x * sn;
				h = y * sn; // h == 0 because y==0
				y = y * cs;

				for( j = 0; j < n; j++)
				{
					x = real( v[j][i-1] );
					w = real( v[j][i] );
					v[j][i-1] = complex_t( x * cs + w * sn, 0.0 );
					v[j][i]   = complex_t( w * cs - x * sn, 0.0 );
				}
				bool tiny_w = false;
				if (DEBUG) cout.precision(32);
//				if (DEBUG) cout << " h = " << h << " f = " << f << " large_f = " << large_f << endl;
				if (abs(h)  < 1e-13 && abs(f) < 1e-13 && large_f != 0) {
				  tiny_w = true;
				}
//				else {
//				  w = sqrt( h * h + f * f );
//				}
				w = sqrt( h * h + f * f );
				if (w == 0 && !tiny_w) {
//				  cout << "ERROR 2: w is exactly 0: h = " << h << " , f = " << f << endl;
//				  cout << " w = " << w << endl;
//				  assert(false);
				  throw("ERROR");
				}

				s[i-1] = w;
				if (tiny_w) {
				  if (DEBUG) cout << "tiny" <<endl;
				  cs = 1.0; // because h==0, so w = f
				  sn = 0;
				} else {
				  cs = f / w;
				  sn = h / w;
				}
//				cs = f / w;
//				sn = h / w;

				f = cs * g + sn * y;
				x = cs * y - sn * g;
				for( j = 0; j < n; j++)
				{
					y = real( u[j][i-1] );
					w = real( u[j][i] );
					u[j][i-1] = complex_t( y * cs + w * sn, 0.0 );
					u[j][i]   = complex_t( w * cs - y * sn, 0.0 );
				}
			}
			t[l] = 0.0;
			t[k] = f;
			s[k] = x;
		}


        if ( w < -1e-13 ) //
		{
			s[k] = - w;
			for( j = 0; j < n; j++){
				v[j][k] = - v[j][k];
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

//        	if (s[i] > 1.0000001) {
//			  cout << "ERROR 3: s[i] > 1 " <<endl;
//			  cout << "s[i] = " << s[i] <<endl;
//			  assert(false);
//			}
			if ( g < s[i] )
			{
				g = s[i];
				j = i;
			}
        }

        if ( j != k )
		{
			s[j] = s[k];
			s[k] = g;

			for( i = 0; i < n; i++)
			{
				q      = v[i][j];
				v[i][j] = v[i][k];
				v[i][k] = q;
			}

			for( i = 0; i < n; i++)
			{
				q      = u[i][j];
				u[i][j] = u[i][k];
				u[i][k] = q;
			}
        }
    }

    for( k = n-1 ; k >= 0; k--)
	{
		if ( b[k] != 0.0 )
		{
			q = -a[k][k] / abs( a[k][k] );
			for( j = 0; j < m; j++){
				u[k][j] = q * u[k][j];
			}
			for( j = 0; j < m; j++)
			{
				q = complex_t( 0.0, 0.0 );
				for( i = k; i < m; i++){
					q = q + conj( a[i][k] ) * u[i][j];
				}
				q = q / ( abs( a[k][k] ) * b[k] );
				for( i = k; i < m; i++){
					u[i][j] = u[i][j] - q * a[i][k];
				}

			}

		}

	}

	for( k = n-1 -1; k >= 0; k--)
	{
		k1 = k + 1;
		if ( c[k1] != 0.0 )
		{
			q = -conj( a[k][k1] ) / abs( a[k][k1] );

			for( j = 0; j < n; j++){
				v[k1][j] = q * v[k1][j];
			}

			for( j = 0; j < n; j++)
			{
				q = complex_t( 0.0, 0.0 );
				for( i = k1 ; i < n; i++){
					q = q + a[k][i] * v[i][j];
				}
				q = q / ( abs( a[k][k1] ) * c[k1] );
				for( i = k1; i < n; i++){
					v[i][j] = v[i][j] - q * conj( a[k][i] );
				}
			}
		}
	}
	// Transpose again if m < n
	if(transposed)
	{
		for (int i = 0; i < max(m,n); i++)
		{
			for (int j = 0; j < max(m,n); j++)
			{
				cswap(u[i][j],v[i][j]);
			}
		}
		swap(m,n);
	}



//	double cut_off = 1.0e-15;
//	for( i = 0; i < m; i++)
//	{
//		if(s[i] < cut_off)  s[i] = 0.0;
//		for( j = 0; j < m; j++)
//		{
//			if(abs(u[i][j].real()) < cut_off)
//				u[i][j].real(0.0);
//			if(abs(u[i][j].imag()) < cut_off)
//				u[i][j].imag(0.0);
//		}
//	}
//	for( i = 0; i < n; i++)
//		for( j = 0; j < n; j++)
//		{
//			if(abs(v[i][j].real()) < cut_off)
//				v[i][j].real(0.0);
//			if(abs(v[i][j].imag()) < cut_off)
//				v[i][j].imag(0.0);
//		}

//	for(int i = 0; i < m; i++)
//		for(int j = 0; j < m; j++)
//		{
//			U[i][j].real((double)u[i][j].real());
//			U[i][j].imag((double)u[i][j].imag());
//		}
//	for(int i = 0; i < n; i++)
//		for(int j = 0; j < n; j++)
//		{
//			V[i][j].real((double)v[i][j].real());
//			V[i][j].imag((double)v[i][j].imag());
//		}
//	for(int i = 0; i < max(m,n); i++)
//		S[i] = (double)s[i] / mul_factor;

}

void csvd (complex_t** a, int m, int n, double* s, complex_t** u, complex_t** v)
{
	complex_t **coppied_a;
	coppied_a = new complex_t*[max(m,n)];
	for(int i = 0; i < max(m,n); ++i)
	{
		coppied_a[i] = new complex_t[max(m,n)];
		for(int j = 0; j < max(m,n); ++j)
		{
			coppied_a[i][j] = a[i][j];
		}
	}

	int times = 0;
	try
	{
		if(DEBUG) cout << "1st try" << endl;
		inside_csvd(a, m, n, s, u, v);
		if(DEBUG) cout << "1st try success" << endl;
	}
	catch(...)
	{
		if(DEBUG) cout << "1st try fail" << endl;
		while(true)
		{
			times++;
			copyMatrix(a,coppied_a,m,n);
			for(int i = 0; i < m; i++)
				for(int j = 0; j < n; j++)
				{
					a[i][j] *= mul_factor;
					coppied_a[i][j] *= mul_factor;
				}
			try
			{
				if(DEBUG) cout << "another try" << endl;
				inside_csvd(a, m, n, s, u, v);
				break;
			}
			catch(...)
			{
				if(DEBUG) cout << "another try fail" << endl;
			}
			if(times == 20)
				assert(false);
		}
	}

	//Divide by mul_factor every singular value after we multiplied matrix a
	for(int i = 0; i < times; i++)
		for(int k = 0; k < max(m,n); k++)
				s[k] /= mul_factor;




	for(int i = 0; i < max(m,n); ++i)
	{
		delete [] coppied_a[i];
	}
	delete [] coppied_a;


}
