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

#define mul_factor 100000000000000000

using namespace std;

void cswap(complex_t &a, complex_t &b)
{

	complex_t temp = a;
	a = b;
	b = temp;
}


void cDagger(complex_t** a, int m, int n)
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

void csvd (complex_t** a, int m, int n, double* s, complex_t** u, complex_t** v)
{
	double b[100] = {0}, c[100] = {0}, t[100] = {0};
	double cs = 0, eps = 0, f = 0 ,g = 0, h = 0, sn = 0 , w = 0, x = 0, y = 0, z = 0;
	double eta = 1.1920929e-10, tol = 1.5e-34;
	int i = 0, j = 0, k = 0, k1 = 0, l = 0, l1 = 0;
	complex_t q = 0;

	// Transpose when m < n
    bool transposed = false;
    if (m < n)
    {
    	transposed = true;
    	cDagger(a,m,n);
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
			q = -conj(a[k][k1] )/abs(a[k][k1]);
			for( i = k1; i < m; i++){
				a[i][k1] = a[i][k1] * q;
			}
		}
		k = k1;
    }
    eps = 0.0;
	for( k = 0; k < n; k++)
	{
		s[k] = b[k];
		t[k] = c[k];
		eps = max( eps, s[k] + t[k] );
	}
	eps = eps * eta;
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
				t[i-1] = w;
				cs = f / w;
				sn = h / w;
				f = x * cs + g * sn;
				long double large_f = 0;
				if (f==0) {
				  if (DEBUG) cout << "f ==0 because " << "x = " << x << ", cs = " << cs << ", g = " << g << ", sn = " << sn  <<endl;
				  long double large_x =   x * mul_factor;
				  long double large_g =   g * mul_factor;
				  long double large_cs = cs * mul_factor;
				  long double large_sn = sn * mul_factor;
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
				if (h ==0 && f==0 && large_f !=0) {
				  tiny_w = true;
				} else {
				  w = sqrt( h * h + f * f );
				}
				if (w == 0 && !tiny_w) {
				  cout << "ERROR 2: w is exactly 0: h = " <<h<< " , f = " << f <<endl;
				  cout << "w = " << w <<endl;
				  assert(false);
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

        if ( w < -1e-13 )
		{
			s[k] = - w;
			for( j = 0; j < n; j++){
				v[j][k] = - v[j][k];
			}
		}
	}

	for( k = 0; k < n; k++)
	{
		g = - 1.0;
		j = k;
        for( i = k; i < n; i++)
		{
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
	}
	double cut_off = 1.0e-15;
	for( i = 0; i < m; i++)
	{
		if(s[i] < cut_off)  s[i] = 0.0;
		for( j = 0; j < m; j++)
		{
			if(abs(u[i][j].real()) < cut_off)
				u[i][j].real(0.0);
			if(abs(u[i][j].imag()) < cut_off)
				u[i][j].imag(0.0);
		}
	}
	for( i = 0; i < n; i++)
		for( j = 0; j < n; j++)
		{
			if(abs(v[i][j].real()) < cut_off)
				v[i][j].real(0.0);
			if(abs(v[i][j].imag()) < cut_off)
				v[i][j].imag(0.0);
		}
}
