#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
cimport cython
from libc.math cimport (fabs, sinh, cosh, exp, pi, sqrt, cos, sin, copysign)

cdef extern from "<complex>" namespace "std" nogil:
    double real(double complex x)
    double imag(double complex x)

#
#          Shanjie Zhang and Jianming Jin
#
#       Copyrighted but permission granted to use code in programs.
# Buy their book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
    
@cython.cdivision(True)
@cython.boundscheck(False)
cdef double complex erf(double complex Z):
    """
    Parameters
    ----------
    Z : double complex
        Input parameter.
    X : double
        Real part of Z.
    Y : double
        Imag part of Z.

    Returns
    -------
    erf(z) : double complex
    """
    
    cdef double EPS = 1e-12
    cdef double X = real(Z)
    cdef double Y = imag(Z)
    cdef double X2 = X * X
    
    cdef double ER, R, W, C0, ER0, ERR, ERI, CS, SS, ER1, EI1, ER2, W1
    
    cdef size_t K, N
    
    if X < 3.5:
        ER = 1.0
        R = 1.0
        W = 0.0
        for K in range(1, 100):
            R = R*X2/(K+0.5)
            ER = ER+R
            if (fabs(ER-W) < EPS*fabs(ER)):
                break
            W = ER
        C0 = 2.0/sqrt(pi)*X*exp(-X2)
        ER0 = C0*ER
    else:
        ER = 1.0
        R=1.0
        for K in range(1, 12):
            R = -R*(K-0.5)/X2
            ER = ER+R
        C0 = exp(-X2)/(X*sqrt(pi))
        ER0 = 1.0-C0*ER
    
    if Y == 0.0:
        ERR = ER0
        ERI = 0.0
    else:
        CS = cos(2.0*X*Y)
        SS = sin(2.0*X*Y)
        ER1 = exp(-X2)*(1.0-CS)/(2.0*pi*X)
        EI1 = exp(-X2)*SS/(2.0*pi*X)
        ER2 = 0.0
        W1 = 0.0
        
        for N in range(1,100):
            ER2 = ER2+exp(-.25*N*N)/(N*N+4.0*X2)*(2.0*X-2.0*X*cosh(N*Y)*CS+N*sinh(N*Y)*SS)
            if (fabs((ER2-W1)/ER2) < EPS):
                break
            W1 = ER2
        
        C0 = 2.0*exp(-X2)/pi
        ERR = ER0+ER1+C0*ER2
        EI2 = 0.0
        W2 = 0.0
        
        for N in range(1,100):
            EI2 = EI2+exp(-.25*N*N)/(N*N+4.0*X2)*(2.0*X*cosh(N*Y)*SS+N*sinh(N*Y)*CS)
            if (fabs((EI2-W2)/EI2) < EPS):
                break
            W2 = EI2
        ERI = EI1+C0*EI2
        
    return ERR + 1j*ERI