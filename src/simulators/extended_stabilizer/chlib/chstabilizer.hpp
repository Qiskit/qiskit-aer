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

#ifndef CH_STABILIZER_HPP
#define CH_STABILIZER_HPP

#include <limits.h>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "core.hpp"

namespace CHSimulator
{
// Clifford simulator based on the CH-form 
class StabilizerState
{
public:
  // constructor creates basis state |phi>=|00...0> 
  StabilizerState(const unsigned n_qubits);
  //Copy constructor
  StabilizerState(const StabilizerState& rhs);
  // n = number of qubits
  // qubits are numbered as q=0,1,...,n-1


  uint_fast64_t NQubits() const
  {
    return n;
  }
  scalar_t Omega() const
  {
    return omega;
  }
  uint_fast64_t Gamma1() const
  {
    return gamma1;
  }
  uint_fast64_t Gamma2() const
  {
    return gamma2;
  }
  std::vector<uint_fast64_t> GMatrix() const
  {
    return G;
  }
  std::vector<uint_fast64_t> FMatrix() const
  {
    return F;
  }
  std::vector<uint_fast64_t> MMatrix() const
  {
    return M;
  }


  // Clifford gates
  void CX(unsigned q, unsigned r); // q=control, r=target
  void CZ(unsigned q, unsigned r);
  void H(unsigned q); 
  void S(unsigned q); 
  void Z(unsigned q); 
  void X(unsigned q);
  void Y(unsigned q);
  void Sdag(unsigned q); // S inverse
 

  // state preps
  void CompBasisVector(uint_fast64_t x); // prepares |x> 
  void HadamardBasisVector(uint_fast64_t x); // prepares H(x)|00...0>


  // measurements
  scalar_t Amplitude(uint_fast64_t x); // computes the  amplitude <x|phi>
  uint_fast64_t Sample(); // returns a sample from the distribution |<x|phi>|^2
  uint_fast64_t Sample(uint_fast64_t v_mask);
  void MeasurePauli(const pauli_t P); // applies a gate (I+P)/2 
                                // where P is an arbitrary Pauli operator
  void MeasurePauliProjector(const std::vector<pauli_t>& generators);

  inline scalar_t ScalarPart() {return omega;} 

  
  //InnerProduct & Norm Estimation
  scalar_t InnerProduct(const uint_fast64_t& A_diag1, const uint_fast64_t& A_diag2, const std::vector<uint_fast64_t>& A);
  
  // Metropolis updates:
  // To initialize Metropolis by a state x call Amplitude(x)
  scalar_t ProposeFlip(unsigned flip_pos); // returns the amplitude <x'|phi> 
                                           // where x'=bitflip(x,q)
                                           // x = current Metropolis state
  inline void AcceptFlip() {P=Q;} // accept the proposed bit flip
    
  friend double NormEstimate(std::vector<StabilizerState>& states,
              const std::vector< std::complex<double> >& phases, 
              const std::vector<uint_fast64_t>& Samples_d1,
              const std::vector<uint_fast64_t> &Samples_d2, 
              const std::vector< std::vector<uint_fast64_t> >& Samples);
  #ifdef _OPENMP
  friend double ParallelNormEstimate(std::vector<StabilizerState>& states,
                        const std::vector< std::complex<double> >& phases, 
                        const std::vector<uint_fast64_t>& Samples_d1,
                        const std::vector<uint_fast64_t> &Samples_d2, 
                        const std::vector< std::vector<uint_fast64_t> >& Samples,
                        int n_threads);
  #endif

private:

  unsigned n; 

  // define CH-data (F,G,M,gamma,v,s,omega)  see Section IIIA for notations

  // stabilizer tableaux of the C-layer
  uint_fast64_t gamma1; //phase vector gamma = gamma1 + 2*gamma2 (mod 4)
  uint_fast64_t gamma2; 
  // each column of F,G,M is represented by uint_fast64_t integer
  std::vector<uint_fast64_t> F; // F[i] = i-th column of F
  std::vector<uint_fast64_t> G; // G[i] = i-th column of G
  std::vector<uint_fast64_t> M; // M[i] = i-th column of M

  uint_fast64_t v; // H-layer U_H=H(v)
  uint_fast64_t s; // initial state |s>
  scalar_t omega; // scalar factor such that |phi>=omega*U_C*U_H|s>

  // multiplies the C-layer on the right by a C-type gate
  void RightCX(unsigned q, unsigned r); // q=control, r=target
  void RightCZ(unsigned q, unsigned r);
  void RightS(unsigned q);
  void RightZ(unsigned q);
  void RightSdag(unsigned q);
  
  // computes a Pauli operator U_C^{-1}X(x)U_C 
  pauli_t GetPauliX(uint_fast64_t x);

  // replace the initial state |s> in the CH-form by a superposition
  // (|t> + i^b |u>)*sqrt(1/2) as described in Proposition 3
  // update the CH-form 
  void UpdateSvector(uint_fast64_t t, uint_fast64_t u, unsigned b);


  // F-transposed and M-transposed. Need this to compute amplitudes
  std::vector<uint_fast64_t> FT; // FT[i] = i-th row of F
  std::vector<uint_fast64_t> MT; // MT[i] = i-th row of M
  // compute the transposed matrices
  void TransposeF();
  void TransposeM();
  bool isReadyFT; // true if F-transposed is available
  bool isReadyMT; // true if M-transposed is available


  // auxiliary Pauli operators
  pauli_t P;
  pauli_t Q;
  
};

typedef std::complex<double> cdouble;

//-------------------------------//
// Implementation                //
//-------------------------------//

//Lookup table for e^(i pi m / 4)
static const int RE_PHASE[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static const int IM_PHASE[8] = {0, 1, 1, 1, 0, -1, -1, -1};

// Clifford simulator based on the CH-form for for n<=64 qubits

StabilizerState::StabilizerState(const unsigned n_qubits):
n(n_qubits), // number of qubits
gamma1(zer),
gamma2(zer),
F(n,zer),
G(n,zer),
M(n,zer),
v(zer),
s(zer),
omega(),
FT(n,zer),
MT(n,zer),
isReadyFT(false),
isReadyMT(false)
{
  // initialize  by the basis vector 00...0
  if(n>63)
  {
    throw std::invalid_argument("The CH simulator only supports up to 63 qubits.\n");
  }

  // G and F are identity matrices
  for (unsigned q=0; q<n; q++)
  {
    G[q]=(one<<q);
    F[q]=(one<<q);
  }
  omega.makeOne();
}

StabilizerState::StabilizerState(const StabilizerState& rhs):
n(rhs.n), // number of qubits
gamma1(rhs.gamma1),
gamma2(rhs.gamma2),
F(rhs.F),
G(rhs.G),
M(rhs.M),
v(rhs.v),
s(rhs.s),
omega(rhs.omega),
FT(rhs.FT),
MT(rhs.MT),
isReadyFT(rhs.isReadyFT),
isReadyMT(rhs.isReadyMT)
{
}

void StabilizerState::CompBasisVector(uint_fast64_t x)
{
  s=x;
  v=zer;
  gamma1=zer;
  gamma2=zer;
  omega.makeOne();
  // G and F are identity matrices, M is zero matrix
  for (unsigned q=0; q<n; q++)
  {
    M[q]=zer;
    G[q]=(one<<q);
    F[q]=(one<<q);
  }
  isReadyFT=false;
  isReadyMT=false;
}

void StabilizerState::HadamardBasisVector(uint_fast64_t x)
{
  s=zer;
  v=x;
  gamma1=zer;
  gamma2=zer;
  omega.makeOne();
  // G and F are identity matrices, M is zero matrix
  for (unsigned q=0; q<n; q++)
  {
    M[q]=zer;
    G[q]=(one<<q);
    F[q]=(one<<q);
  }
  isReadyFT=false;
  isReadyMT=false;
}

void StabilizerState::RightS(unsigned q)
{
  isReadyMT=false;// we are going to change M
  M[q]^=F[q];
  // update phase vector: gamma[p] gets gamma[p] - F_{p,q} (mod 4)   for all p
  gamma2^=F[q]^(gamma1 & F[q]);
  gamma1^=F[q];
}

void StabilizerState::RightSdag(unsigned q)
{
  isReadyMT=false;// we are going to change M
  M[q]^=F[q];
  // update phase vector: gamma[p] gets gamma[p] + F_{p,q} (mod 4)   for all p
  gamma1^=F[q];
  gamma2^=F[q]^(gamma1 & F[q]);
}

void StabilizerState::RightZ(unsigned q)
{
  // update phase vector: gamma[p] gets gamma[p] + 2F_{p,q} (mod 4)   for all p
  gamma2^=F[q];
}

void StabilizerState::S(unsigned q)
{
  isReadyMT=false;// we are going to change M
  uint_fast64_t C=(one<<q);
  for (unsigned p=0; p<n;  p++)
  {
    M[p]^=((G[p]>>q) & one)*C;
  }
  // update phase vector:  gamma[q] gets gamma[q] - 1
  gamma1^=C;
  gamma2^=((gamma1 >> q) & one )*C;
}

void StabilizerState::Sdag(unsigned q)
{
  isReadyMT=false;// we are going to change M
  uint_fast64_t C=(one<<q);
  for (unsigned p=0; p<n;  p++)
  {
    M[p]^=((G[p]>>q) & one)*C;
  }
  // update phase vector:  gamma[q] gets gamma[q] + 1
  gamma2^=((gamma1 >> q) & one )*C;
  gamma1^=C;
}

void StabilizerState::Z(unsigned q)
{
  // update phase vector:  gamma[q] gets gamma[q] + 2
  gamma2^=(one<<q);
}

void StabilizerState::X(unsigned q)
{
  //Commute the Pauli through UC
  if (!isReadyMT)
  {
    TransposeM();
  }
  if (!isReadyFT)
  {
    TransposeF();
  }
  uint_fast64_t x_string = FT[q];
  uint_fast64_t z_string = MT[q];
  //Initial phase correction
  int phase = 2*((gamma1 >> q)&one) + 4*((gamma2>>q) & one);
  //Commute the z_string through the hadamard layer
  // Each z that hits a hadamard becomes a Pauli X
  s ^= (z_string & v);
  //Remaining Z gates add a global phase from their action on the s string
  phase += 4*(hamming_parity((z_string & ~(v)) & s));
  //Commute the x_string through the hadamard layer
  // Any remaining X gates update s
  s ^= (x_string & ~(v));
  //New z gates add a global phase from their action on the s string
  phase += 4*(hamming_parity((x_string & v) & s));
  //Update the global phase
  omega.e = (omega.e + phase)%8;
}

void StabilizerState::Y(unsigned q)
{
  Z(q);
  X(q);
  //Add a global phase of -i
  omega.e = (omega.e + 2)%8;
}

void StabilizerState::RightCX(unsigned q, unsigned r)
{
  G[q]^=G[r];
  F[r]^=F[q];
  M[q]^=M[r];
}

void StabilizerState::CX(unsigned q, unsigned r)
{
  isReadyMT=false;// we are going to change M and F
  isReadyFT=false;
  uint_fast64_t C=(one<<q);
  uint_fast64_t T=(one<<r);
  bool b=false;
  for (unsigned p=0; p<n;  p++)
  {
    b=( b != ( ( M[p] & C ) && ( F[p] & T ) )  );
    G[p]^=((G[p]>>q) & one)*T;
    F[p]^=((F[p]>>r) & one)*C;
    M[p]^=((M[p]>>r) & one)*C;
  }
  // update phase vector as
  // gamma[q] gets gamma[q] + gamma[r] + 2*b (mod 4)
  if (b) gamma2^=C;
  b= ( (gamma1 & C) && (gamma1 & T));
  gamma1^=((gamma1 >> r) & one)*C;
  gamma2^=((gamma2 >> r) & one)*C;
  if (b) gamma2^=C;
}

void StabilizerState::RightCZ(unsigned q, unsigned r)
{
  isReadyMT=false;// we are going to change M
  M[q]^=F[r];
  M[r]^=F[q];
  gamma2^=(F[q] & F[r]);
}

void StabilizerState::CZ(unsigned q, unsigned r)
{
  isReadyMT=false;// we are going to change M
  uint_fast64_t C=(one<<q);
  uint_fast64_t T=(one<<r);
  for (unsigned p=0; p<n;  p++)
  {
    M[p]^=((G[p]>>r) & one)*C;
    M[p]^=((G[p]>>q) & one)*T;
  }
}

pauli_t StabilizerState::GetPauliX(uint_fast64_t x)
{
  // make sure that M-transposed and F-transposed have been already computed
  if (!isReadyMT)
  {
    TransposeM();
  }
  if (!isReadyFT)
  {
    TransposeF();
  }
  pauli_t R;

  for (unsigned pos=0; pos<n; pos++)
  {
    if (x & (one<<pos))
    {
      pauli_t P1; // make P1=U_C^{-1} X_{pos} U_C
      P1.e=1*((gamma1>>pos) & one);
      P1.e+=2*((gamma2>>pos) & one);
      P1.X=FT[pos];
      P1.Z=MT[pos];
      R*=P1;
    }
  }
 return R;
}

scalar_t StabilizerState::Amplitude(uint_fast64_t x)
{
  // compute transposed matrices if needed
  if (!isReadyMT)
  {
    TransposeM();
  }
  if (!isReadyFT)
  {
    TransposeF();
  }
  // compute Pauli U_C^{-1} X(x) U_C
  P=GetPauliX(x);

  if (!omega.eps) return omega; // the state is zero

  // now the amplitude = complex conjugate of <s|U_H P |0^n>
  // Z-part of P is absorbed into 0^n

  scalar_t amp;
  amp.e=2*P.e;
  int p = (int) hamming_weight(v);
  amp.p= -1 * p;// each Hadamard gate contributes 1/sqrt(2)
  bool isNonZero=true;


  for (unsigned q=0; q<n; q++)
  {
    uint_fast64_t pos=(one<<q);
    if (v & pos)
    {
      amp.e += 4*( (s & pos) && (P.X & pos) ); // minus sign that comes from <1|H|1>
    }
    else
    {
      isNonZero=( ((P.X & pos)==(s & pos)) );
    }
    if (!isNonZero) break;
  }


  amp.e%=8;
  if (isNonZero)
  {
    amp.conjugate();
  }
  else
  {
    amp.eps=0;
    return amp;
  }

  // multiply amp by omega
  amp.p+=omega.p;
  amp.e=(amp.e + omega.e) % 8;
  return amp;
}

scalar_t StabilizerState::ProposeFlip(unsigned flip_pos)
{
  // Q gets Pauli operator U_C^{-1} X_{flip_pos} U_C
  Q.e=1*((gamma1>>flip_pos) & one);
  Q.e+=2*((gamma2>>flip_pos) & one);
  Q.X=FT[flip_pos];
  Q.Z=MT[flip_pos];
  Q*=P;

  if (!omega.eps) return omega; // the state is zero

  // the rest is the same as Amplitude() except that P becomes Q

  // now the amplitude = complex conjugate of <s|U_H Q |0^n>
  // Z-part of Q is absorbed into 0^n

  // the rest is the same as Amplitude() except that P is replaced by Q

  scalar_t amp;
  amp.e=2*Q.e;
  amp.p=-1*(hamming_weight(v));// each Hadamard gate contributes 1/sqrt(2)
  bool isNonZero=true;

  for (unsigned q=0; q<n; q++)
  {
    uint_fast64_t pos=(one<<q);
    if (v & pos)
    {
      amp.e+=4*( (s & pos) && (Q.X & pos) ); // minus sign that comes from <1|H|1>
    }
    else
    {
      isNonZero=( ((Q.X & pos)==(s & pos)) );
    }
    if (!isNonZero) break;
  }

  amp.e%=8;
  if (isNonZero)
  {
    amp.conjugate();
  }
  else
  {
    amp.eps=0;
    return amp;
  }

  // multiply amp by omega
  amp.p+=omega.p;
  amp.e=(amp.e + omega.e) % 8;
  return amp;
}

uint_fast64_t StabilizerState::Sample()
{
  uint_fast64_t x=zer;
  for (unsigned q=0; q<n; q++)
  {
    bool w = !!(s & (one<<q));
    w^=( (v & (one<<q)) && (rand() % 2) );
    if (w) x^=G[q];
  }
  return x;
}

uint_fast64_t StabilizerState::Sample(uint_fast64_t v_mask)
{
  //v_mask is a uniform random binary string we use to sample the bits
  //of v in a single step.
  uint_fast64_t x=zer;
  uint_fast64_t masked_v = v&v_mask;
  for (unsigned q=0; q<n; q++)
  {
    bool w = !!(s & (one<<q));
    w^= !!(masked_v & (one<<q));
    if (w) x^=G[q];
  }
  return x;
}


void StabilizerState::TransposeF()
{
  for (unsigned p=0; p<n; p++) // look at p-th row of F
  {
    uint_fast64_t row=zer;
    uint_fast64_t mask=(one<<p);
    for (unsigned q=0; q<n; q++) // look at q-th column of F
        if (F[q] & mask) row^=(one<<q);
    FT[p]=row;
  }
  isReadyFT = true;
}

void StabilizerState::TransposeM()
{
  for (unsigned p=0; p<n; p++) // look at p-th row of M
  {
    uint_fast64_t row=zer;
    uint_fast64_t mask=(one<<p);
    for (unsigned q=0; q<n; q++) // look at q-th column of M
        if (M[q] & mask) row^=(one<<q);
    MT[p]=row;
  }
  isReadyMT = true;
}

void StabilizerState::UpdateSvector(uint_fast64_t t, uint_fast64_t u, unsigned b)
{
// take care of the trivial case: t=u
    if (t==u)  // multiply omega by (1+i^b)/sqrt(2)
        switch(b)
        {
             case 0 :
                omega.p+=1;
                s=t;
                return;
             case 1 :
                s=t;
                omega.e=(omega.e + 1) % 8;
                return;
             case 2 :
                s=t;
                omega.eps=0;
                return;
             case 3 :
                s=t;
                omega.e=(omega.e + 7) % 8;
                return;
             default :
                // we should not get here
                throw std::logic_error("Invalid phase factor found b:" + std::to_string(b) + ".\n");
         }

    // now t and u are distinct
    isReadyFT=false; // below we are going to change F and M
    isReadyMT=false;
    // naming of variables roughly follows Section IIIA
    uint_fast64_t ut=u^t;
    uint_fast64_t nu0 = (~v) & ut;
    uint_fast64_t nu1 = v & ut;
    //
    b%=4;
    unsigned q=0;
    uint_fast64_t qpos=zer;
    if (nu0)
    {

        // the subset nu0 is non-empty
        // find the first element of nu0
        q=0;
        while (!(nu0 & (one<<q))) q++;

        qpos=(one<<q);
        if(!(nu0 & qpos))
        {
          throw std::logic_error("Failed to find first bit of nu despite being non-empty.");
        }

        // if nu0 has size >1 then multiply U_C on the right by the first half of the circuit VC
        nu0^=qpos; // set q-th bit to zero
        if (nu0)
            for (unsigned q1=q+1; q1<n; q1++)
                if (nu0 & (one<<q1))
                    RightCX(q,q1);

        // if nu1 has size >0 then apply the second half of the circuit VC
        if (nu1)
            for (unsigned q1=0; q1<n; q1++)
                if (nu1 & (one<<q1))
                    RightCZ(q,q1);

    }// if (nu0)
    else
    {
        // if we got here when nu0 is empty
        // find the first element of nu1
        q=0;
        while (!(nu1 & (one<<q)))  q++;

        qpos=(one<<q);
        if(!(nu1 & qpos))
        {
          throw std::logic_error("Expect at least one non-zero element in nu1.\n");
        }

        // if nu1 has size >1 then apply the circuit VC
        nu1^=qpos;
        if (nu1)
            for (unsigned q1=q+1; q1<n; q1++)
                if (nu1 & (one<<q1))
                    RightCX(q1,q);

    }// if (nu0) else

    // update the initial state
    // if t_q=1 then switch t_q and u_q
    // uint_fast64_t y,z;
    if (t & qpos)
    {
        s=u;
        omega.e=(omega.e + 2*b) % 8;
        b=(4-b) % 4;
        if(!!(u & qpos))
        {

        }
    }
    else
        s=t;

    // change the order of H and S gates
    // H^{a} S^{b} |+> = eta^{e1} S^{e2} H^{e3} |e4>
    // here eta=exp( i (pi/4) )
    // a=0,1
    // b=0,1,2,3
    //
    // H^0 S^0 |+> = eta^0 S^0 H^1 |0>
    // H^0 S^1 |+> = eta^0 S^1 H^1 |0>
    // H^0 S^2 |+> = eta^0 S^0 H^1 |1>
    // H^0 S^3 |+> = eta^0 S^1 H^1 |1>
    //
    // H^1 S^0 |+> = eta^0 S^0 H^0 |0>
    // H^1 S^1 |+> = eta^1 S^1 H^1 |1>
    // H^1 S^2 |+> = eta^0 S^0 H^0 |1>
    // H^1 S^3 |+> = eta^{7} S^1 H^1 |0>
    //
    // "analytic" formula:
    // e1 = a * (b mod 2) * ( 3*b -2 )
    // e2 = b mod 2
    // e3 = not(a) + a * (b mod 2)
    // e4 = not(a)*(b>=2) + a*( (b==1) || (b==2) )

    bool a=((v & qpos)>0);
    unsigned e1=a*(b % 2)*( 3*b -2);
    unsigned e2 = b % 2;
    bool e3 = ( (!a) != (a && ((b % 2)>0) ) );
    bool e4 = ( ( (!a) && (b>=2) ) != (a && ((b==1) || (b==2)))  );

    // update CH-form
    // set q-th bit of s to e4
    s&=~qpos;
    s^=e4*qpos;

    // set q-th bit of v to e3
    v&=~qpos;
    v^=e3*qpos;

    // update the scalar factor omega
    omega.e=(omega.e  + e1) % 8;

    // multiply the C-layer on the right by S^{e2} on the q-th qubit
    if (e2) RightS(q);
}



void StabilizerState::H(unsigned q)
{
    isReadyMT=false;// we are going to change M and F
    isReadyFT=false;
    // extract the q-th row of F,G,M
    uint_fast64_t rowF=zer;
    uint_fast64_t rowG=zer;
    uint_fast64_t rowM=zer;
    for (unsigned j=0; j<n; j++)
    {
        uint_fast64_t pos=(one<<j);
        rowF^=pos*( (F[j]>>q) & one );
        rowG^=pos*( (G[j]>>q) & one );
        rowM^=pos*( (M[j]>>q) & one );
    }

    // after commuting H through the C and H laters it maps |s> to a state
    // sqrt(0.5)*[  (-1)^alpha |t> + i^{gamma[p]} (-1)^beta |u>  ]
    //
    // compute t,s,alpha,beta
    uint_fast64_t t = s ^ (rowG & v);
    uint_fast64_t u = s ^ (rowF & (~v)) ^ (rowM & v);

    unsigned alpha =  hamming_weight( rowG & (~v) & s );
    unsigned beta =  hamming_weight( (rowM & (~v) & s) ^ (rowF & v & (rowM ^ s)) );

    if (alpha % 2) omega.e=(omega.e+4) % 8;
    // get the phase gamma[q]
    unsigned phase = ((gamma1>>q) & one) + 2*((gamma2>>q) & one);
    unsigned b=(phase + 2*alpha + 2*beta) % 4;


    // now the initial state is sqrt(0.5)*(|t> + i^b |u>)

    // take care of the trivial case
    if (t==u)
    {
        s=t;
        if(!((b==1) || (b==3))) // otherwise the state is not normalized
        {
          throw std::logic_error("State is not properly normalised, b should be 1 or 3.\n");
        }
        if (b==1)
            omega.e=(omega.e + 1) % 8;
        else
            omega.e=(omega.e + 7) % 8;
    }
    else
        UpdateSvector(t,u,b);

}

void StabilizerState::MeasurePauli(pauli_t PP)
{
    // compute Pauli R = U_C^{-1} P U_C
    pauli_t R;
    R.e = PP.e;

    for (unsigned j=0; j<n; j++)
        if ((PP.X>>j) & one)
        {
            // multiply R by U_C^{-1} X_j U_C
            // extract the j-th rows of F and M
            uint_fast64_t rowF=zer;
            uint_fast64_t rowM=zer;
            for (unsigned i=0; i<n; i++)
            {
                rowF^=(one<<i)*( (F[i]>>j) & one);
                rowM^=(one<<i)*( (M[i]>>j) & one);
            }
            R.e+= 2*hamming_weight(R.Z & rowF); // extra sign from Pauli commutation
            R.Z^=rowM;
            R.X^=rowF;
            R.e+= ((gamma1>>j) & one) + 2*((gamma2>>j) & one);
        }
    for (unsigned q=0; q<n; q++)
          R.Z^=(one<<q)*(hamming_weight( PP.Z & G[q]) % 2);

    // now R=U_C^{-1} PP U_C
    // next conjugate R by U_H
    uint_fast64_t tempX = ( (~v) & R.X ) ^ (v & R.Z);
    uint_fast64_t tempZ = ( (~v) & R.Z ) ^ (v & R.X);
    // the sign flips each time a Hadamard hits Y on some qubit
    R.e=(R.e + 2*hamming_weight(v & R.X & R.Z)) % 4;
    R.X=tempX;
    R.Z=tempZ;

    // now the initial state |s> becomes 0.5*(|s> + R |s>) = 0.5*(|s> + i^b |s ^ R.X>)
    unsigned b = (R.e + 2*hamming_weight(R.Z & s) ) % 4;
    UpdateSvector(s, s ^ R.X, b);
    // account for the extra factor sqrt(1/2)
    omega.p-=1;

    isReadyMT=false;// we have changed change M and F
    isReadyFT=false;
}

void StabilizerState::MeasurePauliProjector(const std::vector<pauli_t>& generators)
//Measure generators of a projector.
{
    for (uint_fast64_t i=0; i<generators.size(); i++)
    {
        this->MeasurePauli(generators[i]);
        if (omega.eps == 0)
        {
            break;
        }
    }
}

scalar_t StabilizerState::InnerProduct(const uint_fast64_t& A_diag1,
                                       const uint_fast64_t& A_diag2,
                                       const std::vector<uint_fast64_t>& A)
{
    uint_fast64_t K_diag1 = zer, K_diag2=zer, J_diag1=gamma1, J_diag2=gamma2;
    std::vector<uint_fast64_t> J(n, zer);
    std::vector<uint_fast64_t> K(n, zer);
    std::vector<uint_fast64_t> placeholder(n, zer);
    if(!isReadyMT)
    {
        TransposeM();
    }
    if (!isReadyFT)
    {
        TransposeF();
    }
    //Setup the J matrix
    for (size_t i=0; i<n; i++)
    {
        for (size_t j=i; j<n; j++)
        {
            if (hamming_parity(MT[i] & FT[j]))
            {
                J[i] |= (one << j);
                J[j] |= (one << i);
            }
        }
    }
    //Calculate the matrix J =  A+J
    J_diag2 = (A_diag2 ^ J_diag2 ^ (A_diag1&J_diag1));
    J_diag1 = (A_diag1 ^ J_diag1);
    for (size_t i=0; i<n; i++)
    {
        J[i] ^= A[i];
    }
    //placeholder = J*G)
    for (size_t i=0; i<n; i++)
    {
        //Grab column i of J, it's symmetric
        uint_fast64_t col_i = J[i];
        for (size_t j=0; j<n; j++)
        {
            if (hamming_parity(col_i & G[j]))
            {
                placeholder[j] |= (one<<i);
            }
        }
    }
    //K = GT*placeholder
    for (size_t i=0; i<n; i++)
    {
        uint_fast64_t col_i = G[i];
        uint_fast64_t shift = (one << i);
        for (size_t j=i; j<n; j++)
        {
            if(hamming_parity(col_i & placeholder[j]))
            {
                K[j] |= shift;
                K[i] |= (one << j);
            }
        }
        for (size_t r =0; r<n; r++)
        {
            if ((col_i >> r) & one)
            {
                uint_fast64_t one_bit = (J_diag1 >> r) & one;
                if ((K_diag1 >> i) & one_bit)
                {
                    K_diag2 ^= shift;
                }
                K_diag1 ^= one_bit*shift;
                K_diag2 ^= ((J_diag2 >> r)&one) * shift;
            }
            for (size_t k=r+1; k<n; k++)
            {
                if ((J[k] >> r) & (col_i >> r) & (col_i >> k) & one)
                {
                    K_diag2 ^= shift;
                }
            }
        }
    }
    unsigned col=0;
    QuadraticForm q(hamming_weight(v));
    //We need to setup a quadratic form to evaluate the Exponential Sum
    for (size_t i=0; i<n; i++)
    {
        if ((v>>i) & one)
        {
            uint_fast64_t shift = (one << col);
            //D = Diag(K(1,1)) + 2*[s + s*K](1)
            // J = K(1,1);
            q.D1 ^= ((K_diag1 >> i) & one) * shift;
            q.D2 ^= (( ((K_diag2 >> i) ^ (s >> i)) & one)
                     ^ hamming_parity(K[i] & s)) * shift;
            // q.D2 ^= ((s >> i) & one) * shift;
            // q.D2 ^= hamming_parity(K[i] & s) * shift;
            unsigned row=0;
            for (size_t j=0; j<n; j++)
            {
                if((v>>j) & one)
                {
                    q.J[col] |= (((K[i] >> j) & one) << row);
                    row++;
                }
            }
            col++;
        }
    }
    // Q = 4* (s.v) + sKs
    q.Q = hamming_parity(s&v)*4;
    for (size_t i=0; i<n; i++)
    {
        if ((s>>i) & one)
        {
            q.Q = (q.Q + 4*((K_diag2 >> i) & one) + 2*((K_diag1 >> i) & one))%8;// + 2*((K_diag1 >> i) & one))%8;
            for (size_t j=i+1; j<n; j++)
            {
                if ((s>>j) & (K[j] >> i) & one)
                {
                    q.Q ^= 4;
                }
            }
        }
    }
    scalar_t amp = q.ExponentialSum();
    // Reweight by 2^{-(n+|v|)}/2
    amp.p -= (n+hamming_weight(v));
    // We need to further multiply by omega*
    scalar_t psi_amp(omega);
    psi_amp.conjugate();
    amp *= psi_amp;
    return amp;
}

double NormEstimate(std::vector<StabilizerState>& states, const std::vector< std::complex<double> >& phases, 
                    const std::vector<uint_fast64_t>& Samples_d1, const std::vector<uint_fast64_t> &Samples_d2, 
                    const std::vector< std::vector<uint_fast64_t> >& Samples)
{
    // Norm estimate for a state |psi> = \sum_{i} c_{i}|phi_{i}>
    double xi=0;
    unsigned L = Samples_d1.size();
    // std::vector<double> data = (L,0.);
    for (size_t i=0; i<L; i++)
    {
      double re_eta =0., im_eta = 0.;
      const int_t END = states.size();
      #pragma omp parallel for reduction(+:re_eta) reduction(+:im_eta)
      for (int_t j=0; j<END; j++)
      {
          if(states[j].ScalarPart().eps != 0)
          {
              scalar_t amp = states[j].InnerProduct(Samples_d1[i], Samples_d2[i], Samples[i]);
              if (amp.eps != 0)
              {
                  if (amp.e % 2)
                  {
                    amp.p--;
                  }
                  double mag = pow(2, amp.p/(double) 2);
                  cdouble phase(RE_PHASE[amp.e], IM_PHASE[amp.e]);
                  phase *= conj(phases[j]);
                  re_eta += (mag * real(phase));
                  im_eta += (mag * imag(phase));
              }
          }
      }
      xi += (pow(re_eta,2) + pow(im_eta, 2));
    }
    return std::pow(2, states[0].NQubits()) * (xi/L);
}

double ParallelNormEstimate(std::vector<StabilizerState>& states, const std::vector< std::complex<double> >& phases, 
                    const std::vector<uint_fast64_t>& Samples_d1, const std::vector<uint_fast64_t>& Samples_d2, 
                    const std::vector< std::vector<uint_fast64_t> >& Samples, int n_threads)
{
    double xi=0;
    unsigned L = Samples_d1.size();
    unsigned chi = states.size();
    for (uint_fast64_t i=0; i<L; i++)
    {
        double re_eta =0., im_eta = 0.;
        #pragma omp parallel for reduction(+:re_eta) reduction(+:im_eta) num_threads(n_threads)
        for (int_t j=0; j<chi; j++)
        {
            if(states[j].ScalarPart().eps != 0)
            {
                scalar_t amp = states[j].InnerProduct(Samples_d1[i], Samples_d2[i], Samples[i]);
                if (amp.eps != 0)
                {
                    if (amp.e % 2)
                    {
                      amp.p--;
                    }
                    double mag = pow(2, amp.p/(double) 2);
                    cdouble phase(RE_PHASE[amp.e], IM_PHASE[amp.e]);
                    phase *= conj(phases[j]);
                    re_eta += (mag * real(phase));
                    im_eta += (mag * imag(phase));
                }
            }
        }
        xi += (pow(re_eta,2) + pow(im_eta, 2));
    }
    return pow(2., states[0].NQubits())*(xi/L);
}

}
#endif
