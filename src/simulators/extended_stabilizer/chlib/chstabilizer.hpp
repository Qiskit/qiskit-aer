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

#include <array>
#include <climits>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "core.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace CHSimulator {
// Clifford simulator based on the CH-form
class StabilizerState {
public:
  // constructor creates basis state |phi>=|00...0>
  StabilizerState(const unsigned n_qubits);
  // Copy constructor
  StabilizerState(const StabilizerState &rhs);
  // n = number of qubits
  // qubits are numbered as q=0,1,...,n-1

  uint_fast64_t NQubits() const { return n_qubits_; }
  scalar_t Omega() const { return omega_; }
  BitVector Gamma1() const { return gamma1_; }
  BitVector Gamma2() const { return gamma2_; }
  std::vector<BitVector> GMatrix() const { return G_; }
  std::vector<BitVector> FMatrix() const { return F_; }
  std::vector<BitVector> MMatrix() const { return M_; }

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
  void CompBasisVector(BitVector &x);     // prepares |x>
  void HadamardBasisVector(BitVector &x); // prepares H(x)|00...0>

  // measurements
  scalar_t Amplitude(BitVector &x); // computes the  amplitude <x|phi>
  BitVector Sample(); // returns a sample from the distribution |<x|phi>|^2
  BitVector Sample(BitVector &v_mask);
  void MeasurePauli(const pauli_t P); // applies a gate (I+P)/2
                                      // where P is an arbitrary Pauli operator
  void MeasurePauliProjector(const std::vector<pauli_t> &generators);

  inline scalar_t ScalarPart() { return omega_; }

  // InnerProduct & Norm Estimation
  scalar_t InnerProduct(const BitVector &A_diag1, const BitVector &A_diag2,
                        const std::vector<BitVector> &A);

  // Metropolis updates:
  // To initialize Metropolis by a state x call Amplitude(x)
  scalar_t ProposeFlip(unsigned flip_pos); // returns the amplitude <x'|phi>
                                           // where x'=bitflip(x,q)
                                           // x = current Metropolis state
  inline void AcceptFlip() { P_ = Q_; }    // accept the proposed bit flip

  friend double
  NormEstimate(std::vector<StabilizerState> &states,
               const std::vector<std::complex<double>> &phases,
               const std::vector<uint_fast64_t> &Samples_d1,
               const std::vector<uint_fast64_t> &Samples_d2,
               const std::vector<std::vector<uint_fast64_t>> &Samples);
#ifdef _OPENMP
  friend double
  ParallelNormEstimate(std::vector<StabilizerState> &states,
                       const std::vector<std::complex<double>> &phases,
                       const std::vector<uint_fast64_t> &Samples_d1,
                       const std::vector<uint_fast64_t> &Samples_d2,
                       const std::vector<std::vector<uint_fast64_t>> &Samples,
                       int n_threads);
#endif

private:
  unsigned n_qubits_;

  // define CH-data (F,G,M,gamma,v,s,omega)  see Section IIIA for notations

  // stabilizer tableaux of the C-layer
  BitVector gamma1_; // phase vector gamma = gamma1 + 2*gamma2 (mod 4)
  BitVector gamma2_;
  // each column of F,G,M is represented by uint_fast64_t integer
  std::vector<BitVector> F_; // F[i] = i-th column of F
  std::vector<BitVector> G_; // G[i] = i-th column of G
  std::vector<BitVector> M_; // M[i] = i-th column of M

  BitVector v_;    // H-layer U_H=H(v)
  BitVector s_;    // initial state |s>
  scalar_t omega_; // scalar factor such that |phi>=omega*U_C*U_H|s>

  // multiplies the C-layer on the right by a C-type gate
  void RightCX(unsigned q, unsigned r); // q=control, r=target
  void RightCZ(unsigned q, unsigned r);
  void RightS(unsigned q);
  void RightZ(unsigned q);
  void RightSdag(unsigned q);

  // computes a Pauli operator U_C^{-1}X(x)U_C
  pauli_t GetPauliX(BitVector &x);

  // replace the initial state |s> in the CH-form by a superposition
  // (|t> + i^b |u>)*sqrt(1/2) as described in Proposition 3
  // update the CH-form
  void UpdateSvector(BitVector t, BitVector u, unsigned b);

  // auxiliary Pauli operators
  pauli_t P_;
  pauli_t Q_;
};

using cdouble = std::complex<double>;

//-------------------------------//
// Implementation                //
//-------------------------------//

// Clifford simulator based on the CH-form for for n<=64 qubits

StabilizerState::StabilizerState(const unsigned n_qubits)
    : n_qubits_(n_qubits), // number of qubits
      gamma1_(n_qubits_, true), gamma2_(n_qubits_, true),
      F_(n_qubits_, BitVector(n_qubits_, true)),
      G_(n_qubits_, BitVector(n_qubits_, true)),
      M_(n_qubits_, BitVector(n_qubits_, true)), v_(n_qubits_, true),
      s_(n_qubits_, true), omega_(), P_(n_qubits_), Q_(n_qubits_) {
  // initialize  by the basis vector 00...0

  // G and F are identity matrices
  for (unsigned q = 0; q < n_qubits_; q++) {
    G_[q].set(q, one);
    F_[q].set(q, one);
  }
  omega_.makeOne();
}

StabilizerState::StabilizerState(const StabilizerState &rhs)
    : n_qubits_(rhs.n_qubits_), // number of qubits
      gamma1_(rhs.gamma1_), gamma2_(rhs.gamma2_), F_(rhs.F_), G_(rhs.G_),
      M_(rhs.M_), v_(rhs.v_), s_(rhs.s_), omega_(rhs.omega_), P_(rhs.P_),
      Q_(rhs.Q_) {}

void StabilizerState::CompBasisVector(BitVector &x) {
  s_ = x;
  v_.zero();
  gamma1_.zero();
  gamma2_.zero();
  omega_.makeOne();
  // G and F are identity matrices, M is zero matrix
  for (unsigned q = 0; q < n_qubits_; q++) {
    M_[q].zero();
    G_[q].set(q, one);
    F_[q].set(q, one);
  }
}

void StabilizerState::HadamardBasisVector(BitVector &x) {
  s_.zero();
  v_ = x;
  gamma1_.zero();
  gamma2_.zero();
  omega_.makeOne();
  // G and F are identity matrices, M is zero matrix
  for (unsigned q = 0; q < n_qubits_; q++) {
    M_[q].zero();
    G_[q].set(q, one);
    F_[q].set(q, one);
  }
}

void StabilizerState::RightS(unsigned q) {
  M_[q] ^= F_[q];
  // update phase vector: gamma[p] gets gamma[p] - F_{p,q} (mod 4)   for all p
  gamma2_ ^= F_[q] ^ (gamma1_ & F_[q]);
  gamma1_ ^= F_[q];
}

void StabilizerState::RightSdag(unsigned q) {
  M_[q] ^= F_[q];
  // update phase vector: gamma[p] gets gamma[p] + F_{p,q} (mod 4)   for all p
  gamma1_ ^= F_[q];
  gamma2_ ^= F_[q] ^ (gamma1_ & F_[q]);
}

void StabilizerState::RightZ(unsigned q) {
  // update phase vector: gamma[p] gets gamma[p] + 2F_{p,q} (mod 4)   for all p
  gamma2_ ^= F_[q];
}

void StabilizerState::S(unsigned q) {
  for (unsigned p = 0; p < n_qubits_; p++) {
    M_[p].set_xor(q, G_[p][q]);
  }
  // update phase vector:  gamma[q] gets gamma[q] - 1
  gamma1_.set_xor(q, one);
  gamma2_.set_xor(q, gamma1_[q]);
}

void StabilizerState::Sdag(unsigned q) {
  for (unsigned p = 0; p < n_qubits_; p++) {
    M_[p].set_xor(q, G_[p][q]);
  }
  // update phase vector:  gamma[q] gets gamma[q] + 1
  gamma2_.set_xor(q, gamma1_[q]);
  gamma1_.set_xor(q, one);
}

void StabilizerState::Z(unsigned q) {
  // update phase vector:  gamma[q] gets gamma[q] + 2
  gamma2_.set_xor(q, one);
}

void StabilizerState::X(unsigned q) {
  // Commute the Pauli through UC
  BitVector x_string(n_qubits_);
  BitVector z_string(n_qubits_);

  for (uint_t i = 0; i < n_qubits_; i++) {
    x_string.set(i, F_[i][q]);
    z_string.set(i, M_[i][q]);
  }

  // Initial phase correction
  uint_t phase = 2 * (uint_t)gamma1_[q] + 4 * (uint_t)gamma2_[q];
  // Commute the z_string through the hadamard layer
  //  Each z that hits a hadamard becomes a Pauli X
  s_ ^= (z_string & v_);
  // Remaining Z gates add a global phase from their action on the s string
  z_string = (z_string & ~v_) & s_;
  phase += 4 * z_string.parity();
  // Commute the x_string through the hadamard layer
  //  Any remaining X gates update s
  s_ ^= (x_string & ~v_);
  // New z gates add a global phase from their action on the s string
  x_string = (x_string & v_) & s_;
  phase += 4 * (uint_t)x_string.parity();
  // Update the global phase
  omega_.e = (omega_.e + phase) % 8;
}

void StabilizerState::Y(unsigned q) {
  Z(q);
  X(q);
  // Add a global phase of -i
  omega_.e = (omega_.e + 2) % 8;
}

void StabilizerState::RightCX(unsigned q, unsigned r) {
  G_[q] ^= G_[r];
  F_[r] ^= F_[q];
  M_[q] ^= M_[r];
}

void StabilizerState::CX(unsigned q, unsigned r) {
  bool b = false;
  for (unsigned p = 0; p < n_qubits_; p++) {
    b = (b != (M_[p][q] && F_[p][r]));
    G_[p].set_xor(r, G_[p][q]);
    F_[p].set_xor(q, F_[p][r]);
    M_[p].set_xor(q, M_[p][r]);
  }
  // update phase vector as
  // gamma[q] gets gamma[q] + gamma[r] + 2*b (mod 4)
  if (b)
    gamma2_.set_xor(q, one);
  b = gamma1_[q] && gamma1_[r];
  gamma1_.set_xor(q, gamma1_[r]);
  gamma2_.set_xor(q, gamma2_[r]);
  if (b)
    gamma2_.set_xor(q, one);
}

void StabilizerState::RightCZ(unsigned q, unsigned r) {
  M_[q] ^= F_[r];
  M_[r] ^= F_[q];
  gamma2_ ^= (F_[q] & F_[r]);
}

void StabilizerState::CZ(unsigned q, unsigned r) {
  for (unsigned p = 0; p < n_qubits_; p++) {
    M_[p].set_xor(q, G_[p][r]);
    M_[p].set_xor(r, G_[p][q]);
  }
}

pauli_t StabilizerState::GetPauliX(BitVector &x) {
  // make sure that M-transposed and F-transposed have been already computed
  pauli_t R(n_qubits_);

  for (unsigned pos = 0; pos < n_qubits_; pos++) {
    if (x[pos]) {
      pauli_t P1(n_qubits_); // make P1=U_C^{-1} X_{pos} U_C
      P1.e = 1 * (uint_t)gamma1_[pos];
      P1.e += 2 * (uint_t)gamma2_[pos];
      for (uint_t i = 0; i < n_qubits_; i++) {
        P1.X.set(i, F_[i][pos]);
        P1.Z.set(i, M_[i][pos]);
      }
      R *= P1;
    }
  }
  return R;
}

scalar_t StabilizerState::Amplitude(BitVector &x) {
  // compute Pauli U_C^{-1} X(x) U_C
  P_ = GetPauliX(x);

  if (!omega_.eps)
    return omega_; // the state is zero

  // now the amplitude = complex conjugate of <s|U_H P |0^n>
  // Z-part of P is absorbed into 0^n

  scalar_t amp;
  amp.e = 2 * P_.e;
  int p = (int)v_.popcount();
  amp.p = -1 * p; // each Hadamard gate contributes 1/sqrt(2)
  bool isNonZero = true;

  for (unsigned q = 0; q < n_qubits_; q++) {
    if (v_[q]) {
      amp.e += 4 * (s_[q] && P_.X[q]); // minus sign that comes from <1|H|1>
    } else {
      isNonZero = (P_.X[q] == s_[q]);
    }
    if (!isNonZero)
      break;
  }

  amp.e %= 8;
  if (isNonZero) {
    amp.conjugate();
  } else {
    amp.eps = 0;
    return amp;
  }

  // multiply amp by omega
  amp.p += omega_.p;
  amp.e = (amp.e + omega_.e) % 8;
  return amp;
}

scalar_t StabilizerState::ProposeFlip(unsigned flip_pos) {
  // Q gets Pauli operator U_C^{-1} X_{flip_pos} U_C
  Q_.e = 1 * gamma1_[flip_pos];
  Q_.e += 2 * gamma2_[flip_pos];
  for (uint_t i = 0; i < n_qubits_; i++) {
    Q_.X.set(i, F_[i][flip_pos]);
    Q_.Z.set(i, M_[i][flip_pos]);
  }
  Q_ *= P_;

  if (!omega_.eps)
    return omega_; // the state is zero

  // the rest is the same as Amplitude() except that P becomes Q

  // now the amplitude = complex conjugate of <s|U_H Q |0^n>
  // Z-part of Q is absorbed into 0^n

  // the rest is the same as Amplitude() except that P is replaced by Q

  scalar_t amp;
  amp.e = 2 * Q_.e;
  // each Hadamard gate contributes 1/sqrt(2)
  amp.p = -1 * (int)(v_.popcount());
  bool isNonZero = true;

  for (unsigned q = 0; q < n_qubits_; q++) {
    if (v_[q]) {
      amp.e += 4 * (s_[q] && Q_.X[q]); // minus sign that comes from <1|H|1>
    } else {
      isNonZero = (Q_.X[q] == s_[q]);
    }
    if (!isNonZero)
      break;
  }

  amp.e %= 8;
  if (isNonZero) {
    amp.conjugate();
  } else {
    amp.eps = 0;
    return amp;
  }

  // multiply amp by omega
  amp.p += omega_.p;
  amp.e = (amp.e + omega_.e) % 8;
  return amp;
}

BitVector StabilizerState::Sample() {
  BitVector x(n_qubits_);
  for (unsigned q = 0; q < n_qubits_; q++) {
    bool w = !!(s_[q]);
    w ^= (v_[q] && (rand() % 2));
    if (w)
      x ^= G_[q];
  }
  return x;
}

BitVector StabilizerState::Sample(BitVector &v_mask) {
  // v_mask is a uniform random binary string we use to sample the bits
  // of v in a single step.
  BitVector x(n_qubits_);
  BitVector masked_v = v_ & v_mask;
  for (unsigned q = 0; q < n_qubits_; q++) {
    bool w = !!(s_[q]);
    w ^= !!(masked_v[q]);
    if (w)
      x ^= G_[q];
  }
  return x;
}

void StabilizerState::UpdateSvector(BitVector t, BitVector u, unsigned b) {
  // take care of the trivial case: t=u
  if (t == u) // multiply omega by (1+i^b)/sqrt(2)
    switch (b) {
    case 0:
      omega_.p += 1;
      s_ = t;
      return;
    case 1:
      s_ = t;
      omega_.e = (omega_.e + 1) % 8;
      return;
    case 2:
      s_ = t;
      omega_.eps = 0;
      return;
    case 3:
      s_ = t;
      omega_.e = (omega_.e + 7) % 8;
      return;
    default:
      // we should not get here
      throw std::logic_error(
          "Invalid phase factor found b:" + std::to_string(b) + ".\n");
    }

  // now t and u are distinct
  // naming of variables roughly follows Section IIIA
  BitVector ut = u ^ t;
  BitVector nu0 = (~v_) & ut;
  BitVector nu1 = v_ & ut;
  //
  b %= 4;
  unsigned q = 0;
  if (!nu0.is_zero()) {

    // the subset nu0 is non-empty
    // find the first element of nu0
    q = 0;
    while (!nu0[q])
      q++;

    if (!nu0[q]) {
      throw std::logic_error(
          "Failed to find first bit of nu despite being non-empty.");
    }

    // if nu0 has size >1 then multiply U_C on the right by the first half of
    // the circuit VC
    nu0.set_xor(q, one); // set q-th bit to zero
    if (!nu0.is_zero())
      for (unsigned q1 = q + 1; q1 < n_qubits_; q1++)
        if (nu0[q1])
          RightCX(q, q1);

    // if nu1 has size >0 then apply the second half of the circuit VC
    if (!nu1.is_zero())
      for (unsigned q1 = 0; q1 < n_qubits_; q1++)
        if (nu1[q1])
          RightCZ(q, q1);

  } // if (nu0)
  else {
    // if we got here when nu0 is empty
    // find the first element of nu1
    q = 0;
    while (!nu1[q])
      q++;

    if (!nu1[q]) {
      throw std::logic_error("Expect at least one non-zero element in nu1.\n");
    }

    // if nu1 has size >1 then apply the circuit VC
    nu1.set_xor(q, one);
    if (!nu1.is_zero())
      for (unsigned q1 = q + 1; q1 < n_qubits_; q1++)
        if (nu1[q1])
          RightCX(q1, q);

  } // if (nu0) else

  // update the initial state
  // if t_q=1 then switch t_q and u_q
  // uint_fast64_t y,z;
  if (t[q]) {
    s_ = u;
    omega_.e = (omega_.e + 2 * b) % 8;
    b = (4 - b) % 4;
    if (!!(u[q])) {
    }
  } else
    s_ = t;

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

  bool a = v_[q];
  unsigned e1 = a * (b % 2) * (3 * b - 2);
  unsigned e2 = b % 2;
  bool e3 = ((!a) != (a && ((b % 2) > 0)));
  bool e4 = (((!a) && (b >= 2)) != (a && ((b == 1) || (b == 2))));

  // update CH-form
  // set q-th bit of s to e4
  s_.set(q, e4);

  // set q-th bit of v to e3
  v_.set(q, e3);

  // update the scalar factor omega
  omega_.e = (omega_.e + e1) % 8;

  // multiply the C-layer on the right by S^{e2} on the q-th qubit
  if (e2)
    RightS(q);
}

void StabilizerState::H(unsigned q) {
  // extract the q-th row of F,G,M
  BitVector rowF(n_qubits_, true);
  BitVector rowG(n_qubits_, true);
  BitVector rowM(n_qubits_, true);
  for (unsigned j = 0; j < n_qubits_; j++) {
    rowF.set_xor(j, F_[j][q]);
    rowG.set_xor(j, G_[j][q]);
    rowM.set_xor(j, M_[j][q]);
  }

  // after commuting H through the C and H laters it maps |s> to a state
  // sqrt(0.5)*[  (-1)^alpha |t> + i^{gamma[p]} (-1)^beta |u>  ]
  //
  // compute t,s,alpha,beta
  BitVector t = s_ ^ (rowG & v_);
  BitVector u = s_ ^ (rowF & (~v_)) ^ (rowM & v_);

  BitVector pc = rowG & (~v_) & s_;
  unsigned alpha = pc.popcount();
  pc = (rowM & (~v_) & s_) ^ (rowF & v_ & (rowM ^ s_));
  unsigned beta = pc.popcount();

  if (alpha % 2)
    omega_.e = (omega_.e + 4) % 8;
  // get the phase gamma[q]
  unsigned phase = (unsigned)gamma1_[q] + 2 * (unsigned)gamma2_[q];
  unsigned b = (phase + 2 * alpha + 2 * beta) % 4;

  // now the initial state is sqrt(0.5)*(|t> + i^b |u>)

  // take care of the trivial case
  if (t == u) {
    s_ = t;
    if (!((b == 1) || (b == 3))) // otherwise the state is not normalized
    {
      throw std::logic_error(
          "State is not properly normalised, b should be 1 or 3.\n");
    }
    if (b == 1)
      omega_.e = (omega_.e + 1) % 8;
    else
      omega_.e = (omega_.e + 7) % 8;
  } else
    UpdateSvector(t, u, b);
}

void StabilizerState::MeasurePauli(pauli_t PP) {
  // compute Pauli R = U_C^{-1} P U_C
  pauli_t R(n_qubits_);
  R.e = PP.e;

  for (unsigned j = 0; j < n_qubits_; j++)
    if (PP.X[j]) {
      // multiply R by U_C^{-1} X_j U_C
      // extract the j-th rows of F and M
      BitVector rowF(n_qubits_, true);
      BitVector rowM(n_qubits_, true);
      for (unsigned i = 0; i < n_qubits_; i++) {
        rowF.set_xor(i, F_[i][j]);
        rowM.set_xor(i, M_[i][j]);
      }
      BitVector pc = R.Z;
      pc &= rowF;
      R.e += 2 * pc.popcount(); // extra sign from Pauli commutation
      R.Z ^= rowM;
      R.X ^= rowF;
      R.e += (uint_t)gamma1_[j] + 2 * (uint_t)gamma2_[j];
    }
  for (unsigned q = 0; q < n_qubits_; q++) {
    BitVector pc = PP.Z;
    pc &= G_[q];
    R.Z.set_xor(q, (pc.popcount() & 1));
  }

  // now R=U_C^{-1} PP U_C
  // next conjugate R by U_H
  BitVector tempX = ((~v_) & R.X) ^ (v_ & R.Z);
  BitVector tempZ = ((~v_) & R.Z) ^ (v_ & R.X);
  // the sign flips each time a Hadamard hits Y on some qubit
  BitVector pc = v_;
  pc &= R.X;
  pc &= R.Z;
  R.e = (R.e + 2 * pc.popcount()) % 4;
  R.X = tempX;
  R.Z = tempZ;

  // now the initial state |s> becomes 0.5*(|s> + R |s>) = 0.5*(|s> + i^b |s ^
  // R.X>)
  BitVector sz = s_;
  sz &= R.Z;
  unsigned b = (R.e + 2 * (unsigned)sz.popcount()) % 4;
  BitVector sr = s_;
  sr ^= R.X;
  UpdateSvector(s_, sr, b);
  // account for the extra factor sqrt(1/2)
  omega_.p -= 1;
}

void StabilizerState::MeasurePauliProjector(
    const std::vector<pauli_t> &generators)
// Measure generators of a projector.
{
  for (uint_fast64_t i = 0; i < generators.size(); i++) {
    this->MeasurePauli(generators[i]);
    if (omega_.eps == 0) {
      break;
    }
  }
}

scalar_t StabilizerState::InnerProduct(const BitVector &A_diag1,
                                       const BitVector &A_diag2,
                                       const std::vector<BitVector> &A) {
  BitVector K_diag1(n_qubits_, true), K_diag2(n_qubits_, true);
  BitVector J_diag1 = gamma1_;
  BitVector J_diag2 = gamma2_;
  std::vector<BitVector> J(n_qubits_, BitVector(n_qubits_, true));
  std::vector<BitVector> K(n_qubits_, BitVector(n_qubits_, true));
  std::vector<BitVector> placeholder(n_qubits_, BitVector(n_qubits_, true));

  // Setup the J matrix
  for (size_t i = 0; i < n_qubits_; i++) {
    BitVector mt(n_qubits_, true);
    for (uint_t k = 0; k < n_qubits_; k++)
      mt.set(k, M_[k][i]);

    for (size_t j = i; j < n_qubits_; j++) {
      BitVector ft(n_qubits_, true);
      for (uint_t k = 0; k < n_qubits_; k++)
        ft.set(k, F_[k][j] & mt[k]);
      if (ft.parity()) {
        J[i].set_or(j, true);
        J[j].set_or(i, true);
      }
    }
  }

  // Calculate the matrix J =  A+J
  J_diag2 = (A_diag2 ^ J_diag2 ^ (A_diag1 & J_diag1));
  J_diag1 = (A_diag1 ^ J_diag1);
  for (size_t i = 0; i < n_qubits_; i++) {
    J[i] ^= A[i];
  }
  // placeholder = J*G)
  for (size_t i = 0; i < n_qubits_; i++) {
    // Grab column i of J, it's symmetric
    BitVector col_i = J[i];
    for (size_t j = 0; j < n_qubits_; j++) {
      BitVector p = col_i;
      p &= G_[j];
      if (p.parity()) {
        placeholder[j].set_or(i, true);
      }
    }
  }
  // K = GT*placeholder
  for (size_t i = 0; i < n_qubits_; i++) {
    BitVector col_i(n_qubits_);
    col_i = G_[i];
    for (size_t j = i; j < n_qubits_; j++) {
      BitVector p = col_i & placeholder[j];
      if (p.parity()) {
        K[j].set_or(i, true);
        K[i].set_or(j, true);
      }
    }
    for (size_t r = 0; r < n_qubits_; r++) {
      if (col_i[r]) {
        if (K_diag1[i] & J_diag1[r]) {
          K_diag2.set_xor(i, true);
        }
        K_diag1.set_xor(i, J_diag1[r]);
        K_diag2.set_xor(i, J_diag2[r]);
      }
      for (size_t k = r + 1; k < n_qubits_; k++) {
        if (J[k][r] & col_i[r] & col_i[k]) {
          K_diag2.set_xor(i, true);
        }
      }
    }
  }
  unsigned col = 0;
  QuadraticForm q(v_.popcount());
  // We need to setup a quadratic form to evaluate the Exponential Sum
  for (size_t i = 0; i < n_qubits_; i++) {
    if (v_[i]) {
      // D = Diag(K(1,1)) + 2*[s + s*K](1)
      //  J = K(1,1);
      q.D1.set_xor(col, K_diag1[i]);
      BitVector p = K[i];
      p &= s_;
      q.D2.set_xor(col, K_diag2[i] ^ s_[i] ^ p.parity());
      // q.D2 ^= ((s >> i) & one) * shift;
      // q.D2 ^= AER::Utils::hamming_parity(K[i] & s) * shift;
      unsigned row = 0;
      for (size_t j = 0; j < n_qubits_; j++) {
        if (v_[j]) {
          q.J[col].set_or(row, K[i][j]);
          row++;
        }
      }
      col++;
    }
  }
  // Q = 4* (s.v) + sKs
  BitVector p = s_ & v_;
  q.Q = (uint_t)p.parity() * 4;
  for (size_t i = 0; i < n_qubits_; i++) {
    if (s_[i]) {
      q.Q = (q.Q + 4 * (uint_t)K_diag2[i] + 2 * (uint_t)K_diag1[i]) %
            8; // + 2*((K_diag1 >> i) & one))%8;
      for (size_t j = i + 1; j < n_qubits_; j++) {
        if (s_[j] & K[j][i]) {
          q.Q ^= 4;
        }
      }
    }
  }
  scalar_t amp = q.ExponentialSum();
  // Reweight by 2^{-(n+|v|)}/2
  amp.p -= (n_qubits_ + v_.popcount());
  // We need to further multiply by omega*
  scalar_t psi_amp(omega_);
  psi_amp.conjugate();
  amp *= psi_amp;
  return amp;
}

double NormEstimate(std::vector<StabilizerState> &states,
                    const std::vector<std::complex<double>> &phases,
                    const std::vector<BitVector> &Samples_d1,
                    const std::vector<BitVector> &Samples_d2,
                    const std::vector<std::vector<BitVector>> &Samples) {
  // Norm estimate for a state |psi> = \sum_{i} c_{i}|phi_{i}>
  double xi = 0;
  unsigned L = Samples_d1.size();
  // std::vector<double> data = (L,0.);
  for (size_t i = 0; i < L; i++) {
    double re_eta = 0., im_eta = 0.;
    const int_t END = states.size();
#pragma omp parallel for reduction(+ : re_eta) reduction(+ : im_eta)
    for (int_t j = 0; j < END; j++) {
      if (states[j].ScalarPart().eps != 0) {
        scalar_t amp =
            states[j].InnerProduct(Samples_d1[i], Samples_d2[i], Samples[i]);
        if (amp.eps != 0) {
          if (amp.e % 2) {
            amp.p--;
          }
          double mag = pow(2, amp.p / (double)2);
          cdouble phase(RE_PHASE[amp.e], IM_PHASE[amp.e]);
          phase *= conj(phases[j]);
          re_eta += (mag * real(phase));
          im_eta += (mag * imag(phase));
        }
      }
    }
    xi += (pow(re_eta, 2) + pow(im_eta, 2));
  }
  return std::pow(2, states[0].NQubits()) * (xi / L);
}

double ParallelNormEstimate(std::vector<StabilizerState> &states,
                            const std::vector<std::complex<double>> &phases,
                            const std::vector<BitVector> &Samples_d1,
                            const std::vector<BitVector> &Samples_d2,
                            const std::vector<std::vector<BitVector>> &Samples,
                            int n_threads) {
  double xi = 0;
  unsigned L = Samples_d1.size();
  unsigned chi = states.size();
  for (uint_fast64_t i = 0; i < L; i++) {
    double re_eta = 0., im_eta = 0.;
#pragma omp parallel for reduction(+:re_eta) reduction(+:im_eta) num_threads(n_threads)
    for (int_t j = 0; j < chi; j++) {
      if (states[j].ScalarPart().eps != 0) {
        scalar_t amp =
            states[j].InnerProduct(Samples_d1[i], Samples_d2[i], Samples[i]);
        if (amp.eps != 0) {
          if (amp.e % 2) {
            amp.p--;
          }
          double mag = pow(2, amp.p / (double)2);
          cdouble phase(RE_PHASE[amp.e], IM_PHASE[amp.e]);
          phase *= conj(phases[j]);
          re_eta += (mag * real(phase));
          im_eta += (mag * imag(phase));
        }
      }
    }
    xi += (pow(re_eta, 2) + pow(im_eta, 2));
  }
  return pow(2., states[0].NQubits()) * (xi / L);
}

} // namespace CHSimulator
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
