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

#ifndef CORE_HPP
#define CORE_HPP

#include <array>
#include <climits>
#include <complex>
#include <cstdint>
#include <iostream>
#include <vector>

#include "framework/utils.hpp"

namespace CHSimulator {

static const std::array<int, 8> RE_PHASE = {1, 1, 0, -1, -1, -1, 0, 1};
static const std::array<int, 8> IM_PHASE = {0, 1, 1, 1, 0, -1, -1, -1};

using complex_t = std::complex<double>;
using uint_t = uint_fast64_t;
using int_t = int_fast64_t;

extern const uint_t zer = 0U;
extern const uint_t one = 1U;

struct scalar_t {
  // complex numbers of the form eps * 2^{p/2} * exp(i (pi/4)*e )
  // eps=0,1       p=integer         e=0,1,...,7
  // if eps=0 then p and e are arbitrary
  int eps = 1;
  int p = 0;
  int e = 0;
  // constructor makes number 1
  scalar_t() = default;
  scalar_t(const scalar_t &rhs) = default;
  scalar_t(const std::complex<double> coeff) {
    double abs_val = std::abs(coeff);
    if (std::abs(abs_val - 0.) < 1e-8) {
      eps = 0;
    } else {
      p = (int)std::log2(std::pow(abs_val, 2));
      bool is_real_zero = (std::abs(coeff.real() - 0) < 1e-8);
      bool is_imag_zero = (std::abs(coeff.imag() - 0) < 1e-8);
      bool is_real_posi = (coeff.real() > 0) && !(is_real_zero);
      bool is_imag_posi = (coeff.imag() > 0) && !(is_imag_zero);
      char switch_var = (is_real_zero * 1) + (is_imag_zero * 2) +
                        (is_real_posi * 4) + (is_imag_posi * 8);
      switch (switch_var) {
      case 0:
        e = 5;
        break;
      case 1:
        e = 6;
        break;
      case 2:
        e = 4;
        break;
      case 4:
        e = 7;
        break;
      case 6:
        e = 0;
        break;
      case 8:
        e = 3;
        break;
      case 9:
        e = 2;
        break;
      case 12:
        e = 1;
        break;
      default:
        throw std::runtime_error("Unsure what to do here chief.");
        break;
      }
    }
  }

  // complex conjugation
  inline void conjugate() {
    e %= 8;
    e = (8 - e) % 8;
  }

  inline void makeOne() {
    eps = 1;
    p = 0;
    e = 0;
  }

  scalar_t &operator*=(const scalar_t &rhs);
  scalar_t operator*(const scalar_t &rhs) const;

  std::complex<double> to_complex() const {
    if (eps == 0) {
      return {0., 0.};
    }
    std::complex<double> mag(std::pow(2, p / (double)2), 0.);
    std::complex<double> phase(RE_PHASE[e], IM_PHASE[e]);
    if (e % 2) {
      phase /= std::sqrt(2);
    }
    return mag * phase;
  }

  void Print() {
    std::cout << "eps=" << eps << ", p=" << p << ", e=" << e << std::endl;
  }
};

// Below we represent n-bit strings by uint64_t integers
struct pauli_t {
  // n-qubit Pauli operators: i^e * X(x) * Z(z)
  uint_fast64_t X; // n-bit string
  uint_fast64_t Z; // n-bit string
  unsigned e = 0;  // takes values 0,1,2,3

  // constructor makes the identity Pauli operator
  pauli_t();
  pauli_t(const pauli_t &p) = default;

  // multiplication of Pauli operators
  pauli_t &operator*=(const pauli_t &rhs);
};

struct QuadraticForm {
  unsigned int n;
  int Q;
  uint_fast64_t D1;
  uint_fast64_t D2;
  std::vector<uint_fast64_t> J;
  scalar_t ExponentialSum();
  QuadraticForm(unsigned n_qubits);
  QuadraticForm(const QuadraticForm &rhs);
  QuadraticForm &operator-=(const QuadraticForm &rhs);
  // ~QuadraticForm();
};

bool operator==(const QuadraticForm &lhs, const QuadraticForm &rhs);

std::ostream &operator<<(std::ostream &os, const QuadraticForm &q);

void Print(uint_fast64_t x, unsigned n);              // print a bit string
void Print(std::vector<uint_fast64_t> A, unsigned n); // print a binary matrix
// A[i] represents i-th column of the matrix

//---------------------------------------//
// Implementations                       //
//---------------------------------------//

scalar_t &scalar_t::operator*=(const scalar_t &rhs) {
  p += (rhs.p);
  e += (rhs.e);
  e %= 8;
  eps *= rhs.eps;
  return *this;
}

scalar_t scalar_t::operator*(const scalar_t &rhs) const {
  scalar_t out;
  out.p = (p + rhs.p);
  out.e = (e + rhs.e);
  out.e = (out.e % 8);
  out.eps = (eps * rhs.eps);
  return out;
}

pauli_t::pauli_t() : X(zer), Z(zer) {}

pauli_t &pauli_t::operator*=(const pauli_t &rhs) {
  unsigned overlap =
      AER::Utils::popcount(Z & rhs.X); // commute rhs.X to the left
  X ^= rhs.X;
  Z ^= rhs.Z;
  e = (e + rhs.e + 2 * overlap) % 4;
  return (*this);
}

class QubitException : public std::exception {
  virtual const char *what() const throw() {
    return "Length error: We cannot compute a quadratic form with more that 63 "
           "dimensions.";
  }
} qubex;

QuadraticForm::QuadraticForm(unsigned int n_qubits)
    : n(n_qubits), Q(0), D1(zer), D2(zer), J(n_qubits, zer) {
  if (n > 63) {
    throw qubex;
  }
}

QuadraticForm::QuadraticForm(const QuadraticForm &rhs) : J(rhs.n, zer) {
  n = rhs.n;
  Q = rhs.Q;
  D1 = rhs.D1;
  D2 = rhs.D2;
  for (size_t i = 0; i < n; i++) {
    J[i] = rhs.J[i];
  }
}

QuadraticForm &QuadraticForm::operator-=(const QuadraticForm &rhs) {
  Q = (Q - rhs.Q) % 8;
  if (Q < 0) {
    Q += 8;
  }
  // TODO: D Update
  for (size_t j = 0; j < n; j++) {
    J[j] ^= rhs.J[j];
  }
  return *this;
}

bool operator==(const QuadraticForm &lhs, const QuadraticForm &rhs) {
  if (lhs.Q != rhs.Q) {
    return false;
  }
  if (lhs.D1 != rhs.D1) {
    return false;
  }
  if (lhs.D2 != rhs.D2) {
    return false;
  }
  if (lhs.J != rhs.J) {
    return false;
  }
  return true;
}

std::ostream &operator<<(std::ostream &os, const QuadraticForm &q) {
  os << "Q: " << q.Q << std::endl;
  os << "D:";
  for (size_t i = 0; i < q.n; i++) {
    os << " " << (2 * (2 * ((q.D2 >> i) & one) + ((q.D1 >> i) & one)));
  }
  os << std::endl;
  os << "J:\n";
  for (size_t i = 0; i < q.n; i++) {
    for (size_t j = 0; j < q.n; j++) {
      os << ((q.J[i] >> j) & one) << " ";
    }
    os << "\n";
  }
  return os;
}

// Computes exponential sums  Z=sum_x exp(j*(pi/4)*q(x))
// where q(x) is a quadratic form and x runs over n-bit strings
// Usage: run "mex ExponentialSum.c" in matlab command line
// This creates the mex file.
// Now one can compute the exponential sum by calling
// [p,m,eps]=ExponentialSum(J0,J1,J2)
// from any matlab script
//
// Input parameters define a quadratic form
// q(x)=q(x_1,...,x_n)
// q(x)=J0 + sum_{a=1}^n J1_a x_a + 4*sum_{1<=a<b<=n| J2_{a,b} x_a x_b  (mod 8)
//
// Expects          J0=0,1,...,7
//                  J1_a=0,2,4,6
//                  J2_{a,b}=0,1
//                  J2 = symmetric binary matrix with zero diagonal
// IMPORTANT: J2 MUST HAVE LOGICAL TYPE !!
//
// Output: integers p>=0, m=0,1,2,..,7, and eps=0,1  such that
// Z = sum_x exp(j*(pi/4)*q(x)) = eps*2^(p/2)*exp(j*pi*m/4)
// if eps=0 then p and m can be ignored
scalar_t QuadraticForm::ExponentialSum() {

  // Variables for Z2-valued exponential sums
  // Z=(2^pow2)*sigma
  // where pow2 is integer
  // sigma=0,1
  // if Z=0 then pow2,sigma contain junk and isZero=1
  int pow2_real = 0;
  bool sigma_real = false;
  bool isZero_real = false;

  int pow2_imag = 0;
  bool sigma_imag = false;
  bool isZero_imag = false;
  scalar_t amp;
  amp.makeOne();

  int m = n + 1;

  // Lreal, Limag define the linear part of the Z2-valued forms
  // M defines the quadratic part of the Z2-valued forms
  // each column of M is represented by long integer
  uint_fast64_t Lreal, c = zer;
  std::vector<uint_fast64_t> M(n + 1, zer);
  Lreal = D2;
  for (size_t i = 0; i < n; i++) {
    if ((D1 >> i) & one) {
      c |= (one << i);
      M[n] |= (one << i);
    }
  }
  uint_fast64_t Limag = Lreal;
  Limag ^= (one << n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = (i + 1); j < n; j++) {
      bool x = !!((J[i] >> j) & one);
      bool c1 = !!((c >> i) & one);
      bool c2 = !!((c >> j) & one);
      if (x ^ (c1 & c2)) {
        M[j] ^= (one << i);
      }
    }
  }
  uint_fast64_t active = zer;
  active = ~(active);
  int nActive = m;
  int firstActive = 0;
  while (nActive >= 1) {
    // find the first active variable
    int i1;
    for (i1 = firstActive; i1 < m; i1++) {
      if ((active >> i1) & one) {
        break;
      }
    }
    firstActive = i1;
    // find i2 such that M(i1,i2)!=M(i2,i1)
    int i2;
    bool isFound = false;
    for (i2 = 0; i2 < m; i2++) {
      isFound = (((M[i1] >> i2) & one) != ((M[i2] >> i1) & one));
      if (isFound) {
        break;
      }
    }
    bool L1real = !!(((Lreal >> i1) & one) ^ ((M[i1] >> i1) & one));
    bool L1imag = !!(((Limag >> i1) & one) ^ ((M[i1] >> i1) & one));
    // take care of the trivial cases
    if (!isFound) {
      // the form is linear in the variable i1
      if (L1real)
        isZero_real = true;

      pow2_real++;

      if (L1imag)
        isZero_imag = true;

      pow2_imag++;

      nActive--;
      uint_fast64_t not_shift;
      not_shift = ~(one << i1);
      active &= not_shift;
      // set column i1 to zero
      M[i1] = zer;
      // set row i1 to zero
      for (int j = 0; j < m; j++) {
        M[j] &= not_shift;
      }
      Lreal &= not_shift;
      Limag &= not_shift;

      if (isZero_real && isZero_imag) {
        amp.eps = 0;
        break;
      }
      continue;
    } // trivial cases
    bool L2real = !!(((Lreal >> i2) & one) ^ ((M[i2] >> i2) & one));
    bool L2imag = !!(((Limag >> i2) & one) ^ ((M[i2] >> i2) & one));
    Lreal &= ~(one << i1);
    Lreal &= ~(one << i2);
    Limag &= ~(one << i1);
    Limag &= ~(one << i2);
    // Extract rows i1 and i2 of M
    uint_fast64_t m1 = zer;
    uint_fast64_t m2 = zer;
    for (int j = 0; j < m; j++) {
      m1 ^= ((M[j] >> i1) & one) << j;
      m2 ^= ((M[j] >> i2) & one) << j;
    }
    m1 ^= M[i1];
    m2 ^= M[i2];
    m1 &= ~(one << i1);
    m1 &= ~(one << i2);
    m2 &= ~(one << i1);
    m2 &= ~(one << i2);
    // set columns i1, i2 to zero
    M[i1] = zer;
    M[i2] = zer;
    // set rows i1,i2 to zero
    for (int j = 0; j < m; j++) {
      M[j] &= ~(one << i1);
      M[j] &= ~(one << i2);
    }

    // update the std::vectors Lreal, Limag
    if (!isZero_real) {
      pow2_real++;
      sigma_real ^= (L1real && L2real);
      if (L1real) {
        Lreal ^= m2;
      }
      if (L2real) {
        Lreal ^= m1;
      }
    }
    if (!isZero_imag) {
      pow2_imag++;
      sigma_imag ^= (L1imag && L2imag);
      if (L1imag) {
        Limag ^= m2;
      }
      if (L2imag) {
        Limag ^= m1;
      }
    }
    // update matrix M
    for (int j = 0; j < m; j++) {
      if ((m2 >> j) & one) {
        M[j] ^= m1;
      }
    }
    active &= ~(one << i1);
    active &= ~(one << i2);
    nActive -= 2;
  }
  // int J0 = Q;
  // int J0 = inJ0[0];
  Q %= 8;

  // Combine the real and the imaginary parts
  if (isZero_imag) {
    amp.p = 2 * pow2_real - 2;
    amp.e = (4 * sigma_real + Q) % 8;
  } else if (isZero_real) {
    amp.p = 2 * pow2_imag - 2;
    amp.e = (2 + 4 * sigma_imag + Q) % 8;
  } else {
    // now both parts are nonzero
    amp.p = 2 * pow2_real - 1;
    amp.eps = 1;
    if (sigma_real == 0) {
      if (sigma_imag == 0) {
        amp.e = (1 + Q) % 8;
      } else {
        amp.e = (7 + Q) % 8;
      }
    } else {
      if (sigma_imag == 0) {
        amp.e = (3 + Q) % 8;
      } else {
        amp.e = (5 + Q) % 8;
      }
    }
  }
  return amp;
}

// prints a binary form of x
void Print(uint_fast64_t x, unsigned n) {
  for (unsigned i = 0; i < n; i++) {
    std::cout << ((x & (1ULL << i)) > 0);
  }
  std::cout << std::endl;
}

// prints a binary form of each element of A
// each element of A represents column of a binary matrix
void Print(std::vector<uint_fast64_t> A, unsigned n) {
  for (unsigned row = 0; row < n; row++) {
    for (unsigned col = 0; col < n; col++) {
      std::cout << ((A[col] & (one << row)) > 0);
    }
    std::cout << std::endl;
  }
}

} // namespace CHSimulator
#endif
