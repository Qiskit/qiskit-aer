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

#ifndef _aer_chsimulator_gates_hpp
#define _aer_chsimulator_gates_hpp

#define _USE_MATH_DEFINES
#include <math.h>

#include <complex>
#include <cstdint>

#include "framework/operations.hpp"
#include "framework/types.hpp"

namespace CHSimulator
{
  using uint_t = uint_fast64_t;
  using complex_t = std::complex<double>;

  enum class Gates {
    u0, u1, id, x, y, z, h, s, sdg, t, tdg,
    cx, cz, swap,
    ccx, ccz
  };

  enum class Gatetypes {
    pauli,
    clifford,
    non_clifford
  };

  const AER::stringmap_t<Gatetypes> gate_types_ = {
    {"id", Gatetypes::pauli},     // Pauli-Identity gate
    {"x", Gatetypes::pauli},       // Pauli-X gate
    {"y", Gatetypes::pauli},       // Pauli-Y gate
    {"z", Gatetypes::pauli},       // Pauli-Z gate
    {"s", Gatetypes::clifford},       // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gatetypes::clifford},   // Conjugate-transpose of Phase gate
    {"h", Gatetypes::clifford},       // Hadamard gate (X + Z / sqrt(2))
    {"t", Gatetypes::non_clifford},       // T-gate (sqrt(S))
    {"tdg", Gatetypes::non_clifford},   // Conjguate-transpose of T gate
    // Waltz Gates
    {"u0", Gatetypes::pauli},     // idle gate in multiples of X90
    {"u1", Gatetypes::non_clifford},     // zero-X90 pulse waltz gate
    // Two-qubit gates
    {"CX", Gatetypes::clifford},     // Controlled-X gate (CNOT)
    {"cx", Gatetypes::clifford},     // Controlled-X gate (CNOT)
    {"cz", Gatetypes::clifford},     // Controlled-Z gate
    {"swap", Gatetypes::clifford}, // SWAP gate
    // Three-qubit gates
    {"ccx", Gatetypes::non_clifford},    // Controlled-CX gate (Toffoli)
    {"ccz", Gatetypes::non_clifford}     // Constrolled-CZ gate (H3 Toff H3)
  };

  using sample_branch_t = std::pair<complex_t, Gates>;

  const double root2 = std::sqrt(2);
  const double root1_2 = 1./root2;
  const complex_t pi_over_8_phase(0., M_PI/8);
  const complex_t omega(root1_2, root1_2);
  const complex_t omega_star(root1_2, -1*root1_2);
  const complex_t root_omega = std::exp(pi_over_8_phase);
  const complex_t root_omega_star = std::conj(root_omega);

  const double tan_pi_over_8 = std::tan(M_PI/8.);

  //Base class for sampling over non-Clifford gates in the Sum Over Cliffords routine. 
  struct Sample {
  public:
    Sample() = default;
    virtual ~Sample() = default;
    Sample(const Sample& other) : branches(other.branches) {};
    std::vector<sample_branch_t> branches;

    virtual sample_branch_t sample(double r) const = 0;
  };

  //Functor class that defines how to sample branches over a U1 operation
  //Used for caching each U1 gate angle we encounter in the circuit
struct U1Sample : public Sample
{
  double p_threshold;

  U1Sample() = default;
  U1Sample(double lambda);

  U1Sample(const U1Sample &other) : Sample(other)
  {
    p_threshold = other.p_threshold;
  }

  ~U1Sample() = default;

  sample_branch_t sample(double r) const override;
};

U1Sample::U1Sample(double lambda)
{
  // Shift parameter into +- 2 Pi
  uint_t shift_factor = std::floor(std::abs(lambda)/(2*M_PI));
  if (shift_factor != 0)
  {
    if(lambda < 0)
    {
      lambda += shift_factor*(2*M_PI);
    } 
    else
    {
      lambda -= shift_factor*(2*M_PI);
    }
  }
  // Shift parameter into +- Pi
  if (lambda > M_PI)
  {
    lambda -= 2*M_PI;
  }
  else if (lambda < -1*M_PI)
  {
    lambda += 2*M_PI;
  }
  // Compute the coefficients
  double angle = std::abs(lambda);
  bool s_z_quadrant = (angle > M_PI/2);
  if(s_z_quadrant)
  {
    angle = angle - M_PI/2;
  }
  angle /= 2;
  complex_t coeff_0 = std::cos(angle)-std::sin(angle);
  complex_t coeff_1 = root2*std::sin(angle);
  if(lambda < 0)
  {
    coeff_0 *= root_omega_star;
    coeff_1 = coeff_1 * root_omega;
    if(s_z_quadrant)
    {
      branches = 
      {
        sample_branch_t(coeff_0, Gates::sdg),
        sample_branch_t(coeff_1, Gates::z)
      };
    }
    else
    {
      branches = 
      {
        sample_branch_t(coeff_0, Gates::id),
        sample_branch_t(coeff_1, Gates::sdg)
      };
    }
  }
  else
  {
    coeff_0 *= root_omega;
    coeff_1 = coeff_1 * root_omega_star;
    if(s_z_quadrant)
    {
      branches = 
      {
        sample_branch_t(coeff_0, Gates::s),
        sample_branch_t(coeff_1, Gates::z)
      };
    }
    else
    {
      branches = 
      {
        sample_branch_t(coeff_0, Gates::id),
        sample_branch_t(coeff_1, Gates::s)
      };
    }
  }
  p_threshold = std::abs(coeff_0) / (std::abs(coeff_0)+std::abs(coeff_1));
}

sample_branch_t U1Sample::sample(double r) const
{
  if (r<p_threshold)
  {
    return branches[0];
  }
  else
  {
    return branches[1];
  }
}

  // Stabilizer extent for different non-Clifford unitaries
  // Special case of Eq. 28 in arXiv:1808.00128
  const double t_extent = std::pow(1./(std::cos(M_PI/8.)), 2);
  // Equation 32 in arXiv: 1808.00128
  const double ccx_extent = 16./9.;
  const double ccx_coeff = 1./6.;
  //General result for z rotations, Eq. 28 in arXiv 1809.00128
  inline double u1_extent(double lambda)
  {
    // Shift parameter into +- 2 Pi
    uint_t shift_factor = std::floor(std::abs(lambda)/(2*M_PI));
    if (shift_factor != 0)
    {
      if(lambda < 0)
      {
        lambda += shift_factor*(2*M_PI);
      } 
      else
      {
        lambda -= shift_factor*(2*M_PI);
      }
    }
    // Shift parameter into +- Pi
    if (lambda > M_PI)
    {
      lambda -= 2*M_PI;
    }
    else if (lambda < -1*M_PI)
    {
      lambda += 2*M_PI;
    }
    double angle = std::abs(lambda);
    bool s_z_quadrant = (angle > M_PI/2);
    if(s_z_quadrant)
    {
      angle = angle - M_PI/2;
    }
    angle /= 2;
    return std::pow(std::cos(angle) + tan_pi_over_8*std::sin(angle), 2.);
  }

}



#endif
