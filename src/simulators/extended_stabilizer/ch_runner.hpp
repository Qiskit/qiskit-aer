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

#ifndef chsimulator_runner_hpp
#define chsimulator_runner_hpp

#include "chlib/core.hpp"
#include "chlib/chstabilizer.hpp"
#include "gates.hpp"

#include "framework/json.hpp"
#include "framework/operations.hpp"
#include "framework/rng.hpp"
#include "framework/types.hpp"


#define _USE_MATH_DEFINES
#include <math.h>

#include <cstdint>
#include <complex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace CHSimulator {

using chstabilizer_t = StabilizerState;

const double T_ANGLE = M_PI/4.;
const double TDG_ANGLE = -1.*M_PI/4.;

const U1Sample t_sample(T_ANGLE);
const U1Sample tdg_sample(TDG_ANGLE);

const uint_t ZERO = 0ULL;

thread_local std::unordered_map<double, U1Sample> Z_ROTATIONS;

class Runner
{
private:
  uint_t n_qubits_;
  uint_t num_states_;
  std::vector<chstabilizer_t> states_;
  std::vector<complex_t> coefficients_;
  uint_t num_threads_;
  uint_t omp_threshold_;

  bool accept_;
  complex_t old_ampsum_;
  uint_t x_string_;
  uint_t last_proposal_;

  void init_metropolis(AER::RngEngine &rng);
  void metropolis_step(AER::RngEngine &rng);
  //

  json_t serialize_state(uint_t rank) const;

public:
  Runner(): n_qubits_(0), num_states_(0) {};
  Runner(uint_t n_qubits);
  virtual ~Runner() = default;

  void initialize(uint_t n_qubits);
  void initialize_omp(uint_t n_threads, uint_t threshold_rank);

  bool empty() const
  {
    return (n_qubits_ == 0 || num_states_ == 0);
  }

  uint_t get_num_states() const;
  uint_t get_n_qubits() const;
  bool check_omp_threshold();

  //Convert each state to a json object and return it.
  std::vector<std::string> serialize_decomposition() const;

  //Creates num_states_ copies of the 'base state' in the runner
  //This will be either the |0>^n state, or a stabilizer state
  //produced by applying the first m Clifford gates of the
  //circuit. 
  void initialize_decomposition(uint_t n_states);
  //Check if the coefficient omega is 0
  bool check_eps(uint_t rank);

  //Methods for applying gates
  void apply_cx(uint_t control, uint_t target, uint_t rank);
  void apply_cz(uint_t control, uint_t target, uint_t rank);
  void apply_swap(uint_t qubit_1, uint_t qubit_2, uint_t rank);
  void apply_h(uint_t qubit, uint_t rank);
  void apply_s(uint_t qubit, uint_t rank);
  void apply_sdag(uint_t qubit, uint_t rank);
  void apply_x(uint_t qubit, uint_t rank);
  void apply_y(uint_t qubit, uint_t rank);
  void apply_z(uint_t qubit, uint_t rank);
  //Methods for non-clifford gates, including parameters for the Sampling
  void apply_t(uint_t qubit, double r, int rank);
  void apply_tdag(uint_t qubit, double r, int rank);
  void apply_u1(uint_t qubit, complex_t lambda, double r, int rank);
  void apply_ccx(uint_t control_1, uint_t control_2, uint_t target, uint_t branch, int rank);
  void apply_ccz(uint_t control_1, uint_t control_2, uint_t target, uint_t branch, int rank);
  //Measure a Pauli projector on each term in the decomposition and update their coefficients
  // omega.
  void apply_pauli_projector(const std::vector<pauli_t> &generators);
  void apply_pauli_projector(const std::vector<pauli_t> &generators, uint_t rank);
  //Routine for Norm Estimation, thin wrapper for the CHSimulator method that uses AER::RngEngine
  //to set up the estimation routine.
  double norm_estimation(uint_t n_samples, AER::RngEngine &rng);
  double norm_estimation(uint_t n_samples, std::vector<pauli_t> generators, AER::RngEngine &rng);

  //Metropolis Estimation for sampling from the output distribution
  uint_t metropolis_estimation(uint_t n_steps, AER::RngEngine &rng);
  std::vector<uint_t> metropolis_estimation(uint_t n_steps, uint_t n_shots, AER::RngEngine &rng);
  //Efficient Sampler for the output distribution of a stabilizer state
  uint_t stabilizer_sampler(AER::RngEngine &rng);
  std::vector<uint_t> stabilizer_sampler(uint_t n_shots, AER::RngEngine &rng);
  //Utilities for the state-vector snapshot.
  complex_t amplitude(uint_t x_measure);
  void state_vector(std::vector<complex_t> &svector, AER::RngEngine &rng);

};

//=========================================================================
// Implementation
//=========================================================================

Runner::Runner(uint_t num_qubits)
{
  initialize(num_qubits);
}

void Runner::initialize(uint_t num_qubits)
{
  states_.clear();
  coefficients_.clear();
  n_qubits_ = num_qubits;
  num_states_ = 1;
  num_threads_ = 1;
  states_ = std::vector<chstabilizer_t>(1, chstabilizer_t(num_qubits));
  coefficients_.push_back(complex_t(1.,0.));
}

void Runner::initialize_decomposition(uint_t n_states)
{
  num_states_ = n_states;
  states_.reserve(num_states_);
  coefficients_.reserve(num_states_);
  if(states_.size() > 1 || coefficients_.size() > 1)
  {
    throw std::runtime_error(std::string("CHSimulator::Runner was initialized without") + 
                             std::string("being properly cleared since the last ") +
                             std::string("experiment."));
  }
  chstabilizer_t base_sate(states_[0]);
  complex_t coeff(coefficients_[0]);
  for(uint_t i=1; i<num_states_; i++)
  {
    states_.push_back(base_sate);
    coefficients_.push_back(coeff);
  }
}

void Runner::initialize_omp(uint_t n_threads, uint_t threshold_rank)
{
  num_threads_ = (n_threads == 0 ? 1: n_threads);
  omp_threshold_ = threshold_rank;
}

uint_t Runner::get_num_states() const
{
  return num_states_;
}

uint_t Runner::get_n_qubits() const
{
  return n_qubits_;
}

bool Runner::check_omp_threshold()
{
  return num_states_ > omp_threshold_;
}

//-------------------------------------------------------------------------
// Operations on the decomposition
//-------------------------------------------------------------------------

void Runner::apply_pauli_projector(const std::vector<pauli_t> &generators)
{
  const int_t END = num_states_;
  #pragma omp parallel for if(num_states_ > omp_threshold_ && num_threads_ > 1) num_threads(num_threads_)
  for(int_t i=0; i<END; i++)
  {
    apply_pauli_projector(generators, i);
  }
}

void Runner::apply_pauli_projector(const std::vector<pauli_t> &generators, uint_t rank)
{
  states_[rank].MeasurePauliProjector(generators);
}

bool Runner::check_eps(uint_t rank)
{
  return (states_[rank].Omega().eps == 1);
}

void Runner::apply_cx(uint_t control, uint_t target, uint_t rank)
{
  states_[rank].CX(control, target);
}

void Runner::apply_cz(uint_t control, uint_t target, uint_t rank)
{
  states_[rank].CZ(control, target);
}

void Runner::apply_swap(uint_t qubit_1, uint_t qubit_2, uint_t rank)
{
  states_[rank].CX(qubit_1, qubit_2);
  states_[rank].CX(qubit_2, qubit_1);
  states_[rank].CX(qubit_1, qubit_2);
}


void Runner::apply_h(uint_t qubit, uint_t rank)
{
  states_[rank].H(qubit);
}

void Runner::apply_s(uint_t qubit, uint_t rank)
{
  states_[rank].S(qubit);
}

void Runner::apply_sdag(uint_t qubit, uint_t rank)
{
  states_[rank].Sdag(qubit);
}

void Runner::apply_x(uint_t qubit, uint_t rank)
{
  states_[rank].X(qubit);
}

void Runner::apply_y(uint_t qubit, uint_t rank)
{
  states_[rank].Y(qubit);
}

void Runner::apply_z(uint_t qubit, uint_t rank)
{
  states_[rank].Z(qubit);
}

void Runner::apply_t(uint_t qubit, double r, int rank)
{
  sample_branch_t branch = t_sample.sample(r);
  coefficients_[rank] *= branch.first;
  if (branch.second == Gates::s)
  {
    states_[rank].S(qubit);
  }
}
void Runner::apply_tdag(uint_t qubit, double r, int rank)
{
  sample_branch_t branch = tdg_sample.sample(r);
  coefficients_[rank] *= branch.first;
  if (branch.second == Gates::sdg)
  {
    states_[rank].Sdag(qubit);
  }
}

void Runner::apply_u1(uint_t qubit, complex_t param, double r, int rank)
{
  double lambda = std::real(param);
  auto it = Z_ROTATIONS.find(lambda); //Look for cached z_rotations
  sample_branch_t branch;
  if (it == Z_ROTATIONS.end())
  {
    U1Sample rotation(lambda);
    Z_ROTATIONS.insert({lambda, rotation});
    branch = rotation.sample(r);
  }
  else
  {
    branch = it->second.sample(r);
  }
  coefficients_[rank] *= branch.first;
  switch(branch.second)
  {
    case Gates::s:
      states_[rank].S(qubit);
      break;
    case Gates::sdg:
      states_[rank].Sdag(qubit);
      break;
    case Gates::z:
      states_[rank].Z(qubit);
      break;
    default:
      break;
  }
}

void Runner::apply_ccx(uint_t control_1, uint_t control_2, uint_t target, uint_t branch, int rank)
{
  switch(branch) //Decomposition of the CCX gate into Cliffords
  {
    case 1:
      states_[rank].CZ(control_1, control_2);
      break;
    case 2:
      states_[rank].CX(control_1, target);
      break;
    case 3:
      states_[rank].CX(control_2, target);
      break;
    case 4:
      states_[rank].CZ(control_1, control_2);
      states_[rank].CX(control_1, target);
      states_[rank].Z(control_1);
      break;
    case 5:
      states_[rank].CZ(control_1, control_2);
      states_[rank].CX(control_2, target);
      states_[rank].Z(control_2);
      break;
    case 6:
      states_[rank].CX(control_1, target);
      states_[rank].CX(control_2, target);
      states_[rank].X(target);
      break;
    case 7:
      states_[rank].CZ(control_1, control_2);
      states_[rank].CX(control_1, target);
      states_[rank].CX(control_2, target);
      states_[rank].Z(control_1);
      states_[rank].Z(control_2);
      states_[rank].X(target);
      coefficients_[rank] *= -1; //Additional phase
      break;
    default: //Identity
      break;
  }
}

void Runner::apply_ccz(uint_t control_1, uint_t control_2, uint_t target, uint_t branch, int rank)
{
  switch(branch) //Decomposition of the CCZ gate into Cliffords
  {
    case 1:
      states_[rank].CZ(control_1, control_2);
      break;
    case 2:
      states_[rank].CZ(control_1, target);
      break;
    case 3:
      states_[rank].CZ(control_2, target);
      break;
    case 4:
      states_[rank].CZ(control_1, control_2);
      states_[rank].CZ(control_1, target);
      states_[rank].Z(control_1);
      break;
    case 5:
      states_[rank].CZ(control_1, control_2);
      states_[rank].CZ(control_2, target);
      states_[rank].Z(control_2);
      break;
    case 6:
      states_[rank].CZ(control_1, target);
      states_[rank].CZ(control_2, target);
      states_[rank].Z(target);
      break;
    case 7:
      states_[rank].CZ(control_1, control_2);
      states_[rank].CZ(control_1, target);
      states_[rank].CZ(control_2, target);
      states_[rank].Z(control_1);
      states_[rank].Z(control_2);
      states_[rank].Z(target);
      coefficients_[rank] *= -1; //Additional phase
      break;
    default: //Identity
      break;
  }
}

//-------------------------------------------------------------------------
//Measurement
//-------------------------------------------------------------------------

double Runner::norm_estimation(uint_t n_samples, AER::RngEngine &rng)
{
  std::vector<uint_t> adiag_1(n_samples, 0ULL);
  std::vector<uint_t> adiag_2(n_samples, 0ULL);
  std::vector< std::vector<uint_t> > a(n_samples, std::vector<uint_t>(n_qubits_, 0ULL));
  const int_t NSAMPLES = n_samples;
  const int_t NQUBITS = n_qubits_;
  #pragma omp parallel if (num_threads_ > 1) num_threads(num_threads_)
  {
  #ifdef _WIN32
    #pragma omp for
  #else
    #pragma omp for collapse(2)
  #endif
  for (int_t l=0; l<NSAMPLES; l++)
  {
    for (int_t i=0; i<NQUBITS; i++)
    {
      for (int_t j=i; j<NQUBITS; j++)
      {
          if(rng.rand() < 0.5)
          {
              a[l][i] |= (1ULL << j);
              a[l][j] |= (1ULL << i);
          }
      }
      adiag_1[l] |= (a[l][i] & (1ULL << i));
      if (rng.rand() < 0.5)
      {
          adiag_2[l] |= (1ULL << i);
      }
    }
  }
  } // end omp parallel
  return ParallelNormEstimate(states_, coefficients_, adiag_1, adiag_2, a, num_threads_);
}

double Runner::norm_estimation(uint_t n_samples, std::vector<pauli_t> generators, AER::RngEngine &rng)
{
  apply_pauli_projector(generators);
  return norm_estimation(n_samples, rng);
}

uint_t Runner::metropolis_estimation(uint_t n_steps, AER::RngEngine &rng)
{
  init_metropolis(rng);
  for (uint_t i=0; i<n_steps; i++)
  {
    metropolis_step(rng);
  }
  return x_string_;
}

std::vector<uint_t> Runner::metropolis_estimation(uint_t n_steps, uint_t n_shots, AER::RngEngine &rng)
{
  std::vector<uint_t> shots(n_shots, zer);
  shots[0] = metropolis_estimation(n_steps, rng);
  for (uint_t i=1; i<n_shots; i++)
  {
    metropolis_step(rng);
    shots[i] = x_string_;
  }
  return shots;
}

void Runner::init_metropolis(AER::RngEngine &rng)
{
  accept_ = 0;
  //Random initial x_string from RngEngine
  uint_t max = (1ULL<<n_qubits_) - 1;
  x_string_ = rng.rand_int(ZERO, max);
  last_proposal_=0;
  double local_real=0., local_imag=0.;
  const int_t END = num_states_;
  #pragma omp parallel for if(num_states_ > omp_threshold_ && num_threads_ > 1) num_threads(num_threads_) reduction(+:local_real) reduction(+:local_imag)
  for (int_t i=0; i<END; i++)
  {
    scalar_t amp = states_[i].Amplitude(x_string_);
    if(amp.eps == 1)
    {
      complex_t local = (amp.to_complex() * coefficients_[i]);
      local_real += local.real();
      local_imag += local.imag();
    }
  }
  old_ampsum_ = complex_t(local_real, local_imag);
}

void Runner::metropolis_step(AER::RngEngine &rng)
{
  uint_t proposal = rng.rand(0ULL, n_qubits_);
  if(accept_)
  {
    x_string_ ^= (one << last_proposal_);
  }
  double real_part = 0.,imag_part =0.;
  if (accept_ == 0)
  {
    const int_t END = num_states_;
    #pragma omp parallel for if(num_states_ > omp_threshold_ && num_threads_ > 1) num_threads(num_threads_) reduction(+:real_part) reduction(+:imag_part)
    for (int_t i=0; i<END; i++)
    {
      scalar_t amp = states_[i].ProposeFlip(proposal);
      if(amp.eps == 1)
      {
        complex_t local = (amp.to_complex() * coefficients_[i]);
        real_part += local.real();
        imag_part += local.imag();
      }
    }
  }
  else
  {
    const int_t END = num_states_;
    #pragma omp parallel for if(num_states_ > omp_threshold_ && num_threads_ > 1) num_threads(num_threads_) reduction(+:real_part) reduction(+:imag_part)
    for (int_t i=0; i<END; i++)
    {
      states_[i].AcceptFlip();
      scalar_t amp = states_[i].ProposeFlip(proposal);
      if(amp.eps == 1)
      {
        complex_t local = (amp.to_complex() * coefficients_[i]);
        real_part += local.real();
        imag_part += local.imag();
      }
    }
  }
  complex_t ampsum(real_part, imag_part);
  double p_threshold = std::norm(ampsum)/std::norm(old_ampsum_);
  #ifdef  __FAST_MATH__ //isnan doesn't behave well under fastmath, so use absolute tolerance check instead
  if(std::isinf(p_threshold) || std::abs(std::norm(old_ampsum_)-0.) < 1e-8)
  #else
  if(std::isinf(p_threshold) || std::isnan(p_threshold))
  #endif
  {
    accept_ = 1;
    old_ampsum_ = ampsum;
    last_proposal_ = proposal; //We try to move away from node with 0 probability.
  }
  else
  {
    double rand = rng.rand();
    if (rand < p_threshold)
    {
      accept_ = 1;
      old_ampsum_ = ampsum;
      last_proposal_ = proposal;
    }
    else
    {
      accept_ = 0;
    }
  }
}

uint_t Runner::stabilizer_sampler(AER::RngEngine &rng)
{
  uint_t max = (1ULL << n_qubits_) -1;
  return states_[0].Sample(rng.rand_int(ZERO, max));
}

std::vector<uint_t> Runner::stabilizer_sampler(uint_t n_shots, AER::RngEngine &rng)
{
  if(num_states_ > 1)
  {
    throw std::invalid_argument("CH::Runner::stabilizer_sampler: This method can only be used for a single Stabilizer state.\n");
  }
  std::vector<uint_t> shots;
  shots.reserve(n_shots);
  for(uint_t i=0; i<n_shots; i++)
  {
    shots.push_back(stabilizer_sampler(rng));
  }
  return shots;
}

complex_t Runner::amplitude(uint_t x_measure)
{
  double real_part=0., imag_part=0.;
  //Splitting the reduction guarantees support on more OMP versions.
  const int_t END = num_states_;
  #pragma omp parallel for if(num_states_ > omp_threshold_ && num_threads_ > 1) num_threads(num_threads_) reduction(+:real_part) reduction(+:imag_part)
  for(int_t i=0; i<END; i++)
  {
    complex_t amplitude = states_[i].Amplitude(x_measure).to_complex();
    amplitude *= coefficients_[i];
    real_part += amplitude.real();
    imag_part += amplitude.imag();
  }
  return complex_t(real_part, imag_part);
}

void Runner::state_vector(std::vector<complex_t> &svector, AER::RngEngine &rng)
{
  uint_t ceil = 1ULL << n_qubits_;
  if (!svector.empty())
  {
    svector.clear();
  }
  svector.reserve(ceil);
  // double norm = 1;
  double norm = 1;
  if(num_states_ > 1)
  {
    norm = norm_estimation(40, rng);
  }
  for(uint_t i=0; i<ceil; i++)
  {
    svector.push_back(amplitude(i)/std::sqrt(norm));
  }
}

//=========================================================================
// Utility
//=========================================================================

inline void to_json(json_t &js, const Runner &rn)
{
  js["num_qubits"] = rn.get_n_qubits();
  js["num_states"] = rn.get_num_states();
  js["decomposition"] = rn.serialize_decomposition();
}

std::vector<std::string> Runner::serialize_decomposition() const
{
  std::vector<std::string> serialized_states(num_states_);
  const int_t END = num_states_;
  #pragma omp parallel for if(num_threads_ > 1 && num_states_ > omp_threshold_) num_threads(num_threads_)
  for(int_t i=0; i<END; i++)
  {
    serialized_states[i] = serialize_state(i).dump();
  }
  return serialized_states;
}

json_t Runner::serialize_state(uint_t rank) const
{
  json_t js = json_t::object();
  std::vector<unsigned> gamma;
  std::vector<uint_t> M;
  std::vector<uint_t> F;
  std::vector<uint_t> G;
  gamma.reserve(n_qubits_);
  M = states_[rank].MMatrix();
  F = states_[rank].FMatrix();
  G = states_[rank].GMatrix();
  uint_t gamma1 = states_[rank].Gamma1();
  uint_t gamma2 = states_[rank].Gamma2();
  for(uint_t i=0; i<n_qubits_; i++)
  {
    gamma.push_back(((gamma1 >> i) & 1ULL) + 2*((gamma2 >> i) & 1ULL));
  }
  js["gamma"] = gamma;
  js["M"] = M;
  js["F"] = F;
  js["G"] = G;
  js["internal_cofficient"] = states_[rank].Omega().to_complex();
  js["coefficient"] = coefficients_[rank];
  return js;
}

} // Close namespace CHSimulator
#endif
