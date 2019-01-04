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
  
using complex_t = std::complex<double>;
using uint_t = uint_fast64_t;
using int_t = int_fast64_t;
using chstabilizer_t = StabilizerState;

double t_angle = M_PI/4.;
double tdg_angle = -1.*M_PI/4.;

uint_t zero = 0ULL;

const U1Sample t_sample(t_angle);
const U1Sample tdg_sample(tdg_angle);

static std::unordered_map<double, U1Sample> z_rotations;

class Runner
{
private:
  uint_t n_qubits;
  uint_t chi;
  uint_t three_qubit_gate_count; //Needed for normalising states
  std::vector<chstabilizer_t> states;
  std::vector<complex_t> coefficients;
  uint_t n_omp_threads;
  uint_t omp_threshold;

  bool accept;
  complex_t old_ampsum;
  uint_t x_string;
  uint_t last_proposal;


  void InitMetropolis(AER::RngEngine &rng);
  void MetropolisStep(AER::RngEngine &rng);
  //

  json_t serialise_state(uint_t rank) const;

public:
  Runner() = default;
  Runner(uint_t n_qubits);
  virtual ~Runner() = default;

  void initialize(uint_t n_qubits);
  void initialize_omp(uint_t n_threads, uint_t threshold_rank);

  uint_t get_chi() const;
  uint_t get_n_qubits() const;
  bool check_omp_threshold();

  //Convert each state to a json object and return it.
  std::vector<std::string> serialised_decomposition() const;

  //Creates chi copies of the 'base state' in the runner
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
  inline void apply_pauli_projector(const std::vector<pauli_t> &generators, uint_t rank);
  //Routine for Norm Estimation, thin wrapper for the CHSimulator method that uses AER::RngEngine
  //to set up the estimation routine.
  double NormEstimation(uint_t n_samples, AER::RngEngine &rng);
  double NormEstimation(uint_t n_samples, std::vector<pauli_t> generators, AER::RngEngine &rng);

  //Metropolis Estimation for sampling from the output distribution
  uint_t MetropolisEstimation(uint_t n_steps, AER::RngEngine &rng);
  std::vector<uint_t> MetropolisEstimation(uint_t n_steps, uint_t n_shots, AER::RngEngine &rng);
  //Efficient Sampler for the output distribution of a stabilizer state
  uint_t StabilizerSampler(AER::RngEngine &rng);
  std::vector<uint_t> StabilizerSampler(uint_t n_shots, AER::RngEngine &rng);
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
  if (states.size() > 0)
  {
    states.clear();
  }
  if (coefficients.size() > 0)
  {
    coefficients.clear();
  }
  n_qubits = num_qubits;
  chi = 1;
  n_omp_threads = 1;
  states = std::vector<chstabilizer_t>(1, chstabilizer_t(num_qubits));
  coefficients.push_back(complex_t(1.,0.));
}

void Runner::initialize_decomposition(uint_t n_states)
{
  chi = n_states;
  chstabilizer_t base_sate = states[0];
  complex_t coeff = coefficients[0];
  states = std::vector<chstabilizer_t>(chi, base_sate);
  coefficients = std::vector<complex_t>(chi, coeff);
}

void Runner::initialize_omp(uint_t n_threads, uint_t threshold_rank)
{
  if(n_threads == 0)
  {
    n_omp_threads = 1;
  }
  else
  {
    n_omp_threads = n_threads;
  }
  omp_threshold = threshold_rank;
}

uint_t Runner::get_chi() const
{
  return chi;
}

uint_t Runner::get_n_qubits() const
{
  return n_qubits;
}

bool Runner::check_omp_threshold()
{
  return chi > omp_threshold;
}

//-------------------------------------------------------------------------
// Operations on the decomposition
//-------------------------------------------------------------------------

void Runner::apply_pauli_projector(const std::vector<pauli_t> &generators)
{
  #pragma omp parallel for if(chi > omp_threshold && n_omp_threads > 1) num_threads(n_omp_threads)
  for(uint_t i=0; i<chi; i++)
  {
    apply_pauli_projector(generators, i);
  }
}

void Runner::apply_pauli_projector(const std::vector<pauli_t> &generators, uint_t rank)
{
  states[rank].MeasurePauliProjector(generators);
}

bool Runner::check_eps(uint_t rank)
{
  return (states[rank].Omega().eps == 1);
}

void Runner::apply_cx(uint_t control, uint_t target, uint_t rank)
{
  states[rank].CX(control, target);
}

void Runner::apply_cz(uint_t control, uint_t target, uint_t rank)
{
  states[rank].CZ(control, target);
}

void Runner::apply_swap(uint_t qubit_1, uint_t qubit_2, uint_t rank)
{
  states[rank].CX(qubit_1, qubit_2);
  states[rank].CX(qubit_2, qubit_1);
  states[rank].CX(qubit_1, qubit_2);
}


void Runner::apply_h(uint_t qubit, uint_t rank)
{
  states[rank].H(qubit);
}

void Runner::apply_s(uint_t qubit, uint_t rank)
{
  states[rank].S(qubit);
}

void Runner::apply_sdag(uint_t qubit, uint_t rank)
{
  states[rank].Sdag(qubit);
}

void Runner::apply_x(uint_t qubit, uint_t rank)
{
  states[rank].X(qubit);
}

void Runner::apply_y(uint_t qubit, uint_t rank)
{
  states[rank].Y(qubit);
}

void Runner::apply_z(uint_t qubit, uint_t rank)
{
  states[rank].Z(qubit);
}

void Runner::apply_t(uint_t qubit, double r, int rank)
{
  sample_branch_t branch = t_sample.sample(r);
  coefficients[rank] *= branch.first;
  if (branch.second == Gates::s)
  {
    states[rank].S(qubit);
  }
}
void Runner::apply_tdag(uint_t qubit, double r, int rank)
{
  sample_branch_t branch = tdg_sample.sample(r);
  coefficients[rank] *= branch.first;
  if (branch.second == Gates::s)
  {
    states[rank].S(qubit);
  }
}

void Runner::apply_u1(uint_t qubit, complex_t param, double r, int rank)
{
  double lambda = std::real(param);
  auto it = z_rotations.find(lambda); //Look for cached z_rotations
  sample_branch_t branch;
  if (it == z_rotations.end())
  {
    U1Sample rotation(lambda);
    z_rotations.insert({lambda, rotation});
    branch = rotation.sample(r);
  }
  else
  {
    branch = it->second.sample(r);
  }
  coefficients[rank] *= branch.first;
  switch(branch.second)
  {
    case Gates::s:
      states[rank].S(qubit);
      break;
    case Gates::sdg:
      states[rank].Sdag(qubit);
      break;
    case Gates::z:
      states[rank].Z(qubit);
      break;
    default:
      break;
  }
}

void Runner::apply_ccx(uint_t control_1, uint_t control_2, uint_t target, uint_t branch, int rank)
{
  coefficients[rank] *= ccx_coeff;
  switch(branch) //Decomposition of the CCZ gate into Cliffords
  {
    case 1:
      states[rank].CZ(control_1, control_2);
      break;
    case 2:
      states[rank].CX(control_1, target);
      break;
    case 3:
      states[rank].CZ(control_1, control_2);
      states[rank].CX(control_1, target);
      states[rank].Z(control_1);
      break;
    case 4:
      states[rank].CX(control_2, target);
      break;
    case 5:
      states[rank].CX(control_2, target);
      states[rank].CZ(control_1, control_2);
      states[rank].Z(control_2);
      break;
    case 6:
      states[rank].CX(control_1, target);
      states[rank].CX(control_2, target);
      states[rank].X(target);
      break;
    case 7:
      states[rank].CZ(control_1, control_2);
      states[rank].CX(control_1, target);
      states[rank].CX(control_2, target);
      states[rank].Z(control_1);
      states[rank].Z(control_2);
      states[rank].X(target);
      coefficients[rank] *= -1; //Additional phase
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
      states[rank].CZ(control_1, control_2);
      break;
    case 2:
      states[rank].CZ(control_1, target);
      break;
    case 3:
      states[rank].CZ(control_1, control_2);
      states[rank].CZ(control_1, target);
      states[rank].Z(control_1);
      break;
    case 4:
      states[rank].CZ(control_2, target);
      break;
    case 5:
      states[rank].CZ(control_2, target);
      states[rank].CZ(control_1, control_2);
      states[rank].Z(control_2);
      break;
    case 6:
      states[rank].CZ(control_1, target);
      states[rank].CZ(control_2, target);
      states[rank].Z(target);
      break;
    case 7:
      states[rank].CZ(control_1, control_2);
      states[rank].CZ(control_1, target);
      states[rank].CZ(control_2, target);
      states[rank].Z(control_1);
      states[rank].Z(control_2);
      states[rank].Z(target);
      coefficients[rank] *= -1; //Additional phase
      break;
    default: //Identity
      break;
  }
}

//-------------------------------------------------------------------------
//Measurement
//-------------------------------------------------------------------------

double Runner::NormEstimation(uint_t n_samples, AER::RngEngine &rng)
{
  std::vector<uint_t> adiag_1(n_samples, 0ULL);
  std::vector<uint_t> adiag_2(n_samples, 0ULL);
  std::vector< std::vector<uint_t> > a(n_samples, std::vector<uint_t>(n_qubits, 0ULL));
  #pragma omp parallel for if(n_omp_threads > 1)
  for (size_t l=0; l<n_samples; l++)
  {
    for (size_t i=0; i<n_qubits; i++)
    {
      for (size_t j=i; j<n_qubits; j++)
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
  return ParallelNormEstimate(states, coefficients, adiag_1, adiag_2, a, n_omp_threads);
}

double Runner::NormEstimation(uint_t n_samples, std::vector<pauli_t> generators, AER::RngEngine &rng)
{
  apply_pauli_projector(generators);
  return NormEstimation(n_samples, rng);
}

uint_t Runner::MetropolisEstimation(uint_t n_steps, AER::RngEngine &rng)
{
  InitMetropolis(rng);
  for (uint_t i=0; i<n_steps; i++)
  {
    MetropolisStep(rng);
  }
  return x_string;
}

std::vector<uint_t> Runner::MetropolisEstimation(uint_t n_steps, uint_t n_shots, AER::RngEngine &rng)
{
  std::vector<uint_t> shots(n_shots, zer);
  shots[0] = MetropolisEstimation(n_steps, rng);
  for (uint_t i=1; i<n_shots; i++)
  {
    MetropolisStep(rng);
    shots[i] = x_string;
  }
  return shots;
}

void Runner::InitMetropolis(AER::RngEngine &rng)
{
  accept = 0;
  //Random initial x_string from RngEngine
  uint_t max = (1ULL<<n_qubits) - 1;
  x_string = rng.rand_int(zero, max);
  last_proposal=0;
  // for (uint_t i=0; i<n_qubits; i++)
  // {
  //   if (rng.rand() < 0.5)
  //   {
  //     x_string ^= one << i;
  //   }
  // }
  double local_real=0., local_imag=0.;
  #pragma omp parallel for if(chi > omp_threshold && n_omp_threads > 1) num_threads(n_omp_threads) reduction(+:local_real) reduction(+:local_imag)
  for (uint_t i=0; i<chi; i++)
  {
    scalar_t amp = states[i].Amplitude(x_string);
    if(amp.eps == 1)
    {
      complex_t local = (amp.to_complex() * coefficients[i]);
      local_real += local.real();
      local_imag += local.imag();
    }
  }
  old_ampsum = complex_t(local_real, local_imag);
}

void Runner::MetropolisStep(AER::RngEngine &rng)
{
  uint_t proposal = rng.rand(0ULL, n_qubits);
  if(accept)
  {
    x_string ^= (one << last_proposal);
  }
  double real_part = 0.,imag_part =0.;
  if (accept == 0)
  {
    #pragma omp parallel for if(chi > omp_threshold && n_omp_threads > 1) num_threads(n_omp_threads) reduction(+:real_part) reduction(+:imag_part)
    for (uint_t i=0; i<chi; i++)
    {
      scalar_t amp = states[i].ProposeFlip(proposal);
      if(amp.eps == 1)
      {
        complex_t local = (amp.to_complex() * coefficients[i]);
        real_part += local.real();
        imag_part += local.imag();
      }
    }
  }
  else
  {
    #pragma omp parallel for if(chi > omp_threshold && n_omp_threads > 1) num_threads(n_omp_threads) reduction(+:real_part) reduction(+:imag_part)
    for (uint_t i=0; i<chi; i++)
    {
      states[i].AcceptFlip();
      scalar_t amp = states[i].ProposeFlip(proposal);
      if(amp.eps == 1)
      {
        complex_t local = (amp.to_complex() * coefficients[i]);
        real_part += local.real();
        imag_part += local.imag();
      }
    }
  }
  complex_t ampsum(real_part, imag_part);
  double p_threshold = std::norm(ampsum)/std::norm(old_ampsum);  
  double rand = rng.rand();
  if (rand < p_threshold)
  {
    accept = 1;
    old_ampsum = ampsum;
    last_proposal = proposal;
  }
  else
  {
    accept = 0;
  }
}

uint_t Runner::StabilizerSampler(AER::RngEngine &rng)
{
  uint max = (1ULL << n_qubits) -1;
  return states[0].Sample(rng.rand_int(zero, max));
}

std::vector<uint_t> Runner::StabilizerSampler(uint_t n_shots, AER::RngEngine &rng)
{
  if(chi > 1)
  {
    throw std::invalid_argument("CH::Runner::StabilizerSampler: This method can only be used for a single Stabilizer state.\n");
  }
  std::vector<uint_t> shots;
  shots.reserve(n_shots);
  for(uint_t i=0; i<n_shots; i++)
  {
    shots.push_back(StabilizerSampler(rng));
  }
  return shots;
}

complex_t Runner::amplitude(uint_t x_measure)
{
  double real_part=0., imag_part=0.;
  //Splitting the reduction guarantees support on more OMP versions.
  #pragma omp parallel for if(chi > omp_threshold && n_omp_threads > 1) num_threads(n_omp_threads) reduction(+:real_part) reduction(+:imag_part)
  for(uint_t i=0; i<chi; i++)
  {
    complex_t amplitude = states[i].Amplitude(x_measure).to_complex();
    amplitude *= coefficients[i];
    real_part += amplitude.real();
    imag_part += amplitude.imag();
  }
  return complex_t(real_part, imag_part);
}

void Runner::state_vector(std::vector<complex_t> &svector, AER::RngEngine &rng)
{
  uint_t ceil = 1ULL << n_qubits;
  if (!svector.empty())
  {
    svector.clear();
  }
  svector.reserve(ceil);
  // double norm = 1;
  double norm = 1;
  if(chi > 1)
  {
    norm = NormEstimation(40, rng);
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
  js["chi"] = rn.get_chi();
  js["decomposition"] = rn.serialised_decomposition();
}

std::vector<std::string> Runner::serialised_decomposition() const
{
  std::vector<std::string> serialised_states(chi);
  #pragma omp parallel for if(n_omp_threads > 1 && chi > omp_threshold) num_threads(n_omp_threads)
  for(uint_t i=0; i<chi; i++)
  {
    serialised_states[i] = serialise_state(i);
  }
  return serialised_states;
}

json_t Runner::serialise_state(uint_t rank) const
{
  json_t js = json_t::object();
  std::vector<unsigned> gamma;
  std::vector<uint_t> M;
  std::vector<uint_t> F;
  std::vector<uint_t> G;
  gamma.reserve(n_qubits);
  M = states[rank].MMatrix();
  F = states[rank].FMatrix();
  G = states[rank].GMatrix();
  uint_t gamma1 = states[rank].Gamma1();
  uint_t gamma2 = states[rank].Gamma2();
  for(uint_t i=0; i<n_qubits; i++)
  {
    gamma.push_back(((gamma1 >> i) & 1ULL) + 2*((gamma2 >> i) & 1ULL));
  }
  js["gamma"] = gamma;
  js["M"] = M;
  js["F"] = F;
  js["G"] = G;
  js["internal_cofficient"] = states[rank].Omega().to_complex();
  js["coefficient"] = coefficients[rank];
  return js;
}

} // Close namespace CHSimulator
#endif
