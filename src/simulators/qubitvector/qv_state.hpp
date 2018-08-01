/**
 * Copyright 2017, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    qubitvector_state.hpp
 * @brief   QubitVector Simulator State
 * @authors Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _qubitvector_qv_state_hpp
#define _qubitvector_qv_state_hpp

#include <algorithm>
#include <array>
#include <complex>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "base/state.hpp"
#include "simulators/qubitvector/qubitvector.hpp"

//-----------------------------------------------------------------------
// JSON serialization for QubitVector class
//-----------------------------------------------------------------------
namespace QV {
  inline void to_json(json_t &js, const QubitVector&qv) {
    to_json(js, qv.vector());
  }

  inline void from_json(const json_t &js, QubitVector&qv) {
    cvector_t tmp;
    from_json(js, tmp);
    qv = tmp;
  }
}

namespace AER {
namespace QubitVector {
  
  using state_t = QV::QubitVector;
  using BaseState = Base::State<state_t>;


/*******************************************************************************
 *
 * QubitVector State class
 *
 ******************************************************************************/

class State : public BaseState {

public:

  State() = default;
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Allowed operations are:
  // {"barrier", "measure", "reset", "mat", "dmat", "kraus",
  //  "u0", "u1", "u2", "u3", "cx", "cz",
  //  "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"}
  virtual std::set<std::string> allowed_ops() const override;
    
  // Applies an operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  virtual void apply_op(const Op &op) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize(uint_t num_qubits) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual uint_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Op> &ops) override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 14}
  // TODO: Check optimal default value for a desktop i7 CPU
  virtual void load_config(const json_t &config) override;

  // Allows measurements
  bool has_measure = true;

  virtual reg_t apply_measure(const reg_t& qubits) override;

  virtual rvector_t measure_probs(const reg_t &qubits) const override;

  // Supports Pauli observables
  bool has_pauli_observables = true;

  // Return the complex expectation value for an observable operator
  virtual double pauli_observable_value(const reg_t& qubits, 
                                        const std::string &pauli) const override;

  /* TODO: // Supports Matrix observables
  bool has_matrix_observables = true;
  // Return the complex expectation value for an observable operator
  inline virtual complex_t matrix_observable_value(const Op &op) const {
    (void)op; return complex_t();
  }; */

protected:
  // Allowed operations are:
  // {"barrier", "measure", "reset", "mat", "dmat", "kraus",
  //  "u0", "u1", "u2", "u3", "cx", "cz",
  //  "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"}
  const std::set<std::string> allowed_ops_ = {
    "barrier", "measure", "reset",
    "mat", "dmat", "kraus",
    "u0", "u1", "u2", "u3", "cx", "cz",
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"
  }; 

  // Enum class and gateset map for switching based on gate name
  enum class Gates {
    mat, dmat, kraus, // special
    measure, reset, barrier,
    u0, u1, u2, u3, id, x, y, z, h, s, sdg, t, tdg, // single qubit
    cx, cz, rzz // two qubit
  };
  const static std::map<std::string, Gates> gateset;

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  int omp_qubit_threshold_ = 14; // Test this on desktop i7 to find best setting 

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  // Apply a diagonal matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & dmat);

  //-----------------------------------------------------------------------
  // Apply Kraus Noise
  //-----------------------------------------------------------------------

  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &krausops);

  //-----------------------------------------------------------------------
  // Measurement and Reset Helpers
  //-----------------------------------------------------------------------

  void apply_reset(const reg_t &qubits, const uint_t reset_state = 0);

  // Sample the measurement outcome for qubits
  // return the outcome, and its corresponding probabilities
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_outcome(const reg_t &qubits);


  void measure_reset_update(const std::vector<uint_t> &qubits,
                            const uint_t final_state,
                            const uint_t meas_state,
                            const double meas_prob);

  //-----------------------------------------------------------------------
  // 1-Qubit Gates
  //-----------------------------------------------------------------------
  
  void apply_gate_u3(const uint_t qubit, const double theta, const double phi,
                       const double lambda);

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);


  //-----------------------------------------------------------------------
  // Generate matrices
  //-----------------------------------------------------------------------

  cvector_t waltz_vectorized_matrix(double theta, double phi, double lambda);

  cvector_t rzz_diagonal_matrix(double lambda);

};


//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

std::set<std::string> State::allowed_ops() const {
  return { "barrier", "measure", "reset",
    "mat", "dmat", "kraus",
    "u0", "u1", "u2", "u3", "cx", "cz",
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"};
} 

const std::map<std::string, State::Gates> State::gateset({
  {"measure", Gates::measure}, // Measure is handled by engine
  {"reset", Gates::reset}, // Reset operation
  {"barrier", Gates::barrier}, // barrier does nothing
  // Matrix multiplication
  {"mat", Gates::mat},     // matrix multiplication
  {"dmat", Gates::dmat}, // Diagonal matrix multiplication
  // Single qubit gates
  {"id", Gates::id},   // Pauli-Identity gate
  {"x", Gates::x},    // Pauli-X gate
  {"y", Gates::y},    // Pauli-Y gate
  {"z", Gates::z},    // Pauli-Z gate
  {"s", Gates::z},    // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg}, // Conjugate-transpose of Phase gate
  {"h", Gates::h},    // Hadamard gate (X + Z / sqrt(2))
  {"t", Gates::t},    // T-gate (sqrt(S))
  {"tdg", Gates::tdg}, // Conjguate-transpose of T gate
  // Waltz Gates
  {"u0", Gates::u0},  // idle gate in multiples of X90
  {"u1", Gates::u1},  // zero-X90 pulse waltz gate
  {"u2", Gates::u2},  // single-X90 pulse waltz gate
  {"u3", Gates::u3},  // two X90 pulse waltz gate
  // Two-qubit gates
  {"cx", Gates::cx},  // Controlled-X gate (CNOT)
  {"cz", Gates::cz},  // Controlled-Z gate
  {"rzz", Gates::rzz}, // ZZ-rotation gate
  // Type-2 Noise
  {"kraus", Gates::kraus} // Kraus error
}); 


//============================================================================
// Implementation: Base class method overrides
//============================================================================

uint_t State::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Op> &ops) {
  // An n-qubit state vector as 2^n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  uint_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  uint_t mem_mb = 1ULL << shift_mb;
  return mem_mb;
}


void State::load_config(const json_t &config) {
  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "omp_qubit_threshold", config);
}


void State::initialize(uint_t num_qubits) {
  
  // reset state std::vector to default state
  data_ = state_t(num_qubits);

  // Set maximum threads for QubitVector parallelization
  data_.set_omp_threshold(omp_qubit_threshold_);
  if (threads_ > 0)
    data_.set_omp_threads(threads_); // set allowed OMP threads in qubitvector
  
  data_.initialize(); // initialize qubit vector to all |0> state
}


reg_t State::apply_measure(const reg_t &qubits) {
  // Actual measurement outcome
  const auto meas = sample_measure_outcome(qubits);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  return Utils::int2reg(meas.first, 2, qubits.size());
}


rvector_t State::measure_probs(const reg_t &qubits) const {
  if (qubits.size() == 1) {
    // Probability of P0 outcome
    double p0 = data_.probability(qubits[0], 0);
    return {{p0, 1. - p0}};
  } else
    return data_.probabilities(qubits);
}


double State::pauli_observable_value(const reg_t& qubits, 
                                     const std::string &pauli) const {
  // Copy the quantum state;
  state_t data_copy = data_;
  // Apply each pauli operator as a gate to the corresponding qubit
  for (size_t pos=0; pos < qubits.size(); ++pos) {
    switch (pauli[pos]) {
      case 'I':
        break;
      case 'X':
        data_copy.apply_x(qubits[pos]);
        break;
      case 'Y':
        data_copy.apply_y(qubits[pos]);
        break;
      case 'Z':
        data_copy.apply_z(qubits[pos]);
        break;
      default: {
        std::stringstream msg;
        msg << "QubitVectorState::invalid Pauli string \'" << pauli[pos] << "\'.";
        throw std::invalid_argument(msg.str());
      }
    }
  }
  // Compute the inner_product of data with the updated data_copy
  complex_t inprod = data_copy.inner_product(data_);
  return std::real(std::conj(inprod) * inprod);
}



void State::apply_op(const Op &op) {

  // Check Op is supported by State
  auto it = gateset.find(op.name);
  if (it == gateset.end()) {
    std::stringstream msg;
    msg << "QubitVectorState::invalid operation \'" << op.name << "\'.";
    throw std::invalid_argument(msg.str());
  }

  Gates g = it -> second;
  switch (g) {
  // Matrix multiplication
  case Gates::mat:
    apply_matrix(op.qubits, op.params_m[0]);
    break;
  case Gates::dmat:
    apply_matrix(op.qubits, op.params_z);
    break;
  // Special Noise operations
  case Gates::kraus:
    apply_kraus(op.qubits, op.params_m);
    break;
  // Base gates
  case Gates::u3:
    apply_gate_u3(op.qubits[0], op.params_d[0], op.params_d[1], op.params_d[2]);
    break;
  case Gates::u2:
    apply_gate_u3(op.qubits[0], M_PI / 2., op.params_d[0], op.params_d[1]);
    break;
  case Gates::u1:
    apply_gate_phase(op.qubits[0], std::exp(complex_t(0., op.params_d[0])));
    break;
  case Gates::cx:
    data_.apply_cnot(op.qubits[0], op.qubits[1]);
    break;
  case Gates::cz:
    data_.apply_cz(op.qubits[0], op.qubits[1]);
    break;
  case Gates::reset:
    apply_reset(op.qubits, 0);
    break;
  case Gates::barrier:
    break;
  // Waltz gates
  case Gates::u0: // u0 = id in ideal State
    break;
  // QIP gates
  case Gates::id:
    break;
  case Gates::x:
    data_.apply_x(op.qubits[0]);
    break;
  case Gates::y:
    data_.apply_y(op.qubits[0]);
    break;
  case Gates::z:
    data_.apply_z(op.qubits[0]);
    break;
  case Gates::h:
    apply_gate_u3(op.qubits[0], M_PI / 2., 0., M_PI);
    break;
  case Gates::s:
    apply_gate_phase(op.qubits[0], complex_t(0., 1.));
    break;
  case Gates::sdg:
    apply_gate_phase(op.qubits[0], complex_t(0., -1.));
    break;
  case Gates::t: {
    const double isqrt2{1. / std::sqrt(2)};
    apply_gate_phase(op.qubits[0], complex_t(isqrt2, isqrt2));
  } break;
  case Gates::tdg: {
    const double isqrt2{1. / std::sqrt(2)};
    apply_gate_phase(op.qubits[0], complex_t(isqrt2, -isqrt2));
  } break;
  // ZZ rotation by angle lambda
  case Gates::rzz: {
    const auto dmat = rzz_diagonal_matrix(op.params_d[0]);
    data_.apply_matrix({{op.qubits[0], op.qubits[1]}}, dmat);
  } break;
  // Invalid Gate (we shouldn't get here)
  default:
    std::stringstream msg;
    msg << "QubitVectorState::invalid State operation \'" << op.name << "\'.";
    throw std::invalid_argument(msg.str());
  }
}


//============================================================================
// Implementation: Matrix multiplication
//============================================================================

void State::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (qubits.empty() == false && mat.size() > 0)
    data_.apply_matrix(qubits, Utils::vectorize_matrix(mat));
}


void State::apply_matrix(const reg_t &qubits, const cvector_t &dmat) {
  if (qubits.empty() == false && dmat.size() > 0)
    data_.apply_matrix(qubits, dmat);
}


void State::apply_gate_u3(uint_t qubit, double theta, double phi, double lambda) {
  data_.apply_matrix(qubit, Utils::vectorize_matrix(Utils::Matrix::U3(theta, phi, lambda)));
}


void State::apply_gate_phase(uint_t qubit, complex_t phase) {
  data_.apply_matrix(qubit, cvector_t({1., phase}));
}


cvector_t State::rzz_diagonal_matrix(double lambda) {
  const complex_t one(1.0, 0);
  const complex_t phase = exp(complex_t(0, lambda));
  return cvector_t({one, phase, phase, one});
}


//============================================================================
// Implementation: Reset and Measurement Sampling
//============================================================================

void State::apply_reset(const reg_t &qubits, const uint_t reset_state) {

  // Simulate unobserved measurement
  const auto meas = sample_measure_outcome(qubits);
  // Apply update tp reset state
  measure_reset_update(qubits, reset_state, meas.first, meas.second);
}


std::pair<uint_t, double> State::sample_measure_outcome(const reg_t &qubits) {

  // Sample outcome of a multi-qubit joint Z-measurement
  // Returns a pair [m, p] of outcome (m) and corresponding probability (p)

  if (qubits.size() == 1) {
    // Probability of P0 outcome
    double p0 = data_.probability(qubits[0], 0);
    rvector_t probs = {p0, 1. - p0};
    // randomly pick outcome
    uint_t outcome = rng_.rand_int(probs);
    return std::make_pair(outcome, probs[outcome]);
  } else {
    // Calculate measurement outcome probabilities
    rvector_t probs = data_.probabilities(qubits);
    // Randomly pick outcome and return pair
    const uint_t outcome = rng_.rand_int(probs);
    return std::make_pair(outcome, probs[outcome]);
  }
}


void State::measure_reset_update(const std::vector<uint_t> &qubits,
                                 const uint_t final_state,
                                 const uint_t meas_state,
                                 const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from 
  // sample_measure_outcome function, and a desired post-measurement final_state
  
  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    data_.apply_matrix(qubits[0], mdiag);

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      data_.apply_x(qubits[0]);
    }
  } 
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    data_.apply_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      // build vectorized permutation matrix  
      cvector_t perm(dim * dim, 0.); 
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (size_t j=0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      data_.apply_matrix(qubits, perm);
    }
  }
}


//============================================================================
// Implementation: Kraus Noise
//============================================================================

void State::apply_kraus(const reg_t &qubits,
                        const std::vector<cmatrix_t> &kmats) {
  
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early


  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double r = rng_.rand(0., 1.);
  double accum = 0.;
  bool complete = false;

  // Loop through N-1 kraus operators
  for (size_t j=0; j < kmats.size() - 1; j++) {
    
    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);
    double p = data_.norm(qubits, vmat);
    accum += p;
    
    // check if we need to apply this operator
    if (accum > r) {
      // rescale vmat so projection is normalized
      complex_t renorm = 1 / std::sqrt(p);
      for (auto &v : vmat) {
        v *= renorm;
      }
      // apply Kraus projection operator
      data_.apply_matrix(qubits, vmat);
      complete = true;
      break;
    }
  }

  // check if we haven't applied a kraus operator yet
  if (complete == false) {
    // Compute probability from accumulated
    complex_t renorm = 1 / std::sqrt(1. - accum);
    data_.apply_matrix(qubits, Utils::vectorize_matrix(renorm * kmats.back()));
  }
}

//------------------------------------------------------------------------------
} // end namespace QubitVector
} // end namespace AER
//------------------------------------------------------------------------------
#endif