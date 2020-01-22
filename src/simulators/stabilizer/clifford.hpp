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

#ifndef _clifford_hpp_
#define _clifford_hpp_

#include "pauli.hpp"


namespace Clifford {

/*******************************************************************************
 *
 * Clifford Class
 *
 ******************************************************************************/

class Clifford {
public:
  using phase_t = int8_t;
  using phasevec_t = std::vector<phase_t>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------
  Clifford() = default;
  explicit Clifford(const uint64_t nqubit);

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Get number of qubits of the Clifford table
  uint64_t num_qubits() const {return num_qubits_;}

  // Return true if the number of qubits is 0
  bool empty() const {return (num_qubits_ == 0);}

  // Return JSON serialization of QubitVector;
  json_t json() const;

  // Access stabilizer table
  Pauli::Pauli &operator[](uint64_t j) {return table_[j];}
  const Pauli::Pauli& operator[](uint64_t j) const {return table_[j];}

  // Return reference to internal Stabilizer table
  std::vector<Pauli::Pauli>& table() {return table_;}
  const std::vector<Pauli::Pauli>& table() const {return table_;}

  // Return reference to internal phases vector
  phasevec_t& phases() {return phases_;}
  const phasevec_t& phases() const {return phases_;}

  // Return n-th destabilizer from internal stabilizer table
  Pauli::Pauli &destabilizer(uint64_t n) {return table_[n];}
  const Pauli::Pauli& destabilizer(uint64_t n) const {return table_[n];}

  // Return n-th stabilizer from internal stabilizer table
  Pauli::Pauli &stabilizer(uint64_t n) {return table_[num_qubits_ + n];}
  const Pauli::Pauli& stabilizer(uint64_t n) const {return table_[num_qubits_ + n];}


  //-----------------------------------------------------------------------
  // Apply basic Clifford gates
  //-----------------------------------------------------------------------

  // Apply Controlled-NOT (CX) gate
  void append_cx(const uint64_t qubit_ctrl, const uint64_t qubit_trgt);

  // Apply Hadamard (H) gate
  void append_h(const uint64_t qubit);

  // Apply Phase (S, square root of Z) gate
  void append_s(const uint64_t qubit);

  // Apply Pauli::Pauli X gate
  void append_x(const uint64_t qubit);

  // Apply Pauli::Pauli Y gate
  void append_y(const uint64_t qubit);

  // Apply Pauli::Pauli Z gate
  void append_z(const uint64_t qubit);

  //-----------------------------------------------------------------------
  // Measurement
  //-----------------------------------------------------------------------

  // If we perform a single qubit Z measurement, 
  // will the outcome be random or deterministic.
  bool is_deterministic_outcome(const uint64_t& qubit) const;

  // Return the outcome (0 or 1) of a single qubit Z measurement, and
  // update the stabilizer to the conditional (post measurement) state if
  // the outcome was random.
  bool measure_and_update(const uint64_t qubit, const uint64_t randint);

  // Return 1 or -1: the expectation value of observable Z on all
  // the qubits in the parameter `qubits`
  int64_t expectation_value(const std::vector<uint64_t>& qubits);

  //-----------------------------------------------------------------------
  // Configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() {return json_chop_threshold_;}

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint64_t get_omp_threads() {return omp_threads_;}

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint64_t get_omp_threshold() {return omp_threshold_;}

private:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------

  std::vector<Pauli::Pauli> table_;
  phasevec_t phases_;
  uint64_t num_qubits_ = 0;

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  uint64_t omp_threads_ = 1;          // Disable multithreading by default
  uint64_t omp_threshold_ = 1000;     // Qubit threshold for multithreading when enabled
  double json_chop_threshold_ = 0;  // Threshold for chopping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Helper functions
  //-----------------------------------------------------------------------

  // Check if there exists stabilizer or destabilizer row anticommuting
  // with Z[qubit]. If so return pair (true, row), else return (false, 0)
  std::pair<bool, uint64_t> z_anticommuting(const uint64_t qubit) const;

  // Check if there exists stabilizer or destabilizer row anticommuting
  // with X[qubit]. If so return pair (true, row), else return (false, 0)
  std::pair<bool, uint64_t> x_anticommuting(const uint64_t qubit) const;

  void rowsum_helper(const Pauli::Pauli &row, const phase_t row_phase,
                     Pauli::Pauli &accum, phase_t &accum_phase) const;
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Config settings
//------------------------------------------------------------------------------

void Clifford::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}

void Clifford::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

void Clifford::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

Clifford::Clifford(uint64_t nq) : num_qubits_(nq) {
  // initial state = all zeros
  // add destabilizers
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(nq); i++) {
    Pauli::Pauli P(nq);
    P.X.setValue(1, i);
    table_.push_back(P);
  }
  // add stabilizers
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(nq); i++) {
    Pauli::Pauli P(nq);
    P.Z.setValue(1, i);
    table_.push_back(P);
  }
  // Add phases
  phases_.resize(2 * nq);
}

//------------------------------------------------------------------------------
// Apply Clifford gates
//------------------------------------------------------------------------------

void Clifford::append_cx(const uint64_t qcon, const uint64_t qtar) {
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(2 * num_qubits_); i++) {
    phases_[i] ^= (table_[i].X[qcon] && table_[i].Z[qtar] &&
                  (table_[i].X[qtar] ^ table_[i].Z[qcon] ^ 1));
    table_[i].X.setValue(table_[i].X[qtar] ^ table_[i].X[qcon], qtar);
    table_[i].Z.setValue(table_[i].Z[qtar] ^ table_[i].Z[qcon], qcon);
  }
}

void Clifford::append_h(const uint64_t qubit) {
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(2 * num_qubits_); i++) {
    phases_[i] ^= (table_[i].X[qubit] && table_[i].Z[qubit]);
    // exchange X and Z
    bool b = table_[i].X[qubit];
    table_[i].X.setValue(table_[i].Z[qubit], qubit);
    table_[i].Z.setValue(b, qubit);
  }
}

void Clifford::append_s(const uint64_t qubit) {
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(2 * num_qubits_); i++) {
    phases_[i] ^= (table_[i].X[qubit] && table_[i].Z[qubit]);
    table_[i].Z.setValue(table_[i].Z[qubit] ^ table_[i].X[qubit], qubit);
  }
}

void Clifford::append_x(const uint64_t qubit) {
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(2 * num_qubits_); i++)
    phases_[i] ^= table_[i].Z[qubit];
}

void Clifford::append_y(const uint64_t qubit) {
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(2 * num_qubits_); i++)
    phases_[i] ^= (table_[i].Z[qubit] ^ table_[i].X[qubit]);
}

void Clifford::append_z(const uint64_t qubit) {
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int64_t i = 0; i < static_cast<int64_t>(2 * num_qubits_); i++)
    phases_[i] ^= table_[i].X[qubit];
}


//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

std::pair<bool, uint64_t> Clifford::z_anticommuting(const uint64_t qubit) const {
  for (uint64_t p = num_qubits_; p < 2 * num_qubits_; p++) {
    if (table_[p].X[qubit])
      return std::make_pair(true, p);
  }
  return std::make_pair(false, 0);
}


std::pair<bool, uint64_t> Clifford::x_anticommuting(const uint64_t qubit) const {
  for (uint64_t p = num_qubits_; p < 2 * num_qubits_; p++) {
    if (table_[p].Z[qubit])
      return std::make_pair(true, p);
  }
  return std::make_pair(false, 0);
}

//------------------------------------------------------------------------------
// Measurement
//------------------------------------------------------------------------------

bool Clifford::is_deterministic_outcome(const uint64_t& qubit) const {
  // Clifford state measurements only have three probabilities:
  // (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
  // The random case happens if there is a row anti-commuting with Z[qubit]
  return !z_anticommuting(qubit).first;
}

bool Clifford::measure_and_update(const uint64_t qubit, const uint64_t randint) {
  // Clifford state measurements only have three probabilities:
  // (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
  // The random case happens if there is a row anti-commuting with Z[qubit]
  auto anticom = z_anticommuting(qubit);
  if (anticom.first) {
    bool outcome = (randint == 1);
    auto row = anticom.second;
    for (uint64_t i = 0; i < 2 * num_qubits_; i++) {
      // the last condition is not in the AG paper but we seem to need it
      if ((table_[i].X[qubit]) && (i != row) && (i != (row - num_qubits_))) {
        rowsum_helper(table_[row], phases_[row], table_[i], phases_[i]);
      }
    }
    // Update state
    table_[row - num_qubits_].X = table_[row].X;
    table_[row - num_qubits_].Z = table_[row].Z;
    phases_[row - num_qubits_] = phases_[row];
    table_[row].X.makeZero();
    table_[row].Z.makeZero();
    table_[row].Z.setValue(1, qubit);
    phases_[row] = outcome;
    return outcome;
  } else {
    // Deterministic outcome
    Pauli::Pauli accum(num_qubits_);
    phase_t outcome = 0;
    for (uint64_t i = 0; i < num_qubits_; i++) {
      if (table_[i].X[qubit]) {
        rowsum_helper(table_[i + num_qubits_], phases_[i + num_qubits_],
                      accum, outcome);
      }
    }
    return outcome;
  }
}

int64_t Clifford::expectation_value(const std::vector<uint64_t>& qubits) {
  // Check if there is a row that anti-commutes with an odd number of qubits
  // If so expectation value is 0
  for (uint64_t p = num_qubits_; p < 2 * num_qubits_; p++) {
    uint64_t num_of_x = 0;
    for (auto qubit : qubits) {
      if (table_[p].X[qubit])
	num_of_x ++;
    }
    if(num_of_x % 2 == 1)
      return 0;
  }

  // Otherwise the expectation value is +1 or -1
  uint64_t sum_of_outcomes = 0;
  for (auto qubit : qubits) {
    Pauli::Pauli accum(num_qubits_);
    phase_t outcome = 0;
    for (uint64_t i = 0; i < num_qubits_; i++) {
      if (table_[i].X[qubit]) {
        rowsum_helper(table_[i + num_qubits_], phases_[i + num_qubits_],
                      accum, outcome);
      }
    }
    sum_of_outcomes += outcome;
  }

  return (sum_of_outcomes % 2 == 0) ? 1 : -1;
}

void Clifford::rowsum_helper(const Pauli::Pauli &row, const phase_t row_phase,
                             Pauli::Pauli &accum, phase_t &accum_phase) const {
  int8_t newr = ((2 * row_phase + 2 * accum_phase) +
                 Pauli::Pauli::phase_exponent(row, accum)) % 4;
  // Since we are only using +1 and -1 phases in our Clifford phases
  // the exponent must be 0 (for +1) or 2 (for -1)
  if ((newr != 0) && (newr != 2)) {
    throw std::runtime_error("Clifford: rowsum error");
  }
  accum_phase = (newr == 2);
  accum.X += row.X;
  accum.Z += row.Z;
}

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

json_t Clifford::json() const {
  json_t js = json_t::object();
  // Add destabilizers
  json_t stab;
  for (size_t i = 0; i < num_qubits_; i++) {
    // Destabilizer
    std::string label = (phases_[i] == 0) ? "" : "-";
    label += table_[i].str();
    js["destabilizers"].push_back(label);

    // Stabilizer
    label = (phases_[num_qubits_ + i] == 0) ? "" : "-";
    label += table_[num_qubits_ + i].str();
    js["stabilizers"].push_back(label);
  }
  return js;
}

inline void to_json(json_t &js, const Clifford &clif) {
  js = clif.json();
}

inline void from_json(const json_t &js, Clifford &clif) {
  bool has_keys = JSON::check_keys({"stabilizers", "destabilizers"}, js);
  if (!has_keys)
    throw std::invalid_argument("Invalid Clifford JSON.");

  const std::vector<std::string> stab = js["stabilizers"];
  const std::vector<std::string> destab = js["destabilizers"];
  const auto nq = stab.size();
  if (nq != destab.size()) {
    throw std::invalid_argument("Invalid Clifford JSON: stabilizer and destabilizer lengths do not match.");
  }

  clif = Clifford(nq);
  for (size_t i = 0; i < nq; i++) {
    std::string label;
    // Get destabilizer
    label = destab[i];
    switch (label[0]) {
      case '-':
        clif.phases()[i] = 1;
        clif.table()[i] = Pauli::Pauli(label.substr(1, nq));
        break;
      case '+':
        clif.table()[i] = Pauli::Pauli(label.substr(1, nq));
        break;
      case 'I':
      case 'X':
      case 'Y':
      case 'Z':
        clif.table()[i] = Pauli::Pauli(label);
        break;
      default:
        throw std::invalid_argument("Invalid Stabilizer JSON string.");
    }
    // Get stabilizer
    label = stab[i];
    switch (label[0]) {
      case '-':
        clif.phases()[i + nq] = 1;
        clif.table()[i + nq] = Pauli::Pauli(label.substr(1, nq));
        break;
      case '+':
        clif.table()[i + nq] = Pauli::Pauli(label.substr(1, nq));
        break;
      case 'I':
      case 'X':
      case 'Y':
      case 'Z':
        clif.table()[i + nq] = Pauli::Pauli(label);
        break;
      default:
        throw std::invalid_argument("Invalid Stabilizer JSON string.");
    }
  }
}


//------------------------------------------------------------------------------
} // end namespace Clifford
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class statevector_t>
std::ostream &operator<<(std::ostream &out, const Clifford::Clifford &clif) {
  out << clif.json().dump();
  return out;
}

//------------------------------------------------------------------------------
#endif
