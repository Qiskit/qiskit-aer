/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    optimization.hpp
 * @brief   Optimization for circuits
 * @authors Hiroshi Horii <horii@jp.ibm.com>
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "framework/circuit.hpp"
#include "framework/utils.hpp"

#define UNUSED(x) ((void)x)

namespace AER {
namespace QubitVector {

using Operations::Op;

/*******************************************************************************
 *
 * Optimization Class
 *
 ******************************************************************************/
class Optimization {

public:
  enum class OpType {
    mat, dmat, kraus, // special
    measure, reset, barrier,
    u0, u1, u2, u3, id, x, y, z, h, s, sdg, t, tdg, // single qubit
    cx, cz, rzz, // two qubit
    unknown
  };

  const static std::unordered_map<std::string, OpType> optypes;

  Optimization();

  virtual bool optimize(const std::vector<Circuit>& original, std::vector<Circuit>& optimized) const;
  virtual std::string name() const { return "base_optimization"; };

protected:
  virtual bool optimize(const Circuit &original, Circuit &optimized) const { UNUSED(original); UNUSED(optimized); return false; };

  virtual Circuit generate_circuit(const Circuit &original, std::vector<Op>& ops) const;

  OpType type(const Op& op) const;

  bool is_qubit(const Op& op) const;
  bool is_control_qubit(const Op& op) const;
  bool is_target_qubit(const Op& op) const;
  bool fusionable(const Op& op) const;

  uint_t qubit(const Op& op) const;
  uint_t control_qubit(const Op& op) const;
  uint_t target_qubit(const Op& op) const;
  reg_t qubits(const Op& op) const;
  cmatrix_t matrix(const Op& op) const;

  double lambda(const Op& op) const;
  double theta(const Op& op) const;
  double phi(const Op& op) const;

  uint_t creg(const Op& op) const;
  double n(const Op& op) const;
  uint_t reset_state(const Op& op) const;
  bool is_condition(const Op& op) const;
};

const std::unordered_map<std::string, Optimization::OpType> Optimization::optypes({
    {"reset", OpType::reset},     // Reset operation
    {"measure", OpType::measure}, // measure operation
    {"barrier", OpType::barrier}, // barrier does nothing
    // Matrix multiplication
    {"mat", OpType::mat},         // matrix multiplication
    {"dmat", OpType::dmat},       // Diagonal matrix multiplication
    // Single qubit gates
    {"id", OpType::id},           // Pauli-Identity gate
    {"x", OpType::x},             // Pauli-X gate
    {"y", OpType::y},             // Pauli-Y gate
    {"z", OpType::z},             // Pauli-Z gate
    {"s", OpType::s},             // Phase gate (aka sqrt(Z) gate)
    {"sdg", OpType::sdg},         // Conjugate-transpose of Phase gate
    {"h", OpType::h},             // Hadamard gate (X + Z / sqrt(2))
    {"t", OpType::t},             // T-gate (sqrt(S))
    {"tdg", OpType::tdg},         // Conjguate-transpose of T gate
    // Waltz OpType
    {"u0", OpType::u0},           // idle gate in multiples of X90
    {"u1", OpType::u1},           // zero-X90 pulse waltz gate
    {"u2", OpType::u2},           // single-X90 pulse waltz gate
    {"u3", OpType::u3},           // two X90 pulse waltz gate
    // Two-qubit gates
    {"cx", OpType::cx},           // Controlled-X gate (CNOT)
    {"cz", OpType::cz},           // Controlled-Z gate
    {"rzz", OpType::rzz},         // ZZ-rotation gate
    // Type-2 Noise
    {"kraus", OpType::kraus}      // Kraus error
  });

Optimization::Optimization() {
}

Optimization::OpType Optimization::type(const Op& op) const {
  auto it = optypes.find(op.name);
  if (it == optypes.end())
    return OpType::unknown;
  else
    return it->second;
}

bool Optimization::fusionable(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  case OpType::mat:               // matrix multiplication
  case OpType::dmat:              // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
  case OpType::y:                 // Pauli-Y gate
  case OpType::z:                 // Pauli-Z gate
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
  case OpType::sdg:               // Conjugate-transpose of Phase gate
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
  case OpType::t:                 // T-gate (sqrt(S))
  case OpType::tdg:               // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
  case OpType::u2:                // single-X90 pulse waltz gate
  case OpType::u3:                // two X90 pulse waltz gate
  case OpType::cx:                // Controlled-X gate (CNOT)
  case OpType::cz:                // Controlled-Z gate
  case OpType::rzz:               // ZZ-rotation gate
    // Type-2 Noise
    //case OpType::kraus:         // Kraus error
    return true;
  default:
    return false;
  }
}

uint_t Optimization::qubit(const Op& op) const {
  switch (type(op)) {
  case OpType::reset:             // Reset operation
  case OpType::measure:           // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
  case OpType::y:                 // Pauli-Y gate
  case OpType::z:                 // Pauli-Z gate
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
  case OpType::sdg:               // Conjugate-transpose of Phase gate
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
  case OpType::t:                 // T-gate (sqrt(S))
  case OpType::tdg:               // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
  case OpType::u2:                // single-X90 pulse waltz gate
  case OpType::u3:                // two X90 pulse waltz gate
  //case OpType::cx:              // Controlled-X gate (CNOT)
  //case OpType::cz:              // Controlled-Z gate
  //case OpType::rzz:             // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
    return op.qubits[0];
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.qubit()";
    throw std::runtime_error(msg.str());
  }
}

bool Optimization::is_qubit(const Op& op) const {
  switch (type(op)) {
  case OpType::reset:             // Reset operation
  case OpType::measure:           // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
  case OpType::y:                 // Pauli-Y gate
  case OpType::z:                 // Pauli-Z gate
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
  case OpType::sdg:               // Conjugate-transpose of Phase gate
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
  case OpType::t:                 // T-gate (sqrt(S))
  case OpType::tdg:               // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
  case OpType::u2:                // single-X90 pulse waltz gate
  case OpType::u3:                // two X90 pulse waltz gate
  //case OpType::cx:              // Controlled-X gate (CNOT)
  //case OpType::cz:              // Controlled-Z gate
  //case OpType::rzz:             // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
    return true;
  default:
    return false;
  }
}

uint_t Optimization::control_qubit(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  //case OpType::x:               // Pauli-X gate
  //case OpType::y:               // Pauli-Y gate
  //case OpType::z:               // Pauli-Z gate
  //case OpType::s:               // Phase gate (aka sqrt(Z) gate)
  //case OpType::sdg:             // Conjugate-transpose of Phase gate
  //case OpType::h:               // Hadamard gate (X + Z / sqrt(2))
  //case OpType::t:               // T-gate (sqrt(S))
  //case OpType::tdg:             // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  //case OpType::u1:              // zero-X90 pulse waltz gate
  //case OpType::u2:              // single-X90 pulse waltz gate
  //case OpType::u3:              // two X90 pulse waltz gate
  case OpType::cx:                // Controlled-X gate (CNOT)
  case OpType::cz:                // Controlled-Z gate
  case OpType::rzz:               // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
    return op.qubits[0];
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.control_qubit()";
    throw std::runtime_error(msg.str());
  }
}

bool Optimization::is_control_qubit(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  //case OpType::x:               // Pauli-X gate
  //case OpType::y:               // Pauli-Y gate
  //case OpType::z:               // Pauli-Z gate
  //case OpType::s:               // Phase gate (aka sqrt(Z) gate)
  //case OpType::sdg:             // Conjugate-transpose of Phase gate
  //case OpType::h:               // Hadamard gate (X + Z / sqrt(2))
  //case OpType::t:               // T-gate (sqrt(S))
  //case OpType::tdg:             // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  //case OpType::u1:              // zero-X90 pulse waltz gate
  //case OpType::u2:              // single-X90 pulse waltz gate
  //case OpType::u3:              // two X90 pulse waltz gate
  case OpType::cx:                // Controlled-X gate (CNOT)
  case OpType::cz:                // Controlled-Z gate
  case OpType::rzz:               // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
    return true;
  default:
    return false;
  }
}

uint_t Optimization::target_qubit(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  //case OpType::x:               // Pauli-X gate
  //case OpType::y:               // Pauli-Y gate
  //case OpType::z:               // Pauli-Z gate
  //case OpType::s:               // Phase gate (aka sqrt(Z) gate)
  //case OpType::sdg:             // Conjugate-transpose of Phase gate
  //case OpType::h:               // Hadamard gate (X + Z / sqrt(2))
  //case OpType::t:               // T-gate (sqrt(S))
  //case OpType::tdg:             // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  //case OpType::u1:              // zero-X90 pulse waltz gate
  //case OpType::u2:              // single-X90 pulse waltz gate
  //case OpType::u3:              // two X90 pulse waltz gate
  case OpType::cx:                // Controlled-X gate (CNOT)
  case OpType::cz:                // Controlled-Z gate
  case OpType::rzz:               // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
    return op.qubits[1];
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.target_qubit()";
    throw std::runtime_error(msg.str());
  }
}

bool Optimization::is_target_qubit(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  //case OpType::x:               // Pauli-X gate
  //case OpType::y:               // Pauli-Y gate
  //case OpType::z:               // Pauli-Z gate
  //case OpType::s:               // Phase gate (aka sqrt(Z) gate)
  //case OpType::sdg:             // Conjugate-transpose of Phase gate
  //case OpType::h:               // Hadamard gate (X + Z / sqrt(2))
  //case OpType::t:               // T-gate (sqrt(S))
  //case OpType::tdg:             // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  //case OpType::u1:              // zero-X90 pulse waltz gate
  //case OpType::u2:              // single-X90 pulse waltz gate
  //case OpType::u3:              // two X90 pulse waltz gate
  case OpType::cx:                // Controlled-X gate (CNOT)
  case OpType::cz:                // Controlled-Z gate
  case OpType::rzz:               // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
    return true;
  default:
    return false;
  }
}

reg_t Optimization::qubits(const Op& op) const {
  switch (type(op)) {
  case OpType::reset:             // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  case OpType::mat:               // matrix multiplication
  case OpType::dmat:              // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
  case OpType::y:                 // Pauli-Y gate
  case OpType::z:                 // Pauli-Z gate
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
  case OpType::sdg:               // Conjugate-transpose of Phase gate
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
  case OpType::t:                 // T-gate (sqrt(S))
  case OpType::tdg:               // Conjguate-transpose of T gate
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
  case OpType::u2:                // single-X90 pulse waltz gate
  case OpType::u3:                // two X90 pulse waltz gate
  case OpType::cx:                // Controlled-X gate (CNOT)
  case OpType::cz:                // Controlled-Z gate
  case OpType::rzz:               // ZZ-rotation gate
  // Type-2 Noise
  case OpType::kraus:           // Kraus error
    return op.qubits;
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.qubits()";
    throw std::runtime_error(msg.str());
  }
}

cmatrix_t Optimization::matrix(const Op& op) const {
  const complex_t one(1., 0.);
  switch (type(op)) {
  //case OpType::reset:             // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  case OpType::mat:               // matrix multiplication
    return op.mats[0];
  case OpType::dmat:              // Diagonal matrix multiplication
    return Utils::devectorize_matrix<complex_t>(op.params);
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
    return Utils::Matrix::X;
  case OpType::y:                 // Pauli-Y gate
    return Utils::Matrix::Y;
  case OpType::z:                 // Pauli-Z gate
    return Utils::Matrix::Z;
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
    return Utils::Matrix::S;
  case OpType::sdg:               // Conjugate-transpose of Phase gate
    return Utils::Matrix::SDG;
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
    return Utils::Matrix::H;
  case OpType::t:                 // T-gate (sqrt(S))
   return Utils::Matrix::T;
  case OpType::tdg:               // Conjguate-transpose of T gate
    return Utils::Matrix::TDG;
  //case OpType::u0:                // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
    return Utils::Matrix::U1(lambda(op));
  case OpType::u2:                // single-X90 pulse waltz gate
    return Utils::Matrix::U2(phi(op), lambda(op));
  case OpType::u3:                // two X90 pulse waltz gate
    return Utils::Matrix::U3(theta(op), phi(op), lambda(op));
  case OpType::cx:                // Controlled-X gate (CNOT)
    return Utils::Matrix::CX;
  case OpType::cz:                // Controlled-Z gate
    return Utils::Matrix::CZ;
  case OpType::rzz:               // ZZ-rotation gate
    return Utils::devectorize_matrix<complex_t>(cvector_t({one, std::real(op.params[0]), std::real(op.params[0]), one}));
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.matrix()";
    throw std::runtime_error(msg.str());
  }
}

double Optimization::theta(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
  case OpType::y:                 // Pauli-Y gate
    return M_PI;
  case OpType::z:                 // Pauli-Z gate
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
  case OpType::sdg:               // Conjugate-transpose of Phase gate
    return 0.;
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
    return M_PI / 2.;
  case OpType::t:                 // T-gate (sqrt(S))
  case OpType::tdg:               // Conjguate-transpose of T gate
    return 0.;
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
    return 0.;
  case OpType::u2:                // single-X90 pulse waltz gate
    return M_PI / 2.;
  case OpType::u3:                // two X90 pulse waltz gate
    return std::real(op.params[0]);
  //case OpType::cx:              // Controlled-X gate (CNOT)
  //case OpType::cz:              // Controlled-Z gate
  //case OpType::rzz:             // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.theta()";
    throw std::runtime_error(msg.str());
  }
}

double Optimization::phi(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
    return 0;
  case OpType::y:                 // Pauli-Y gate
    return M_PI / 2.;
  case OpType::z:                 // Pauli-Z gate
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
  case OpType::sdg:               // Conjugate-transpose of Phase gate
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
  case OpType::t:                 // T-gate (sqrt(S))
  case OpType::tdg:               // Conjguate-transpose of T gate
    return 0.;
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
    return 0.;
  case OpType::u2:                // single-X90 pulse waltz gate
    return std::real(op.params[0]);
  case OpType::u3:                // two X90 pulse waltz gate
    return std::real(op.params[1]);
  //case OpType::cx:              // Controlled-X gate (CNOT)
  //case OpType::cz:              // Controlled-Z gate
  //case OpType::rzz:             // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.phi()";
    throw std::runtime_error(msg.str());
  }
}

double Optimization::lambda(const Op& op) const {
  switch (type(op)) {
  //case OpType::reset:           // Reset operation
  //case OpType::measure:         // measure operation
  //case OpType::barrier:         // barrier does nothing
  //case OpType::mat:             // matrix multiplication
  //case OpType::dmat:            // Diagonal matrix multiplication
  //case OpType::id:              // Pauli-Identity gate
  case OpType::x:                 // Pauli-X gate
    return M_PI;
  case OpType::y:                 // Pauli-Y gate
    return M_PI / 2.;
  case OpType::z:                 // Pauli-Z gate
    return M_PI;
  case OpType::s:                 // Phase gate (aka sqrt(Z) gate)
    return M_PI / 2.;
  case OpType::sdg:               // Conjugate-transpose of Phase gate
    return -M_PI / 2.;
  case OpType::h:                 // Hadamard gate (X + Z / sqrt(2))
    return M_PI;
  case OpType::t:                 // T-gate (sqrt(S))
    return M_PI / 4.;
  case OpType::tdg:               // Conjguate-transpose of T gate
    return -M_PI / 4.;
  //case OpType::u0:              // idle gate in multiples of X90
  case OpType::u1:                // zero-X90 pulse waltz gate
    return std::real(op.params[0]);
  case OpType::u2:                // single-X90 pulse waltz gate
    return std::real(op.params[1]);
  case OpType::u3:                // two X90 pulse waltz gate
    return std::real(op.params[2]);
  //case OpType::cx:              // Controlled-X gate (CNOT)
  //case OpType::cz:              // Controlled-Z gate
  //case OpType::rzz:             // ZZ-rotation gate
  // Type-2 Noise
  //case OpType::kraus:           // Kraus error
  default:
    std::stringstream msg;
    msg << "invalid operation:" << op.name << "\'.lambda()";
    throw std::runtime_error(msg.str());
  }
}

Circuit Optimization::generate_circuit(const Circuit &original, std::vector<Op>& ops) const {
  Circuit ret(ops);
  ret.shots = original.shots;
  ret.seed = original.seed;
  ret.header = original.header;
  ret.config = original.config;
  ret.measure_sampling_flag = original.measure_sampling_flag;
  return ret;
}


bool Optimization::optimize(const std::vector<Circuit>& original, std::vector<Circuit>& optimized) const {
  if (!optimized.empty())
    optimized.clear();

  bool same = true;
  for (uint_t i = 0; i < original.size(); ++i) {
    Circuit optimized_circuit;
    if (optimize(original[i], optimized_circuit)) {
      optimized.push_back(optimized_circuit);
      same = false;
    } else {
      optimized.push_back(original[i]);
    }
  }

  if (same)
    return false;

  return true;
}

/*******************************************************************************
 *
 * Fusion Class
 *
 ******************************************************************************/
class Fusion: public Optimization {
public:
  Fusion(uint_t max);

  std::string name() const override { return "fusion"; };

  virtual bool optimize(const Circuit &original, Circuit &optimized) const override;

  uint_t max;

  double cost_factor = 2.5;

private:
  void add_fusion_qubits(std::vector<uint_t>& fusioned, const Op& op) const;
  virtual double calculate_cost(const std::vector<Op>& ops, int from, int until) const;
  Op generate_fusion(std::vector<Op> &ops) const;
  void swap_cols_and_rows(const uint_t idx1, const uint_t idx2, cmatrix_t &mat) const;
  cmatrix_t sort_matrix(const std::vector<uint_t> &src, const std::vector<uint_t> &sorted, const cmatrix_t &mat) const;
};

Fusion::Fusion(uint_t max) {
  this->max = max;
}

bool Fusion::optimize(const Circuit &original, Circuit &optimized) const {
  std::vector<double> costs;
  std::vector<uint_t> fusion_from;

  bool same = true;
  // calculate the minimal path to each operation in the circuit
  for (int i = 0; i < original.ops.size(); ++i) {
    // first, fusion from i-th to i-th
    fusion_from.push_back(i);

    // calculate the initial cost from i-th to i-th
    if (i == 0) {
      // if this is the first op, no fusion
      costs.push_back(cost_factor);
    } else {
      // otherwise, i-th cost is calculated from (i-1)-th cost
      costs.push_back(costs[i - 1] + cost_factor);
    }

    // if this is not fusionable, no fusion
    if (i == 0 || !fusionable(original.ops[i]))
      continue;

    for (uint_t num_fusion = 2; num_fusion <= max; ++num_fusion) {
      // calculate cost if we introduce a fusion of (num_fusion) qubits

      // qubits to be handled
      std::vector<uint_t> fusioned;

      // add qubits of this op to be handled
      add_fusion_qubits(fusioned, original.ops[i]);

      for (int j = i - 1; j >= 0; --j) {
        // attempt to add qubits of j-th op
        if (!fusionable(original.ops[j]))
          break;

        add_fusion_qubits(fusioned, original.ops[j]);

        // cannot add (j-th) op if size of fusioned exceeds num_fusion
        if (fusioned.size() > num_fusion)
          break;

        // calculate a new cost of (i-th) by adding
        double cost = //
            calculate_cost(original.ops, j, i) // fusion gate from j-th to i-th, and
            + (j == 0 ? 0.0 : costs[j - 1]); // cost of (j-1)-th cost

        // if the new cost is lower than existing, update (i-th) cost
        if (cost < costs[i]) {
          costs[i] = cost;
          fusion_from[i] = j;
          same = false;
        }
      }
    }
  }

  if (same)
    return false;

  // generate a new circuit with the minimal path to the last operation in the circuit
  std::vector<Op> optimized_ops;

  for (int i = original.ops.size() - 1; i >= 0;) {
    uint_t from = fusion_from[i];

    if (from == (uint_t) i) {
      optimized_ops.insert(optimized_ops.begin(), original.ops[i]);
    } else {
      std::vector<Op> fusioned;
      for (int j = from; j <= i; ++j)
        fusioned.push_back(original.ops[j]);
      optimized_ops.insert(optimized_ops.begin(), generate_fusion(fusioned));
    }
    i = from - 1;
  }
  optimized = generate_circuit(original, optimized_ops);

  return true;
}

void Fusion::add_fusion_qubits(std::vector<uint_t>& fusioned, const Op& op) const {
  if (is_control_qubit(op)) {
    if (find(fusioned.begin(), fusioned.end(), control_qubit(op)) == fusioned.end())
      fusioned.push_back(control_qubit(op));
    if (find(fusioned.begin(), fusioned.end(), target_qubit(op)) == fusioned.end())
      fusioned.push_back(target_qubit(op));
  } else if (is_qubit(op)) {
    if (find(fusioned.begin(), fusioned.end(), qubit(op)) == fusioned.end())
      fusioned.push_back(qubit(op));
  }
}

double Fusion::calculate_cost(const std::vector<Op>& ops, int from, int until) const {
  std::vector<uint_t> fusioned;
  for (int i = from; i <= until; ++i)
    add_fusion_qubits(fusioned, ops[i]);

  switch (fusioned.size()) {
  case 1:
    return 1.0;
  case 2:
    if (until - from == 1)
      return 1.0;
    else
      return 1.5;
  case 3:
    return 1.8;
  default:
    return 1.8 * pow(cost_factor, (double) (fusioned.size() - 3));
  }
}

cmatrix_t Fusion::sort_matrix(const std::vector<uint_t> &src, const std::vector<uint_t> &sorted, const cmatrix_t &mat) const {
  cmatrix_t ret = mat;
  std::vector<uint_t> current = src;

  while (current != sorted) {
    uint_t from;
    uint_t to;
    for (from = 0; from < current.size(); ++from)
      if (current[from] != sorted[from])
        break;
    if (from == current.size())
      break;
    for (to = from + 1; to < current.size(); ++to)
      if (current[from] == sorted[to])
        break;
    if (to == current.size()) {
      std::stringstream ss;
      ss << "should not reach here : sort_matrix, src=" << src << ", sorted=" << sorted << ", current=" << current << ", from=" << from;
      throw std::runtime_error(ss.str());
    }
    swap_cols_and_rows(from, to, ret);

    uint_t cache = current[from];
    current[from] = current[to];
    current[to] = cache;
  }

  return ret;
}

void Fusion::swap_cols_and_rows(const uint_t idx1, const uint_t idx2, cmatrix_t &mat) const {

  uint_t size = mat.GetColumns();
  uint_t mask1 = (1UL << idx1);
  uint_t mask2 = (1UL << idx2);

  for (uint_t first = 0; first < size; ++first) {
    if ((first & mask1) && !(first & mask2)) {
      uint_t second = (first ^ mask1) | mask2;

      for (uint_t i = 0; i < size; ++i) {
        complex_t cache = mat(first, i);
        mat(first, i) = mat(second, i);
        mat(second, i) = cache;
      }
      for (uint_t i = 0; i < size; ++i) {
        complex_t cache = mat(i, first);
        mat(i, first) = mat(i, second);
        mat(i, second) = cache;
      }
    }
  }
}

Op Fusion::generate_fusion(std::vector<Op> &ops) const {

  reg_t fusioned_qubits;
  std::vector<reg_t> op_qubitss;
  std::vector<cmatrix_t> op_mats;
  op_mats.reserve(ops.size());

  for (Op &op : ops) {
    reg_t op_qubits = qubits(op);
    for (uint_t qubit : op_qubits) {
      if (find(fusioned_qubits.begin(), fusioned_qubits.end(), qubit) == fusioned_qubits.end()) {
        fusioned_qubits.push_back(qubit);
      }
    }
    if (op_qubits.size() == 1) {
      op_qubitss.push_back(op_qubits);
      op_mats.push_back(matrix(op));
    } else {
      reg_t sorted_qubits = op_qubits;
      std::sort(sorted_qubits.begin(), sorted_qubits.end());
      op_qubitss.push_back(sorted_qubits);
      if (sorted_qubits != op_qubits)
        op_mats.push_back(sort_matrix(op_qubits, sorted_qubits, matrix(op)));
      else
        op_mats.push_back(matrix(op));
    }
  }

  std::sort(fusioned_qubits.begin(), fusioned_qubits.end());

  const complex_t ZERO(0., 0.);
  const complex_t ONE(1., 0.);

  const uint_t matrix_size = (1UL << fusioned_qubits.size());

  cmatrix_t U(matrix_size, matrix_size);

  for (uint_t i = 0; i < matrix_size; ++i)
    for (uint_t j = 0; j < matrix_size; ++j)
      U(i, j) = ZERO;

  bool first = true;
  for (uint_t idx = 0; idx < op_qubitss.size(); ++idx) {
    cmatrix_t &op_mat = op_mats[idx];
    reg_t &op_qubits = op_qubitss[idx];

    // generate a matrix for op
    cmatrix_t u(matrix_size, matrix_size);

    // 0. initialize u
    for (uint_t i = 0; i < matrix_size * matrix_size; ++i)
      u[i] = ZERO;

    if (op_qubits.size() == 1) { //1-qubit operation
      // 1. identify delta
      uint_t index = 0;
      for (uint_t qubit : fusioned_qubits)
        if (qubit != op_qubits.front())
          ++index;
        else
          break;
      uint_t delta = (1UL << index);

      // 2. find op_mat(0, 0) position in U
      for (uint_t i = 0; i < matrix_size; ++i) {
        bool exist = false;
        for (uint_t j = 0; j < matrix_size; ++j)
          if (u(i, j) != ZERO)
            exist = true; // row-i has a value. need to go the next line
        if (exist)
          continue; // go the next line

        //  3. allocate op.u to u based on u(i, i) and delta
        u(i, i) = op_mat(0, 0);
        u(i, (i + delta)) = op_mat(0, 1);
        u((i + delta), i) = op_mat(1, 0);
        u((i + delta), (i + delta)) = op_mat(1, 1);
      }
    } else if (op_qubits.size() == 2) { //2-qubit operation
      // 1. identify low and high delta
      uint_t low = 0;
      uint_t high = 0;
      for (uint_t qubit : fusioned_qubits)
        if (qubit == op_qubits.front())
          break;
        else
          ++low;
      for (uint_t qubit : fusioned_qubits)
        if (qubit == op_qubits.back())
          break;
        else
          ++high;

      uint_t low_delta = (1UL << low);
      uint_t high_delta = (1UL << high);

      // 2. find op.u(0, 0) position in U
      for (uint_t i = 0; i < matrix_size; ++i) {
        bool exist = false;
        for (uint_t j = 0; j < matrix_size; ++j)
          if (u(i, j) != ZERO)
            exist = true; // row-i has a value. need to go the next line
        if (exist)
          continue; // go the next line

        //  3. allocate op_mat to u based on u(i, i) and delta
        u(i, i) = op_mat(0, 0);
        u(i, (i + low_delta)) = op_mat(0, 1);
        u(i, (i + high_delta)) = op_mat(0, 2);
        u(i, (i + low_delta + high_delta)) = op_mat(0, 3);
        u((i + low_delta), i) = op_mat(1, 0);
        u((i + low_delta), (i + low_delta)) = op_mat(1, 1);
        u((i + low_delta), (i + high_delta)) = op_mat(1, 2);
        u((i + low_delta), (i + low_delta + high_delta)) = op_mat(1, 3);
        u((i + high_delta), i) = op_mat(2, 0);
        u((i + high_delta), (i + low_delta)) = op_mat(2, 1);
        u((i + high_delta), (i + high_delta)) = op_mat(2, 2);
        u((i + high_delta), (i + low_delta + high_delta)) = op_mat(2, 3);
        u((i + low_delta + high_delta), i) = op_mat(3, 0);
        u((i + low_delta + high_delta), (i + low_delta)) = op_mat(3, 1);
        u((i + low_delta + high_delta), (i + high_delta)) = op_mat(3, 2);
        u((i + low_delta + high_delta), (i + low_delta + high_delta)) = op_mat(3, 3);
      }
    } else {
      std::stringstream msg;
      msg << "illegal qubit number: " << op_qubits.size();
      throw std::runtime_error(msg.str());
    }

    // 4. for the first time, copy u to U
    if (first) {
      for (unsigned int i = 0; i < matrix_size; ++i)
        for (unsigned int j = 0; j < matrix_size; ++j)
          U(i, j) = u(i, j);
      first = false;
    } else {
      // 5. otherwise, multiply u and U
      cmatrix_t u_tmp(matrix_size, matrix_size);
      for (uint_t i = 0; i < matrix_size; ++i)
        for (uint_t j = 0; j < matrix_size; ++j)
          u_tmp(i, j) = ZERO;

      for (unsigned int i = 0; i < matrix_size; ++i)
        for (unsigned int j = 0; j < matrix_size; ++j)
          for (unsigned int k = 0; k < matrix_size; ++k)
            u_tmp(i, j) += u(i, k) * U(k, j);

      for (unsigned int i = 0; i < matrix_size; ++i)
        for (unsigned int j = 0; j < matrix_size; ++j)
          U(i, j) = u_tmp(i, j);
    }
//    std::cout << "idx: " << idx << ": "<< std::endl;
//    for (unsigned int i = 0; i < matrix_size; ++i) {
//      for (unsigned int j = 0; j < matrix_size; ++j) {
//        std::cout << U(i, j) << ", ";
//      }
//      std::cout << std::endl;
//    }
  }

  Op matrix_op;
  matrix_op.name = "mat";
  matrix_op.mats.push_back(U);
  matrix_op.qubits = fusioned_qubits;

//  std::cout << "fusioned_matrix qubits=: " << fusioned_qubits << std::endl;
//  for (unsigned int i = 0; i < matrix_size; ++i) {
//    for (unsigned int j = 0; j < matrix_size; ++j) {
//      std::cout << U(i, j) << ", ";
//    }
//    std::cout << std::endl;
//  }

  return matrix_op;
}

}
}
