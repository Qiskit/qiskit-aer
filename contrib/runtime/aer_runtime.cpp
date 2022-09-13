/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#include <cmath>
#include "controllers/state_controller.hpp"

// initialize and return state
extern "C" {

void* aer_state() {
  AER::AerState* handler = new AER::AerState();
  return handler;
};

void* aer_state_initialize(void* handler) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->initialize();
  return handler;
};

// finalize state
void aer_state_finalize(void* handler) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  delete(state);
};

// configure state
void aer_state_configure(void* handler, char* key, char* value) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->configure(key, value);
};

// allocate qubits and return the first qubit index.
// following qubits are indexed with incremented indices.
uint_t aer_allocate_qubits(void* handler, uint_t num_qubits) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  auto qubit_ids = state->allocate_qubits(num_qubits);
  return qubit_ids[0];
};

// measure qubits
uint_t aer_apply_measure(void* handler, uint_t* qubits_, size_t num_qubits) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  std::vector<uint_t> qubits;
  qubits.insert(qubits.end(), &(qubits_[0]), &(qubits[num_qubits - 1]));
  return state->apply_measure(qubits);
};

// return probability of a specific bitstring
double aer_probability(void* handler, uint_t outcome) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  return state->probability(outcome);
};

// return probability amplitude of a specific bitstring
complex_t aer_amplitude(void* handler, uint_t outcome) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  return state->amplitude(outcome);
};

// return probability amplitudes
// returned pointer must be freed in the caller
complex_t* aer_release_statevector(void* handler) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  AER::Vector<complex_t> sv = state->move_to_vector();
  return sv.move_to_buffer();
};

// phase gate
void aer_apply_p(void* handler, uint_t qubit, double lambda) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcphase({qubit}, lambda);
};

// Pauli gate: bit-flip or NOT gate
void aer_apply_x(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcx({qubit});
};

// Pauli gate: bit and phase flip
void aer_apply_y(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcy({qubit});
};

// Pauli gate: phase flip
void aer_apply_z(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcz({qubit});
};

// Clifford gate: Hadamard
void aer_apply_h(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcu({qubit}, M_PI / 2.0, 0, M_PI);
};

// Clifford gate: sqrt(Z) or S gate
void aer_apply_s(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcu({qubit}, 0, 0, M_PI / 2.0);
};

// Clifford gate: inverse of sqrt(Z)
void aer_apply_sdg(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcu({qubit}, 0, 0, - M_PI / 2.0);
};

// // sqrt(S) or T gate
void aer_apply_t(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcu({qubit}, 0, 0, M_PI / 4.0);
};

// inverse of sqrt(S)
void aer_apply_tdg(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcu({qubit}, 0, 0, - M_PI / 4.0);
};

// sqrt(NOT) gate
void aer_apply_sx(void* handler, uint_t qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcrx({qubit}, - M_PI / 4.0);
};

// Rotation around X-axis
void aer_apply_rx(void* handler, uint_t qubit, double theta) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcrx({qubit}, theta);
};

// rotation around Y-axis
void aer_apply_ry(void* handler, uint_t qubit, double theta) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcry({qubit}, theta);
};

// rotation around Z axis
void aer_apply_rz(void* handler, uint_t qubit, double theta) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcrz({qubit}, theta);
};

// controlled-NOT
void aer_apply_cx(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcx({ctrl_qubit, tgt_qubit});
};

// controlled-Y
void aer_apply_cy(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcy({ctrl_qubit, tgt_qubit});
};

// controlled-Z
void aer_apply_cz(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcz({ctrl_qubit, tgt_qubit});
};

// controlled-phase
void aer_apply_cp(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit, double lambda) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcphase({ctrl_qubit, tgt_qubit}, lambda);
};

// controlled-rx
void aer_apply_crx(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit, double theta) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcrx({ctrl_qubit, tgt_qubit}, theta);
};

// controlled-ry
void aer_apply_cry(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit, double theta) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcry({ctrl_qubit, tgt_qubit}, theta);
};

// controlled-rz
void aer_apply_crz(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit, double theta) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcrz({ctrl_qubit, tgt_qubit}, theta);
};

// controlled-H
void aer_apply_ch(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcu({ctrl_qubit, tgt_qubit}, M_PI / 2.0, 0, M_PI);
};

// swap
void aer_apply_swap(void* handler, uint_t qubit0, uint_t qubit1) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcswap({qubit0, qubit1});
};

// Toffoli
void aer_apply_ccx(void* handler, uint_t qubit0, uint_t qubit1, uint_t qubit2) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcx({qubit0, qubit1, qubit2});
};

// // controlled-swap
void aer_apply_cswap(void* handler, uint_t ctrl_qubit, uint_t qubit0, uint_t qubit1) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcswap({ctrl_qubit, qubit0, qubit1});
};

// four parameter controlled-U gate with relative phase γ
void aer_apply_cu(void* handler, uint_t ctrl_qubit, uint_t tgt_qubit, double theta, double phi, double lambda, double gamma) {
  AER::AerState* state = reinterpret_cast<AER::AerState*>(handler);
  state->apply_mcphase({ctrl_qubit}, gamma);
  state->apply_mcu({ctrl_qubit, tgt_qubit}, theta, phi, lambda);
};
}
