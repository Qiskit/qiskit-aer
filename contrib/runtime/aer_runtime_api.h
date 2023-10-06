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
#include <complex.h>
#include <stdint.h>

typedef uint_fast64_t uint_t; 

// construct an aer state
void* aer_state();

// initialize aer state
void aer_state_initialize(void* state);

// finalize state
void aer_state_finalize(void* state);

// configure state
void aer_state_configure(void* state, char* key, char* value);

// allocate qubits and return the first qubit index.
// following qubits are indexed with incremented indices.
uint_t aer_allocate_qubits(void* state, uint_t num_qubits);

// measure qubits
uint_t aer_apply_measure(void* state, uint_t* qubits, size_t num_qubits);

// return probability of a specific bitstring
double aer_probability(void* state, uint_t outcome);

// return probability amplitude of a specific bitstring
double complex aer_amplitude(void* state, uint_t outcome);

// return probability amplitudes
// returned pointer must be freed in the caller
double complex* aer_release_statevector(void* state);

// u3 gate
void aer_apply_u3(void* state, uint_t qubit, double theta, double phi, double lambda);

// phase gate
void aer_apply_p(void* state, uint_t qubit, double lambda);

// Pauli gate: bit-flip or NOT gate
void aer_apply_x(void* state, uint_t qubit);
// Pauli gate: bit and phase flip
void aer_apply_y(void* state, uint_t qubit);
// Pauli gate: phase flip
void aer_apply_z(void* state, uint_t qubit);

// Clifford gate: Hadamard
void aer_apply_h(void* state, uint_t qubit);
// Clifford gate: sqrt(Z) or S gate
void aer_apply_s(void* state, uint_t qubit);
// Clifford gate: inverse of sqrt(Z)
void aer_apply_sdg(void* state, uint_t qubit);

// sqrt(S) or T gate
void aer_apply_t(void* state, uint_t qubit);
// inverse of sqrt(S)
void aer_apply_tdg(void* state, uint_t qubit);

// sqrt(NOT) gate
void aer_apply_sx(void* state, uint_t qubit);

// Rotation around X-axis
void aer_apply_rx(void* state, uint_t qubit, double theta);
// rotation around Y-axis
void aer_apply_ry(void* state, uint_t qubit, double theta);
// rotation around Z axis
void aer_apply_rz(void* state, uint_t qubit, double theta);

// controlled-NOT
void aer_apply_cx(void* state, uint_t ctrl_qubit, uint_t tgt_qubit);
// controlled-Y
void aer_apply_cy(void* state, uint_t ctrl_qubit, uint_t tgt_qubit);
// controlled-Z
void aer_apply_cz(void* state, uint_t ctrl_qubit, uint_t tgt_qubit);
// controlled-phase
void aer_apply_cp(void* state, uint_t ctrl_qubit, uint_t tgt_qubit, double lambda);
// controlled-rx
void aer_apply_crx(void* state, uint_t ctrl_qubit, uint_t tgt_qubit, double theta);
// controlled-ry
void aer_apply_cry(void* state, uint_t ctrl_qubit, uint_t tgt_qubit, double theta);
// controlled-rz
void aer_apply_crz(void* state, uint_t ctrl_qubit, uint_t tgt_qubit, double theta);
// controlled-H
void aer_apply_ch(void* state, uint_t ctrl_qubit, uint_t tgt_qubit);

// swap
void aer_apply_swap(void* state, uint_t qubit0, uint_t qubit1);

// Toffoli
void aer_apply_ccx(void* state, uint_t qubit0, uint_t qubit1, uint_t qubit2);
// controlled-swap
void aer_apply_cswap(void* state, uint_t ctrl_qubit, uint_t qubit0, uint_t qubit1);

// four parameter controlled-U gate with relative phase γ
void aer_apply_cu(void* state, uint_t ctrl_qubit, uint_t tgt_qubit, double theta, double phi, double lambda, double gamma);