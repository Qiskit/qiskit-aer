# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import StatePreparation
from test.terra.reference import ref_multiplexer
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

from qiskit_aer.backends.aer_compiler import AerCompiler, assemble_circuit


class TestMultiplexer(SimulatorTestCase):
    """AerSimulator multiplexer gate tests in default basis."""

    # ---------------------------------------------------------------------
    # Test multiplexer-cx-gate
    # ---------------------------------------------------------------------
    def test_multiplexer_cx_gate_deterministic(self):
        """Test multiplxer cx-gate circuits compiling to backend default basis_gates."""
        backend = self.backend()
        shots = 100
        circuits = ref_multiplexer.multiplexer_cx_gate_circuits_deterministic(final_measure=True)
        targets = ref_multiplexer.multiplexer_cx_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_multiplexer_cx_gate_nondeterministic(self):
        """Test multiplexer cx-gate circuits compiling to backend default basis_gates."""
        backend = self.backend()
        shots = 4000
        circuits = ref_multiplexer.multiplexer_cx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_multiplexer.multiplexer_cx_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test multiplexer-gate
    # ---------------------------------------------------------------------
    def test_multiplexer_cxx_gate_deterministic(self):
        """Test multiplexer-gate gate circuits"""
        backend = self.backend()
        shots = 100
        circuits = ref_multiplexer.multiplexer_ccx_gate_circuits_deterministic(final_measure=True)
        targets = ref_multiplexer.multiplexer_ccx_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_multiplexer_cxx_gate_nondeterministic(self):
        """Test multiplexer ccx-gate gate circuits"""
        backend = self.backend()
        shots = 4000
        circuits = ref_multiplexer.multiplexer_ccx_gate_circuits_nondeterministic(
            final_measure=True
        )
        targets = ref_multiplexer.multiplexer_ccx_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_multiplexer_without_control_qubits(self):
        """Test multiplexer without control qubits"""
        backend = self.backend()
        shots = 4000
        circuits = ref_multiplexer.multiplexer_no_control_qubits(final_measure=True)
        target_circuits = transpile(circuits, basis_gates=["u", "measure"])
        result = backend.run(circuits, shots=shots).result()
        counts = [result.get_counts(circuit) for circuit in circuits]
        target_results = backend.run(target_circuits, shots=shots).result()
        targets = [target_results.get_counts(target_circuit) for target_circuit in target_circuits]
        self.assertSuccess(result)
        for actual, target in zip(counts, targets):
            self.assertDictAlmostEqual(actual, target, delta=0.05 * shots)

    def test_state_preparation_inverse_with_empty_multiplexer_params(self):
        """Test a definition-backed multiplexer without matrix parameters."""
        backend = self.backend()
        circuit = QuantumCircuit(1)
        circuit.append(StatePreparation([0.5, math.sqrt(0.75)]).inverse(), [0])
        circuit.save_statevector()

        circuit = transpile(circuit, backend)
        self.assertIn(
            ("multiplexer", 0), [(op.operation.name, len(op.operation.params)) for op in circuit]
        )

        result = backend.run(circuit).result()
        self.assertSuccess(result)
        np.testing.assert_allclose(result.get_statevector(), [0.5, -math.sqrt(0.75)])

    def test_empty_multiplexer_params_inside_control_flow(self):
        """Test a definition-backed multiplexer in a control-flow body."""
        backend = self.backend()
        circuit = QuantumCircuit(1, 1)
        with circuit.if_test((circuit.clbits[0], False)):
            circuit.append(StatePreparation([0.5, math.sqrt(0.75)]).inverse(), [0])
        circuit.save_statevector()

        circuit = transpile(circuit, backend)
        compiled = AerCompiler().compile(circuit)[0]
        self.assertNotIn(
            ("multiplexer", 0),
            [(op.operation.name, len(op.operation.params)) for op in compiled],
        )

        result = backend.run(circuit).result()
        self.assertSuccess(result)
        np.testing.assert_allclose(result.get_statevector(), [0.5, -math.sqrt(0.75)])

    def test_empty_multiplexer_is_rejected(self):
        """Test an empty native multiplexer raises instead of crashing."""
        circuit = QuantumCircuit(1)
        circuit.append(Gate("multiplexer", 1, []), [0])

        with self.assertRaisesRegex(ValueError, "multiplexer matrices cannot be empty"):
            assemble_circuit(circuit)
