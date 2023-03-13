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

from qiskit import transpile
from test.terra.reference import ref_multiplexer
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


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
