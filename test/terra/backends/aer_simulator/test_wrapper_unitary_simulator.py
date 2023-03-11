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
UnitarySimulator Integration Tests
"""

from ddt import ddt
from numpy import exp, pi

from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_unitary_gate
from test.terra.reference import ref_diagonal_gate

from qiskit import transpile
from qiskit_aer import UnitarySimulator, AerError
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_devices


@ddt
class TestUnitarySimulator(SimulatorTestCase):
    """UnitarySimulator tests."""

    BACKEND = UnitarySimulator

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------
    @supported_devices
    def test_unitary_gate(self, device):
        """Test simulation with unitary gate circuit instructions."""
        backend = self.backend(device=device)
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_unitary_gate.unitary_gate_unitary_deterministic()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets)

    @supported_devices
    def test_unitary_gate_circuit_run(self, device):
        """Test simulation with unitary gate circuit instructions."""
        backend = self.backend(device=device)
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_unitary_gate.unitary_gate_unitary_deterministic()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets)

    @supported_devices
    def test_diagonal_gate(self, device):
        """Test simulation with diagonal gate circuit instructions."""
        backend = self.backend(device=device)
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_diagonal_gate.diagonal_gate_unitary_deterministic()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test global phase
    # ---------------------------------------------------------------------

    @supported_devices
    def test_qobj_global_phase(self, device):
        """Test qobj global phase."""
        backend = self.backend(device=device)
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_1q_clifford.h_gate_unitary_nondeterministic()
        for iter, circuit in enumerate(circuits):
            global_phase = (-1) ** iter * (pi / 4)
            circuit.global_phase += global_phase
            targets[iter] = exp(1j * global_phase) * targets[iter]
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets, ignore_phase=False)

    # ---------------------------------------------------------------------
    # Test legacy methods
    # ---------------------------------------------------------------------
    @supported_devices
    def test_legacy_method(self, device):
        """Test legacy device method options."""
        backend = self.backend()
        legacy_method = f"unitary_{device.lower()}"
        with self.assertWarns(DeprecationWarning):
            backend.set_options(method=legacy_method)
        self.assertEqual(backend.options.device, device)

    def test_unsupported_methods(self):
        """Test unsupported AerSimulator method raises AerError."""
        backend = self.backend()
        with self.assertWarns(DeprecationWarning):
            self.assertRaises(AerError, backend.set_options, method="automatic")
