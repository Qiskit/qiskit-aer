# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
StatevectorSimulator Integration Tests
"""

from ddt import ddt
from numpy import exp, pi

from test.terra.reference import ref_measure
from test.terra.reference import ref_reset
from test.terra.reference import ref_initialize
from test.terra.reference import ref_conditionals
from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_unitary_gate
from test.terra.reference import ref_diagonal_gate

from qiskit import transpile
from qiskit_aer import StatevectorSimulator, AerError
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_devices


@ddt
class TestStatevectorSimulator(SimulatorTestCase):
    """StatevectorSimulator tests."""

    BACKEND = StatevectorSimulator

    # ---------------------------------------------------------------------
    # Test initialize
    # ---------------------------------------------------------------------

    @supported_devices
    def test_initialize_1(self, device):
        """Test StatevectorSimulator initialize"""
        backend = self.backend(device=device)
        circuits = ref_initialize.initialize_circuits_1(final_measure=False)
        targets = ref_initialize.initialize_statevector_1()
        result = backend.run(circuits).result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_initialize_2(self, device):
        """Test StatevectorSimulator initialize"""
        backend = self.backend(device=device)
        circuits = ref_initialize.initialize_circuits_2(final_measure=False)
        targets = ref_initialize.initialize_statevector_2()
        result = backend.run(circuits).result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    @supported_devices
    def test_reset_deterministic(self, device):
        """Test StatevectorSimulator reset with for circuits with deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        backend = self.backend(device=device)
        circuits = ref_reset.reset_circuits_deterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_reset.reset_statevector_deterministic()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_reset_nondeterministic(self, device):
        """Test StatevectorSimulator reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        backend = self.backend(device=device)
        circuits = ref_reset.reset_circuits_nondeterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_reset.reset_statevector_nondeterministic()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    @supported_devices
    def test_measure(self, device):
        """Test StatevectorSimulator measure with deterministic counts"""
        backend = self.backend(device=device)
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_measure.measure_statevector_deterministic()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_devices
    def test_conditional_gate_1bit(self, device):
        """Test conditional gates on 1-bit conditional register."""
        backend = self.backend(device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_1bit()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_unitary_1bit(self, device):
        """Test conditional unitaries on 1-bit conditional register."""
        backend = self.backend(device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=False, conditional_type="unitary"
        )
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_1bit()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_gate_2bit(self, device):
        """Test conditional gates on 2-bit conditional register."""
        backend = self.backend(device=device)
        circuits = ref_conditionals.conditional_circuits_2bit(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_2bit()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_unitary_2bit(self, device):
        """Test conditional unitary on 2-bit conditional register."""
        backend = self.backend(device=device)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=False, conditional_type="unitary"
        )
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_2bit()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_gate_64bit(self, device):
        """Test conditional gates on 64-bit conditional register."""
        backend = self.backend(device=device)
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=False, conditional_type="gate"
        )
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_unitary_64bit(self, device):
        """Test conditional unitary on 64-bit conditional register."""
        backend = self.backend(device=device)
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=False, conditional_type="unitary"
        )
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_gate_132bit(self, device):
        """Test conditional gates on 132-bit conditional register."""
        backend = self.backend(device=device)
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=False, conditional_type="gate"
        )
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_conditional_unitary_132bit(self, device):
        """Test conditional unitary on 132-bit conditional register."""
        backend = self.backend(device=device)
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=False, conditional_type="unitary"
        )
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

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
        targets = ref_unitary_gate.unitary_gate_statevector_deterministic()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_unitary_gate_circuit_run(self, device):
        """Test simulation with unitary gate circuit instructions."""
        backend = self.backend(device=device)
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_unitary_gate.unitary_gate_statevector_deterministic()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    @supported_devices
    def test_diagonal_gate(self, device):
        """Test simulation with diagonal gate circuit instructions."""
        backend = self.backend(device=device)
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(final_measure=False)
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        targets = ref_diagonal_gate.diagonal_gate_statevector_deterministic()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test global phase
    # ---------------------------------------------------------------------

    @supported_devices
    def test_qobj_global_phase(self, device):
        """Test qobj global phase."""
        backend = self.backend(device=device)
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_nondeterministic()
        for iter, circuit in enumerate(circuits):
            global_phase = (-1) ** iter * (pi / 4)
            circuit.global_phase += global_phase
            targets[iter] = exp(1j * global_phase) * targets[iter]
        circuits = transpile(circuits, backend, optimization_level=1)
        result = backend.run(circuits, shots=1).result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets, ignore_phase=False)

    # ---------------------------------------------------------------------
    # Test legacy methods
    # ---------------------------------------------------------------------

    @supported_devices
    def test_legacy_method(self, device):
        """Test legacy device method options."""
        backend = self.backend()
        legacy_method = f"statevector_{device.lower()}"
        with self.assertWarns(DeprecationWarning):
            backend.set_options(method=legacy_method)
        self.assertEqual(backend.options.device, device)

    def test_unsupported_methods(self):
        """Test unsupported AerSimulator method raises AerError."""
        backend = self.backend()
        with self.assertWarns(DeprecationWarning):
            self.assertRaises(AerError, backend.set_options, method="automatic")
