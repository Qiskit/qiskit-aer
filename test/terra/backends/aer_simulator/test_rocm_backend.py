# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator ROCm Integration Tests
"""
import os
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


class TestROCmBackend(SimulatorTestCase):
    """AerSimulator tests for ROCm backend"""

    @pytest.mark.skipif(
        "AER_THRUST_BACKEND" not in os.environ or os.environ["AER_THRUST_BACKEND"] != "ROCM",
        reason="Skipping ROCm-specific tests",
    )
    @supported_methods(["statevector"])
    def test_rocm_backend_initialization(self, method, device):
        """Test if Qiskit Aer initializes with ROCm backend."""
        backend = self.backend(method=method, device=device)
        self.assertEqual(backend.backend_name, "aer_simulator")

    @pytest.mark.skipif(
        "AER_THRUST_BACKEND" not in os.environ or os.environ["AER_THRUST_BACKEND"] != "ROCM",
        reason="Skipping ROCm-specific tests",
    )
    @supported_methods(["statevector"])
    def test_rocm_statevector(self, method, device):
        """Test a simple circuit on the ROCm backend."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        backend = self.backend(method=method, device=device)
        result = backend.run(qc).result()
        statevector = result.get_statevector()

        self.assertEqual(len(statevector), 2**2)

    @pytest.mark.skipif(
        "AER_THRUST_BACKEND" not in os.environ or os.environ["AER_THRUST_BACKEND"] != "ROCM",
        reason="Skipping ROCm-specific tests",
    )
    @supported_methods(["statevector"])
    def test_rocm_expectation_value(self, method, device):
        """Test expectation value calculation on ROCm."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        backend = self.backend(method=method, device=device)
        result = backend.run(qc).result()
        statevector = result.get_statevector()

        # Check probability of |00⟩ state (around 50% for H + CNOT)
        prob_00 = abs(statevector[0]) ** 2
        self.assertTrue(0.49 < prob_00 < 0.51)
