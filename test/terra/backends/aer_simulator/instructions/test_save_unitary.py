# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QasmSimulator Integration Tests for SaveStatevector instruction
"""
from ddt import ddt
import qiskit.quantum_info as qi
from qiskit import transpile
from qiskit.circuit.library import QuantumVolume
from test.terra.backends.aer_simulator.aer_simulator_test_case import (
    AerSimulatorTestCase, supported_methods)


@ddt
class TestSaveUnitary(AerSimulatorTestCase):
    """SaveUnitary instruction tests."""

    @supported_methods(["automatic", "unitary"])
    def test_save_unitary(self, method, device):
        """Test save unitary instruction"""
        backend = self.backend(method=method, device=device)

        # Test circuit
        SEED = 5426
        circ = QuantumVolume(3, seed=SEED)

        # Target unitary
        target = qi.Operator(circ)

        # Add save to circuit
        label = 'state'
        circ.save_unitary(label=label)

        # Run
        result = backend.run(transpile(
            circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.Operator(simdata[label])
        self.assertEqual(value, target)
