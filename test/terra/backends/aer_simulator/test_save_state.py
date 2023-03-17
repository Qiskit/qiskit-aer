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
Integration Tests for SaveState instruction
"""

import numpy as np
from ddt import ddt
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
from qiskit import QuantumCircuit, transpile
from qiskit_aer.library import (
    SaveStatevector,
    SaveDensityMatrix,
    SaveStabilizer,
    SaveMatrixProductState,
    SaveUnitary,
    SaveSuperOp,
)


@ddt
class TestSaveState(SimulatorTestCase):
    """Test instructions for saving simulator state."""

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "stabilizer",
            "matrix_product_state",
            "unitary",
            "superop",
            "tensor_network",
        ]
    )
    def test_save_state(self, method, device):
        """Test save_amplitudes instruction"""

        REFERENCE_SAVE = {
            "automatic": SaveStabilizer,
            "stabilizer": SaveStabilizer,
            "statevector": SaveStatevector,
            "density_matrix": SaveDensityMatrix,
            "matrix_product_state": SaveMatrixProductState,
            "unitary": SaveUnitary,
            "superop": SaveSuperOp,
            "tensor_network": SaveStatevector,
        }

        backend = self.backend(method=method, device=device)
        if method == "automatic":
            label = "stabilizer"
        else:
            label = method

        # Stabilizer test circuit
        num_qubits = 2
        target_instr = REFERENCE_SAVE[method](num_qubits, label="target")
        circ = QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(i - 1, i)
        circ.save_state()
        circ.append(target_instr, range(num_qubits))

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        self.assertIn("target", simdata)
        value = simdata[label]
        target = simdata["target"]
        if method == "matrix_product_state":
            for val, targ in zip(value[0], target[0]):
                self.assertTrue(np.allclose(val, targ))
            for val, targ in zip(value[1], target[1]):
                self.assertTrue(np.allclose(val, targ))
        else:
            self.assertEqual(value, target)

    @supported_methods(["statevector", "density_matrix"])
    def test_save_state_cache_blocking(self, method, device):
        """Test save_amplitudes instruction"""

        REFERENCE_SAVE = {
            "statevector": SaveStatevector,
            "density_matrix": SaveDensityMatrix,
        }

        backend = self.backend(
            method=method, device=device, blocking_qubits=2, max_parallel_threads=1
        )

        # Stabilizer test circuit
        num_qubits = 4
        target_instr = REFERENCE_SAVE[method](num_qubits, label="target")
        circ = QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(i - 1, i)
        circ.save_state()
        circ.append(target_instr, range(num_qubits))

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(method, simdata)
        self.assertIn("target", simdata)
        value = simdata[method]
        target = simdata["target"]
        self.assertEqual(value, target)
