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
QasmSimulator Integration Tests for SaveState instruction
"""

import numpy as np

from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.library import (
    SaveStatevector, SaveDensityMatrix, SaveStabilizer,
    SaveMatrixProductState)


class QasmSaveStateTests:
    """QasmSimulator SaveState instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_save_state(self):
        """Test save_amplitudes instruction"""

        REFERENCE_SAVE = {
            'automatic': SaveStabilizer,
            'stabilizer': SaveStabilizer,
            'statevector': SaveStatevector,
            'statevector_gpu': SaveStatevector,
            'statevector_thrust': SaveStatevector,
            'density_matrix': SaveDensityMatrix,
            'density_matrix_gpu': SaveDensityMatrix,
            'density_matrix_thrust': SaveDensityMatrix,
            'matrix_product_state': SaveMatrixProductState
        }
        REFERENCE_LABEL = {
            'automatic': 'stabilizer',
            'stabilizer': 'stabilizer',
            'statevector': 'statevector',
            'statevector_gpu': 'statevector',
            'statevector_thrust': 'statevector',
            'density_matrix': 'density_matrix',
            'density_matrix_gpu': 'density_matrix',
            'density_matrix_thrust': 'density_matrix',
            'matrix_product_state': 'matrix_product_state'
        }

        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')

        if method in REFERENCE_SAVE:

            # Stabilizer test circuit
            num_qubits = 4
            target_instr = REFERENCE_SAVE[method](num_qubits, label='target')
            circ = QuantumCircuit(num_qubits)
            circ.h(0)
            for i in range(1, num_qubits):
                circ.cx(i - 1, i)
            circ.save_state()
            circ.append(target_instr, range(num_qubits))
            label = REFERENCE_LABEL[method]

            # Run
            qobj = assemble(circ, self.SIMULATOR)
            result = self.SIMULATOR.run(qobj, **opts).result()
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            self.assertIn('target', data)
            value = data[label]
            target = data['target']
            if method == 'matrix_product_state':
                for val, targ in zip(value[0], target[0]):
                    self.assertTrue(np.allclose(val, targ))
                for val, targ in zip(value[1], target[1]):
                    self.assertTrue(np.allclose(val, targ))
            else:
                self.assertTrue(np.all(value == target))
