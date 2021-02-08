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


import unittest
import numpy as np

from qiskit import QuantumCircuit, assemble
from qiskit.extensions.exceptions import ExtensionError
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import UnitarySimulator
from qiskit.providers.aer.extensions import Snapshot

import qiskit.quantum_info as qi

class UnitarySnapshotTests:
    """Unitary Snapshot tests"""

    def test_unitary_snap(self):
        """Test Unitary matrix snaps on a random circuit"""
        backend = UnitarySimulator()
        target = qi.random_unitary(2 ** 4, seed=111)
        circ = QuantumCircuit(4)
        circ.append(target, [0, 1, 2, 3])
        circ.append(Snapshot("final", "unitary", 4), [0, 1, 2, 3])
        qobj = assemble(circ, backend=backend, shots=1)
        job = backend.run(qobj)
        result = job.result()
        self.assertSuccess(result)
        snaps = result.data(0)['snapshots']['unitary']['final']
        for arr in snaps:
            self.assertTrue(isinstance(arr, np.ndarray))
            self.assertEqual(qi.Operator(arr), target)


if __name__ == '__main__':
    unittest.main()
