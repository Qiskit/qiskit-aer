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
        backend_opts = {}
        circ = QuantumCircuit(4)
        circ.append(qi.random_unitary(2 ** 4), [0, 1, 2, 3])
        circ.append(Snapshot("final", "unitary", 4), [0, 1, 2, 3])
        qobj = assemble(circ, backend=backend)
        aer_input = backend._format_qobj(qobj, self.BACKEND_OPTS, None)
        aer_output = backend._controller(aer_input)
        self.assertIsInstance(aer_output, dict)
        self.assertTrue(aer_output['success'])
        snaps = aer_output['results'][0]['data']['snapshots']['unitary']['final']
        self.assertTrue(all([isinstance(arr, np.ndarray) for arr in snaps]))

if __name__ == '__main__':
    unittest.main()
