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

from qiskit import QuantumCircuit, assemble
from qiskit.extensions.exceptions import ExtensionError
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import UnitarySimulator
from qiskit.providers.aer.extensions import Snapshot

from qiskit.providers.aer.extensions.snapshot_density_matrix import SnapshotDensityMatrix
from qiskit.providers.aer.extensions.snapshot_expectation_value import SnapshotExpectationValue
from qiskit.providers.aer.extensions.snapshot_probabilities import SnapshotProbabilities
import qiskit.quantum_info as qi

class TestSnapshotsExtension(unittest.TestCase):
    """Snapshot extension tests"""

    def test_density_mat_snap(self):
        """Test density matrix snaps on a random circuit"""
        backend = QasmSimulator()
        backend_opts = {}
        circ = QuantumCircuit(4)
        circ.append(qi.random_unitary(2 ** 4), [0, 1, 2, 3])
        circ.snapshot_density_matrix('dmat_snap')
        qobj = assemble(circ, backend=backend)
        result = backend.run(qobj, backend_options=backend_opts).result()
        self.assertTrue(result.success)

    def test_unitary_snap(self):
        """Test density matrix snaps on a random circuit"""
        backend = UnitarySimulator()
        backend_opts = {}
        circ = QuantumCircuit(4)
        circ.append(qi.random_unitary(2 ** 4), [0, 1, 2, 3])
        circ.append(Snapshot("final", "unitary", 4), [0, 1, 2, 3])
        qobj = assemble(circ, backend=backend)
        result = backend.run(qobj, backend_options=backend_opts).result()
        self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
