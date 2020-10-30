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
import copy
import pickle
from multiprocessing import Pool

from qiskit import assemble, transpile, QuantumCircuit
from qiskit.providers.aer.backends import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.aer.backends.controller_wrappers import (qasm_controller_execute,
                                                               statevector_controller_execute,
                                                               unitary_controller_execute)
from qiskit.providers.aer.backends.backend_utils import LIBRARY_DIR
from test.terra.reference import ref_algorithms, ref_measure, ref_1q_clifford
from test.terra.common import QiskitAerTestCase


class TestControllerExecuteWrappers(QiskitAerTestCase):
    """Basic functionality tests for pybind-generated wrappers"""

    CFUNCS = [qasm_controller_execute(),
              statevector_controller_execute(),
              unitary_controller_execute()]

    def test_deepcopy(self):
        """Test that the functors are deepcopy-able."""
        for cfunc in self.CFUNCS:
            cahpy = copy.deepcopy(cfunc)

    def test_pickleable(self):
        """Test that the functors are pickle-able (directly)."""
        for cfunc in self.CFUNCS:
            bites = pickle.dumps(cfunc)
            cahpy = pickle.loads(bites)

    def _create_qobj(self, backend, noise_model=None):
        num_qubits = 2
        circuit = QuantumCircuit(num_qubits)
        circuit.x(list(range(num_qubits)))
        qobj = assemble(transpile(circuit, backend), backend)
        opts = {'max_parallel_threads': 1,
                'library_dir': LIBRARY_DIR}
        fqobj = backend._format_qobj(qobj, **opts, noise_model=noise_model)
        return fqobj.to_dict()

    def _map_and_test(self, cfunc, qobj):
        n = 2
        with Pool(processes=1) as p:
            rs = p.map(cfunc, [copy.deepcopy(qobj) for _ in range(n)])

        self.assertEqual(len(rs), n)
        for r in rs:
            self.assertTrue(r['success'])

    def test_mappable_qasm(self):
        """Test that the qasm controller can be mapped."""
        cfunc = qasm_controller_execute()
        sim = QasmSimulator()
        fqobj = self._create_qobj(sim)
        self._map_and_test(cfunc, fqobj)

    def test_mappable_statevector(self):
        """Test that the statevector controller can be mapped."""
        cfunc = statevector_controller_execute()
        sim = StatevectorSimulator()
        fqobj = self._create_qobj(sim)
        self._map_and_test(cfunc, fqobj)

    def test_mappable_unitary(self):
        """Test that the unitary controller can be mapped."""
        cfunc = unitary_controller_execute()
        sim = UnitarySimulator()
        fqobj = self._create_qobj(sim)
        self._map_and_test(cfunc, fqobj)


if __name__ == '__main__':
    unittest.main()
