# the LICENSE.txt file in the root directory of this source tree.


# This file will be modified when the QCircuit interface is updated
# with new simulator commands (Issue #10).
# Thoughout, code segments that are to be removed in this change are surrounded with *** and TODO


import test.terra.common as common
import unittest
import numpy as np
from itertools import repeat

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit,
                    compile, execute)
from qiskit.qobj import Qobj
from qiskit_aer.backends import QasmSimulator
from qiskit import Aer


class TestFallback(common.QiskitAerTestCase):
    """Test the final statevector in circuits whose simulation is deterministic,
    i.e., contain no measurement or noise"""

    def setUp(self):
        # ***
        self.backend = QasmSimulator()
        # Restricted to 2 qubits until Issue #46 is solved
        # The test still fails if all gates operate on a single qubit
        self._number_of_qubits = 2

    def single_circuit_comparison(self, circuit):
        """ Test that the fallback for QV simulator
        returns the same results for the given circuit."""

        # TODO  Add circuit.state_snapshot('final')
        qobj_dict_qv = compile(circuit, backend=self.backend, shots=1).as_dict()
        qobj_dict_qv['experiments'][0]['instructions'].append(
            {'name': 'snapshot',
             'type': 'statevector',
             'label': 'final'}
        )
        qobj_qv = Qobj.from_dict(qobj_dict_qv)
        result_qv = self.backend.run(qobj_qv).result()
        # TODO Replace with  result_qv = execute(circuit, backend='local_qv_simulator').result()
        self.assertEqual(result_qv.get_status(), 'COMPLETED')
        # ***
        vector_qv = result_qv.get_snapshots()['statevector']['final'][0]
        # TODO Replace with vector_qv = result_qv.get_state_snapshot(slot='final')

        # Compare with the result of the fallback Python simulator
        result = execute(circuit, Aer.get_backend('statevector_simulator_py'))
        result_py = result.result()
        self.assertEqual(result_py.get_status(), 'COMPLETED')
        vector_py = result_py.get_statevector()

        # Verify the same statevector
        # for the Python and Aer simulators
        # TODO consider comparing the vectors using fidelity
        self.assertAlmostEqual(np.linalg.norm(vector_qv - vector_py), 0.,
                               msg=(lambda circ: 'Error on circuit: ' + circ.qasm())(circuit))


    def test_qv_fallback(self):
        """ Test that the fallback for QV simulator returns the same results."""
        for _ in repeat(None, 30):
            circuit = common.generate_random_circuit(self._number_of_qubits, 4,
                                                     ['u1', 'u2', 'u3', 'iden', 'x', 'y', 'z', 'h', 's', 'sdg', 't',
                                                      'tdg', 'cx', 'cz'])
            self.single_circuit_comparison(circuit)


if __name__ == '__main__':
    unittest.main()
