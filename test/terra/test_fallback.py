# the LICENSE.txt file in the root directory of this source tree.


# This file will be modified when the QCircuit interface is updated with new simulator commands (Issue #10).
# Thoughout, code segments that are to be removed in this change are surrounded with *** and !!!


import test.terra.common as common
import unittest
import numpy as np
from itertools import repeat

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit,
                    register, compile, execute)
from qiskit.qobj import Qobj
from qiskit_addon_qv import AerQvSimulator
from qiskit import Aer


class TestFallback(common.QiskitAerTestCase):
    """Test the final statevector in circuits whose simulation is deterministic, i.e., contain no measurement or noise"""

    def setUp(self):
        # ***
        self.backend_qv = AerQvSimulator()
        # TODO Replace with register(provider_class=AerQvProvider)
        # Restricted to 2 qubits until Issue #46 is solved
        self._number_of_qubits = 2

    def test_qv_fallback(self):
        """ Test that the fallback for QV simulator returns the same results."""
        for _ in repeat(None, 30):
            circuit = common.generate_random_circuit(6, 4,
                ['u1', 'u2', 'u3', 'iden', 'x', 'y', 'z', 'h', 's', 'sdg', 't',
                'tdg', 'cx', 'cz'])
            """Test the final statevector in circuits whose simulation is deterministic, i.e., contain no measurement or noise"""

            # TODO  Add circuit.state_snapshot('final')
            qobj_dict_qv = compile(circuit, backend=self.backend_qv, shots=1).as_dict()
            qobj_dict_qv['experiments'][0]['instructions'].append(
                {'name': 'snapshot',
                'type': 'state',
                'label': 'final'}
            )
            qobj_qv = Qobj.from_dict(qobj_dict_qv)
            result_qv = self.backend_qv.run(qobj_qv).result()
            # TODO Replace with  result_qv = execute(circuit, backend='local_qv_simulator').result()
            self.assertEqual(result_qv.get_status(), 'COMPLETED')
            # ***
            vector_qv = result_qv.get_snapshots()['state']['final'][0]
            # TODO Replace with vector_qv = result_qv.get_state__snapshot(slot='final')
            # Compare with the result of the fallback Python simulator
            result = execute(circuit, Aer.get_backend('statevector_simulator_py'))
            result_py = result.result()
            self.assertEqual(result_py.get_status(), 'COMPLETED')
            vector_py = result_py.get_statevector()

            # Verify the same statevector
            # for the Python and Aer simulators
            for qv_amplitude, py_amplitude in zip(vector_qv, vector_py):
                self.assertAlmostEqual(qv_amplitude, py_amplitude,
                    msg=lambda circuit: 'Error on circuit: ' + circuit.qasm())


if __name__ == '__main__':
    unittest.main()
