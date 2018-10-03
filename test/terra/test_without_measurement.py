# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


# This file will be modified when the QCircuit interface is updated
# with new simulator commands (Issue #10).
# Thoughout, code segments that are to be removed in this change are surrounded with *** and !!!


'''
Test the final statevector in circuits whose simulation is deterministic,
i.e., contain no measurement or noise
'''


import test.terra.common as common
from itertools import repeat
import unittest
import numpy as np

from qiskit.qobj import Qobj
import qiskit

# ***
from qiskit_addon_qv import AerQvSimulator
# !!!  Replace with  from qiskit_addon_qv import AerQvProvider


class QvNoMeasurementTest(common.QiskitAerTestCase):
    '''
    Test the final statevector in circuits whose simulation is deterministic,
    i.e., contain no measurement or noise
    '''

    def setUp(self):
        # ***
        self.backend_qv = AerQvSimulator()
        # !!!  Replace with register(provider_class=AerQvProvider)
        # Restricted to 2 qubits until Issue #46 is solved
        self._number_of_qubits = 2


    def get_qv_snapshot(self, circuit):
        """Run the qv simulator and return the final state"""

        exception_msg = self.generate_circuit_exception_msg(circuit)

        # ***
        qobj_dict_qv = qiskit.compile(circuit, backend=self.backend_qv, shots=1).as_dict()
        qobj_dict_qv['experiments'][0]['instructions'].append({'name': 'snapshot',
                                                               'type': 'state', 'label': 'final'})
        qobj_qv = Qobj.from_dict(qobj_dict_qv)
        result_qv = self.backend_qv.run(qobj_qv).result()
        # !!!  Replace with
        #      result_qv = qiskit.execute(circuit, backend='local_qv_simulator').result()
        self.assertEqual(result_qv.get_status(), 'COMPLETED', msg=exception_msg)
        # ***
        vector_qv = result_qv.get_snapshots()['state']['final'][0]
        # !!!  Replace with vector_qv = result_qv.get_state_snapshot(slot='final')

        return vector_qv


    def get_py_snapshot(self, circuit):
        """Run the Python simulator and return the final state"""

        exception_msg = self.generate_circuit_exception_msg(circuit)

        result_py = qiskit.execute(circuit, backend='local_statevector_simulator_py').result()
        self.assertEqual(result_py.get_status(), 'COMPLETED', msg=exception_msg)
        vector_py = result_py.get_statevector()

        return vector_py


    def single_circuit_test(self, circuit):
        '''
        Test the final statevector in circuits whose simulation is deterministic,
        i.e., contain no measurement or noise
        '''

        # ***
        # !!!  Add circuit.state_snapshot('final')

        exception_msg = self.generate_circuit_exception_msg(circuit)

        vector_qv = self.get_qv_snapshot(circuit)
        vector_py = self.get_py_snapshot(circuit)

        # Verify the same statevector
        # for the Python and Aer simulators
        for qv_amplitude, py_amplitude in zip(vector_qv, vector_py):
            self.assertAlmostEqual(qv_amplitude, py_amplitude, msg=exception_msg)


    def test_random_circuits(self):
        '''
        Test the simulator for 30 random circuits
        '''

        for _ in repeat(None, 30):
            circuit = common.generate_random_circuit(6, 4,
                                                     ['u1', 'u2', 'u3', 'iden', 'x',
                                                      'y', 'z', 'h', 's', 'sdg',
                                                      't', 'tdg', 'cx', 'cz'])
            self.single_circuit_test(circuit)


    def test_bell(self):
        '''
        Test the simulator for a circuit that generates the Bell state
        '''

        qr = qiskit.QuantumRegister(2)
        circuit = qiskit.QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        self.single_circuit_test(circuit)


if __name__ == '__main__':
    unittest.main()
