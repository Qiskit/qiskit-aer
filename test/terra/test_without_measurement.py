# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
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
# ***
from qiskit_addon_qv import AerQvSimulator
# !!!  Replace with  from qiskit_addon_qv import AerQvProvider


class QvNoMeasurementTest(common.QiskitAerTestCase):
    """Test the final statevector in circuits whose simulation is deterministic, i.e., contain no measurement or noise"""

    def setUp(self):        
        # ***
        self.backend_qv = AerQvSimulator()
        # !!!  Replace with register(provider_class=AerQvProvider)
        # Restricted to 2 qubits until Issue #46 is solved
        self._number_of_qubits = 2
        

    def single_circuit_test(self, circuit):     
        """Test the final statevector in circuits whose simulation is deterministic, i.e., contain no measurement or noise"""

        # ***
        # !!!  Add circuit.state_snapshot('final')

        exception_msg = self.generate_circuit_exception_msg(circuit)

        # ***
        qobj_dict_qv = compile(circuit, backend=self.backend_qv, shots=1).as_dict()
        qobj_dict_qv['experiments'][0]['instructions'].append({'name': 'snapshot', 'type': 'state', 'label': 'final'})
        qobj_qv = Qobj.from_dict(qobj_dict_qv)
        result_qv = self.backend_qv.run(qobj_qv).result()
        # !!!  Replace with  result_qv = execute(circuit, backend='local_qv_simulator').result()        
        self.assertEqual(result_qv.get_status(), 'COMPLETED', msg=exception_msg)
        # ***
        vector_qv_raw = result_qv.get_snapshots()['state']['final']
        # !!!  Replace with vector_qv = result_qv.get_state_snapshot(slot='final')

        # The following lines are needed because the statevector represents complex numbers by a pair of real numbers.
        # See Issue #46.
        vector_qv = [np.complex(real, imag) for [real, imag] in vector_qv_raw[0]]
        
        # Compare with the result of the Python simulator
        result_py = execute(circuit, backend='local_statevector_simulator_py').result()
        self.assertEqual(result_py.get_status(), 'COMPLETED', msg=exception_msg)
        vector_py = result_py.get_statevector()

        # Verify the same statevector
        # for the Python and Aer simulators
        for a, b in zip(vector_qv, vector_py):
            self.assertAlmostEqual(a, b, msg=exception_msg)


    def test_random_circuits(self):
        for _ in repeat(None, 30):
            circuit = common.generate_random_circuit(6, 4, ['u1', 'u2', 'u3', 'iden', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'cx', 'cz'])
            self.single_circuit_test(circuit)
        

if __name__ == '__main__':
    unittest.main()
