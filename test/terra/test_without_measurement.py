# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from test.terra.common import QiskitQvTestCase

import unittest

from qiskit import (execute, QuantumRegister,
                    ClassicalRegister, QuantumCircuit, register)
from qiskit.extensions.simulator.snapshot import snapshot
from qiskit_addon_qv import AerQvProvider


class QvNoMeasurementTest(QiskitQvTestCase):
    """Test the final statevector in circuits whose simulation is deterministic, i.e., costain no measurement or noise"""

    def setUp(self):
        register(provider_class=AerQvProvider)
        self._number_of_qubits = 6

    def test_no_measurement(self):     
        """Test the final statevector in circuits whose simulation is deterministic, i.e., constain no measurement or noise"""

        # A very simple circuit,
        # to be replaced by random circuits covering all types of gates
        q = QuantumRegister(self._number_of_qubits)
        circuit = QuantumCircuit(q)
        circuit.h(q[0])
        circuit.cx(q[0], q[1])

        # Waiting for Issue #10 to be resolved
        #circuit.state_snapshot('0')

        result = execute(circuit, backend='local_qv_simulator').result()
        self.assertEqual(result.get_status(), 'COMPLETED')

        # Compare with the result of the Python simulator
        result_py = execute(circuit, backend='local_statevector_simulator_py').result()
        self.assertEqual(result_py.get_status(), 'COMPLETED')

        # Assuming that the Python simulator supports statevector,
        # and that it supports the same gates as the aer simulator:
        # Add lines to verify the same statevector
        # for the Python and aer simulators


if __name__ == '__main__':
    unittest.main()
