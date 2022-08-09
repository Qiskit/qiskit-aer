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
"""
Integration Tests for Parameterized Qobj execution, testing qasm_simulator,
statevector_simulator, and expectation value snapshots.
"""

import unittest
from math import pi
import numpy as np

from test.terra import common
from qiskit.providers import JobStatus, JobError

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.aer import AerSimulator

class TestIBMQSemantics(common.QiskitAerTestCase):
    """Test ibmq_semantics option"""

    def test_jobstatus_error(self):
        """Test JobStatus.Error is set when simulation fails"""

        backend = AerSimulator(method="stabilizer")

        circ = QuantumCircuit(1, 1)
        circ.t(0)
        circ.measure(0, 0)

        job = backend.run(circ, ibmq_semantics=True)

        try:
            job.result()
        except Exception as e:
            pass

        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_parameter_binds_convert(self):
        """Test paramter binding conversion"""

        backend = AerSimulator()

        p = Parameter('a')
        circ = QuantumCircuit(1, 1)
        circ.ry(p, 0)
        circ.measure(0, 0)

        result_aer = backend.run(circ, parameter_binds=[{p:[1.0]}], seed_simulator=0).result()
        result_ibmq = backend.run(circ, parameter_binds=[{p:1.0}], ibmq_semantics=True, seed_simulator=0).result()
        self.assertEqual(len(result_aer.results), len(result_ibmq.results))
        for idx in range(len(result_ibmq.results)):
            self.assertEqual(result_aer.get_counts(idx), result_ibmq.get_counts(idx))

        result_aer = backend.run([circ, circ], parameter_binds=[{p:[1.0, 2.0]}, {p:[1.0, 2.0]}], seed_simulator=0).result()
        result_ibmq = backend.run([circ, circ], parameter_binds=[{p:1.0}, {p:2.0}], ibmq_semantics=True, seed_simulator=0).result()

        self.assertEqual(len(result_aer.results), len(result_ibmq.results))
        for idx in range(len(result_ibmq.results)):
            self.assertEqual(result_aer.get_counts(idx), result_ibmq.get_counts(idx))


if __name__ == '__main__':
    unittest.main()
