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

# pylint: disable=arguments-differ

import unittest
import logging
from ddt import ddt, data

from qiskit import transpile, assemble
from qiskit.providers import JobError
from qiskit.circuit.random import random_circuit
from qiskit_aer.jobs import split_qobj
from test.terra.reference.ref_save_expval import (
    save_expval_circuits,
    save_expval_circuit_parameterized,
)
from test.terra.backends.simulator_test_case import SimulatorTestCase


@ddt
class TestJobSplitting(SimulatorTestCase):
    """Test job splitting option"""

    @staticmethod
    def parameterized_circuits():
        """Return ParameterizedQobj for settings."""
        pcirc1, param1 = save_expval_circuit_parameterized(
            pershot=False,
            measure=True,
            snapshot=False,
        )
        circuits2to4 = save_expval_circuits(
            pauli=True,
            skip_measure=False,
            pershot=False,
        )
        pcirc2, param2 = save_expval_circuit_parameterized(
            pershot=False,
            measure=True,
            snapshot=False,
        )
        circuits = [pcirc1] + circuits2to4 + [pcirc2]
        params = [param1, [], [], [], param2]
        return circuits, params

    def split_compare(self, circs, backend, parameterizations=None):
        """Qobj split test"""
        qobj = assemble(circs, parameterizations=parameterizations, qobj_id="testing")
        if parameterizations:
            qobjs = [
                assemble(c, parameterizations=[p], qobj_id="testing")
                for (c, p) in zip(circs, parameterizations)
            ]
        else:
            qobjs = [assemble(c, qobj_id="testing") for c in circs]

        test_qobjs = split_qobj(qobj, max_size=1, qobj_id="testing")
        self.assertEqual(len(test_qobjs[0]), len(qobjs))
        for ref, test in zip(qobjs, test_qobjs[0]):
            self.assertEqual(ref, test)

    def add_custom_instruction(self):
        backend = self.backend(max_job_size=1, max_shot_size=1)
        circ = random_circuit(num_qubits=2, depth=4)
        circ.save_statevector()
        circ = transpile(circ, backend)
        qobj = assemble(circ)
        split_qobj(qobj, max_size=1, max_shot_size=1, qobj_id="testing")

    def test_split(self):
        """Circuits split test"""
        backend = self.backend(max_job_size=1)
        circs = [random_circuit(num_qubits=2, depth=4, measure=True, seed=i) for i in range(2)]
        circs = transpile(circs, backend)
        self.split_compare(circs, backend)

    def test_parameterized_split(self):
        """Parameterized circuits split test"""
        backend = self.backend(max_job_size=1)
        circs, params = self.parameterized_circuits()
        self.split_compare(circs, backend, parameterizations=params)

    def test_custom_instruction_error(self):
        with self.assertRaises(JobError):
            self.add_custom_instruction()


if __name__ == "__main__":
    unittest.main()
