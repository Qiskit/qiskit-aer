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
import numpy as np

from test.terra import common

from qiskit import assemble
from test.terra.reference.ref_save_expval import (
    save_expval_circuits,
    save_expval_counts,
    save_expval_labels,
    save_expval_pre_meas_values,
    save_expval_circuit_parameterized,
    save_expval_final_statevecs,
)
from qiskit.providers.aer import AerSimulator


class TestParameterizedQobj(common.QiskitAerTestCase):
    """Parameterized Qobj extension tests"""

    BACKEND_OPTS = {
        "seed_simulator": 2113
    }

    @staticmethod
    def parameterized_qobj(
        backend, shots=1000, measure=True, snapshot=False, save_state=False,
    ):
        """Return ParameterizedQobj for settings."""
        pershot = shots == 1
        pcirc1, param1 = save_expval_circuit_parameterized(
            pershot=pershot, measure=measure, snapshot=snapshot,
        )
        circuits2to4 = save_expval_circuits(
            pauli=True, skip_measure=(not measure), pershot=pershot,
        )
        pcirc2, param2 = save_expval_circuit_parameterized(
            pershot=pershot, measure=measure, snapshot=snapshot,
        )
        circuits = [pcirc1] + circuits2to4 + [pcirc2]
        if save_state:
            for circuit in circuits:
                circuit.save_statevector(pershot=pershot)
        params = [param1, [], [], [], param2]
        qobj = assemble(circuits,
                        backend=backend,
                        shots=shots,
                        parameterizations=params)
        return qobj

    def test_parameterized_qobj_qasm_save_expval(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        shots = 1000
        labels = save_expval_labels() * 3
        counts_targets = save_expval_counts(shots) * 3
        value_targets = save_expval_pre_meas_values() * 3

        backend = AerSimulator()
        qobj = self.parameterized_qobj(backend=backend,
                                       shots=1000,
                                       measure=True,
                                       snapshot=True)
        self.assertIn('parameterizations', qobj.to_dict()['config'])
        job = backend.run(qobj, **self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        num_circs = len(result.to_dict()['results'])
        self.assertTrue(success)
        self.compare_counts(result,
                            range(num_circs),
                            counts_targets,
                            delta=0.1 * shots)
        # Check snapshots
        for j, target in enumerate(value_targets):
            data = result.data(j)
            for label in labels:
                self.assertAlmostEqual(data[label], target[label], delta=1e-7)

    def test_parameterized_qobj_statevector(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        statevec_targets = save_expval_final_statevecs() * 3

        backend = AerSimulator(method="statevector")
        qobj = self.parameterized_qobj(
            backend=backend, measure=False, snapshot=False, save_state=True,
        )
        self.assertIn('parameterizations', qobj.to_dict()['config'])
        job = backend.run(qobj, **self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        num_circs = len(result.to_dict()['results'])
        self.assertTrue(success)

        for j in range(num_circs):
            statevector = result.get_statevector(j)
            np.testing.assert_array_almost_equal(statevector, statevec_targets[j].data, decimal=7)

if __name__ == '__main__':
    unittest.main()
