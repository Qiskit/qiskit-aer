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
from qiskit.circuit import QuantumCircuit, Parameter
from test.terra.reference.ref_snapshot_expval import (
    snapshot_expval_circuits, snapshot_expval_counts, snapshot_expval_labels,
    snapshot_expval_pre_meas_values, snapshot_expval_circuit_parameterized,
    snapshot_expval_final_statevecs)
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, AerSimulator


class TestParameterizedQobj(common.QiskitAerTestCase):
    """Parameterized Qobj extension tests"""

    BACKEND_OPTS = {
        "seed_simulator": 2113
    }

    @staticmethod
    def expval_snapshots(data, labels):
        """Format snapshots as nested dicts"""
        # Check snapshot entry exists in data
        output = {}
        for label in labels:
            snaps = data.get("snapshots", {}).get("expectation_value",
                                                  {}).get(label, [])
            # Convert list into dict
            inner = {}
            for snap_dict in snaps:
                inner[snap_dict['memory']] = snap_dict['value']
            output[label] = inner
        return output

    @staticmethod
    def parameterized_qobj(backend, shots=1000, measure=True, snapshot=False):
        """Return ParameterizedQobj for settings."""
        single_shot = shots == 1
        pcirc1, param1 = snapshot_expval_circuit_parameterized(single_shot=single_shot,
                                                               measure=measure,
                                                               snapshot=snapshot)
        circuits2to4 = snapshot_expval_circuits(pauli=True,
                                                skip_measure=(not measure),
                                                single_shot=single_shot)
        pcirc2, param2 = snapshot_expval_circuit_parameterized(single_shot=single_shot,
                                                               measure=measure,
                                                               snapshot=snapshot)
        circuits = [pcirc1] + circuits2to4 + [pcirc2]
        params = [param1, [], [], [], param2]
        qobj = assemble(circuits,
                        backend=backend,
                        shots=shots,
                        parameterizations=params)
        return qobj

    def test_parameterized_qobj_qasm_snapshot_expval(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        shots = 1000
        labels = snapshot_expval_labels() * 3
        counts_targets = snapshot_expval_counts(shots) * 3
        value_targets = snapshot_expval_pre_meas_values() * 3

        backend = QasmSimulator()
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
        for j in range(num_circs):
            data = result.data(j)
            all_snapshots = self.expval_snapshots(data, labels)
            for label in labels:
                snaps = all_snapshots.get(label, {})
                self.assertTrue(len(snaps), 1)
                for memory, value in snaps.items():
                    target = value_targets[j].get(label,
                                                  {}).get(memory, {})
                    self.assertAlmostEqual(value, target, delta=1e-7)

    def test_parameterized_qobj_statevector(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        statevec_targets = snapshot_expval_final_statevecs() * 3

        backend = StatevectorSimulator()
        qobj = self.parameterized_qobj(backend=backend,
                                       measure=False,
                                       snapshot=False)
        self.assertIn('parameterizations', qobj.to_dict()['config'])
        job = backend.run(qobj, **self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        num_circs = len(result.to_dict()['results'])
        self.assertTrue(success)

        for j in range(num_circs):
            statevector = result.get_statevector(j)
            np.testing.assert_array_almost_equal(statevector, statevec_targets[j].data, decimal=7)

    def test_run_path(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, 3.14, 6.28]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': 1000}, {'11': 1000}, {'00': 1000}])

    def test_run_path_with_expressions(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        theta_squared = theta*theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, 3.14, 6.28]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': 1000}, {'11': 1000}, {'00': 1000}])

    def test_run_path_with_more_params_than_expressions(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        theta_squared = theta*theta
        phi = Parameter('phi')
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.ry(phi, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, 3.14, 6.28], phi: [0, 1, 3.14]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        for index, expected in enumerate([{'00': 1000}, {'01': 250, '11': 750}, {'10': 1000}]):
            self.assertDictAlmostEqual(counts[index], expected, delta=50)


if __name__ == '__main__':
    unittest.main()
