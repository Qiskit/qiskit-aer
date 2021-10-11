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

from qiskit.compiler import assemble, transpile
from qiskit.circuit import QuantumCircuit, Parameter
from test.terra.reference.ref_save_expval import (
    save_expval_circuits,
    save_expval_counts,
    save_expval_labels,
    save_expval_pre_meas_values,
    save_expval_circuit_parameterized,
    save_expval_final_statevecs,
)
from qiskit.providers.aer import AerSimulator, AerError


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
        with self.assertWarns(DeprecationWarning):
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
                    self.assertAlmostEqual(
                        data[label], target[label], delta=1e-7)

    def test_parameterized_qobj_statevector(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        statevec_targets = save_expval_final_statevecs() * 3

        backend = AerSimulator(method="statevector")
        qobj = self.parameterized_qobj(
            backend=backend, measure=False, snapshot=False, save_state=True,
        )
        self.assertIn('parameterizations', qobj.to_dict()['config'])
        with self.assertWarns(DeprecationWarning):
            job = backend.run(qobj, **self.BACKEND_OPTS)
            result = job.result()
            success = getattr(result, 'success', False)
            num_circs = len(result.to_dict()['results'])
            self.assertTrue(success)

            for j in range(num_circs):
                statevector = result.get_statevector(j)
                np.testing.assert_array_almost_equal(
                    statevector, statevec_targets[j].data, decimal=7)

    def test_run_path(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'11': shots}, {'00': shots}])

    def test_run_path_already_bound_parameter_expression(self):
        """Test parameterizations with a parameter expression that's already bound."""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        tmp = Parameter('x')
        theta = Parameter('theta')
        expr = tmp - tmp
        bound_expr = expr.bind({tmp: 1})
        circuit.rx(theta, 0)
        circuit.rx(bound_expr, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'11': shots}, {'00': shots}])

    def test_run_path_already_transpiled_parameter_expression(self):
        """Test parameterizations with a transpiled parameter expression."""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(1)
        theta = Parameter('theta')
        circuit.rx(theta, 0)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        tqc = transpile(circuit, basis_gates=['u3'])
        res = backend.run(tqc, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'0': shots}, {'1': shots}, {'0': shots}])


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
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'11': shots}, {'00': shots}])

    def test_run_path_with_expressions_multiple_params_per_instruction(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        theta_squared = theta*theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'01': shots}, {'00': shots}])

    def test_run_path_with_more_params_than_expressions(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 2000
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
        parameter_binds = [{theta: [0, pi, 2 * pi], phi: [0, 1, pi]}]
        res = backend.run(circuit, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        for index, expected in enumerate([{'00': shots}, {'01': 0.25*shots, '11': 0.75*shots}, {'10': shots}]):
            self.assertDictAlmostEqual(counts[index], expected, delta=0.05*shots)

    def test_run_path_multiple_circuits(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]*3
        res = backend.run([circuit]*3, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'11': shots}, {'00': shots}] * 3)

    def test_run_path_with_expressions_multiple_circuits(self):
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
        parameter_binds = [{theta: [0, pi, 2 * pi]}]*3
        res = backend.run([circuit]*3, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'11': shots}, {'00': shots}] * 3)

    def test_run_path_with_expressions_multiple_params_per_instruction(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        theta_squared = theta*theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]*3
        res = backend.run([circuit]*3, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{'00': shots}, {'01': shots}, {'00': shots}] * 3)

    def test_run_path_with_more_params_than_expressions_multiple_circuits(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 2000
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
        parameter_binds = [{theta: [0, pi, 2 * pi], phi: [0, 1, pi]}]*3
        res = backend.run([circuit]*3, shots=shots, parameter_binds=parameter_binds).result()
        counts = res.get_counts()
        for index, expected in enumerate([{'00': shots}, {'01': 0.25*shots, '11': 0.75*shots}, {'10': shots}] * 3):
            self.assertDictAlmostEqual(counts[index], expected, delta=0.05*shots)

    def test_run_path_multiple_circuits_mismatch_length(self):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = AerSimulator()
        circuit = QuantumCircuit(2)
        theta = Parameter('theta')
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        with self.assertRaises(AerError):
            backend.run([circuit]*3, shots=shots, parameter_binds=[parameter_binds]).result()



if __name__ == '__main__':
    unittest.main()
