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
import platform
from math import pi
from ddt import ddt
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
from qiskit_aer.library import SaveStatevector
from qiskit_aer import AerSimulator, AerError

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import pauli_error, amplitude_damping_error


from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

SUPPORTED_METHODS = [
    "statevector",
]


@ddt
class TestRuntimeParameterization(SimulatorTestCase):
    """Runtime Parameterization tests"""

    BACKEND_OPTS = {
        "seed_simulator": 2113,
        "shot_branching_enable": False,
        "runtime_parameter_bind_enable": True,
    }

    @staticmethod
    def runtime_parameterization(
        backend,
        shots=1000,
        measure=True,
        snapshot=False,
        save_state=False,
    ):
        """Return ParameterizedQobj for settings."""
        pershot = shots == 1
        pcirc1, param1 = save_expval_circuit_parameterized(
            pershot=pershot,
            measure=measure,
            snapshot=snapshot,
        )
        circuits2to4 = save_expval_circuits(
            pauli=True,
            skip_measure=(not measure),
            pershot=pershot,
        )
        pcirc2, param2 = save_expval_circuit_parameterized(
            pershot=pershot,
            measure=measure,
            snapshot=snapshot,
        )
        circuits = [pcirc1] + circuits2to4 + [pcirc2]
        if save_state:
            for circuit in circuits:
                circuit.save_statevector(pershot=pershot)
        params = [param1, [], [], [], param2]
        qobj = assemble(circuits, backend=backend, shots=shots, parameterizations=params)
        return qobj

    def test_runtime_parameterization_qasm_save_expval(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        shots = 1000
        labels = save_expval_labels() * 3
        counts_targets = save_expval_counts(shots) * 3
        value_targets = save_expval_pre_meas_values() * 3

        backend = AerSimulator()
        qobj = self.runtime_parameterization(
            backend=backend, shots=1000, measure=True, snapshot=True
        )
        self.assertIn("parameterizations", qobj.to_dict()["config"])
        with self.assertWarns(DeprecationWarning):
            job = backend.run(qobj, **self.BACKEND_OPTS)
            result = job.result()
            success = getattr(result, "success", False)
            num_circs = len(result.to_dict()["results"])
            self.assertTrue(success)
            self.compare_counts(result, range(num_circs), counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, target in enumerate(value_targets):
                data = result.data(j)
                for label in labels:
                    self.assertAlmostEqual(data[label], target[label], delta=1e-7)

    def test_runtime_parameterization_statevector(self):
        """Test parameterized qobj with Expectation Value snapshot and qasm simulator."""
        statevec_targets = save_expval_final_statevecs() * 3

        backend = AerSimulator(method="statevector")
        qobj = self.runtime_parameterization(
            backend=backend,
            measure=False,
            snapshot=False,
            save_state=True,
        )
        self.assertIn("parameterizations", qobj.to_dict()["config"])
        with self.assertWarns(DeprecationWarning):
            job = backend.run(qobj, **self.BACKEND_OPTS)
            result = job.result()
            success = getattr(result, "success", False)
            num_circs = len(result.to_dict()["results"])
            self.assertTrue(success)

            for j in range(num_circs):
                statevector = result.get_statevector(j)
                np.testing.assert_array_almost_equal(
                    statevector, statevec_targets[j].data, decimal=7
                )

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"11": shots}, {"00": shots}])

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_already_bound_parameter_expression(self, method, device):
        """Test parameterizations with a parameter expression that's already bound."""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        tmp = Parameter("x")
        theta = Parameter("theta")
        expr = tmp - tmp
        bound_expr = expr.bind({tmp: 1})
        circuit.rx(theta, 0)
        circuit.rx(bound_expr, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"11": shots}, {"00": shots}])

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_already_transpiled_parameter_expression(self, method, device):
        """Test parameterizations with a transpiled parameter expression."""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(1)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        tqc = transpile(circuit, basis_gates=["u3"])
        res = backend.run(
            tqc,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"0": shots}, {"1": shots}, {"0": shots}])

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_expressions(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"11": shots}, {"00": shots}])

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_expressions_multiple_params_per_instruction(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"01": shots}, {"00": shots}])

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_more_params_than_expressions(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 2000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        phi = Parameter("phi")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.ry(phi, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi], phi: [0, 1, pi]}]
        res = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        for index, expected in enumerate(
            [{"00": shots}, {"01": 0.25 * shots, "11": 0.75 * shots}, {"10": shots}]
        ):
            self.assertDictAlmostEqual(counts[index], expected, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_multiple_circuits(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}] * 3
        res = backend.run(
            [circuit] * 3,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"11": shots}, {"00": shots}] * 3)

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_multiple_different_circuits(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)

        circuit1 = QuantumCircuit(2)
        theta1 = Parameter("theta1")
        circuit1.rx(theta1, 0)
        circuit1.cx(0, 1)
        circuit1.measure_all()

        circuit2 = QuantumCircuit(2)
        theta2 = Parameter("theta2")
        circuit2.rx(theta2, 0)
        circuit2.cx(0, 1)
        circuit2.measure_all()

        circuit3 = QuantumCircuit(2)
        theta3_1 = Parameter("theta3_1")
        theta3_2 = Parameter("theta3_2")
        circuit3.rx(theta3_1, 0)
        circuit3.rx(theta3_2, 0)
        circuit3.cx(0, 1)
        circuit3.measure_all()

        parameter_binds = [
            {theta1: [0, pi, 2 * pi]},
            {theta2: [0, pi, 2 * pi]},
            {theta3_1: [0, pi / 2, pi], theta3_2: [0, pi / 2, pi]},
        ]
        res = backend.run(
            [circuit1, circuit2, circuit3],
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"11": shots}, {"00": shots}] * 3)

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_expressions_multiple_circuits(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}] * 3
        res = backend.run(
            [circuit] * 3,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"11": shots}, {"00": shots}] * 3)

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_expressions_multiple_params_per_instruction(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}] * 3
        res = backend.run(
            [circuit] * 3,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"01": shots}, {"00": shots}] * 3)

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_more_params_than_expressions_multiple_circuits(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 2000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        phi = Parameter("phi")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.ry(phi, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi], phi: [0, 1, pi]}] * 3
        res = backend.run(
            [circuit] * 3,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        for index, expected in enumerate(
            [{"00": shots}, {"01": 0.25 * shots, "11": 0.75 * shots}, {"10": shots}] * 3
        ):
            self.assertDictAlmostEqual(counts[index], expected, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_multiple_circuits_mismatch_length(self, method, device):
        """Test parameterized circuit path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        with self.assertRaises(AerError):
            backend.run(
                [circuit] * 3,
                shots=shots,
                parameter_binds=[parameter_binds],
                shot_branching_enable=False,
                runtime_parameter_bind_enable=True,
            ).result()

    @supported_methods(SUPPORTED_METHODS)
    def test_run_path_with_truncation(self, method, device):
        """Test parameterized circuits with truncation"""
        backend = self.backend(method=method, device=device)
        theta = Parameter("theta")
        circuit = QuantumCircuit(5, 2)
        for q in range(5):
            circuit.ry(theta, q)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        for q in range(5):
            circuit.ry(theta, q)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.append(SaveStatevector(3, label="sv", pershot=False, conditional=False), range(3))

        param_map = {theta: [0.1 * i for i in range(3)]}
        param_sets = [{theta: 0.1 * i} for i in range(3)]

        resolved_circuits = [circuit.assign_parameters(param_set) for param_set in param_sets]

        result = backend.run(
            circuit,
            parameter_binds=[param_map],
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)

        result_without_parameters = backend.run(resolved_circuits).result()
        self.assertSuccess(result_without_parameters)

        for actual_result in result.results:
            metadata = actual_result.metadata
            self.assertEqual(metadata["active_input_qubits"], [q for q in range(3)])
        for i in range(3):
            self.assertEqual(result.data(i)["sv"], result_without_parameters.data(i)["sv"])

    @supported_methods(SUPPORTED_METHODS)
    def test_different_seed(self, method, device):
        """Test parameterized circuits have different seeds"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]
        res = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        seed_simulator_list = [result.seed_simulator for result in res.results]
        self.assertEqual(len(seed_simulator_list), len(np.unique(seed_simulator_list)))

        res2 = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            seed_simulator=seed_simulator_list[0],
        ).result()
        self.assertEqual(seed_simulator_list, [result.seed_simulator for result in res2.results])

    @supported_methods(SUPPORTED_METHODS)
    def test_run_empty(self, method, device):
        """Test parameterized circuit with empty dict path via backed.run()"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        parameter_binds = [{}]
        with self.assertRaises(AerError):
            res = backend.run(
                circuit,
                shots=shots,
                parameter_binds=parameter_binds,
                shot_branching_enable=False,
                runtime_parameter_bind_enable=True,
            ).result()

    @supported_methods(SUPPORTED_METHODS)
    def test_parameters_with_barrier(self, method, device):
        """Test parameterized circuit path with barrier"""
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(3)
        theta = Parameter("theta")
        phi = Parameter("phi")
        circuit.rx(theta, 0)
        circuit.rx(theta, 1)
        circuit.rx(theta, 2)
        circuit.barrier()
        circuit.rx(phi, 0)
        circuit.rx(phi, 1)
        circuit.rx(phi, 2)
        circuit.barrier()
        circuit.measure_all()

        parameter_binds = [{theta: [pi / 2], phi: [pi / 2]}]
        res = backend.run(
            [circuit],
            shots=1024,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()

        self.assertSuccess(res)
        self.assertEqual(res.get_counts(), {"111": 1024})

    @supported_methods(SUPPORTED_METHODS)
    def test_dynamic_circuit(self, method, device):
        """Test parameterized dynamic circuit"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.h(0)
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.reset(0)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]

        result = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)
        counts = result.get_counts()

        result_pre_bind = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=False,
        ).result()
        self.assertSuccess(result_pre_bind)
        counts_pre_bind = result_pre_bind.get_counts()

        self.assertEqual(counts, counts_pre_bind)

    @supported_methods(SUPPORTED_METHODS)
    @unittest.skipIf(platform.system() == "Darwin", "skip MacOS tentatively")
    def test_dynamic_circuit_with_shot_branching(self, method, device):
        """Test parameterized dynamic circuit"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.h(0)
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.reset(0)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]

        result = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=True,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)
        counts = result.get_counts()

        result_pre_bind = backend.run(
            circuit,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=False,
        ).result()
        self.assertSuccess(result_pre_bind)
        counts_pre_bind = result_pre_bind.get_counts()

        self.assertEqual(counts, counts_pre_bind)

    @supported_methods(SUPPORTED_METHODS)
    def test_fusion(self, method, device):
        """Test parameterized circuit with fusion"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}] * 3
        res = backend.run(
            [circuit] * 3,
            shots=shots,
            parameter_binds=parameter_binds,
            fusion_enable=True,
            fusion_threshold=1,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        counts = res.get_counts()
        self.assertEqual(counts, [{"00": shots}, {"01": shots}, {"00": shots}] * 3)

    @supported_methods(SUPPORTED_METHODS)
    def test_pauli_noise(self, method, device):
        """Test parameterized circuit with Pauli noise"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.h(0)
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]

        error = pauli_error([("X", 0.2), ("Y", 0.2), ("Z", 0.2), ("I", 0.4)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["h", "rx", "rz", "u"])

        result = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)
        counts = result.get_counts()

        result_pre_bind = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=False,
        ).result()
        self.assertSuccess(result_pre_bind)
        counts_pre_bind = result_pre_bind.get_counts()

        self.assertEqual(counts, counts_pre_bind)

    @supported_methods(SUPPORTED_METHODS)
    def test_kraus_noise(self, method, device):
        """Test parameterized circuit with Kraus noise"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.h(0)
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]

        error = amplitude_damping_error(0.75, 0.25)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["h", "rx", "rz", "u"])

        result = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)
        counts = result.get_counts()

        result_pre_bind = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=False,
        ).result()
        self.assertSuccess(result_pre_bind)
        counts_pre_bind = result_pre_bind.get_counts()

        self.assertEqual(counts, counts_pre_bind)

    @supported_methods(SUPPORTED_METHODS)
    @unittest.skipIf(platform.system() == "Darwin", "skip MacOS tentatively")
    def test_pauli_noise_with_shot_branching(self, method, device):
        """Test parameterized circuit with Pauli noise"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.h(0)
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]

        error = pauli_error([("X", 0.2), ("Y", 0.2), ("Z", 0.2), ("I", 0.4)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["h", "rx", "rz", "u"])

        result = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=True,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)
        counts = result.get_counts()

        result_pre_bind = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=False,
        ).result()
        self.assertSuccess(result_pre_bind)
        counts_pre_bind = result_pre_bind.get_counts()

        self.assertEqual(counts, counts_pre_bind)

    @supported_methods(SUPPORTED_METHODS)
    @unittest.skipIf(platform.system() == "Darwin", "skip MacOS tentatively")
    def test_kraus_noise_with_shot_branching(self, method, device):
        """Test parameterized circuit with Kraus noise"""
        shots = 1000
        backend = self.backend(method=method, device=device)
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        theta_squared = theta * theta
        circuit.h(0)
        circuit.rx(theta, 0)
        circuit.cx(0, 1)
        circuit.rz(theta_squared, 1)
        circuit.u(theta, theta_squared, theta, 1)
        circuit.measure_all()
        parameter_binds = [{theta: [0, pi, 2 * pi]}]

        error = amplitude_damping_error(0.75, 0.25)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["h", "rx", "rz", "u"])

        result = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=True,
            runtime_parameter_bind_enable=True,
        ).result()
        self.assertSuccess(result)
        counts = result.get_counts()

        result_pre_bind = backend.run(
            circuit,
            noise_model=noise_model,
            shots=shots,
            parameter_binds=parameter_binds,
            shot_branching_enable=False,
            runtime_parameter_bind_enable=False,
        ).result()
        self.assertSuccess(result_pre_bind)
        counts_pre_bind = result_pre_bind.get_counts()

        self.assertEqual(counts, counts_pre_bind)


if __name__ == "__main__":
    unittest.main()
