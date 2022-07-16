# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
NoiseModel class integration tests
"""

import unittest

import numpy as np
from qiskit.providers.aer.backends import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.device.models import _excited_population
from qiskit.providers.aer.noise.errors import QuantumError
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit.providers.aer.noise.errors.standard_errors import kraus_error
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from qiskit.providers.aer.noise.errors.standard_errors import reset_error
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.providers.aer.utils.noise_transformation import transform_noise_model

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate, XGate
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import (
    FakeBackend, FakeAlmaden, FakeLagos, FakeSingapore, FakeMumbai,
    FakeBackendV2, FakeLagosV2
)
from test.terra.common import QiskitAerTestCase


class TestNoiseModel(QiskitAerTestCase):
    """Testing noise model"""

    def test_amplitude_damping_error(self):
        """Test amplitude damping error damps to correct state"""
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)  # prepare + state
        for _ in range(30):
            # Add noisy identities
            circuit.barrier(qr)
            circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        shots = 4000
        backend = AerSimulator()
        # test noise model
        error = amplitude_damping_error(0.75, 0.25)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # Execute
        target = {'0x0': 3 * shots / 4, '0x1': shots / 4}
        circuit = transpile(circuit, basis_gates=noise_model.basis_gates, optimization_level=0)
        result = backend.run(circuit, shots=shots, noise_model=noise_model).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_noise_model_basis_gates(self):
        """Test noise model basis_gates"""
        basis_gates = ['u1', 'u2', 'u3', 'cx']
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        self.assertEqual(model.basis_gates, target)

        # Check adding readout errors doesn't add to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        self.assertEqual(model.basis_gates, target)
        model.add_readout_error([[0.9, 0.1], [0, 1]], [2], False)
        self.assertEqual(model.basis_gates, target)

        # Check a reset instruction error isn't added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_quantum_error(reset_error(0.2), ['reset'], False)
        self.assertEqual(model.basis_gates, target)

        # Check a non-standard gate isn't added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_quantum_error(reset_error(0.2), ['label'], False)
        self.assertEqual(model.basis_gates, target)

        # Check a standard gate is added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates + ['h'])
        model.add_all_qubit_quantum_error(reset_error(0.2), ['h'], False)
        self.assertEqual(model.basis_gates, target)

    def test_noise_model_noise_instructions(self):
        """Test noise instructions"""
        model = NoiseModel()
        target = []
        self.assertEqual(model.noise_instructions, target)

        # Check a non-standard gate is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ['label'], False)
        target = ['label']
        self.assertEqual(model.noise_instructions, target)

        # Check a standard gate is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ['h'], False)
        target = ['h']
        self.assertEqual(model.noise_instructions, target)

        # Check a reset is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ['reset'], False)
        target = ['reset']
        self.assertEqual(model.noise_instructions, target)

        # Check a measure is added to noise instructions for readout error
        model = NoiseModel()
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        target = ['measure']
        self.assertEqual(model.noise_instructions, target)

    def test_noise_model_noise_qubits(self):
        """Test noise instructions"""
        model = NoiseModel()
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a default error isn't added to noise qubits
        model = NoiseModel()
        model.add_all_qubit_quantum_error(pauli_error([['XX', 1]]), ['label'], False)
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a local error adds to noise qubits
        model = NoiseModel()
        model.add_quantum_error(pauli_error([['XX', 1]]), ['label'], [1, 0], False)
        target = sorted([0, 1])
        self.assertEqual(model.noise_qubits, target)

        # Check adding a non-local error adds to noise qubits
        model = NoiseModel()
        with self.assertWarns(DeprecationWarning):
            model.add_nonlocal_quantum_error(pauli_error([['XX', 1]]), ['label'], [0], [1, 2], False)
        target = sorted([0, 1, 2])
        self.assertEqual(model.noise_qubits, target)

        # Check adding a default error isn't added to noise qubits
        model = NoiseModel()
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a local error adds to noise qubits
        model = NoiseModel()
        model.add_readout_error([[0.9, 0.1], [0, 1]], [2], False)
        target = [2]
        self.assertEqual(model.noise_qubits, target)

    def test_noise_models_equal(self):
        """Test two noise models are Equal"""
        roerror = [[0.9, 0.1], [0.5, 0.5]]
        error1 = kraus_error([np.diag([1, 0]), np.diag([0, 1])])
        error2 = pauli_error([("I", 0.5), ("Z", 0.5)])

        model1 = NoiseModel()
        model1.add_all_qubit_quantum_error(error1, ['u3'], False)
        model1.add_quantum_error(error1, ['u3'], [2], False)
        with self.assertWarns(DeprecationWarning):
            model1.add_nonlocal_quantum_error(error1, ['cx'], [0, 1], [3], False)
        model1.add_all_qubit_readout_error(roerror, False)
        model1.add_readout_error(roerror, [0], False)

        model2 = NoiseModel()
        model2.add_all_qubit_quantum_error(error2, ['u3'], False)
        model2.add_quantum_error(error2, ['u3'], [2], False)
        with self.assertWarns(DeprecationWarning):
            model2.add_nonlocal_quantum_error(error2, ['cx'], [0, 1], [3], False)
        model2.add_all_qubit_readout_error(roerror, False)
        model2.add_readout_error(roerror, [0], False)
        self.assertEqual(model1, model2)

    def test_noise_models_not_equal(self):
        """Test two noise models are not equal"""
        error = pauli_error([['X', 1]])

        model1 = NoiseModel()
        model1.add_all_qubit_quantum_error(error, ['u3'], False)

        model2 = NoiseModel(basis_gates=['u3', 'cx'])
        model2.add_all_qubit_quantum_error(error, ['u3'], False)

    def test_noise_model_from_backend_singapore(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = FakeSingapore()
        noise_model = NoiseModel.from_backend(backend)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_backend_almaden(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = FakeAlmaden()
        noise_model = NoiseModel.from_backend(backend)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_mumbai(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = FakeMumbai()
        noise_model = NoiseModel.from_backend(backend)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_backend_v2(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = FakeBackendV2()
        noise_model = NoiseModel.from_backend(backend)
        self.assertEquals([0, 1], noise_model.noise_qubits)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_lagos_v2(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.cx(0, 1)
        circ.measure_all()

        backend = FakeLagosV2()
        noise_model = NoiseModel.from_backend(backend)
        self.assertEquals([0, 1, 2, 3, 4, 5, 6], noise_model.noise_qubits)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_invalid_t2_backend(self):
        """Test if issue user warning when creating a noise model from invalid t2 backend"""
        from qiskit.providers.models.backendproperties import BackendProperties, Gate, Nduv
        import datetime

        t1_ns, invalid_t2_ns = 75_1000, 200_1000
        u3_time_ns = 320
        frequency = 4919.96800692

        class InvalidT2Fake1Q(FakeBackend):
            def __init__(self):
                mock_time = datetime.datetime.now()
                dt = 1.3333
                configuration = BackendProperties(
                    backend_name="invalid_t2",
                    backend_version="0.0.0",
                    num_qubits=1,
                    basis_gates=["u3"],
                    qubits=[
                        [
                            Nduv(date=mock_time, name="T1", unit="µs", value=t1_ns/1000),
                            Nduv(date=mock_time, name="T2", unit="µs", value=invalid_t2_ns/1000),
                            Nduv(date=mock_time, name="frequency", unit="MHz", value=frequency),
                        ],
                    ],
                    gates=[
                        Gate(
                            gate="u3",
                            name="u3_0",
                            qubits=[0],
                            parameters=[
                                Nduv(date=mock_time, name="gate_error", unit="", value=0.001),
                                Nduv(date=mock_time, name="gate_length", unit="ns", value=u3_time_ns),
                            ],
                        ),
                    ],
                    last_update_date=mock_time,
                    general=[],
                )
                super().__init__(configuration)

            def defaults(self):
                """defaults == configuration"""
                return self._configuration

            def properties(self):
                """properties == configuration"""
                return self._configuration

        backend = InvalidT2Fake1Q()
        with self.assertWarns(UserWarning):
            noise_model = NoiseModel.from_backend(backend, gate_error=False)
            expected = thermal_relaxation_error(
                t1=t1_ns,
                t2=2*t1_ns,
                time=u3_time_ns,
                excited_state_population=_excited_population(frequency, temperature=0)
            )
            self.assertEqual(expected, noise_model._local_quantum_errors["u3"][(0, )])

    def test_transform_noise(self):
        org_error = reset_error(0.2)
        new_error = pauli_error([("I", 0.5), ("Z", 0.5)])

        model = NoiseModel()
        model.add_all_qubit_quantum_error(org_error, ['x'])
        model.add_quantum_error(org_error, ['sx'], [0])
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]])

        def map_func(noise):
            return new_error if noise == org_error else None

        actual = transform_noise_model(model, map_func)

        expected = NoiseModel()
        expected.add_all_qubit_quantum_error(new_error, ['x'])
        expected.add_quantum_error(new_error, ['sx'], [0])
        expected.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]])

        self.assertEqual(actual, expected)

    def test_can_run_circuits_with_delay_noise(self):
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure_all()

        backend = FakeLagos()
        noise_model = NoiseModel.from_backend(backend)

        qc = transpile(circ, backend, scheduling_method="alap")
        result = AerSimulator().run(qc, noise_model=noise_model).result()
        self.assertTrue(result.success)

        # test another path
        noisy_sim = AerSimulator().from_backend(backend)
        qc = transpile(circ, noisy_sim, scheduling_method="alap")
        result = noisy_sim.run(qc).result()
        self.assertTrue(result.success)

        # no scheduling = no delay noise
        qc = transpile(circ, backend)
        result = AerSimulator().run(qc, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_from_dict(self):
        noise_ops_1q = [((IGate(), [0]), 0.9),
                     ((XGate(), [0]), 0.1)]

        noise_ops_2q = [((PauliGate('II'), [0, 1]), 0.9),
                     ((PauliGate('IX'), [0, 1]), 0.045),
                     ((PauliGate('XI'), [0, 1]), 0.045),
                     ((PauliGate('XX'), [0, 1]), 0.01)]

        noise_model = NoiseModel()
        with self.assertWarns(DeprecationWarning):
            noise_model.add_quantum_error(QuantumError(noise_ops_1q, 1), 'h', [0])
            noise_model.add_quantum_error(QuantumError(noise_ops_1q, 1), 'h', [1])
            noise_model.add_quantum_error(QuantumError(noise_ops_2q, 2), 'cx', [0, 1])
            noise_model.add_quantum_error(QuantumError(noise_ops_2q, 2), 'cx', [1, 0])
            deserialized = NoiseModel.from_dict(noise_model.to_dict())
            self.assertEqual(noise_model, deserialized)


if __name__ == '__main__':
    unittest.main()
