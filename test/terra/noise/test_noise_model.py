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
import warnings

import numpy as np
from qiskit_aer.backends import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.device.models import _excited_population
from qiskit_aer.noise.errors import PauliError, PauliLindbladError
from qiskit_aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit_aer.noise.errors.standard_errors import kraus_error
from qiskit_aer.noise.errors.standard_errors import pauli_error
from qiskit_aer.noise.errors.standard_errors import reset_error
from qiskit_aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit_aer.utils.noise_transformation import transform_noise_model

import qiskit
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap, Target
from qiskit.providers import QubitProperties, BackendV2, Options

if qiskit.__version__.startswith("0."):
    from qiskit.providers.fake_provider import (
        FakeBackend,
        FakeAlmaden as Fake20QV1,
        FakeMumbai as Fake27QPulseV1,
        FakeLagosV2,
    )

    def fake_7q_v2():
        """Generate a dummy 7q V2 backend."""
        return FakeLagosV2()

else:
    from qiskit.providers.fake_provider import (
        FakeBackend,
        Fake20QV1,
        Fake27QPulseV1,
        GenericBackendV2,
    )

    def fake_7q_v2():
        """Generate a dummy 7q V2 backend."""
        return GenericBackendV2(num_qubits=7, coupling_map=CouplingMap.from_ring(7), seed=0)


from test.terra.common import QiskitAerTestCase


class TestNoiseModel(QiskitAerTestCase):
    """Testing noise model"""

    def test_amplitude_damping_error(self):
        """Test amplitude damping error damps to correct state"""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)  # prepare + state
        for _ in range(30):
            # Add noisy identities
            circuit.barrier(qr)
            circuit.id(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        shots = 4000
        backend = AerSimulator()
        # test noise model
        error = amplitude_damping_error(0.75, 0.25)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, "id")
        # Execute
        target = {"0x0": 3 * shots / 4, "0x1": shots / 4}
        circuit = transpile(circuit, basis_gates=noise_model.basis_gates, optimization_level=0)
        result = backend.run(circuit, shots=shots, noise_model=noise_model).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_noise_model_basis_gates(self):
        """Test noise model basis_gates"""
        basis_gates = ["u1", "u2", "u3", "cx"]
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
        model.add_all_qubit_quantum_error(reset_error(0.2), ["reset"], False)
        self.assertEqual(model.basis_gates, target)

        # Check a non-standard gate isn't added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_quantum_error(reset_error(0.2), ["label"], False)
        self.assertEqual(model.basis_gates, target)

        # Check a standard gate is added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates + ["h"])
        model.add_all_qubit_quantum_error(reset_error(0.2), ["h"], False)
        self.assertEqual(model.basis_gates, target)

    def test_noise_model_noise_instructions(self):
        """Test noise instructions"""
        model = NoiseModel()
        target = []
        self.assertEqual(model.noise_instructions, target)

        # Check a non-standard gate is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ["label"], False)
        target = ["label"]
        self.assertEqual(model.noise_instructions, target)

        # Check a standard gate is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ["h"], False)
        target = ["h"]
        self.assertEqual(model.noise_instructions, target)

        # Check a reset is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ["reset"], False)
        target = ["reset"]
        self.assertEqual(model.noise_instructions, target)

        # Check a measure is added to noise instructions for readout error
        model = NoiseModel()
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        target = ["measure"]
        self.assertEqual(model.noise_instructions, target)

    def test_noise_model_noise_qubits(self):
        """Test noise instructions"""
        model = NoiseModel()
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a default error isn't added to noise qubits
        model = NoiseModel()
        model.add_all_qubit_quantum_error(pauli_error([["XX", 1]]), ["label"], False)
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a local error adds to noise qubits
        model = NoiseModel()
        model.add_quantum_error(pauli_error([["XX", 1]]), ["label"], [1, 0], False)
        target = sorted([0, 1])
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
        model1.add_all_qubit_quantum_error(error1, ["u3"], False)
        model1.add_quantum_error(error1, ["u3"], [2], False)
        model1.add_all_qubit_readout_error(roerror, False)
        model1.add_readout_error(roerror, [0], False)

        model2 = NoiseModel()
        model2.add_all_qubit_quantum_error(error2, ["u3"], False)
        model2.add_quantum_error(error2, ["u3"], [2], False)
        model2.add_all_qubit_readout_error(roerror, False)
        model2.add_readout_error(roerror, [0], False)
        self.assertEqual(model1, model2)

    def test_noise_models_not_equal(self):
        """Test two noise models are not equal"""
        error = pauli_error([["X", 1]])

        model1 = NoiseModel()
        model1.add_all_qubit_quantum_error(error, ["u3"], False)

        model2 = NoiseModel(basis_gates=["u3", "cx"])
        model2.add_all_qubit_quantum_error(error, ["u3"], False)

    def test_noise_model_from_backend_20(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = Fake20QV1()
        noise_model = NoiseModel.from_backend(backend)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_backend_27_pulse(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = Fake27QPulseV1()
        noise_model = NoiseModel.from_backend(backend)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_backend_v2(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.cx(0, 1)
        circ.measure_all()

        backend = fake_7q_v2()
        noise_model = NoiseModel.from_backend(backend)
        self.assertEqual([0, 1, 2, 3, 4, 5, 6], noise_model.noise_qubits)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_backend_v2_with_non_operational_qubits(self):
        """Test if possible to create a noise model from backend with non-operational qubits.
        See issues #1779 and #1815 for the details."""
        backend = fake_7q_v2()
        # tweak target to have non-operational qubits
        faulty_qubits = [0, 1]
        for qubit in faulty_qubits:
            backend.target.qubit_properties[qubit] = QubitProperties(t1=None, t2=None, frequency=0)

        noise_model = NoiseModel.from_backend(backend)

        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure_all()
        circ = transpile(circ, backend, scheduling_method="alap")
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_invalid_t2_backend(self):
        """Test if silently truncate invalid T2 values when creating a noise model from backend"""
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
                            Nduv(date=mock_time, name="T1", unit="µs", value=t1_ns / 1000),
                            Nduv(date=mock_time, name="T2", unit="µs", value=invalid_t2_ns / 1000),
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
                                Nduv(
                                    date=mock_time, name="gate_length", unit="ns", value=u3_time_ns
                                ),
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
        noise_model = NoiseModel.from_backend(backend, gate_error=False)
        expected = thermal_relaxation_error(
            t1=t1_ns,
            t2=2 * t1_ns,
            time=u3_time_ns,
            excited_state_population=_excited_population(frequency, temperature=0),
        )
        self.assertEqual(expected, noise_model._local_quantum_errors["u3"][(0,)])

    def test_create_noise_model_without_user_warnings(self):
        """Test if never issue user warnings when creating a noise model from backend.
        See issue#1631 for the details."""

        class BadlyCalibratedBackendV2(BackendV2):
            """A backend with `t2 > 2*t1` due to awkward calibration statistics."""

            @property
            def target(self):
                return Target(
                    num_qubits=1, qubit_properties=[QubitProperties(t1=1.2e-3, t2=2.5e-3)]
                )

            @property
            def max_circuits(self):
                return None

            @classmethod
            def _default_options(cls):
                return Options()

            def run(self, run_input, **options):
                raise NotImplementedError

        with warnings.catch_warnings(record=True) as warns:
            NoiseModel.from_backend(BadlyCalibratedBackendV2())
            user_warnings = [w for w in warns if issubclass(w.category, UserWarning)]
            self.assertEqual(len(user_warnings), 0)

    def test_noise_model_from_backend_properties(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = Fake20QV1()
        backend_propeties = backend.properties()
        noise_model = NoiseModel.from_backend_properties(backend_propeties)
        circ = transpile(circ, backend, optimization_level=0)
        result = AerSimulator().run(circ, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_transform_noise(self):
        org_error = reset_error(0.2)
        new_error = pauli_error([("I", 0.5), ("Z", 0.5)])

        model = NoiseModel()
        model.add_all_qubit_quantum_error(org_error, ["x"])
        model.add_quantum_error(org_error, ["sx"], [0])
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]])

        def map_func(noise):
            return new_error if noise == org_error else None

        actual = transform_noise_model(model, map_func)

        expected = NoiseModel()
        expected.add_all_qubit_quantum_error(new_error, ["x"])
        expected.add_quantum_error(new_error, ["sx"], [0])
        expected.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]])

        self.assertEqual(actual, expected)

    def test_can_run_circuits_with_delay_noise(self):
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure_all()

        backend = fake_7q_v2()
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

    def test_pauli_error_equiv(self):
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure_all()

        perr1 = PauliError(["I", "X"], [0.9, 0.1])
        perr2 = PauliError(["II", "XX"], [0.9, 0.1])

        noise_model1 = NoiseModel()
        noise_model1.add_all_qubit_quantum_error(perr1, ["h"])
        noise_model1.add_all_qubit_quantum_error(perr2, ["cx"])

        noise_model2 = NoiseModel()
        noise_model2.add_all_qubit_quantum_error(perr1.to_quantum_error(), ["h"])
        noise_model2.add_all_qubit_quantum_error(perr2.to_quantum_error(), ["cx"])

        seed = 1234
        result1 = (
            AerSimulator()
            .run(circ, shots=5000, noise_model=noise_model1, seed_simulator=seed)
            .result()
        )
        result2 = (
            AerSimulator()
            .run(circ, shots=5000, noise_model=noise_model2, seed_simulator=seed)
            .result()
        )
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertEqual(result1.get_counts(), result2.get_counts())

    def test_pauli_lindblad_error_sampling(self):
        num_qubits = 100
        rate = 0.1
        plerr = PauliLindbladError([num_qubits * "X"], [rate])
        prob_err = 0.5 - 0.5 * np.exp(-2 * rate)

        circ = QuantumCircuit(num_qubits)
        circ.append(plerr, range(num_qubits))
        circ.measure_all()

        shots = 10000
        backend = AerSimulator(method="stabilizer", seed_simulator=123)
        result = backend.run(circ, shots=shots).result()
        counts = result.get_counts()
        self.assertEqual(len(counts), 2)
        self.assertEqual(sum(counts.values()), shots)
        self.assertIn(num_qubits * "0", counts)
        self.assertIn(num_qubits * "1", counts)
        np.testing.assert_allclose(counts[num_qubits * "1"] / shots, prob_err, atol=1e-3)

    def test_pauli_lindblad_error_equiv(self):
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure_all()

        perr1 = PauliLindbladError(["X", "Y", "Z"], [0.1, 0.2, 0.3])
        perr2 = PauliLindbladError(["XX", "YY", "ZZ"], [0.1, 0.2, 0.3])

        noise_model1 = NoiseModel()
        noise_model1.add_all_qubit_quantum_error(perr1, ["h"])
        noise_model1.add_all_qubit_quantum_error(perr2, ["cx"])

        noise_model2 = NoiseModel()
        noise_model2.add_all_qubit_quantum_error(perr1.to_quantum_error(), ["h"])
        noise_model2.add_all_qubit_quantum_error(perr2.to_quantum_error(), ["cx"])

        seed = 1234
        result1 = (
            AerSimulator()
            .run(circ, shots=5000, noise_model=noise_model1, seed_simulator=seed)
            .result()
        )
        result2 = (
            AerSimulator()
            .run(circ, shots=5000, noise_model=noise_model2, seed_simulator=seed)
            .result()
        )
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertEqual(result1.get_counts(), result2.get_counts())

    def test_pauli_lindblad_error_sampling_equiv(self):
        plerr = PauliLindbladError(["IX", "XI"], [0.5, 0.1])
        circ1 = QuantumCircuit(2)
        circ1.append(plerr, range(2))
        circ1.measure_all()

        circ2 = QuantumCircuit(2)
        circ2.append(plerr.to_quantum_error(), range(2))
        circ2.measure_all()

        shots = 100000
        backend = AerSimulator(seed_simulator=123)
        result = backend.run([circ1, circ2], shots=shots).result()
        counts1 = result.get_counts(0)
        counts2 = result.get_counts(1)
        probs1 = [counts1.get(i, 0) / shots for i in ["00", "01", "10", "11"]]
        probs2 = [counts2.get(i, 0) / shots for i in ["00", "01", "10", "11"]]
        np.testing.assert_allclose(probs1, probs2, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
