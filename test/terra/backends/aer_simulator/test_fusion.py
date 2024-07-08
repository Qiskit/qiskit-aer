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
AerSimulator Integration Tests
"""
# pylint: disable=no-member
import copy
import numpy as np
import math
from ddt import ddt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QuantumVolume, QFT, RealAmplitudes
from qiskit.compiler import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, depolarizing_error, amplitude_damping_error
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestGateFusion(SimulatorTestCase):
    """AerSimulator fusion tests."""

    def create_statevector_circuit(self):
        """Creates a simple circuit for running in the statevector"""
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        circuit = QuantumCircuit(qr, cr)
        circuit.u(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        return circuit

    def noise_model_depol(self):
        """Creates a new noise model for testing purposes"""
        readout_error = [0.01, 0.1]
        params = {"u3": (1, 0.001), "cx": (2, 0.02)}
        noise = NoiseModel()
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in params.items():
            noise.add_all_qubit_quantum_error(depolarizing_error(gate_error, num_qubits), gate)
            return noise

    def noise_model_kraus(self):
        """Creates a new noise model for testing purposes"""
        readout_error = [0.01, 0.1]
        params = {"u3": (1, 0.001), "cx": (2, 0.02)}
        noise = NoiseModel()
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in params.items():
            noise.add_all_qubit_quantum_error(amplitude_damping_error(gate_error, num_qubits), gate)
            return noise

    def fusion_options(self, enabled=None, threshold=None, verbose=None, parallelization=None):
        """Return default backend_options dict."""
        backend_options = {}
        if enabled is not None:
            backend_options["fusion_enable"] = enabled
        if verbose is not None:
            backend_options["fusion_verbose"] = verbose
        if threshold is not None:
            backend_options["fusion_threshold"] = threshold
        if parallelization is not None:
            backend_options["fusion_parallelization_threshold"] = 1
            backend_options["max_parallel_threads"] = parallelization
            backend_options["max_parallel_shots"] = 1
            backend_options["max_parallel_state_update"] = parallelization
        return backend_options

    def fusion_metadata(self, result):
        """Return fusion metadata dict"""
        metadata = result.results[0].metadata
        return metadata.get("fusion", {})

    @supported_methods(["automatic", "statevector", "density_matrix"])
    def test_fusion_threshold(self, method, device):
        """Test fusion threshsold"""
        shots = 100
        num_qubits = 5
        backend = self.backend(method=method, device=device)
        circuit = transpile(QFT(num_qubits), backend, optimization_level=0)
        circuit.measure_all()

        with self.subTest(msg="at threshold"):
            backend.set_options(**self.fusion_options(enabled=True, threshold=num_qubits))
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get("enabled"))
            self.assertFalse(meta.get("applied"))

        with self.subTest(msg="below threshold"):
            backend.set_options(**self.fusion_options(enabled=True, threshold=num_qubits + 1))
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get("enabled"))
            self.assertFalse(meta.get("applied"))

        with self.subTest(msg="above threshold"):
            backend.set_options(**self.fusion_options(enabled=True, threshold=num_qubits - 1))
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get("enabled"))
            self.assertTrue(meta.get("applied"))

    @supported_methods(["stabilizer", "matrix_product_state", "extended_stabilizer"])
    def test_method_no_fusion(self, method, device):
        """Test fusion MPS threshsold"""
        shots = 100
        backend = self.backend(method=method, device=device)
        num_qubits = 5
        circuit = QuantumCircuit(num_qubits)
        circuit.h(range(num_qubits))
        circuit.x(range(num_qubits))
        circuit.measure_all()
        circuit = transpile(circuit, backend, optimization_level=0)

        with self.subTest(msg="below fusion threshold"):
            options = self.fusion_options(enabled=True, threshold=num_qubits + 1)
            result = backend.run(circuit, shots=shots, **options).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertFalse(meta.get("enabled"))
            self.assertFalse(meta.get("applied"))

        with self.subTest(msg="above fusion threshold"):
            options = self.fusion_options(enabled=True, threshold=num_qubits - 1)
            result = backend.run(circuit, shots=shots, **options).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertFalse(meta.get("enabled"))
            self.assertFalse(meta.get("applied"))

    def test_fusion_verbose(self):
        """Test Fusion with verbose option"""
        shots = 100
        backend = self.backend(method="statevector")
        circuit = transpile(self.create_statevector_circuit(), backend, optimization_level=0)

        with self.subTest(msg="verbose enabled"):
            backend.set_options(**self.fusion_options(enabled=True, verbose=True, threshold=1))
            result = backend.run(circuit, shots=shots).result()
            # Assert fusion applied succesfully
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get("applied", False))
            # Assert verbose meta data in output
            self.assertIn("output_ops", meta)

        with self.subTest(msg="verbose disabled"):
            backend.set_options(**self.fusion_options(enabled=True, verbose=False, threshold=1))
            result = backend.run(circuit, shots=shots).result()
            # Assert fusion applied succesfully
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get("applied", False))
            # Assert verbose meta data not in output
            self.assertNotIn("output_ops", meta)

        with self.subTest(msg="verbose default"):
            backend.set_options(**self.fusion_options(enabled=True, threshold=1))
            result = backend.run(circuit, shots=shots).result()
            # Assert fusion applied succesfully
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get("applied", False))
            # Assert verbose meta data not in output
            self.assertNotIn("output_ops", meta)

    @supported_methods(["statevector", "density_matrix", "unitary", "superop"])
    def test_fusion_default(self, method, device):
        """Test fusion threshsold"""
        shots = 100
        num_qubits = 5
        backend = self.backend(method=method, device=device)
        circuit = transpile(QFT(num_qubits), backend, optimization_level=0)
        if method == "unitary":
            circuit.save_unitary()
        elif method == "superop":
            circuit.save_superop()
        else:
            circuit.measure_all()

        backend.set_options(**self.fusion_options(enabled=True))
        result = backend.run(circuit, shots=shots).result()

        expected_max_qubits = 5
        expected_threshold = 14
        if method == "density_matrix":
            expected_max_qubits = 2
            expected_threshold = 7
        elif method == "unitary":
            expected_max_qubits = 5
            expected_threshold = 7
        elif method == "superop":
            expected_max_qubits = 2
            expected_threshold = 7

        meta = result.results[0].metadata.get("fusion", None)
        self.assertEqual(meta.get("max_fused_qubits", None), expected_max_qubits)
        self.assertEqual(meta.get("threshold", None), expected_threshold)

    @supported_methods(["statevector", "density_matrix"])
    def test_kraus_noise_fusion(self, method, device):
        """Test Fusion with kraus noise model option"""
        shots = 100
        fusion_options = self.fusion_options(enabled=True, threshold=1)
        backend = self.backend(
            method=method, device=device, noise_model=self.noise_model_kraus(), **fusion_options
        )
        circuit = transpile(self.create_statevector_circuit(), backend, optimization_level=0)
        result = backend.run(circuit, shots=shots).result()
        meta = self.fusion_metadata(result)
        if method == "density_matrix":
            target_method = "superop"
        else:
            target_method = "kraus"
        self.assertSuccess(result)
        self.assertTrue(meta.get("applied", False), msg="fusion should have been applied.")
        self.assertEqual(meta.get("method", None), target_method)

    @supported_methods(["statevector", "density_matrix"])
    def test_non_kraus_noise_fusion(self, method, device):
        """Test Fusion with non-kraus noise model option"""
        shots = 100
        fusion_options = self.fusion_options(enabled=True, threshold=1)
        backend = self.backend(
            method=method, device=device, noise_model=self.noise_model_depol(), **fusion_options
        )
        circuit = transpile(self.create_statevector_circuit(), backend, optimization_level=0)
        result = backend.run(circuit, shots=shots).result()
        meta = self.fusion_metadata(result)
        if method == "density_matrix":
            target_method = "superop"
        else:
            target_method = "unitary"
        self.assertSuccess(result)
        self.assertTrue(meta.get("applied", False), msg="fusion should have been applied.")
        self.assertEqual(meta.get("method", None), target_method)

    def test_control_fusion(self):
        """Test Fusion enable/disable option"""
        shots = 100
        backend = self.backend(method="statevector")
        circuit = transpile(
            self.create_statevector_circuit(), backend=backend, optimization_level=0
        )

        with self.subTest(msg="fusion enabled"):
            backend_options = self.fusion_options(enabled=True, threshold=1)
            result = backend.run(circuit, shots=shots, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertTrue(meta.get("enabled"))
            self.assertTrue(meta.get("applied", False))

        with self.subTest(msg="fusion disabled"):
            backend_options = backend_options = self.fusion_options(enabled=False, threshold=1)
            result = backend.run(circuit, shots=shots, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertFalse(meta.get("enabled"))

        with self.subTest(msg="fusion default"):
            backend_options = self.fusion_options()
            result = backend.run(circuit, shots=shots, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertTrue(meta.get("enabled"))
            self.assertFalse(meta.get("applied", False), msg=meta)

    def test_fusion_operations(self):
        """Test Fusion enable/disable option"""
        shots = 100
        num_qubits = 8

        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)

        circuit.h(qr)
        circuit.barrier(qr)

        circuit.u(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[1])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)

        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[2], qr[3])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)

        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[2], qr[3])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)
        circuit.u(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)

        circuit.measure(qr, cr)

        backend = self.backend(method="statevector")
        circuit = transpile(circuit, backend, optimization_level=0)

        result_disabled = backend.run(
            circuit, **self.fusion_options(enabled=False, threshold=1)
        ).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        result_enabled = backend.run(
            circuit, **self.fusion_options(enabled=True, threshold=1)
        ).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, "success", "False"))
        self.assertTrue(getattr(result_enabled, "success", "False"))
        self.assertFalse(meta_disabled.get("enabled"))
        self.assertTrue(meta_enabled.get("enabled"))
        self.assertTrue(meta_enabled.get("applied"))
        self.assertDictAlmostEqual(
            result_enabled.get_counts(circuit),
            result_disabled.get_counts(circuit),
            delta=0.0,
            msg="fusion for qft was failed",
        )

    def test_fusion_qv(self):
        """Test Fusion with quantum volume"""
        shots = 100
        num_qubits = 6
        depth = 2
        backend = self.backend(method="statevector")
        circuit = transpile(
            QuantumVolume(num_qubits, depth, seed=0), backend=backend, optimization_level=0
        )
        circuit.measure_all()

        options_disabled = self.fusion_options(enabled=False, threshold=1, verbose=True)
        result_disabled = backend.run(circuit, shots=shots, **options_disabled).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        options_enabled = self.fusion_options(enabled=True, threshold=1, verbose=True)
        result_enabled = backend.run(circuit, shots=shots, **options_enabled).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, "success", "False"))
        self.assertTrue(getattr(result_enabled, "success", "False"))
        self.assertFalse(meta_disabled.get("applied", False))
        self.assertTrue(meta_enabled.get("applied", False))
        self.assertDictAlmostEqual(
            result_enabled.get_counts(circuit),
            result_disabled.get_counts(circuit),
            delta=0.0,
            msg="fusion for qft was failed",
        )

    def test_fusion_qft(self):
        """Test Fusion with qft"""
        shots = 100
        num_qubits = 8
        backend = self.backend(method="statevector")
        circuit = transpile(QFT(num_qubits), backend, optimization_level=0)
        circuit.measure_all()

        options_disabled = self.fusion_options(enabled=False, threshold=1)
        result_disabled = backend.run(circuit, shots=shots, **options_disabled).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        options_enabled = self.fusion_options(enabled=True, threshold=1)
        result_enabled = backend.run(circuit, shots=shots, **options_enabled).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, "success", "False"))
        self.assertTrue(getattr(result_enabled, "success", "False"))
        self.assertFalse(meta_disabled.get("enabled"))
        self.assertTrue(meta_enabled.get("enabled"))
        self.assertTrue(meta_enabled.get("applied"))
        self.assertDictAlmostEqual(
            result_enabled.get_counts(circuit),
            result_disabled.get_counts(circuit),
            delta=0.0,
            msg="fusion for qft was failed",
        )

    def test_fusion_parallelization(self):
        """Test Fusion parallelization option"""
        shots = 100
        num_qubits = 8
        depth = 2
        parallelization = 2

        backend = self.backend(method="statevector")
        circuit = transpile(
            QuantumVolume(num_qubits, depth, seed=0), backend=backend, optimization_level=0
        )
        circuit.measure_all()

        options_serial = self.fusion_options(enabled=True, threshold=1, parallelization=1)
        result_serial = backend.run(circuit, shots=shots, **options_serial).result()
        meta_serial = self.fusion_metadata(result_serial)

        options_parallel = self.fusion_options(enabled=True, threshold=1, parallelization=2)
        result_parallel = backend.run(circuit, shots=shots, **options_parallel).result()
        meta_parallel = self.fusion_metadata(result_parallel)

        self.assertTrue(getattr(result_serial, "success", "False"))
        self.assertTrue(getattr(result_parallel, "success", "False"))
        self.assertEqual(meta_serial.get("parallelization"), 1)
        self.assertEqual(meta_parallel.get("parallelization"), parallelization)
        self.assertDictAlmostEqual(
            result_parallel.get_counts(circuit),
            result_serial.get_counts(circuit),
            delta=0.0,
            msg="parallelized fusion was failed",
        )

    def test_fusion_two_qubits(self):
        """Test 2-qubit fusion"""
        shots = 100
        num_qubits = 8
        reps = 3
        backend = self.backend(method="statevector")

        circuit = RealAmplitudes(num_qubits=num_qubits, entanglement="linear", reps=reps)
        circuit.measure_all()

        np.random.seed(12345)
        param_binds = {}
        for param in circuit.parameters:
            param_binds[param] = np.random.random()

        circuit = transpile(circuit.assign_parameters(param_binds), backend, optimization_level=0)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        backend_options["fusion_verbose"] = True

        backend_options["fusion_enable.2_qubits"] = False
        result_disabled = backend.run(circuit, shots=shots, **backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        backend_options["fusion_enable.2_qubits"] = True
        result_enabled = backend.run(circuit, shots=shots, **backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, "success", "False"))
        self.assertTrue(getattr(result_enabled, "success", "False"))

        self.assertTrue(
            len(meta_enabled["output_ops"])
            if "output_ops" in meta_enabled
            else (
                len(circuit.ops) < len(meta_disabled["output_ops"])
                if "output_ops" in meta_disabled
                else len(circuit.ops)
            )
        )

    def test_fusion_diagonal(self):
        """Test diagonal fusion"""
        shots = 100
        num_qubits = 8
        backend = self.backend(method="statevector")

        circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit.p(0.1, i)

        for i in range(num_qubits - 1):
            circuit.cp(0.1, i, i + 1)

        circuit = transpile(circuit, backend, optimization_level=0)
        circuit.measure_all()

        backend_options = self.fusion_options(enabled=True, threshold=1)
        backend_options["fusion_verbose"] = True

        backend_options["fusion_enable.cost_base"] = False
        result = backend.run(circuit, shots=shots, **backend_options).result()
        meta = self.fusion_metadata(result)

        method = result.results[0].metadata.get("method")
        if method not in ["statevector"]:
            return

        for op in meta["output_ops"]:
            op_name = op["name"]
            if op_name == "measure":
                break
            self.assertEqual(op_name, "diagonal")

    def test_parallel_fusion_diagonal(self):
        """Test diagonal fusion with parallelization"""
        backend = self.backend(method="statevector")

        num_of_qubits = 10
        circuit = QuantumCircuit(num_of_qubits)
        size = 3

        for q in range(num_of_qubits):
            circuit.h(q)

        np.random.seed(12345)
        for qubit in range(num_of_qubits):
            for q in range(size):
                ctrl = (qubit + q) % num_of_qubits
                tgt = (ctrl + 1) % num_of_qubits
                circuit.cx(ctrl, tgt)

            circuit.p(np.random.random(), tgt)

            for q in range(size):
                ctrl = (qubit + size - q - 1) % num_of_qubits
                tgt = (ctrl + 1) % num_of_qubits
                circuit.cx(ctrl, tgt)

        circuit.save_statevector()

        thread = 8
        result = backend.run(
            circuit,
            fusion_threshold=1,
            **{
                "fusion_enable.diagonal": True,
                "fusion_enable.cost_based": False,
                "fusion_enable.n_qubits": False,
            },
            fusion_parallelization_threshold=3,
            max_parallel_threads=thread,
        ).result()
        actual = result.get_statevector(0)

        result = backend.run(circuit, fusion_enable=False).result()
        expected = result.get_statevector(0)

        self.assertEqual(actual, expected)
