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
QasmSimulator Integration Tests
"""
# pylint: disable=no-member

from test.terra.reference import ref_2q_clifford
from test.benchmark.tools import quantum_volume_circuit, qft_circuit

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError, depolarizing_error


class QasmFusionTests:
    """QasmSimulator fusion tests."""

    SIMULATOR = QasmSimulator()

    def create_statevector_circuit(self):
        """ Creates a simple circuit for running in the statevector """
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        circuit = QuantumCircuit(qr, cr)
        circuit.u3(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        return circuit

    def noise_model(self):
        """ Creates a new noise model for testing purposes """
        readout_error = [0.01, 0.1]
        depolarizing = {'u3': (1, 0.001), 'cx': (2, 0.02)}
        noise = NoiseModel()
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in depolarizing.items():
            noise.add_all_qubit_quantum_error(
                depolarizing_error(gate_error, num_qubits), gate)
            return noise

    def fusion_options(self, enabled=None, threshold=None, verbose=None):
        """Return default backend_options dict."""
        backend_options = self.BACKEND_OPTS.copy()
        if enabled is not None:
            backend_options['fusion_enable'] = enabled
        if verbose is not None:
            backend_options['fusion_verbose'] = verbose
        if threshold is not None:
            backend_options['fusion_threshold'] = threshold
        return backend_options

    def fusion_metadata(self, result):
        """Return fusion metadata dict"""
        metadata = result.results[0].metadata
        return metadata.get('fusion', {})

    def test_fusion_theshold(self):
        """Test fusion threhsold"""
        shots = 100
        threshold = 10
        backend_options = self.fusion_options(enabled=True, threshold=threshold)

        with self.subTest(msg='below fusion threshold'):
            circuit = qft_circuit(threshold - 1, measure=True)
            qobj = assemble([circuit],
                            self.SIMULATOR,
                            shots=shots)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertTrue(getattr(result, 'success', False))
            self.assertFalse(meta.get('applied', False))

        with self.subTest(msg='at fusion threshold'):
            circuit = qft_circuit(threshold, measure=True)
            qobj = assemble([circuit],
                            self.SIMULATOR,
                            shots=shots)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertTrue(getattr(result, 'success', False))
            self.assertTrue(meta.get('applied', False))

        with self.subTest(msg='above fusion threshold'):
            circuit = qft_circuit(threshold + 1, measure=True)
            qobj = assemble([circuit],
                            self.SIMULATOR,
                            shots=shots)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertTrue(getattr(result, 'success', False))
            self.assertTrue(meta.get('applied', False))

    def test_fusion_verbose(self):
        """Test Fusion with verbose option"""
        circuit = self.create_statevector_circuit()
        shots = 100
        qobj = assemble([circuit], self.SIMULATOR, shots=shots)

        with self.subTest(msg='verbose enabled'):
            backend_options = self.fusion_options(enabled=True, verbose=True, threshold=1)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            # Assert fusion applied succesfully
            self.assertTrue(getattr(result, 'success', False))
            self.assertTrue(meta.get('applied', False))
            # Assert verbose meta data in output
            self.assertIn('input_ops', meta)
            self.assertIn('output_ops', meta)

        with self.subTest(msg='verbose disabled'):
            backend_options = self.fusion_options(enabled=True, verbose=False, threshold=1)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            # Assert fusion applied succesfully
            self.assertTrue(getattr(result, 'success', False))
            self.assertTrue(meta.get('applied', False))
            # Assert verbose meta data not in output
            self.assertNotIn('input_ops', meta)
            self.assertNotIn('output_ops', meta)

        with self.subTest(msg='verbose default'):
            backend_options = self.fusion_options(enabled=True, threshold=1)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            # Assert fusion applied succesfully
            self.assertTrue(getattr(result, 'success', False))
            self.assertTrue(meta.get('applied', False))
            # Assert verbose meta data not in output
            self.assertNotIn('input_ops', meta)
            self.assertNotIn('output_ops', meta)

    def test_noise_fusion(self):
        """Test Fusion with noise model option"""
        shots = 100
        circuit = self.create_statevector_circuit()
        noise_model = self.noise_model()
        circuit = transpile([circuit],
                            backend=self.SIMULATOR,
                            basis_gates=noise_model.basis_gates)
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)
        backend_options = self.fusion_options(enabled=True, threshold=1)
        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    backend_options=backend_options).result()
        meta = self.fusion_metadata(result)

        self.assertTrue(getattr(result, 'success', False))
        self.assertTrue(meta.get('applied', False),
                        msg='fusion should have been applied.')

    def test_control_fusion(self):
        """Test Fusion enable/disable option"""
        shots = 100
        circuit = self.create_statevector_circuit()
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        with self.subTest(msg='fusion enabled'):
            backend_options = self.fusion_options(enabled=True, threshold=1)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertTrue(getattr(result, 'success', False))
            self.assertTrue(meta.get('applied', False))

        with self.subTest(msg='fusion disabled'):
            backend_options = backend_options = self.fusion_options(enabled=False, threshold=1)
            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertTrue(getattr(result, 'success', False))
            self.assertFalse(meta.get('applied', False))

        with self.subTest(msg='fusion default'):
            backend_options = self.fusion_options()

            result = self.SIMULATOR.run(
                qobj, backend_options=backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertTrue(getattr(result, 'success', False))
            self.assertFalse(meta.get('applied', False))

    def test_fusion_operations(self):
        """Test Fusion enable/disable option"""
        shots = 100

        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        circuit = QuantumCircuit(qr, cr)

        for i in range(10):
            circuit.h(qr[i])
            circuit.barrier(qr)

        circuit.u3(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[0])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[1])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[3])
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
        circuit.u3(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[3])
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
        circuit.u3(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1, 0.1, 0.1, qr[3])
        circuit.barrier(qr)

        circuit.measure(qr, cr)

        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=False, threshold=1)
        result_disabled = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        result_enabled = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        self.assertEqual(meta_disabled, {})
        self.assertTrue(meta_enabled.get('applied', False))
        self.assertDictAlmostEqual(result_enabled.get_counts(circuit),
                                   result_disabled.get_counts(circuit),
                                   delta=0.0,
                                   msg="fusion for qft was failed")

    def test_fusion_qv(self):
        """Test Fusion with quantum volume"""
        shots = 100

        circuit = quantum_volume_circuit(10, 2, measure=True, seed=0)
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=False, threshold=1)
        result_disabled = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        result_enabled = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        self.assertEqual(meta_disabled, {})
        self.assertTrue(meta_enabled.get('applied', False))
        self.assertDictAlmostEqual(result_enabled.get_counts(circuit),
                                   result_disabled.get_counts(circuit),
                                   delta=0.0,
                                   msg="fusion for qft was failed")

    def test_fusion_qft(self):
        """Test Fusion with qft"""
        shots = 100

        circuit = qft_circuit(10, measure=True)
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=False, threshold=1)
        result_disabled = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        result_enabled = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        self.assertEqual(meta_disabled, {})
        self.assertTrue(meta_enabled.get('applied', False))
        self.assertDictAlmostEqual(result_enabled.get_counts(circuit),
                                   result_disabled.get_counts(circuit),
                                   delta=0.0,
                                   msg="fusion for qft was failed")
