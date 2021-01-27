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
import copy
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QuantumVolume, QFT, RealAmplitudes
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import (ReadoutError,
                                               depolarizing_error,
                                               amplitude_damping_error)


class QasmFusionTests:
    """QasmSimulator fusion tests."""

    SIMULATOR = QasmSimulator()

    def create_statevector_circuit(self):
        """ Creates a simple circuit for running in the statevector """
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
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

    def noise_model_depol(self):
        """ Creates a new noise model for testing purposes """
        readout_error = [0.01, 0.1]
        params = {'u3': (1, 0.001), 'cx': (2, 0.02)}
        noise = NoiseModel()
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in params.items():
            noise.add_all_qubit_quantum_error(
                depolarizing_error(gate_error, num_qubits), gate)
            return noise

    def noise_model_kraus(self):
        """ Creates a new noise model for testing purposes """
        readout_error = [0.01, 0.1]
        params = {'u3': (1, 0.001), 'cx': (2, 0.02)}
        noise = NoiseModel()
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in params.items():
            noise.add_all_qubit_quantum_error(
                amplitude_damping_error(gate_error, num_qubits), gate)
            return noise

    def fusion_options(self, enabled=None, threshold=None, verbose=None, 
                       parallelization=None):
        """Return default backend_options dict."""
        backend_options = self.BACKEND_OPTS.copy()
        if enabled is not None:
            backend_options['fusion_enable'] = enabled
        if verbose is not None:
            backend_options['fusion_verbose'] = verbose
        if threshold is not None:
            backend_options['fusion_threshold'] = threshold
        if parallelization is not None:
            backend_options['fusion_parallelization_threshold'] = 1
            backend_options['max_parallel_threads'] = parallelization
            backend_options['max_parallel_shots'] = 1
            backend_options['max_parallel_state_update'] = parallelization
        return backend_options

    def fusion_metadata(self, result):
        """Return fusion metadata dict"""
        metadata = result.results[0].metadata
        return metadata.get('fusion', {})

    def test_fusion_theshold(self):
        """Test fusion threhsold"""
        shots = 100
        threshold = 6
        backend_options = self.fusion_options(enabled=True, threshold=threshold)

        with self.subTest(msg='below fusion threshold'):
            circuit = transpile(QFT(threshold - 1),
                                self.SIMULATOR, basis_gates=['u1', 'u2', 'u3', 'cx', 'cz'],
                                optimization_level=0)
            circuit.measure_all()
            qobj = assemble(circuit, self.SIMULATOR, shots=shots)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get('enabled'))
            self.assertFalse(meta.get('applied'))

        with self.subTest(msg='at fusion threshold'):
            circuit = transpile(QFT(threshold),
                                self.SIMULATOR, basis_gates=['u1', 'u2', 'u3', 'cx', 'cz'],
                                optimization_level=0)
            circuit.measure_all()
            qobj = assemble(circuit, self.SIMULATOR, shots=shots)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get('enabled'))
            self.assertFalse(meta.get('applied'))

        with self.subTest(msg='above fusion threshold'):
            circuit = transpile(QFT(threshold + 1),
                                self.SIMULATOR, basis_gates=['u1', 'u2', 'u3', 'cx', 'cz'],
                                optimization_level=0)
            circuit.measure_all()
            qobj = assemble(circuit, self.SIMULATOR, shots=shots)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get('enabled'))
            self.assertTrue(meta.get('applied'))

    def test_fusion_verbose(self):
        """Test Fusion with verbose option"""
        circuit = self.create_statevector_circuit()
        shots = 100
        qobj = assemble(transpile([circuit], self.SIMULATOR, optimization_level=0),
                        shots=shots)

        with self.subTest(msg='verbose enabled'):
            backend_options = self.fusion_options(enabled=True, verbose=True, threshold=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            # Assert fusion applied succesfully
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get('applied', False))
            # Assert verbose meta data in output
            self.assertIn('output_ops', meta)

        with self.subTest(msg='verbose disabled'):
            backend_options = self.fusion_options(enabled=True, verbose=False, threshold=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            # Assert fusion applied succesfully
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get('applied', False))
            # Assert verbose meta data not in output
            self.assertNotIn('output_ops', meta)

        with self.subTest(msg='verbose default'):
            backend_options = self.fusion_options(enabled=True, threshold=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()

            # Assert fusion applied succesfully
            self.assertSuccess(result)
            meta = self.fusion_metadata(result)
            self.assertTrue(meta.get('applied', False))
            # Assert verbose meta data not in output
            self.assertNotIn('output_ops', meta)

    def test_kraus_noise_fusion(self):
        """Test Fusion with kraus noise model option"""
        shots = 100
        circuit = self.create_statevector_circuit()
        noise_model = self.noise_model_kraus()
        circuit = transpile([circuit],
                            backend=self.SIMULATOR,
                            basis_gates=noise_model.basis_gates,
                            optimization_level=0)
        qobj = assemble(circuit,
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)
        backend_options = self.fusion_options(enabled=True, threshold=1)
        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **backend_options).result()
        method = result.results[0].metadata.get('method')
        meta = self.fusion_metadata(result)
        if method in ['density_matrix', 'density_matrix_thrust', 'density_matrix_gpu']:
            target_method = 'superop'
        else:
            target_method = 'kraus'

        self.assertSuccess(result)
        self.assertTrue(meta.get('applied', False),
                        msg='fusion should have been applied.')
        self.assertEqual(meta.get('method', None), target_method)

    def test_non_kraus_noise_fusion(self):
        """Test Fusion with non-kraus noise model option"""
        shots = 100
        circuit = self.create_statevector_circuit()
        noise_model = self.noise_model_depol()
        circuit = transpile([circuit],
                            backend=self.SIMULATOR,
                            basis_gates=noise_model.basis_gates,
                            optimization_level=0)
        qobj = assemble(circuit,
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)
        backend_options = self.fusion_options(enabled=True, threshold=1)
        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **backend_options).result()
        meta = self.fusion_metadata(result)
        method = result.results[0].metadata.get('method')
        if method in ['density_matrix', 'density_matrix_thrust', 'density_matrix_gpu']:
            target_method = 'superop'
        else:
            target_method = 'unitary'

        self.assertSuccess(result)
        self.assertTrue(meta.get('applied', False),
                        msg='fusion should have been applied.')
        self.assertEqual(meta.get('method', None), target_method)

    def test_control_fusion(self):
        """Test Fusion enable/disable option"""
        shots = 100
        circuit = transpile(self.create_statevector_circuit(),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        with self.subTest(msg='fusion enabled'):
            backend_options = self.fusion_options(enabled=True, threshold=1)
            result = self.SIMULATOR.run(
                copy.deepcopy(qobj), **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertTrue(meta.get('enabled'))
            self.assertTrue(meta.get('applied', False))

        with self.subTest(msg='fusion disabled'):
            backend_options = backend_options = self.fusion_options(enabled=False, threshold=1)
            result = self.SIMULATOR.run(
                copy.deepcopy(qobj), **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertFalse(meta.get('enabled'))

        with self.subTest(msg='fusion default'):
            backend_options = self.fusion_options()

            result = self.SIMULATOR.run(
                copy.deepcopy(qobj), **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertTrue(meta.get('enabled'))
            self.assertFalse(meta.get('applied', False), msg=meta)

    def test_fusion_operations(self):
        """Test Fusion enable/disable option"""
        shots = 100
        num_qubits = 8

        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)

        circuit.h(qr)
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

        circuit = transpile(circuit,
                            backend=self.SIMULATOR,
                            optimization_level=0)
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=False, threshold=1)
        result_disabled = self.SIMULATOR.run(
            qobj, **backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        result_enabled = self.SIMULATOR.run(
            qobj, **backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        self.assertFalse(meta_disabled.get('enabled'))
        self.assertTrue(meta_enabled.get('enabled'))
        self.assertTrue(meta_enabled.get('applied'))
        self.assertDictAlmostEqual(result_enabled.get_counts(circuit),
                                   result_disabled.get_counts(circuit),
                                   delta=0.0,
                                   msg="fusion for qft was failed")

    def test_fusion_qv(self):
        """Test Fusion with quantum volume"""
        shots = 100
        num_qubits = 6
        depth = 2
        circuit = transpile(QuantumVolume(num_qubits, depth, seed=0),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        qobj_disabled = assemble([circuit], self.SIMULATOR, shots=shots,
                                 **self.fusion_options(enabled=False, threshold=1, verbose=True))
        result_disabled = self.SIMULATOR.run(qobj_disabled).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        qobj_enabled = assemble([circuit], self.SIMULATOR, shots=shots,
                                **self.fusion_options(enabled=True, threshold=1, verbose=True))
        result_enabled = self.SIMULATOR.run(qobj_enabled).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        self.assertFalse(meta_disabled.get('applied', False))
        self.assertTrue(meta_enabled.get('applied', False))
        self.assertDictAlmostEqual(result_enabled.get_counts(circuit),
                                   result_disabled.get_counts(circuit),
                                   delta=0.0,
                                   msg="fusion for qft was failed")

    def test_fusion_qft(self):
        """Test Fusion with qft"""
        shots = 100
        num_qubits = 8
        circuit = transpile(QFT(num_qubits),
                            backend=self.SIMULATOR,
                            basis_gates=['u1', 'u2', 'u3', 'cx', 'cz'],
                            optimization_level=0)
        circuit.measure_all()
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=False, threshold=1)
        result_disabled = self.SIMULATOR.run(
            qobj, **backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        result_enabled = self.SIMULATOR.run(
            qobj, **backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)

        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        self.assertFalse(meta_disabled.get('enabled'))
        self.assertTrue(meta_enabled.get('enabled'))
        self.assertTrue(meta_enabled.get('applied'))
        self.assertDictAlmostEqual(result_enabled.get_counts(circuit),
                                   result_disabled.get_counts(circuit),
                                   delta=0.0,
                                   msg="fusion for qft was failed")

    def test_fusion_parallelization(self):
        """Test Fusion parallelization option"""
        shots = 100
        num_qubits = 8
        depth = 2
        parallelization = 2

        circuit = transpile(QuantumVolume(num_qubits, depth, seed=0),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=True, threshold=1,
                                              parallelization=1)
        result_serial = self.SIMULATOR.run(qobj, **backend_options).result()
        meta_serial = self.fusion_metadata(result_serial)

        backend_options = self.fusion_options(enabled=True, threshold=1, 
                                              parallelization=2)
        result_parallel = self.SIMULATOR.run(qobj, **backend_options).result()
        meta_parallel = self.fusion_metadata(result_parallel)

        self.assertTrue(getattr(result_serial, 'success', 'False'))
        self.assertTrue(getattr(result_parallel, 'success', 'False'))
        self.assertEqual(meta_serial.get('parallelization'), 1)
        self.assertEqual(meta_parallel.get('parallelization'), parallelization)
        self.assertDictAlmostEqual(result_parallel.get_counts(circuit),
                                   result_serial.get_counts(circuit),
                                   delta=0.0,
                                   msg="parallelized fusion was failed")

    def test_fusion_two_qubits(self):
        """Test 2-qubit fusion"""
        shots = 100
        num_qubits = 8
        reps = 3
        
        circuit = RealAmplitudes(num_qubits=num_qubits, entanglement='linear', reps=reps)
        param_binds = {}
        for param in circuit.parameters:
            param_binds[param] = np.random.random()

        circuit = transpile(circuit.bind_parameters(param_binds),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        backend_options['fusion_verbose'] =  True
        
        backend_options['fusion_enable.2_qubits'] =  False
        result_disabled = self.SIMULATOR.run(qobj, **backend_options).result()
        meta_disabled = self.fusion_metadata(result_disabled)
        
        backend_options['fusion_enable.2_qubits'] =  True
        result_enabled = self.SIMULATOR.run(qobj, **backend_options).result()
        meta_enabled = self.fusion_metadata(result_enabled)
        
        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(getattr(result_enabled, 'success', 'False'))
        
        self.assertTrue(len(meta_enabled['output_ops']) if 'output_ops' in meta_enabled else len(circuit.ops) < 
                        len(meta_disabled['output_ops']) if 'output_ops' in meta_disabled else len(circuit.ops))

    def test_fusion_diagonal(self):
        """Test diagonal fusion"""
        shots = 100
        num_qubits = 8

        circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit.p(0.1, i)
        
        for i in range(num_qubits - 1):
            circuit.cp(0.1, i, i + 1)
        
        circuit = transpile(circuit,
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        
        qobj = assemble([circuit],
                        self.SIMULATOR,
                        shots=shots,
                        seed_simulator=1)

        backend_options = self.fusion_options(enabled=True, threshold=1)
        backend_options['fusion_verbose'] =  True
        
        backend_options['fusion_enable.cost_base'] =  False
        result = self.SIMULATOR.run(qobj, **backend_options).result()
        meta = self.fusion_metadata(result)
        
        method = result.results[0].metadata.get('method')
        if method not in ['statevector']:
            return

        for op in meta['output_ops']:
            op_name = op['name']
            if op_name == 'measure':
                break
            self.assertEqual(op_name, 'diagonal')
