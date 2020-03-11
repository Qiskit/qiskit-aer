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
from test.terra.reference import ref_2q_clifford
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError, depolarizing_error
from test.benchmark.tools import quantum_volume_circuit, qft_circuit

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

    def check_mat_exist(self, result):
        fusion_gates = result.to_dict(
        )['results'][0]['metadata']['fusion_verbose']
        for gate in fusion_gates:
            print(gate)

    def test_clifford_no_fusion(self):
        """Test Fusion with clifford simulator"""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_verbose'] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        self.assertTrue(getattr(result, 'success', False))

        self.assertTrue(
            'results' in result.to_dict(), msg="results must exist in result")
        self.assertTrue(
            'metadata' in result.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' not in result.to_dict()['results'][0]['metadata'],
            msg="fusion must not work for clifford")

    def test_noise_fusion(self):
        """Test Fusion with noise model option"""
        circuit = self.create_statevector_circuit()

        shots = 100
        noise_model = self.noise_model()
        circuit = transpile([circuit],
                            backend=self.SIMULATOR,
                            basis_gates=noise_model.basis_gates)
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = self.SIMULATOR.run(
            qobj,
            noise_model=noise_model,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result, 'success', False))

        self.assertTrue(
            'results' in result.to_dict(), msg="results must exist in result")
        self.assertTrue(
            'metadata' in result.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' in result.to_dict()['results'][0]['metadata'],
            msg="verbose must work with noise")

    def test_fusion_verbose(self):
        """Test Fusion with verbose option"""
        circuit = self.create_statevector_circuit()

        shots = 100
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1
    
        result_verbose = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_verbose, 'success', 'False'))
        self.assertTrue(
            'results' in result_verbose.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_verbose.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' in result_verbose.to_dict()['results'][0]
            ['metadata'],
            msg="fusion must work for satevector")

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = False
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_nonverbose = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_nonverbose, 'success', 'False'))
        self.assertTrue(
            'results' in result_nonverbose.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_nonverbose.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' not in result_nonverbose.to_dict()['results'][0]
            ['metadata'],
            msg="verbose must not work if fusion_verbose is False")

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_default = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        self.assertTrue(getattr(result_default, 'success', 'False'))
        self.assertTrue(
            'results' in result_default.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_default.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' not in result_default.to_dict()['results'][0]
            ['metadata'],
            msg="verbose must not work if fusion_verbose is False")

    def test_control_fusion(self):
        """Test Fusion enable/disable option"""
        shots = 100
        circuit = self.create_statevector_circuit()
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_verbose = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_verbose, 'success', 'False'))
        self.assertTrue(
            'results' in result_verbose.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_verbose.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' in result_verbose.to_dict()['results'][0]
            ['metadata'],
            msg="fusion must work for satevector")

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = False
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_disabled = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_disabled, 'success', 'False'))
        self.assertTrue(
            'results' in result_disabled.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_disabled.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' not in result_disabled.to_dict()['results'][0]
            ['metadata'],
            msg="fusion must not work with fusion_enable is False")

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_verbose'] = True

        result_default = self.SIMULATOR.run(
            qobj, backend_options=backend_options).result()
        self.assertTrue(getattr(result_default, 'success', 'False'))
        self.assertTrue(
            'results' in result_default.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_default.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' not in result_default.to_dict()['results'][0]
            ['metadata'],
            msg="fusion must not work by default for satevector")


    def test_default_fusion(self):
        """Test default Fusion option"""
        default_threshold = 20
        shots = 100
        circuit = qft_circuit(default_threshold - 1, measure=True)
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_verbose'] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_verbose = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_verbose, 'success', 'False'))
        self.assertTrue(
            'results' in result_verbose.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_verbose.to_dict()['results'][0],
            msg="metadata must not exist in results[0]")
        self.assertTrue(
            'fusion_verbose' not in result_verbose.to_dict()['results'][0]
            ['metadata'],
            msg="fusion must work for satevector")

        circuit = qft_circuit(default_threshold, measure=True)
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)
        result_verbose = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_verbose, 'success', 'False'))
        self.assertTrue(
            'results' in result_verbose.to_dict(),
            msg="results must exist in result")
        self.assertTrue(
            'metadata' in result_verbose.to_dict()['results'][0],
            msg="metadata must exist in results[0]")
        self.assertTrue(
            'fusion_verbose' in result_verbose.to_dict()['results'][0]
            ['metadata'],
            msg="fusion must work for satevector")
        
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

        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_fusion = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_fusion, 'success', 'False'))

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = False
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_nonfusion = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_nonfusion, 'success', 'False'))

        self.assertDictAlmostEqual(
            result_fusion.get_counts(circuit),
            result_nonfusion.get_counts(circuit),
            delta=0.0,
            msg="fusion x-x-x was failed")


    def test_fusion_qv(self):
        """Test Fusion with quantum volume"""
        shots = 100
        
        circuit = quantum_volume_circuit(10, 1, measure=True, seed=0)
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)
        
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_fusion = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_fusion, 'success', 'False'))

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = False
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_nonfusion = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_nonfusion, 'success', 'False'))
        
        self.assertDictAlmostEqual(
            result_fusion.get_counts(circuit),
            result_nonfusion.get_counts(circuit),
            delta=0.0,
            msg="fusion for qv was failed")
        
    def test_fusion_qft(self):
        """Test Fusion with qft"""
        shots = 100
        
        circuit = qft_circuit(10, measure=True)
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)
        
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_fusion = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_fusion, 'success', 'False'))

        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = False
        backend_options['fusion_verbose'] = True
        backend_options['fusion_threshold'] = 1
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result_nonfusion = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertTrue(getattr(result_nonfusion, 'success', 'False'))
        
        self.assertDictAlmostEqual(
            result_fusion.get_counts(circuit),
            result_nonfusion.get_counts(circuit),
            delta=0.0,
            msg="fusion for qft was failed")