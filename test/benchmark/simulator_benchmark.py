# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Base class of Qiskit Aer Benchmarking
"""
import sys
import numpy as np
from time import time
from qiskit.compiler import transpile, assemble
from qiskit.providers.aer import QasmSimulator, UnitarySimulator
from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

from benchmark.circuit_library_circuits import CircuitLibraryCircuits

QOBJS = {}
QASM_SIMULATOR = QasmSimulator()

class SimulatorBenchmarkSuite(CircuitLibraryCircuits):

    RUNTIME_STATEVECTOR_CPU = 'statevector'
    RUNTIME_STATEVECTOR_GPU = 'statevector_gpu'
    RUNTIME_MPS_CPU = 'matrix_product_state'
    RUNTIME_DENSITY_MATRIX_CPU = 'density_matrix'
    RUNTIME_DENSITY_MATRIX_GPU = 'density_matrix_gpu'
    RUNTIME_STABILIZER_CPU = 'stabilizer'
    RUNTIME_EXTENDED_STABILIZER_CPU = 'extended_stabilizer'
    RUNTIME_UNITARY_MATRIX_CPU = 'unitary_matrix'
    RUNTIME_UNITARY_MATRIX_GPU = 'unitary_matrix_gpu'
    
    RUNTIME_CPU = [
        RUNTIME_STATEVECTOR_CPU,
        RUNTIME_MPS_CPU,
        RUNTIME_DENSITY_MATRIX_CPU,
        RUNTIME_STABILIZER_CPU,
        RUNTIME_EXTENDED_STABILIZER_CPU,
        RUNTIME_UNITARY_MATRIX_CPU
        ]
    
    RUNTIME_GPU = [
        RUNTIME_STATEVECTOR_GPU,
        RUNTIME_DENSITY_MATRIX_GPU,
        RUNTIME_UNITARY_MATRIX_GPU
        ]
    
    DEFAULT_RUNTIME = [
        RUNTIME_STATEVECTOR_CPU,
        RUNTIME_MPS_CPU,
        RUNTIME_DENSITY_MATRIX_CPU,
        RUNTIME_STATEVECTOR_GPU
        ]
    
    DEFAULT_QUBITS = [10, 15, 20, 25]
    
    MEASUREMENT_SAMPLING = 'sampling'
    MEASUREMENT_EXPVAL = 'expval'
    
    DEFAULT_MEASUREMENT_METHODS = [ MEASUREMENT_SAMPLING ]
    DEFAULT_MEASUREMENT_COUNTS = [ 1000 ]

    NOISE_IDEAL = 'ideal'
    NOISE_DAMPING = 'damping'
    NOISE_DEPOLARIZING = 'depolarizing'
    
    DEFAULT_NOISE_MODELS = [ NOISE_IDEAL ]
    
    def __init__(self,
                 name = 'simulator_benchmark',
                 apps = {},
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = DEFAULT_NOISE_MODELS):
        self.timeout = 60 * 10
        self.__name__ = name
        
        self.apps = apps if isinstance(apps, list) else [app for app in apps]
        self.reps = [None] * len(apps) if isinstance(apps, list) else [apps[app] for app in apps]
        self.qubits = qubits
        self.runtime_names = runtime_names
        self.measures = measures
        self.measure_counts = measure_counts
        self.noise_model_names = noise_model_names 

        self.params = (self.apps, self.measures, self.measure_counts, self.noise_model_names, self.qubits)
        self.param_names = ["application", "measure_method", "measure_counts", "noise", "qubit", "repeats"]
        
        all_simulators = [ QASM_SIMULATOR ]
        
        self.simulators = {}
        self.backend_options_list = {}
        self.backend_qubits = {}

        self.noise_models = {}
        self.noise_models[self.NOISE_IDEAL] = None
        if self.NOISE_DAMPING in self.noise_model_names:
            noise_model = NoiseModel()
            error = amplitude_damping_error(1e-3)
            noise_model.add_all_qubit_quantum_error(error, ['u3'])
            self.noise_models[self.NOISE_DAMPING] = noise_model
        if self.NOISE_DEPOLARIZING in self.noise_model_names:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ['u3'])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ['cx'])
            self.noise_models[self.NOISE_DEPOLARIZING] = noise_model

        if self.RUNTIME_STATEVECTOR_CPU in runtime_names:
            self.simulators[self.RUNTIME_STATEVECTOR_CPU] = QASM_SIMULATOR
            self.backend_options_list[self.RUNTIME_STATEVECTOR_CPU] = { 'method': self.RUNTIME_STATEVECTOR_CPU }
            self.backend_qubits[self.RUNTIME_STATEVECTOR_CPU] = self.qubits
        
        if self.RUNTIME_STATEVECTOR_GPU in runtime_names:
            self.simulators[self.RUNTIME_STATEVECTOR_GPU] = QASM_SIMULATOR
            self.backend_options_list[self.RUNTIME_STATEVECTOR_GPU] = { 'method': self.RUNTIME_STATEVECTOR_GPU }
            self.backend_qubits[self.RUNTIME_STATEVECTOR_GPU] = self.qubits
        
        if self.RUNTIME_MPS_CPU in runtime_names:
            self.simulators[self.RUNTIME_MPS_CPU] = QASM_SIMULATOR
            self.backend_options_list[self.RUNTIME_MPS_CPU] = { 'method': self.RUNTIME_MPS_CPU }
            self.backend_qubits[self.RUNTIME_MPS_CPU] = self.qubits
        
        if self.RUNTIME_DENSITY_MATRIX_CPU in runtime_names:
            self.simulators[self.RUNTIME_DENSITY_MATRIX_CPU] = QASM_SIMULATOR
            self.backend_options_list[self.RUNTIME_DENSITY_MATRIX_CPU] = { 'method': self.RUNTIME_DENSITY_MATRIX_CPU }
            self.backend_qubits[self.RUNTIME_DENSITY_MATRIX_CPU] = [qubit for qubit in qubits if qubit <= 15]
        
        if self.RUNTIME_DENSITY_MATRIX_GPU in runtime_names:
            self.simulators[self.RUNTIME_DENSITY_MATRIX_GPU] = QASM_SIMULATOR
            self.backend_options_list[self.RUNTIME_DENSITY_MATRIX_GPU] = { 'method': self.RUNTIME_DENSITY_MATRIX_GPU }
            self.backend_qubits[self.RUNTIME_DENSITY_MATRIX_GPU] = [qubit for qubit in qubits if qubit <= 15]
        
        #if self.RUNTIME_STABILIZER_CPU in runtime_names:
        #    self.simulators[self.RUNTIME_STABILIZER_CPU] = QASM_SIMULATOR
        #    self.backend_options_list[self.RUNTIME_STABILIZER_CPU] = { 'method': self.RUNTIME_STABILIZER_CPU }
        #    self.backend_qubits[self.RUNTIME_STABILIZER_CPU] = self.qubits
        
        #if self.RUNTIME_EXTENDED_STABILIZER_CPU in runtime_names:
        #    self.simulators[self.RUNTIME_EXTENDED_STABILIZER_CPU] = QASM_SIMULATOR
        #    self.backend_options_list[self.RUNTIME_EXTENDED_STABILIZER_CPU] = { 'method': self.RUNTIME_EXTENDED_STABILIZER_CPU }
        #    self.backend_qubits[self.RUNTIME_EXTENDED_STABILIZER_CPU] = self.qubits
        
        def add_measure_all(base):
            circuit = base.copy()
            circuit.measure_all()
            return circuit
        
        def add_expval(base, num_terms):
            circuit = base.copy()
            from qiskit.providers.aer.extensions import snapshot_expectation_value
            from numpy.random import default_rng
            rng = default_rng(1)
            paulis = [''.join(s) for s in 
                      rng.choice(['I', 'X', 'Y', 'Z'], size=(num_terms, qubit))]
            pauli_op = [(1 / num_terms, pauli) for pauli in paulis]
            circuit.snapshot_expectation_value('expval', pauli_op, range(qubit))
            return circuit

        for app, rep in zip(self.apps, self.reps):
            for qubit in self.qubits:
                if (all_simulators[0], app, self.measures[0], self.measure_counts[0], qubit) in QOBJS:
                    continue
                base_circuit = None
                try:
                    base_circuit = eval('self.{0}'.format(app))(qubit, rep)
                    print('circuit construction: circ={0}, qubit={1}'.format(app, qubit), file=sys.stderr)
                except ValueError as e:
                    print('circuit construction error: circ={0}, qubit={1}, info={2}'.format(app, qubit, e), file=sys.stderr)
                    continue
                
                if len(base_circuit.parameters) > 0:
                    param_binds = {}
                    for param in base_circuit.parameters:
                        param_binds[param] = np.random.random()
                    base_circuit = base_circuit.bind_parameters(param_binds)

                for measure in self.measures:
                    if measure == self.MEASUREMENT_SAMPLING:
                        circuit = add_measure_all(base_circuit)
                        for measure_count in self.measure_counts:
                            for simulator in all_simulators:
                                QOBJS[(simulator, app, measure, measure_count, qubit)] = assemble(circuit, simulator, shots=measure_count)
                        
                    elif measure == self.MEASUREMENT_EXPVAL:
                        for measure_count in self.measure_counts:
                            circuit = add_expval(base_circuit, measure_count)
                            for simulator in all_simulators:
                                QOBJS[(simulator, app, measure, measure_count, qubit)] = assemble(circuit, simulator, shots=1)

    def _transpile(self, circuit, basis_gates):
        from qiskit import transpile
        return transpile(circuit, basis_gates=basis_gates)

    def transpile(self, circuit):
        return self._transpile(circuit, [
            'u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg',
            't', 'tdg', 'swap', 'ccx', 'unitary', 'diagonal', 'initialize',
            'cu1', 'cu2', 'cu3', 'cswap', 'mcx', 'mcy', 'mcz',
            'mcu1', 'mcu2', 'mcu3', 'mcswap', 'multiplexer', 'kraus', 'roerror'])
    
#     def transpile_for_mps(self, circuit):
#         return self.transpile([
#             'u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg',
#             't', 'tdg', 'swap', 'ccx'#, 'unitary', 'diagonal', 'initialize',
#             'cu1', #'cu2', 'cu3', 'cswap', 'mcx', 'mcy', 'mcz',
#             #'mcu1', 'mcu2', 'mcu3', 'mcswap', 'multiplexer', 'kraus', 
#             'roerror'
#             ])

    def _run(self, runtime, app, measure, measure_count, noise_name, qubit):
        if runtime not in self.simulators or runtime not in self.backend_options_list:
            raise ValueError('unknown runtime: {0}'.format(runtime))
        simulator = self.simulators[runtime]
        backend_options = self.backend_options_list[runtime]
        noise_model = self.noise_models[noise_name]
        
        if qubit not in self.backend_qubits[runtime]:
            raise ValueError('out of qubit range: qubit={0}, list={1}'.format(qubit, self.backend_qubits[runtime]))
        
        if (simulator, app, measure, measure_count, qubit) not in QOBJS:
            raise ValueError('no qobj: measure={0}:{1}, qubit={2}'.format(measure, measure_count, qubit))
        
        qobj = QOBJS[(simulator, app, measure, measure_count, qubit)]
        
        result = simulator.run(qobj, backend_options=backend_options, noise_model=noise_model).result()
        if result.status != 'COMPLETED':
            try:
                reason = None
                ret_dict = result.to_dict()
                if 'results' in ret_dict:
                    if len (ret_dict['results']) > 0 and 'status' in ret_dict['results'][0]:
                        reason = ret_dict['results'][0]['status']
                if reason is None and 'status' in ret_dict:
                    reason = ret_dict['status']
                if reason is None:
                    reason = 'unknown'
            except:
                reason = 'unknown'
            raise ValueError('simulation error ({0})'.format(reason))
    
    #def time_statevector(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_STATEVECTOR_CPU, app, measure, measure_count, noise_name, qubit)

    #def time_statevector_gpu(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_STATEVECTOR_GPU, app, measure, measure_count, noise_name, qubit)

    #def time_matrix_product_state(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_MPS_CPU, app, measure, measure_count, noise_name, qubit)
        
    #def time_density_matrix(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_DENSITY_MATRIX_CPU, app, measure, measure_count, noise_name, qubit)
        
    #def time_density_matrix_gpu(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_DENSITY_MATRIX_GPU, app, qubit)
        
    #def time_stabilizer(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_STABILIZER_CPU, app, measure, measure_count, noise_name, qubit)
        
    #def time_extended_stabilizer(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_EXTENDED_STABILIZER_CPU, app, measure, measure_count, noise_name, qubit)
        
    #def time_unitary_matrix(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_UNITARY_MATRIX_CPU, app, qubit)
        
    #def time_unitary_matrix_gpu(self, app, measure, measure_count, noise_name, qubit):
    #    self._run(self.RUNTIME_UNITARY_MATRIX_GPU, app, qubit)

    
    def run_manual(self):
        import timeout_decorator
        @timeout_decorator.timeout(self.timeout)
        def run_with_timeout (suite, runtime, app, measure, measure_count, noise_name, qubit):
            start = time()
            eval('suite.time_{0}'.format(runtime))(app, measure, measure_count, noise_name, qubit)
            return time() - start
        
        #for runtime in self.runtime_names:
        for noise_name in self.noise_model_names:
            for runtime in self.runtime_names:
                for app, repeats in zip(self.apps, self.reps):
                    app_name = app if repeats is None else '{0}:{1}'.format(app, repeats)
                    for qubit in self.qubits:
                        for measure in self.measures:
                            for measure_count in self.measure_counts:
                                print ('{0},{1},{2},{3},{4},{5},{6},'.format(self.__name__, app_name, runtime, measure, measure_count, noise_name, qubit), end="")
                                try:
                                    elapsed = run_with_timeout(self, runtime, app, measure, measure_count, noise_name, qubit)
                                    print ('{0}'.format(elapsed))
                                except ValueError as e:
                                    print ('{0}'.format(e))
                                except:
                                    import traceback
                                    traceback.print_exc(file=sys.stderr)
                                    print ('unknown error')
