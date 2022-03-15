# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import numpy as np
from time import time
from abc import ABC

from qiskit import QuantumCircuit, transpile
import qiskit.quantum_info as qi
from qiskit.circuit.library import QuantumVolume, QFT, RealAmplitudes
from qiskit.providers.aer.noise import NoiseModel, ReadoutError, amplitude_damping_error, depolarizing_error, phase_amplitude_damping_error

class _Base(ABC):

    def __init__(self,
                 qubits=[5, 15, 25]):

        self._simulator = None
        self.qubits = qubits
        self.params = (qubits)
        self.param_names = ["qubits"]
        
    def simulator(self):
        if self._simulator:
            return self._simulator
        try:
            from qiskit.providers.aer import AerSimulator
            self._simulator = AerSimulator()
        except ImportError:
            from qiskit.providers.aer import QasmSimulator
            self._simulator = QasmSimulator()
        return self._simulator

    def _run(self, circuit, *args, **kwargs):
        if self.simulator().__class__.__name__ == 'AerSimulator':
            result = self.simulator().run(circuit, *args, **kwargs).result()
            if not result.success:
                raise ValueError(result.status)            
        elif self.simulator().__class__.__name__ in ('QasmSimulator', 'UnitarySimulator'):
            from qiskit import assemble
            circuit = assemble(circuit, self.simulator())
            result = self.simulator().run(circuit, *args, **kwargs).result()
            if not result.success:
                raise ValueError(result.status)            
        else:
            raise ValueError(f'unknown simulator class: {self._simulator.__class__.__name__}')


class Benchmark(_Base):

    def __init__(self,
                 qubits=[5, 15, 25]):
        
        super().__init__(qubits)
        self._qv_circs = {qubit: self._setup_circuit(QuantumVolume(qubit)) for qubit in qubits}
        self._qft_circs = {qubit: self._setup_circuit(QFT(qubit)) for qubit in qubits}
        self._ra_circs = {qubit: self._setup_circuit(RealAmplitudes(qubit)) for qubit in qubits}
        self._ra_full_circs = {qubit: self._setup_circuit(RealAmplitudes(qubit, entanglement='full')) for qubit in qubits}

    def _setup_circuit(self, circuit):
        self.add_measure(circuit)
        circuit = transpile(circuit, self.simulator())
        if circuit.num_parameters > 0:
            params = [ np.random.random() for _ in range(circuit.num_parameters) ]
            circuit = circuit.bind_parameters(params)
        return circuit

    def add_measure(self, circuit):
        """append measurement"""
        circuit.measure_all()

    def time_qv(self, qubit):
        """simulation time of QuantumVolume"""
        self._run(self._qv_circs[qubit])
    
    def time_qft(self, qubit):
        """simulation time of QFT"""
        self._run(self._qft_circs[qubit])

    def time_real_amplitudes(self, qubit):
        """simulation time of RealAmplitudes"""
        self._run(self._ra_circs[qubit])

    def time_real_amplitudes_full(self, qubit):
        """simulation time of RealAmplitudes"""
        self._run(self._ra_full_circs[qubit])

class ExpVal(_Base):

    def __init__(self,
                 qubits=[10, 15, 25]):

        super().__init__(qubits)
        self._expval_circs = { qubit: self.expval_circuit(qubit) for qubit in qubits } 
        self._expval_var_circs = { qubit: self.expval_var_circuit(qubit) for qubit in qubits } 

    def expval_circuit(self, qubit):
        terms = 1000
        circuit = QuantumCircuit(qubit)
        for i in range(qubit):
            circuit.h(i)

        rng = np.random.default_rng(1)
        pauli_strings = [''.join(s) for s in rng.choice(['I', 'X', 'Y', 'Z'], size=(terms, qubit))]
        try:
            op = None
            for pauli_string in pauli_strings:
                if op is None:
                    op = 1 / terms * qi.SparsePauliOp(pauli_string)
                else:
                    op += 1 / terms * qi.SparsePauliOp(pauli_string)
            circuit.save_expectation_value(op, range(qubit))
        except ImportError:
            from qiskit.providers.aer.extensions import snapshot_expectation_value
            circuit.snapshot_expectation_value('expval', [(1/terms, pauli) for pauli in pauli_strings], range(qubit))   
        return circuit

    def time_expval(self, qubit):
        """time to calculate expectation values with 1K pauli-strings"""
        self._run(self._expval_circs[qubit])

    def expval_var_circuit(self, qubit):
        terms = 100
        circuit = QuantumCircuit(qubit)
        for i in range(qubit):
            circuit.h(i)

        rng = np.random.default_rng(1)
        pauli_strings = [''.join(s) for s in rng.choice(['I', 'X', 'Y', 'Z'], size=(terms, qubit))]
        try:
            op = None
            for pauli_string in pauli_strings:
                if op is None:
                    op = rng.random() * qi.SparsePauliOp(pauli_string)
                else:
                    op += rng.random() * qi.SparsePauliOp(pauli_string)
            circuit.save_expectation_value_variance(op, range(qubit))
        except ImportError:
            raise ValueError('no save_expectation_value_variance')

        return circuit

    def time_expval_var(self, qubit):
        """time to calculate expectation value variances with 100 pauli-strings"""
        self._run(self._expval_var_circs[qubit])

class Noise(_Base):

    def __init__(self,
                 qubits=[10, 15]):

        super().__init__(qubits)
        self._circuits = {qubit: self.quantum_volume(qubit) for qubit in qubits}
        self._depolar_noise = self.depolarizing_error_model()
        self._damping_noise = self.amplitude_damping_error_model()
        self._roerror_noise = self.readout_error_model()

    def depolarizing_error_model(self):
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ['u'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ['cx'])
        return noise_model

    def amplitude_damping_error_model(self):
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(amplitude_damping_error(1e-3), 'u')
        cx_error = amplitude_damping_error(1e-2)
        cx_error = cx_error.tensor(cx_error)
        noise_model.add_all_qubit_quantum_error(cx_error, 'cx')
        return noise_model

    def readout_error_model(self):
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))
        return noise_model

    def quantum_volume(self, qubit):
        circuit = QuantumVolume(qubit)
        circuit.measure_all()
        circuit = transpile(circuit, self.simulator(), basis_gates=['u', 'cx'])
        return circuit

    def time_depolarizing_error(self, qubit):
        """time to simulate quantum volume transpiled with basis gates U and CX with depolarizing error"""
        self._run(self._circuits[qubit], noise_model=self._depolar_noise)

    def time_amplitude_damping_error(self, qubit):
        """time to simulate quantum volume transpiled with basis gates U and CX with amplitude damping error"""
        self._run(self._circuits[qubit], noise_model=self._damping_noise)

    def time_readout_error(self, qubit):
        self._run(self._circuits[qubit], noise_model=self._roerror_noise)

class ParameterizedCircuit(_Base):

    def __init__(self, qubits=[15]):
        super().__init__(qubits)
        self._circuits = {qubit: self.parameterized_circuits(qubit) for qubit in qubits}
        self._param_maps = {qubit: self.parameter_map(self._circuits[qubit]) for qubit in qubits}

    def parameterized_circuits(self, qubit):
        circuit = RealAmplitudes(qubit, reps=100)
        circuit.measure_all()
        circuit = transpile(circuit, self.simulator())
        return circuit

    def parameter_map(self, circuit):
        num_of_params = 10
        param_map = {}
        for param in circuit.parameters:
            param_values = [ np.random.random() for _ in range(num_of_params) ]
            param_map[param] = param_values
        return param_map

    def time_parameterized_circuits(self, qubit):
        """simulate parameterized circuits: 100 sets x 1000 parameters"""
        self._run(self._circuits[qubit], parameter_binds=[self._param_maps[qubit]])
