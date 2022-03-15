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
        self.params = (qubits)
        self.param_names = ["qubits"]
        
    def simulator(self):
        if self._simulator:
            return self._simulator
        try:
            from qiskit.providers.aer import AerSimulator
            self._simulator = AerSimulator()
        except:
            from qiskit.providers.aer import QasmSimulator
            self._simulator = QasmSimulator()
        return self._simulator

    def _run(self, circuit, *args, **kwargs):
        if self.simulator().__class__.__name__ == 'AerSimulator':
            return self.simulator().run(circuit, *args, **kwargs).result()
        elif self.simulator().__class__.__name__ in ('QasmSimulator', 'UnitarySimulator'):
            from qiskit import assemble
            circuit = assemble(circuit, self.simulator())
            return self.simulator().run(circuit, *args, **kwargs).result()
        else:
            raise ValueError(f'unknown simulator class: {self._simulator.__class__.__name__}')


class Benchmark(_Base):

    def __init__(self,
                 qubits=[5, 15, 25]):
        
        super().__init__(qubits)

    def _track(self, circuit):
        """track only simulation time of given circuit"""

        self.add_measure(circuit)

        circuit = transpile(circuit, self.simulator())

        if circuit.num_parameters > 0:
            params = [ np.random.random() for _ in range(circuit.num_parameters) ]
            circuit = circuit.bind_parameters(params)

        # benchmark start
        start_ts = time()
        result = self._run(circuit)
        end_ts = time()
        # benchmark end

        if result.success:
            return end_ts - start_ts
        else:
            raise ValueError(result.status)

    def add_measure(self, circuit):
        """append measurement"""
        circuit.measure_all()

    def track_qv(self, qubit):
        """simulation time of QuantumVolume"""
        return self._track(QuantumVolume(qubit))

    track_qv.unit = "s"

    def track_qft(self, qubit):
        """simulation time of QFT"""
        return self._track(QFT(qubit))

    track_qft.unit = 'ms'

    def track_real_amplitudes(self, qubit):
        """simulation time of RealAmplitudes"""
        return self._track(RealAmplitudes(qubit))

    track_real_amplitudes.unit = 'ms'

    def track_real_amplitudes_full(self, qubit):
        """simulation time of RealAmplitudes"""
        return self._track(RealAmplitudes(qubit, entanglement='full'))

    track_real_amplitudes_full.unit = 'ms'


class ExpVal(_Base):

    def __init__(self,
                 qubits=[10, 15, 25]):

        super().__init__(qubits)

    def track_expval(self, qubit):
        """track only time to calculate expectation values of RealAmplitudes with 1K pauli-strings"""
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
        except:
            from qiskit.providers.aer.extensions import snapshot_expectation_value
            circuit.snapshot_expectation_value('expval', [(1/terms, pauli) for pauli in pauli_strings], range(qubit))   

        start_ts = time()
        result = self._run(circuit)
        end_ts = time()
        if not result.success:
            raise ValueError(result.status)

        return end_ts - start_ts

    track_expval.unit = "s"

    def track_expval_var(self, qubit):
        """track only time to calculate expectation value variances of RealAmplitudes with 100 pauli-strings"""

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
        except:
            raise ValueError('no save_expectation_value_variance')

        start_ts = time()
        result = self._run(circuit)
        end_ts = time()
        if not result.success:
            raise ValueError(result.status)
        return end_ts - start_ts

    track_expval_var.unit = "s"


class Noise(_Base):

    def __init__(self,
                 qubits=[10, 15]):

        super().__init__(qubits)

    def track_depolarizing_error(self, qubit):
        """track only time to simulate quantum volume transpiled with basis gates U and CX with depolarizing error"""
        circuit = QuantumVolume(qubit)
        circuit.measure_all()

        circuit = transpile(circuit, self.simulator(), basis_gates=['u', 'cx'])

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ['u'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ['cx'])

        start_ts = time()
        result = self._run(circuit, noise_model=noise_model)
        end_ts = time()
        if not result.success:
            raise ValueError(result.status)

        return end_ts - start_ts

    track_depolarizing_error.unit = "s"

    def track_amplitude_damping_error(self, qubit):
        """track only time to simulate quantum volume transpiled with basis gates U and CX with amplitude damping error"""
        circuit = QuantumVolume(qubit)
        circuit.measure_all()

        circuit = transpile(circuit, self.simulator(), basis_gates=['u', 'cx'])

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(amplitude_damping_error(1e-3), 'u')
        cx_error = amplitude_damping_error(1e-2)
        cx_error = cx_error.tensor(cx_error)
        noise_model.add_all_qubit_quantum_error(cx_error, 'cx')

        start_ts = time()
        result = self._run(circuit, noise_model=noise_model)
        end_ts = time()
        if not result.success:
            raise ValueError(result.status)

        return end_ts - start_ts

    track_amplitude_damping_error.unit = "s"

    def track_readout_error(self, qubit):
        """track only time to simulate quantum volume transpiled with basis gates U and CX with amplitude damping error"""
        circuit = QuantumVolume(qubit)
        circuit.measure_all()

        circuit = transpile(circuit, self.simulator())

        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))

        start_ts = time()
        result = self._run(circuit, noise_model=noise_model)
        end_ts = time()
        if not result.success:
            raise ValueError(result.status)

        return end_ts - start_ts

    track_readout_error.unit = "s"


class ParameterizedCircuit(_Base):

    def __init__(self):

        super().__init__([15])

    def track_parameterized_circuits(self, qubit):
        """track parameterized circuits: 100 sets x 1000 parameters"""
        circuit = RealAmplitudes(qubit, reps=100)
        circuit.measure_all()
        
        circuit = transpile(circuit, self.simulator())

        num_of_params = 10
        param_map = {}
        for param in circuit.parameters:
            param_values = [ np.random.random() for _ in range(num_of_params) ]
            param_map[param] = param_values

        start_ts = time()
        result = self._run(circuit, parameter_binds=[param_map])
        end_ts = time()
        if not result.success:
            raise ValueError(result.status)

        return end_ts - start_ts

    track_parameterized_circuits.unit = "s"