# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import qiskit as Terra
from qiskit import QiskitError
from qiskit.providers.aer import QasmSimulator
from .tools import simple_u3_circuit
from .tools import simple_cnot_circuit
from .tools import mixed_unitary_noise_model
from .tools import reset_noise_model
from .tools import kraus_noise_model


class SimpleU3TimeSuite:
    """
    Benchmark simple circuits with just one U3 gate

    The methods defined in this class will be executed by ASV framework as many
    times as the combination of all parameters exist in `self.params`, for
    exmaple: self.params = ([1,2,3],[4,5,6]), will run all methdos 9 times:
        time_method(1,4)
        time_method(1,5)
        time_method(1,6)
        time_method(2,4)
        time_method(2,5)
        time_method(2,6)
        time_method(3,4)
        time_method(3,5)
        time_method(3,6)

    For each noise model, we want to test various configurations of number of
    qubits
    """

    def __init__(self):
        self.timeout = 60 * 20
        self.backend = QasmSimulator()
        self.circuits = []
        for i in 5, 10, 15:
            circuit = simple_u3_circuit(i)
            self.circuits.append(Terra.compile(circuit, self.backend, shots=1))

        self.param_names = [
            "Simple u3 circuits (5/16/20/30 qubits)", "Noise model"
        ]
        self.params = (self.circuits, [
            None,
            mixed_unitary_noise_model(),
            reset_noise_model(),
            kraus_noise_model()
        ])

    def time_simple_u3(self, qobj, noise_model):
        result = self.backend.run(qobj, noise_model=noise_model).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


class SimpleCxTimeSuite:
    """
    Benchmark simple circuits with just on CX gate

    For each noise model, we want to test various configurations of number of
    qubits
    """

    def __init__(self):
        self.timeout = 60 * 20
        self.backend = QasmSimulator()
        self.circuits = []
        self.param_names = [
            "Simple cnot circuits (5/16/20/30 qubits)", "Noise model"
        ]
        for i in 5, 10, 15:
            circuit = simple_cnot_circuit(i)
            self.circuits.append(Terra.compile(circuit, self.backend, shots=1))
        self.params = (self.circuits, [
            None,
            mixed_unitary_noise_model(),
            reset_noise_model(),
            kraus_noise_model()
        ])

    def time_simple_cx(self, qobj, noise_model):
        result = self.backend.run(qobj, noise_model=noise_model).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)
