# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import qiskit as Terra
from qiskit import QiskitError
from qiskit.providers.aer import QasmSimulator
from .tools import simple_u3_circuit, simple_cnot_circuit, \
                    mixed_unitary_noise_model, reset_noise_model, \
                    kraus_noise_model

class SimpleU3TimeSuite:
    """
    Benchmarking times for simple circuits with just one gate but with various
    noise configurations:
    - ideal (no noise)
    - mixed state
    - reset
    - kraus

    For each noise model, we want to test various configurations of number of
    qubits
    """

    def __init__(self):

        self.timeout = 60 * 20
        self.backend = QasmSimulator()
        self.circuits = []
        for i in 5, 10, 15:
            circuit = simple_u3_circuit(i)
            self.circuits.append(
                Terra.compile(circuit, self.backend, shots=1)
            )

        self.param_names = ["Simple u3 circuits (5/16/20/30 qubits)"]
        self.params = (self.circuits)

    def time_ideal_simple_u3(self, qobj):
        result = self.backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_mixed_unitary_simple_u3(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=mixed_unitary_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_reset_simple_u3(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=reset_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_kraus_simple_u3(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=kraus_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


class SimpleCxTimeSuite:
    """
    Benchmarking times for simple circuits with just one gate but with various
    noise configurations:
    - ideal (no noise)
    - mixed state
    - reset
    - kraus

    For each noise model, we want to test various configurations of number of
    qubits
    """

    def __init__(self):

        self.timeout = 60 * 20
        self.backend = QasmSimulator()
        self.circuits = []
        self.param_names = ["Simple cnot circuits (5/16/20/30 qubits)"]
        for i in 5, 10, 15:
            circuit = simple_cnot_circuit(i)
            self.circuits.append(
                Terra.compile(circuit, self.backend, shots=1)
            )
        self.params = (self.circuits)

    def time_ideal_simple_u3(self, qobj):
        result = self.backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_mixed_unitary_simple_u3(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=mixed_unitary_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_reset_simple_u3(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=reset_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_kraus_simple_u3(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=kraus_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)
