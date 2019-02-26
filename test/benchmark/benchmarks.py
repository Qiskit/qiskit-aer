# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import qiskit as Terra
from qiskit import QiskitError
from qiskit.providers.aer import QasmSimulator
from .quantumvolume import quantum_volume_circuit
from .tools import tools_mixed_unitary_noise_model, \
                   tools_reset_noise_model, \
                   tools_kraus_noise_model


class TimeSuite:
    """
    Benchmarking times for Quantum Volume with various noise configurations
    - ideal (no noise)
    - mixed state
    - reset
    - kraus

    For each noise model, we want to test various configurations of
    qubits/depth.
    """

    def __init__(self):
        # We want always the same SEED, as we want always the same circuits
        # for the same value pairs of qubits,depth
        self.timeout = 60 * 20
        self.qv_circuits = []
        self.backend = QasmSimulator()
        for num_qubits in 16 ,:
            for depth in 10 ,:
                circ = quantum_volume_circuit(num_qubits, depth, seed=1)
                self.qv_circuits.append(
                    Terra.compile(circ, self.backend, shots=1024)
                )
        self.param_names = ["Quantum Volume (16qubits 10depth)"]
        self.params = (self.qv_circuits)

    def setup(self, qobj):
        pass


    def time_ideal_quantum_volume(self, qobj):
        result = self.backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_mixed_unitary_quantum_volume(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=tools_mixed_unitary_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_reset_quantum_volume(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=tools_reset_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    def time_kraus_quantum_volume(self, qobj):
        result = self.backend.run(
            qobj,
            noise_model=tools_kraus_noise_model()
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)
