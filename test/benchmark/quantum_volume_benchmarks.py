# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
"""Quantum Voluming benchmark suite"""

import qiskit as Terra
from qiskit import QiskitError
from qiskit.providers.aer import QasmSimulator
from .tools import quantum_volume_circuit
from .tools import mixed_unitary_noise_model
from .tools import reset_noise_model
from .tools import kraus_noise_model


class QuantumVolumeTimeSuite:
    """
    Benchmarking times for Quantum Volume with various noise configurations
    - ideal (no noise)
    - mixed state
    - reset
    - kraus

    For each noise model, we want to test various configurations of number of
    qubits

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
    """

    def __init__(self):
        self.timeout = 60 * 20
        self.qv_circuits = []
        self.backend = QasmSimulator()
        for num_qubits in (16, ):
            for depth in (10, ):
                # We want always the same seed, as we want always the same circuits
                # for the same value pairs of qubits,depth
                circ = quantum_volume_circuit(num_qubits, depth, seed=1)
                self.qv_circuits.append(
                    Terra.compile(
                        circ, self.backend, shots=1, basis_gates=['u3', 'cx']))
        self.param_names = ["Quantum Volume (16qubits 10depth)", "Noise Model"]
        # This will run every benchmark for one of the combinations we have here:
        # bench(qv_circuits, None) => bench(qv_circuits, mixed()) =>
        # bench(qv_circuits, reset) => bench(qv_circuits, kraus())
        self.params = (self.qv_circuits, [
            None,
            mixed_unitary_noise_model(),
            reset_noise_model(),
            kraus_noise_model()
        ])

    def setup(self, qobj):
        pass

    def time_quantum_volume(self, qobj, noise_model):
        result = self.backend.run(qobj, noise_model=noise_model).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)
