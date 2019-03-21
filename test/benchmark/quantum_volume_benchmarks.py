# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import qiskit as Terra
from qiskit import QiskitError
from qiskit.providers.aer import QasmSimulator
from .tools import quantum_volume_circuit, mixed_unitary_noise_model, \
                    reset_noise_model, kraus_noise_model


class QuantumVolumeTimeSuite:
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
        self.timeout = 60 * 20
        self.qv_circuits = []
        self.backend = QasmSimulator()
        for num_qubits in 16 ,:
            for depth in 10 ,:
                # We want always the same seed, as we want always the same circuits
                # for the same value pairs of qubits,depth
                circ = quantum_volume_circuit(num_qubits, depth, seed=1)
                self.qv_circuits.append(
                    Terra.compile(circ, self.backend, shots=1)
                )
        self.param_names = ["Quantum Volume (16qubits 10depth)"]
        # This will run every benchmark for one of the combinations we have here:
        # bench(qv_circuits, None) => bench(qv_circuits, mixed()) =>
        # bench(qv_circuits, reset) => bench(qv_circuits, kraus())
        self.params = (
            self.qv_circuits,
            [
                None,
                mixed_unitary_noise_model(),
                reset_noise_model(),
                kraus_noise_model()
            ]
        )

    def setup(self, qobj):
        pass


    def time_quantum_volume(self, qobj, noise_model):
        result = self.backend.run(
            qobj,
            noise_model=noise_model
        ).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


    # def time_mixed_unitary_quantum_volume(self, qobj):
    #     result = self.backend.run(
    #         qobj,
    #         noise_model=mixed_unitary_noise_model()
    #     ).result()
    #     if result.status != 'COMPLETED':
    #         raise QiskitError("Simulation failed. Status: " + result.status)


    # def time_reset_quantum_volume(self, qobj):
    #     result = self.backend.run(
    #         qobj,
    #         noise_model=reset_noise_model()
    #     ).result()
    #     if result.status != 'COMPLETED':
    #         raise QiskitError("Simulation failed. Status: " + result.status)


    # def time_kraus_quantum_volume(self, qobj):
    #     result = self.backend.run(
    #         qobj,
    #         noise_model=kraus_noise_model()
    #     ).result()
    #     if result.status != 'COMPLETED':
    #         raise QiskitError("Simulation failed. Status: " + result.status)
