# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from qiskit.providers.aer import QasmSimulator
from .quantumvolume import quantum_volume_circuit
from .benchmark_tools import tools_create_backend,
                             tools_mixed_unitary_noise_model,
                             tools_reset_noise_model,
                             tools_kraus_noise_model


class TimeSuite:
    """
    Benchmarking times for Quantum Volume with various noise configurations
    - ideal (no noise)
    - mixed state
    - reset
    - kraus

    For each noise model, we want to test various configurations of
    qubits/depth/threads:
    - 5,16,20 qubits
    - 10,100,1000 depth
    - 1,2,4,16 threads
    """

    # We want always the same SEED, as we want always the same circuits
    # for the same value pairs of qubits,depth
    SEED = 1

    def setup(self):
        self.qv_circuits = []
        for num_qubits in 5,16,20:
            for depth in 10,100,1000:
                self.qv_circuits.add(
                    quantum_volume_circuit(num_qubits, depth, seed=SEED)
                )
        self.max_thread_list = [1,2,4,16]
        self.params = (self.max_thread_list, self.qv_circuits)

    def time_ideal_quantum_volume(self, max_threads, qv_circuit):
        backend = tools_create_backend(QasmSimulator, max_threads)
        qobj = qiskit.compile(qv_circuit, backend, shots=num_shots)
        result = backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)

    def time_mixed_unitary_quantum_volume(self, max_threads, qv_circuit):
        backend = tools_create_backend(QasmSimulator, max_threads)
        # Load noise model
        backend.set_noise_model(tools_mixed_unitary_noise_model())
        qobj = qiskit.compile(qv_circuit, backend, shots=num_shots)
        result = backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)

    def time_reset_quantum_volume(self, max_threads_list, qv_circuit):
        backend = tools_create_backend(QasmSimulator, max_threads)
        # Load noise model
        backend.set_noise_model(tools_reset_noise_model())
        qobj = qiskit.compile(qv_circuit, backend, shots=num_shots)
        result = backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)

    def time_kraus_quantum_volume(self, max_threads_list, qv_circuit):
        backend = tools_create_backend(QasmSimulator, max_threads)
        # Load noise model
        backend.set_noise_model(tools_kraus_noise_model())
        qobj = qiskit.compile(qv_circuit, backend, shots=num_shots)
        result = backend.run(qobj).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)
