"""
Benchmarking utility functions.
"""

import time
from multiprocessing import cpu_count

import qiskit
from qiskit import QiskitError
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_aer.backends.qasm_simulator import QasmSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error
from qiskit_aer.noise.errors import amplitude_damping_error
from qiskit_aer.noise.errors import thermal_relaxation_error


def benchmark_circuits_qasm_simulator(circuits,
                                      shots=1,
                                      threads=1,
                                      parallel_mode='state',
                                      noise_model=None,
                                      config=None):
    """Return average execution time for a list if list of circuits.

    Args:
        circuits (list(QuantumCircuit)): a list of quantum circuits.
        shots (int): number of shots for each circuit.
        max_threads (int): The maximum number of threads to use for
                           OpenMP parallelization.
        parallel_mode (str): The method of parallelization to use.
                             options are 'state', 'circuit', 'shot'.
        noise_model (dict): Noise dictionary.
        config (dict): Backend configuration dictionary.

    Returns
        float: The total execution time for all circuits divided by the
        number of circuits.
    """

    if threads is None:
        threads = cpu_count()
    if threads > cpu_count():
        print("Warning: threads ({0})".format(threads) +
              " is greater than cpu_count ({1}).".format(cpu_count()))
        threads = cpu_count()

    # Generate backend
    backend = aer_benchmark_backend(max_threads=threads, parallel_mode=parallel_mode,
                                    noise_model=noise_model, config=config)
    # time circuits
    return benchmark_circuits(backend, circuits, shots=shots)


def benchmark_circuits(backend, circuits, shots=1):
    """Return average execution time for a list of circuits.

    Args:
        backend (Backend): A qiskit backend object.
        circuits list(QuantumCircuit): a list of quantum circuits.
        shots (int): Number of shots for each circuit.

    Returns
        float: The total execution time for all circuits divided by the
        number of circuits.

    Raises:
        QiskitError: If the simulation execution fails.
    """
    qobj = qiskit.compile(circuits, backend, shots=shots)
    start_time = time.time()
    result = backend.run(qobj).result()
    end_time = time.time()
    if isinstance(circuits, QuantumCircuit):
        average_time = end_time - start_time
    else:
        average_time = (end_time - start_time) / len(circuits)
    if result.status != 'COMPLETED':
        raise QiskitError("Simulation failed. Status: " + result.status)
    return average_time


def aer_benchmark_backend(max_threads=-1,
                          parallel_mode='state',
                          noise_model=None,
                          config=None):
    """Return an Aer simulator backend for benchmarking.

    Args:
        max_threads (int): The maximum number of threads to use for
                           OpenMP parallelization.
        parallel_mode (str): The method of parallelization to use.
                            options are 'state', 'circuit', 'shot'.
        noise_model (dict): Noise dictionary.
        config (dict): Backend configuration dictionary.

    Returns:
        AerQvSimulator: backend object with parallelization options set
    """

    # Set max threads for parallelization mode
    max_threads_state = max_threads
    max_threads_circuit = 1
    max_threads_shot = 1
    if parallel_mode == 'circuit':
        max_threads_circuit = max_threads
    elif parallel_mode == 'shot':
        max_threads_shot = max_threads

    # Set threads
    backend = QasmSimulator()
    backend.set_max_threads_state(max_threads_state)
    backend.set_max_threads_circuit(max_threads_circuit)
    backend.set_max_threads_shot(max_threads_shot)
    # Load config
    backend.set_config(config)
    # Load noise model
    backend.set_noise_model(noise_model)
    return backend


def add_measurement(circuit, measure_opt=True):
    """Append measurements to an input circuit.

    Args:
        circuit (QuantumCircuit): A circuit without measurements.
        measure_opt (bool): Enable measurement sampling optimization.

    Returns:
        QuantumCircuit: The input quantum circuit with classical registers
        and measure operations added.
    """
    qregs = list(circuit.get_qregs().values())
    cregs = []
    for qreg in qregs:
        cregs.append(ClassicalRegister(qreg.size))

    measure_circuit = QuantumCircuit(*qregs, *cregs)
    for qreg in qregs:
        measure_circuit.barrier(qreg)
    for qreg, creg in zip(qregs, cregs):
        measure_circuit.measure(qreg, creg)
    if measure_opt is False:
        for qreg, creg in zip(qregs, cregs):
            measure_circuit.barrier(qreg)
            measure_circuit.iden(qreg)
    return circuit + measure_circuit


def mixed_unitary_noise_model():
    """Return test rest mixed unitary noise model"""
    noise_model = NoiseModel()
    error1 = depolarizing_error(0.1, 1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = depolarizing_error(0.1, 2)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model


def kraus_noise_model():
    """Return test Kraus noise model"""
    noise_model = NoiseModel()
    error1 = amplitude_damping_error(0.1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = error1.kron(error1)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model


def reset_noise_model():
    """Return test reset noise model"""
    noise_model = NoiseModel()
    error1 = thermal_relaxation_error(50, 50, 0.1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = error1.kron(error1)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model
