"""
Benchmarking utility functions.
"""

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error
from qiskit.providers.aer.noise.errors import amplitude_damping_error
from qiskit.providers.aer.noise.errors import thermal_relaxation_error

def tools_create_backend(backendClass, max_threads):
    backend = backendClass()
    backend.set_max_threads_state(max_threads)
    backend.set_max_threads_circuit(max_threads)
    backend.set_max_threads_shot(max_threads)
    return backend

def tools_mixed_unitary_noise_model():
    """Return test rest mixed unitary noise model"""
    noise_model = NoiseModel()
    error1 = depolarizing_error(0.1, 1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = depolarizing_error(0.1, 2)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model


def tools_reset_noise_model():
    """Return test reset noise model"""
    noise_model = NoiseModel()
    error1 = thermal_relaxation_error(50, 50, 0.1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = error1.kron(error1)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model


def tools_kraus_noise_model():
    """Return test Kraus noise model"""
    noise_model = NoiseModel()
    error1 = amplitude_damping_error(0.1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = error1.kron(error1)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model

