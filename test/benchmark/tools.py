"""
Benchmarking utility functions.
"""

import math
from itertools import repeat
from numpy import random
from scipy import linalg
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info.operators import Unitary
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error
from qiskit.providers.aer.noise.errors import amplitude_damping_error
from qiskit.providers.aer.noise.errors import thermal_relaxation_error


def _add_measurements(circuit, qr):
    cr = ClassicalRegister(qr.size)
    meas = QuantumCircuit(qr, cr)
    meas.barrier(qr)
    meas.measure(qr, cr)
    return circuit + meas


def mixed_unitary_noise_model():
    """Return test rest mixed unitary noise model"""
    noise_model = NoiseModel()
    error1 = depolarizing_error(0.1, 1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = depolarizing_error(0.1, 2)
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


def kraus_noise_model():
    """Return test Kraus noise model"""
    noise_model = NoiseModel()
    error1 = amplitude_damping_error(0.1)
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
    error2 = error1.kron(error1)
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model


def quantum_volume_circuit(num_qubits, depth, measure=True, seed=None):
    """Create a quantum volume circuit without measurement.

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    Args:
        num_qubits (int): number of qubits
        depth (int): ideal depth of each model circuit (over SU(4))
        measure (bool): include measurement in circuit.
        seed (int): the seed for the random number generator

    Returns:
        QuantumCircuit: A quantum volume circuit.
    """
    # Create random number generator with possibly fixed seed
    rng = random.RandomState(seed)
    # Create quantum/classical registers of size n
    qr = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr)
    # For each layer
    for _ in repeat(None, depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = rng.permutation(num_qubits)
        # For each consecutive pair in Pj, generate Haar random SU(4)
        # Decompose each SU(4) into CNOT + SU(2) and add to Ci
        for k in range(math.floor(num_qubits / 2)):
            # Generate random SU(4) matrix
            X = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
            SU4, _ = linalg.qr(X)  # Q is a unitary matrix
            SU4 /= pow(linalg.det(SU4), 1 / 4)  # make Q a special unitary
            qubits = [qr[int(perm[2 * k])], qr[int(perm[2 * k + 1])]]
            circuit.append(Unitary(SU4), qubits)
    if measure is True:
        circuit = _add_measurements(circuit, qr)
    return circuit


def simple_u3_circuit(num_qubits, measure=True):
    """Creates a simple circuit composed by u3 gates, with measurements or not
    at the end of each qubit.

    Args:
        num_qubits (int): Number of qubits
        measure (bool): Add measurements at the end of each qubit

    Returns:
        QuantumCircuit: The simple quantum circuit
    """
    qr = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr)
    for i in range(num_qubits):
        circuit.u3(1.1, 2.2, 3.3, qr[i])

    if measure:
        circuit = _add_measurements(circuit, qr)
    return circuit


def simple_cnot_circuit(num_qubits, measure=True):
    """Creates a simple circuit composed by cnot gates, with measurements or not
    at the end of each qubit.

    Args:
        num_qubits (int): Number of qubits
        measure (bool): Add measurements at the end of each qubit

    Returns:
        QuantumCircuit: The simple quantum circuit
    """
    qr = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr)
    for i in range(num_qubits):
        # for the last qubit, we exchange control and target qubits
        target_qubit = i + 1 if num_qubits - 1 > i else i - 1
        circuit.cx(qr[i], qr[target_qubit])

    if measure:
        circuit = _add_measurements(circuit, qr)
    return circuit
