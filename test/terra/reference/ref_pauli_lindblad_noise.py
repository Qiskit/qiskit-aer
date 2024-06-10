# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
AerSimulator readout error NoiseModel integration tests
"""
from math import log, inf
from test.terra.utils.utils import list2dict

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import PauliLindbladError


# ==========================================================================
# Pauli Gate Errors
# ==========================================================================


def _pauli_rate(prob_error):
    """Convert single error probability to generator rate"""
    # NOTE: PauliLindbladError cannot produce a Pauli error with error prob > 50%
    if prob_error > 0.5:
        raise ValueError("Error probability cannot be > 50%")
    if prob_error == 0.5:
        return inf
    if prob_error == 0:
        return 0.0
    return -0.5 * log(1 - 2 * prob_error)


def pauli_lindblad_gate_error_circuits():
    """Local PauliLindbladError gate error noise model circuits"""
    circuits = []

    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")

    # 25% all-qubit Pauli error on "id" gates
    circuit = QuantumCircuit(qr, cr)
    circuit.id(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% all-qubit Pauli error on "id" gates on qubit-0
    circuit = QuantumCircuit(qr, cr)
    circuit.id(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 50% Pauli error on conditional gate that doesn't get applied
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr).c_if(cr, 1)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 50% Pauli error on conditional gate that does get applied
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr).c_if(cr, 0)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_lindblad_gate_error_noise_models():
    """Local Pauli gate error noise models"""
    noise_models = []

    # 25% all-qubit Pauli error on "id" gates
    error = PauliLindbladError(["X"], [_pauli_rate(0.25)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, "id")
    noise_models.append(noise_model)

    # 25% all-qubit Pauli error on "id" gates on qubit-0
    error = PauliLindbladError(["X"], [_pauli_rate(0.25)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, "id", [0])
    noise_models.append(noise_model)

    # 50% Pauli error on conditional gate that doesn't get applied
    error = PauliLindbladError(["X"], [_pauli_rate(0.5)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, "x")
    noise_models.append(noise_model)

    # 50% Pauli error on conditional gate that does get applied
    error = PauliLindbladError(["X"], [_pauli_rate(0.5)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, "x")
    noise_models.append(noise_model)

    return noise_models


def pauli_lindblad_gate_error_counts(shots, hex_counts=True):
    """Pauli gate error circuits reference counts"""
    counts_lists = []

    # 25% all-qubit Pauli error on "id" gates
    counts = [9 * shots / 16, 3 * shots / 16, 3 * shots / 16, shots / 16]
    counts_lists.append(counts)

    # 25% all-qubit Pauli error on "id" gates on qubit-0
    counts = [3 * shots / 4, shots / 4, 0, 0]
    counts_lists.append(counts)

    # 50% Pauli error on conditional gate that doesn't get applied
    counts = [shots, 0, 0, 0]
    counts_lists.append(counts)

    # 50% Pauli error on conditional gate that does get applied
    counts = 4 * [shots / 4]
    counts_lists.append(counts)

    return [list2dict(i, hex_counts) for i in counts_lists]


# ==========================================================================
# Pauli Measure Errors
# ==========================================================================


def pauli_lindblad_measure_error_circuits():
    """Local Pauli measure error noise model circuits"""
    circuits = []

    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")

    # 25% all-qubit Pauli error on measure
    circuit = QuantumCircuit(qr, cr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% local Pauli error on measure of qubit 1
    circuit = QuantumCircuit(qr, cr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_lindblad_measure_error_noise_models():
    """Local Pauli measure error noise models"""
    noise_models = []

    # 25% all-qubit Pauli error on measure
    error = PauliLindbladError(["X"], [_pauli_rate(0.25)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, "measure")
    noise_models.append(noise_model)

    # 25% local Pauli error on measure of qubit 1
    error = PauliLindbladError(["X"], [_pauli_rate(0.25)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, "measure", [1])
    noise_models.append(noise_model)

    return noise_models


def pauli_lindblad_measure_error_counts(shots, hex_counts=True):
    """Local Pauli measure error circuits reference counts"""
    counts_lists = []

    # 25% all-qubit Pauli error on measure
    counts = [9 * shots / 16, 3 * shots / 16, 3 * shots / 16, shots / 16]
    counts_lists.append(counts)

    # 25% local Pauli error on measure of qubit 1
    counts = [3 * shots / 4, 0, shots / 4, 0]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]


# ==========================================================================
# Pauli Reset Errors
# ==========================================================================


def pauli_lindblad_reset_error_circuits():
    """Local Pauli reset error noise model circuits"""
    circuits = []

    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")

    # 25% all-qubit Pauli error on reset
    circuit = QuantumCircuit(qr, cr)
    circuit.barrier(qr)
    circuit.reset(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% local Pauli error on reset of qubit 1
    circuit = QuantumCircuit(qr, cr)
    circuit.barrier(qr)
    circuit.reset(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_lindblad_reset_error_noise_models():
    """Local Pauli reset error noise models"""
    noise_models = []

    # 25% all-qubit Pauli error on reset
    error = PauliLindbladError(["X"], [_pauli_rate(0.25)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, "reset")
    noise_models.append(noise_model)

    # 25% local Pauli error on reset of qubit 1
    error = PauliLindbladError(["X"], [_pauli_rate(0.25)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, "reset", [1])
    noise_models.append(noise_model)

    return noise_models


def pauli_lindblad_reset_error_counts(shots, hex_counts=True):
    """Local Pauli reset error circuits reference counts"""
    counts_lists = []

    # 25% all-qubit Pauli error on reset
    counts = [9 * shots / 16, 3 * shots / 16, 3 * shots / 16, shots / 16]
    counts_lists.append(counts)

    # 25% local Pauli error on reset of qubit 1
    counts = [3 * shots / 4, 0, shots / 4, 0]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]
