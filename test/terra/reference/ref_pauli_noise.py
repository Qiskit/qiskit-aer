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
QasmSimulator readout error NoiseModel integration tests
"""

from test.terra.utils.utils import list2dict

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


# ==========================================================================
# Pauli Gate Errors
# ==========================================================================

def pauli_gate_error_circuits():
    """Local Pauli gate error noise model circuits"""
    circuits = []

    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')

    # 100% all-qubit Pauli error on "id" gate
    circuit = QuantumCircuit(qr, cr)
    circuit.i(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% all-qubit Pauli error on "id" gates
    circuit = QuantumCircuit(qr, cr)
    circuit.i(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 100% Pauli error on "id" gates on qubit-1
    circuit = QuantumCircuit(qr, cr)
    circuit.i(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% all-qubit Pauli error on "id" gates on qubit-0
    circuit = QuantumCircuit(qr, cr)
    circuit.i(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% Pauli-X error on spectator for CX gate on [0, 1]
    qr = QuantumRegister(3, 'qr')
    cr = ClassicalRegister(3, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.cx(qr[0], qr[1])
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_gate_error_noise_models():
    """Local Pauli gate error noise models"""
    noise_models = []

    # 100% all-qubit Pauli error on "id" gates
    error = pauli_error([('X', 1)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'id')
    noise_models.append(noise_model)

    # 25% all-qubit Pauli error on "id" gates
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'id')
    noise_models.append(noise_model)

    # 100% Pauli error on "id" gates on qubit-1
    error = pauli_error([('X', 1)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, 'id', [1])
    noise_models.append(noise_model)

    # 25% all-qubit Pauli error on "id" gates on qubit-0
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, 'id', [0])
    noise_models.append(noise_model)

    # 25% Pauli-X error on spectator for CX gate on [0, 1]
    error = pauli_error([('XII', 0.25), ('III', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_nonlocal_quantum_error(error, 'cx', [0, 1], [0, 1, 2])
    noise_models.append(noise_model)

    return noise_models


def pauli_gate_error_counts(shots, hex_counts=True):
    """Pauli gate error circuits reference counts"""
    counts_lists = []

    # 100% all-qubit Pauli error on "id" gates
    counts = [0, 0, 0, shots]
    counts_lists.append(counts)

    # 25% all-qubit Pauli error on "id" gates
    counts = [9 * shots / 16, 3 * shots / 16, 3 * shots / 16, shots / 16]
    counts_lists.append(counts)

    # 100% Pauli error on "id" gates on qubit-1
    counts = [0, 0, shots, 0]
    counts_lists.append(counts)

    # 25% all-qubit Pauli error on "id" gates on qubit-0
    counts = [3 * shots / 4, shots / 4, 0, 0]
    counts_lists.append(counts)

    # 25% Pauli-X error on spectator for CX gate on [0, 1]
    counts = [3 * shots / 4, 0, 0, 0, shots / 4, 0, 0, 0]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]


# ==========================================================================
# Pauli Measure Errors
# ==========================================================================

def pauli_measure_error_circuits():
    """Local Pauli measure error noise model circuits"""
    circuits = []

    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')

    # 25% all-qubit Pauli error on measure
    circuit = QuantumCircuit(qr, cr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% local Pauli error on measure of qubit 1
    circuit = QuantumCircuit(qr, cr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25 % non-local Pauli error on qubit 1 for measure of qubit-1
    circuit = QuantumCircuit(qr, cr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_measure_error_noise_models():
    """Local Pauli measure error noise models"""
    noise_models = []

    # 25% all-qubit Pauli error on measure
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'measure')
    noise_models.append(noise_model)

    # 25% local Pauli error on measure of qubit 1
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, 'measure', [1])
    noise_models.append(noise_model)

    # 25 % non-local Pauli error on qubit 1 for measure of qubit-1
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_nonlocal_quantum_error(error, 'measure', [0], [1])
    noise_models.append(noise_model)

    return noise_models


def pauli_measure_error_counts(shots, hex_counts=True):
    """Local Pauli measure error circuits reference counts"""
    counts_lists = []

    # 25% all-qubit Pauli error on measure
    counts = [9 * shots / 16, 3 * shots / 16, 3 * shots / 16, shots / 16]
    counts_lists.append(counts)

    # 25% local Pauli error on measure of qubit 1
    counts = [3 * shots / 4, 0, shots / 4, 0]
    counts_lists.append(counts)

    # 25 % non-local Pauli error on qubit 1 for measure of qubit-1
    counts = [3 * shots / 4, 0, shots / 4, 0]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]


# ==========================================================================
# Pauli Reset Errors
# ==========================================================================


def pauli_reset_error_circuits():
    """Local Pauli reset error noise model circuits"""
    circuits = []

    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')

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

    # 25 % non-local Pauli error on qubit 1 for reset of qubit-0
    circuit = QuantumCircuit(qr, cr)
    circuit.barrier(qr)
    circuit.reset(qr[1])
    circuit.barrier(qr)
    circuit.reset(qr[0])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_reset_error_noise_models():
    """Local Pauli reset error noise models"""
    noise_models = []

    # 25% all-qubit Pauli error on reset
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'reset')
    noise_models.append(noise_model)

    # 25% local Pauli error on reset of qubit 1
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, 'reset', [1])
    noise_models.append(noise_model)

    # 25 % non-local Pauli error on qubit 1 for reset of qubit-0
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    noise_model = NoiseModel()
    noise_model.add_nonlocal_quantum_error(error, 'reset', [0], [1])
    noise_models.append(noise_model)

    return noise_models


def pauli_reset_error_counts(shots, hex_counts=True):
    """Local Pauli reset error circuits reference counts"""
    counts_lists = []

    # 25% all-qubit Pauli error on reset
    counts = [9 * shots / 16, 3 * shots / 16, 3 * shots / 16, shots / 16]
    counts_lists.append(counts)

    # 25% local Pauli error on reset of qubit 1
    counts = [3 * shots / 4, 0, shots / 4, 0]
    counts_lists.append(counts)

    # 25 % non-local Pauli error on qubit 1 for reset of qubit-0
    counts = [3 * shots / 4, 0, shots / 4, 0]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]
