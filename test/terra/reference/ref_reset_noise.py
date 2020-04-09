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
QasmSimulator reset error NoiseModel integration tests
"""

from test.terra.utils.utils import list2dict

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import reset_error

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


# ==========================================================================
# Reset Gate Errors
# ==========================================================================

def reset_gate_error_circuits():
    """Reset gate error noise model circuits"""
    circuits = []

    # 50% reset to 0 state on qubit 0
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% reset to 0 state on qubit 1
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 100% reset error to 0 on all qubits
    qr = QuantumRegister(1, 'qr')
    cr = ClassicalRegister(1, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 100% reset error to 1 on all qubits
    qr = QuantumRegister(1, 'qr')
    cr = ClassicalRegister(1, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.i(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    # 25% reset error to 0 and 1 on all qubits
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.i(qr[0])
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def reset_gate_error_noise_models():
    """Reset gate error noise models"""
    noise_models = []

    # 50% reset to 0 state on qubit 0
    error = reset_error(0.5)
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, 'x', [0])
    noise_models.append(noise_model)

    # 25% reset to 0 state on qubit 1
    error = reset_error(0.25)
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, 'x', [1])
    noise_models.append(noise_model)

    # 100% reset error to 0 on all qubits
    error = reset_error(1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'x'])
    noise_models.append(noise_model)

    # 100% reset error to 1 on all qubits
    error = reset_error(0, 1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'x'])
    noise_models.append(noise_model)

    # 25% reset error to 0 and 1 on all qubits
    error = reset_error(0.25, 0.25)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'x'])
    noise_models.append(noise_model)

    return noise_models


def reset_gate_error_counts(shots, hex_counts=True):
    """Reset gate error circuits reference counts"""
    counts_lists = []

    # 50% reset to 0 state on qubit 0
    counts = [0, 0, shots / 2, shots / 2]
    counts_lists.append(counts)

    # 25% reset to 0 state on qubit 1
    counts = [0, shots / 4, 0, 3 * shots / 4]
    counts_lists.append(counts)

    # 100% reset error to 0 on all qubits
    counts = [shots, 0, 0, 0]
    counts_lists.append(counts)

    # 100% reset error to 1 on all qubits
    counts = [0, shots, 0, 0]
    counts_lists.append(counts)

    # 25% reset error to 0 and 1 on all qubits
    counts = [3 * shots / 16, shots / 16, 9 * shots / 16, 3 * shots / 16]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]
