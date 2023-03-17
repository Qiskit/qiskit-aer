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

from test.terra.utils.utils import list2dict

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit_aer.noise import NoiseModel


# ==========================================================================
#  Readout error
# ==========================================================================

# Error matrices used in tests
ROERROR_1Q = [[0.9, 0.1], [0.3, 0.7]]
ROERROR_2Q = [[0.3, 0, 0, 0.7], [0, 0.6, 0.4, 0], [0, 0, 1, 0], [0.1, 0, 0, 0.9]]


def readout_error_circuits():
    """Readout error test circuits"""

    circuits = []

    # Test circuit: ideal bell state for 1-qubit readout errors
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[1])
    # Ensure qubit 0 is measured before qubit 1
    circuit.barrier(qr)
    circuit.measure(qr[0], cr[0])
    circuit.barrier(qr)
    circuit.measure(qr[1], cr[1])

    # Add three copies of circuit
    circuits += 3 * [circuit]

    # 2-qubit correlated readout error circuit
    measure2 = Instruction("measure", 2, 2, [])  # 2-qubit measure
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.append(measure2, [0, 1], [0, 1])

    circuits.append(circuit)

    return circuits


def readout_error_noise_models():
    """Readout error test circuit noise models."""
    noise_models = []

    # 1-qubit readout error on qubit 0
    noise_model = NoiseModel()
    noise_model.add_readout_error(ROERROR_1Q, [0])
    noise_models.append(noise_model)

    # 1-qubit readout error on qubit 1
    noise_model = NoiseModel()
    noise_model.add_readout_error(ROERROR_1Q, [1])
    noise_models.append(noise_model)

    # 1-qubit readout error on qubit 1
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(ROERROR_1Q)
    noise_models.append(noise_model)

    # 2-qubit readout error on qubits 0,1
    noise_model = NoiseModel()
    noise_model.add_readout_error(ROERROR_2Q, [0, 1])
    noise_models.append(noise_model)

    return noise_models


def readout_error_counts(shots, hex_counts=True):
    """Readout error test circuits reference counts."""
    counts_lists = []

    # 1-qubit readout error on qubit 0
    counts = [
        ROERROR_1Q[0][0] * shots / 2,
        ROERROR_1Q[0][1] * shots / 2,
        ROERROR_1Q[1][0] * shots / 2,
        ROERROR_1Q[1][1] * shots / 2,
    ]
    counts_lists.append(counts)

    # 1-qubit readout error on qubit 1
    counts = [
        ROERROR_1Q[0][0] * shots / 2,
        ROERROR_1Q[1][0] * shots / 2,
        ROERROR_1Q[0][1] * shots / 2,
        ROERROR_1Q[1][1] * shots / 2,
    ]
    counts_lists.append(counts)

    # 1-qubit readout error on qubit 1
    p00 = 0.5 * (ROERROR_1Q[0][0] ** 2 + ROERROR_1Q[1][0] ** 2)
    p01 = 0.5 * (ROERROR_1Q[0][0] * ROERROR_1Q[0][1] + ROERROR_1Q[1][0] * ROERROR_1Q[1][1])
    p10 = 0.5 * (ROERROR_1Q[0][0] * ROERROR_1Q[0][1] + ROERROR_1Q[1][0] * ROERROR_1Q[1][1])
    p11 = 0.5 * (ROERROR_1Q[0][1] ** 2 + ROERROR_1Q[1][1] ** 2)
    counts = [p00 * shots, p01 * shots, p10 * shots, p11 * shots]
    counts_lists.append(counts)

    # 2-qubit readout error on qubits 0,1
    probs_ideal = [0.25, 0.25, 0.25, 0.25]
    p00 = sum([ideal * noise[0] for ideal, noise in zip(probs_ideal, ROERROR_2Q)])
    p01 = sum([ideal * noise[1] for ideal, noise in zip(probs_ideal, ROERROR_2Q)])
    p10 = sum([ideal * noise[2] for ideal, noise in zip(probs_ideal, ROERROR_2Q)])
    p11 = sum([ideal * noise[3] for ideal, noise in zip(probs_ideal, ROERROR_2Q)])
    counts = [p00 * shots, p01 * shots, p10 * shots, p11 * shots]
    counts_lists.append(counts)

    return [list2dict(i, hex_counts) for i in counts_lists]
