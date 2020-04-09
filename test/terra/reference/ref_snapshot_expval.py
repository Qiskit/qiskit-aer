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
Test circuits and reference outputs for snapshot state instructions.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info.states import Statevector
from qiskit.providers.aer.extensions.snapshot_expectation_value import *

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


def snapshot_expval_labels():
    """List of labels for exp val snapshots."""
    return [
        "<H[0]>", "<H[1]>", "<X[0]>", "<X[1]>", "<Z[0]>", "<Z[1]>",
        "<H[0], I[1]>", "<I[0], H[1]>", "<X[0], I[1]>", "<I[0], X[1]>",
        "<Z[0], I[1]>", "<I[0], Z[1]>", "<X[0], X[1]>", "<Z[0], Z[1]>"
    ]


def snapshot_expval_params(pauli=False):
    """Dictionary of labels and params, qubits for exp val snapshots."""
    if pauli:
        X_wpo = [[1, 'X']]
        Z_wpo = [[1, 'Z']]
        H_wpo = [[1 / np.sqrt(2), 'X'], [1 / np.sqrt(2), 'Z']]
        IX_wpo = [[1, 'IX']]
        IZ_wpo = [[1, 'IZ']]
        IH_wpo = [[1 / np.sqrt(2), 'IX'], [1 / np.sqrt(2), 'IZ']]
        XX_wpo = [[1, 'XX']]
        ZZ_wpo = [[1, 'ZZ']]
    else:
        X_wpo = np.array([[0, 1], [1, 0]], dtype=complex)
        Z_wpo = np.array([[1, 0], [0, -1]], dtype=complex)
        H_wpo = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        IX_wpo = np.kron(np.eye(2), X_wpo)
        IZ_wpo = np.kron(np.eye(2), Z_wpo)
        IH_wpo = np.kron(np.eye(2), H_wpo)
        XX_wpo = np.kron(X_wpo, X_wpo)
        ZZ_wpo = np.kron(Z_wpo, Z_wpo)
    return {
        "<H[0]>": (H_wpo, [0]),
        "<H[1]>": (H_wpo, [1]),
        "<X[0]>": (X_wpo, [0]),
        "<X[1]>": (X_wpo, [1]),
        "<Z[0]>": (Z_wpo, [0]),
        "<Z[1]>": (Z_wpo, [1]),
        "<H[0], I[1]>": (IH_wpo, [0, 1]),
        "<I[0], H[1]>": (IH_wpo, [1, 0]),
        "<X[0], I[1]>": (IX_wpo, [0, 1]),
        "<I[0], X[1]>": (IX_wpo, [1, 0]),
        "<Z[0], I[1]>": (IZ_wpo, [0, 1]),
        "<I[0], Z[1]>": (IZ_wpo, [1, 0]),
        "<X[0], X[1]>": (XX_wpo, [0, 1]),
        "<Z[0], Z[1]>": (ZZ_wpo, [0, 1]),
    }


def snapshot_expval_circuits(pauli=False,
                             single_shot=False,
                             variance=False,
                             post_measure=False,
                             skip_measure=False):
    """SnapshotExpectationValue test circuits with deterministic counts"""

    circuits = []
    num_qubits = 2
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    # State |+1>
    circuit = QuantumCircuit(*regs)
    circuit.x(0)
    circuit.h(1)
    if not post_measure:
        for label, (params,
                    qubits) in snapshot_expval_params(pauli=pauli).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot,
                                               variance=variance)
    circuit.barrier(qr)
    if not skip_measure:
        circuit.measure(qr, cr)
    circuit.barrier(qr)
    if post_measure:
        for label, (params,
                    qubits) in snapshot_expval_params(pauli=pauli).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot,
                                               variance=variance)
    circuits.append(circuit)

    # State |00> + |11>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    if not post_measure:
        for label, (params,
                    qubits) in snapshot_expval_params(pauli=pauli).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot,
                                               variance=variance)
    circuit.barrier(qr)
    if not skip_measure:
        circuit.measure(qr, cr)
    circuit.barrier(qr)
    if post_measure:
        for label, (params,
                    qubits) in snapshot_expval_params(pauli=pauli).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot,
                                               variance=variance)
    circuits.append(circuit)

    # State |10> -i|01>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.sdg(0)
    circuit.cx(0, 1)
    circuit.x(1)
    if not post_measure:
        for label, (params,
                    qubits) in snapshot_expval_params(pauli=pauli).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot,
                                               variance=variance)
    circuit.barrier(qr)
    if not skip_measure:
        circuit.measure(qr, cr)
    circuit.barrier(qr)
    if post_measure:
        for label, (params,
                    qubits) in snapshot_expval_params(pauli=pauli).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot,
                                               variance=variance)
    circuits.append(circuit)
    return circuits


def snapshot_expval_counts(shots):
    """SnapshotExpectationValue test circuits reference counts."""
    targets = []
    # State |+1>
    targets.append({'0x1': shots / 2, '0x3': shots / 2})
    # State |00> + |11>
    targets.append({'0x0': shots / 2, '0x3': shots / 2})
    # State |01> -i|01>
    targets.append({'0x1': shots / 2, '0x2': shots / 2})
    return targets


def snapshot_expval_final_statevecs():
    """SnapshotExpectationValue test circuits pre meas statevecs"""
    # Get pre-measurement statevectors
    statevecs = []
    # State |+1>
    statevec = Statevector.from_label('+1')
    statevecs.append(statevec)
    # State |00> + |11>
    statevec = (Statevector.from_label('00') +
                Statevector.from_label('11')) / np.sqrt(2)
    statevecs.append(statevec)
    # State |10> -i|01>
    statevec = (Statevector.from_label('10') -
                1j * Statevector.from_label('01')) / np.sqrt(2)
    statevecs.append(statevec)
    return statevecs


def snapshot_expval_pre_meas_values():
    """SnapshotExpectationValue test circuits reference final probs"""
    targets = []
    for statevec in snapshot_expval_final_statevecs():
        values = {}
        for label, (mat, qubits) in snapshot_expval_params().items():
            values[label] = {
                '0x0': statevec.data.conj().dot(statevec.evolve(mat, qubits).data)
            }
        targets.append(values)
    return targets


def snapshot_expval_post_meas_values():
    """SnapshotExpectationValue test circuits reference final statevector"""
    targets = []
    for statevec in snapshot_expval_final_statevecs():
        values = {}
        for label, (mat, qubits) in snapshot_expval_params().items():
            inner_dict = {}
            for j in ['00', '01', '10', '11']:
                # Check if non-zero measurement probability for given
                # measurement outcome for final statevector
                vec = Statevector.from_label(j)
                if not np.isclose(vec.data.dot(statevec.data), 0):
                    # If outcome is non-zero compute expectation value
                    # with post-selected outcome state
                    inner_dict[hex(int(j, 2))] = vec.data.conj().dot(vec.evolve(mat, qubits).data)
            values[label] = inner_dict
        targets.append(values)
    return targets


def snapshot_expval_circuit_parameterized(single_shot=False,
                                          measure=True,
                                          snapshot=False):
    """SnapshotExpectationValue test circuits, rewritten as a single parameterized circuit and
    parameterizations array. """

    num_qubits = 2
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    circuit = QuantumCircuit(*regs)
    circuit.u3(0, 0, 0, 0)
    circuit.u1(0, 0)
    circuit.u3(0, 0, 0, 1)
    circuit.cu3(0, 0, 0, 0, 1)
    circuit.u3(0, 0, 0, 1)
    circuit.i(0)
    if snapshot:
        for label, (params, qubits) in snapshot_expval_params(pauli=True).items():
            circuit.snapshot_expectation_value(label,
                                               params,
                                               qubits,
                                               single_shot=single_shot)
    if measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        circuit.barrier(qr)

    # Parameterizations

    # State |+1>
    plus_one_params = {
        # X on 0
        (0, 0): np.pi,
        (0, 1): 0,
        (0, 2): np.pi,
        # No rZ
        (1, 0): 0,
        # H on 1
        (2, 0): np.pi / 2,
        (2, 2): np.pi,
        # No CrX
        (3, 0): 0,
        (3, 1): 0,
        (3, 2): 0,
        # No X
        (4, 0): 0,
        (4, 1): 0,
        (4, 2): 0,
    }
    # State |00> + |11>
    bell_params = {
        # H 0
        (0, 0): np.pi / 2,
        (0, 1): 0,
        (0, 2): np.pi,
        # No rZ
        (1, 0): 0,
        # No H
        (2, 0): 0,
        (2, 2): 0,
        # CX from 0 on 1
        (3, 0): np.pi,
        (3, 1): 0,
        (3, 2): np.pi,
        # No X
        (4, 0): 0,
        (4, 1): 0,
        (4, 2): 0,
    }
    # State |10> -i|01>
    iminus_bell_params = {
        # H 0
        (0, 0): np.pi / 2,
        (0, 1): 0,
        (0, 2): np.pi,
        # S 0
        (1, 0): - np.pi / 2,
        # No H
        (2, 0): 0,
        (2, 2): 0,
        # CX from 0 on 1
        (3, 0): np.pi,
        (3, 1): 0,
        (3, 2): np.pi,
        # X 1
        (4, 0): np.pi,
        (4, 1): 0,
        (4, 2): np.pi,
    }
    param_mat = np.transpose([list(plus_one_params.values()),
                              list(bell_params.values()),
                              list(iminus_bell_params.values())]).tolist()
    parameterizations = [[list(index), params]
                         for (index, params) in zip(plus_one_params.keys(), param_mat)]

    return circuit, parameterizations
