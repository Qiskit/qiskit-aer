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
Test circuits and reference outputs for 1-qubit Clifford gate instructions.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# H-gate
# ==========================================================================


def h_gate_circuits_deterministic(final_measure=True):
    """H-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)
    # HH=I
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def h_gate_counts_deterministic(shots, hex_counts=True):
    """H-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # HH=I
        targets.append({"0x0": shots})
    else:
        # HH=I
        targets.append({"0": shots})
    return targets


def h_gate_statevector_deterministic():
    """H-gate circuits reference statevectors."""
    targets = []
    # HH=I
    targets.append(np.array([1, 0]))
    return targets


def h_gate_unitary_deterministic():
    """H-gate circuits reference unitaries."""
    targets = []
    # HH=I
    targets.append(np.eye(2))
    return targets


def h_gate_circuits_nondeterministic(final_measure=True):
    """X-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)
    # H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def h_gate_counts_nondeterministic(shots, hex_counts=True):
    """H-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # H
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
    else:
        # H
        targets.append({"0": shots / 2, "1": shots / 2})
    return targets


def h_gate_statevector_nondeterministic():
    """H-gate circuits reference statevectors."""
    targets = []
    # H
    targets.append(np.array([1, 1]) / np.sqrt(2))
    return targets


def h_gate_unitary_nondeterministic():
    """H-gate circuits reference unitaries."""
    targets = []
    # HH=I
    targets.append(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    return targets


# ==========================================================================
# X-gate
# ==========================================================================


def x_gate_circuits_deterministic(final_measure=True):
    """X-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # X
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # XX = I
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.x(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # HXH=Z
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def x_gate_counts_deterministic(shots, hex_counts=True):
    """X-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # X
        targets.append({"0x1": shots})
        # XX = I
        targets.append({"0x0": shots})
        # HXH=Z
        targets.append({"0x0": shots})
    else:
        # X
        targets.append({"1": shots})
        # XX = I
        targets.append({"0": shots})
        # HXH=Z
        targets.append({"0": shots})
    return targets


def x_gate_statevector_deterministic():
    """X-gate circuits reference statevectors."""
    targets = []
    # X
    targets.append(np.array([0, 1]))
    # XX = I
    targets.append(np.array([1, 0]))
    # HXH=Z
    targets.append(np.array([1, 0]))
    return targets


def x_gate_unitary_deterministic():
    """X-gate circuits reference unitaries."""
    targets = []
    # X
    targets.append(np.array([[0, 1], [1, 0]]))
    # XX = I
    targets.append(np.eye(2))
    # HXH=Z
    targets.append(np.array([[1, 0], [0, -1]]))
    return targets


# ==========================================================================
# Z-gate
# ==========================================================================


def z_gate_circuits_deterministic(final_measure=True):
    """Z-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Z alone
    circuit = QuantumCircuit(*regs)
    circuit.z(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # HZH = X
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.z(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # HZZH = I
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.z(qr)
    circuit.barrier(qr)
    circuit.z(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def z_gate_counts_deterministic(shots, hex_counts=True):
    """Z-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Z
        targets.append({"0x0": shots})
        # HZH = X
        targets.append({"0x1": shots})
        # HZZH = I
        targets.append({"0x0": shots})
    else:
        # Z
        targets.append({"0": shots})
        # HZH = X
        targets.append({"1": shots})
        # HZZH = I
        targets.append({"0": shots})
    return targets


def z_gate_statevector_deterministic():
    """Z-gate circuits reference statevectors."""
    targets = []
    # Z
    targets.append(np.array([1, 0]))
    # HZH = X
    targets.append(np.array([0, 1]))
    # HZZH = I
    targets.append(np.array([1, 0]))
    return targets


def z_gate_unitary_deterministic():
    """Z-gate circuits reference unitaries."""
    targets = []
    # Z
    targets.append(np.array([[1, 0], [0, -1]]))
    # HZH = X
    targets.append(np.array([[0, 1], [1, 0]]))
    # HZZH = I
    targets.append(np.eye(2))
    return targets


# ==========================================================================
# Y-gate
# ==========================================================================


def y_gate_circuits_deterministic(final_measure=True):
    """Y-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Y
    circuit = QuantumCircuit(*regs)
    circuit.y(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    # YY = I
    circuit = QuantumCircuit(*regs)
    circuit.y(qr)
    circuit.barrier(qr)
    circuit.y(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    # HYH = -Y
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.y(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def y_gate_counts_deterministic(shots, hex_counts=True):
    """Y-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Y
        targets.append({"0x1": shots})
        # YY = I
        targets.append({"0x0": shots})
        # HYH = -Y
        targets.append({"0x1": shots})
    else:
        # Y
        targets.append({"1": shots})
        # YY = I
        targets.append({"0": shots})
        # HYH = -Y
        targets.append({"1": shots})
    return targets


def y_gate_statevector_deterministic():
    """Y-gate circuits reference statevectors."""
    targets = []
    # Y
    targets.append(np.array([0, 1j]))
    # YY = I
    targets.append(np.array([1, 0]))
    # HYH = -Y
    targets.append(np.array([0, -1j]))
    return targets


def y_gate_unitary_deterministic():
    """Y-gate circuits reference unitaries."""
    targets = []
    # Y
    targets.append(np.array([[0, -1j], [1j, 0]]))
    # YY = I
    targets.append(np.eye(2))
    # HYH = -Y
    targets.append(np.array([[0, 1j], [-1j, 0]]))
    return targets


# ==========================================================================
# S-gate
# ==========================================================================


def s_gate_circuits_deterministic(final_measure=True):
    """S-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # S
    circuit = QuantumCircuit(*regs)
    circuit.s(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # S.X
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.s(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # HSSH = HZH = X
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.s(qr)
    circuit.barrier(qr)
    circuit.s(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    circuit.barrier(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    return circuits


def s_gate_counts_deterministic(shots, hex_counts=True):
    """S-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # S
        targets.append({"0x0": shots})
        # S.X
        targets.append({"0x1": shots})
        # HSSH = HZH = X
        targets.append({"0x1": shots})
    else:
        # S
        targets.append({"0": shots})
        # S.X
        targets.append({"1": shots})
        # HSSH = HZH = X
        targets.append({"1": shots})
    return targets


def s_gate_statevector_deterministic():
    """S-gate circuits reference statevectors."""
    targets = []
    # S
    targets.append(np.array([1, 0]))
    # S.X
    targets.append(np.array([0, 1j]))
    # HSSH = HZH = X
    targets.append(np.array([0, 1]))
    return targets


def s_gate_unitary_deterministic():
    """S-gate circuits reference unitaries."""
    targets = []
    # S
    targets.append(np.diag([1, 1j]))
    # S.X
    targets.append(np.array([[0, 1], [1j, 0]]))
    # HSSH = HZH = X
    targets.append(np.array([[0, 1], [1, 0]]))
    return targets


def s_gate_circuits_nondeterministic(final_measure=True):
    """S-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # SH
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.s(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # HSH
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.s(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def s_gate_counts_nondeterministic(shots, hex_counts=True):
    """S-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # S.H
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # H.S.H
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
    else:
        # S.H
        targets.append({"0": shots / 2, "1": shots / 2})
        # H.S.H
        targets.append({"0": shots / 2, "1": shots / 2})
    return targets


def s_gate_statevector_nondeterministic():
    """S-gate circuits reference statevectors."""
    targets = []
    # S.H
    targets.append(np.array([1, 1j]) / np.sqrt(2))
    # H.S.H
    targets.append(np.array([1 + 1j, 1 - 1j]) / 2)
    return targets


def s_gate_unitary_nondeterministic():
    """S-gate circuits reference unitaries."""
    targets = []
    # S.H
    targets.append(np.array([[1, 1], [1j, -1j]]) / np.sqrt(2))
    # H.S.H
    targets.append(np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)
    return targets


# ==========================================================================
# S^dagger-gate
# ==========================================================================


def sdg_gate_circuits_deterministic(final_measure=True):
    """Sdg-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Sdg
    circuit = QuantumCircuit(*regs)
    circuit.sdg(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.Sdg.Sdg.H = H.Z.H = X
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.sdg(qr)
    circuit.barrier(qr)
    circuit.sdg(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.Sdg.S.H = I
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.s(qr)
    circuit.barrier(qr)
    circuit.sdg(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def sdg_gate_counts_deterministic(shots, hex_counts=True):
    """Sdg-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Sdg
        targets.append({"0x0": shots})
        # H.Sdg.Sdg.H = H.Z.H = X
        targets.append({"0x1": shots})
        # H.Sdg.S.H = I
        targets.append({"0x0": shots})
    else:
        # Sdg
        targets.append({"0": shots})
        # H.Sdg.Sdg.H = H.Z.H = X
        targets.append({"1": shots})
        # H.Sdg.S.H = I
        targets.append({"0": shots})
    return targets


def sdg_gate_statevector_deterministic():
    """Sdg-gate circuits reference statevectors."""
    targets = []
    # Sdg
    targets.append(np.array([1, 0]))
    # H.Sdg.Sdg.H = H.Z.H = X
    targets.append(np.array([0, 1]))
    # H.Sdg.S.H = I
    targets.append(np.array([1, 0]))
    return targets


def sdg_gate_unitary_deterministic():
    """Sdg-gate circuits reference unitaries."""
    targets = []
    # Sdg
    targets.append(np.diag([1, -1j]))
    # H.Sdg.Sdg.H = H.Z.H = X
    targets.append(np.array([[0, 1], [1, 0]]))
    # H.Sdg.S.H = I
    targets.append(np.eye(2))
    return targets


def sdg_gate_circuits_nondeterministic(final_measure=True):
    """Sdg-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Sdg.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.sdg(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.Sdg.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.sdg(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def sdg_gate_counts_nondeterministic(shots, hex_counts=True):
    """Sdg-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Sdg.H
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # H.Sdg.H
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
    else:
        # Sdg.H
        targets.append({"0": shots / 2, "1": shots / 2})
        # H.Sdg.H
        targets.append({"0": shots / 2, "1": shots / 2})
    return targets


def sdg_gate_statevector_nondeterministic():
    """Sdg-gate circuits reference statevectors."""
    targets = []
    # Sdg.H
    targets.append(np.array([1, -1j]) / np.sqrt(2))
    # H.Sdg.H
    targets.append(np.array([1 - 1j, 1 + 1j]) / 2)
    return targets


def sdg_gate_unitary_nondeterministic():
    """Sdg-gate circuits reference unitaries."""
    targets = []
    # Sdg.H
    targets.append(np.array([[1, 1], [-1j, 1j]]) / np.sqrt(2))
    # H.Sdg.H
    targets.append(np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)
    return targets


# ==========================================================================
# Pauli gate
# ==========================================================================


def pauli_gate_circuits_deterministic(final_measure=True):
    """pauli gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr,)

    circuit = QuantumCircuit(*regs)
    circuit.pauli("ZYX", qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # HZH = X
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.h(qr[2])
    circuit.pauli("ZZ", [qr[0], qr[2]])
    circuit.h(qr[0])
    circuit.h(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    circuit = QuantumCircuit(*regs)
    circuit.pauli("XYZ", qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def pauli_gate_counts_deterministic(shots, hex_counts=True):
    """multipauli-gate circuits reference counts."""
    targets = []
    if hex_counts:
        targets.append({"0x3": shots})
        targets.append({"0x5": shots})
        targets.append({"0x6": shots})
    else:
        targets.append({"110": shots})
        targets.append({"101": shots})
        targets.append({"011": shots})
    return targets


# ==========================================================================
# I-gate
# ==========================================================================


def id_gate_circuits_deterministic(final_measure=True):
    """I-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    circuit = QuantumCircuit(*regs)
    circuit.id(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.id(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def id_gate_counts_deterministic(shots, hex_counts=True):
    """I-gate circuits reference counts."""
    targets = []
    if hex_counts:
        targets.append({"0x0": shots})
        targets.append({"0x1": shots})
    else:
        targets.append({"0": shots})
        targets.append({"1": shots})
    return targets


def id_gate_statevector_deterministic():
    """I-gate circuits reference statevectors."""
    targets = []
    targets.append(np.array([1, 0]))
    targets.append(np.array([0, 1]))
    return targets


def id_gate_unitary_deterministic():
    """delay-gate circuits reference unitaries."""
    targets = []
    targets.append(np.eye(2))
    targets.append(np.array([[0, 1], [1, 0]]))
    return targets


# ==========================================================================
# delay-gate
# ==========================================================================


def delay_gate_circuits_deterministic(final_measure=True):
    """delay-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    circuit = QuantumCircuit(*regs)
    circuit.delay(1, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.delay(1, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def delay_gate_counts_deterministic(shots, hex_counts=True):
    """delay-gate circuits reference counts."""
    targets = []
    if hex_counts:
        targets.append({"0x0": shots})
        targets.append({"0x1": shots})
    else:
        targets.append({"0": shots})
        targets.append({"1": shots})
    return targets


def delay_gate_statevector_deterministic():
    """delay-gate circuits reference statevectors."""
    targets = []
    targets.append(np.array([1, 0]))
    targets.append(np.array([0, 1]))
    return targets


def delay_gate_unitary_deterministic():
    """delay-gate circuits reference unitaries."""
    targets = []
    targets.append(np.eye(2))
    targets.append(np.array([[0, 1], [1, 0]]))
    return targets
