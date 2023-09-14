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
Test circuits and reference outputs for rotation gate instructions.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# RX-gate
# ==========================================================================


def rx_gate_circuits_deterministic(final_measure=True):
    """X-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # RX(pi/2)
    circuit = QuantumCircuit(*regs)
    circuit.rx(np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(pi) = X
    circuit = QuantumCircuit(*regs)
    circuit.rx(np.pi, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(3*pi/2)
    circuit = QuantumCircuit(*regs)
    circuit.rx(3 * np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(4*pi/2) = I
    circuit = QuantumCircuit(*regs)
    circuit.rx(4 * np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def rx_gate_counts_deterministic(shots, hex_counts=True):
    """RX-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # pi/2
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # 2*pi/2
        targets.append({"0x1": shots})
        # 3*pi/2
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # 4*pi/2
        targets.append({"0x0": shots})
    else:
        # pi/2
        targets.append({"0": shots / 2, "1": shots / 2})
        # 2*pi/2
        targets.append({"1": shots})
        # 3*pi/2
        targets.append({"0": shots / 2, "1": shots / 2})
        # 4*pi/2
        targets.append({"0": shots})
    return targets


# ==========================================================================
# Z-gate
# ==========================================================================


def rz_gate_circuits_deterministic(final_measure=True):
    """RZ-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # RZ(pi/2) = S
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.rz(np.pi / 2, qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RZ(pi) = Z
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.rz(np.pi, qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RZ(3*pi/2) = Sdg
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.rz(3 * np.pi / 2, qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RZ(4*pi/2) = I
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.rz(4 * np.pi / 2, qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def rz_gate_counts_deterministic(shots, hex_counts=True):
    """RZ-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # pi/2 = S
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # 2*pi/2 = Z
        targets.append({"0x1": shots})
        # 3*pi/2 = Sdg
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # 4*pi/2 = I
        targets.append({"0x0": shots})
    else:
        # pi/2 = S
        targets.append({"0": shots / 2, "1": shots / 2})
        # 2*pi/2 = Z
        targets.append({"1": shots})
        # 3*pi/2 = Sdg
        targets.append({"0": shots / 2, "1": shots / 2})
        # 4*pi/2 = I
        targets.append({"0": shots})
    return targets


# ==========================================================================
# Y-gate
# ==========================================================================


def ry_gate_circuits_deterministic(final_measure=True):
    """RY-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # RX(pi/2)
    circuit = QuantumCircuit(*regs)
    circuit.ry(np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(pi) = Y
    circuit = QuantumCircuit(*regs)
    circuit.ry(np.pi, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(3*pi/2)
    circuit = QuantumCircuit(*regs)
    circuit.ry(3 * np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(4*pi/2) = I
    circuit = QuantumCircuit(*regs)
    circuit.ry(4 * np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def ry_gate_counts_deterministic(shots, hex_counts=True):
    """RY-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # pi/2
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # 2*pi/2
        targets.append({"0x1": shots})
        # 3*pi/2
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # 4*pi/2
        targets.append({"0x0": shots})
    else:
        # pi/2
        targets.append({"0": shots / 2, "1": shots / 2})
        # 2*pi/2
        targets.append({"1": shots})
        # 3*pi/2
        targets.append({"0": shots / 2, "1": shots / 2})
        # 4*pi/2
        targets.append({"0": shots})
    return targets
