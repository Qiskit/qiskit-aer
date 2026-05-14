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


# ==========================================================================
# RX-gate (Clifford angles, sign-distinguishing)
# ==========================================================================


def rx_gate_clifford_circuits(final_measure=True):
    """RX-gate test circuits at Clifford angles (k * pi/2) with sign-sensitive outcomes.

    The circuit for pi/2 and 3*pi/2 applies RX then Sdg then H, which maps the
    Y-axis eigenstates to deterministic Z-basis outcomes and distinguishes +pi/2
    from -pi/2 (both would otherwise give 50/50 counts when measuring directly).
    """
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # RX(pi/2)|0> = |-y> = (|0>-i|1>)/sqrt(2); Sdg H |-y> = |1>
    circuit = QuantumCircuit(*regs)
    circuit.rx(np.pi / 2, qr)
    circuit.sdg(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(pi) = X: |0> -> |1>
    circuit = QuantumCircuit(*regs)
    circuit.rx(np.pi, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(3*pi/2)|0> = |+y> = (|0>+i|1>)/sqrt(2); Sdg H |+y> = |0>
    circuit = QuantumCircuit(*regs)
    circuit.rx(3 * np.pi / 2, qr)
    circuit.sdg(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RX(4*pi/2) = I: |0> -> |0>
    circuit = QuantumCircuit(*regs)
    circuit.rx(4 * np.pi / 2, qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def rx_gate_clifford_counts(shots, hex_counts=True):
    """RX-gate Clifford circuits reference counts."""
    targets = []
    if hex_counts:
        targets.append({"0x1": shots})   # pi/2: |1>
        targets.append({"0x1": shots})   # pi:   |1>
        targets.append({"0x0": shots})   # 3*pi/2: |0>
        targets.append({"0x0": shots})   # 4*pi/2 = I: |0>
    else:
        targets.append({"1": shots})   # pi/2: |1>
        targets.append({"1": shots})   # pi:   |1>
        targets.append({"0": shots})   # 3*pi/2: |0>
        targets.append({"0": shots})   # 4*pi/2 = I: |0>
    return targets


# ==========================================================================
# RZZ-gate (Clifford angles, sign-distinguishing)
# ==========================================================================


def rzz_gate_clifford_circuits(final_measure=True):
    """RZZ-gate test circuits at Clifford angles (k * pi/2) with sign-sensitive outcomes.

    For pi/2 and 3*pi/2 cases the circuit H0 -> RZZ -> Sdg0 -> H0 is used.
    Starting from |00>, H0 prepares |+0>. After RZZ(theta) the state is
    e^{-i*theta/2}(|00> + e^{i*theta}|10>)/sqrt(2). Sdg0 then H0 maps qubit 0
    to |0> for theta=pi/2 and |1> for theta=3*pi/2, distinguishing the two cases.
    For pi and 2*pi the H0 H1 -> RZZ -> H0 H1 sandwich is used.
    """
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # RZZ(pi/2): qubit0 -> |0>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.rzz(np.pi / 2, qr[0], qr[1])
    circuit.sdg(qr[0])
    circuit.h(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RZZ(pi): H H -> RZZ(pi) -> H H maps |00> to |11>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.rzz(np.pi, qr[0], qr[1])
    circuit.h(qr[0])
    circuit.h(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RZZ(3*pi/2): qubit0 -> |1>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.rzz(3 * np.pi / 2, qr[0], qr[1])
    circuit.sdg(qr[0])
    circuit.h(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # RZZ(4*pi/2) = I: H H -> I -> H H maps |00> to |00>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.rzz(4 * np.pi / 2, qr[0], qr[1])
    circuit.h(qr[0])
    circuit.h(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def rzz_gate_clifford_counts(shots, hex_counts=True):
    """RZZ-gate Clifford circuits reference counts."""
    targets = []
    if hex_counts:
        targets.append({"0x0": shots})   # pi/2: qubit0=0, qubit1=0 -> |00>
        targets.append({"0x3": shots})   # pi:   |11>
        targets.append({"0x1": shots})   # 3*pi/2: qubit0=1, qubit1=0 -> |01>
        targets.append({"0x0": shots})   # 4*pi/2 = I: |00>
    else:
        targets.append({"00": shots})    # pi/2: |00>
        targets.append({"11": shots})    # pi:   |11>
        targets.append({"01": shots})    # 3*pi/2: qubit0=1 -> "01"
        targets.append({"00": shots})    # 4*pi/2 = I: |00>
    return targets
