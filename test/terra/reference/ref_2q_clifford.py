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
Test circuits and reference outputs for 2-qubit Clifford gate instructions.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# CX-gate
# ==========================================================================


def cx_gate_circuits_deterministic(final_measure=True):
    """CX-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(X^I), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(I^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(I^X), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(X^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(X^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(X^X), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # test ctrl_states=0
    circuit = QuantumCircuit(*regs)
    circuit.cx(qr[0], qr[1], ctrl_state=0)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def cx_gate_counts_deterministic(shots, hex_counts=True):
    """CX-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CX01, |00> state
        targets.append({"0x0": shots})  # {"00": shots}
        # CX10, |00> state
        targets.append({"0x0": shots})  # {"00": shots}
        # CX01.(X^I), |10> state
        targets.append({"0x2": shots})  # {"00": shots}
        # CX10.(I^X), |01> state
        targets.append({"0x1": shots})  # {"00": shots}
        # CX01.(I^X), |11> state
        targets.append({"0x3": shots})  # {"00": shots}
        # CX10.(X^I), |11> state
        targets.append({"0x3": shots})  # {"00": shots}
        # CX01.(X^X), |01> state
        targets.append({"0x1": shots})  # {"00": shots}
        # CX10.(X^X), |10> state
        targets.append({"0x2": shots})  # {"00": shots}
        # test ctrl_states=0
        targets.append({"0x2": shots})  # {"00": shots}
    else:
        # CX01, |00> state
        targets.append({"00": shots})  # {"00": shots}
        # CX10, |00> state
        targets.append({"00": shots})  # {"00": shots}
        # CX01.(X^I), |10> state
        targets.append({"10": shots})  # {"00": shots}
        # CX10.(I^X), |01> state
        targets.append({"01": shots})  # {"00": shots}
        # CX01.(I^X), |11> state
        targets.append({"11": shots})  # {"00": shots}
        # CX10.(X^I), |11> state
        targets.append({"11": shots})  # {"00": shots}
        # CX01.(X^X), |01> state
        targets.append({"01": shots})  # {"00": shots}
        # CX10.(X^X), |10> state
        targets.append({"10": shots})  # {"00": shots}
        # test ctrl_states=0
        targets.append({"10": shots})  # {"00": shots}
    return targets


def cx_gate_statevector_deterministic():
    """CX-gate test circuits with deterministic counts."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX10, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX01.(X^I), |10> state
    targets.append(np.array([0, 0, 1, 0]))
    # CX10.(I^X), |01> state
    targets.append(np.array([0, 1, 0, 0]))
    # CX01.(I^X), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    # CX10.(X^I), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    # CX01.(X^X), |01> state
    targets.append(np.array([0, 1, 0, 0]))
    # CX10.(X^X), |10> state
    targets.append(np.array([0, 0, 1, 0]))
    return targets


def cx_gate_unitary_deterministic():
    """CX-gate circuits reference unitaries."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
    # CX10, |00> state
    targets.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    # CX01.(X^I), |10> state
    targets.append(np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
    # CX10.(I^X), |01> state
    targets.append(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    # CX01.(I^X), |11> state
    targets.append(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]))
    # CX10.(X^I), |11> state
    targets.append(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
    # CX01.(X^X), |01> state
    targets.append(np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    # CX10.(X^X), |10> state
    targets.append(np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]))
    return targets


def cx_gate_circuits_nondeterministic(final_measure=True):
    """CX-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # CX01.(I^H), Bell state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(H^I), Bell state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # test ctrl_states=0
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[1], ctrl_state=0)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def cx_gate_counts_nondeterministic(shots, hex_counts=True):
    """CX-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CX01.(I^H), Bell state
        targets.append({"0x0": shots / 2, "0x3": shots / 2})
        # CX10.(I^H), Bell state
        targets.append({"0x0": shots / 2, "0x3": shots / 2})
        # test ctrl_states=0
        targets.append({"0x1": shots / 2, "0x2": shots / 2})
    else:
        # CX01.(I^H), Bell state
        targets.append({"00": shots / 2, "11": shots / 2})
        # CX10.(I^H), Bell state
        targets.append({"00": shots / 2, "11": shots / 2})
        # test ctrl_states=0
        targets.append({"01": shots / 2, "10": shots / 2})
    return targets


def cx_gate_statevector_nondeterministic():
    """CX-gate circuits reference statevectors."""
    targets = []
    # CX01.(I^H), Bell state
    targets.append(np.array([1, 0, 0, 1]) / np.sqrt(2))
    # CX10.(I^H), Bell state
    targets.append(np.array([1, 0, 0, 1]) / np.sqrt(2))
    return targets


def cx_gate_unitary_nondeterministic():
    """CX-gate circuits reference unitaries."""
    targets = []
    # CX01.(I^H), Bell state
    targets.append(
        np.array([[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]]) / np.sqrt(2)
    )
    # CX10.(I^H), Bell state
    targets.append(
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]]) / np.sqrt(2)
    )
    return targets


# ==========================================================================
# CZ-gate
# ==========================================================================


def cz_gate_circuits_deterministic(final_measure=True):
    """CZ-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # CZ, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.cz(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.cz(qr[0], qr[1])
    circuit.h(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[1])
    circuit.cz(qr[1], qr[0])
    circuit.h(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (I^H).CZ.(X^H) = CX10.(X^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.cz(qr[0], qr[1])
    circuit.barrier(qr)
    circuit.h(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (H^I).CZ.(H^X) = CX01.(I^X), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.cz(qr[0], qr[1])
    circuit.barrier(qr)
    circuit.h(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def cz_gate_counts_deterministic(shots, hex_counts=True):
    """CZ-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CZ, |00> state
        targets.append({"0x0": shots})
        # CX10, |00> state
        targets.append({"0x0": shots})
        # CX01, |00> state
        targets.append({"0x0": shots})
        # (I^H).CZ.(X^H) = CX10.(X^I), |11> state
        targets.append({"0x3": shots})
        # (H^I).CZ.(H^X) = CX01.(I^H), |11> state
        targets.append({"0x3": shots})
    else:
        # CZ, |00> state
        targets.append({"00": shots})
        # CX10, |00> state
        targets.append({"00": shots})
        # CX01, |00> state
        targets.append({"00": shots})
        # (I^H).CZ.(X^H) = CX10.(X^I), |11> state
        targets.append({"11": shots})
        # (H^I).CZ.(H^X) = CX01.(I^H), |11> state
        targets.append({"11": shots})
    return targets


def cz_gate_statevector_deterministic():
    """CZ-gate test circuits with deterministic counts."""
    targets = []
    # CZ, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX10, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX01, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # (I^H).CZ.(X^H) = CX10.(X^I), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    # (H^I).CZ.(H^X) = CX01.(I^H), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    return targets


def cz_gate_unitary_deterministic():
    """CZ-gate circuits reference unitaries."""
    targets = []
    # CZ, |00> state
    targets.append(np.diag([1, 1, 1, -1]))
    # CX10, |00> state
    targets.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    # CX01, |00> state
    targets.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
    # (I^H).CZ.(X^H) = CX10.(X^I), |11> state
    targets.append(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
    # (H^I).CZ.(H^X) = CX01.(I^X), |11> state
    targets.append(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]))
    return targets


def cz_gate_circuits_nondeterministic(final_measure=True):
    """CZ-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # (I^H).CZ.(H^H) = CX10.(H^I), Bell state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.cz(qr[0], qr[1])
    circuit.barrier(qr)
    circuit.h(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (H^I).CZ.(H^H) = CX01.(I^H), Bell state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.cz(qr[0], qr[1])
    circuit.h(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def cz_gate_counts_nondeterministic(shots, hex_counts=True):
    """CZ-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # (I^H).CZ.(H^H) = CX10.(H^I), Bell state
        targets.append({"0x0": shots / 2, "0x3": shots / 2})
        # (H^I).CZ.(H^H) = CX01.(I^H), Bell state
        targets.append({"0x0": shots / 2, "0x3": shots / 2})
    else:
        # (I^H).CZ.(H^H) = CX10.(H^I), Bell state
        targets.append({"00": shots / 2, "11": shots / 2})
        # (H^I).CZ.(H^H) = CX01.(I^H), Bell state
        targets.append({"00": shots / 2, "11": shots / 2})
    return targets


def cz_gate_statevector_nondeterministic():
    """CZ-gate circuits reference statevectors."""
    targets = []
    # (I^H).CZ.(H^H) = CX10.(H^I), Bell state
    targets.append(np.array([1, 0, 0, 1]) / np.sqrt(2))
    # (H^I).CZ.(H^H) = CX01.(I^H), Bell state
    targets.append(np.array([1, 0, 0, 1]) / np.sqrt(2))
    return targets


def cz_gate_unitary_nondeterministic():
    """CZ-gate circuits reference unitaries."""
    targets = []
    # (I^H).CZ.(H^H) = CX10.(H^I), Bell state
    targets.append(
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]]) / np.sqrt(2)
    )
    # (H^I).CZ.(H^H) = CX01.(I^H), Bell state
    targets.append(
        np.array([[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]]) / np.sqrt(2)
    )
    return targets


# ==========================================================================
# SWAP-gate
# ==========================================================================


def swap_gate_circuits_deterministic(final_measure=True):
    """SWAP-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Swap(0,1), |00> state
    circuit = QuantumCircuit(*regs)
    circuit.swap(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Swap(0,1).(I^X), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.swap(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def swap_gate_counts_deterministic(shots, hex_counts=True):
    """SWAP-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Swap(0,1), |00> state
        targets.append({"0x0": shots})
        # Swap(0,1).(I^X), |10> state
        targets.append({"0x2": shots})
    else:
        # Swap(0,1), |00> state
        targets.append({"00": shots})
        # Swap(0,1).(I^X), |10> state
        targets.append({"10": shots})
    return targets


def swap_gate_statevector_deterministic():
    """SWAP-gate test circuits with deterministic counts."""
    targets = []
    # Swap(0,1), |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # Swap(0,1).(I^X), |10> state
    targets.append(np.array([0, 0, 1, 0]))
    return targets


def swap_gate_unitary_deterministic():
    """SWAP-gate circuits reference unitaries."""
    targets = []
    # Swap(0,1), |00> state
    targets.append(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    # Swap(0,1).(I^X), |10> state
    targets.append(np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]))
    return targets


def swap_gate_circuits_nondeterministic(final_measure=True):
    """SWAP-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr,)
    # initial state as |10+>

    # Swap(0,1).(X^I^H), Permutation (0,1,2) -> (1,0,2)
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.swap(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Swap(0,2).(X^I^H), # Permutation (0,1,2) -> (2,1,0),
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.swap(qr[0], qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Swap(2,0).Swap(0,1).(X^I^H), Permutation (0,1,2) -> (2,0,1)
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.swap(qr[0], qr[1])
    circuit.barrier(qr)
    circuit.swap(qr[2], qr[0])
    circuit.barrier(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    return circuits


def swap_gate_counts_nondeterministic(shots, hex_counts=True):
    """SWAP-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # initial state as |10+>
        # Swap(0,1).(X^I^H), Permutation (0,1,2) -> (1,0,2)
        targets.append({"0x4": shots / 2, "0x6": shots / 2})
        # Swap(0,2).(X^I^H), # Permutation (0,1,2) -> (2,1,0),
        targets.append({"0x1": shots / 2, "0x5": shots / 2})
        # Swap(2,0).Swap(0,1).(X^I^H), Permutation (0,1,2) -> (2,0,1)
        targets.append({"0x1": shots / 2, "0x3": shots / 2})
    else:
        # initial state as |10+>
        # Swap(0,1).(X^I^H), Permutation (0,1,2) -> (1,0,2)
        targets.append({"100": shots / 2, "110": shots / 2})
        # Swap(0,2).(X^I^H), # Permutation (0,1,2) -> (2,1,0),
        targets.append({"001": shots / 2, "101": shots / 2})
        # Swap(2,0).Swap(0,1).(X^I^H), Permutation (0,1,2) -> (2,0,1)
        targets.append({"001": shots / 2, "011": shots / 2})
    return targets


def swap_gate_statevector_nondeterministic():
    """SWAP-gate circuits reference statevectors."""
    targets = []
    # initial state as |10+>
    # Swap(0,1).(X^I^H), Permutation (0,1,2) -> (1,0,2), |1+0>
    targets.append(np.array([0, 0, 0, 0, 1, 0, 1, 0]) / np.sqrt(2))
    # Swap(0,2).(X^I^H), # Permutation (0,1,2) -> (2,1,0),
    targets.append(np.array([0, 1, 0, 0, 0, 1, 0, 0]) / np.sqrt(2))
    # Swap(2,0).Swap(0,1).(X^I^H), Permutation (0,1,2) -> (2,0,1)
    targets.append(np.array([0, 1, 0, 1, 0, 0, 0, 0]) / np.sqrt(2))
    return targets


def swap_gate_unitary_nondeterministic():
    """SWAP-gate circuits reference unitaries."""
    targets = []
    # initial state as |10+>
    # Swap(0,1).(X^I^H), Permutation (0,1,2) -> (1,0,2)
    targets.append(
        np.array(
            [
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, -1],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0, 0, 0],
            ]
        )
        / np.sqrt(2)
    )
    # Swap(0,2).(X^I^H), # Permutation (0,1,2) -> (2,1,0),
    targets.append(
        np.array(
            [
                [0, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0],
                [1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, -1],
                [0, 0, 1, -1, 0, 0, 0, 0],
            ]
        )
        / np.sqrt(2)
    )
    # Swap(2,0).Swap(0,1).(X^I^H), Permutation (0,1,2) -> (2,0,1)
    targets.append(
        np.array(
            [
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, -1],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0, 0, 0],
            ]
        )
        / np.sqrt(2)
    )
    return targets


# ==========================================================================
# ECR gate
# ==========================================================================


def ecr_gate_circuits_nondeterministic(final_measure=True):
    """ECR-gate test circuits with nondeterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # ECR, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.ecr(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # ECR, |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.ecr(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # ECR, |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.ecr(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # ECR, |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.ecr(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def ecr_gate_counts_nondeterministic(shots, hex_counts=True):
    """ECR-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # ECR, |00> state
        targets.append({"0x1": shots / 2, "0x3": shots / 2})
        # ECR, |01> state
        targets.append({"0x0": shots / 2, "0x2": shots / 2})
        # ECR, |10> state
        targets.append({"0x1": shots / 2, "0x3": shots / 2})
        # ECR, |11> state
        targets.append({"0x0": shots / 2, "0x2": shots / 2})

    else:
        # ECR, |00> state
        targets.append({"01": shots / 2, "11": shots / 2})
        # ECR, |01> state
        targets.append({"00": shots / 2, "10": shots / 2})
        # ECR, |10> state
        targets.append({"01": shots / 2, "11": shots / 2})
        # ECR, |11> state
        targets.append({"00": shots / 2, "10": shots / 2})
    return targets
