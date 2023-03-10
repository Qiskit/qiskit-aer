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
Test circuits and reference outputs for reset instruction.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# Deterministic output
# ==========================================================================


def reset_circuits_deterministic(final_measure=True):
    """Reset test circuits with deterministic count output"""

    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Reset 0 from |11>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.reset(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 1 from |11>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.reset(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,1 from |11>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.reset(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,1 from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.reset(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def reset_counts_deterministic(shots, hex_counts=True):
    """Reset test circuits reference counts."""
    targets = []
    if hex_counts:
        # Reset 0 from |11>
        targets.append({"0x2": shots})
        # Reset 1 from |11>
        targets.append({"0x1": shots})
        # Reset 0,1 from |11>
        targets.append({"0x0": shots})
        # Reset 0,1 from |++>
        targets.append({"0x0": shots})
    else:
        # Reset 0 from |11>
        targets.append({"10": shots})
        # Reset 1 from |11>
        targets.append({"01": shots})
        # Reset 0,1 from |11>
        targets.append({"00": shots})
        # Reset 0,1 from |++>
        targets.append({"00": shots})
    return targets


def reset_statevector_deterministic():
    """Reset test circuits reference counts."""
    targets = []
    # Reset 0 from |11>
    targets.append(array([0, 0, 1, 0]))
    # Reset 1 from |11>
    targets.append(array([0, 1, 0, 0]))
    # Reset 0,1 from |11>
    targets.append(array([1, 0, 0, 0]))
    # Reset 0,1 from |++>
    targets.append(array([1, 0, 0, 0]))
    return targets


# ==========================================================================
# Non-Deterministic output
# ==========================================================================


def reset_circuits_nondeterministic(final_measure=True):
    """Reset test circuits with deterministic count output"""

    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Reset 0 from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.reset(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 1 from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.reset(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def reset_counts_nondeterministic(shots, hex_counts=True):
    """Reset test circuits reference counts."""
    targets = []
    if hex_counts:
        # Reset 0 from |++>
        targets.append({"0x0": shots / 2, "0x2": shots / 2})
        # Reset 1 from |++>
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
    else:
        # Reset 0 from |++>
        targets.append({"00": shots / 2, "10": shots / 2})
        # Reset 1 from |++>
        targets.append({"00": shots / 2, "01": shots / 2})
    return targets


def reset_statevector_nondeterministic():
    """Reset test circuits reference counts."""
    targets = []
    # Reset 0 from |++>
    targets.append(array([1, 0, 1, 0]) / sqrt(2))
    # Reset 1 from |++>
    targets.append(array([1, 1, 0, 0]) / sqrt(2))
    return targets


# ==========================================================================
# Repeated Resets
# ==========================================================================


def reset_circuits_repeated():
    """Test circuit for repeated measure reset"""
    qr = QuantumRegister(1)
    cr = ClassicalRegister(2)
    qc = QuantumCircuit(qr, cr)
    qc.x(qr[0])
    qc.measure(qr[0], cr[0])
    qc.reset(qr[0])
    qc.measure(qr[0], cr[1])
    qc.reset(qr[0])
    return [qc]


def reset_counts_repeated(shots, hex_counts=True):
    """Sampling optimization counts"""
    if hex_counts:
        return [{"0x1": shots}]
    else:
        return [{"01": shots}]


# ==========================================================================
# Sampling optimization
# ==========================================================================


def reset_circuits_sampling_optimization():
    """Test sampling optimization"""
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    qc = QuantumCircuit(qr, cr)

    # The optimization should not be triggerred
    # because the reset operation performs randomizations
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.reset([qr[0]])
    qc.measure(qr, cr)

    return [qc]


def reset_counts_sampling_optimization(shots, hex_counts=True):
    """Sampling optimization counts"""
    if hex_counts:
        return [{"0x0": shots / 2, "0x2": shots / 2}]
    else:
        return [{"00": shots / 2, "10": shots / 2}]


def reset_circuits_with_entangled_and_moving_qubits(final_measure=True):
    """Reset test circuits with entangled and moving qubits count output"""

    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # Reset 0 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 1 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 2 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,1 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([0, 1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,2 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([0, 2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 1,2 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([1, 2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,1,2 from |000+111>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.reset([0, 1, 2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def reset_counts_with_entangled_and_moving_qubits(shots, hex_counts=True):
    """Reset test circuits reference counts."""
    targets = []
    if hex_counts:
        # Reset 0 from |000+111>
        targets.append({"0x0": shots / 2, "0x6": shots / 2})
        # Reset 1 from |000+111>
        targets.append({"0x0": shots / 2, "0x5": shots / 2})
        # Reset 2 from |000+111>
        targets.append({"0x0": shots / 2, "0x3": shots / 2})
        # Reset 0,1 from |000+111>
        targets.append({"0x0": shots / 2, "0x4": shots / 2})
        # Reset 0,2 from |000+111>
        targets.append({"0x0": shots / 2, "0x2": shots / 2})
        # Reset 1,2 from |000+111>
        targets.append({"0x0": shots / 2, "0x1": shots / 2})
        # Reset 0,1,2 from |000+111>
        targets.append({"0x0": shots})
    else:
        # Reset 0 from |000+111>
        targets.append({"000": shots / 2, "110": shots / 2})
        # Reset 1 from |000+111>
        targets.append({"000": shots / 2, "101": shots / 2})
        # Reset 2 from |000+111>
        targets.append({"000": shots / 2, "011": shots / 2})
        # Reset 0,1 from |000+111>
        targets.append({"000": shots / 2, "100": shots / 2})
        # Reset 0,2 from |000+111>
        targets.append({"000": shots / 2, "010": shots / 2})
        # Reset 1,2 from |000+111>
        targets.append({"000": shots / 2, "001": shots / 2})
        # Reset 0,1,2 from |000+111>
        targets.append({"000": shots})
    return targets
