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
Test circuits and reference outputs for multiplexer gates.
"""
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from test.terra.utils.multiplexer import multiplexer_multi_controlled_x
from test.terra.reference.ref_2q_clifford import (
    cx_gate_counts_nondeterministic,
    cx_gate_counts_deterministic,
)
from test.terra.reference.ref_non_clifford import (
    ccx_gate_counts_nondeterministic,
    ccx_gate_counts_deterministic,
)
from qiskit.quantum_info.states import Statevector
from qiskit.circuit.library import Isometry, UCGate


def multiplexer_cx_gate_circuits_deterministic(final_measure=True):
    """multiplexer-gate simulating cx gate, test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    num_control_qubits = 1

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[1], qr[0]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(X^I), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[1], qr[0]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(I^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(I^X), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[1], qr[0]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(X^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(X^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[1], qr[0]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(X^X), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def multiplexer_cx_gate_circuits_nondeterministic(final_measure=True):
    """Multiplexer CX-like gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # cx gate only has one control qubit
    num_control_qubits = 1

    # CX01.(I^H), Bell state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[1], qr[0]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(H^I), Bell state
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def multiplexer_cx_gate_counts_deterministic(shots, hex_counts=True):
    """The counts are exactly the same as the cx gate"""
    return cx_gate_counts_deterministic(shots, hex_counts)


def multiplexer_cx_gate_counts_nondeterministic(shots, hex_counts=True):
    """The counts are exactly the same as the cx gate"""
    return cx_gate_counts_nondeterministic(shots, hex_counts)


# ==========================================================================
# Multiplexer-gate (CCX-like)
# ==========================================================================
def multiplexer_ccx_gate_circuits_deterministic(final_measure=True):
    """multiplexer-gate simulating ccx gate, test circuits with deterministic counts."""

    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # because ccx has two control qubits and one target
    num_control_qubits = 2

    # CCX(0,1,2)
    circuit = QuantumCircuit(*regs)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[2], qr[0], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (I^X^X).CCX(0,1,2).(I^X^X) -> |100>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[2], qr[0], qr[1]])
    circuit.barrier(qr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CCX(2,1,0)
    circuit = QuantumCircuit(*regs)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[2], qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^X^I).CCX(2,1,0).(X^X^I) -> |001>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[2], qr[1]])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def multiplexer_ccx_gate_circuits_nondeterministic(final_measure=True):
    """mukltiplexer CCX-like gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr,)

    # because ccx has two control qubits and one target
    num_control_qubits = 2

    # (I^X^I).CCX(0,1,2).(I^X^H) -> |000> + |101>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[2], qr[0], qr[1]])
    circuit.barrier(qr)
    circuit.x(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^I^I).CCX(2,1,0).(X^H^I) -> |000> + |011>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.append(multiplexer_multi_controlled_x(num_control_qubits), [qr[0], qr[2], qr[1]])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def multiplexer_ccx_gate_counts_deterministic(shots, hex_counts=True):
    """The counts are exactly the same as the ccx gate"""
    return ccx_gate_counts_deterministic(shots, hex_counts)


def multiplexer_ccx_gate_counts_nondeterministic(shots, hex_counts=True):
    """The counts are exactly the same as the ccx gate"""
    return ccx_gate_counts_nondeterministic(shots, hex_counts)


def multiplexer_no_control_qubits(final_measure=True):
    qc = QuantumCircuit(1, 1)
    vector = [0.2, 0.1]
    vector_circuit = QuantumCircuit(1)
    vector_circuit.append(Isometry(vector / np.linalg.norm(vector), 0, 0), [0])
    vector_circuit = vector_circuit.inverse()
    qc.append(vector_circuit, [0])

    sv = Statevector(qc)
    gate_list = [np.array([[sv[0], -sv[1]], [sv[1], sv[0]]])]
    qc = QuantumCircuit(1, 1)
    qc.append(UCGate(gate_list), [0])

    if final_measure:
        qc.measure(0, 0)
    return [transpile(qc, basis_gates=["multiplexer", "measure"])]
