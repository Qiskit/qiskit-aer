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
Test circuits and reference outputs for standard algorithms.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


def grovers_circuit(final_measure=True, allow_sampling=True):
    """Testing a circuit originated in the Grover algorithm"""

    circuits = []

    # 6-qubit grovers
    qr = QuantumRegister(6)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )
    circuit = QuantumCircuit(*regs)

    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.x(qr[2])
    circuit.x(qr[3])
    circuit.x(qr[0])
    circuit.cx(qr[0], qr[2])
    circuit.x(qr[0])
    circuit.cx(qr[1], qr[3])
    circuit.ccx(qr[2], qr[3], qr[4])
    circuit.cx(qr[1], qr[3])
    circuit.x(qr[0])
    circuit.cx(qr[0], qr[2])
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.x(qr[4])
    circuit.h(qr[4])
    circuit.ccx(qr[0], qr[1], qr[4])
    circuit.h(qr[4])
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.x(qr[4])
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.h(qr[4])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
    circuits.append(circuit)

    return circuits


def grovers_counts(shots, hex_counts=True):
    """Reference counts for Grovers algorithm"""
    targets = []
    if hex_counts:
        # 6-qubit grovers
        targets.append({'0x0': 5 * shots / 8, '0x1': shots / 8,
                        '0x2': shots / 8, '0x3': shots / 8})
    else:
        # 6-qubit grovers
        targets.append({'00': 5 * shots / 8, '01': shots / 8,
                        '10': shots / 8, '11': shots / 8})
    return targets


def teleport_circuit():
    """Testing a circuit originated in the teleportation algorithm"""

    circuits = []

    # Classic 3-qubit teleportation
    qr = QuantumRegister(3)
    c0 = ClassicalRegister(1)
    c1 = ClassicalRegister(1)
    c2 = ClassicalRegister(1)
    # Compiles to creg order [c2, c1, c0]
    circuit = QuantumCircuit(qr, c0, c1, c2)

    # Teleport the |0> state from qr[0] to qr[2]
    circuit.h(qr[1])
    circuit.cx(qr[1], qr[2])
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    circuit.h(qr[0])
    circuit.measure(qr[0], c0[0])
    circuit.measure(qr[1], c1[0])
    circuit.z(qr[2]).c_if(c0, 1)
    circuit.x(qr[2]).c_if(c1, 1)
    circuit.measure(qr[2], c2[0])
    circuits.append(circuit)

    return circuits


def teleport_counts(shots, hex_counts=True):
    """Reference counts for teleport circuits"""
    targets = []
    if hex_counts:
        # Classical 3-qubit teleport
        targets.append({'0x0': shots / 4, '0x1': shots / 4,
                        '0x2': shots / 4, '0x3': shots / 4})
    else:
        # Classical 3-qubit teleport
        targets.append({'0 0 0': shots / 4, '0 0 1': shots / 4,
                        '0 1 0': shots / 4, '0 1 1': shots / 4})
    return targets
