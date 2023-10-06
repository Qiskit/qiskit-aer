# -*- coding: utf-8 -*-

# Copyright 2021, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import json
import os

from qiskit import ClassicalRegister
from qiskit.compiler import assemble, transpile
from qiskit import QuantumCircuit
from qiskit import QuantumRegister


def grovers_circuit(final_measure=True, allow_sampling=True):
    """Testing a circuit originated in the Grover algorithm"""

    circuits = []

    # 6-qubit grovers
    qr = QuantumRegister(6)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr,)
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
        circuit.id(qr)
    circuits.append(circuit)

    return circuits


if __name__ == "__main__":
    # Run qasm simulator
    shots = 4000
    circuits = grovers_circuit(final_measure=True, allow_sampling=True)
    if os.getenv("USE_MPI", False):
        qobj = assemble(transpile(circuits), shots=shots, blocking_enable=True, blocking_qubits=2)
    else:
        qobj = assemble(transpile(circuits), shots=shots)
    with open("qobj.json", "wt") as fp:
        json.dump(qobj.to_dict(), fp)
