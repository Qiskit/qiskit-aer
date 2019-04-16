# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for measure instruction.
"""


import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit.mapper import two_qubit_kak

def quantum_volume(qubit, final_measure=True, depth=10, seed=0):
    qreg = QuantumRegister(qubit)
    width = qubit
    
    np.random.seed(seed)
    name = "Qvolume: %s by %s, seed: %s" %(width, depth, seed)
    circuit = QuantumCircuit(qreg, name=name)
    
    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = np.random.permutation(width)

        # For each pair p in Pj, generate Haar random U(4)
        # Decompose each U(4) into CNOT + SU(2)
        for k in range(width // 2):
            U = random_unitary_matrix(4)
            for gate in two_qubit_kak(U):
                qs = [qreg[int(perm[2 * k + i])] for i in gate["args"]]
                pars = gate["params"]
                name = gate["name"]
                if name == "cx":
                    circuit.cx(qs[0], qs[1])
                elif name == "u1":
                    circuit.u1(pars[0], qs[0])
                elif name == "u2":
                    circuit.u2(*pars[:2], qs[0])
                elif name == "u3":
                    circuit.u3(*pars[:3], qs[0])
                elif name == "id":
                    pass  # do nothing
                else:
                    raise Exception("Unexpected gate name: %s" % name)
    
    circuit.barrier(qreg)
    
    if final_measure:
        creg = ClassicalRegister(qubit)
        circuit.add_register(creg)
        circuit.measure(qreg, creg)
    
    return circuit
    
