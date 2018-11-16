"""
Generate quantum volume circuits.
"""

import math
from numpy import random
from scipy import linalg
from itertools import repeat
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.mapper import two_qubit_kak


def quantum_volume_circuit(n, depth, measure=True, seed=None):
    """Create a quantum volume circuit without measurement.

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    Args:
        n (int): number of qubits
        depth (int): ideal depth of each model circuit (over SU(4))
        measure (bool): include measurement in circuit.

    Returns:
        QuantumCircuit: A quantum volume circuit.
    """
    # Create random number generator with possibly fixed seed
    rng = random.RandomState(seed)
    # Create quantum/classical registers of size n
    qr = QuantumRegister(n)
    circuit = QuantumCircuit(qr)
    # For each layer
    for _ in repeat(None, depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = rng.permutation(n)
        # For each consecutive pair in Pj, generate Haar random SU(4)
        # Decompose each SU(4) into CNOT + SU(2) and add to Ci
        for k in range(math.floor(n / 2)):
            # Generate random SU(4) matrix
            X = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
            SU4, _ = linalg.qr(X)           # Q is a unitary matrix
            SU4 /= pow(linalg.det(SU4), 1 / 4)   # make Q a special unitary
            decomposed_SU4 = two_qubit_kak(SU4)  # Decompose into CX and U gates
            qubits = [int(perm[2 * k]), int(perm[2 * k + 1])]
            for gate in decomposed_SU4:
                i0 = qubits[gate["args"][0]]
                if gate["name"] == "cx":
                    i1 = qubits[gate["args"][1]]
                    circuit.cx(qr[i0], qr[i1])
                elif gate["name"] == "u1":
                    circuit.u1(gate["params"][2], qr[i0])
                elif gate["name"] == "u2":
                    circuit.u2(gate["params"][1], gate["params"][2], qr[i0])
                elif gate["name"] == "u3":
                    circuit.u3(gate["params"][0], gate["params"][1],
                               gate["params"][2], qr[i0])
                elif gate["name"] == "id":
                    pass
    if measure is True:
        cr = ClassicalRegister(n)
        meas = QuantumCircuit(qr, cr)
        meas.barrier(qr)
        meas.measure(qr, cr)
        circuit = circuit + meas
    return circuit
