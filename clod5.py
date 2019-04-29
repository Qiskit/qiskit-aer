import numpy as np
import sympy
import json

from qiskit import *
from qiskit.compiler import *
from qiskit.circuit import Gate
from qiskit.providers.aer import *

# CNOT multiplexer
I = sympy.Matrix(np.eye(2, dtype=complex))
X = sympy.Matrix(np.array([[0, 1],[1, 0]],  dtype=complex))

mplex = Gate('multiplexer', 2, [I, X])

qr = QuantumRegister(2)
circ = QuantumCircuit(qr)
circ.h(qr[0])
circ.append(mplex, [qr[0], qr[1]])

qobj = assemble_circuits(circ, RunConfig(shots=1))
print(json.dumps(qobj.as_dict()))

