from qiskit.qobj.models.qasm import QasmQobjInstruction, QasmQobjExperiment
from qiskit.qobj.models.base import QobjHeader
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import copy
from copy import deepcopy
from pprint import pprint
import random

from qiskit.assembler import assemble_circuits

def create_circuit(num_qubits, number_of_gates, seed):
    # Generate a quantum circuit
    q = QuantumRegister(num_qubits)
    c = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(q, c, name="tn_circuit")

    q_header = QobjHeader(backend_name="qasm", backend_version="0.2.0")
    qasm_qobj = assemble_circuits([qc], run_config=[], qobj_id="1", qobj_header=q_header) #returns QasmQobj

    random.seed(seed)
    kind_of_gate = ['one qubit', 'two qubits']
    one_qubit = ['x', 'y', 'z', 'h', 't', 's', 'tdg', 'sdg', 'id', 'u1', 'u2','u3']
    two_qubits = ['cx', 'cz', 'swap']

    gates = []
    for num_gates in range(number_of_gates):
        qubit = random.choice(range(num_qubits))
        phase_params = []
        qubits = [qubit]

        kind = random.choice(kind_of_gate)
        if(kind == 'one qubit'):
            gate = random.choice(one_qubit)
        else:
            gate = random.choice(two_qubits)
        gates.append(gate);

        if gate in ['u1','u2','u3']:
           lambda_ = random.uniform(0, np.pi)
           phase_params.append(lambda_)
        if gate in ['u2','u3']:
           phi_ = random.uniform(0, np.pi)
           phase_params.append(phi_)
        if gate in ['u3']:
           theta_ = random.uniform(0, np.pi)
           phase_params.append(theta_)

        if gate in ['cx', 'cz', 'swap']:
           choices = list(range(num_qubits))
           choices.remove(qubit)
           second_qubit = random.choice(choices)
           qubits.append(second_qubit)

        if gate in ['u1', 'u2', 'u3']:
           next_op = QasmQobjInstruction(name = gate, qubits = qubits, params = phase_params)
        else:
           next_op = QasmQobjInstruction(name = gate, qubits = qubits)
        qasm_qobj.experiments[0].instructions.append(next_op)	

    return qasm_qobj

