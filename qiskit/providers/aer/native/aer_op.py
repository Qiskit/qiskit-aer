import numpy
from ..backends.controller_wrappers import *

def to_aer_vec(data, T):
    if isinstance(data, list):
        return [T(d) for d in data]
    else:
        raise ValueError(f"unknown data type for conversion to complex vector: {data.__class__ }")

# OpType.barrier
def barrier(qubits):
    op = AerOp()
    op.type = OpType.barrier
    op.name = "barrier"
    op.qubits = qubits
    return op

# OpType.reset:
def reset(qubits):
    op = AerOp()
    op.type = OpType.reset
    op.name = "reset"
    op.qubits = qubits
    return op

# OpType.initialize:
def initialize(qubits, initial_state):
    op = AerOp()
    op.type = OpType.initialize
    op.name = "initialize"
    op.qubits = qubits
    op.params = to_aer_vec(initial_state, complex)
    return op

# OpType.measure:
def measure(qubits, memory):
    op = AerOp()
    op.type = OpType.measure
    op.name = "measure"
    op.qubits = qubits
    op.memory = memory
    op.registers = memory
    return op

# OpType.bfunc:

# OpType.roerror:
def _roerror(memory, probs):
    op = AerOp()
    op.type = OpType.roerror
    op.name = "roerror"
    op.memory = memory
    op.probs = to_aer_vec(probs, float)
    return op

# OpType.gate:
def u1(qubit, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "u1"
    op.qubits = [qubit]
    op.params = [lam]
    return op

def u2(qubit, phi, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "u2"
    op.qubits = [qubit]
    op.params = [phi, lam]
    return op

def u3(qubit, theta, phi, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "u3"
    op.qubits = [qubit]
    op.params = [theta, phi, lam]
    return op

def cx(ctrl_qubit, tgt_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cx"
    op.qubits = [ctrl_qubit, tgt_qubit]
    return op

def cz(ctrl_qubit, tgt_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cz"
    op.qubits = [ctrl_qubit, tgt_qubit]
    return op

def cy(ctrl_qubit, tgt_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cy"
    op.qubits = [ctrl_qubit, tgt_qubit]
    return op

def cp(ctrl_qubit, tgt_qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cp"
    op.qubits = [ctrl_qubit, tgt_qubit]
    op.params = [theta]
    return op

def cu1(ctrl_qubit, qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu1"
    op.qubits = [ctrl_qubit, qubit]
    op.params = [theta]
    return op

def cu2(ctrl_qubit, qubit, phi, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu2"
    op.qubits = [ctrl_qubit, qubit]
    op.params = [phi, lam]
    return op

def cu3(ctrl_qubit, qubit, theta, phi, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu3"
    op.qubits = [ctrl_qubit, qubit]
    op.params = [theta, phi, lam]
    return op

def swap(qubit1, qubit2):
    op = AerOp()
    op.type = OpType.gate
    op.name = "swap"
    op.qubits = [qubit1, qubit2]
    return op

def id(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "id"
    op.qubits = [qubit]
    return op

def p(qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "p"
    op.qubits = [qubit]
    op.params = [theta]
    return op

def x(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "x"
    op.qubits = [qubit]
    return op

def y(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "y"
    op.qubits = [qubit]
    return op

def z(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "z"
    op.qubits = [qubit]
    return op

def h(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "h"
    op.qubits = [qubit]
    return op

def s(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "s"
    op.qubits = [qubit]
    return op

def sdg(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "sdg"
    op.qubits = [qubit]
    return op

def t(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "s"
    op.qubits = [qubit]
    return op

def tdg(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "tdg"
    op.qubits = [qubit]
    return op

def r(qubit, theta, phi):
    op = AerOp()
    op.type = OpType.gate
    op.name = "r"
    op.qubits = [qubit]
    op.params = [theta, phi]
    return op

def rx(qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "rx"
    op.qubits = [qubit]
    op.params = [theta]
    return op

def ry(qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "ry"
    op.qubits = [qubit]
    op.params = [theta]
    return op

def rz(qubit, phi):
    op = AerOp()
    op.type = OpType.gate
    op.name = "rz"
    op.qubits = [qubit]
    op.params = [phi]
    return op

def rxx(qubit1, qubit2, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "rxx"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op

def ryy(qubit1, qubit2, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "ryy"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op

def rzz(qubit1, qubit2, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "rzz"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op

def rzx(qubit1, qubit2, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "rzx"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op

def ccx(control_qubit1, control_qubit2, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "ccx"
    op.qubits = [control_qubit1, control_qubit2, target_qubit]
    return op

def cswap(control_qubit, target_qubit1, target_qubit2):
    op = AerOp()
    op.type = OpType.gate
    op.name = "cswap"
    op.qubits = [control_qubit, target_qubit1, target_qubit2]
    return op

def mcx(control_qubits, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcx"
    op.qubits = [*control_qubits, target_qubit]
    return op

def mcy(control_qubits, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcy"
    op.qubits = [*control_qubits, target_qubit]
    return op

def mcz(control_qubits, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcz"
    op.qubits = [*control_qubits, target_qubit]
    return op

def mcu1(control_qubits, target_qubit, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu1"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [lam]
    return op

def mcu2(control_qubits, target_qubit, phi, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu2"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [phi, lam]
    return op

def mcu3(control_qubits, target_qubit, theta, phi, lam):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu3"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta, phi, lam]
    return op

def mcswap(control_qubits, target_qubit1, target_qubit2):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcswap"
    op.qubits = [*control_qubits, target_qubit1, target_qubits2]
    return op

def mcphase(control_qubits, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcphase"
    op.qubits = [*control_qubits, target_qubit]
    return op

def mcr(control_qubits, target_qubit, theta, phi):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcr"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta, phi]
    return op

def mcrx(control_qubits, target_qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcrx"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta]
    return op

def mcry(control_qubits, target_qubit, theta):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcry"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta]
    return op

def sx(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "sx"
    op.qubits = [qubit]
    return op

def csx(control_qubit, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "csx"
    op.qubits = [control_qubit, target_qubit]
    return op

def mcsx(control_qubits, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcsx"
    op.qubits = [*control_qubits, target_qubit]
    return op

def delay(qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "delay"
    op.qubits = [qubit]
    return op

def pauli(qubits, pauli_string):
    op = AerOp()
    op.type = OpType.gate
    op.name = "pauli"
    op.qubits = qubits
    op.string_params = [pauli_string]
    return op

def mcx_gray(control_qubits, target_qubit):
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcx_gray"
    op.qubits = [*control_qubits, target_qubit]
    op.qubits.push_back(target_qubit)
    return op

# OpType.snapshot:
def snapshot_statevector(qubits, name):
#     {{"statevector", Snapshots::statevector},
    op = AerOp()
    op.type = OpType.snapshot
    op.name = 'statevector'
    op.string_params = [name]
    return op

def snapshot_probabilities(qubits, name):
#      {"probabilities", Snapshots::probs},
    op = AerOp()
    op.type = OpType.snapshot
    op.name = 'probabilities'
    op.string_params = [name]
    return op

def snapshot_expectation_value_pauli(qubits, coeff_and_paulistring_list):
#      {"expectation_value_pauli", Snapshots::expval_pauli},
    op = AerOp()
    op.type = OpType.snapshot
    op.name = 'expectation_value_pauli'
    os.params_expval_pauli = coeff_and_paulistring_list
    return op

#      {"expectation_value_matrix", Snapshots::expval_matrix},
#      {"probabilities_with_variance", Snapshots::probs_var},
#      {"density_matrix", Snapshots::densmat},
#      {"density_matrix_with_variance", Snapshots::densmat_var},
#      {"expectation_value_pauli_with_variance", Snapshots::expval_pauli_var},
#      {"expectation_value_matrix_with_variance", Snapshots::expval_matrix_var},
#      {"expectation_value_pauli_single_shot", Snapshots::expval_pauli_shot},
#      {"expectation_value_matrix_single_shot", Snapshots::expval_matrix_shot},
#      {"memory", Snapshots::cmemory},
#      {"register", Snapshots::cregister}});

# OpType.matrix:
def unitary(qubits, mat):
    op = AerOp()
    op.type = OpType.matrix
    op.name = 'unitary'
    op.qubits = qubits
    op.mats = [mat]
    return op

# OpType.diagonal_matrix:
def diagonal(qubits, vec):
    op = AerOp()
    op.type = OpType.diagonal_matrix
    op.name = "diagonal"
    op.qubits = qubits
    op.params = vec
    return op

# OpType.multiplexer:

# OpType.kraus:
def _kraus(qubits, mats):
    op = AerOp()
    op.type = OpType.kraus
    op.name = "kraus"
    op.qubits = qubits
    op.mats = mats
    return op

def _superop(qubits, mat):
    op = AerOp()
    op.type = OpType.superop
    op.name = "superop"
    op.qubits = qubits
    op.mats = [mat]
    return op

def gen_aer_op(instruction, qubits, clbits):
    params = instruction.params
    if instruction.name == 'barrier':
        return barrier(qubits)
    elif instruction.name == 'reset':
        return reset(qubits)
    elif instruction.name == 'initialize':
        return initialize(qubits, params[0])
    elif instruction.name == 'measure':
        return measure(qubits, clbits)
    #TODO roerror
    elif instruction.name == 'u1':
        return u1(qubits[0], params[0])
    elif instruction.name == 'u2':
        return u2(qubits[0], params[0], params[1])
    elif instruction.name == 'u3':
        return u3(qubits[0], params[0], params[1], params[2])
    elif instruction.name == 'cx':
        return cx(qubits[0], qubits[1])
    elif instruction.name == 'cz':
        return cz(qubits[0], qubits[1])
    elif instruction.name == 'cy':
        return cy(qubits[0], qubits[1])
    elif instruction.name == 'cp':
        return cp(qubits[0], qubits[1], params[0])
    elif instruction.name == 'cu1':
        return cu1(qubits[0], qubits[1], params[0])
    elif instruction.name == 'cu2':
        return cu2(qubits[0], qubits[1], params[0], params[1])
    elif instruction.name == 'cu3':
        return cu2(qubits[0], qubits[1], params[0], params[1], params[2])
    elif instruction.name == 'swap':
        return swap(qubits[0], qubits[1])
    elif instruction.name == 'id':
        return id(qubits[0])
    elif instruction.name == 'p':
        return p(qubits[0], params[0])
    elif instruction.name == 'x':
        return x(qubits[0])
    elif instruction.name == 'y':
        return y(qubits[0])
    elif instruction.name == 'z':
        return z(qubits[0])
    elif instruction.name == 'h':
        return h(qubits[0])
    elif instruction.name == 's':
        return s(qubits[0])
    elif instruction.name == 'sdg':
        return sdg(qubits[0])
    elif instruction.name == 't':
        return t(qubits[0])
    elif instruction.name == 'tdg':
        return tdg(qubits[0])
    elif instruction.name == 'r':
        return r(qubits[0], params[0], params[1])
    elif instruction.name == 'rx':
        return rx(qubits[0], params[0])
    elif instruction.name == 'ry':
        return ry(qubits[0], params[0])
    elif instruction.name == 'rz':
        return rz(qubits[0], params[0])
    elif instruction.name == 'rxx':
        return rxx(qubits[0], qubits[1], params[0])
    elif instruction.name == 'ryy':
        return ryy(qubits[0], qubits[1], params[0])
    elif instruction.name == 'rzz':
        return rzz(qubits[0], qubits[1], params[0])
    elif instruction.name == 'rzx':
        return rzx(qubits[0], qubits[1], params[0])
    elif instruction.name == 'cxx':
        return cxx(qubits[0], qubits[1], qubits[2])
    elif instruction.name == 'cswap':
        return cswap(qubits[0], qubits[1], qubits[2])
    elif instruction.name == 'mcx':
        return mcx(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcy':
        return mcy(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcz':
        return mcz(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcu1':
        return mcu1(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'mcu2':
        return mcu2(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0], params[1])
    elif instruction.name == 'mcu3':
        return mcu3(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0], params[1], params[2])
    elif instruction.name == 'mcswap':
        return mcswap(qubits[0: len(qubits) - 2], qubits[len(qubits) - 2], qubits[len(qubits) - 2])
    elif instruction.name == 'mcphase':
        return mcphase(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'mcr':
        return mcr(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0], params[1])
    elif instruction.name == 'mcx':
        return mcx(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'mcry':
        return mcry(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'sx':
        return sx(qubits[0])
    elif instruction.name == 'csx':
        return csx(qubits[0], qubits[1])
    elif instruction.name == 'mcsx':
        return mcsx(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'delay':
        return delay(qubits[0])
    elif instruction.name == 'pauli':
        return pauli(qubits, params[0])
    elif instruction.name == 'mcx_gray':
        return mcx_gray(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcx_gray':
        return mcx_gray(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'unitary':
        return unitary(qubits, params[0])
    elif instruction.name == 'diagonal':
        return diagonal(qubits, to_aer_vec(diagonal_gate.params[0], complex))
    else:
        raise ValueError(f'not implemented yet: {instruction}')
    
def gen_aer_circuit(circuit):

    global_phase = circuit.global_phase

    qubit_indices = {bit: index for index, bit in enumerate(circuit.qubits)}
    clbit_indices = {bit: index for index, bit in enumerate(circuit.clbits)}

    circ = AerCircuit([gen_aer_op(instruction[0],
                                  [qubit_indices[qubit] for qubit in instruction[1]],
                                  [clbit_indices[clbit] for clbit in instruction[2]])
                                  for instruction in circuit.data])
    circ.global_phase_angle = global_phase
    return circ
