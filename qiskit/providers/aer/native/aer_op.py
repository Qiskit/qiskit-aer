# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Qiskit Aer helper functions to use C++ objects with pybind11.
"""
# pylint: disable=import-error, no-name-in-module, invalid-name
from qiskit.providers.aer.backends.controller_wrappers import AerOp, AerCircuit, OpType


def to_aer_vec(data, type_):
    """conversion to list"""
    if isinstance(data, list):
        return [type_(d) for d in data]
    else:
        raise ValueError(f"unknown data type for conversion to complex vector: {data.__class__ }")


# OpType.barrier
def barrier(qubits):
    """barrier"""
    op = AerOp()
    op.type = OpType.barrier
    op.name = "barrier"
    op.qubits = qubits
    return op


# OpType.reset:
def reset(qubits):
    """reset"""
    op = AerOp()
    op.type = OpType.reset
    op.name = "reset"
    op.qubits = qubits
    return op


# OpType.initialize:
def initialize(qubits, initial_state):
    """initialize"""
    op = AerOp()
    op.type = OpType.initialize
    op.name = "initialize"
    op.qubits = qubits
    op.params = to_aer_vec(initial_state, complex)
    return op


# OpType.measure:
def measure(qubits, memory):
    """measure"""
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
    """roerror"""
    op = AerOp()
    op.type = OpType.roerror
    op.name = "roerror"
    op.memory = memory
    op.probs = to_aer_vec(probs, float)
    return op


# OpType.gate:
def u1(qubit, lam):
    """u1"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u1"
    op.qubits = [qubit]
    op.params = [lam]
    return op


def u2(qubit, phi, lam):
    """u2"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u2"
    op.qubits = [qubit]
    op.params = [phi, lam]
    return op


def u3(qubit, theta, phi, lam):
    """u3"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u3"
    op.qubits = [qubit]
    op.params = [theta, phi, lam]
    return op


def cx(ctrl_qubit, tgt_qubit):
    """cx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cx"
    op.qubits = [ctrl_qubit, tgt_qubit]
    return op


def cz(ctrl_qubit, tgt_qubit):
    """cz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cz"
    op.qubits = [ctrl_qubit, tgt_qubit]
    return op


def cy(ctrl_qubit, tgt_qubit):
    """cy"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cy"
    op.qubits = [ctrl_qubit, tgt_qubit]
    return op


def cp(ctrl_qubit, tgt_qubit, theta):
    """cp"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cp"
    op.qubits = [ctrl_qubit, tgt_qubit]
    op.params = [theta]
    return op


def cu1(ctrl_qubit, qubit, theta):
    """cu1"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu1"
    op.qubits = [ctrl_qubit, qubit]
    op.params = [theta]
    return op


def cu2(ctrl_qubit, qubit, phi, lam):
    """cu2"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu2"
    op.qubits = [ctrl_qubit, qubit]
    op.params = [phi, lam]
    return op


def cu3(ctrl_qubit, qubit, theta, phi, lam):
    """cu3"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu3"
    op.qubits = [ctrl_qubit, qubit]
    op.params = [theta, phi, lam]
    return op


def swap(qubit1, qubit2):
    """swap"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "swap"
    op.qubits = [qubit1, qubit2]
    return op


def _id(qubit):
    """_id"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "id"
    op.qubits = [qubit]
    return op


def p(qubit, theta):
    """p"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "p"
    op.qubits = [qubit]
    op.params = [theta]
    return op


def x(qubit):
    """x"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "x"
    op.qubits = [qubit]
    return op


def y(qubit):
    """y"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "y"
    op.qubits = [qubit]
    return op


def z(qubit):
    """z"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "z"
    op.qubits = [qubit]
    return op


def h(qubit):
    """h"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "h"
    op.qubits = [qubit]
    return op


def s(qubit):
    """s"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "s"
    op.qubits = [qubit]
    return op


def sdg(qubit):
    """sdg"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "sdg"
    op.qubits = [qubit]
    return op


def t(qubit):
    """t"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "s"
    op.qubits = [qubit]
    return op


def tdg(qubit):
    """tdg"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "tdg"
    op.qubits = [qubit]
    return op


def r(qubit, theta, phi):
    """r"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "r"
    op.qubits = [qubit]
    op.params = [theta, phi]
    return op


def rx(qubit, theta):
    """rx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rx"
    op.qubits = [qubit]
    op.params = [theta]
    return op


def ry(qubit, theta):
    """ry"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "ry"
    op.qubits = [qubit]
    op.params = [theta]
    return op


def rz(qubit, phi):
    """rz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rz"
    op.qubits = [qubit]
    op.params = [phi]
    return op


def rxx(qubit1, qubit2, theta):
    """rxx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rxx"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op


def ryy(qubit1, qubit2, theta):
    """ryy"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "ryy"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op


def rzz(qubit1, qubit2, theta):
    """rzz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rzz"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op


def rzx(qubit1, qubit2, theta):
    """rzx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rzx"
    op.qubits = [qubit1, qubit2]
    op.params = [theta]
    return op


def ccx(control_qubit1, control_qubit2, target_qubit):
    """ccx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "ccx"
    op.qubits = [control_qubit1, control_qubit2, target_qubit]
    return op


def cswap(control_qubit, target_qubit1, target_qubit2):
    """cswap"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cswap"
    op.qubits = [control_qubit, target_qubit1, target_qubit2]
    return op


def mcx(control_qubits, target_qubit):
    """mcx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcx"
    op.qubits = [*control_qubits, target_qubit]
    return op


def mcy(control_qubits, target_qubit):
    """mcy"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcy"
    op.qubits = [*control_qubits, target_qubit]
    return op


def mcz(control_qubits, target_qubit):
    """mcz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcz"
    op.qubits = [*control_qubits, target_qubit]
    return op


def mcu1(control_qubits, target_qubit, lam):
    """mcu1"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu1"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [lam]
    return op


def mcu2(control_qubits, target_qubit, phi, lam):
    """mcu2"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu2"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [phi, lam]
    return op


def mcu3(control_qubits, target_qubit, theta, phi, lam):
    """mcu3"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu3"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta, phi, lam]
    return op


def mcswap(control_qubits, target_qubit1, target_qubit2):
    """mcswap"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcswap"
    op.qubits = [*control_qubits, target_qubit1, target_qubit2]
    return op


def mcphase(control_qubits, target_qubit, phase):
    """mcphase"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcphase"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [phase]
    return op


def mcr(control_qubits, target_qubit, theta, phi):
    """XXXX"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcr"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta, phi]
    return op


def mcrx(control_qubits, target_qubit, theta):
    """mcrx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcrx"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta]
    return op


def mcry(control_qubits, target_qubit, theta):
    """mcry"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcry"
    op.qubits = [*control_qubits, target_qubit]
    op.params = [theta]
    return op


def sx(qubit):
    """sx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "sx"
    op.qubits = [qubit]
    return op


def csx(control_qubit, target_qubit):
    """csx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "csx"
    op.qubits = [control_qubit, target_qubit]
    return op


def mcsx(control_qubits, target_qubit):
    """mcsx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcsx"
    op.qubits = [*control_qubits, target_qubit]
    return op


def delay(qubit):
    """delay"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "delay"
    op.qubits = [qubit]
    return op


def pauli(qubits, pauli_string):
    """pauli"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "pauli"
    op.qubits = qubits
    op.string_params = [pauli_string]
    return op


def mcx_gray(control_qubits, target_qubit):
    """mcx_gray"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcx_gray"
    op.qubits = [*control_qubits, target_qubit]
    op.qubits.push_back(target_qubit)
    return op


# OpType.snapshot:
def snapshot_statevector(qubits, name):
    """snapshot for statevector"""
#     {{"statevector", Snapshots::statevector},
    op = AerOp()
    op.type = OpType.snapshot
    op.name = 'statevector'
    op.qubits = qubits
    op.string_params = [name]
    return op


def snapshot_probabilities(qubits, name):
    """snapshot for probabilities"""
#      {"probabilities", Snapshots::probs},
    op = AerOp()
    op.type = OpType.snapshot
    op.name = 'probabilities'
    op.qubits = qubits
    op.string_params = [name]
    return op


def snapshot_expectation_value_pauli(qubits, coeff_and_paulistring_list):
    """snapshot for expectation_value_pauli"""
#      {"expectation_value_pauli", Snapshots::expval_pauli},
    op = AerOp()
    op.type = OpType.snapshot
    op.name = 'expectation_value_pauli'
    op.qubits = qubits
    op.params_expval_pauli = coeff_and_paulistring_list
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
    """unitary"""
    op = AerOp()
    op.type = OpType.matrix
    op.name = 'unitary'
    op.qubits = qubits
    op.mats = [mat]
    return op


# OpType.diagonal_matrix:
def diagonal(qubits, vec):
    """diagonal"""
    op = AerOp()
    op.type = OpType.diagonal_matrix
    op.name = "diagonal"
    op.qubits = qubits
    op.params = vec
    return op

# OpType.multiplexer:


# OpType.kraus:
def _kraus(qubits, mats):
    """kraus"""
    op = AerOp()
    op.type = OpType.kraus
    op.name = "kraus"
    op.qubits = qubits
    op.mats = mats
    return op


def _superop(qubits, mat):
    """superop"""
    op = AerOp()
    op.type = OpType.superop
    op.name = "superop"
    op.qubits = qubits
    op.mats = [mat]
    return op


def gen_aer_op(instruction, qubits, clbits):
    """Generate an Aer operation from an instruction"""
    params = instruction.params
    op = None
    if instruction.name == 'barrier':
        op = barrier(qubits)
    elif instruction.name == 'reset':
        op = reset(qubits)
    elif instruction.name == 'initialize':
        op = initialize(qubits, params[0])
    elif instruction.name == 'measure':
        op = measure(qubits, clbits)
    elif instruction.name == 'u1':
        op = u1(qubits[0], params[0])
    elif instruction.name == 'u2':
        op = u2(qubits[0], params[0], params[1])
    elif instruction.name == 'u3':
        op = u3(qubits[0], params[0], params[1], params[2])
    elif instruction.name == 'cx':
        op = cx(qubits[0], qubits[1])
    elif instruction.name == 'cz':
        op = cz(qubits[0], qubits[1])
    elif instruction.name == 'cy':
        op = cy(qubits[0], qubits[1])
    elif instruction.name == 'cp':
        op = cp(qubits[0], qubits[1], params[0])
    elif instruction.name == 'cu1':
        op = cu1(qubits[0], qubits[1], params[0])
    elif instruction.name == 'cu2':
        op = cu2(qubits[0], qubits[1], params[0], params[1])
    elif instruction.name == 'cu3':
        op = cu3(qubits[0], qubits[1], params[0], params[1], params[2])
    elif instruction.name == 'swap':
        op = swap(qubits[0], qubits[1])
    elif instruction.name == 'id':
        op = _id(qubits[0])
    elif instruction.name == 'p':
        op = p(qubits[0], params[0])
    elif instruction.name == 'x':
        op = x(qubits[0])
    elif instruction.name == 'y':
        op = y(qubits[0])
    elif instruction.name == 'z':
        op = z(qubits[0])
    elif instruction.name == 'h':
        op = h(qubits[0])
    elif instruction.name == 's':
        op = s(qubits[0])
    elif instruction.name == 'sdg':
        op = sdg(qubits[0])
    elif instruction.name == 't':
        op = t(qubits[0])
    elif instruction.name == 'tdg':
        op = tdg(qubits[0])
    elif instruction.name == 'r':
        op = r(qubits[0], params[0], params[1])
    elif instruction.name == 'rx':
        op = rx(qubits[0], params[0])
    elif instruction.name == 'ry':
        op = ry(qubits[0], params[0])
    elif instruction.name == 'rz':
        op = rz(qubits[0], params[0])
    elif instruction.name == 'rxx':
        op = rxx(qubits[0], qubits[1], params[0])
    elif instruction.name == 'ryy':
        op = ryy(qubits[0], qubits[1], params[0])
    elif instruction.name == 'rzz':
        op = rzz(qubits[0], qubits[1], params[0])
    elif instruction.name == 'rzx':
        op = rzx(qubits[0], qubits[1], params[0])
    elif instruction.name == 'ccx':
        op = ccx(qubits[0], qubits[1], qubits[2])
    elif instruction.name == 'cswap':
        op = cswap(qubits[0], qubits[1], qubits[2])
    elif instruction.name == 'mcx':
        op = mcx(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcy':
        op = mcy(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcz':
        op = mcz(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcu1':
        op = mcu1(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'mcu2':
        op = mcu2(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0], params[1])
    elif instruction.name == 'mcu3':
        op = mcu3(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1],
                  params[0], params[1], params[2])
    elif instruction.name == 'mcswap':
        op = mcswap(qubits[0: len(qubits) - 2], qubits[len(qubits) - 2], qubits[len(qubits) - 2])
    elif instruction.name == 'mcphase':
        op = mcphase(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'mcr':
        op = mcr(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0], params[1])
    elif instruction.name == 'mcx':
        op = mcx(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcry':
        op = mcry(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1], params[0])
    elif instruction.name == 'sx':
        op = sx(qubits[0])
    elif instruction.name == 'csx':
        op = csx(qubits[0], qubits[1])
    elif instruction.name == 'mcsx':
        op = mcsx(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'delay':
        op = delay(qubits[0])
    elif instruction.name == 'pauli':
        op = pauli(qubits, params[0])
    elif instruction.name == 'mcx_gray':
        op = mcx_gray(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'mcx_gray':
        op = mcx_gray(qubits[0: len(qubits) - 1], qubits[len(qubits) - 1])
    elif instruction.name == 'unitary':
        op = unitary(qubits, params[0])
    elif instruction.name == 'diagonal':
        op = diagonal(qubits, to_aer_vec(params[0], complex))
    else:
        raise ValueError(f'not implemented yet: {instruction}')

    return op


def gen_aer_circuit(circuit):
    """convert QuantumCircuit to AerCircuit"""

    global_phase = circuit.global_phase

    qubit_indices = {bit: index for index, bit in enumerate(circuit.qubits)}
    clbit_indices = {bit: index for index, bit in enumerate(circuit.clbits)}

    circ = AerCircuit([gen_aer_op(instruction[0],
                                  [qubit_indices[qubit] for qubit in instruction[1]],
                                  [clbit_indices[clbit] for clbit in instruction[2]])
                       for instruction in circuit.data])
    circ.global_phase_angle = global_phase
    return circ
