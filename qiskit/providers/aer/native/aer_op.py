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
# pylint: disable=import-error, no-name-in-module, invalid-name, unused-argument
import qiskit
from qiskit.providers.aer.backends.controller_wrappers import (AerCircuit, AerOp,
                                                               OpType, make_unitary)


# OpType.barrier
def barrier(inst, qubits, clbits):
    """barrier"""
    op = AerOp()
    op.type = OpType.barrier
    op.name = "barrier"
    op.qubits = qubits
    return op


# OpType.reset:
def reset(inst, qubits, clbits):
    """reset"""
    op = AerOp()
    op.type = OpType.reset
    op.name = "reset"
    op.qubits = qubits
    return op


# OpType.initialize:
def initialize(inst, qubits, clbits):
    """initialize"""
    op = AerOp()
    op.type = OpType.initialize
    op.name = "initialize"
    op.qubits = qubits
    op.params = [complex(d) for d in inst.params[0]]
    return op


# OpType.measure:
def measure(inst, qubits, clbits):
    """measure"""
    op = AerOp()
    op.type = OpType.measure
    op.name = "measure"
    op.qubits = qubits
    op.memory = clbits
    op.registers = clbits
    return op

# OpType.bfunc:

# OpType.roerror:


# OpType.gate:
def u1(inst, qubits, clbits):
    """u1"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u1"
    op.qubits = qubits
    op.params = inst.params
    return op


def u2(inst, qubits, clbits):
    """u2"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u2"
    op.qubits = qubits
    op.params = inst.params
    return op


def u3(inst, qubits, clbits):
    """u3"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u3"
    op.qubits = qubits
    op.params = inst.params
    return op


def cx(inst, qubits, clbits):
    """cx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cx"
    op.qubits = qubits
    return op


def cz(inst, qubits, clbits):
    """cz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cz"
    op.qubits = qubits
    return op


def cy(inst, qubits, clbits):
    """cy"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cy"
    op.qubits = qubits
    return op


def cp(inst, qubits, clbits):
    """cp"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cp"
    op.qubits = qubits
    op.params = inst.params
    return op


def cu1(inst, qubits, clbits):
    """cu1"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu1"
    op.qubits = qubits
    op.params = inst.params
    return op


def cu2(inst, qubits, clbits):
    """cu2"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu2"
    op.qubits = qubits
    op.params = inst.params
    return op


def cu3(inst, qubits, clbits):
    """cu3"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu3"
    op.qubits = qubits
    op.params = inst.params
    return op


def swap(inst, qubits, clbits):
    """swap"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "swap"
    op.qubits = qubits
    return op


def id_(inst, qubits, clbits):
    """_id"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "id"
    op.qubits = qubits
    return op


def p(inst, qubits, clbits):
    """p"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "p"
    op.qubits = qubits
    op.params = inst.params
    return op


def x(inst, qubits, clbits):
    """x"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "x"
    op.qubits = qubits
    return op


def y(inst, qubits, clbits):
    """y"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "y"
    op.qubits = qubits
    return op


def z(inst, qubits, clbits):
    """z"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "z"
    op.qubits = qubits
    return op


def h(inst, qubits, clbits):
    """h"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "h"
    op.qubits = qubits
    return op


def s(inst, qubits, clbits):
    """s"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "s"
    op.qubits = qubits
    return op


def sdg(inst, qubits, clbits):
    """sdg"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "sdg"
    op.qubits = qubits
    return op


def t(inst, qubits, clbits):
    """t"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "s"
    op.qubits = qubits
    return op


def tdg(inst, qubits, clbits):
    """tdg"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "tdg"
    op.qubits = qubits
    return op


def r(inst, qubits, clbits):
    """r"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "r"
    op.qubits = qubits
    op.params = inst.params
    return op


def rx(inst, qubits, clbits):
    """rx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rx"
    op.qubits = qubits
    op.params = inst.params
    return op


def ry(inst, qubits, clbits):
    """ry"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "ry"
    op.qubits = qubits
    op.params = inst.params
    return op


def rz(inst, qubits, clbits):
    """rz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rz"
    op.qubits = qubits
    op.params = inst.params
    return op


def rxx(inst, qubits, clbits):
    """rxx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rxx"
    op.qubits = qubits
    op.params = inst.params
    return op


def ryy(inst, qubits, clbits):
    """ryy"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "ryy"
    op.qubits = qubits
    op.params = inst.params
    return op


def rzz(inst, qubits, clbits):
    """rzz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rzz"
    op.qubits = qubits
    op.params = inst.params
    return op


def rzx(inst, qubits, clbits):
    """rzx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "rzx"
    op.qubits = qubits
    op.params = inst.params
    return op


def ccx(inst, qubits, clbits):
    """ccx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "ccx"
    op.qubits = qubits
    return op


def cswap(inst, qubits, clbits):
    """cswap"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cswap"
    op.qubits = qubits
    return op


def mcx(inst, qubits, clbits):
    """mcx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcx"
    op.qubits = qubits
    return op


def mcy(inst, qubits, clbits):
    """mcy"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcy"
    op.qubits = qubits
    return op


def mcz(inst, qubits, clbits):
    """mcz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcz"
    op.qubits = qubits
    return op


def mcu1(inst, qubits, clbits):
    """mcu1"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu1"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcu2(inst, qubits, clbits):
    """mcu2"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu2"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcu3(inst, qubits, clbits):
    """mcu3"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu3"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcswap(inst, qubits, clbits):
    """mcswap"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcswap"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcp(inst, qubits, clbits):
    """mcp"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcphase"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcr(inst, qubits, clbits):
    """XXXX"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcr"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcrx(inst, qubits, clbits):
    """mcrx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcrx"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcry(inst, qubits, clbits):
    """mcry"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcry"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcrz(inst, qubits, clbits):
    """mcrz"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcrz"
    op.qubits = qubits
    op.params = inst.params
    return op


def sx(inst, qubits, clbits):
    """sx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "sx"
    op.qubits = qubits
    return op


def csx(inst, qubits, clbits):
    """csx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "csx"
    op.qubits = qubits
    op.params = inst.params
    return op


def mcsx(inst, qubits, clbits):
    """mcsx"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcsx"
    op.qubits = qubits
    op.params = inst.params
    return op


def delay(inst, qubits, clbits):
    """delay"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "delay"
    op.qubits = qubits
    op.params = inst.params
    return op


def pauli(inst, qubits, clbits):
    """pauli"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "pauli"
    op.qubits = qubits
    op.string_params = inst.params
    return op


def mcx_gray(inst, qubits, clbits):
    """mcx_gray"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcx_gray"
    op.qubits = qubits
    return op


# OpType.snapshot:
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
def unitary(inst, qubits, clbits):
    """unitary"""
    mat = inst.params[0]
    return make_unitary(qubits, mat, mat.flags.carray)


# OpType.diagonal_matrix:
def diagonal(inst, qubits, clbits):
    """diagonal"""
    op = AerOp()
    op.type = OpType.diagonal_matrix
    op.name = "diagonal"
    op.qubits = qubits
    op.params = inst.params
    return op

# OpType.multiplexer:


# OpType.kraus:

_gen_op_funcs = {
    qiskit.extensions.unitary.UnitaryGate: unitary,
    qiskit.circuit.measure.Measure: measure,
    qiskit.circuit.barrier.Barrier: barrier,
    qiskit.circuit.library.standard_gates.u1.U1Gate: u1,
    qiskit.circuit.library.standard_gates.u2.U2Gate: u2,
    qiskit.circuit.library.standard_gates.u3.U3Gate: u3,
    qiskit.circuit.library.standard_gates.u.UGate: u3,
    qiskit.circuit.library.standard_gates.x.CXGate: cx,
    qiskit.circuit.library.standard_gates.y.CYGate: cy,
    qiskit.circuit.library.standard_gates.z.CZGate: cz,
    qiskit.circuit.library.standard_gates.p.CPhaseGate: cp,
    qiskit.circuit.library.standard_gates.u1.CU1Gate: cu1,
    qiskit.circuit.library.standard_gates.u3.CU3Gate: cu3,
    qiskit.circuit.library.standard_gates.u1.MCU1Gate: mcu1,
    qiskit.circuit.library.standard_gates.swap.SwapGate: swap,
    qiskit.circuit.library.standard_gates.i.IGate: id_,
    qiskit.circuit.library.standard_gates.p.PhaseGate: p,
    qiskit.circuit.library.standard_gates.x.XGate: x,
    qiskit.circuit.library.standard_gates.y.YGate: y,
    qiskit.circuit.library.standard_gates.z.ZGate: z,
    qiskit.circuit.library.standard_gates.h.HGate: h,
    qiskit.circuit.library.standard_gates.s.SGate: s,
    qiskit.circuit.library.standard_gates.s.SdgGate: sdg,
    qiskit.circuit.library.standard_gates.t.TGate: t,
    qiskit.circuit.library.standard_gates.t.TdgGate: tdg,
    qiskit.circuit.library.standard_gates.r.RGate: r,
    qiskit.circuit.library.standard_gates.rx.RXGate: rx,
    qiskit.circuit.library.standard_gates.ry.RYGate: ry,
    qiskit.circuit.library.standard_gates.rz.RZGate: rz,
    qiskit.circuit.library.standard_gates.rxx.RXXGate: rxx,
    qiskit.circuit.library.standard_gates.ryy.RYYGate: ryy,
    qiskit.circuit.library.standard_gates.rzz.RZZGate: rzz,
    qiskit.circuit.library.standard_gates.x.CCXGate: ccx,
    qiskit.circuit.library.standard_gates.swap.CSwapGate: cswap,
    qiskit.circuit.library.standard_gates.x.C3XGate: mcx,
    qiskit.circuit.library.standard_gates.x.C4XGate: mcx,
    qiskit.circuit.library.standard_gates.p.MCPhaseGate: mcp,
    qiskit.circuit.library.standard_gates.rx.CRXGate: mcrx,
    qiskit.circuit.library.standard_gates.ry.CRYGate: mcry,
    qiskit.circuit.library.standard_gates.rz.CRZGate: mcrz,
    qiskit.circuit.library.standard_gates.sx.SXGate: sx,
    qiskit.circuit.library.standard_gates.sx.CSXGate: csx,
    qiskit.circuit.delay.Delay: delay,
    qiskit.circuit.library.generalized_gates.pauli.PauliGate: pauli,
    qiskit.circuit.library.generalized_gates.diagonal.Diagonal: diagonal,
    qiskit.circuit.library.standard_gates.x.MCXGrayCode: mcx_gray,
}


def gen_aer_op(inst, qubits, clbits):
    """Generate an Aer operation from an inst"""
    if inst.__class__ not in _gen_op_funcs:
        raise ValueError(f'unsupported instruction: {inst.__class__}')

    return _gen_op_funcs[inst.__class__](inst, qubits, clbits)


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
