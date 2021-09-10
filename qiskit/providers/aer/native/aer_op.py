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
from qiskit.providers.aer.library.save_instructions import (SaveState,
                                                            SaveStatevector,
                                                            SaveStatevectorDict,
                                                            SaveExpectationValue,
                                                            SaveExpectationValueVariance,
                                                            SaveProbabilities,
                                                            SaveProbabilitiesDict,
                                                            SaveUnitary,
                                                            SaveDensityMatrix,
                                                            SaveAmplitudes,
                                                            SaveAmplitudesSquared,
                                                            SaveStabilizer,
                                                            SaveMatrixProductState,
                                                            SaveSuperOp)
from qiskit.providers.aer.library.set_instructions import (SetStatevector,
                                                           SetDensityMatrix,
                                                           SetUnitary,
                                                           SetStabilizer,
                                                           SetSuperOp,
                                                           SetMatrixProductState)
from qiskit.providers.aer.backends.controller_wrappers import (AerCircuit, AerOp,
                                                               OpType, DataSubType,
                                                               make_unitary,
                                                               make_multiplexer,
                                                               make_set_clifford)


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
    op.params = [complex(d) for d in inst.params]
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


def append_measure(circuit, inst, qubits, clbits):
    """measure"""
    memory = clbits
    return circuit.measure(qubits, memory, clbits)

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


def u(inst, qubits, clbits):
    """u"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "u"
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


def cu(inst, qubits, clbits):
    """cu"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "cu"
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
    op.name = "t"
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


def mcu(inst, qubits, clbits):
    """mcu"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "mcu"
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


def sxdg(inst, qubits, clbits):
    """sxdg"""
    op = AerOp()
    op.type = OpType.gate
    op.name = "sxdg"
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


_data_subtype = {
    "single": DataSubType.single,
    "c_single": DataSubType.c_single,
    "average": DataSubType.average,
    "c_average": DataSubType.c_average,
    "list": DataSubType.list,
    "c_list": DataSubType.c_list,
    "accum": DataSubType.accum,
    "c_accum": DataSubType.c_accum,
}


def _save_operation(op_type, inst, qubits, clbits):
    """save operation base"""
    op = AerOp()
    op.name = inst.name
    op.type = op_type
    op.save_type = _data_subtype[inst._subtype]
    op.qubits = qubits
    op.string_params = [inst.label]
    return op


def save_state(inst, qubits, clbits):
    """save state"""
    return _save_operation(OpType.save_state, inst, qubits, clbits)


def save_statevector(inst, qubits, clbits):
    """save satevector"""
    return _save_operation(OpType.save_statevec, inst, qubits, clbits)


def save_statevector_dict(inst, qubits, clbits):
    """save satevector dict"""
    return _save_operation(OpType.save_statevec_dict, inst, qubits, clbits)


def _save_expval(save_expval_type, inst, qubits, clbits):
    """save expectation value base"""
    op = _save_operation(save_expval_type, inst, qubits, clbits)
    expval_params = [(paulistr, coeff[0], coeff[1])
                     for paulistr, coeff in inst.params]
    if not expval_params:
        expval_params = [(['I'] * len(qubits), 0.0, 0.0)]
    op.expval_params = expval_params
    return op


def save_expval(inst, qubits, clbits):
    """save expectation value"""
    return _save_expval(OpType.save_expval, inst, qubits, clbits)


def save_expval_var(inst, qubits, clbits):
    """save expectation value variance"""
    return _save_expval(OpType.save_expval_var, inst, qubits, clbits)


def save_probabilities(inst, qubits, clbits):
    """save probabilities"""
    return _save_operation(OpType.save_probs, inst, qubits, clbits)


def save_probabilities_dict(inst, qubits, clbits):
    """save probabilities dict"""
    return _save_operation(OpType.save_probs_ket, inst, qubits, clbits)


def save_amplitudes(inst, qubits, clbits):
    """save amplitudes"""
    op = _save_operation(OpType.save_amps, inst, qubits, clbits)
    op.int_params = inst.params
    return op


def save_amplitudes_squared(inst, qubits, clbits):
    """save amplitudes squared"""
    op = _save_operation(OpType.save_amps_sq, inst, qubits, clbits)
    op.int_params = inst.params
    return op


def save_density_matrix(inst, qubits, clbits):
    """save density matrix"""
    return _save_operation(OpType.save_densmat, inst, qubits, clbits)


def save_unitary(inst, qubits, clbits):
    """save unitary"""
    return _save_operation(OpType.save_unitary, inst, qubits, clbits)


def save_stabilizer(inst, qubits, clbits):
    """save stabilizer state"""
    return _save_operation(OpType.save_stabilizer, inst, qubits, clbits)


def save_mps(inst, qubits, clbits):
    """save mps state"""
    return _save_operation(OpType.save_mps, inst, qubits, clbits)


def save_superop(inst, qubits, clbits):
    """save superop state"""
    return _save_operation(OpType.save_superop, inst, qubits, clbits)


def _set_state(_type, inst, qubits, clbits):
    """set state base"""
    op = AerOp()
    op.name = inst.name
    op.type = _type
    op.qubits = qubits
    return op


def set_statevector(inst, qubits, clbits):
    """set statevector"""
    op = _set_state(OpType.set_statevec, inst, qubits, clbits)
    op.params = inst.params[0]
    return op


def set_density_matrix(inst, qubits, clbits):
    """set density matrix"""
    op = _set_state(OpType.set_densmat, inst, qubits, clbits)
    op.mats = inst.params
    return op


def set_unitary(inst, qubits, clbits):
    """set unitary"""
    op = _set_state(OpType.set_unitary, inst, qubits, clbits)
    op.mats = inst.params
    return op


def set_stabilizer(inst, qubits, clbits):
    """set stabilizer"""
    return make_set_clifford(qubits, inst.params[0]['stabilizer'], inst.params[0]['destabilizer'])


def set_superop(inst, qubits, clbits):
    """set superop"""
    op = _set_state(OpType.set_superop, inst, qubits, clbits)
    op.mats = inst.params
    return op


def set_mps(inst, qubits, clbits):
    """set mps"""
    op = _set_state(OpType.set_mps, inst, qubits, clbits)
    op.mps = inst.params[0]
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
    op = make_unitary(qubits, mat, mat.flags.carray)
    if inst.label:
        op.string_params = [inst.label]
    return op


def append_unitary(circuit, inst, qubits, clbits):
    """append_ unitary"""
    mat = inst.params[0]
    circuit.unitary(qubits, mat, mat.flags.carray)


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
def multiplexer(inst, qubits, clbits):
    """multiplexer"""
    label = inst.label if inst.label else ""
    return make_multiplexer(qubits, inst.params, label)


# OpType.kraus:
def kraus(inst, qubits, clbits):
    """kraus"""
    op = AerOp()
    op.type = OpType.kraus
    op.name = "kraus"
    op.qubits = qubits
    op.mats = inst.params
    return op


# OpType.superop:
def superop(inst, qubits, clbits):
    """superop"""
    op = AerOp()
    op.type = OpType.superop
    op.name = "superop"
    op.qubits = qubits
    op.mats = inst.params
    return op


_gen_op_funcs_by_name = {
    'superop': superop,
    'kraus': kraus,
    'measure': measure,
    'multiplexer': multiplexer,
}


def general_instruction(inst, qubits, clbits):
    """Generate an Aer operation from a name of inst"""
    if inst.name not in _gen_op_funcs_by_name:
        raise ValueError(f'unsupported instruction: {inst.name}')

    return _gen_op_funcs_by_name[inst.name](inst, qubits, clbits)


_gen_op_funcs = {
    qiskit.extensions.unitary.UnitaryGate: unitary,
    qiskit.circuit.reset.Reset: reset,
    qiskit.circuit.measure.Measure: measure,
    qiskit.circuit.barrier.Barrier: barrier,
    qiskit.circuit.library.standard_gates.u1.U1Gate: u1,
    qiskit.circuit.library.standard_gates.u2.U2Gate: u2,
    qiskit.circuit.library.standard_gates.u3.U3Gate: u3,
    qiskit.circuit.library.standard_gates.u.UGate: u,
    qiskit.circuit.library.standard_gates.x.CXGate: cx,
    qiskit.circuit.library.standard_gates.y.CYGate: cy,
    qiskit.circuit.library.standard_gates.z.CZGate: cz,
    qiskit.circuit.library.standard_gates.p.CPhaseGate: cp,
    qiskit.circuit.library.standard_gates.u1.CU1Gate: cu1,
    qiskit.circuit.library.standard_gates.u3.CU3Gate: cu3,
    qiskit.circuit.library.standard_gates.u.CUGate: cu,
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
    qiskit.circuit.library.standard_gates.rzx.RZXGate: rzx,
    qiskit.circuit.library.standard_gates.x.CCXGate: ccx,
    qiskit.circuit.library.standard_gates.swap.CSwapGate: cswap,
    qiskit.circuit.library.standard_gates.x.C3XGate: mcx,
    qiskit.circuit.library.standard_gates.x.C4XGate: mcx,
    qiskit.circuit.library.standard_gates.x.MCXGate: mcx,
    qiskit.circuit.library.standard_gates.p.MCPhaseGate: mcp,
    qiskit.circuit.library.standard_gates.rx.CRXGate: mcrx,
    qiskit.circuit.library.standard_gates.ry.CRYGate: mcry,
    qiskit.circuit.library.standard_gates.rz.CRZGate: mcrz,
    qiskit.circuit.library.standard_gates.sx.SXGate: sx,
    qiskit.circuit.library.standard_gates.sx.SXdgGate: sxdg,
    qiskit.circuit.library.standard_gates.sx.CSXGate: csx,
    qiskit.circuit.delay.Delay: delay,
    qiskit.circuit.library.generalized_gates.pauli.PauliGate: pauli,
    qiskit.circuit.library.generalized_gates.diagonal.Diagonal: diagonal,
    qiskit.circuit.library.standard_gates.x.MCXGrayCode: mcx_gray,
    SaveState: save_state,
    SaveStatevector: save_statevector,
    SaveStatevectorDict: save_statevector_dict,
    SaveExpectationValue: save_expval,
    SaveExpectationValueVariance: save_expval_var,
    SaveProbabilities: save_probabilities,
    SaveProbabilitiesDict: save_probabilities_dict,
    SaveAmplitudes: save_amplitudes,
    SaveAmplitudesSquared: save_amplitudes_squared,
    SaveDensityMatrix: save_density_matrix,
    SaveUnitary: save_unitary,
    SaveStabilizer: save_stabilizer,
    SaveMatrixProductState: save_mps,
    SaveSuperOp: save_superop,
    SetStatevector: set_statevector,
    SetDensityMatrix: set_density_matrix,
    SetUnitary: set_unitary,
    SetStabilizer: set_stabilizer,
    SetSuperOp: set_superop,
    SetMatrixProductState: set_mps,
    qiskit.circuit.instruction.Instruction: general_instruction,
    qiskit.extensions.quantum_initializer.diagonal.DiagonalGate: diagonal,
    qiskit.extensions.quantum_initializer.initializer.Initialize: initialize,
}

_append_funcs = {
    qiskit.extensions.unitary.UnitaryGate: append_unitary,
    # qiskit.circuit.reset.Reset: append_reset,
    qiskit.circuit.measure.Measure: append_measure,
    # qiskit.circuit.barrier.Barrier: append_barrier,
    # qiskit.circuit.library.standard_gates.u1.U1Gate: append_u1,
    # qiskit.circuit.library.standard_gates.u2.U2Gate: append_u2,
    # qiskit.circuit.library.standard_gates.u3.U3Gate: append_u3,
    # qiskit.circuit.library.standard_gates.u.UGate: append_u,
    # qiskit.circuit.library.standard_gates.x.CXGate: append_cx,
    # qiskit.circuit.library.standard_gates.y.CYGate: append_cy,
    # qiskit.circuit.library.standard_gates.z.CZGate: append_cz,
    # qiskit.circuit.library.standard_gates.p.CPhaseGate: append_cp,
    # qiskit.circuit.library.standard_gates.u1.CU1Gate: append_cu1,
    # qiskit.circuit.library.standard_gates.u3.CU3Gate: append_cu3,
    # qiskit.circuit.library.standard_gates.u.CUGate: append_cu,
    # qiskit.circuit.library.standard_gates.u1.MCU1Gate: append_mcu1,
    # qiskit.circuit.library.standard_gates.swap.SwapGate: append_swap,
    # qiskit.circuit.library.standard_gates.i.IGate: append_id_,
    # qiskit.circuit.library.standard_gates.p.PhaseGate: append_p,
    # qiskit.circuit.library.standard_gates.x.XGate: append_x,
    # qiskit.circuit.library.standard_gates.y.YGate: append_y,
    # qiskit.circuit.library.standard_gates.z.ZGate: append_z,
    # qiskit.circuit.library.standard_gates.h.HGate: append_h,
    # qiskit.circuit.library.standard_gates.s.SGate: append_s,
    # qiskit.circuit.library.standard_gates.s.SdgGate: append_sdg,
    # qiskit.circuit.library.standard_gates.t.TGate: append_t,
    # qiskit.circuit.library.standard_gates.t.TdgGate: append_tdg,
    # qiskit.circuit.library.standard_gates.r.RGate: append_r,
    # qiskit.circuit.library.standard_gates.rx.RXGate: append_rx,
    # qiskit.circuit.library.standard_gates.ry.RYGate: append_ry,
    # qiskit.circuit.library.standard_gates.rz.RZGate: append_rz,
    # qiskit.circuit.library.standard_gates.rxx.RXXGate: append_rxx,
    # qiskit.circuit.library.standard_gates.ryy.RYYGate: append_ryy,
    # qiskit.circuit.library.standard_gates.rzz.RZZGate: append_rzz,
    # qiskit.circuit.library.standard_gates.rzx.RZXGate: append_rzx,
    # qiskit.circuit.library.standard_gates.x.CCXGate: append_ccx,
    # qiskit.circuit.library.standard_gates.swap.CSwapGate: append_cswap,
    # qiskit.circuit.library.standard_gates.x.C3XGate: append_mcx,
    # qiskit.circuit.library.standard_gates.x.C4XGate: append_mcx,
    # qiskit.circuit.library.standard_gates.x.MCXGate: append_mcx,
    # qiskit.circuit.library.standard_gates.p.MCPhaseGate: append_mcp,
    # qiskit.circuit.library.standard_gates.rx.CRXGate: append_mcrx,
    # qiskit.circuit.library.standard_gates.ry.CRYGate: append_mcry,
    # qiskit.circuit.library.standard_gates.rz.CRZGate: append_mcrz,
    # qiskit.circuit.library.standard_gates.sx.SXGate: append_sx,
    # qiskit.circuit.library.standard_gates.sx.SXdgGate: append_sxdg,
    # qiskit.circuit.library.standard_gates.sx.CSXGate: append_csx,
    # qiskit.circuit.delay.Delay: append_delay,
    # qiskit.circuit.library.generalized_gates.pauli.PauliGate: append_pauli,
    # qiskit.circuit.library.generalized_gates.diagonal.Diagonal: append_diagonal,
    # qiskit.circuit.library.standard_gates.x.MCXGrayCode: append_mcx_gray,
    # SaveState: append_save_state,
    # SaveStatevector: append_save_statevector,
    # SaveStatevectorDict: append_save_statevector_dict,
    # SaveExpectationValue: append_save_expval,
    # SaveExpectationValueVariance: append_save_expval_var,
    # SaveProbabilities: append_save_probabilities,
    # SaveProbabilitiesDict: append_save_probabilities_dict,
    # SaveAmplitudes: append_save_amplitudes,
    # SaveAmplitudesSquared: append_save_amplitudes_squared,
    # SaveDensityMatrix: append_save_density_matrix,
    # SaveUnitary: append_save_unitary,
    # SaveStabilizer: append_save_stabilizer,
    # SaveMatrixProductState: append_save_mps,
    # SaveSuperOp: append_save_superop,
    # SetStatevector: append_set_statevector,
    # SetDensityMatrix: append_set_density_matrix,
    # SetUnitary: append_set_unitary,
    # SetStabilizer: append_set_stabilizer,
    # SetSuperOp: append_set_superop,
    # SetMatrixProductState: append_set_mps,
    # qiskit.circuit.instruction.Instruction: append_general_instruction,
    # qiskit.extensions.quantum_initializer.diagonal.DiagonalGate: append_diagonal,
    # qiskit.extensions.quantum_initializer.initializer.Initialize: append_initialize,
}


def gen_aer_op(inst, qubits, clbits):
    """Generate an Aer operation from an inst"""
    if inst.__class__ not in _gen_op_funcs:
        return general_instruction(inst, qubits, clbits)

    return _gen_op_funcs[inst.__class__](inst, qubits, clbits)


def gen_aer_circuit(circuit, seed=None, shots=None, num_memory=0, enable_truncation=False):
    """convert QuantumCircuit to AerCircuit"""

    if isinstance(circuit, AerCircuit):
        return circuit

    global_phase = circuit.global_phase

    qubit_indices = {bit: index for index, bit in enumerate(circuit.qubits)}
    clbit_indices = {bit: index for index, bit in enumerate(circuit.clbits)}

#     circ = AerCircuit(circuit.name,
#                       [gen_aer_op(instruction[0],
#                                   [qubit_indices[qubit] for qubit in instruction[1]],
#                                   [clbit_indices[clbit] for clbit in instruction[2]])
#                        for instruction in circuit.data],
#                       enable_truncation)

    circ = AerCircuit(circuit.name)
    for instruction in circuit.data:
        qubits = [qubit_indices[qubit] for qubit in instruction[1]]
        clbits = [clbit_indices[clbit] for clbit in instruction[2]]
        if instruction[0].__class__ in _append_funcs:
            _append_funcs[instruction[0].__class__](circ, instruction[0], qubits, clbits)
        else:
            op = gen_aer_op(instruction[0], qubits, clbits)
            circ.append_op(op)
    circ.global_phase_angle = global_phase
    circ.num_memory = num_memory
    if seed:
        circ.seed = seed
    if shots:
        circ.shots = shots
    circ.initialize(enable_truncation)
    if not enable_truncation:
        circ.num_qubits = len(qubit_indices)
    return circ
