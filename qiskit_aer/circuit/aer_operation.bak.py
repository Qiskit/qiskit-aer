# This code is part of Qiskit.
#
# (C) Copyright IBM 2017-2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from abc import ABC
from typing import List, Dict, Optional, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit_aer.backends.controller_wrappers import AerCircuit_, AerOp_, OpType_, DataSubType_
from qiskit_aer.aererror import AerError

"""Directly simulatable circuit operation in Aer."""
class AerOp(ABC):
    """A class of an internal ciruict operation of Aer
    """
    def __init__(
        self,
        optype,
        name,
        qubits: Optional[List[int]] = None,
        params: Optional[List[complex]] = None,
        string_params: Optional[List[str]] = None,
        relation: Optional[str] = None,
        registers: Optional[List[int]] = None,
        memory: Optional[List[int]] = None,
        mats: Optional[List[np.ndarray]] = None,
        save_type: Optional[DataSubType_] = None,
        expval_params: Optional[List[Tuple[str, float, float]]] = None,
        conditional_reg: int = -1
    ):
        self._optype = optype
        self._name = name
        if qubits:
            self._qubits = qubits
        self._params = params
        self._string_params = string_params
        self._relation = relation
        self._registers = registers
        self._memory = memory
        self._mats = mats
        self._save_type = save_type
        self._conditional = conditional_reg >= 0
        self._expval_params = expval_params
        self._conditional_reg = conditional_reg
    
    def set_conditional(
        self,
        conditional_reg: int
    )-> None:
        self._conditional = conditional_reg >= 0
        self._conditional_reg = conditional_reg

    def assemble_native(
        self
    )-> AerOp_:
        op = AerOp_()
        op.type = self._optype
        op.name = self._name
        if self._qubits:
            op.qubits = self._qubits
        if self._params:
            op.params = self._params
        if self._string_params:
            op.string_params = self._string_params
        if self._relation:
            op.relation = self._relation
        if self._registers:
            op.registers = self._registers
        if self._memory:
            op.memory = self._memory
        if self._mats:
            op.mats = self._mats
        if self._save_type:
            op.save_type = self._save_type
        op.conditional = self._conditional
        if self._conditional:
            op.conditional_reg = self._conditional_reg
        if self._expval_params:
            op.expval_params = self._expval_params
        return op

class BinaryFuncOp(AerOp):
    """bfunc operation"""
    def __init__(
        self,
        mask: str,
        relation: str,
        val: str,
        register: int,
        memory: int = -1,
    ):
        super().__init__(
            OpType_.bfunc,
            "bfunc",
            string_params=[mask, val],
            relation=relation,
            reigsters=[register],
            memory=[memory]
            )

class BarrierOp(AerOp):
    """barrier operation"""
    def __init__(
        self,
        qubits: List[int],
        conditional_reg: int=-1
    ):
        super().__init__(
            OpType_.barrier,
            "barrier",
            qubits=qubits,
            conditional_reg=conditional_reg
        )

class MeasureOp(AerOp):
    """measure operation"""
    def __init__(
        self,
        qubits: List[int],
        memory: List[int],
        registers: List[int]
    ):
        super().__init__(
            OpType_.measure,
            "measure",
            qubits=qubits,
            memory=memory,
            registers=registers
        )

class ResetOp(AerOp):
    """reset operation"""
    def __init__(
        self,
        qubits: List[int]
    ):
        super().__init__(
            OpType_.reset,
            "reset",
            qubits=qubits
        )

class InitializeOp(AerOp):
    """initialize operation"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.initialize,
            "initialize",
            qubits=qubits,
            params=init_data
        )

class UnitaryOp(AerOp):
    """unitary operation"""
    def __init__(
        self,
        qubits: List[int],
        mat: np.ndarray,
        label: str = None,
        conditional_reg: int=-1
    ):
        super().__init__(
            OpType_.matrix,
            "unitary",
            qubits=qubits,
            mats=[mat],
            string_params=[label] if label != None else None,
            conditional_reg=conditional_reg
        )

class DiagonalOp(AerOp):
    """diagonal operation"""
    def __init__(
        self,
        qubits: List[int],
        vmat: np.ndarray,
        label: str = None,
        conditional_reg: int=-1
    ):
        super().__init__(
            OpType_.diagonal_matrix,
            "diagonal",
            qubits=qubits,
            params=vmat,
            string_params=[label] if label != None else None,
            conditional_reg=conditional_reg
        )

class SuperOp(AerOp):
    """super operator operation"""
    def __init__(
        self,
        qubits: List[int],
        params: np.ndarray,
        label: str = None,
        conditional_reg: int=-1
    ):
        super().__init__(
            OpType_.superop,
            "superop",
            qubits=qubits,
            params=params,
            string_params=[label] if label != None else None,
            conditional_reg=conditional_reg
        )

_SAVE_SUB_TYPES = {
    "single": DataSubType_.single,
    "c_single": DataSubType_.c_single,
    "average": DataSubType_.average,
    "c_average": DataSubType_.c_average,
    "list": DataSubType_.list,
    "c_list": DataSubType_.c_list,
    "accum": DataSubType_.accum,
    "c_accum": DataSubType_.c_accum,    
}

class SaveStateOp(AerOp):
    """super operator operation"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        expval_params: Optional[List[Tuple[str, float, float]]] = None,
        label: str = None
    ):
        super().__init__(
            OpType_.save_state,
            "save_state",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveExpValOp(AerOp):
    """save expectation value"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        expval_params: List[Tuple[str, float, float]],
        label: str = None
    ):
        if len(expval_params) == 0:
            expval_params.append(('I' * len(qubits), 0.0, 0.0))
        super().__init__(
            OpType_.save_expval,
            "save_expval",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            expval_params=expval_params,
            string_params=[label] if label != None else None
        )

class SaveExpValVarOp(AerOp):
    """save expectation value with variant"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        expval_params: List[Tuple[str, float, float]],
        label: str = None
    ):
        if len(expval_params) == 0:
            expval_params.append(('I' * len(qubits), 0.0, 0.0))
        super().__init__(
            OpType_.save_expval_var,
            "save_expval_var",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            expval_params=expval_params,
            string_params=[label] if label != None else None
        )

class SaveStatevectorOp(AerOp):
    """save statevector state"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_statevec,
            "save_statevector",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveStatevectorDictOp(AerOp):
    """save statevector state"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_statevec_dict,
            "save_statevector_dict",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveStabilizerOp(AerOp):
    """save stabilizer state"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_stabilizer,
            "save_stabilizer",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveCliffordOp(AerOp):
    """save clifford state"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_clifford,
            "save_clifford",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveUnitaryOp(AerOp):
    """save unitary matrix"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_unitary,
            "save_unitary",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveSuperopOp(AerOp):
    """save super operator state"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_superop,
            "save_superop",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveDensityMatrixOp(AerOp):
    """save density matrix"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_density_matrix,
            "save_density_matrix",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveProbabirlitiesOp(AerOp):
    """save proberbilities"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_probs,
            "save_probabilities",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveProbabirlitiesOp(AerOp):
    """save proberbilities"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_probs,
            "save_probabilities",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveProbabirlitiesKetOp(AerOp):
    """save proberbilities dict"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_probs_ket,
            "save_probs_ket",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveMatrixProductStateOp(AerOp):
    """save mps state"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_mps,
            "save_mps",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveAmbilitudesOp(AerOp):
    """save ambilitudes"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_amplitudes,
            "save_amplitudes",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SaveAmbilitudesSqOp(AerOp):
    """squared statevector amplitudes"""
    def __init__(
        self,
        qubits: List[int],
        subtype: str,
        label: str = None
    ):
        super().__init__(
            OpType_.save_amplitudes_sq,
            "save_amplitudes_sq",
            qubits=qubits,
            save_type=_SAVE_SUB_TYPES[subtype],
            string_params=[label] if label != None else None
        )

class SetStatevectorOp(AerOp):
    """set statevector"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.set_statevec,
            "set_statevecor",
            qubits=qubits,
            params=init_data
        )

class SetDensityMatrixOp(AerOp):
    """set densitymatrix"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.set_densmat,
            "set_densmat",
            qubits=qubits,
            params=init_data
        )

class SetUnitaryOp(AerOp):
    """set unitary matrix"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.set_unitary,
            "set_unitary",
            qubits=qubits,
            params=init_data
        )

class SetSuperopOp(AerOp):
    """set super operator state"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.set_superop,
            "set_superop",
            qubits=qubits,
            params=init_data
        )

class SetStabilizerOp(AerOp):
    """set stabilizer state"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.set_stabilizer,
            "set_stabilizer",
            qubits=qubits,
            params=init_data
        )

class SetMatrixProductStateOp(AerOp):
    """set mps state"""
    def __init__(
        self,
        qubits: List[int],
        init_data: List[complex]
    ):
        super().__init__(
            OpType_.set_matrix_product_state,
            "set_matrix_product_state",
            qubits=qubits,
            params=init_data
        )

class ErrorLocationOp(AerOp):
    """set error location"""
    def __init__(
        self,
        qubits: List[int],
        label: str,
        conditional_reg: int=-1
    ):
        super().__init__(
            OpType_.qerror_loc,
            label,
            qubits=qubits,
            conditional_reg=conditional_reg
        )

class MultiplexerOp(AerOp):
    """multiplexer op"""
    def __init__(
        self,
        qubits: List[int],
        label: str,
        conditional_reg: int=-1
    ):
        super().__init__(
            OpType_.multiplexer,
            "multiplexer",
            qubits=qubits,
            conditional_reg=conditional_reg
        )

def generate_aer_operation(
    inst: CircuitInstruction,
    qubit_indices: Dict[int, int],
    clbit_indices: Dict[int, int],
    register_clbits: bool
) -> AerOp:
    qubits = []
    memory = []
    registers = []

    if inst.qubits:
        qubits = [qubit_indices[qubit] for qubit in inst.qubits]
    if inst.clbits:
        memory = [clbit_indices[qubit] for qubit in inst.clbits]
        if inst.operation.name == "measure" and register_clbits:
            registers = [clbit_indices[clbit] for clbit in cargs]

    params=[]
    if hasattr(inst.operation, "params"):
        params = [x.evalf(x) if hasattr(x, "evalf") else x for x in inst.operation.params]
    
    if inst.operation.name == 'barrier':
        return BarrierOp(qubits)
    elif inst.operation.name == 'measure':
        return MeasureOp(qubits, memory, registers)
    elif inst.operation.name == 'reset':
        return ResetOp(qubits, qubits)
    elif inst.operation.name == 'initialize':
        return InitializeOp(qubits, params)
    elif inst.operation.name == 'unitary':
        return UnitaryOp(qubits, params[0], inst.operation.label)
    elif inst.operation.name in ('diagonal', 'diag'):
        return DiagonalOp(qubits, params, inst.operation.label)
    elif inst.operation.name == 'superop':
        return SuperOp(qubits, params, inst.operation.label)
    elif inst.operation.name == 'save_state':
        return SaveStateOp(qubits, inst.operation._subtype, inst.operation.label)
    elif inst.operation.name == 'save_expval':
        return SaveExpValOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_expval_var':
        return SaveExpValVarOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_statevector':
        return SaveStatevectorOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_statevector_dict':
        return SaveStatevectorDictOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_stabilizer':
        return SaveStabilizerOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_clifford':
        return SaveCliffordOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_unitary':
        return SaveUnitaryOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_superop':
        return SaveSuperopOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_density_matrix':
        return SaveDensityMatrixOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_probabilities':
        return SaveProbabirlitiesOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_matrix_product_state':
        return SaveMatrixProductStateOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_probabilities_dict':
        return SaveProbabirlitiesKetOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_amplitudes':
        return SaveAmplitudesOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'save_amplitudes_sq':
        return SaveAmplitudesSqOp(qubits, inst.operation._subtype, inst.params, inst.operation.label)
    elif inst.operation.name == 'set_statevector':
        return SetStatevectorOp(qubits, inst.params)
    elif inst.operation.name == 'set_density_matrix':
        return SetDensityMatrixOp(qubits, inst.params)
    elif inst.operation.name == 'set_unitary':
        return SetUnitaryOp(qubits, inst.params)
    elif inst.operation.name == 'set_superop':
        return SetSuperopOp(qubits, inst.params)
    elif inst.operation.name == 'set_stabilizer':
        return SetStabilizerOp(qubits, inst.params)
    elif inst.operation.name == 'set_matrix_product_state':
        return SetMatrixProductStateOp(qubits, inst.params)
    elif inst.operation.name == 'qerror_loc':
        return ErrorLocationOp(qubits, inst.params)
    elif inst.operation.name == 'multiplexer':
        return MultiplexerOp(qubits, inst.params)
    else:
        raise AerError("unknown instruction: " + inst.operation.name)

#   // Set

#   if (name == "multiplexer")
#     return input_to_op_multiplexer(input);
#   if (name == "kraus")
#     return input_to_op_kraus(input);
#   if (name == "roerror")
#     return input_to_op_roerror(input);
#   if (name == "pauli")
#     return input_to_op_pauli(input);

#   //Control-flow
#   if (name == "jump")
#     return input_to_op_jump(input);
#   if (name == "mark")
#     return input_to_op_mark(input);
#   // Default assume gate
#   return input_to_op_gate(input);