# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
"""
Qiskit Aer simulator name mapping for Target object
"""
from qiskit.circuit import ControlledGate, Parameter
from qiskit.circuit.reset import Reset
from qiskit.circuit.library import (
    SXGate,
    MCPhaseGate,
    MCXGate,
    RZGate,
    RXGate,
    U2Gate,
    U1Gate,
    U3Gate,
    YGate,
    ZGate,
    PauliGate,
    SwapGate,
    RGate,
    MCXGrayCode,
    RYGate,
)
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    ContinueLoopOp,
    BreakLoopOp,
    SwitchCaseOp,
)
from qiskit.extensions import Initialize, UnitaryGate
from qiskit.extensions.quantum_initializer import DiagonalGate, UCGate
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

from ..library import (
    SaveExpectationValue,
    SaveAmplitudes,
    SaveStatevectorDict,
    SaveSuperOp,
    SaveClifford,
    SaveMatrixProductState,
    SaveDensityMatrix,
    SaveProbabilities,
    SaveStatevector,
    SetDensityMatrix,
    SetUnitary,
    SaveState,
    SetMatrixProductState,
    SaveUnitary,
    SetSuperOp,
    SaveExpectationValueVariance,
    SaveStabilizer,
    SetStatevector,
    SetStabilizer,
    SaveAmplitudesSquared,
    SaveProbabilitiesDict,
)
from ..noise.errors import ReadoutError
from ..noise.noise_model import QuantumErrorLocation


class MCSXGate(ControlledGate):
    """mcsx gate"""

    def __init__(self, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcsx",
            1 + num_ctrl_qubits,
            [],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=SXGate(),
        )


class MCYGate(ControlledGate):
    """mcy gate"""

    def __init__(self, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcy",
            1 + num_ctrl_qubits,
            [],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=YGate(),
        )


class MCZGate(ControlledGate):
    """mcz gate"""

    def __init__(self, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcz",
            1 + num_ctrl_qubits,
            [],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=ZGate(),
        )


class MCRXGate(ControlledGate):
    """mcrx gate"""

    def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcrx",
            1 + num_ctrl_qubits,
            [theta],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=RXGate(theta),
        )


class MCRYGate(ControlledGate):
    """mcry gate"""

    def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcry",
            1 + num_ctrl_qubits,
            [theta],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=RYGate(theta),
        )


class MCRZGate(ControlledGate):
    """mcrz gate"""

    def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcrz",
            1 + num_ctrl_qubits,
            [theta],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=RZGate(theta),
        )


class MCRGate(ControlledGate):
    """mcr gate"""

    def __init__(self, theta, phi, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcr",
            1 + num_ctrl_qubits,
            [theta, phi],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=RGate(theta, phi),
        )


class MCU1Gate(ControlledGate):
    """mcu1 gate"""

    def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcu1",
            1 + num_ctrl_qubits,
            [theta],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=U1Gate(theta),
        )


class MCU2Gate(ControlledGate):
    """mcu2 gate"""

    def __init__(self, theta, lam, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcu2",
            1 + num_ctrl_qubits,
            [theta, lam],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=U2Gate(theta, lam),
        )


class MCU3Gate(ControlledGate):
    """mcu3 gate"""

    def __init__(self, theta, lam, phi, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcu3",
            1 + num_ctrl_qubits,
            [theta, phi, lam],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=U3Gate(theta, phi, lam),
        )


class MCUGate(ControlledGate):
    """mcu gate"""

    def __init__(self, theta, lam, phi, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcu",
            1 + num_ctrl_qubits,
            [theta, phi, lam],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=U3Gate(theta, phi, lam),
        )


class MCSwapGate(ControlledGate):
    """mcswap gate"""

    def __init__(self, num_ctrl_qubits, ctrl_state=None):
        super().__init__(
            "mcswap",
            2 + num_ctrl_qubits,
            [],
            None,
            num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=SwapGate(),
        )


PHI = Parameter("phi")
LAM = Parameter("lam")
NAME_MAPPING = {
    "mcsx": MCSXGate,
    "mcp": MCPhaseGate,
    "mcphase": MCPhaseGate,
    "initialize": Initialize,
    "quantum_channel": QuantumChannel,
    "save_expval": SaveExpectationValue,
    "diagonal": DiagonalGate,
    "save_amplitudes": SaveAmplitudes,
    "roerror": ReadoutError,
    "mcrx": MCRXGate,
    "kraus": Kraus,
    "save_statevector_dict": SaveStatevectorDict,
    "mcx": MCXGate,
    "mcu1": MCU1Gate,
    "mcu2": MCU2Gate,
    "mcu3": MCU3Gate,
    "save_superop": SaveSuperOp,
    "multiplexer": UCGate,
    "mcy": MCYGate,
    "superop": SuperOp,
    "save_clifford": SaveClifford,
    "save_matrix_product_state": SaveMatrixProductState,
    "save_density_matrix": SaveDensityMatrix,
    "save_probabilities": SaveProbabilities,
    "if_else": IfElseOp,
    "while_loop": WhileLoopOp,
    "for_loop": ForLoopOp,
    "switch_case": SwitchCaseOp,
    "break_loop": BreakLoopOp,
    "continue_loop": ContinueLoopOp,
    "save_statevector": SaveStatevector,
    "mcu": MCUGate,
    "set_density_matrix": SetDensityMatrix,
    "qerror_loc": QuantumErrorLocation,
    "unitary": UnitaryGate,
    "mcz": MCZGate,
    "pauli": PauliGate,
    "set_unitary": SetUnitary,
    "save_state": SaveState,
    "mcswap": MCSwapGate,
    "set_matrix_product_state": SetMatrixProductState,
    "save_unitary": SaveUnitary,
    "mcr": MCRGate,
    "mcx_gray": MCXGrayCode,
    "mcrz": MCRZGate,
    "set_superop": SetSuperOp,
    "save_expval_var": SaveExpectationValueVariance,
    "save_stabilizer": SaveStabilizer,
    "set_statevector": SetStatevector,
    "mcry": MCRYGate,
    "set_stabilizer": SetStabilizer,
    "save_amplitudes_sq": SaveAmplitudesSquared,
    "save_probabilities_dict": SaveProbabilitiesDict,
    "save_probs_ket": SaveProbabilitiesDict,
    "save_probs": SaveProbabilities,
    "cu2": U2Gate(PHI, LAM).control(),
    "reset": Reset(),
}
