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
from qiskit.circuit import Parameter


from qiskit.circuit.library import (
    MCPhaseGate,
    MCXGate,
    MCU1Gate,
    U2Gate,
    PauliGate,
    MCXGrayCode,
    UnitaryGate,
    UCGate,
    Initialize,
    DiagonalGate,
)
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    ContinueLoopOp,
    BreakLoopOp,
    SwitchCaseOp,
)


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

PHI = Parameter("phi")
LAM = Parameter("lam")
NAME_MAPPING = {
    "mcp": MCPhaseGate,
    "mcphase": MCPhaseGate,
    "quantum_channel": QuantumChannel,
    "initialize": Initialize,
    "save_expval": SaveExpectationValue,
    "diagonal": DiagonalGate,
    "save_amplitudes": SaveAmplitudes,
    "roerror": ReadoutError,
    "kraus": Kraus,
    "save_statevector_dict": SaveStatevectorDict,
    "mcx": MCXGate,
    "mcu1": MCU1Gate,
    "save_superop": SaveSuperOp,
    "multiplexer": UCGate,
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
    "set_density_matrix": SetDensityMatrix,
    "qerror_loc": QuantumErrorLocation,
    "unitary": UnitaryGate,
    "pauli": PauliGate,
    "set_unitary": SetUnitary,
    "save_state": SaveState,
    "set_matrix_product_state": SetMatrixProductState,
    "save_unitary": SaveUnitary,
    "mcx_gray": MCXGrayCode,
    "set_superop": SetSuperOp,
    "save_expval_var": SaveExpectationValueVariance,
    "save_stabilizer": SaveStabilizer,
    "set_statevector": SetStatevector,
    "set_stabilizer": SetStabilizer,
    "save_amplitudes_sq": SaveAmplitudesSquared,
    "save_probabilities_dict": SaveProbabilitiesDict,
    "cu2": U2Gate(PHI, LAM).control(),
}
