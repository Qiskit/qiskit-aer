# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Simulator instruction to set a program counter
"""

from qiskit.circuit import Instruction
from qiskit.circuit.classical.expr import Expr


class AerJump(Instruction):
    """
    Jump instruction

    This instruction sets a program counter to specified mark instruction.
    """

    _directive = True

    def __init__(self, jump_to, num_qubits, num_clbits=0):
        super().__init__("jump", num_qubits, num_clbits, [jump_to])
        self.condition_expr = None
        self.condition = None

    def set_conditional(self, cond):
        """Set condition to perform this jump instruction.

        Args:
            cond (Expr or tuple): `Expr` to call `eval_bool` or tuple for `c_if`

        Returns:
            AerJump: jump instruction added specified condition
        """
        if isinstance(cond, Expr):
            self.condition_expr = cond
        else:
            self.condition = cond
        return self
