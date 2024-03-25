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


class AerStore(Instruction):
    """
    Store instruction for Aer to work wround transpilation issue
    of qiskit.circuit.Store
    """

    _directive = True

    def __init__(self, num_qubits, num_clbits, store):
        super().__init__("aer_store", num_qubits, num_clbits, [store.lvalue, store.rvalue])
        self.store = store
