# This code is part of Qiskit.
#
# (C) Copyright IBM 2018-2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Base quantum error class for Aer noise model
"""
import copy
import uuid
from abc import abstractmethod

from qiskit.circuit import QuantumCircuit, Instruction, QuantumRegister
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel import Kraus


class BaseQuantumError(BaseOperator):
    """Base quantum error class for Aer noise model"""

    def __init__(self, num_qubits: int):
        """Base class for a quantum error supported by Aer."""
        # Unique ID for BaseQuantumError
        self._id = uuid.uuid4().hex
        super().__init__(num_qubits=num_qubits)

    def _repr_name(self) -> str:
        return f"{type(self).__name__}[{self.id}]"

    def __repr__(self):
        """Display QuantumError."""
        return f"<{self._repr_name()}>"

    def __hash__(self):
        return hash(self._id)

    @property
    def id(self):  # pylint: disable=invalid-name
        """Return unique ID string for error"""
        return self._id

    def copy(self):
        """Make a copy of current QuantumError."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return copy.deepcopy(self)

    def to_instruction(self):
        """Convert the QuantumError to a circuit Instruction."""
        return QuantumChannelInstruction(self)

    @abstractmethod
    def ideal(self):
        """Return True if this error object is composed only of identity operations.
        Note that the identity check is best effort and up to global phase."""

    @abstractmethod
    def to_quantumchannel(self):
        """Convert the QuantumError to a SuperOp quantum channel.
        Required to enable SuperOp(QuantumError)."""

    @abstractmethod
    def to_dict(self):
        """Return the current error as a dictionary."""

    @abstractmethod
    def compose(self, other, qargs=None, front=False):
        pass

    @abstractmethod
    def tensor(self, other):
        pass

    @abstractmethod
    def expand(self, other):
        return other.tensor(self)

    def __rmul__(self, other):
        raise NotImplementedError(
            f"'{type(self).__name__}' does not support scalar multiplication."
        )

    def __truediv__(self, other):
        raise NotImplementedError(f"'{type(self).__name__}' does not support division.")

    def __add__(self, other):
        raise NotImplementedError(f"'{type(self).__name__}' does not support addition.")

    def __sub__(self, other):
        raise NotImplementedError(f"'{type(self).__name__}' does not support subtraction.")

    def __neg__(self):
        raise NotImplementedError(f"'{type(self).__name__}' does not support negation.")


class QuantumChannelInstruction(Instruction):
    """Container instruction for adding BaseQuantumError to circuit"""

    def __init__(self, quantum_error):
        """Initialize a quantum error circuit instruction.

        Args:
            quantum_error (BaseQuantumError): the error to add as an instruction.
        """
        super().__init__("quantum_channel", quantum_error.num_qubits, 0, [])
        self._quantum_error = quantum_error

    def _define(self):
        """Allow unrolling to a Kraus instruction"""
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc._append(Kraus(self._quantum_error).to_instruction(), q, [])
        self.definition = qc
