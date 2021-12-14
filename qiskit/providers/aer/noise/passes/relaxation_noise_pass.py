# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Thermal relaxation noise pass.
"""
import warnings
from typing import Optional, Union, Sequence, List

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from .local_noise_pass import LocalNoisePass
from ..errors.standard_errors import thermal_relaxation_error


class RelaxationNoisePass(LocalNoisePass):
    """Add duration dependent thermal relaxation noise after instructions."""

    def __init__(
            self,
            t1s: List[float],
            t2s: List[float],
            dt: float,
            op_types: Optional[Union[type, Sequence[type]]] = None,
            excited_state_populations: Optional[List[float]] = None,
    ):
        """Initialize RelaxationNoisePass.

        Args:
            t1s: List of T1 times in seconds for each qubit.
            t2s: List of T2 times in seconds for each qubit.
            dt: Backend sample time (resolution) in seconds. This is used as the time
                unit when converting scheduled circuit op durations to times.
            op_types: Optional, the operation types to add relaxation to. If None
                relaxation will be added to all operations.
            excited_state_populations: Optional, list of excited state populations
                for each qubit at thermal equilibrium. If not supplied or obtained
                from the backend this will be set to 0 for each qubit.
        """
        self._t1s = np.asarray(t1s)
        self._t2s = np.asarray(t2s)
        if excited_state_populations is not None:
            self._p1s = np.asarray(excited_state_populations)
        else:
            self._p1s = np.zeros(len(t1s))
        self._dt = dt
        super().__init__(self._thermal_relaxation_error, op_types=op_types, method="append")

    def _thermal_relaxation_error(
            self,
            op: Instruction,
            qubits: Sequence[int]
    ):
        """Return thermal relaxation error on each operand qubit"""
        if not op.duration:
            if op.duration is None:
                warnings.warn("RelaxationNoisePass ignores instructions without duration,"
                              " you may need to schedule circuit in advance.", UserWarning)
            return None

        # convert duration in seconds
        duration = op.duration * self._dt

        t1s = self._t1s[qubits]
        t2s = self._t2s[qubits]
        p1s = self._p1s[qubits]

        # pylint: disable=invalid-name
        if op.num_qubits == 1:
            t1, t2, p1 = t1s[0], t2s[0], p1s[0]
            if t1 == np.inf and t2 == np.inf:
                return None
            return thermal_relaxation_error(t1, t2, duration, p1)

        # General multi-qubit case
        noise = QuantumCircuit(op.num_qubits)
        for qubit, (t1, t2, p1) in enumerate(zip(t1s, t2s, p1s)):
            if t1 == np.inf and t2 == np.inf:
                # No relaxation on this qubit
                continue
            error = thermal_relaxation_error(t1, t2, duration, p1)
            noise.append(error.to_instruction(), [qubit])

        return noise
