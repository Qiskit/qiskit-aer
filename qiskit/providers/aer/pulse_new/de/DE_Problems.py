# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from typing import Callable, Union, List, Optional

from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.operator_models import OperatorModel
from qiskit.providers.aer.pulse_new.de.type_utils import StateTypeConverter

class BMDE_Problem:
    """Class for representing Bilinear Matrix Differential Equations.
    """

    def __init__(self,
                 generator: OperatorModel,
                 y0: Optional[np.ndarray] = None,
                 t0: Optional[float] = None,
                 interval: Optional[List[float]] = None,
                 frame_operator: Optional[Union[str, Operator, np.ndarray]] = 'auto',
                 cutoff_freq: Optional[float] = None,
                 state_type_converter: Optional[StateTypeConverter] = None):
        """fill in
        """

        # set state parameters
        self._state_type_converter = state_type_converter
        self.y0 = y0

        # set initial time or interval
        if (interval is not None) and (t0 is not None):
            raise Exception('Only specify one of t0 or interval.')

        self.t0 = t0
        self.interval = interval

        # set up frame
        solver_frame_operator = None
        if generator.frame_operator is not None:
            # if the generator has a frame specified, leave it as
            solver_frame_operator = generator.frame_operator
            self._user_in_frame = True
        else:
            # if auto, go into the drift part of the generator, otherwise
            # set it to whatever as passed
            if frame_operator == 'auto':
                solver_frame_operator = anti_herm_part(generator.drift)
            else:
                solver_frame_operator = frame_operator

            self._user_in_frame = False

        # set up cutoff freq
        solver_cutoff_freq = None
        if generator.cutoff_freq is not None and cutoff_freq is not None:
            raise Exception("""Cutoff frequency specified in generator and in
                                solver settings.""")

        if generator.cutoff_freq is not None:
            solver_cutoff_freq = generator.cutoff_freq
        else:
            solver_cutoff_freq = cutoff_freq

        # set up signals
        if generator._signal_params is not None:
            signals = generator._signal_params
        else:
            signals = generator._signals

        # construct new internal generator for solving
        self._generator = OperatorModel(operators=generator._operators,
                                        signals=signals,
                                        signal_mapping=generator.signal_mapping,
                                        frame_operator=solver_frame_operator,
                                        cutoff_freq=solver_cutoff_freq)


def anti_herm_part(A: Union[np.ndarray, Operator]):
    """Get the anti-hermitian part of an operator.
    """
    return 0.5 * (A - A.conj().transpose())
