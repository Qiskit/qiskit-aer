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

from DE_Methods import ODE_Method, method_from_string
from DE_Problems import BMDE_Problem
from DE_Options import DE_Options
from type_utils import StateTypeConverter

class BMDE_Solver:
    """An intermediate interface to underlying DE solver methods."""

    def __init__(self,
                 bmde_problem: BMDE_Problem,
                 method: Optional[ODE_Method],
                 options: Optional[DE_Options] = None):
        """fill in
        """

        # determine initial time
        t0 = None
        if bmde_problem.t0 is not None:
            t0 = bmde_problem.t0
        elif bmde_problem.interval is not None:
            t0 = bmde_problem.interval[0]

        self.generator = bmde_problem._generator

        # setup solver method
        if options is None:
            options = DE_Options()

        # if no method explicitly provided, use the one in options
        if method is None:
            method = options.method

        Method = None
        if isinstance(method, str):
            Method = method_from_string(method)
        elif issubclass(method, ODE_Method):
            Method = method

        # instantiate method with minimal parameters
        self._method = Method(t0, y0=None, rhs=None, options=options)

        # flag signifying whether to return results in frame or not
        self._return_in_frame = self._frame_from_model

        self._state_type_converter = bmde_problem._state_type_converter
        if bmde_problem._y0 is not None:
            self.y = bmde_problem.y0

        # set RHS functions to evaluate in frame basis
        rhs_dict = {'rhs': lambda t, y: self.generator.lmult(t, y,
                                                             in_frame_basis=True),
                    'generator': lambda t: self.generator.evaluate(t,
                                                                   in_frame_basis=True)}
        self._method.set_rhs(rhs_dict)

    @property
    def t(self):
        return self._method.t

    @t.setter
    def t(self, new_t):
        self._method.t = new_t

    @property
    def y(self):
        return self.get_y()

    @y.setter
    def y(self, new_y):
        if new_y is not None:
            self.set_y(new_y)

    def set_y(self, y: np.ndarray, y_in_frame: Optional[bool] = None):
        """Set the state of the BMDE.

        State is internally represented in the frame of the internal generator,
        and in the basis in which the frame operator is diagonal.

        Args:
            y: new state
            y_in_frame: whether or not y is specified in the rotating frame
        """

        # if frame of y is not specified, assume it's specified in
        # the return frame
        if y_in_frame is None:
            y_in_frame = self._return_in_frame


        # convert y into internal representation
        new_y = None
        if self._state_type_converter is None:
            new_y = y
        else:
            new_y = self._state_type_converter.outer_to_inner(y)


        # convert y into the frame for the bmde, and also into the frame basis
        if y_in_frame:
            # if y is already in the frame, only need to convert into
            # frame basis
            new_y = self.generator._frame_freq_helper.state_into_frame_basis(new_y)
        else:
            # if y not in frame, convert it into frame and into frame basis
            new_y = self.generator._frame_freq_helper.state_into_frame(self.t,
                                                                       new_y,
                                                                       y_in_frame_basis=False,
                                                                       return_in_frame_basis=True)

        # set the converted state into the internal method state
        self._method.y = new_y


    def get_y(self, return_in_frame: Optional[bool] = None):
        """Return the state of the BMDE.

        Args:
            return_in_frame: whether or not to return in the solver frame
        """

        solver_state = self._method.y

        if return_in_frame is None:
            return_in_frame = self._return_in_frame

        if return_in_frame:
            y = solver_state
        else:
            y = self.generator._frame_freq_helper.state_out_of_frame(self.t,
                                                                     solver_state,
                                                                     True)
