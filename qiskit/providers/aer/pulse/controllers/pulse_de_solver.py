# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
# pylint: disable=invalid-name

"""Set up DE solver for problems in qutip format."""

from ..de.DE_Methods import method_from_string


def setup_de_solver(exp, y0, pulse_de_model, de_options):
    """ Constructs a scipy ODE solver for a given exp and op_system

    Parameters:
        exp (dict): dict containing experiment description
        y0 (array): initial state
        pulse_de_model (PulseInternalDEModel): container for de model
        de_options (DE_Options): options for DE method

    Returns:
        solver: ODE_Method instance initialized with state and rhs function
    """

    method = method_from_string(de_options.method)

    rhs = pulse_de_model.init_rhs(exp)
    solver = method(0.0, y0, rhs, de_options)
    return solver
