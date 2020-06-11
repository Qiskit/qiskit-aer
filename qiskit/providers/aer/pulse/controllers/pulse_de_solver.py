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
# pylint: disable=no-value-for-parameter, invalid-name, import-error

"""Set up DE solver for problems in qutip format."""

from ..de.DE_Methods import QiskitZVODE
from ..de.DE_Options import DE_Options


def construct_pulse_zvode_solver(exp, y0, op_system, ode_options):
    """ Constructs a scipy ODE solver for a given exp and op_system

    Parameters:
        exp (dict): dict containing experimental
        op_system (PulseSimDescription): container for simulation information

    Returns:
        ode: scipy ode
    """

    rhs = op_system.init_rhs(exp)

    options = DE_Options(method=ode_options.method,
                         order=ode_options.order,
                         atol=ode_options.atol,
                         rtol=ode_options.rtol,
                         nsteps=ode_options.nsteps,
                         first_step=ode_options.first_step,
                         min_step=ode_options.min_step,
                         max_step=ode_options.max_step)

    qiskit_zvode = QiskitZVODE(0.0, y0, rhs, options)
    return qiskit_zvode
