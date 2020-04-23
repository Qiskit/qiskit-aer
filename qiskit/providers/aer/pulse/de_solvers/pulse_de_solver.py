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

"""Pulse DE solver for problems in qutip format."""

import numpy as np
from scipy.integrate import ode
from scipy.integrate._ode import zvode
# pylint: disable=no-name-in-module
from .pulse_utils import td_ode_rhs_static


def construct_pulse_zvode_solver(exp, op_system):
    """ Constructs a scipy ODE solver for a given exp and op_system

    Parameters:
        exp (dict): dict containing experimental
        op_system (PulseSimDescription): container for simulation information

    Returns:
        ode: scipy ode
    """

    # extract relevant data from op_system
    global_data = op_system.global_data
    ode_options = op_system.ode_options
    channels = dict(op_system.channels)

    # Init register
    register = np.ones(global_data['n_registers'], dtype=np.uint8)

    ODE = ode(td_ode_rhs_static)

    ODE.set_f_params(global_data, exp, op_system.system, channels, register)

    ODE._integrator = qiskit_zvode(method=ode_options.method,
                                   order=ode_options.order,
                                   atol=ode_options.atol,
                                   rtol=ode_options.rtol,
                                   nsteps=ode_options.nsteps,
                                   first_step=ode_options.first_step,
                                   min_step=ode_options.min_step,
                                   max_step=ode_options.max_step
                                   )
    # Forces complex ODE solving
    if not ODE._y:
        ODE.t = 0.0
        ODE._y = np.array([0.0], complex)
    ODE._integrator.reset(len(ODE._y), ODE.jac is not None)

    # Since all experiments are defined to start at zero time.
    ODE.set_initial_value(global_data['initial_state'], 0)

    return ODE


class qiskit_zvode(zvode):
    """Modifies the stepper for ZVODE so that
    it always stops at a given time in tlist;
    by default, it over shoots the time.
    """
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r
