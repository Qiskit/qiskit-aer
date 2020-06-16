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

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
# pylint: disable=invalid-name, attribute-defined-outside-init

"""DE methods."""

from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.integrate._ode import zvode
from .DE_Options import DE_Options
from .type_utils import StateTypeConverter


class ODE_Method(ABC):
    """Abstract wrapper class for an ODE solving method, providing an expected interface
    for integrating a new method/solver.

    Class Attributes:
        method_spec (dict): Container of general information about the method.
                            Currently supports keys:
                                - 'inner_state_spec': description of the datatype a solver requires,
                                                      with accepted descriptions given in type_utils

    Instance attributes:
        _t, t (float): private and public time variable.
        _y, y (array): private and public state variable.
        rhs (dict): rhs-related functions as values Currently supports key 'rhs'.
    """

    method_spec = {'inner_state_spec': {'type': 'array'}}

    def __init__(self, t0=None, y0=None, rhs=None, options=None):

        # set_options is first as options may influence the behaviour of other functions
        self.set_options(options)

        self._t = t0
        self.set_y(y0, reset=False)
        self.set_rhs(rhs)

        # attributes for error reporting
        self._successful = True
        self._return_code = None

    def integrate_over_interval(self, y0, interval, rhs=None, **kwargs):
        """Integrate over an interval, with additional options to reset the rhs functions.

        Args:
            y0 (array): state at the start of the interval
            interval (tuple or list): initial and start time, e.g. (t0, tf)
            rhs (callable or dict): Either the rhs function itself, or a dict of rhs-related
                                    functions. If not given, will use the already-stored rhs.
            kwargs (dict): additional keyword arguments for the integrate function of a concrete
                           method

        Returns:
            state: state at the end of the interval
        """
        t0 = interval[0]
        tf = interval[1]

        self._t = t0
        self.set_y(y0, reset=False)
        if rhs is not None:
            self.set_rhs(rhs, reset=False)

        self._reset_method()

        self.integrate(tf, **kwargs)

        return self.y

    @property
    def t(self):
        """Time property."""
        return self._t

    @t.setter
    def t(self, new_t):
        """Time setter."""
        self._t = new_t
        self._reset_method()

    @property
    def y(self):
        """State property."""
        return self._state_type_converter.inner_to_outer(self._y)

    @y.setter
    def y(self, new_y):
        """State setter."""
        self.set_y(new_y)

    def set_y(self, new_y, reset=True):
        """Method for logic of setting internal state of solver with more control
        """

        # instantiate internal StateTypeConverter based on the provided new_y and the
        # general type required internally by the solver
        type_spec = self.method_spec.get('inner_state_spec')
        self._state_type_converter = \
            StateTypeConverter.from_outer_instance_inner_type_spec(new_y, type_spec)

        # set internal state
        self._y = self._state_type_converter.outer_to_inner(new_y)

        self._reset_method(reset)

    def set_rhs(self, rhs=None, reset=True):
        """Set rhs functions.

        Args:
            rhs (dict or callable): Either a dict with callable values,
                                    e.g. {'rhs': f}, or a callable f, which
                                    produces equivalent behaviour as the input {'rhs': f}
            reset (bool): Whether or not to reset solver

        Raises:
            Exception: if rhs dict is mis-specified
        """

        if rhs is None:
            rhs = {'rhs': None}

        if callable(rhs):
            rhs = {'rhs': rhs}

        if 'rhs' not in rhs:
            raise Exception('ODE_Method requires at minimum a specification of an rhs function.')

        # transform rhs function into a function that accepts/returns inner state type
        self.rhs = self._state_type_converter.transform_rhs_funcs(rhs)

        self._reset_method(reset)

    def successful(self):
        """Return if whether method is successful."""
        return self._successful

    def return_code(self):
        """Get return code."""
        return self._return_code

    @abstractmethod
    def integrate(self, tf, **kwargs):
        """Integrate up to a time tf.

        Args:
            tf (float): time to integrate up to
            kwargs (dict): key word arguments specific to a given method
        """
        pass

    def _reset_method(self, reset=True):
        """Reset any parameters of internal numerical solving method, e.g. delete persistent memory
        for multi-step methods.

        Args:
            reset (bool): Whether or not to reset method
        """
        pass

    def set_options(self, options):
        """Setup options for the method."""
        pass


class ScipyODE(ODE_Method):
    """Method wrapper for scipy.integrate.solve_ivp.

    To use:
        - Specify a method acceptable by the keyword argument 'method' scipy.integrate.solve_ivp
          in DE_Options attribute 'method'. Methods that currently work are:
            - 'RK45', 'RK23', and 'BDF'
            - Default if not specified is 'RK45'

    Additional notes:
        - solve_ivp requires states to be 1d
        - Enabling other methods requires adding dtype handling to type_utils for solvers that
          do not handle complex types
    """

    method_spec = {'inner_state_spec': {'type': 'array', 'ndim': 1}}

    def integrate(self, tf, **kwargs):
        """Integrate up to a time tf.
        """
        t0 = self.t
        y0 = self._y
        rhs = self.rhs.get('rhs')

        # solve problem and silence warnings for options that don't apply to a given method
        kept_warnings = []
        with warnings.catch_warnings(record=True) as ws:
            results = solve_ivp(rhs, (t0, tf), y0,
                                method=self.options.method,
                                atol=self.options.atol,
                                rtol=self.options.rtol,
                                max_step=self.options.max_step,
                                min_step=self.options.min_step,
                                first_step=self.options.first_step,
                                **kwargs)

            # update the internal state
            self._y = results.y[:, -1]
            self._t = results.t[-1]

            # discard warnings for arguments with no effect
            for w in ws:
                if 'The following arguments have no effect' not in str(w.message):
                    kept_warnings.append(w)

        # display warnings we don't want to silence
        for w in kept_warnings:
            warnings.warn(w.message, type(w))

    def set_options(self, options):
        # establish method
        if options is None:
            options = DE_Options()
            options.method = 'RK45'
        else:
            options = options.copy()
            if 'scipy-' in options.method:
                options.method = options.method[6:]

        self.options = options

        # handle defaults for None-type arguments
        if self.options.max_step is None:
            self.options.max_step = np.inf


class QiskitZVODE(ODE_Method):
    """Wrapper for zvode solver available through Scipy.

    Notes:
        - Internally this
    """

    method_spec = {'inner_state_spec': {'type': 'array', 'ndim': 1}}

    def __init__(self, t0=None, y0=None, rhs=None, options=None):

        # all de specification arguments are necessary to instantiate scipy ode object
        if (t0 is None) or (y0 is None) or (rhs is None):
            raise Exception('QiskitZVODE solver requires t0, y0, and rhs at instantiation.')

        # initialize internal attribute for storing scipy ode object
        self._ODE = None

        super().__init__(t0, y0, rhs, options)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, new_t):
        self._t = new_t
        self._ODE.t = new_t
        self._reset_method()

    def set_y(self, new_y, reset=True):
        """Method for logic of setting internal state of solver with more control
        """
        type_spec = self.method_spec.get('inner_state_spec')
        self._state_type_converter = \
            StateTypeConverter.from_outer_instance_inner_type_spec(new_y, type_spec)

        self._y = self._state_type_converter.outer_to_inner(new_y)

        if self._ODE is not None:
            self._ODE._y = self._y

        self._reset_method(reset)

    def set_rhs(self, rhs=None, reset=True):
        """This set_rhs function fully instantiates the scipy ode object behind the scenes."""

        if rhs is None:
            rhs = {'rhs': None}

        if callable(rhs):
            rhs = {'rhs': rhs}

        if 'rhs' not in rhs:
            raise Exception('ODE_Method requires at minimum a specification of an rhs function.')

        self.rhs = self._state_type_converter.transform_rhs_funcs(rhs)

        self._ODE = ode(self.rhs['rhs'])

        self._ODE._integrator = qiskit_zvode(method=self.options.method,
                                             order=self.options.order,
                                             atol=self.options.atol,
                                             rtol=self.options.rtol,
                                             nsteps=self.options.nsteps,
                                             first_step=self.options.first_step,
                                             min_step=self.options.min_step,
                                             max_step=self.options.max_step
                                             )

        # Forces complex ODE solving
        if not self._ODE._y:
            self._ODE.t = 0.0
            self._ODE._y = np.array([0.0], complex)
        self._ODE._integrator.reset(len(self._ODE._y), self._ODE.jac is not None)

        self._ODE.set_initial_value(self._y, self._t)

        self._reset_method(reset)

    def integrate(self, tf, **kwargs):
        """Integrate up to a time tf.

        Args:
            tf (float): time to integrate up to
            kwargs (dict): Supported kwargs:
                            - 'step': if False, integrates up to tf, if True, only implements a
                                      single step of the solver
        """

        step = kwargs.get('step', False)

        self._ODE.integrate(tf, step=step)

        # update state stored locally
        self._y = self._ODE.y
        self._t = self._ODE.t

        # update success parameters
        self._successful = self._ODE.successful()
        self._return_code = self._ODE.get_return_code()

    def _reset_method(self, reset=True):
        """Discard internal memory."""
        if reset:
            self._ODE._integrator.call_args[3] = 1

    def set_options(self, options):
        # establish method
        if options is None:
            options = DE_Options(method='adams')
        else:
            options = options.copy()
            if 'zvode-' in options.method:
                options.method = options.method[6:]

        # handle None-type defaults
        if options.first_step is None:
            options.first_step = 0

        if options.max_step is None:
            options.max_step = 0

        if options.min_step is None:
            options.min_step = 0

        self.options = options


class qiskit_zvode(zvode):
    """Customized ZVODE with modified stepper so that
    it always stops at a given time in tlist;
    by default, it over shoots the time.
    """
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        # pylint: disable=no-value-for-parameter
        r = self.run(*args)
        self.call_args[2] = itask
        return r


class RK4(ODE_Method):
    """Single-step RK4 solver. Serves as a simple/minimal example of a concrete ODE_Method
    subclass.
    """

    def integrate(self, tf, **kwargs):
        """Integrate up to a time tf.
        """

        delta_t = tf - self.t
        steps = int((delta_t // self._max_dt) + 1)
        h = delta_t / steps
        for _ in range(steps):
            self._integration_step(h)

    def _integration_step(self, h):
        """Integration step for RK4
        """
        y0 = self._y
        t0 = self._t
        rhs = self.rhs.get('rhs')

        k1 = rhs(t0, y0)
        t_mid = t0 + (h / 2)
        k2 = rhs(t_mid, y0 + (h * k1 / 2))
        k3 = rhs(t_mid, y0 + (h * k2 / 2))
        t_end = t0 + h
        k4 = rhs(t_end, y0 + h * k3)
        self._y = y0 + (1. / 6) * h * (k1 + (2 * k2) + (2 * k3) + k4)
        self._t = t_end

    def set_options(self, options):
        self._max_dt = options.max_dt


def method_from_string(method_str):
    """Returns an ODE_Method specified by a string.

    Args:
        method_str (str): string specifying method

    Returns:
        method: instance of an ODE_Method object
    """

    if 'scipy-' in method_str:
        return ScipyODE

    if 'zvode-' in method_str:
        return QiskitZVODE

    method_dict = {'RK4': RK4,
                   'scipy': ScipyODE,
                   'zvode': QiskitZVODE}

    return method_dict.get(method_str)
