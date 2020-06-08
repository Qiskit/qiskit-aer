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

"""DE methods."""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from .type_utils import StateTypeConverter


class ODE_Method(ABC):
    """Abstract wrapper class for an ODE solving method, providing an expected interface
    for integrating a new method/solver.

    Class Attributes:
        method_spec (dict): Container of general information about the method. Currently
                            supports key 'inner_state_spec', containing a description of
                            of the data type that the underlying method requires. Must be
                            understandable by StateTypeConverter, which will automatically
                            handle conversions of the state and rhs functions.

    Instance attributes:
        _t, t (float): private and public time variable
        _y, y (array): private and public state variable
        rhs (dict): rhs-related functions as values
    """

    method_spec = {'inner_state_spec': {'type': 'array'}}

    def __init__(self, t0=None, y0=None, rhs=None, solver_options={}):

        # set_options should be first as options may influence the behaviour of other functions
        self.set_options(solver_options)

        self._t = t0
        self.set_y(y0, reset=False)
        self.set_rhs(rhs)

        # default to True, only to be changed to false if a failure occurs
        self._successful = True

    def integrate_over_interval(self, y0, interval, rhs=None):
        """Integrate over an interval, with additional options to reset the rhs functions.

        Args:
            y0 (array): state at the start of the interval
            interval (tuple or list): initial and start time, e.g. (t0, tf)
            rhs (callable or dict): Either the rhs function itself, or a dict of rhs-related
                                    functions

        Returns:
            state of the solver at the end of the integral
        """
        t0 = interval[0]
        tf = interval[1]

        self._t = t0
        self.set_y(y0, reset=False)
        if rhs is not None:
            self.set_rhs(rhs, reset=False)

        self._reset_method()

        self.integrate(tf)

        return self.y

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, new_t):
        self._t = new_t
        self._reset_method()

    @property
    def y(self):
        return self._state_type_converter.inner_to_outer(self._y)

    @y.setter
    def y(self, new_y):
        self.set_y(new_y)

    def set_y(self, new_y, reset=True):
        """Method for logic of setting internal state of solver with more control
        """
        type_spec = self.method_spec.get('inner_state_spec')
        self._state_type_converter = \
                        StateTypeConverter.from_outer_instance_inner_type_spec(new_y, type_spec)

        self._y = self._state_type_converter.outer_to_inner(new_y)

        self._reset_method(reset)


    def set_rhs(self, rhs=None, reset=True):
        """Set rhs functions.

        Args:
            rhs (dict or callable): Either a dict with callable values,
                                    e.g. {'rhs': f, 'rhs_jac': g}, or a callable f, which
                                    produces equivalent behaviour as the input {'rhs', f}
        """

        if rhs is None:
            rhs = {'rhs': None}

        if callable(rhs):
            rhs = {'rhs': rhs}

        if 'rhs' not in rhs:
            raise Exception('ODE_Method requires at minimum a specification of an rhs function.')

        self.rhs = self._state_type_converter.transform_rhs_funcs(rhs)

        self._reset_method(reset)

    def successful(self):
        return self._successful


    """
    Functions to implement in concrete subclasses
    """

    @abstractmethod
    def integrate(self, tf):
        """Integrate up to a time tf.

        Args:
            tf (float): time to integrate up to
        """
        pass

    def _reset_method(self, reset=True):
        """Reset any parameters of internal numerical solving method, e.g. delete persistent memory
        for multi-step methods.

        Args:
            reset (bool): Whether or not to reset method
        """
        pass

    def set_options(self, solver_options):
        pass


class ScipyODE(ODE_Method):
    """Method wrapper for scipy.integrate.solve_ivp

    To use:
        - Specify a method acceptable by scipy.integrate.solve_ivp in solver_options using key
          'method'
        - Options for solve_ivp in the form of Keyword arguments may also be passed as a dict
          with key 'scipy_options' in solver_options
    """

    method_spec = {'inner_state_spec': {'type': 'array', 'ndim': 1}}

    def integrate(self, tf):
        """Integrate up to a time tf.
        """
        t0 = self.t
        y0 = self._y
        rhs = self.rhs.get('rhs')

        results = solve_ivp(rhs, (t0, tf), y0, method=self._scipy_method, **self._scipy_options)

        self._y = results.y[:, -1]
        self._t = results.t[-1]

    def set_options(self, solver_options):
        """Only option is max step size
        """
        if 'method' not in solver_options:
            raise Exception("""ScipyODE requires a 'method' key in solver_options with value a
                            method string acceptable by scipy.integrate.solve_ivp.""")
        self._scipy_method = solver_options.get('method')

        self._scipy_options = solver_options.get('scipy_options', {})


class QiskitZVODE(ODE_Method):
    """Wrapper for zvode solver available through Scipy."""

    method_spec = {'inner_state_spec': {'type': 'array', 'ndim': 1}}

    def __init__(self, t0=None, y0=None, rhs=None, solver_options={}):
        """This method requires t0, y0, and rhs to specified on instantiation, as these are
        necessary to properly instantiate the underlying solver object
        """

        # Add check for t0 and y0

        self._ODE = None

        super().__init__(t0, y0, rhs, solver_options)

        # remove after
        # set_options should be first as options may influence the behaviour of other functions
        self.set_options(solver_options)

        self._t = t0
        self.set_y(y0, reset=False)
        self.set_rhs(rhs)

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
        """Set rhs functions. rhs may either be a dict specifying multiple functions related
        to the rhs, (e.g. {'rhs': f, 'rhs_jac': g}), or a callable, in which case it will be
        assumed to be the standard rhs function.
        """

        if rhs is None:
            rhs = {'rhs': None}

        if callable(rhs):
            rhs = {'rhs': rhs}

        if 'rhs' not in rhs:
            raise Exception('ODE_Method requires at minimum a specification of an rhs function.')

        self.rhs = self._state_type_converter.transform_rhs_funcs(rhs)

        self._ODE = ode(self.rhs['rhs'])

        """
        self._ODE._integrator = qiskit_zvode(method=ode_options.method,
                                             order=ode_options.order,
                                             atol=ode_options.atol,
                                             rtol=ode_options.rtol,
                                             nsteps=ode_options.nsteps,
                                             first_step=ode_options.first_step,
                                             min_step=ode_options.min_step,
                                             max_step=ode_options.max_step
                                             )
        """
        self._ODE._integrator = qiskit_zvode(method=self.solver_options.get('method', 'adams'),
                                             order=self.solver_options.get('order', 12),
                                             atol=self.solver_options.get('atol', 10**-8),
                                             rtol=self.solver_options.get('rtol', 10**-6),
                                             nsteps=self.solver_options.get('nsteps', 50000),
                                             first_step=self.solver_options.get('first_step', 0),
                                             min_step=self.solver_options.get('min_step', 0),
                                             max_step=self.solver_options.get('max_step', 0)
                                             )

        # Forces complex ODE solving
        if not self._ODE._y:
            self._ODE.t = 0.0
            self._ODE._y = np.array([0.0], complex)
        self._ODE._integrator.reset(len(self._ODE._y), self._ODE.jac is not None)

        self._ODE.set_initial_value(self._y, self._t)

        self._reset_method(reset)


    def integrate(self, tf, step=False):
        """Integrate up to a time tf.

        Args:
            tf (float): time to integrate up to
            step (bool): if False, integrates up to tf, if True, only implements a single step
        """
        self._ODE.integrate(tf, step=step)
        self._y = self._ODE.y
        self._t = self._ODE.t
        self._successful = self._ODE.successful()

    def _reset_method(self, reset=True):
        if reset:
            self._ODE._integrator.call_args[3] = 1

    def set_options(self, solver_options):
        self.solver_options = solver_options


class qiskit_zvode(zvode):
    """Customized ZVODE with modified stepper so that
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


class RK4(ODE_Method):
    """
    Simple single-step RK4 solver
    """

    def integrate(self, tf):
        """Integrate up to a time tf.
        """

        delta_t = tf - self.t
        steps = int((delta_t // self._max_dt) + 1)
        h = delta_t / steps
        for k in range(steps):
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

    def set_options(self, solver_options):
        """Only option is max step size
        """
        if 'max_dt' not in solver_options:
            raise Exception('Solver requires max_dt setting')
        self._max_dt = solver_options['max_dt']


def method_from_string(method_str):
    """Factory function that returns a method specified by a string, along with any additional
    required options.

    Args:
        method_str (str): string specifying method

    Returns:
        (method, additional_options): method is the ODE_Method object, and additional_options
                                      is a dict containing any necessary options for that solver
    """

    method_dict = {'RK4': RK4}

    if method_str in method_dict:
        return method_dict.get(method_str), {}

    if 'scipy-' in method_str:
        return ScipyODE, {'method': method_str[6:]}
