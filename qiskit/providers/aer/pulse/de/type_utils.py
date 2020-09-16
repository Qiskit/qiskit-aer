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
# pylint: disable=invalid-name

"""Utilities for type handling/conversion for DE classes.

A type specification is a dictionary describing a specific expected type, e.g. an array of a given
shape. Currently only handled types are numpy arrays, specified via:
    - {'type': 'array', 'shape': tuple}
"""

import numpy as np


class StateTypeConverter:
    """Contains descriptions of two type specifications for DE solvers/methods, with functions for
    converting states and rhs functions between representations.

    While this class stores exact type specifications, it can be instantiated with a
    concrete type and a more general type. This facilitates the situation
    in which a solver requires a 1d array, which is specified by the type:
        - {'type': 'array', 'ndim': 1}
    """

    def __init__(self, inner_type_spec, outer_type_spec=None):
        """Instantiate with the inner and return types for the state.

        Args:
            inner_type_spec (dict): inner type
            outer_type_spec (dict): outer type
        """

        self.inner_type_spec = inner_type_spec

        self.outer_type_spec = self.inner_type_spec if outer_type_spec is None else outer_type_spec

    @classmethod
    def from_instances(cls, inner_y, outer_y=None):
        """Instantiate from concrete instances. Type of instances must be supported by
        type_spec_from_instance. If outer_y is None the outer type is set to the inner type

        Args:
            inner_y (array): concrete representative of inner type
            outer_y (array): concrete representative of outer type

        Returns:
            StateTypeConverter: type converter as specified by args
        """
        inner_type_spec = type_spec_from_instance(inner_y)

        outer_type_spec = None
        if outer_y is not None:
            outer_type_spec = type_spec_from_instance(outer_y)

        return cls(inner_type_spec, outer_type_spec)

    @classmethod
    def from_outer_instance_inner_type_spec(cls, outer_y, inner_type_spec=None):
        """Instantiate from concrete instance of the outer type, and an inner type-spec.
        The inner type spec can be either be fully specified, or be more general (i.e. to
        facilitate the situation in which a solver needs a 1d array).

        Accepted general data types:
            - {'type': 'array'}
            - {'type': 'array', 'ndim': 1}

        Args:
            outer_y (array): concrete outer data type
            inner_type_spec (dict): inner, potentially general, type spec

        Returns:
            StateTypeConverter: type converter as specified by args

        Raises:
            Exception: if inner_type_spec is not properly specified or is not a handled type
        """

        # if no inner_type_spec given just instantiate both inner and outer to the outer_y
        if inner_type_spec is None:
            return cls.from_instances(outer_y)

        inner_type = inner_type_spec.get('type')
        if inner_type is None:
            raise Exception("inner_type_spec needs a 'type' key.")

        if inner_type == 'array':
            outer_y_as_array = np.array(outer_y)

            # if a specific shape is given attempt to instantiate from a reshaped outer_y
            shape = inner_type_spec.get('shape')
            if shape is not None:
                return cls.from_instances(outer_y_as_array.reshape(shape), outer_y)

            # handle the case that ndim == 1 is given
            ndim = inner_type_spec.get('ndim')
            if ndim == 1:
                return cls.from_instances(outer_y_as_array.flatten(), outer_y)

            # if neither shape nor ndim is given, assume it can be an array of any shape
            return cls.from_instances(outer_y_as_array, outer_y)

        raise Exception('inner_type_spec not a handled type.')

    def inner_to_outer(self, y):
        """Convert a state of inner type to one of outer type."""
        return convert_state(y, self.outer_type_spec)

    def outer_to_inner(self, y):
        """Convert a state of outer type to one of inner type."""
        return convert_state(y, self.inner_type_spec)

    def transform_rhs_funcs(self, rhs_funcs):
        """Convert RHS funcs passed in a dictionary from functions taking/returning outer type,
        to functions taking/returning inner type.

        Currently supports:
            - rhs_funcs['rhs'] - standard differential equation rhs function f(t, y)

        Args:
            rhs_funcs (dict): contains various rhs functions

        Returns:
            dict: transformed rhs funcs
        """

        new_rhs_funcs = {}

        # transform standard rhs function
        rhs = rhs_funcs.get('rhs')

        if rhs is not None:
            def new_rhs(t, y):
                outer_y = self.inner_to_outer(y)
                rhs_val = rhs(t, outer_y)
                return self.outer_to_inner(rhs_val)

            new_rhs_funcs['rhs'] = new_rhs

        return new_rhs_funcs


def convert_state(y, type_spec):
    """Convert the de state y into the type specified by type_spec. Accepted values of type_spec
    are given at the beginning of the file."""

    new_y = None

    if type_spec['type'] == 'array':
        # default array data type to complex
        new_y = np.array(y, dtype=type_spec.get('dtype', 'complex'))

        shape = type_spec.get('shape')
        if shape is not None:
            new_y = new_y.reshape(shape)

    return new_y


def type_spec_from_instance(y):
    """Determine type spec from an instance."""
    type_spec = {}
    if isinstance(y, np.ndarray):
        type_spec['type'] = 'array'
        type_spec['shape'] = y.shape

    return type_spec
