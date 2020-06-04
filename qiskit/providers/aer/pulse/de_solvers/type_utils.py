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

"""Utilities for type handling/conversion for DE classes."""

import numpy as np

class StateTypeConverter:
    """Contains descriptions of two data types for DE solvers/methods, with functions for
    converting states and rhs functions between representations.

    Stores two descriptions: inner_type_spec (internal representation of data) and
    outer_type_spec (expected return type). Currently only supports conversion between
    numpy array shapes, represented as dictionaries:
        - {'type': 'array', 'shape': tuple}

    This class stores exact type information to implement conversions.

    While this class is meant to store exact type information, it can be instantiated with a
    concrete type and a more general type. This is specifically to facilitate the situation
    in which internally a solver requires a 1d array, which is specified by the type:
        - {'type': 'array', 'ndim': 1}
    """

    def __init__(self, inner_type_spec, outer_type_spec=None):
        """Instantiate with the inner and return types for the state.

        Args:
            inner_type_spec (dict): inner type
            outer_type_spec: outer type
        """

        self.inner_type_spec = inner_type_spec

        if outer_type_spec is not None:
            self.outer_type_spec = outer_type_spec
        else:
            self.outer_type_spec = self.inner_type_spec

    @classmethod
    def from_instances(cls, inner_y, outer_y=None):
        """Instantiate from concrete instances. Type of instances must be supported by
        type_spec_from_instance. If outer_y is None the outer type is set to the inner type

        Args:
            inner_y: concerete representative of inner type
            outer_y: concrete representative of outer type
        """
        inner_type_spec = type_spec_from_instance(inner_y)

        outer_type_spec = None
        if outer_y is not None:
            outer_type_spec = type_spec_from_instance(outer_y)

        return cls(inner_type_spec, outer_type_spec)

    @classmethod
    def from_outer_instance_inner_type_spec(cls, outer_y, inner_type_spec=None):
        """Instantiate from concrete instance of the outer type, and an inner type-spec.
        The inner type spec can be a fully specified on, or more general, to facilitate the
        situation in which a solver needs a 1d array.

        Accepted general data types:
            - {'type': 'array'}
            - {'type': 'array', 'ndim': 1}

        Args:
            outer_y: concrete outer data type
            inner_type_spec (dict): inner, potentially general, type spec
        """

        # if no inner_spec given just instantiate both inner and outer to the outer_y
        if inner_type_spec is None:
            return cls.from_instances(outer_y)

        inner_type = inner_type_spec.get('type')
        if inner_type is None:
            raise Exception("inner_type_spec needs a 'type' key.")

        # if an array, if shape is given in the spec, use that
        # if ndim == 1 (i.e. representing that the inner shape needs to be a vector)
        #   flatten outer_y and instantiate with those
        elif inner_type == 'array':
            outer_y_as_array = np.array(outer_y)

            shape = inner_type_spec.get('shape')
            if shape is not None:
                return cls.from_instances(outer_y_as_array.reshape(shape), outer_y)

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
            - rhs_funcs['generator'] - generator for a BMDE

        Need to implement:
            - rhs_funcs['rhs_jac'] - jacobian of rhs function, Jf(t)

        Assumptions:
            - For rhs_funcs['generator'], either inner_type == outer_type, or
              outer_type = {'type': 'array', 'shape': (d0,d1)} and
              inner_type = {'type': 'array', 'shape': (d0*d1,)}, i.e. the internal representation
              is the vectorized version of the outer

        Args:
            rhs_funcs (dict): contains various rhs functions
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

        # transform rhs_jac function
        rhs_jac = rhs_funcs.get('rhs_jac')

        if rhs_jac is not None:
            pass

        # transform generator
        generator = rhs_funcs.get('generator')

        if generator is not None:
            # With the above assumptions, returns a new generator function G' so that, for a
            # generator G and state y of outer_shape,
            # (Gy).reshape(inner_shape) = G' y.reshape(inner_shape)
            if self.inner_type_spec == self.outer_type_spec:
                new_rhs_funcs['generator'] = generator
            else:

                # raise exceptions based on assumptions
                if (self.inner_type_spec['type'] != 'array') or (self.outer_type_spec['type'] != 'array'):
                    raise Exception("""RHS generator transformation only valid for state types
                                       np.array.""")
                if len(self.inner_type_spec['shape']) != 1:
                    raise Exception("""RHS generator transformation only valid if inner_type is
                                       1d.""")

                def new_generator(t):
                    # create identity of size the second dimension fo the outer type
                    ident = np.eye(self.outer_type_spec['shape'][1])
                    return np.kron(generator(t), ident)

                new_rhs_funcs['generator'] = new_generator

        return new_rhs_funcs

def convert_state(y, type_spec):
    """Convert the de state y into the type specified by type_spec."""

    if type_spec['type'] == 'array':
        # default array data type to complex
        new_y = np.array(y, dtype=type_spec.get('dtype', 'complex'))

        shape = type_spec.get('shape')
        if shape is not None:
            return new_y.reshape(shape)
        else:
            return new_y

def type_spec_from_instance(y):
    """Determine type spec from an instance."""
    type_spec = {}
    if isinstance(y, np.ndarray):
        type_spec['type'] = 'array'
        type_spec['shape'] = y.shape

    return type_spec
