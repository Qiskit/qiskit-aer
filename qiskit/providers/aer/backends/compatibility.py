# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Compatibility classes for changes to save instruction return types.

These subclasses include a `__getattr__` method that allows existing
code which used numpy.ndarray attributes to continue functioning with
a deprecation warning.

Numpy functions that consumed these classes should already work due to
them having an `__array__` method for implicit array conversion.
"""

import warnings
import numpy as np
import qiskit.quantum_info as qi


def _forward_attr(attr):
    """Return True if attribute should be passed to legacy class"""
    if attr[:2] == '__' or attr in ['_data', '_op_shape']:
        return False
    return True


class Statevector(qi.Statevector):
    """Aer result backwards compatibility wrapper for qiskit Statevector."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            warnings.warn(
                "The return type of saved statevectors has been changed from"
                " a `numpy.ndarray` to a `qiskit.quantum_info.Statevector` as"
                "of qiskit-aer 0.10. Accessing numpy array attributes is deprecated"
                " and will result in an error in a future release. Use the `.data`"
                " property to access the the stored ndarray for the Statevector.",
                DeprecationWarning, stacklevel=2)
            return getattr(self.data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other)) and
            self._op_shape == other._op_shape and
            np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )


class DensityMatrix(qi.DensityMatrix):
    """Aer result backwards compatibility wrapper for qiskit DensityMatrix."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            warnings.warn(
                "The return type of saved density matrices has been changed from"
                " a `numpy.ndarray` to a `qiskit.quantum_info.DensityMatrix` as"
                "of qiskit-aer 0.10. Accessing numpy array attributes is deprecated"
                " and will result in an error in a future release. Use the `.data`"
                " property to access the the stored ndarray for the DensityMatrix.",
                DeprecationWarning, stacklevel=2)
            return getattr(self.data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other)) and
            self._op_shape == other._op_shape and
            np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )


class Operator(qi.Operator):
    """Aer result backwards compatibility wrapper for qiskit Operator."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            warnings.warn(
                "The return type of saved unitaries has been changed from"
                " a `numpy.ndarray` to a `qiskit.quantum_info.Operator` as"
                "of qiskit-aer 0.10. Accessing numpy array attributes is deprecated"
                " and will result in an error in a future release. Use the `.data`"
                " property to access the the stored ndarray for the Operator.",
                DeprecationWarning, stacklevel=2)
            return getattr(self.data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other)) and
            self._op_shape == other._op_shape and
            np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )


class SuperOp(qi.SuperOp):
    """Aer result backwards compatibility wrapper for qiskit SuperOp."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            warnings.warn(
                "The return type of saved superoperators has been changed from"
                " a `numpy.ndarray` to a `qiskit.quantum_info.SuperOp` as"
                "of qiskit-aer 0.10. Accessing numpy array attributes is deprecated"
                " and will result in an error in a future release. Use the `.data`"
                " property to access the the stored ndarray for the SuperOp.",
                DeprecationWarning, stacklevel=2)
            return getattr(self.data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other)) and
            self._op_shape == other._op_shape and
            np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )


class StabilizerState(qi.StabilizerState):
    """Aer result backwards compatibility wrapper for qiskit StabilizerState."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(dict, attr):
            warnings.warn(
                "The return type of saved stabilizers has been changed from"
                " a `dict` to a `qiskit.quantum_info.StabilizerState` as of qiskit-aer 0.10."
                " Accessing dict attributes is deprecated and will result in an"
                " error in a future release. Use the `.clifford.to_dict()` methods to access "
                " the stored Clifford operator and convert to a dictionary.",
                DeprecationWarning, stacklevel=2)
            return getattr(self._data.to_dict(), attr)
        return getattr(super(), attr)

    def __getitem__(self, item):
        if item in ["stabilizer", "destabilizer"]:
            warnings.warn(
                "The return type of saved stabilizers has been changed from"
                " a `dict` to a `qiskit.quantum_info.StabilizerState` as of qiskit-aer 0.10."
                " Accessing dict items is deprecated and will result in an"
                " error in a future release. Use the `.clifford.to_dict()` methods to access "
                " the stored Clifford operator and convert to a dictionary.",
                DeprecationWarning, stacklevel=2)
            return self._data.to_dict()[item]
        raise TypeError("'StabilizerState object is not subscriptable'")

    def _add(self, other):
        raise NotImplementedError(f"{type(self)} does not support addition")

    def _multiply(self, other):
        raise NotImplementedError(f"{type(self)} does not support scalar multiplication")
