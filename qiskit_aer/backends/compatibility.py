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
    # Pass Iterable magic methods on to the Numpy array.  We can't pass
    # `__getitem__` (and consequently `__setitem__`, `__delitem__`) on because
    # `Statevector` implements them itself.
    if attr[:2] == "__" or attr in ["_data", "_op_shape"]:
        return False
    return True


def _deprecation_warning(instance, instances_name):
    class_name = instance.__class__.__name__
    warnings.warn(
        f"The return type of saved {instances_name} has been changed from"
        f" a `numpy.ndarray` to a `qiskit.quantum_info.{class_name}` as"
        " of qiskit-aer 0.10. Accessing numpy array attributes is deprecated"
        " and will result in an error in a future release. To continue using"
        " saved result objects as arrays you can explicitly cast them using "
        " `np.asarray(object)`.",
        DeprecationWarning,
        stacklevel=3,
    )


class Statevector(qi.Statevector):
    """Aer result backwards compatibility wrapper for qiskit Statevector."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            _deprecation_warning(self, "statevectors")
            return getattr(self._data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other))
            and self._op_shape == other._op_shape
            and np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )

    def __bool__(self):
        # Explicit override to the default behaviour for Python objects to
        # prevent the new `__len__` from messing with it.
        return True

    # Magic methods for the iterable/collection interface that need forwarding,
    # but bypass `__getattr__`.  `__getitem__` is defined by `Statevector`, so
    # that can't be forwarded (and consequently neither can `__setitem__`
    # without introducing an inconsistency).

    def __len__(self):
        _deprecation_warning(self, "statevectors")
        return self._data.__len__()

    def __iter__(self):
        _deprecation_warning(self, "statevectors")
        return self._data.__iter__()

    def __contains__(self, value):
        _deprecation_warning(self, "statevectors")
        return self._data.__contains__(value)

    def __reversed__(self):
        _deprecation_warning(self, "statevectors")
        return self._data.__reversed__()


class DensityMatrix(qi.DensityMatrix):
    """Aer result backwards compatibility wrapper for qiskit DensityMatrix."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            _deprecation_warning(self, "density matrices")
            return getattr(self._data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other))
            and self._op_shape == other._op_shape
            and np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )

    def __len__(self):
        _deprecation_warning(self, "density matrices")
        return self._data.__len__()

    def __iter__(self):
        _deprecation_warning(self, "density matrices")
        return self._data.__iter__()

    def __contains__(self, value):
        _deprecation_warning(self, "density matrices")
        return self._data.__contains__(value)

    def __reversed__(self):
        _deprecation_warning(self, "density matrices")
        return self._data.__reversed__()

    def __getitem__(self, key):
        _deprecation_warning(self, "density matrices")
        return self._data.__getitem__(key)

    def __setitem__(self, key, value):
        _deprecation_warning(self, "density matrices")
        return self._data.__setitem__(key, value)


class Operator(qi.Operator):
    """Aer result backwards compatibility wrapper for qiskit Operator."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            _deprecation_warning(self, "unitaries")
            return getattr(self._data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other))
            and self._op_shape == other._op_shape
            and np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )

    def __bool__(self):
        # Explicit override to the default behaviour for Python objects to
        # prevent the new `__len__` from messing with it.
        return True

    def __len__(self):
        _deprecation_warning(self, "unitaries")
        return self._data.__len__()

    def __iter__(self):
        _deprecation_warning(self, "unitaries")
        return self._data.__iter__()

    def __contains__(self, value):
        _deprecation_warning(self, "unitaries")
        return self._data.__contains__(value)

    def __reversed__(self):
        _deprecation_warning(self, "unitaries")
        return self._data.__reversed__()

    def __getitem__(self, key):
        _deprecation_warning(self, "unitaries")
        return self._data.__getitem__(key)

    def __setitem__(self, key, value):
        _deprecation_warning(self, "unitaries")
        return self._data.__setitem__(key, value)


class SuperOp(qi.SuperOp):
    """Aer result backwards compatibility wrapper for qiskit SuperOp."""

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(self._data, attr):
            _deprecation_warning(self, "superoperators")
            return getattr(self._data, attr)
        return getattr(super(), attr)

    def __eq__(self, other):
        return (
            isinstance(self, type(other))
            and self._op_shape == other._op_shape
            and np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)
        )

    def __bool__(self):
        # Explicit override to the default behaviour for Python objects to
        # prevent the new `__len__` from messing with it.
        return True

    def __len__(self):
        _deprecation_warning(self, "superoperators")
        return self._data.__len__()

    def __iter__(self):
        _deprecation_warning(self, "superoperators")
        return self._data.__iter__()

    def __contains__(self, value):
        _deprecation_warning(self, "superoperators")
        return self._data.__contains__(value)

    def __reversed__(self):
        _deprecation_warning(self, "superoperators")
        return self._data.__reversed__()

    def __getitem__(self, key):
        _deprecation_warning(self, "superoperators")
        return self._data.__getitem__(key)

    def __setitem__(self, key, value):
        _deprecation_warning(self, "superoperators")
        return self._data.__setitem__(key, value)


class StabilizerState(qi.StabilizerState):
    """Aer result backwards compatibility wrapper for qiskit StabilizerState."""

    def __deprecation_warning(self):
        warnings.warn(
            "The return type of saved stabilizers has been changed from"
            " a `dict` to a `qiskit.quantum_info.StabilizerState` as of qiskit-aer 0.10."
            " Accessing dict attributes is deprecated and will result in an"
            " error in a future release. Use the `.clifford.to_dict()` methods to access "
            " the stored Clifford operator and convert to a dictionary.",
            DeprecationWarning,
            stacklevel=3,
        )

    def __getattr__(self, attr):
        if _forward_attr(attr) and hasattr(dict, attr):
            self.__deprecation_warning()
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
                DeprecationWarning,
                stacklevel=2,
            )
            return self._data.to_dict()[item]
        raise TypeError("'StabilizerState object is not subscriptable'")

    def __bool__(self):
        # Explicit override to the default behaviour for Python objects to
        # prevent the new `__len__` from messing with it.
        return True

    def __len__(self):
        self.__deprecation_warning()
        return self._data.to_dict().__len__()

    def __iter__(self):
        self.__deprecation_warning()
        return self._data.to_dict().__iter__()

    def __contains__(self, value):
        self.__deprecation_warning()
        return self._data.to_dict().__contains__(value)

    def __reversed__(self):
        self.__deprecation_warning()
        return self._data.to_dict().__reversed__()

    def _add(self, other):
        raise NotImplementedError(f"{type(self)} does not support addition")

    def _multiply(self, other):
        raise NotImplementedError(f"{type(self)} does not support scalar multiplication")
