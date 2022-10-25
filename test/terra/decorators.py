# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Decorator for using with Qiskit Aer unit tests."""

import multiprocessing
import unittest

from qiskit import QuantumCircuit, execute

from qiskit_aer import AerProvider, AerSimulator

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


def is_method_available(backend, method):
    """Check if input method is available for the qasm simulator."""
    if isinstance(backend, str):
        backend = AerProvider().get_backend(backend)
    avail = backend.available_methods()
    if method in avail:
        return True
    else:
        return False


def requires_method(backend, method):
    """Decorator that skips test if a simulation method is unavailable.

    Args:
        backend (str or AerBackend): backend to check method for.
        method (str): the method string

    Returns:
        decorator: the decorator for testing input method.
    """
    reason = 'method "{}" is unavailable, skipping test'.format(method)
    skip = not is_method_available(backend, method)
    return unittest.skipIf(skip, reason)


def requires_omp(test_item):
    """Decorator that skips test if OpenMP is not available.

    Args:
        test_item (callable): function or class to be decorated.

    Returns:
        callable: the decorated function.
    """
    # Run dummy circuit to check OpenMP status
    result = AerSimulator().run(QuantumCircuit(1)).result()
    omp_enabled = result.metadata.get('omp_enabled', False)
    skip = not omp_enabled
    reason = 'OpenMP not available, skipping test'
    return unittest.skipIf(skip, reason)(test_item)


def requires_multiprocessing(test_item):
    """Decorator that skips test if run on single-core CPU.

    Args:
        test_item (callable): function or class to be decorated.

    Returns:
        callable: the decorated function.
    """
    skip = multiprocessing.cpu_count() <= 1
    reason = 'Multicore CPU not available, skipping test'
    return unittest.skipIf(skip, reason)(test_item)


def deprecated(method):
    """Decorator that is for deprecated methods.

    Args:
        method (callable): a deprecated method.

    Returns:
        callable: the decorated method.
    """

    def _deprecated_method(self, *args, **kwargs):
        with self.assertWarns(DeprecationWarning):
            method(self, *args, **kwargs)

    return _deprecated_method
