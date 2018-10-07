# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Shared functionality and helpers for the unit tests.
"""

from enum import Enum
import functools
import inspect
import logging
import os
import unittest
from unittest.util import safe_repr
import numpy as np
from itertools import repeat
from random import choice, sample
from math import pi

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit)
from qiskit_addon_qv import __path__ as main_path


class Path(Enum):
    """Helper with paths commonly used during the tests."""
    MAIN = main_path[0]
    TEST = os.path.dirname(__file__)
    # Examples path:    examples/
    EXAMPLES = os.path.join(MAIN, '../examples')


class QiskitAerTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)

        # Set logging to file and stdout if the LOG_LEVEL environment variable
        # is set.
        if os.getenv('LOG_LEVEL'):
            # Set up formatter.
            log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                       ' %(message)s'.format(cls.__name__))
            formatter = logging.Formatter(log_fmt)

            # Set up the file handler.
            log_file_name = '%s.log' % cls.moduleName
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(os.getenv('LOG_LEVEL'),
                                             logging.INFO)
            cls.log.setLevel(level)

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """ Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertNoLogs(self, logger=None, level=None):
        """
        Context manager to test that no message is sent to the specified
        logger and level (the opposite of TestCase.assertLogs()).
        """
        # pylint: disable=invalid-name
        return _AssertNoLogsContext(self, logger, level)

    def assertDictAlmostEqual(self, dict1, dict2, delta=None, msg=None,
                              places=None, default_value=0):
        """
        Assert two dictionaries with numeric values are almost equal.

        Fail if the two dictionaries are unequal as determined by
        comparing that the difference between values with the same key are
        not greater than delta (default 1e-8), or that difference rounded
        to the given number of decimal places is not zero. If a key in one
        dictionary is not in the other the default_value keyword argument
        will be used for the missing value (default 0). If the two objects
        compare equal then they will automatically compare almost equal.

        Args:
            dict1 (dict): a dictionary.
            dict2 (dict): a dictionary.
            delta (number): threshold for comparison (defaults to 1e-8).
            msg (str): return a custom message on failure.
            places (int): number of decimal places for comparison.
            default_value (number): default value for missing keys.

        Raises:
            TypeError: raises TestCase failureException if the test fails.
        """
        if dict1 == dict2:
            # Shortcut
            return
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        if places is not None:
            success = True
            standard_msg = ''
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            if success is True:
                return
            standard_msg = standard_msg[:-2] + ' within %s places' % places

        else:
            if delta is None:
                delta = 1e-8  # default delta value
            success = True
            standard_msg = ''
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            if success is True:
                return
            standard_msg = standard_msg[:-2] + ' within %s delta' % delta

        msg = self._formatMessage(msg, standard_msg)
        raise self.failureException(msg)


class _AssertNoLogsContext(unittest.case._AssertLogsContext):
    """A context manager used to implement TestCase.assertNoLogs()."""

    # pylint: disable=inconsistent-return-statements
    def __exit__(self, exc_type, exc_value, tb):
        """
        This is a modified version of TestCase._AssertLogsContext.__exit__(...)
        """
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return False

        if self.watcher.records:
            msg = 'logs of level {} or higher triggered on {}:\n'.format(
                logging.getLevelName(self.level), self.logger.name)
            for record in self.watcher.records:
                msg += 'logger %s %s:%i: %s\n' % (record.name, record.pathname,
                                                  record.lineno,
                                                  record.getMessage())

            self._raiseFailure(msg)

def _is_ci_fork_pull_request():
    """
    Check if the tests are being run in a CI environment and if it is a pull
    request.

    Returns:
        bool: True if the tests are executed inside a CI tool, and the changes
            are not against the "master" branch.
    """
    if os.getenv('TRAVIS'):
        # Using Travis CI.
        if os.getenv('TRAVIS_PULL_REQUEST_BRANCH'):
            return True
    elif os.getenv('APPVEYOR'):
        # Using AppVeyor CI.
        if os.getenv('APPVEYOR_PULL_REQUEST_NUMBER'):
            return True
    return False


def generate_random_circuit(n_qubits, n_gates, gate_types):
    """
    Generation of a random circuit has a history in Qiskit.
    Terra used to have a file _random_circuit_generator.py, but it is not there anymore.
    This file was located in folder `test`, hence accessible only to Qiskit developers and not to users.
    Currently, as far as I know, each test that requires random circuits has its own implementation of a random circuit generator.
    This includes tests in qiskit-addon-sympy and test_visualization in terra.
    Aqua had an issue of writing a random circuit generator, which was closed with the justification that it is moved to ignes.
    """
    qr = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qr)

    for _ in repeat(None, n_gates):

        # Choose the next gate
        op_name = choice(gate_types)
        op = eval('QuantumCircuit.' + op_name)

        # Check if op is one of u1, u2, u3
        if op_name[0] == 'u' and op_name[1].isdigit():
            # Number of angles
            n_angles = int(op_name[1])
            # Number of qubits manipulated by the gate
            n_params = 1
        else:
            n_angles = 0
            n_params = len(inspect.signature(op).parameters) - 1

        # Choose qubits
        qubit_indices = sample(range(n_qubits), n_params)
        qubits = [qr[i] for i in qubit_indices]

        # Choose angles
        angles = np.random.rand(n_angles)*pi

        # Add operation to the circuit
        op(circuit, *angles, *qubits)

    return circuit
