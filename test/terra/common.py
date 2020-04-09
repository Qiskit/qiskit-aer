# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Shared functionality and helpers for the unit tests.
"""

from enum import Enum

import inspect
import logging
import os
import unittest
from unittest.util import safe_repr
from itertools import repeat
from random import choice, sample
from math import pi
import numpy as np
from numpy.linalg import norm

from qiskit.quantum_info import state_fidelity
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import __path__ as main_path


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

    def check_position(self, obj, items, precision=15):
        """Return position of numeric object in a list."""
        for pos, item in enumerate(items):
            # Try numeric difference first
            try:
                delta = round(np.linalg.norm(np.array(obj) - np.array(item)),
                              precision)
                if delta == 0:
                    return pos
            # If objects aren't numeric try direct equality comparison
            except:
                try:
                    if obj == item:
                        return pos
                except:
                    return None
        return None

    def remove_if_found(self, obj, items, precision=15):
        """If obj is in list of items, remove first instance"""
        pos = self.check_position(obj, items)
        if pos is not None:
            items.pop(pos)

    def compare_statevector(self, result, circuits, targets,
                            global_phase=True, places=None):
        """Compare final statevectors to targets."""
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            output = result.get_statevector(circuit)
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = " {} != {}".format(output, target)
                if global_phase:
                    # Test equal including global phase
                    self.assertAlmostEqual(
                        norm(output - target), 0, places=places,
                        msg=msg)
                else:
                    # Test equal ignorning global phase
                    self.assertAlmostEqual(
                        state_fidelity(output, target) - 1, 0, places=places,
                        msg=msg + " up to global phase")

    def compare_unitary(self, result, circuits, targets,
                        global_phase=True, places=None):
        """Compare final unitary matrices to targets."""
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            output = result.get_unitary(circuit)
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = "\n{}\n {} != {}".format(circuit, output, target)
                if global_phase:
                    # Test equal including global phase
                    self.assertAlmostEqual(
                        norm(output - target), 0, places=places, msg=msg)
                else:
                    # Test equal ignorning global phase
                    delta = np.trace(np.dot(
                        np.conj(np.transpose(output)), target)) - len(output)
                    self.assertAlmostEqual(
                        delta, 0, places=places, msg=msg + " up to global phase")

    def compare_counts(self, result, circuits, targets, hex_counts=True, delta=0):
        """Compare counts dictionary to targets."""
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            if hex_counts:
                # Don't use get_counts method which converts hex
                output = result.data(circuit)["counts"]
            else:
                # Use get counts method which converts hex
                output = result.get_counts(circuit)
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = " {} != {}".format(output, target)
                self.assertDictAlmostEqual(
                    output, target, delta=delta, msg=msg)

    def compare_memory(self, result, circuits, targets, hex_counts=True):
        """Compare memory list to target."""
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            self.assertIn("memory", result.data(circuit))
            if hex_counts:
                # Don't use get_counts method which converts hex
                output = result.data(circuit)["memory"]
            else:
                # Use get counts method which converts hex
                output = result.get_memory(circuit)
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = " {} != {}".format(output, target)
                self.assertEqual(output, target, msg=msg)

    def compare_result_metadata(self, result, circuits, key, targets):
        """Compare result metadata key value."""
        if not isinstance(targets, (list, tuple)):
            targets = len(circuits) * [targets]
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            value = None
            metadata = getattr(result.results[0], 'metadata')
            if metadata:
                value = metadata.get(key)
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = " metadata {} value {} != {}".format(key, value, target)
                self.assertEqual(value, target, msg=msg)

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
    This file was located in folder `test`,
    hence accessible only to Qiskit developers and not to users.
    Currently, as far as I know, each test that requires random circuits has its own
    implementation of a random circuit generator.
    This includes tests in qiskit-addon-sympy and test_visualization in terra.
    Aqua had an issue of writing a random circuit generator, which was closed
    with the justification that it is moved to ignes.
    """
    qr = QuantumRegister(n_qubits, 'qr')
    cr = ClassicalRegister(n_qubits, 'cr')
    circuit = QuantumCircuit(qr, cr)

    for _ in repeat(None, n_gates):

        # Choose the next gate
        op_name = choice(gate_types)
        if op_name == 'id':
            op_name = 'iden'
        operation = eval('QuantumCircuit.' + op_name)

        # Check if operation is one of u1, u2, u3
        if op_name[0] == 'u' and op_name[1].isdigit():
            # Number of angles
            n_angles = int(op_name[1])
            # Number of qubits manipulated by the gate
            n_params = 1
        else:
            n_angles = 0
            if op_name == 'measure':
                n_params = 1
            else:
                n_params = len(inspect.signature(operation).parameters) - 1

        # Choose qubits
        qubit_indices = sample(range(n_qubits), n_params)
        qubits = [qr[i] for i in qubit_indices]

        # Choose angles
        angles = np.random.rand(n_angles) * pi

        # Measurement operation
        # In all measure operations, the classical register is not random,
        # but has the same index as the quantum register
        if op_name == 'measure':
            classical_regs = [cr[i] for i in qubit_indices]
        else:
            classical_regs = []

        # Add operation to the circuit
        operation(circuit, *angles, *qubits, *classical_regs)

    return circuit
