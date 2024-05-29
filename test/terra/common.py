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

import inspect
import logging
import os
import warnings
import unittest
from enum import Enum
from itertools import repeat
from math import pi
from random import choice, sample
from unittest.util import safe_repr

import fixtures
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import __path__ as main_path
from qiskit.quantum_info import Operator, Statevector
from qiskit.quantum_info.operators.predicates import matrix_equal
from .decorators import enforce_subclasses_call


class Path(Enum):
    """Helper with paths commonly used during the tests."""

    MAIN = main_path[0]
    TEST = os.path.dirname(__file__)
    # Examples path:    examples/
    EXAMPLES = os.path.join(MAIN, "../examples")


@enforce_subclasses_call(["setUp", "setUpClass", "tearDown", "tearDownClass"])
class BaseQiskitAerTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__setup_called = False
        self.__teardown_called = False

    def setUp(self):
        super().setUp()
        if self.__setup_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.setUp was already called. Do not explicitly call "
                "setUp from your tests. In your own setUp, use super to call "
                "the base setUp." % (sys.modules[self.__class__.__module__].__file__,)
            )
        self.__setup_called = True

    def tearDown(self):
        super().tearDown()
        if self.__teardown_called:
            raise ValueError(
                "In File: %s\n"
                "TestCase.tearDown was already called. Do not explicitly call "
                "tearDown from your tests. In your own tearDown, use super to "
                "call the base tearDown." % (sys.modules[self.__class__.__module__].__file__,)
            )
        self.__teardown_called = True

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.

        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertDictAlmostEqual(
        self, dict1, dict2, delta=None, msg=None, places=None, default_value=0
    ):
        """Assert two dictionaries with numeric values are almost equal.

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
            TypeError: if the arguments are not valid (both `delta` and
                `places` are specified).
            AssertionError: if the dictionaries are not almost equal.
        """

        error_msg = dicts_almost_equal(dict1, dict2, delta, places, default_value)

        if error_msg:
            msg = self._formatMessage(msg, error_msg)
            raise self.failureException(msg)


def dicts_almost_equal(dict1, dict2, delta=None, places=None, default_value=0):
    """Test if two dictionaries with numeric values are almost equal.

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
        places (int): number of decimal places for comparison.
        default_value (number): default value for missing keys.

    Raises:
        TypeError: if the arguments are not valid (both `delta` and
            `places` are specified).

    Returns:
        String: Empty string if dictionaries are almost equal. A description
            of their difference if they are deemed not almost equal.
    """

    def valid_comparison(value):
        """compare value to delta, within places accuracy"""
        if places is not None:
            return round(value, places) == 0
        else:
            return value < delta

    # Check arguments.
    if dict1 == dict2:
        return ""
    if places is not None:
        if delta is not None:
            raise TypeError("specify delta or places not both")
        msg_suffix = " within %s places" % places
    else:
        delta = delta or 1e-8
        msg_suffix = " within %s delta" % delta

    # Compare all keys in both dicts, populating error_msg.
    error_msg = ""
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key, default_value)
        val2 = dict2.get(key, default_value)
        if not valid_comparison(abs(val1 - val2)):
            error_msg += f"({safe_repr(key)}: {safe_repr(val1)} != {safe_repr(val2)}), "

    if error_msg:
        return error_msg[:-2] + msg_suffix
    else:
        return ""


class QiskitAerTestCase(BaseQiskitAerTestCase):
    """Helper class that contains common functionality."""

    def setUp(self):
        super().setUp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        allow_DeprecationWarning_modules = [
            "cvxpy",
        ]
        for mod in allow_DeprecationWarning_modules:
            warnings.filterwarnings("default", category=DeprecationWarning, module=mod)

        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)

        # Set logging to file and stdout if the LOG_LEVEL environment variable
        # is set.
        if os.getenv("LOG_LEVEL"):
            # Set up formatter.
            log_fmt = "{}.%(funcName)s:%(levelname)s:%(asctime)s:" " %(message)s".format(
                cls.__name__
            )
            formatter = logging.Formatter(log_fmt)

            # Set up the file handler.
            log_file_name = "%s.log" % cls.moduleName
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(os.getenv("LOG_LEVEL"), logging.INFO)
            cls.log.setLevel(level)

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertSuccess(self, result):
        """Assert that simulation executed without errors"""
        success = getattr(result, "success", False)
        msg = result.status
        if not success:
            for i, res in enumerate(getattr(result, "results", [])):
                if res.status != "DONE":
                    msg += ", (Circuit {}) {}".format(i, res.status)
        self.assertTrue(success, msg=msg)

    def assertNotSuccess(self, result):
        """Assert that simulation executed with errors"""
        success = getattr(result, "success", False)
        msg = result.status
        self.assertFalse(success, msg=msg)

    @staticmethod
    def gate_circuits(gate_cls, num_angles=0, has_ctrl_qubits=False, rng=None, basis_states=None):
        """
        Construct circuits from a gate class.
        Example of basis_states: ['010, '100'].
        When basis_states is None, tests all basis states
        with the gate's number of qubits.
        """
        if rng is None:
            rng = np.random.default_rng()

        if num_angles:
            params = list(rng.random(num_angles))
        else:
            params = []

        if has_ctrl_qubits:
            params.append(5)

        gate = gate_cls(*params)

        if basis_states is None:
            basis_states = [bin(i)[2:].zfill(gate.num_qubits) for i in range(1 << gate.num_qubits)]

        circs = []
        qubit_permutation = list(rng.permutation(gate.num_qubits))
        for state in basis_states:
            circ = QuantumCircuit(gate.num_qubits)
            for i in qubit_permutation:
                if state[i] == "1":
                    circ.x(i)

            circ.append(gate, qubit_permutation)
            circs.append(circ)

        return circs

    @staticmethod
    def check_position(obj, items, precision=15):
        """Return position of numeric object in a list."""
        for pos, item in enumerate(items):
            # Try numeric difference first
            try:
                delta = round(np.linalg.norm(np.array(obj) - np.array(item)), precision)
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

    @staticmethod
    def remove_if_found(obj, items, precision=15):
        """If obj is in list of items, remove first instance"""
        pos = QiskitAerTestCase.check_position(obj, items)
        if pos is not None:
            items.pop(pos)

    def compare_statevector(
        self, result, circuits, targets, ignore_phase=False, atol=1e-8, rtol=1e-5
    ):
        """Compare final statevectors to targets."""
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            target = Statevector(target)
            output = Statevector(result.get_statevector(circuit))
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = " {} != {}".format(output, target)
                delta = matrix_equal(
                    output.data, target.data, ignore_phase=ignore_phase, atol=atol, rtol=rtol
                )
                self.assertTrue(delta, msg=msg)

    def compare_unitary(self, result, circuits, targets, ignore_phase=False, atol=1e-8, rtol=1e-5):
        """Compare final unitary matrices to targets."""
        for pos, test_case in enumerate(zip(circuits, targets)):
            circuit, target = test_case
            target = Operator(target)
            output = Operator(result.get_unitary(circuit))
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = test_msg + " {} != {}".format(output.data, target.data)
                delta = matrix_equal(
                    output.data, target.data, ignore_phase=ignore_phase, atol=atol, rtol=rtol
                )
                self.assertTrue(delta, msg=msg)

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
                msg = test_msg + " {} != {}".format(output, target)
                self.assertDictAlmostEqual(output, target, delta=delta, msg=msg)

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
            metadata = getattr(result.results[0], "metadata")
            if metadata:
                value = metadata.get(key)
            test_msg = "Circuit ({}/{}):".format(pos + 1, len(circuits))
            with self.subTest(msg=test_msg):
                msg = " metadata {} value {} != {}".format(key, value, target)
                self.assertEqual(value, target, msg=msg)

    def assertDictAlmostEqual(
        self, dict1, dict2, delta=None, msg=None, places=None, default_value=0
    ):
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
            standard_msg = ""
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += "(%s: %s != %s), " % (
                        safe_repr(key),
                        safe_repr(val1),
                        safe_repr(val2),
                    )
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += "(%s: %s != %s), " % (
                        safe_repr(key),
                        safe_repr(val1),
                        safe_repr(val2),
                    )
            if success is True:
                return
            standard_msg = standard_msg[:-2] + " within %s places" % places

        else:
            if delta is None:
                delta = 1e-8  # default delta value
            success = True
            standard_msg = ""
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += "(%s: %s != %s), " % (
                        safe_repr(key),
                        safe_repr(val1),
                        safe_repr(val2),
                    )
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += "(%s: %s != %s), " % (
                        safe_repr(key),
                        safe_repr(val1),
                        safe_repr(val2),
                    )
            if success is True:
                return
            standard_msg = standard_msg[:-2] + " within %s delta" % delta

        msg = self._formatMessage(msg, standard_msg)
        raise self.failureException(msg)


def _is_ci_fork_pull_request():
    """
    Check if the tests are being run in a CI environment and if it is a pull
    request.

    Returns:
        bool: True if the tests are executed inside a CI tool, and the changes
            are not against the "main" branch.
    """
    if os.getenv("TRAVIS"):
        # Using Travis CI.
        if os.getenv("TRAVIS_PULL_REQUEST_BRANCH"):
            return True
    elif os.getenv("APPVEYOR"):
        # Using AppVeyor CI.
        if os.getenv("APPVEYOR_PULL_REQUEST_NUMBER"):
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
    qr = QuantumRegister(n_qubits, "qr")
    cr = ClassicalRegister(n_qubits, "cr")
    circuit = QuantumCircuit(qr, cr)

    for _ in repeat(None, n_gates):
        # Choose the next gate
        op_name = choice(gate_types)
        if op_name == "id":
            op_name = "iden"
        operation = eval("QuantumCircuit." + op_name)

        # Check if operation is one of u1, u2, u3
        if op_name[0] == "u" and op_name[1].isdigit():
            # Number of angles
            n_angles = int(op_name[1])
            # Number of qubits manipulated by the gate
            n_params = 1
        else:
            n_angles = 0
            if op_name == "measure":
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
        if op_name == "measure":
            classical_regs = [cr[i] for i in qubit_indices]
        else:
            classical_regs = []

        # Add operation to the circuit
        operation(circuit, *angles, *qubits, *classical_regs)

    return circuit
