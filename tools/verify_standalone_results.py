# -*- coding: utf-8 -*-

# Copyright 2021, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import json
import os
import sys

from qiskit.result import Result


def assertDictAlmostEqual(dict1, dict2, delta=None, msg=None, places=None, default_value=0):
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
                standard_msg += "(%s: %s != %s), " % (key, val1, val2)
        # check values for keys in counts, not in target
        keys2 = set(dict2.keys()) - keys1
        for key in keys2:
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if round(abs(val1 - val2), places) != 0:
                success = False
                standard_msg += "(%s: %s != %s), " % (key, val1, val2)
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
                standard_msg += "(%s: %s != %s), " % (key, val1, val2)
        # check values for keys in counts, not in target
        keys2 = set(dict2.keys()) - keys1
        for key in keys2:
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if abs(val1 - val2) > delta:
                success = False
                standard_msg += "(%s: %s != %s), " % (key, val1, val2)
        if success is True:
            return
        standard_msg = standard_msg[:-2] + " within %s delta" % delta

    raise Exception(standard_msg)


def compare_counts(result, target, delta=0):
    """Compare counts dictionary to targets."""
    # Don't use get_counts method which converts hex
    output = result.data(0)["counts"]
    assertDictAlmostEqual(output, target, delta=delta)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "rt") as fp:
            result_dict = json.load(fp)
    else:
        result_dict = json.load(sys.stdin)

    result = Result.from_dict(result_dict)
    assert result.status == "COMPLETED"
    assert result.success is True
    if os.getenv("USE_MPI", False):
        assert result.metadata["num_mpi_processes"] > 1
    shots = result.results[0].shots
    targets = {"0x0": 5 * shots / 8, "0x1": shots / 8, "0x2": shots / 8, "0x3": shots / 8}
    compare_counts(result, targets, delta=0.05 * shots)
    print("Input result JSON is valid!")
