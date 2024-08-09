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


import unittest
import copy
import pickle
from multiprocessing import Pool

from qiskit import transpile, QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit_aer.backends.controller_wrappers import aer_controller_execute
from qiskit_aer.backends.backend_utils import LIBRARY_DIR
from test.terra.common import QiskitAerTestCase


class TestControllerExecuteWrappers(QiskitAerTestCase):
    """Basic functionality tests for pybind-generated wrappers"""

    CFUNCS = [aer_controller_execute()]

    def test_deepcopy(self):
        """Test that the functors are deepcopy-able."""
        for cfunc in self.CFUNCS:
            cahpy = copy.deepcopy(cfunc)

    def test_pickleable(self):
        """Test that the functors are pickle-able (directly)."""
        for cfunc in self.CFUNCS:
            bites = pickle.dumps(cfunc)
            cahpy = pickle.loads(bites)


if __name__ == "__main__":
    unittest.main()
