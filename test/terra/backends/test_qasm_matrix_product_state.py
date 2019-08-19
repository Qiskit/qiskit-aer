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
Matrix Product State Integration Tests
"""

import unittest
from test.terra import common
from test.terra.backends.qasm_simulator.matrix_product_state_method import QasmMatrixProductStateMethodTests
from test.terra.backends.qasm_simulator.matrix_product_state_measure import QasmMatrixProductStateMeasureTests


class TestQasmMatrixProductStateSimulator(common.QiskitAerTestCase,
                                   QasmMatrixProductStateMethodTests,
                                   QasmMatrixProductStateMeasureTests):

    BACKEND_OPTS = {"method": "matrix_product_state"}

if __name__ == '__main__':
    unittest.main()
