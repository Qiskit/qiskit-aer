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
Tensor Network Integration Tests
"""

import unittest
from test.terra import common
from test.terra.backends.qasm_simulator.tensor_network_method import QasmTensorNetworkMethodTests
from test.terra.backends.qasm_simulator.tensor_network_measure import QasmTensorNetworkMeasureTests


class TestQasmTensorNetworkSimulator(common.QiskitAerTestCase,
                                   QasmTensorNetworkMethodTests,
                                   QasmTensorNetworkMeasureTests):

    BACKEND_OPTS = {"method": "tensor_network"}

if __name__ == '__main__':
    unittest.main()
