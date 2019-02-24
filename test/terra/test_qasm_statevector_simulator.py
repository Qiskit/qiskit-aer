# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

import unittest
from test.terra.utils.qasm_simulator.qasm_method import QasmMethodTests
from test.terra.utils.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.utils.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.utils.qasm_simulator.qasm_conditional import QasmConditionalTests
from test.terra.utils.qasm_simulator.qasm_cliffords import QasmCliffordTests
from test.terra.utils.qasm_simulator.qasm_cliffords import QasmCliffordTestsWaltzBasis
from test.terra.utils.qasm_simulator.qasm_cliffords import QasmCliffordTestsMinimalBasis
from test.terra.utils.qasm_simulator.qasm_noncliffords import QasmNonCliffordTests
from test.terra.utils.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsWaltzBasis
from test.terra.utils.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsMinimalBasis
from test.terra.utils.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
from test.terra.utils.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsWaltzBasis
from test.terra.utils.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsMinimalBasis
from test.terra.utils.qasm_simulator.qasm_extra import QasmExtraTests


class TestQasmStatevectorSimulator(QasmMethodTests,
                                   QasmMeasureTests,
                                   QasmResetTests,
                                   QasmConditionalTests,
                                   QasmCliffordTests,
                                   QasmCliffordTestsWaltzBasis,
                                   QasmCliffordTestsMinimalBasis,
                                   QasmNonCliffordTests,
                                   QasmNonCliffordTestsWaltzBasis,
                                   QasmNonCliffordTestsMinimalBasis,
                                   QasmAlgorithmTests,
                                   QasmAlgorithmTestsWaltzBasis,
                                   QasmAlgorithmTestsMinimalBasis,
                                   QasmExtraTests):
    """QasmSimulator statevector method tests."""

    BACKEND_OPTS = {"method": "statevector"}


if __name__ == '__main__':
    unittest.main()
