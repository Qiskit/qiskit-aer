# This code is part of Qiskit.
#
# (C) Copyright IBM Corp. 2017 and later.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
QasmSimulator Integration Tests
"""

import unittest
from test.terra import common
from test.terra.backends.qasm_simulator.qasm_method import QasmMethodTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTests
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_extra import QasmExtraTests


class TestQasmStatevectorSimulator(common.QiskitAerTestCase,
                                   QasmMethodTests,
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
