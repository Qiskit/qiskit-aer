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
UnitarySimulator Integration Tests
"""

import unittest
from test.terra import common
from test.terra.decorators import requires_gpu
# Basic circuit instruction tests
from test.terra.backends.unitary_simulator.unitary_basics import UnitaryBasicsTests

@requires_gpu
class TestUnitaryGPUSimulator(common.QiskitAerTestCase,
                           UnitaryBasicsTests):
    """QasmSimulator automatic method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 2113,
        "method": "unitarymatrix_gpu"
    }

if __name__ == '__main__':
    unittest.main()
