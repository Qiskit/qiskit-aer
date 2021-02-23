# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QasmSimulator Integration Tests for Save instructions
"""

from .qasm_save_expval import QasmSaveExpectationValueTests
from .qasm_save_statevector import QasmSaveStatevectorTests
from .qasm_save_density_matrix import QasmSaveDensityMatrixTests
from .qasm_save_stabilizer import QasmSaveStabilizerTests
from .qasm_save_probabilities import QasmSaveProbabilitiesTests
from .qasm_save_amplitudes import QasmSaveAmplitudesTests


class QasmSaveDataTests(QasmSaveExpectationValueTests,
                        QasmSaveStatevectorTests,
                        QasmSaveDensityMatrixTests,
                        QasmSaveStabilizerTests,
                        QasmSaveProbabilitiesTests,
                        QasmSaveAmplitudesTests):
    """QasmSimulator SaveData instruction tests."""
