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
from .qasm_save_statevector_dict import QasmSaveStatevectorDictTests
from .qasm_save_density_matrix import QasmSaveDensityMatrixTests
from .qasm_save_stabilizer import QasmSaveStabilizerTests
from .qasm_save_probabilities import QasmSaveProbabilitiesTests
from .qasm_save_amplitudes import QasmSaveAmplitudesTests
from .qasm_save_matrix_product_state import QasmSaveMatrixProductStateTests
from .qasm_save_state import QasmSaveStateTests


class QasmSaveDataTests(QasmSaveExpectationValueTests,
                        QasmSaveStatevectorTests,
                        QasmSaveStatevectorDictTests,
                        QasmSaveDensityMatrixTests,
                        QasmSaveStabilizerTests,
                        QasmSaveProbabilitiesTests,
                        QasmSaveAmplitudesTests,
                        QasmSaveMatrixProductStateTests,
                        QasmSaveStateTests):

    """QasmSimulator SaveData instruction tests."""
