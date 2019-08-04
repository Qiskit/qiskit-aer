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
Matrix product state integration tests
This set of tests runs all the circuits from test/terra/references/: ref_1q_clifford and ref_2q_clifford,
with the exception of those with the multiplexer gate which is not supported yet.
"""

import unittest
import logging
import pprint

from test.terra import common
from test.terra.reference import ref_1q_clifford, ref_2q_clifford

from qiskit import *
from qiskit.providers.aer import QasmSimulator
#from qiskit.extensions.simulator import Snapshot

logger = logging.getLogger(__name__)

class QasmMatrixProductStateMethodTests:
    """QasmSimulator matrix_product_state method tests."""

    BACKEND_OPTS = {"method": "matrix_product_state"}
    
    def test_method_deterministic_without_sampling(self):
        """Test matrix product state method with deterministic counts without sampling"""
        deterministic_tests = [(ref_1q_clifford.h_gate_circuits_deterministic, ref_1q_clifford.h_gate_counts_deterministic),
             (ref_1q_clifford.x_gate_circuits_deterministic, ref_1q_clifford.x_gate_counts_deterministic),
             (ref_1q_clifford.z_gate_circuits_deterministic, ref_1q_clifford.z_gate_counts_deterministic),
             (ref_1q_clifford.y_gate_circuits_deterministic, ref_1q_clifford.y_gate_counts_deterministic),
             (ref_1q_clifford.s_gate_circuits_deterministic, ref_1q_clifford.s_gate_counts_deterministic),
             (ref_1q_clifford.sdg_gate_circuits_deterministic, ref_1q_clifford.sdg_gate_counts_deterministic),
             (ref_2q_clifford.cx_gate_circuits_deterministic,ref_2q_clifford.cx_gate_counts_deterministic),
             (ref_2q_clifford.cz_gate_circuits_deterministic, ref_2q_clifford.cz_gate_counts_deterministic),
             (ref_2q_clifford.swap_gate_circuits_deterministic, ref_2q_clifford.swap_gate_counts_deterministic)]

        determ_test_list = {'list':deterministic_tests, 'shots':100, 'delta':0}

        nondeterministic_tests = [(ref_1q_clifford.h_gate_circuits_nondeterministic, ref_1q_clifford.h_gate_counts_nondeterministic),
             (ref_1q_clifford.s_gate_circuits_nondeterministic, ref_1q_clifford.s_gate_counts_nondeterministic),
             (ref_1q_clifford.sdg_gate_circuits_nondeterministic, ref_1q_clifford.sdg_gate_counts_nondeterministic),
             (ref_2q_clifford.cx_gate_circuits_nondeterministic, ref_2q_clifford.cx_gate_counts_nondeterministic),
             (ref_2q_clifford.cz_gate_circuits_nondeterministic, ref_2q_clifford.cz_gate_counts_nondeterministic),
             (ref_2q_clifford.swap_gate_circuits_nondeterministic, ref_2q_clifford.swap_gate_counts_nondeterministic)]

        nondeterm_test_list = {'list':nondeterministic_tests, 'shots':2000, 'delta':0.05}

        test_list = [determ_test_list, nondeterm_test_list]
        for list in test_list:
            for test in list['list']:
                delta = list['delta']
                shots = list['shots']
                
                circuits = test[0](final_measure=True)
                targets  = test[1](shots)
                job = execute(circuits, QasmSimulator(), backend_options=self.BACKEND_OPTS, shots=shots)
                result = job.result()
                self.is_completed(result)
                self.compare_counts(result, circuits, targets, delta = delta*shots)

