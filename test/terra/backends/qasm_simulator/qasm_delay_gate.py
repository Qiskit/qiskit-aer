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
QasmSimulator Integration Tests
"""

from test.terra.reference import ref_1q_clifford
from qiskit.providers.aer import QasmSimulator
from qiskit import assemble

import numpy as np

class QasmDelayGateTests:
    """QasmSimulator delay gate tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_delay_gate(self):
        """Test delay gate circuits"""
        method = self.BACKEND_OPTS.get('method', 'automatic')
        self.SIMULATOR.set_options(method=method)
        
        shots = 100
        circuits = ref_1q_clifford.delay_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.delay_gate_counts_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots,
                        **self.BACKEND_OPTS)
        result = self.SIMULATOR.run(qobj).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)
