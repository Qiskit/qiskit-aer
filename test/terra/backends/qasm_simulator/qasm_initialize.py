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

from qiskit import QuantumCircuit
from test.terra.reference import ref_initialize
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator

import numpy as np

class QasmInitializeTests:
    """QasmSimulator initialize tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test initialize instr make it through the wrapper
    # ---------------------------------------------------------------------
    def test_initialize_wrapper_1(self):
        """Test QasmSimulator initialize"""
        shots = 100
        lst = [0, 1]
        init_states = [
            np.array(lst),
            np.array(lst, dtype=float),
            np.array(lst, dtype=np.float32),
            np.array(lst, dtype=complex),
            np.array(lst, dtype=np.complex64) ]
        circuits = []
        [ circuits.extend(ref_initialize.initialize_circuits_w_1(init_state)) for init_state in init_states ]
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
 
    # ---------------------------------------------------------------------
    # Test initialize instr make it through the wrapper
    # ---------------------------------------------------------------------
    def test_initialize_wrapper_2(self):
        """Test QasmSimulator initialize"""
        shots = 100
        lst = [0, 1, 0, 0]
        init_states = [
            np.array(lst),
            np.array(lst, dtype=float),
            np.array(lst, dtype=np.float32),
            np.array(lst, dtype=complex),
            np.array(lst, dtype=np.complex64) ]
        circuits = []
        [ circuits.extend(ref_initialize.initialize_circuits_w_2(init_state)) for init_state in init_states ]
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)

    # ---------------------------------------------------------------------
    # Test initialize
    # ---------------------------------------------------------------------
    def test_initialize_1(self):
        """Test QasmSimulator initialize"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 1000
        circuits = ref_initialize.initialize_circuits_1(final_measure=True)
        targets = ref_initialize.initialize_counts_1(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        opts = self.BACKEND_OPTS.copy()
        result = self.SIMULATOR.run(qobj, **opts).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_initialize_2(self):
        """Test QasmSimulator initialize"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 1000
        circuits = ref_initialize.initialize_circuits_2(final_measure=True)
        targets = ref_initialize.initialize_counts_2(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        opts = self.BACKEND_OPTS.copy()
        result = self.SIMULATOR.run(qobj, **opts).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_initialize_sampling_opt(self):
        """Test sampling optimization"""
        shots = 1000
        circuits = ref_initialize.initialize_sampling_optimization()
        targets = ref_initialize.initialize_counts_sampling_optimization(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        opts = self.BACKEND_OPTS.copy()
        result = self.SIMULATOR.run(qobj, **opts).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_initialize_entangled_qubits(self):
        """Test initialize entangled qubits"""
        shots = 1000
        circuits = ref_initialize.initialize_entangled_qubits()
        targets = ref_initialize.initialize_counts_entangled_qubits(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        opts = self.BACKEND_OPTS.copy()
        result = self.SIMULATOR.run(qobj, **opts).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_initialize_sampling_opt_enabled(self):
        """Test sampling optimization"""
        shots = 1000
        circuit = QuantumCircuit(2)
        circuit.initialize([0, 1], [1])
        circuit.h([0, 1])
        circuit.initialize([0, 0, 1, 0], [0, 1])
        circuit.measure_all()
        qobj = assemble(circuit, self.SIMULATOR, shots=shots)
        opts = self.BACKEND_OPTS.copy()
        result = self.SIMULATOR.run(qobj, **opts).result()
        self.assertSuccess(result)
        sampling = result.results[0].metadata.get('measure_sampling', None)
        self.assertTrue(sampling)

    def test_initialize_sampling_opt_disabled(self):
        """Test sampling optimization"""
        shots = 1000
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.initialize([0, 1], [1])
        circuit.measure_all()
        qobj = assemble(circuit, self.SIMULATOR, shots=shots)
        opts = self.BACKEND_OPTS.copy()
        result = self.SIMULATOR.run(qobj, **opts).result()
        self.assertSuccess(result)
        sampling = result.results[0].metadata.get('measure_sampling', None)
        self.assertFalse(sampling)
