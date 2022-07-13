# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests for circuit library standard gates
"""

from ddt import ddt
from numpy.random import default_rng
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)

from qiskit import transpile
import qiskit.quantum_info as qi

from qiskit.circuit.library.standard_gates import (
    CXGate, CYGate, CZGate, DCXGate, HGate, IGate, SGate, SXGate, SXdgGate,
    SdgGate, SwapGate, XGate, YGate, ZGate, TGate, TdgGate, iSwapGate, C3XGate,
    C4XGate, CCXGate, CHGate, CSXGate, CSwapGate, CPhaseGate, CRXGate, CRYGate,
    CRZGate, CUGate, CU1Gate, CU3Gate, CUGate, PhaseGate, RC3XGate, RCCXGate, RGate,
    RXGate, RXXGate, RYGate, RYYGate, RZGate, RZXGate, RZZGate, UGate, U1Gate, U2Gate,
    U3Gate, UGate, MCXGate, MCPhaseGate, MCXGrayCode)


CLIFFORD_GATES = [
    # Clifford Gates
    (CXGate, 0, False),
    (CYGate, 0, False),
    (CZGate, 0, False),
    (DCXGate, 0, False),
    (HGate, 0, False),
    (IGate, 0, False),
    (SGate, 0, False),
    (SXGate, 0, False),
    (SXdgGate, 0, False),
    (SdgGate, 0, False),
    (SwapGate, 0, False),
    (XGate, 0, False),
    (YGate, 0, False),
    (ZGate, 0, False),
]

NONCLIFFORD_GATES = [
    # Non-Clifford Gates
    (TGate, 0, False),
    (TdgGate, 0, False),
    (iSwapGate, 0, False),
    (CCXGate, 0, False),
    (CHGate, 0, False),
    (CSXGate, 0, False),
    (CSwapGate, 0, False),
    # Parameterized Gates
    (CPhaseGate, 1, False),
    (CRXGate, 1, False),
    (CRYGate, 1, False),
    (CRZGate, 1, False),
    (CU1Gate, 1, False),
    (CU3Gate, 3, False),
    (CUGate, 4, False),
    (PhaseGate, 1, False),
    (RGate, 2, False),
    (RXGate, 1, False),
    (RXXGate, 1, False),
    (RYGate, 1, False),
    (RYYGate, 1, False),
    (RZGate, 1, False),
    (RZXGate, 1, False),
    (RZZGate, 1, False),
    (U1Gate, 1, False),
    (U2Gate, 2, False),
    (U3Gate, 3, False),
    (UGate, 3, False),
]

MC_GATES = [
    (C3XGate, 0, False),
    (C4XGate, 0, False),
    # Parameterized Gates
    (RC3XGate, 0, False),
    (RCCXGate, 0, False),
    (MCXGate, 0, True),
    (MCPhaseGate, 1, True),
    (MCXGrayCode, 0, True),
]

# Convert to label keys for nicer display format with ddt
CLIFFORD_GATES_DICT = {i[0].__name__: i for i in CLIFFORD_GATES}
NONCLIFFORD_GATES_DICT = {i[0].__name__: i for i in NONCLIFFORD_GATES}
MC_GATES_DICT = {i[0].__name__: i for i in MC_GATES}


@ddt
class TestGates(SimulatorTestCase):
    """Test standard gate library."""

    SEED = 8181
    RNG = default_rng(seed=SEED)

    def _test_gate(self, gate, gates_dict, **options):
        """Test standard gates."""

        backend = self.backend(**options)

        gate_cls, num_angles, has_ctrl_qubits = gates_dict[gate]
        circuits = self.gate_circuits(gate_cls,
                                      num_angles=num_angles,
                                      has_ctrl_qubits=has_ctrl_qubits,
                                      rng=self.RNG)

        label = 'final'
        method = backend.options.method
        for circuit in circuits:
            if method == 'density_matrix':
                target = qi.DensityMatrix(circuit)
                circuit.save_density_matrix(label=label)
                fidelity_fn = qi.state_fidelity
            elif method == 'stabilizer':
                target = qi.StabilizerState(qi.Clifford(circuit))
                circuit.save_stabilizer(label=label)
                fidelity_fn = qi.process_fidelity
            elif method == 'unitary':
                target = qi.Operator(circuit)
                circuit.save_unitary(label=label)
                fidelity_fn = qi.process_fidelity
            elif method == 'superop':
                target = qi.SuperOp(circuit)
                circuit.save_superop(label=label)
                fidelity_fn = qi.process_fidelity
            else:
                target = qi.Statevector(circuit)
                circuit.save_statevector(label=label)
                fidelity_fn = qi.state_fidelity

            result = backend.run(transpile(
                circuit, backend, optimization_level=0), shots=1).result()

            # Check results
            success = getattr(result, 'success', False)
            self.assertTrue(success)
            data = result.data(0)
            self.assertIn(label, data)
            value = data[label]
            fidelity = fidelity_fn(target, value)
            self.assertGreater(fidelity, 0.9999)

    @supported_methods(
        ["automatic", "stabilizer", "statevector", "density_matrix", "matrix_product_state",
         "unitary", "superop"],
        CLIFFORD_GATES_DICT)
    def test_clifford_gate(self, method, device, gate):
        """Test Clifford standard gates."""
        self._test_gate(gate, CLIFFORD_GATES_DICT, method=method, device=device)

    @supported_methods(
        ["statevector", "density_matrix"],
        CLIFFORD_GATES_DICT)
    def test_clifford_gate_cache_blocking(self, method, device, gate):
        """Test Clifford standard gates."""
        self._test_gate(gate, CLIFFORD_GATES_DICT, method=method, device=device,
                        blocking_qubits=2, max_parallel_threads=1)

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state",
         "unitary", "superop"],
        NONCLIFFORD_GATES_DICT)
    def test_nonclifford_gate(self, method, device, gate):
        """Test non-Clifford standard gates."""
        self._test_gate(gate, NONCLIFFORD_GATES_DICT, method=method, device=device)

    @supported_methods(
        ["statevector", "density_matrix"],
        NONCLIFFORD_GATES_DICT)
    def test_nonclifford_gate_cache_blocking(self, method, device, gate):
        """Test non-Clifford standard gates."""
        self._test_gate(gate, NONCLIFFORD_GATES_DICT, method=method, device=device,
                        blocking_qubits=2, max_parallel_threads=1)

    @supported_methods(["automatic", "statevector", "unitary"], MC_GATES_DICT)
    def test_multictrl_gate(self, method, device, gate):
        """Test multi-controlled standard gates."""
        self._test_gate(gate, MC_GATES_DICT, method=method, device=device)

    @supported_methods(["statevector"], MC_GATES_DICT)
    def test_multictrl_gate_cache_blocking(self, method, device, gate):
        """Test multi-controlled standard gates."""
        self._test_gate(gate, MC_GATES_DICT, method=method, device=device,
                        blocking_qubits=2, max_parallel_threads=1)
