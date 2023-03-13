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
from qiskit.circuit.quantumcircuit import QuantumCircuit
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

from qiskit import transpile
import qiskit.quantum_info as qi


@ddt
class TestPauliGate(SimulatorTestCase):
    """Test standard gate library."""

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "unitary",
            "superop",
            "extended_stabilizer",
            "tensor_network",
        ],
        ["I", "X", "Y", "Z", "XY", "ZXY"],
    )
    def test_pauli_gate(self, method, device, pauli):
        """Test multi-qubit Pauli gate."""
        pauli = qi.Pauli(pauli)
        circuit = QuantumCircuit(pauli.num_qubits)
        circuit.append(pauli, range(pauli.num_qubits))

        backend = self.backend(method=method, device=device)
        label = "final"
        if method == "density_matrix":
            target = qi.DensityMatrix(circuit)
            circuit.save_density_matrix(label=label)
            fidelity_fn = qi.state_fidelity
        elif method == "stabilizer":
            target = qi.StabilizerState(qi.Clifford(circuit))
            circuit.save_stabilizer(label=label)
            fidelity_fn = qi.process_fidelity
        elif method == "unitary":
            target = qi.Operator(circuit)
            circuit.save_unitary(label=label)
            fidelity_fn = qi.process_fidelity
        elif method == "superop":
            target = qi.SuperOp(circuit)
            circuit.save_superop(label=label)
            fidelity_fn = qi.process_fidelity
        else:
            target = qi.Statevector(circuit)
            circuit.save_statevector(label=label)
            fidelity_fn = qi.state_fidelity

        result = backend.run(transpile(circuit, backend, optimization_level=0), shots=1).result()

        # Check results
        success = getattr(result, "success", False)
        self.assertTrue(success, msg="Simulation unexpectedly failed")
        data = result.data(0)
        self.assertIn(label, data)
        fidelity = fidelity_fn(target, data[label])

        threshold = 0.9999
        self.assertGreater(fidelity, threshold, msg="Fidelity {fidelity} not > {threshold}")
