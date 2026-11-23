# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Integration tests for mid-circuit measurement support.

Qiskit IBM Runtime provides a dedicated ``MidCircuitMeasure`` instruction
whose default instruction name is ``measure_2``.  Aer should treat
``measure_*`` instructions as measurement operations while preserving the
instruction name.

The test suite intentionally defines a small local equivalent of
``MidCircuitMeasure`` instead of importing qiskit-ibm-runtime.  Qiskit Aer
must not acquire a runtime-package dependency just to test support for the
instruction representation.

These tests cover:

* ordinary mid-circuit measurement regression behaviour;
* transpilation of ``measure_2`` against ``AerSimulator``;
* execution of ``measure_*`` instructions;
* coexistence of ``measure_*`` and ordinary ``measure``;
* classical feedback from a named mid-circuit measurement;
* reset after a named mid-circuit measurement;
* independent noise targeting of ``measure_*``;
* custom ``measure_*`` variants;
* SamplerV2 execution;
* preservation of the final-measurement sampling fast path.
"""

import unittest

from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit import (
    ClassicalRegister,
    Instruction,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, pauli_error
from qiskit_aer.primitives import SamplerV2
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase,
    supported_methods,
)


SUPPORTED_METHODS = [
    "statevector",
    "density_matrix",
    "matrix_product_state",
]


class MidCircuitMeasure(Instruction):
    """Local equivalent of qiskit_ibm_runtime.circuit.MidCircuitMeasure.

    The Runtime instruction is a one-qubit, one-clbit Instruction whose name
    follows the ``measure_<identifier>`` convention.  Its default name is
    ``measure_2``.

    Keeping this small equivalent here lets Aer test compatibility with the
    instruction representation without depending on qiskit-ibm-runtime.
    """

    def __init__(self, name="measure_2", label=None):
        if not name.startswith("measure_"):
            raise ValueError(
                "Invalid name for mid-circuit measure instruction. "
                "The provided name must start with `measure_`"
            )

        super().__init__(name, 1, 1, [], label=label)


@ddt
class TestMidCircuitMeasurement(SimulatorTestCase):
    """Tests for standard and named mid-circuit measurement operations."""

    # ------------------------------------------------------------------
    # Standard Measure regression
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_standard_mid_circuit_measure(self, method, device):
        """Ordinary Measure must continue to work in the middle of a circuit."""

        shots = 200

        circuit = QuantumCircuit(1, 1)

        circuit.x(0)
        circuit.measure(0, 0)
        circuit.x(0)
        circuit.measure(0, 0)

        backend = self.backend(method=method, device=device)
        result = backend.run(circuit, shots=shots).result()

        self.assertSuccess(result)
        self.assertEqual(result.get_counts(), {"0": shots})

    # ------------------------------------------------------------------
    # Core regression: measure_2 must transpile against AerSimulator
    # ------------------------------------------------------------------

    def test_named_mcm_transpiles(self):
        """Regression: Aer target must accept ``measure_2``.

        This covers the user-facing failure:

            TranspilerError:
            'HighLevelSynthesis is unable to synthesize "measure_2"'

        A named mid-circuit measurement must survive transpilation against
        AerSimulator instead of being treated as an unknown operation.
        """

        circuit = QuantumCircuit(1, 1)

        circuit.h(0)
        circuit.append(MidCircuitMeasure(), [0], [0])
        circuit.x(0)
        circuit.measure(0, 0)

        backend = AerSimulator()

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        operation_names = [instruction.operation.name for instruction in transpiled.data]

        self.assertIn(
            "measure_2",
            operation_names,
            msg=(
                "MidCircuitMeasure instruction 'measure_2' should be preserved "
                "when transpiling for AerSimulator"
            ),
        )

    # ------------------------------------------------------------------
    # End-to-end transpile + execution
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_transpile_and_execute(self, method, device):
        """``measure_2`` must transpile and execute with measurement semantics."""

        shots = 200

        circuit = QuantumCircuit(1, 1)

        # Prepare |1>.
        circuit.x(0)

        # Runtime-style mid-circuit measurement.
        # This should deterministically measure 1.
        circuit.append(MidCircuitMeasure(), [0], [0])

        # The operation after measurement makes this genuinely mid-circuit.
        # |1> -> |0>.
        circuit.x(0)

        # Standard terminal measurement.
        circuit.measure(0, 0)

        backend = self.backend(method=method, device=device)

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)
        self.assertEqual(
            result.get_counts(),
            {"0": shots},
            msg="measure_2 should have the same quantum measurement semantics as measure",
        )

    # ------------------------------------------------------------------
    # Direct execution through Aer compiler
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_direct_execution(self, method, device):
        """Aer compiler must directly recognize a ``measure_*`` operation."""

        shots = 200

        circuit = QuantumCircuit(1, 1)

        circuit.x(0)
        circuit.append(MidCircuitMeasure(), [0], [0])
        circuit.x(0)
        circuit.measure(0, 0)

        backend = self.backend(method=method, device=device)

        result = backend.run(circuit, shots=shots).result()

        self.assertSuccess(result)
        self.assertEqual(result.get_counts(), {"0": shots})

    # ------------------------------------------------------------------
    # measure_2 and ordinary measure must coexist
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_and_standard_measure_coexist(self, method, device):
        """Named MCM and ordinary Measure must remain independent operations."""

        shots = 200

        circuit = QuantumCircuit(2, 2)

        # q0 = |1>
        circuit.x(0)

        # Named MCM records c0 = 1.
        circuit.append(MidCircuitMeasure(), [0], [0])

        # Ensure the first measurement is genuinely mid-circuit.
        circuit.x(0)

        # q1 stays |0> and uses ordinary Measure.
        circuit.measure(1, 1)

        backend = self.backend(method=method, device=device)

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        operation_names = [instruction.operation.name for instruction in transpiled.data]

        self.assertIn("measure_2", operation_names)
        self.assertIn("measure", operation_names)

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)

        # c1 = 0, c0 = 1.
        self.assertEqual(result.get_counts(), {"01": shots})

    # ------------------------------------------------------------------
    # Classical feedback
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_classical_feedback(self, method, device):
        """Result of ``measure_2`` must be usable by downstream control flow."""

        shots = 200

        qreg = QuantumRegister(1, "q")
        condition = ClassicalRegister(1, "condition")
        output = ClassicalRegister(1, "output")

        circuit = QuantumCircuit(qreg, condition, output)

        # Deterministically prepare |1>.
        circuit.x(qreg[0])

        # measure_2 writes 1 into condition[0].
        circuit.append(
            MidCircuitMeasure(),
            [qreg[0]],
            [condition[0]],
        )

        # The measurement result must be available to the condition.
        with circuit.if_test((condition, 1)):
            circuit.x(qreg[0])

        # Conditional X should have returned q0 to |0>.
        circuit.measure(qreg[0], output[0])

        backend = self.backend(method=method, device=device)

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)

        counts = result.get_counts()

        # Output register is displayed before condition register.
        # output=0 and condition=1.
        self.assertEqual(
            counts,
            {"0 1": shots},
            msg=(
                "The result written by measure_2 must be available to "
                "downstream classical control"
            ),
        )

    # ------------------------------------------------------------------
    # Named MCM followed by reset
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_then_reset(self, method, device):
        """Reset must work correctly after a named mid-circuit measurement."""

        shots = 200

        qreg = QuantumRegister(1, "q")
        mcm = ClassicalRegister(1, "mcm")
        final = ClassicalRegister(1, "final")

        circuit = QuantumCircuit(qreg, mcm, final)

        circuit.x(0)

        # Deterministically reads 1.
        circuit.append(MidCircuitMeasure(), [qreg[0]], [mcm[0]])

        # Reset state to |0>.
        circuit.reset(0)

        circuit.measure(qreg[0], final[0])

        backend = self.backend(method=method, device=device)

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)

        # final=0, mcm=1
        self.assertEqual(result.get_counts(), {"0 1": shots})

    # ------------------------------------------------------------------
    # Noise must be able to target measure_2 separately
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_quantum_error(self, method, device):
        """A QuantumError on ``measure_2`` must not affect ordinary measure."""

        shots = 200

        circuit = QuantumCircuit(2, 2)

        # q0 starts in |0>.
        # X error immediately before measure_2 makes the MCM read 1.
        circuit.append(MidCircuitMeasure(), [0], [0])

        # Prevent the MCM from becoming a terminal-only operation.
        circuit.x(0)

        # q1 remains |0> and is measured by the ordinary measure operation.
        circuit.measure(1, 1)

        noise_model = NoiseModel()
        noise_model.add_quantum_error(
            pauli_error([("X", 1.0)]),
            "measure_2",
            [0],
        )

        backend = self.backend(
            method=method,
            device=device,
            noise_model=noise_model,
        )

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)

        # c0 = 1 from X-before-measure_2.
        # c1 = 0 because ordinary measure is unaffected.
        self.assertEqual(
            result.get_counts(),
            {"01": shots},
            msg=(
                "QuantumError targeting measure_2 should affect only measure_2 "
                "and not the ordinary measure instruction"
            ),
        )

    # ------------------------------------------------------------------
    # measure_2 and another measure_* implementation can be distinct
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_independent_quantum_errors(self, method, device):
        """Different ``measure_*`` instructions can receive independent errors."""

        shots = 200

        circuit = QuantumCircuit(2, 2)

        circuit.append(MidCircuitMeasure("measure_2"), [0], [0])
        circuit.append(MidCircuitMeasure("measure_3"), [1], [1])

        noise_model = NoiseModel()

        # X before measure_2: |0> -> |1>, therefore c0 = 1.
        noise_model.add_quantum_error(
            pauli_error([("X", 1.0)]),
            "measure_2",
            [0],
        )

        # Z before measure_3 leaves |0> unchanged, therefore c1 = 0.
        noise_model.add_quantum_error(
            pauli_error([("Z", 1.0)]),
            "measure_3",
            [1],
        )

        backend = self.backend(
            method=method,
            device=device,
            noise_model=noise_model,
        )

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)

        self.assertEqual(
            result.get_counts(),
            {"01": shots},
            msg="measure_2 and measure_3 should be independently addressable",
        )

    # ------------------------------------------------------------------
    # Custom measure_* names
    # ------------------------------------------------------------------

    @data("measure_2", "measure_3")
    def test_named_mcm_variants_transpile(self, name):
        """Aer should support the measurement instruction family, not only measure_2."""

        circuit = QuantumCircuit(1, 1)

        circuit.x(0)
        circuit.append(MidCircuitMeasure(name), [0], [0])
        circuit.x(0)
        circuit.measure(0, 0)

        # Explicitly include the named measurement in the simulator's basis
        # through the noise model.  This also verifies that NoiseModel correctly
        # propagates arbitrary measure_* instruction names.
        noise_model = NoiseModel()
        noise_model.add_quantum_error(
            pauli_error([("I", 1.0)]),
            name,
            [0],
        )

        backend = AerSimulator(
            method="statevector",
            noise_model=noise_model,
        )

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        operation_names = [instruction.operation.name for instruction in transpiled.data]

        self.assertIn(name, operation_names)

        result = backend.run(transpiled, shots=100).result()

        self.assertSuccess(result)
        self.assertEqual(result.get_counts(), {"0": 100})

    # ------------------------------------------------------------------
    # Readout noise
    # ------------------------------------------------------------------

    @supported_methods(SUPPORTED_METHODS)
    def test_named_mcm_with_readout_noise(self, method, device):
        """Named mid-circuit measurement must work with readout noise."""

        shots = 1000

        readout_error = ReadoutError(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ]
        )

        noise_model = NoiseModel()
        noise_model.add_all_qubit_readout_error(readout_error)

        # Ensure measure_2 is part of the noisy simulator basis.
        noise_model.add_quantum_error(
            pauli_error([("I", 1.0)]),
            "measure_2",
            [0],
        )

        qreg = QuantumRegister(1, "q")
        condition = ClassicalRegister(1, "condition")
        output = ClassicalRegister(1, "output")

        circuit = QuantumCircuit(qreg, condition, output)

        circuit.x(0)
        circuit.append(
            MidCircuitMeasure(),
            [qreg[0]],
            [condition[0]],
        )

        with circuit.if_test((condition, 1)):
            circuit.x(0)

        circuit.measure(qreg[0], output[0])

        backend = self.backend(
            method=method,
            device=device,
            noise_model=noise_model,
        )

        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
        )

        result = backend.run(transpiled, shots=shots).result()

        self.assertSuccess(result)

        counts = result.get_counts()

        self.assertEqual(sum(counts.values()), shots)

    # ------------------------------------------------------------------
    # SamplerV2
    # ------------------------------------------------------------------

    def test_named_mcm_sampler_v2(self):
        """SamplerV2 must execute a circuit containing ``measure_2``."""

        shots = 200

        qreg = QuantumRegister(1, "q")
        condition = ClassicalRegister(1, "condition")
        output = ClassicalRegister(1, "output")

        circuit = QuantumCircuit(qreg, condition, output)

        circuit.x(0)

        circuit.append(
            MidCircuitMeasure(),
            [qreg[0]],
            [condition[0]],
        )

        with circuit.if_test((condition, 1)):
            circuit.x(0)

        circuit.measure(qreg[0], output[0])

        sampler = SamplerV2(
            default_shots=shots,
            seed=42,
        )

        result = sampler.run([circuit]).result()

        pub_result = result[0]

        condition_counts = pub_result.data.condition.get_counts()
        output_counts = pub_result.data.output.get_counts()

        self.assertEqual(
            condition_counts,
            {"1": shots},
            msg="measure_2 should deterministically record 1",
        )

        self.assertEqual(
            output_counts,
            {"0": shots},
            msg="conditional correction should leave the final state at |0>",
        )

    # ------------------------------------------------------------------
    # Existing terminal-measurement optimization must not regress
    # ------------------------------------------------------------------

    @data("statevector", "density_matrix")
    def test_final_measure_sampling_regression(self, method):
        """Ordinary final measurements must still use measure_sampling."""

        shots = 500

        circuit = QuantumCircuit(2, 2)

        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        backend = AerSimulator(
            method=method,
            seed_simulator=42,
        )

        result = backend.run(circuit, shots=shots).result()

        self.assertSuccess(result)

        for experiment_result in result.results:
            self.assertIn(
                "measure_sampling",
                experiment_result.metadata,
                msg="measure_sampling key missing from result metadata",
            )
            self.assertTrue(
                experiment_result.metadata["measure_sampling"],
                msg=(
                    "Ordinary final-only measurement circuits should retain "
                    "the fast measure_sampling path"
                ),
            )


if __name__ == "__main__":
    unittest.main()
