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
AerSimulator Integration Tests
"""
import unittest
import platform

from ddt import ddt
from test.terra.reference import ref_measure
from test.terra.reference import ref_reset
from test.terra.reference import ref_initialize
from test.terra.reference import ref_kraus_noise
from test.terra.reference import ref_pauli_noise
from test.terra.reference import ref_readout_noise
from test.terra.reference import ref_reset_noise
from test.terra.reference import ref_conditionals

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, depolarizing_error
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info.random import random_unitary
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

from qiskit_aer import noise

import qiskit.quantum_info as qi
from qiskit.circuit.library import QFT
from qiskit.circuit import QuantumCircuit, Reset
from qiskit.circuit.library.standard_gates import IGate, HGate
from qiskit.quantum_info.states.densitymatrix import DensityMatrix

from qiskit.circuit import Parameter, Qubit, Clbit, QuantumRegister, ClassicalRegister
from qiskit.circuit.controlflow import *
from qiskit_aer.library.default_qubits import default_qubits
from qiskit_aer.library.control_flow_instructions import AerMark, AerJump

import numpy as np

SUPPORTED_METHODS = [
    "statevector",
    "density_matrix",
]
# tensor_network is tested in other test cases by setting shot_branching_enable by default

SUPPORTED_METHODS_INITIALIZE = [
    "statevector",
]


@ddt
@unittest.skipIf(platform.system() == "Darwin", "skip MacOS tentatively")
class TestShotBranching(SimulatorTestCase):
    """AerSimulator measure tests."""

    OPTIONS = {"seed_simulator": 41411}

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_measure_nondeterministic_with_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_measure_nondeterministic_without_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        delta = 0.05
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_measure_sampling_with_quantum_noise(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        depolarizing = {"u3": (1, 0.001), "cx": (2, 0.02)}
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in depolarizing.items():
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(gate_error, num_qubits), gate
            )

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        sampling = method == "density_matrix" or method == "tensor_network"
        self.compare_result_metadata(result, circuits, "measure_sampling", sampling)

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_measure_nondeterministic_multi_qubit_with_sampling(
        self, method, device
    ):
        """Test AerSimulator measure with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_measure_nondeterministic_multi_qubit_without_sampling(
        self, method, device
    ):
        """Test AerSimulator measure with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        delta = 0.05
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_reset_nondeterministic(self, method, device):
        """Test AerSimulator reset with for circuits with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 4000
        circuits = ref_reset.reset_circuits_nondeterministic(final_measure=True)
        targets = ref_reset.reset_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_repeated_resets(self, method, device):
        """Test repeated reset operations"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_reset.reset_circuits_repeated()
        targets = ref_reset.reset_counts_repeated(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_reset_moving_qubits(self, method, device):
        """Test AerSimulator reset with for circuits where qubits have moved"""
        backend = self.backend(method=method, device=device)
        # count output circuits
        shots = 1000
        circuits = ref_reset.reset_circuits_with_entangled_and_moving_qubits(final_measure=True)
        targets = ref_reset.reset_counts_with_entangled_and_moving_qubits(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test initialize instr make it through the wrapper
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_wrapper_1(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        shots = 100
        if "tensor_network" in method:
            shots = 10
        lst = [0, 1]
        init_states = [
            np.array(lst),
            np.array(lst, dtype=float),
            np.array(lst, dtype=np.float32),
            np.array(lst, dtype=complex),
            np.array(lst, dtype=np.complex64),
        ]
        circuits = []
        [
            circuits.extend(ref_initialize.initialize_circuits_w_1(init_state))
            for init_state in init_states
        ]
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)

    # ---------------------------------------------------------------------
    # Test initialize instr make it through the wrapper
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_wrapper_2(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        shots = 100
        lst = [0, 1, 0, 0]
        init_states = [
            np.array(lst),
            np.array(lst, dtype=float),
            np.array(lst, dtype=np.float32),
            np.array(lst, dtype=complex),
            np.array(lst, dtype=np.complex64),
        ]
        circuits = []
        [
            circuits.extend(ref_initialize.initialize_circuits_w_2(init_state))
            for init_state in init_states
        ]
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)

    # ---------------------------------------------------------------------
    # Test initialize
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_1(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 1000
        delta = 0.05
        circuits = ref_initialize.initialize_circuits_1(final_measure=True)
        targets = ref_initialize.initialize_counts_1(shots)
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)

    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_2(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 1000
        delta = 0.05
        circuits = ref_initialize.initialize_circuits_2(final_measure=True)
        targets = ref_initialize.initialize_counts_2(shots)
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)

    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_entangled_qubits(self, method, device):
        """Test initialize entangled qubits"""
        backend = self.backend(method=method, device=device)
        shots = 1000
        delta = 0.05
        circuits = ref_initialize.initialize_entangled_qubits()
        targets = ref_initialize.initialize_counts_entangled_qubits(shots)
        result = backend.run(
            circuits, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)

    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_sampling_opt_disabled(self, method, device):
        """Test sampling optimization"""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.initialize([0, 1], [1])
        circuit.measure_all()
        result = backend.run(
            circuit, shots=shots, shot_branching_enable=True, shot_branching_sampling_enable=True
        ).result()
        self.assertSuccess(result)
        sampling = result.results[0].metadata.get("measure_sampling", None)
        self.assertFalse(sampling)

    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_with_labels(self, method, device):
        """Test sampling optimization"""
        backend = self.backend(method=method, device=device)

        circ = QuantumCircuit(4)
        circ.initialize("+-rl")
        circ.save_statevector()
        actual = (
            backend.run(circ, shot_branching_enable=True, shot_branching_sampling_enable=True)
            .result()
            .get_statevector(circ)
        )

        for q4, p4 in enumerate([1, 1]):
            for q3, p3 in enumerate([1, -1]):
                for q2, p2 in enumerate([1, 1j]):
                    for q1, p1 in enumerate([1, -1j]):
                        index = int("{}{}{}{}".format(q4, q3, q2, q1), 2)
                        self.assertAlmostEqual(actual[index], 0.25 * p1 * p2 * p3 * p4)

    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_with_int(self, method, device):
        """Test sampling with int"""
        backend = self.backend(method=method, device=device)

        circ = QuantumCircuit(4)
        circ.initialize(5, [0, 1, 2])
        circ.save_statevector()
        actual = (
            backend.run(circ, shot_branching_enable=True, shot_branching_sampling_enable=True)
            .result()
            .get_statevector(circ)
        )

        self.assertAlmostEqual(actual[5], 1)

    @supported_methods(SUPPORTED_METHODS_INITIALIZE)
    def test_shot_branching_initialize_with_int_twice(self, method, device):
        """Test sampling with int twice"""
        backend = self.backend(method=method, device=device)

        circ = QuantumCircuit(4)
        circ.initialize(1, [0])
        circ.initialize(1, [2])
        circ.save_statevector()
        actual = (
            backend.run(circ, shot_branching_enable=True, shot_branching_sampling_enable=True)
            .result()
            .get_statevector(circ)
        )

        self.assertAlmostEqual(actual[5], 1)

    # ---------------------------------------------------------------------
    # Test noise
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_empty_circuit_noise(self, method, device):
        """Test simulation with empty circuit and noise model."""
        backend = self.backend(method=method, device=device)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(0.1, 1), ["x"])
        result = backend.run(
            QuantumCircuit(), shots=1, noise_model=noise_model, shot_branching_enable=True
        ).result()
        self.assertSuccess(result)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_readout_noise(self, method, device):
        """Test simulation with classical readout error noise model."""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 4000
        circuits = ref_readout_noise.readout_error_circuits()
        noise_models = ref_readout_noise.readout_error_noise_models()
        targets = ref_readout_noise.readout_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots, shot_branching_enable=True).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS, [noise.QuantumError, noise.PauliError])
    def test_shot_branching_pauli_gate_noise(self, method, device, qerror_cls):
        """Test simulation with Pauli gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_gate_error_circuits()
        noise_models = ref_pauli_noise.pauli_gate_error_noise_models(qerror_cls)
        targets = ref_pauli_noise.pauli_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots, shot_branching_enable=True).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS, [noise.QuantumError, noise.PauliError])
    def test_shot_branching_pauli_reset_noise(self, method, device, qerror_cls):
        """Test simulation with Pauli reset error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_reset_error_circuits()
        noise_models = ref_pauli_noise.pauli_reset_error_noise_models(qerror_cls)
        targets = ref_pauli_noise.pauli_reset_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots, shot_branching_enable=True).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS, [noise.QuantumError, noise.PauliError])
    def test_shot_branching_pauli_measure_noise(self, method, device, qerror_cls):
        """Test simulation with Pauli measure error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_measure_error_circuits()
        noise_models = ref_pauli_noise.pauli_measure_error_noise_models(qerror_cls)
        targets = ref_pauli_noise.pauli_measure_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots, shot_branching_enable=True).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_reset_gate_noise(self, method, device):
        """Test simulation with reset gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_reset_noise.reset_gate_error_circuits()
        noise_models = ref_reset_noise.reset_gate_error_noise_models()
        targets = ref_reset_noise.reset_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots, shot_branching_enable=True).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_kraus_gate_noise(self, method, device):
        """Test simulation with Kraus gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_kraus_noise.kraus_gate_error_circuits()
        noise_models = ref_kraus_noise.kraus_gate_error_noise_models()
        targets = ref_kraus_noise.kraus_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots, shot_branching_enable=True).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_kraus_gate_noise_on_QFT(self, method, device):
        """Test Kraus noise on a QFT circuit"""
        shots = 10000

        # Build noise model
        error1 = noise.amplitude_damping_error(0.2)
        error2 = error1.tensor(error1)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error1, ["h"])
        noise_model.add_all_qubit_quantum_error(error2, ["cp", "swap"])

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        ideal_circuit = transpile(QFT(3), backend)

        # manaully build noise circuit
        noise_circuit = QuantumCircuit(3)
        for inst, qargs, cargs in ideal_circuit.data:
            noise_circuit.append(inst, qargs, cargs)
            if inst.name == "h":
                noise_circuit.append(error1, qargs)
            elif inst.name in ["cp", "swap"]:
                noise_circuit.append(error2, qargs)
        # compute target counts
        noise_state = DensityMatrix(noise_circuit)
        ref_target = {i: shots * p for i, p in noise_state.probabilities_dict().items()}

        # Run sim
        ideal_circuit.measure_all()
        result = backend.run(ideal_circuit, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(
            result, [ideal_circuit], [ref_target], hex_counts=False, delta=0.1 * shots
        )

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_clifford_circuit_noise(self, method, device):
        """Test simulation with mixed Clifford quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        error1 = noise.QuantumError(
            [([(IGate(), [0])], 0.8), ([(Reset(), [0])], 0.1), ([(HGate(), [0])], 0.1)]
        )

        error2 = noise.QuantumError(
            [
                ([(IGate(), [0])], 0.75),
                ([(Reset(), [0])], 0.1),
                ([(Reset(), [1])], 0.1),
                ([(Reset(), [0]), (Reset(), [1])], 0.05),
            ]
        )

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(error1, [0])
        qc.cx(0, 1)
        qc.append(error2, [0, 1])
        target_probs = qi.DensityMatrix(qc).probabilities_dict()

        # Add measurement
        qc.measure_all()
        result = backend.run(qc, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        probs = {key: val / shots for key, val in result.get_counts(0).items()}
        self.assertDictAlmostEqual(target_probs, probs, delta=0.1)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_kraus_circuit_noise(self, method, device):
        """Test simulation with Kraus quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        error0 = noise.amplitude_damping_error(0.05)
        error1 = noise.amplitude_damping_error(0.15)
        error01 = error1.tensor(error0)

        # Target Circuit 0
        tc0 = QuantumCircuit(2)
        tc0.h(0)
        tc0.append(qi.Kraus(error0), [0])
        tc0.cx(0, 1)
        tc0.append(qi.Kraus(error01), [0, 1])
        target_probs0 = qi.DensityMatrix(tc0).probabilities_dict()

        # Sim circuit 0
        qc0 = QuantumCircuit(2)
        qc0.h(0)
        qc0.append(error0, [0])
        qc0.cx(0, 1)
        qc0.append(error01, [0, 1])
        qc0.measure_all()

        # Target Circuit 1
        tc1 = QuantumCircuit(2)
        tc1.h(1)
        tc1.append(qi.Kraus(error0), [1])
        tc1.cx(1, 0)
        tc1.append(qi.Kraus(error01), [1, 0])
        target_probs1 = qi.DensityMatrix(tc1).probabilities_dict()

        # Sim circuit 1
        qc1 = QuantumCircuit(2)
        qc1.h(1)
        qc1.append(error0, [1])
        qc1.cx(1, 0)
        qc1.append(error01, [1, 0])
        qc1.measure_all()

        result = backend.run([qc0, qc1], shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        probs = [{key: val / shots for key, val in result.get_counts(i).items()} for i in range(2)]
        self.assertDictAlmostEqual(target_probs0, probs[0], delta=0.1)
        self.assertDictAlmostEqual(target_probs1, probs[1], delta=0.1)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_gates_1bit(self, method, device):
        """Test conditional gate operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="gate"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_gates_2bit(self, method, device):
        """Test conditional gate operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="gate"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_gates_64bit(self, method, device):
        """Test conditional gate operations on 64-bit conditional register."""
        shots = 100
        # [value of conditional register, list of condtional values]
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="gate"
        )
        # not using hex counts because number of leading zeros in results
        # doesn't seem consistent
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_gates_132bit(self, method, device):
        """Test conditional gate operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="gate"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        circuits = circuits[0:1]
        targets = targets[0:1]
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_unitary_1bit(self, method, device):
        """Test conditional unitary operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_unitary_2bit(self, method, device):
        """Test conditional unitary operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_unitary_64bit(self, method, device):
        """Test conditional unitary operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_unitary_132bit(self, method, device):
        """Test conditional unitary operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_unitary_1bit(self, method, device):
        """Test conditional kraus operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_kraus_2bit(self, method, device):
        """Test conditional kraus operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_kraus_64bit(self, method, device):
        """Test conditional kraus operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_conditional_kraus_132bit(self, method, device):
        """Test conditional kraus operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    # ---------------------------------------------------------------------
    # Test control flow
    # ---------------------------------------------------------------------
