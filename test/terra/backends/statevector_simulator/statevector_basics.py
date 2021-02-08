# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
StatevectorSimulator Integration Tests
"""

from numpy import exp, pi

from test.terra.reference import ref_measure
from test.terra.reference import ref_reset
from test.terra.reference import ref_initialize
from test.terra.reference import ref_conditionals
from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_unitary_gate
from test.terra.reference import ref_diagonal_gate

from qiskit import execute, transpile, assemble
from qiskit.providers.aer import StatevectorSimulator


class StatevectorSimulatorTests:
    """StatevectorSimulator tests."""

    SIMULATOR = StatevectorSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test initialize
    # ---------------------------------------------------------------------
    def test_initialize_1(self):
        """Test StatevectorSimulator initialize"""
        circuits = ref_initialize.initialize_circuits_1(final_measure=False)
        targets = ref_initialize.initialize_statevector_1()
        qobj = assemble(circuits, shots=1)
        sim_job = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS)
        result = sim_job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    def test_initialize_2(self):
        """Test StatevectorSimulator initialize"""
        circuits = ref_initialize.initialize_circuits_2(final_measure=False)
        targets = ref_initialize.initialize_statevector_2()
        qobj = assemble(circuits, shots=1)
        sim_job = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS)
        result = sim_job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    def test_reset_deterministic(self):
        """Test StatevectorSimulator reset with for circuits with deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        circuits = ref_reset.reset_circuits_deterministic(final_measure=False)
        targets = ref_reset.reset_statevector_deterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    def test_reset_nondeterministic(self):
        """Test StatevectorSimulator reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        circuits = ref_reset.reset_circuits_nondeterministic(
            final_measure=False)
        targets = ref_reset.reset_statevector_nondeterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    def test_measure(self):
        """Test StatevectorSimulator measure with deterministic counts"""
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=True)
        targets = ref_measure.measure_statevector_deterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_gate_1bit(self):
        """Test conditional gates on 1-bit conditional register."""
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=False)
        targets = ref_conditionals.conditional_statevector_1bit()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    def test_conditional_unitary_1bit(self):
        """Test conditional unitaries on 1-bit conditional register."""
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=False, conditional_type='unitary')
        targets = ref_conditionals.conditional_statevector_1bit()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    def test_conditional_gate_2bit(self):
        """Test conditional gates on 2-bit conditional register."""
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=False)
        targets = ref_conditionals.conditional_statevector_2bit()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        
        self.compare_statevector(result, circuits, targets)

    def test_conditional_unitary_2bit(self):
        """Test conditional unitary on 2-bit conditional register."""
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=False, conditional_type='unitary')
        targets = ref_conditionals.conditional_statevector_2bit()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    def test_conditional_gate_64bit(self):
        """Test conditional gates on 64-bit conditional register."""
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(64, cases,
            final_measure=False, conditional_type='gate')
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        job = execute(circuits, self.SIMULATOR, shots=1, **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)

        self.compare_statevector(result, circuits, targets)

    def test_conditional_unitary_64bit(self):
        """Test conditional unitary on 64-bit conditional register."""
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(64, cases,
            final_measure=False, conditional_type='unitary')
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        job = execute(circuits, self.SIMULATOR, shots=1, **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)

        self.compare_statevector(result, circuits, targets)

    def test_conditional_gate_132bit(self):
        """Test conditional gates on 132-bit conditional register."""
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(132, cases,
            final_measure=False, conditional_type='gate')
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        job = execute(circuits, self.SIMULATOR, shots=1, **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)

        self.compare_statevector(result, circuits, targets)

    def test_conditional_unitary_132bit(self):
        """Test conditional unitary on 132-bit conditional register."""
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(132, cases,
            final_measure=False, conditional_type='unitary')
        targets = ref_conditionals.conditional_statevector_nbit(cases)
        job = execute(circuits, self.SIMULATOR, shots=1, **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)

        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------
    def test_unitary_gate(self):
        """Test simulation with unitary gate circuit instructions."""
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(
            final_measure=False)
        targets = ref_unitary_gate.unitary_gate_statevector_deterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)

    def test_diagonal_gate(self):
        """Test simulation with diagonal gate circuit instructions."""
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(
            final_measure=False)
        targets = ref_diagonal_gate.diagonal_gate_statevector_deterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets)


        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test global phase
    # ---------------------------------------------------------------------

    def test_qobj_global_phase(self):
        """Test qobj global phase."""

        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_nondeterministic()

        qobj = assemble(transpile(circuits, self.SIMULATOR),
                        shots=1, **self.BACKEND_OPTS)
        # Set global phases
        for i, _ in enumerate(circuits):
            global_phase = (-1) ** i * (pi / 4)
            qobj.experiments[i].header.global_phase = global_phase
            targets[i] = exp(1j * global_phase) * targets[i]
        result = self.SIMULATOR.run(qobj).result()
        self.assertSuccess(result)
        self.compare_statevector(result, circuits, targets, ignore_phase=False)
