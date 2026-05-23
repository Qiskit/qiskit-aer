# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
RelaxationNoisePass class tests
"""

import ddt
from qiskit_aer.noise import thermal_relaxation_error, RelaxationNoisePass

import qiskit.quantum_info as qi
from qiskit.circuit import QuantumCircuit, Delay
from qiskit.circuit.library import HGate, CXGate, Measure
from qiskit.compiler import transpile
from qiskit.transpiler import TranspilerError, Target, InstructionProperties
from qiskit.providers.fake_provider import GenericBackendV2
from test.terra.common import QiskitAerTestCase


@ddt.ddt
class TestRelaxationNoisePass(QiskitAerTestCase):
    """Testing RelaxationNoisePass class"""

    def test_delay_circuit_on_single_qubit(self):
        t1s = [0.10, 0.11]
        t2s = [0.20, 0.21]
        dt = 0.01
        duration = 100

        qc = QuantumCircuit(1)
        qc.delay(duration, 0)

        relax_pass = RelaxationNoisePass(t1s=t1s, t2s=t2s, dt=dt)
        actual = qi.SuperOp(relax_pass(qc))
        expected = qi.SuperOp(thermal_relaxation_error(t1s[0], t2s[0], duration * dt))
        self.assertEqual(expected, actual)

    def test_delay_circuit_on_multi_qubits(self):
        t1s = [0.10, 0.11]
        t2s = [0.20, 0.21]
        dt = 0.01
        duration = 100

        qc = QuantumCircuit(2)
        qc.delay(duration, 0)
        qc.delay(duration, 1)

        relax_pass = RelaxationNoisePass(t1s=t1s, t2s=t2s, dt=dt)
        actual = qi.SuperOp(relax_pass(qc))

        noise0 = thermal_relaxation_error(t1s[0], t2s[0], duration * dt)
        noise1 = thermal_relaxation_error(t1s[1], t2s[1], duration * dt)
        expected = qi.SuperOp(noise0.expand(noise1))
        self.assertEqual(expected, actual)

    def test_default_with_scheduled_circuit(self):
        """Test adding noises to all ops in a scheduled circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        target = GenericBackendV2(
            2,
            coupling_map=[[0, 1]],
            basis_gates=["cx", "h"],
            seed=42,
        ).target
        target.dt = 10e-2
        target.update_instruction_properties("cx", (0, 1), InstructionProperties(50 * 10e-2))
        target.update_instruction_properties("h", (0,), InstructionProperties(10 * 10e-2))
        target.update_instruction_properties("h", (1,), InstructionProperties(10 * 10e-2))
        target.update_instruction_properties("measure", (0,), InstructionProperties(200 * 10e-2))
        target.update_instruction_properties("measure", (1,), InstructionProperties(200 * 10e-2))

        sched_circ = transpile(qc, scheduling_method="alap", target=target)

        noise_pass = RelaxationNoisePass(t1s=[0.10, 0.11], t2s=[0.20, 0.21], dt=0.01, target=target)
        noisy_circ = noise_pass(sched_circ)
        self.assertEqual(6, noisy_circ.decompose().decompose().count_ops()["kraus"])

    def test_default_with_non_scheduled_circuit(self):
        """Test never adding noises to non-scheduled circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        noise_pass = RelaxationNoisePass(t1s=[0.10, 0.11], t2s=[0.20, 0.21], dt=0.01)
        with self.assertWarns(UserWarning):
            noisy_circ = noise_pass(qc)

        self.assertEqual(qc, noisy_circ)

    def test_raise_if_supplied_invalid_ops(self):
        with self.assertRaises(TranspilerError):
            RelaxationNoisePass(t1s=[1], t2s=[1], dt=1, op_types="delay")  # str is invalid

    def test_ops_types(self):
        """Test adding noises only to delays in a scheduled circuit."""
        t1s = [0.10, 0.11]
        t2s = [0.20, 0.21]
        dt = 0.01

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        target = GenericBackendV2(
            2,
            coupling_map=[[0, 1]],
            basis_gates=["cx", "h"],
            seed=42,
        ).target
        target.dt = 10e-2
        target.update_instruction_properties("cx", (0, 1), InstructionProperties(50 * 10e-2))
        target.update_instruction_properties("h", (0,), InstructionProperties(10 * 10e-2))
        target.update_instruction_properties("h", (1,), InstructionProperties(10 * 10e-2))
        target.update_instruction_properties("measure", (0,), InstructionProperties(200 * 10e-2))
        target.update_instruction_properties("measure", (1,), InstructionProperties(200 * 10e-2))

        sched_circ = transpile(qc, scheduling_method="alap", target=target)

        noise_pass = RelaxationNoisePass(t1s=t1s, t2s=t2s, dt=dt, op_types=Delay, target=target)
        noisy_circ = noise_pass(sched_circ)

        expected = QuantumCircuit(2, 2)
        expected.h(0)
        expected.delay(10, 1)
        expected.append(thermal_relaxation_error(t1s[1], t2s[1], 10 * dt).to_instruction(), [1])
        expected.cx(0, 1)
        expected.measure([0, 1], [0, 1])

        self.assertEqual(expected, noisy_circ)

    @ddt.data((1e5, "dt"), (1e4, "ns"), (1e1, "us"), (1e-2, "ms"), (1e-5, "s"))
    @ddt.unpack
    def test_delay_units(self, duration, unit):
        """Test un-scheduled delay with different units."""
        t1 = 0.004
        t2 = 0.008
        dt = 1e-10  # 0.1 ns
        target_duration = 1e-5

        qc = QuantumCircuit(1)
        qc.delay(duration, 0, unit=unit)

        relax_pass = RelaxationNoisePass(t1s=[t1], t2s=[t2], dt=dt)
        actual = qi.SuperOp(relax_pass(qc))
        expected = qi.SuperOp(thermal_relaxation_error(t1, t2, target_duration))
        self.assertEqual(expected, actual)
