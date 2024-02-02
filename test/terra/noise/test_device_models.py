# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for utility functions to create device noise model.
"""
import numpy as np
from test.terra.common import QiskitAerTestCase

import qiskit
from qiskit.circuit import library, Reset, Measure, Parameter
from qiskit.providers import convert_to_target
from qiskit.transpiler import CouplingMap, Target, QubitProperties, InstructionProperties

if qiskit.__version__.startswith("0."):
    from qiskit.providers.fake_provider import FakeQuito as Fake5QV1
else:
    from qiskit.providers.fake_provider import Fake5QV1

from qiskit_aer.noise.device.models import basic_device_gate_errors
from qiskit_aer.noise.errors.standard_errors import thermal_relaxation_error


def target_7q():
    """Build an arbitrary 7q ``Target`` with noisy instructions."""
    num_qubits = 7
    qubit_properties = [
        QubitProperties(t1=1.5e-4, t2=1.5e-4, frequency=4_700_000_000.0 + q * 50_000_000.0)
        for q in range(num_qubits)
    ]
    target = Target(num_qubits=7, qubit_properties=qubit_properties)
    for gate in (library.SXGate(), library.XGate(), library.RZGate(Parameter("a"))):
        target.add_instruction(
            gate,
            properties={
                (q,): InstructionProperties(duration=50e-9, error=2e-4) for q in range(num_qubits)
            },
        )
    target.add_instruction(
        library.CXGate(),
        properties={
            link: InstructionProperties(duration=300e-9, error=1e-3)
            for link in CouplingMap.from_ring(num_qubits)
        },
    )
    target.add_instruction(
        Reset(),
        properties={
            (q,): InstructionProperties(duration=4e-6, error=None) for q in range(num_qubits)
        },
    )
    target.add_instruction(
        Measure(),
        properties={
            (q,): InstructionProperties(duration=3e-6, error=1.5e-2) for q in range(num_qubits)
        },
    )
    return target


class TestDeviceNoiseModel(QiskitAerTestCase):
    """Testing device noise model"""

    def test_basic_device_gate_errors_from_target(self):
        """Test if the resulting gate errors never include errors on non-gate instructions"""
        target = target_7q()
        gate_errors = basic_device_gate_errors(target=target)
        errors_on_measure = [name for name, _, _ in gate_errors if name == "measure"]
        errors_on_reset = [name for name, _, _ in gate_errors if name == "reset"]
        self.assertEqual(len(errors_on_measure), 0)
        self.assertEqual(len(errors_on_reset), 7)
        self.assertEqual(len(gate_errors), 42)

    def test_basic_device_gate_errors_from_target_with_non_operational_qubits(self):
        """Test if no thermal relaxation errors are generated for qubits with undefined T1 and T2."""
        target = target_7q()
        # tweak target to have non-operational qubits
        faulty_qubits = (1, 2)
        for q in faulty_qubits:
            target.qubit_properties[q] = QubitProperties(t1=None, t2=None, frequency=0)
        # build gate errors with only relaxation errors i.e. without depolarizing errors
        gate_errors = basic_device_gate_errors(target=target, gate_error=False)
        errors_on_sx = {qubits: error for name, qubits, error in gate_errors if name == "sx"}
        errors_on_cx = {qubits: error for name, qubits, error in gate_errors if name == "cx"}
        self.assertEqual(len(gate_errors), 42)
        # check if no errors are added on sx gates on qubits without T1 and T2 definitions
        for q in faulty_qubits:
            self.assertTrue(errors_on_sx[(q,)].ideal())
        # check if no error is added on cx gate on a qubit pair without T1 and T2 definitions
        self.assertTrue(errors_on_cx[faulty_qubits].ideal())

    def test_basic_device_gate_errors_from_target_and_properties(self):
        """Test if the device same gate errors are produced both from target and properties"""
        backend = Fake5QV1()
        target = convert_to_target(
            configuration=backend.configuration(),
            properties=backend.properties(),
        )
        errors_from_properties = basic_device_gate_errors(properties=backend.properties())
        errors_from_target = basic_device_gate_errors(target=target)
        self.assertEqual(len(errors_from_properties), len(errors_from_target))
        errors_from_properties_s = sorted(errors_from_properties)
        errors_from_target_s = sorted(errors_from_target)
        for err_properties, err_target in zip(errors_from_properties_s, errors_from_target_s):
            name1, qargs1, err1 = err_properties
            name2, qargs2, err2 = err_target
            self.assertEqual(name1, name2)
            self.assertEqual(tuple(qargs1), qargs2)
            self.assertEqual(err1, err2)

    def test_basic_device_gate_errors_from_target_with_no_t2_value(self):
        """Test if gate errors are successfully created from a target with qubits not reporting T2.
        See https://github.com/Qiskit/qiskit-aer/issues/1896 for the details."""
        target = target_7q()
        target.qubit_properties[0].t2 = None
        basic_device_gate_errors(target=target)

    def test_non_zero_temperature(self):
        """Test if non-zero excited_state_population is obtained when positive temperature is supplied.
        See https://github.com/Qiskit/qiskit-aer/issues/1937 for the details."""
        t1, t2, frequency, duration = 1e-4, 1e-4, 5e9, 5e-8
        target = Target(qubit_properties=[QubitProperties(t1=t1, t2=t2, frequency=frequency)])
        target.add_instruction(library.XGate(), {(0,): InstructionProperties(duration=duration)})
        errors = basic_device_gate_errors(target=target, gate_error=False, temperature=100)
        _, _, x_error = errors[0]
        no_excitation_error = thermal_relaxation_error(t1, t2, duration, excited_state_population=0)
        x_error_matrix = x_error.to_quantumchannel().data
        no_excitation_error_matrix = no_excitation_error.to_quantumchannel().data
        self.assertFalse(np.allclose(x_error_matrix, no_excitation_error_matrix))
