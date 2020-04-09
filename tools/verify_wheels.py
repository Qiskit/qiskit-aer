# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np
from numpy.linalg import norm

from qiskit import ClassicalRegister
from qiskit.compiler import assemble, transpile
from qiskit import execute
from qiskit import QuantumCircuit
from qiskit import QuantumRegister

from qiskit.providers.aer.pulse.duffing_model_generators import duffing_system_model
from qiskit.pulse import Schedule, Play, Acquire, Gaussian, DriveChannel, AcquireChannel, MemorySlot

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import StatevectorSimulator
from qiskit.providers.aer import UnitarySimulator
from qiskit.providers.aer import PulseSimulator

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


def assertAlmostEqual(first, second, places=None, msg=None,
                      delta=None):
    """Test of 2 object are almost equal.

    Fail if the two objects are unequal as determined by their
    difference rounded to the given number of decimal places
    (default 7) and comparing to zero, or by comparing that the
    difference between the two objects is more than the given
    delta.
    Note that decimal places (from zero) are usually not the same
    as significant digits (measured from the most significant digit).
    If the two objects compare equal then they will automatically
    compare almost equal.
    """
    if first == second:
        # shortcut
        return
    if delta is not None and places is not None:
        raise TypeError("specify delta or places not both")

    diff = abs(first - second)
    if delta is not None:
        if diff <= delta:
            return

        standardMsg = '%s != %s within %s delta (%s difference)' % (
            first,
            second,
            delta,
            diff)
    else:
        if places is None:
            places = 7

        if round(diff, places) == 0:
            return

        standardMsg = '%s != %s within %r places (%s difference)' % (
            first,
            second,
            places,
            diff)
    raise Exception(standardMsg)


def grovers_circuit(final_measure=True, allow_sampling=True):
    """Testing a circuit originated in the Grover algorithm"""

    circuits = []

    # 6-qubit grovers
    qr = QuantumRegister(6)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )
    circuit = QuantumCircuit(*regs)

    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.x(qr[2])
    circuit.x(qr[3])
    circuit.x(qr[0])
    circuit.cx(qr[0], qr[2])
    circuit.x(qr[0])
    circuit.cx(qr[1], qr[3])
    circuit.ccx(qr[2], qr[3], qr[4])
    circuit.cx(qr[1], qr[3])
    circuit.x(qr[0])
    circuit.cx(qr[0], qr[2])
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.x(qr[4])
    circuit.h(qr[4])
    circuit.ccx(qr[0], qr[1], qr[4])
    circuit.h(qr[4])
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.x(qr[4])
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.h(qr[4])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
    circuits.append(circuit)

    return circuits


def assertDictAlmostEqual(dict1, dict2, delta=None, msg=None,
                          places=None, default_value=0):
    """Assert two dictionaries with numeric values are almost equal.

    Fail if the two dictionaries are unequal as determined by
    comparing that the difference between values with the same key are
    not greater than delta (default 1e-8), or that difference rounded
    to the given number of decimal places is not zero. If a key in one
    dictionary is not in the other the default_value keyword argument
    will be used for the missing value (default 0). If the two objects
    compare equal then they will automatically compare almost equal.

    Args:
        dict1 (dict): a dictionary.
        dict2 (dict): a dictionary.
        delta (number): threshold for comparison (defaults to 1e-8).
        msg (str): return a custom message on failure.
        places (int): number of decimal places for comparison.
        default_value (number): default value for missing keys.

    Raises:
        TypeError: raises TestCase failureException if the test fails.
    """
    if dict1 == dict2:
        # Shortcut
        return
    if delta is not None and places is not None:
        raise TypeError("specify delta or places not both")

    if places is not None:
        success = True
        standard_msg = ''
        # check value for keys in target
        keys1 = set(dict1.keys())
        for key in keys1:
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if round(abs(val1 - val2), places) != 0:
                success = False
                standard_msg += '(%s: %s != %s), ' % (key,
                                                      val1,
                                                      val2)
        # check values for keys in counts, not in target
        keys2 = set(dict2.keys()) - keys1
        for key in keys2:
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if round(abs(val1 - val2), places) != 0:
                success = False
                standard_msg += '(%s: %s != %s), ' % (key,
                                                      val1,
                                                      val2)
        if success is True:
            return
        standard_msg = standard_msg[:-2] + ' within %s places' % places

    else:
        if delta is None:
            delta = 1e-8  # default delta value
        success = True
        standard_msg = ''
        # check value for keys in target
        keys1 = set(dict1.keys())
        for key in keys1:
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if abs(val1 - val2) > delta:
                success = False
                standard_msg += '(%s: %s != %s), ' % (key,
                                                      val1,
                                                      val2)
        # check values for keys in counts, not in target
        keys2 = set(dict2.keys()) - keys1
        for key in keys2:
            val1 = dict1.get(key, default_value)
            val2 = dict2.get(key, default_value)
            if abs(val1 - val2) > delta:
                success = False
                standard_msg += '(%s: %s != %s), ' % (key,
                                                      val1,
                                                      val2)
        if success is True:
            return
        standard_msg = standard_msg[:-2] + ' within %s delta' % delta

    raise Exception(standard_msg)


def compare_counts(result, circuits, targets, hex_counts=True, delta=0):
    """Compare counts dictionary to targets."""
    for pos, test_case in enumerate(zip(circuits, targets)):
        circuit, target = test_case
        if hex_counts:
            # Don't use get_counts method which converts hex
            output = result.data(circuit)["counts"]
        else:
            # Use get counts method which converts hex
            output = result.get_counts(circuit)
        assertDictAlmostEqual(output, target, delta=delta)


def cx_gate_circuits_deterministic(final_measure=True):
    """CX-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(X^I), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(I^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(I^X), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(X^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(X^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.cx(qr[0], qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(X^X), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.cx(qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def cx_gate_statevector_deterministic():
    """CX-gate test circuits with deterministic counts."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX10, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX01.(X^I), |10> state
    targets.append(np.array([0, 0, 1, 0]))
    # CX10.(I^X), |01> state
    targets.append(np.array([0, 1, 0, 0]))
    # CX01.(I^X), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    # CX10.(X^I), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    # CX01.(X^X), |01> state
    targets.append(np.array([0, 1, 0, 0]))
    # CX10.(X^X), |10> state
    targets.append(np.array([0, 0, 1, 0]))
    return targets


def cx_gate_unitary_deterministic():
    """CX-gate circuits reference unitaries."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0]]))
    # CX10, |00> state
    targets.append(np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]]))
    # CX01.(X^I), |10> state
    targets.append(np.array([[0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]]))
    # CX10.(I^X), |01> state
    targets.append(np.array([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]))
    # CX01.(I^X), |11> state
    targets.append(np.array([[0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [1, 0, 0, 0]]))
    # CX10.(X^I), |11> state
    targets.append(np.array([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0]]))
    # CX01.(X^X), |01> state
    targets.append(np.array([[0, 0, 0, 1],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0]]))
    # CX10.(X^X), |10> state
    targets.append(np.array([[0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0]]))
    return targets


def compare_statevector(result, circuits, targets,
                        global_phase=True, places=None):
    """Compare final statevectors to targets."""
    for pos, test_case in enumerate(zip(circuits, targets)):
        circuit, target = test_case
        output = result.get_statevector(circuit)
        msg = ("Circuit ({}/{}):".format(pos + 1, len(circuits)) +
               " {} != {}".format(output, target))
        assertAlmostEqual(norm(output - target), 0, places=places,
                          msg=msg)


def compare_unitary(result, circuits, targets,
                    global_phase=True, places=None):
    """Compare final unitary matrices to targets."""
    for pos, test_case in enumerate(zip(circuits, targets)):
        circuit, target = test_case
        output = result.get_unitary(circuit)
        msg = ("Circuit ({}/{}):".format(pos + 1, len(circuits)) +
               " {} != {}".format(output, target))
        if (global_phase):
            # Test equal including global phase
            assertAlmostEqual(norm(output - target), 0,
                              places=places, msg=msg)
        else:
            # Test equal ignorning global phase
            delta = np.trace(
                np.dot(np.conj(np.transpose(output)), target)) - len(output)
            assertAlmostEqual(delta, 0, places=places)

def model_and_pi_schedule():
    """Return a simple model and schedule for pulse simulation"""

    # construct model
    model = duffing_system_model(dim_oscillators=2,
                                 oscillator_freqs=[5.0],
                                 anharm_freqs=[0],
                                 drive_strengths=[1.0],
                                 coupling_dict={},
                                 dt=1.0)

    # note: parameters set so that area under curve is 1/4
    gauss_pulse = Gaussian(duration=10,
                amp=(1.0/4)/2.506627719963857,
                sigma=1)

    # construct schedule
    schedule = Schedule(name='test_sched')
    schedule |= Play(gauss_pulse, DriveChannel(0))
    schedule += Acquire(10, AcquireChannel(0), MemorySlot(0)) << schedule.duration

    return model, schedule

if __name__ == '__main__':
    # Run qasm simulator
    shots = 2000
    circuits = grovers_circuit(final_measure=True, allow_sampling=True)
    targets = [{'0x0': 5 * shots / 8, '0x1': shots / 8,
                '0x2': shots / 8, '0x3': shots / 8}]
    simulator = QasmSimulator()
    qobj = assemble(transpile(circuits, simulator), simulator, shots=shots)
    result = simulator.run(qobj).result()
    assert result.status == 'COMPLETED'
    compare_counts(result, circuits, targets, delta=0.05 * shots)
    assert result.success is True

    # Run statevector simulator
    circuits = cx_gate_circuits_deterministic(final_measure=False)
    targets = cx_gate_statevector_deterministic()
    job = execute(circuits, StatevectorSimulator(), shots=1)
    result = job.result()
    assert result.status == 'COMPLETED'
    assert result.success is True
    compare_statevector(result, circuits, targets)

    # Run unitary simulator
    circuits = cx_gate_circuits_deterministic(final_measure=False)
    targets = cx_gate_unitary_deterministic()
    job = execute(circuits, UnitarySimulator(), shots=1,
                  basis_gates=['u1', 'u2', 'u3', 'cx'])
    result = job.result()
    assert result.status == 'COMPLETED'
    assert result.success is True
    compare_unitary(result, circuits, targets)

    # Run pulse simulator
    system_model, schedule = model_and_pi_schedule()
    backend_sim = PulseSimulator()
    qobj = assemble([schedule],
                    backend=backend_sim,
                    qubit_lo_freq=[5.0],
                    meas_level=1,
                    meas_return='avg',
                    shots=1)
    results = backend_sim.run(qobj, system_model).result()
    state = results.get_statevector(0)
    assertAlmostEqual(state[0], 0, delta=10**-5)
    assertAlmostEqual(state[1], -1j, delta=10**-5)
