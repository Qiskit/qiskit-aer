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
Test circuits and reference outputs for snapshot amplitude instructions.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.extensions.snapshot import Snapshot
from qiskit.providers.aer.extensions.snapshot_amplitudes import *
from qiskit.providers.aer.extensions.snapshot_statevector import *

def snapshot_amplitudes_labels_params():
    """Dictionary of labels and params for 3-qubit amplitude snapshots"""
    return {
        "[0]": [0],
        "[7]": [7],
        "[0,1]": [0,1],
        "[7,3,5,1]": [7,3,5,1],
        "[4,1,5]": [4,1,5],
        "[6,2]": [6,2],
        "all": [0,1,2,3,4,5,6,7]
    }


def snapshot_amplitudes_circuits(post_measure=False):
    """Snapshot Amplitudes test circuits"""

    circuits = []
    num_qubits = 3
    qubits = list(range(3))
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    # Amplitudes snapshot instruction acting on all qubits

    # Snapshot |000>
    circuit = QuantumCircuit(*regs)
    if not post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params, qubits)
            circuit.snapshot_statevector(label)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params, qubits)
            circuit.snapshot_statevector(label)

    circuits.append(circuit)

    # Snapshot |111>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    if not post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params, qubits)
            circuit.snapshot_statevector(label)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params, qubits)
            circuit.snapshot_statevector(label)

    circuits.append(circuit)

    # Snapshot 0.25*(|001>+|011>+|100>+|101>)
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.h(2)
    if not post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params, qubits)
            circuit.snapshot_statevector(label)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params, qubits)
            circuit.snapshot_statevector(label)

    circuits.append(circuit)

    return circuits


def snapshot_amplitudes_counts(shots):
    """Snapshot Amplitudes test circuits reference counts."""
    targets = []
    # Snapshot |000>
    targets.append({'0x0': shots})
    # Snapshot |111>
    targets.append({'0x7': shots})
    # Snapshot 0.25*(|001>+|011>+|100>+|101>)
    targets.append({'0x0': shots/4, '0x1': shots/4, '0x4': shots/4, '0x5': shots/4,})
    return targets


def snapshot_amplitudes_pre_measure_amplitudes():
    """Snapshot Amplitudes test circuits reference final amplitudes"""
    targets = []
    amplitudes = {  #amplitudes for circuit 1
    # Amplitude |0> = |000>
    '[0]': array([1], dtype=complex),
    # Amplitude |7> = |111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([1,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([1,0,0,0,0,0,0,0], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 2
    # Amplitude |0> = |000>
    '[0]':array([0], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([1], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([1,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0,0,0,0,0,0,0,1], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 3
    # Amplitude |0> = |000>
    '[0]':array([0.5], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0.5,0.5], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0.5,0.5], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0.5,0.5,0.5], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0.5,0.5,0,0,0.5,0.5,0,0], dtype=complex)
        }
    targets.append(amplitudes)
    
    return targets


def snapshot_amplitudes_post_measure_amplitudes():
    """Snapshot Amplitudes test circuits reference final amplitudes"""
    targets = []
    amplitudes = {  #amplitudes for circuit 1
    # Amplitude |0> = |000>
    '[0]': array([1], dtype=complex),
    # Amplitude |7> = |111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([1,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([1,0,0,0,0,0,0,0], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 2
    # Amplitude |0> = |000>
    '[0]':array([0], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([1], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([1,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0,0,0,0,0,0,0,1], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 3
    # Amplitude |0> = |000>
    '[0]':array([0], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0.5,0.5], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0.5,0.5], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0.5,0.5,0.5], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0.5,0.5,0,0,0.5,0.5,0,0], dtype=complex)
        }
    targets.append(amplitudes)
    
    return targets


def snapshot_amplitudes_pre_measure_amplitudes():
    """Snapshot Amplitudes test circuits reference final amplitudes"""
    targets = []
    amplitudes = {  #amplitudes for circuit 1
    # Amplitude |0> = |000>
    '[0]': array([1], dtype=complex),
    # Amplitude |7> = |111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([1,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([1,0,0,0,0,0,0,0], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 2
    # Amplitude |0> = |000>
    '[0]':array([0], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([1], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([1,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0,0,0,0,0,0,0,1], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 3
    # Amplitude |0> = |000>
    '[0]':array([0.5], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0.5,0.5], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0.5,0.5], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0.5,0.5,0.5], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0.5,0.5,0,0,0.5,0.5,0,0], dtype=complex)
        }
    targets.append(amplitudes)
    
    return targets


def snapshot_amplitudes_post_measure_amplitudes():
    """Snapshot Amplitudes test circuits reference final amplitudes"""
    targets = []
    amplitudes = {  #amplitudes for circuit 1
    # Amplitude |0> = |000>
    '[0]': array([1], dtype=complex),
    # Amplitude |7> = |111>
    '[7]':array([0], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([1,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([0,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([1,0,0,0,0,0,0,0], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 2
    # Amplitude |0> = |000>
    '[0]':array([0], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([1], dtype=complex),
    # Amplitudes[0>, |1> = !000>, |001>
    '[0,1]': array([0,0], dtype=complex),
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': array([1,0,0,0], dtype=complex),
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': array([0,0,0], dtype=complex),
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':array([0,0,0,0,0,0,0,1], dtype=complex)
        }
    targets.append(amplitudes)

    amplitudes = {  #amplitudes for circuit 3
    # Amplitude |0> = |000>
    '[0]':array([0], dtype=complex),
     # Amplitude |7> =|111>
    '[7]':array([0], dtype=complex),
        
    # Amplitudes[0>, |1> = |000>, |001>
    '[0,1]': [[1,0,0],
              [0,1.0]],
    
    # Amplitudes[7>,|3>,|5>,|1> = !111>,|011>,!101>,|001>
    '[7,3,5,1]': [array([0,0,1,0], dtype=complex),
                  array([0,0,0,1], dtype=complex)],
        
    # Amplitudes[4>,|1>,|5> = !100>,|001>,!101>
    '[4,1,5]': [array([1,0,0], dtype=complex),
                array([0,1,0], dtype=complex),
                array([0,0,1], dtype=complex)],
    # Amplitudes[6>, |2> = !110>, |010>
    '[6,2]': array([0,0], dtype=complex),
    # All amplitudes
    'all':[array([1,0,0,0,0,0,0,0], dtype=complex),
           array([0,1,0,0,0,0,0,0], dtype=complex),
           array([0,0,0,0,1,0,0,0], dtype=complex),
           array([1,0,0,0,0,1,0,0], dtype=complex)]
        }
    targets.append(amplitudes)
    
    return targets

