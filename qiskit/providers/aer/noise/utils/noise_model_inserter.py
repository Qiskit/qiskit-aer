# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Noise model inserter module
The goal of this module is to add QuantumError gates (Kraus gates) to a circuit
based on a given noise model.
"""
import qiskit.compiler


def insert_noise(circuits, noise_model, transpile=False):
    """Return a noisy version of a QuantumCircuit.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): Input noise-free circuits.
        noise_model (NoiseModel):  The noise model containing the errors to add
        transpile (Boolean): Should the circuit be transpiled into the noise model basis gates

    Returns:
        QuantumCircuit: The new circuit with the Kraus noise instructions inserted.

    Additional Information:
        The noisy circuit return by this function will consist of the
        original circuit with ``Kraus`` instructions inserted after all
        instructions referenced in the ``noise_model``. The resulting circuit
        cannot be ran on a quantum computer but can be executed on the
        :class:`~qiskit.providers.aer.QasmSimulator`.
    """
    is_circuits_list = isinstance(circuits, (list, tuple))
    circuits = circuits if is_circuits_list else [circuits]
    result_circuits = []
    errors = noise_model._default_quantum_errors
    for circuit in circuits:
        if transpile:
            transpiled_circuit = qiskit.compiler.transpile(circuit,
                                                           basis_gates=noise_model.basis_gates)
        else:
            transpiled_circuit = circuit
        result_circuit = circuit.copy(name=transpiled_circuit.name + '_with_noise')
        result_circuit.data = []
        for inst, qargs, cargs in transpiled_circuit.data:
            result_circuit.data.append((inst, qargs, cargs))
            if inst.name in errors.keys():
                error = errors[inst.name]
                result_circuit.append(error.to_instruction(), qargs)
        result_circuits.append(result_circuit)
    return result_circuits if is_circuits_list else result_circuits[0]
