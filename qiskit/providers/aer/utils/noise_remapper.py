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
Remap qubits in a NoiseModel.
"""

import logging

from ..noise.noise_model import NoiseModel
from ..noise.noiseerror import NoiseError

logger = logging.getLogger(__name__)


def remap_noise_model(noise_model, remapping, discard_qubits=False, warnings=True):
    """Remap qubits in a noise model.

    This remaps the specified gate qubits for local quantum errors, the gate
    and noise qubits for non-local quantum errors, and the gate qubits for
    local ReadoutErrors. All-qubit quantum and readout errors are unaffected.

    Args:
        noise_model (NoiseModel): a noise model to remap qubits.
        remapping (list): list or remappings of old qubit to new qubit.
                          See Additional Information.
        discard_qubits (bool): if True discard qubits not in remapping keys,
                               if False an identity mapping wil be assumed
                               for unnamed qubits (Default: False).
        warnings (bool): display warnings if qubits being remapped are not
                         in the input noise model (Default: True).

    Returns:
        NoiseModel: a new noise model with the same errors but remapped
        gate and noise qubits for local and non-local errors.

    Raises:
        NoiseError: if remapping has duplicate qubits in the remapped qubits.

    Additional Information:
        * The remapping map be specified as either a list of pairs:
          ``[(old, new), ...]``, or a list of old qubits where the new qubit is
          inferred from the position: ``[old0, old1, ...]`` is treated as
          ``[(old0, 0), (old1, 1), ...]``.

        * If ``discard_qubits`` is ``False``, any qubits in the noise model not
          specified in the list of old qubits will be added to the remapping as
          a trivial mapping ``(qubit, qubit)``.
    """
    if not isinstance(noise_model, NoiseModel):
        raise NoiseError("Input must be a NoiseModel.")

    if warnings:
        # Warning if remapped qubit isn't in noise model
        for qubit in remapping:
            if qubit not in noise_model.noise_qubits:
                logger.warning('Warning: qubit %s is not in noise model', qubit)

    # Convert remapping into a inverse mapping dict
    inv_map = {}
    for pos, item in enumerate(remapping):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            inv_map[int(item[0])] = int(item[1])
        elif isinstance(item, int):
            inv_map[item] = pos
    # Add noise model qubits not in mapping as identity mapping
    if not discard_qubits:
        for qubit in noise_model.noise_qubits:
            if qubit not in inv_map:
                inv_map[qubit] = qubit

    # Check mapping doesn't have duplicate qubits in output
    new_qubits = list(inv_map.values())
    if len(set(new_qubits)) != len(new_qubits):
        raise NoiseError('Duplicate qubits in remapping: {}'.format(inv_map))

    # Convert noise model to dict
    nm_dict = noise_model.to_dict()

    # Indexes of errors to keep
    new_errors = []
    for error in nm_dict['errors']:
        keep_error = True
        gate_qubits = error.get('gate_qubits', [])
        noise_qubits = error.get('noise_qubits', [])
        for qubits in gate_qubits + noise_qubits:
            for qubit in qubits:
                if qubit not in inv_map:
                    keep_error = False
                    break

        # If any qubits were not in the full_mapping we discard error
        if keep_error:
            new_gate_qubits = gate_qubits
            for i, qubits in enumerate(gate_qubits):
                for j, qubit in enumerate(qubits):
                    new_gate_qubits[i][j] = inv_map[qubit]

            # Otherwise we remap the noise and gate qubits
            new_noise_qubits = noise_qubits
            for i, qubits in enumerate(noise_qubits):
                for j, qubit in enumerate(qubits):
                    new_noise_qubits[i][j] = inv_map[qubit]
            if new_gate_qubits:
                error['gate_qubits'] = new_gate_qubits
            if new_noise_qubits:
                error['noise_qubits'] = new_noise_qubits
            new_errors.append(error)

    # Update errors and convert back to NoiseModel
    nm_dict['errors'] = new_errors
    new_noise_model = NoiseModel.from_dict(nm_dict)

    # Update basis gates from original model
    new_noise_model._basis_gates = noise_model._basis_gates
    return new_noise_model
