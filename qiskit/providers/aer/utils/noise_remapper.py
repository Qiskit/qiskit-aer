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
from warnings import warn

from ..noise.noise_model import NoiseModel
from ..noise.noiseerror import NoiseError

logger = logging.getLogger(__name__)


def remap_noise_model(noise_model, remapping, discard_qubits=False, warnings=True):
    """[Deprecated] Remap qubits in a noise model.

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
    warn(
        'The "remap_noise_model" function has been deprecated as of qiskit-aer 0.10.0'
        ' and will be removed no earlier than 3 months from that release date.',
        DeprecationWarning, stacklevel=2)

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

    # qubits not to be remapped
    discarded_qubits = set(noise_model.noise_qubits) - set(inv_map.keys())

    def in_target(qubits):  # check if qubits are to be remapped or not
        if not discard_qubits:
            return True
        for q in qubits:
            if q in discarded_qubits:
                return False
        return True

    def remapped(qubits):
        return tuple(inv_map[q] for q in qubits)

    # Copy from original noise model
    new_model = NoiseModel()
    new_model._basis_gates = noise_model._basis_gates
    # No remapping for default errors
    for inst_name, noise in noise_model._default_quantum_errors.items():
        new_model.add_all_qubit_quantum_error(noise, inst_name)
    if noise_model._default_readout_error:
        new_model.add_all_qubit_readout_error(noise_model._default_readout_error)
    # Remapping
    for inst_name, noise_dic in noise_model._local_quantum_errors.items():
        for qubits, noise in noise_dic.items():
            if in_target(qubits):
                new_model.add_quantum_error(noise, inst_name, remapped(qubits))
    for inst_name, outer_dic in noise_model._nonlocal_quantum_errors.items():
        for qubits, inner_dic in outer_dic.items():
            if in_target(qubits):
                for noise_qubits, noise in inner_dic.items():
                    if in_target(noise_qubits):
                        new_model.add_nonlocal_quantum_error(
                            noise, inst_name, remapped(qubits), remapped(noise_qubits)
                        )
    for qubits, noise in noise_model._local_readout_errors.items():
        if in_target(qubits):
            new_model.add_readout_error(noise, remapped(qubits))
    return new_model
