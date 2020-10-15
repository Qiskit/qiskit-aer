# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=eval-used, exec-used, invalid-name, missing-return-type-doc

"HamiltonianModel class for system specification for the PulseSimulator"

from collections import OrderedDict
import numpy as np
import numpy.linalg as la
from ...aererror import AerError
from .string_model_parser.string_model_parser import HamiltonianParser


class HamiltonianModel():
    """Hamiltonian model for pulse simulator."""

    def __init__(self,
                 system=None,
                 variables=None,
                 subsystem_dims=None):
        """Initialize a Hamiltonian model.

        Args:
            system (list): List of Qobj objects representing operator form of the Hamiltonian.
            variables (OrderedDict): Ordered dict for parameter values in Hamiltonian.
            subsystem_dims (dict): dict of subsystem dimensions.

        Raises:
            AerError: if arguments are invalid.
        """

        # Initialize internal variables
        # The system Hamiltonian in numerical format
        self._system = system
        # System variables
        self._variables = variables
        # Channels in the Hamiltonian string
        # Qubit subspace dimensinos
        self._subsystem_dims = subsystem_dims or {}

        # The rest are computed from the previous

        # These tell the order in which the channels are evaluated in
        # the RHS solver.
        self._channels = None
        # Diagonal elements of the hamiltonian
        self._h_diag = None
        # Eigenvalues of the time-independent hamiltonian
        self._evals = None
        # Eigenstates of the time-indepedent hamiltonian
        self._estates = None

        # populate self._channels
        self._calculate_hamiltonian_channels()

        if len(self._channels) == 0:
            raise AerError('HamiltonianModel must contain channels to simulate.')

        # populate self._h_diag, self._evals, self._estates
        self._compute_drift_data()

    @classmethod
    def from_dict(cls, hamiltonian, subsystem_list=None):
        """Initialize from a Hamiltonian string specification.

        Args:
            hamiltonian (dict): dictionary representing Hamiltonian in string specification.
            subsystem_list (list or None): List of subsystems to extract from the hamiltonian.

        Returns:
            HamiltonianModel: instantiated from hamiltonian dictionary

        Raises:
            ValueError: if arguments are invalid.
        """

        _hamiltonian_pre_parse_exceptions(hamiltonian)

        # get variables
        variables = OrderedDict()
        if 'vars' in hamiltonian:
            variables = OrderedDict(hamiltonian['vars'])

        # Get qubit subspace dimensions
        if 'qub' in hamiltonian:
            if subsystem_list is None:
                subsystem_list = [int(qubit) for qubit in hamiltonian['qub']]
            else:
                # if user supplied, make a copy and sort it
                subsystem_list = subsystem_list.copy()
                subsystem_list.sort()

            # force keys in hamiltonian['qub'] to be ints
            qub_dict = {
                int(key): val
                for key, val in hamiltonian['qub'].items()
            }

            subsystem_dims = {
                int(qubit): qub_dict[int(qubit)]
                for qubit in subsystem_list
            }
        else:
            subsystem_dims = {}

        # Get oscillator subspace dimensions
        if 'osc' in hamiltonian:
            oscillator_dims = {
                int(key): val
                for key, val in hamiltonian['osc'].items()
            }
        else:
            oscillator_dims = {}

        # Parse the Hamiltonian
        system = HamiltonianParser(h_str=hamiltonian['h_str'],
                                   dim_osc=oscillator_dims,
                                   dim_qub=subsystem_dims)
        system.parse(subsystem_list)
        system = system.compiled

        return cls(system, variables, subsystem_dims)

    def get_qubit_lo_from_drift(self):
        """ Computes a list of qubit frequencies corresponding to the exact energy
        gap between the ground and first excited states of each qubit.

        If the keys in self._subsystem_dims skips over a qubit, it will default to outputting
        a 0 frequency for that qubit.

        Returns:
            qubit_lo_freq (list): the list of frequencies
        """
        # need to specify frequencies up to max qubit index
        qubit_lo_freq = [0] * (max(self._subsystem_dims.keys()) + 1)

        # compute difference between first excited state of each qubit and
        # the ground energy
        min_eval = np.min(self._evals)
        for q_idx in self._subsystem_dims.keys():
            single_excite = _first_excited_state(q_idx, self._subsystem_dims)
            dressed_eval = _eval_for_max_espace_overlap(
                single_excite, self._evals, self._estates)
            qubit_lo_freq[q_idx] = (dressed_eval - min_eval) / (2 * np.pi)

        return qubit_lo_freq

    def _calculate_hamiltonian_channels(self):
        """ Get all the qubit channels D_i and U_i in the string
        representation of a system Hamiltonian.

        Raises:
            Exception: Missing index on channel.
        """
        channels = []
        for _, ham_str in self._system:
            chan_idx = [
                i for i, letter in enumerate(ham_str) if letter in ['D', 'U']
            ]
            for ch in chan_idx:
                if (ch + 1) == len(ham_str) or not ham_str[ch + 1].isdigit():
                    raise Exception('Channel name must include' +
                                    'an integer labeling the qubit.')
            for kk in chan_idx:
                done = False
                offset = 0
                while not done:
                    offset += 1
                    if not ham_str[kk + offset].isdigit():
                        done = True
                    # In case we hit the end of the string
                    elif (kk + offset + 1) == len(ham_str):
                        done = True
                        offset += 1
                temp_chan = ham_str[kk:kk + offset]
                if temp_chan not in channels:
                    channels.append(temp_chan)
        channels.sort(key=lambda x: (int(x[1:]), x[0]))

        channel_dict = OrderedDict()
        for idx, val in enumerate(channels):
            channel_dict[val] = idx

        self._channels = channel_dict

    def _compute_drift_data(self):
        """Calculate the the drift Hamiltonian.

        This computes the dressed frequencies and eigenstates of the
        diagonal part of the Hamiltonian.

        Raises:
            Exception: Missing index on channel.
        """
        # Get the diagonal elements of the hamiltonian with all the
        # drive terms set to zero
        for chan in self._channels:
            exec('%s=0' % chan)

        # might be a better solution to replace the 'var' in the hamiltonian
        # string with 'op_system.vars[var]'
        for var in self._variables:
            exec('%s=%f' % (var, self._variables[var]))

        full_dim = np.prod(list(self._subsystem_dims.values()))

        ham_full = np.zeros((full_dim, full_dim), dtype=complex)
        for ham_part in self._system:
            ham_full += ham_part[0].full() * eval(ham_part[1])
        # Remap eigenvalues and eigenstates
        evals, estates = la.eigh(ham_full)

        evals_mapped = np.zeros(evals.shape, dtype=evals.dtype)
        estates_mapped = np.zeros(estates.shape, dtype=estates.dtype)

        # order the eigenvalues and eigenstates according to overlap with computational basis
        pos_list = []
        min_overlap = 1
        for i, estate in enumerate(estates.T):
            # make a copy and set entries with indices in pos_list to 0
            estate_copy = estate.copy()
            estate_copy[pos_list] = 0

            pos = np.argmax(np.abs(estate_copy))
            pos_list.append(pos)
            min_overlap = min(np.abs(estate_copy)[pos]**2, min_overlap)

            evals_mapped[pos] = evals[i]
            estates_mapped[:, pos] = estate

        self._evals = evals_mapped
        self._estates = estates_mapped
        self._h_diag = np.ascontiguousarray(np.diag(ham_full).real)


def _hamiltonian_pre_parse_exceptions(hamiltonian):
    """Raises exceptions for hamiltonian specification.

    Parameters:
        hamiltonian (dict): dictionary specification of hamiltonian
    Returns:
    Raises:
        AerError: if some part of the hamiltonian dictionary is unsupported
    """

    ham_str = hamiltonian.get('h_str', [])
    if ham_str in ([], ['']):
        raise AerError("Hamiltonian dict requires a non-empty 'h_str' entry.")

    if hamiltonian.get('qub', {}) == {}:
        raise AerError("Hamiltonian dict requires non-empty 'qub' entry with subsystem dimensions.")

    if hamiltonian.get('osc', {}) != {}:
        raise AerError('Oscillator-type systems are not supported.')


def _first_excited_state(qubit_idx, subsystem_dims):
    """
    Returns the vector corresponding to all qubits in the 0 state, except for
    qubit_idx in the 1 state.

    Parameters:
        qubit_idx (int): the qubit to be in the 1 state

        subsystem_dims (dict): a dictionary with keys being subsystem index, and
                        value being the dimension of the subsystem

    Returns:
        vector: the state with qubit_idx in state 1, and the rest in state 0
    """
    vector = np.array([1.])
    # iterate through qubits, tensoring on the state
    qubit_indices = [int(qubit) for qubit in subsystem_dims]
    qubit_indices.sort()
    for idx in qubit_indices:
        new_vec = np.zeros(subsystem_dims[idx])
        if idx == qubit_idx:
            new_vec[1] = 1
        else:
            new_vec[0] = 1
        vector = np.kron(new_vec, vector)

    return vector


def _eval_for_max_espace_overlap(u, evals, evecs, decimals=14):
    """Return the eigenvalue for eigenvector closest to input.

    Given an eigenvalue decomposition evals, evecs, as output from
    get_diag_hamiltonian, returns the eigenvalue from evals corresponding
    to the eigenspace that the vector vec has the maximum overlap with.

    Args:
        u (numpy.array): the vector of interest
        evals (numpy.array): list of eigenvalues
        evecs (numpy.array): eigenvectors corresponding to evals
        decimals (int): rounding option, to try to handle numerical
                        error if two evals should be the same but are
                        slightly different

    Returns:
        complex: eigenvalue corresponding to eigenspace for which vec has
        maximal overlap.
    """
    # get unique evals (with rounding for numerical error)
    rounded_evals = evals.copy().round(decimals=decimals)
    unique_evals = np.unique(rounded_evals)

    # compute overlaps to unique evals
    overlaps = np.zeros(len(unique_evals))
    for idx, val in enumerate(unique_evals):
        overlaps[idx] = _proj_norm(evecs[:, val == rounded_evals], u)

    # return eval with largest overlap
    return unique_evals[np.argmax(overlaps)]


def _proj_norm(mat, vec):
    """
    Compute the projection form of a vector an matrix.

    Given a matrix ``mat`` and vector ``vec``, computes the norm of the
    projection of ``vec`` onto the column space of ``mat`` using least
    squares.

    Note: ``mat`` can also be specified as a 1d numpy.array, in which
    case it will convert it into a matrix with one column

    Parameters:
        mat (numpy.array): 2d array, a matrix.
        vec (numpy.array): 1d array, a vector.

    Returns:
        complex: the norm of the projection
    """

    # if specified as a single vector, turn it into a column vector
    if mat.ndim == 1:
        mat = np.array([mat]).T

    lsq_vec = la.lstsq(mat, vec, rcond=None)[0]

    return la.norm(mat @ lsq_vec)
