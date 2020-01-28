# -*- coding: utf-8 -*-

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
# pylint: disable=eval-used, exec-used, invalid-name, missing-return-type-doc

"HamiltonianModel class for system specification for the PulseSimulator"

from collections import OrderedDict
import numpy as np
import numpy.linalg as la
from .qobj.opparse import HamiltonianParser
from ..aererror import AerError


class HamiltonianModel():
    """Hamiltonian model for pulse simulator."""

    def __init__(self,
                 system=None,
                 variables=None,
                 qubit_dims=None,
                 oscillator_dims=None):
        """Initialize a Hamiltonian model.

        Args:
            system (list): List of Qobj objects representing operator form of the Hamiltonian.
            variables (OrderedDict): Ordered dict for parameter values in Hamiltonian.
            qubit_dims (dict): dict of qubit dimensions.
            oscillator_dims (dict): dict of oscillator dimensions.

        Raises:
            ValueError: if arguments are invalid.
        """

        # Initialize internal variables
        # The system Hamiltonian in numerical format
        self._system = system
        # System variables
        self._variables = variables
        # Channels in the Hamiltonian string
        # Qubit subspace dimensinos
        self._qubit_dims = qubit_dims or {}
        # Oscillator subspace dimensions
        self._oscillator_dims = oscillator_dims or {}

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

        # populate self._h_diag, self._evals, self._estates
        self._compute_drift_data()

    @classmethod
    def from_dict(cls, hamiltonian, qubit_list=None):
        """Initialize from a Hamiltonian string specification.

        Args:
            hamiltonian (dict): dictionary representing Hamiltonian in string specification.
            qubit_list (list or None): List of qubits to extract from the hamiltonian.

        Returns:
            HamiltonianModel: instantiated from hamiltonian dictionary

        Raises:
            ValueError: if arguments are invalid.
        """

        _hamiltonian_parse_exceptions(hamiltonian)

        # get variables
        variables = OrderedDict(hamiltonian['vars'])

        # Get qubit subspace dimensions
        if 'qub' in hamiltonian:
            if qubit_list is None:
                qubit_list = [int(qubit) for qubit in hamiltonian['qub']]

            qubit_dims = {
                int(key): val
                for key, val in hamiltonian['qub'].items()
            }
        else:
            qubit_dims = {}

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
                                   dim_qub=qubit_dims)
        system.parse(qubit_list)
        system = system.compiled

        return cls(system, variables, qubit_dims, oscillator_dims)

    def get_qubit_lo_from_drift(self):
        """ Computes a list of qubit frequencies corresponding to the exact energy
        gap between the ground and first excited states of each qubit.

        Returns:
            qubit_lo_freq (list): the list of frequencies
        """
        qubit_lo_freq = [0] * len(self._qubit_dims)

        # compute difference between first excited state of each qubit and
        # the ground energy
        min_eval = np.min(self._evals)
        for q_idx in range(len(self._qubit_dims)):
            single_excite = _first_excited_state(q_idx, self._qubit_dims)
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

        ham_full = np.zeros(np.shape(self._system[0][0].full()), dtype=complex)
        for ham_part in self._system:
            ham_full += ham_part[0].full() * eval(ham_part[1])
        # Remap eigenvalues and eigenstates
        evals, estates = la.eigh(ham_full)

        evals_mapped = np.zeros(evals.shape, dtype=evals.dtype)
        estates_mapped = np.zeros(estates.shape, dtype=estates.dtype)

        for i, estate in enumerate(estates.T):
            pos = np.argmax(np.abs(estate))
            evals_mapped[pos] = evals[i]
            estates_mapped[:, pos] = estate

        self._evals = evals_mapped
        self._estates = estates_mapped
        self._h_diag = np.ascontiguousarray(np.diag(ham_full).real)


def _hamiltonian_parse_exceptions(hamiltonian):
    """Raises exceptions for hamiltonian specification.

    Parameters:
        hamiltonian (dict): dictionary specification of hamiltonian
    Returns:
    Raises:
        AerError: if some part of the hamiltonian dictionary is unsupported
    """
    if hamiltonian.get('osc', {}) != {}:
        raise AerError('Oscillator-type systems are not supported.')


def _first_excited_state(qubit_idx, qubit_dims):
    """
    Returns the vector corresponding to all qubits in the 0 state, except for
    qubit_idx in the 1 state.

    Assumption: the keys in dim_qub consist exactly of the str version of the int
                in range(len(dim_qub)). They don't need to be in order, but they
                need to be of this format

    Parameters:
        qubit_idx (int): the qubit to be in the 1 state

        qubit_dims (dict): a dictionary with keys being qubit index, and
                        value being the dimension of the qubit

    Returns:
        vector: the state with qubit_idx in state 1, and the rest in state 0
    """
    vector = np.array([1.])
    # iterate through qubits, tensoring on the state
    qubit_indices = [int(qubit) for qubit in qubit_dims]
    qubit_indices.sort()
    for idx in qubit_indices:
        new_vec = np.zeros(qubit_dims[idx])
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
