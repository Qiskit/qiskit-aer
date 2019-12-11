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
# pylint: disable=eval-used, exec-used, invalid-name

"PulseModel class for PulseSimulator"

from collections import OrderedDict
import numpy as np
import numpy.linalg as la
from .qobj.opparse import HamiltonianParser
# pylint: disable=no-name-in-module,import-error
from .qobj import op_qobj as op


class HamiltonianModel():
    """Hamiltonian model for pulse simulator."""
    def __init__(self, hamiltonian, qubits=None):
        """Initialize a Hamiltonian model.

        Args:
            hamiltonian (dict): Hamiltonian dictionary.
            qubits (list or None): List of qubits to extract from the hamiltonian.

        Raises:
            ValueError: if arguments are invalid.
        """
        # Initialize internal variables
        # The system Hamiltonian in numerical format
        self._system = None
        # System variables
        self._vars = None
        # Channels in the Hamiltonian string
        # These tell the order in which the channels are evaluated in
        # the RHS solver.
        self._channels = None
        # Diagonal elements of the hamiltonian
        self._h_diag = None
        # Eigenvalues of the time-independent hamiltonian
        self._evals = None
        # Eigenstates of the time-indepedent hamiltonian
        self._estates = None
        # Qubit subspace dimensinos
        self._dim_qub = {}
        # Oscillator subspace dimensions
        self._dim_osc = {}

        # Parse Hamiltonian
        # TODO: determine n_qubits from hamiltonian if qubits is None
        n_qubits = len(qubits) if qubits else None
        if not n_qubits:
            raise ValueError("TODO: Need to infer n_qubits from "
                             "Hamiltonian if qubits list is not specified")

        self._vars = OrderedDict(hamiltonian['vars'])

        # Get qubit subspace dimensions
        if 'qub' in hamiltonian:
            self._dim_qub = {int(key): val for key, val in hamiltonian['qub'].items()}
        else:
            self._dim_qub = {}.fromkeys(range(n_qubits), 2)

        # Get oscillator subspace dimensions
        if 'osc' in hamiltonian:
            self._dim_osc = {int(key): val for key, val in hamiltonian['osc'].items()}

        # Step 1: Parse the Hamiltonian
        system = HamiltonianParser(h_str=hamiltonian['h_str'],
                                   dim_osc=self._dim_osc,
                                   dim_qub=self._dim_qub)
        system.parse(qubits)
        self._system = system.compiled

        # Step #2: Determine Hamiltonian channels
        self._get_hamiltonian_channels()

        # Step 3: Calculate diagonal hamiltonian
        self._get_diag_hamiltonian()

    def initial_state(self):
        """Return the initial state of the time-independent hamiltonian"""
        # Set initial state
        ground_state = 0 * op.basis(len(self._evals), 1)
        for idx, estate_coef in enumerate(self._estates[:, 0]):
            ground_state += estate_coef * op.basis(len(self._evals), idx)
        return ground_state

    def calculate_frequencies(self, qubit_lo_freq=None, u_channel_lo=None):
        """Calulate frequencies"""
        # Setup freqs for the channels
        freqs = OrderedDict()

        # Set qubit frequencies from hamiltonian
        if not qubit_lo_freq or (qubit_lo_freq == 'from_hamiltonian'
                                 and len(self._dim_osc) == 0):
            qubit_lo_freq = np.zeros(len(self._dim_qub))
            min_eval = np.min(self._evals)
            for q_idx in range(len(self._dim_qub)):
                single_excite = _first_excited_state(q_idx, self._dim_qub)
                dressed_eval = _eval_for_max_espace_overlap(single_excite,
                                                            self._evals,
                                                            self._estates)
                qubit_lo_freq[q_idx] = (dressed_eval - min_eval) / (2 * np.pi)

        # TODO: set u_channel_lo from hamiltonian
        if not u_channel_lo:
            raise ValueError("u_channel_lo cannot be None.")

        # Set frequencies
        for key in self._channels.keys():
            chidx = int(key[1:])
            if key[0] == 'D':
                freqs[key] = qubit_lo_freq[chidx]
            elif key[0] == 'U':
                freqs[key] = 0
                for u_lo_idx in u_channel_lo[chidx]:
                    if u_lo_idx['q'] < len(qubit_lo_freq):
                        qfreq = qubit_lo_freq[u_lo_idx['q']]
                        qscale = u_lo_idx['scale'][0]
                        freqs[key] += qfreq * qscale
            else:
                raise ValueError("Channel is not D or U")
        return freqs

    def _get_hamiltonian_channels(self):
        """ Get all the qubit channels D_i and U_i in the string
        representation of a system Hamiltonian.

        Raises:
            Exception: Missing index on channel.
        """
        channels = []
        for _, ham_str in self._system:
            chan_idx = [i for i, letter in enumerate(ham_str) if
                        letter in ['D', 'U']]
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

    def _get_diag_hamiltonian(self):
        """ Get the diagonal elements of the hamiltonian and get the
        dressed frequencies and eigenstates

        Raises:
            Exception: Missing index on channel.
        """
        # Get the diagonal elements of the hamiltonian with all the
        # drive terms set to zero
        for chan in self._channels:
            exec('%s=0' % chan)

        # might be a better solution to replace the 'var' in the hamiltonian
        # string with 'op_system.vars[var]'
        for var in self._vars:
            exec('%s=%f' % (var, self._vars[var]))

        H_full = np.zeros(np.shape(self._system[0][0].full()), dtype=complex)

        for hpart in self._system:
            H_full += hpart[0].full() * eval(hpart[1])

        evals, estates = la.eigh(H_full)

        eval_mapping = []
        for ii in range(len(evals)):
            eval_mapping.append(np.argmax(np.abs(estates[:, ii])))

        # Remap eigenvalues and eigenstates
        evals_mapped = evals.copy()
        estates_mapped = estates.copy()

        for ii, val in enumerate(eval_mapping):
            evals_mapped[val] = evals[ii]
            estates_mapped[:, val] = estates[:, ii]

        self._evals = evals_mapped
        self._estates = estates_mapped
        self._h_diag = np.ascontiguousarray(np.diag(H_full).real)


def _first_excited_state(qubit_idx, dim_qub):
    """
    Returns the vector corresponding to all qubits in the 0 state, except for
    qubit_idx in the 1 state.

    Assumption: the keys in dim_qub consist exactly of the str version of the int
                in range(len(dim_qub)). They don't need to be in order, but they
                need to be of this format

    Parameters:
        qubit_idx (int): the qubit to be in the 1 state

        dim_qub (dict): a dictionary with keys being qubit index as a string, and
                        value being the dimension of the qubit

    Returns:
        vector: the state with qubit_idx in state 1, and the rest in state 0
    """
    vector = np.array([1.])

    # iterate through qubits, tensoring on the state
    for idx, dim in enumerate(dim_qub):
        new_vec = np.zeros(dim)
        if idx == qubit_idx:
            new_vec[1] = 1
        else:
            new_vec[0] = 1

        vector = np.kron(new_vec, vector)

    return vector


def _eval_for_max_espace_overlap(u, evals, evecs, decimals=14):
    """ Given an eigenvalue decomposition evals, evecs, as output from
    get_diag_hamiltonian, returns the eigenvalue from evals corresponding
    to the eigenspace that the vector vec has the maximum overlap with.

    Parameters:
        u (numpy.array): the vector of interest

        evals (numpy.array): list of eigenvalues

        evecs (numpy.array): eigenvectors corresponding to evals

        decimals (int): rounding option, to try to handle numerical
                        error if two evals should be the same but are
                        slightly different

    Returns:
        eval: eigenvalue corresponding to eigenspace for which vec has
              maximal overlap

    Raises:
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


def _proj_norm(A, b):
    """
    Given a matrix A and vector b, computes the norm of the projection of
    b onto the column space of A using least squares.

    Note: A can also be specified as a 1d numpy.array, in which case it will
    convert it into a matrix with one column

    Parameters:
        A (numpy.array): 2d array, a matrix

        b (numpy.array): 1d array, a vector

    Returns:
        norm: the norm of the projection

    Raises:
    """

    # if specified as a single vector, turn it into a column vector
    if A.ndim == 1:
        A = np.array([A]).T

    x = la.lstsq(A, b, rcond=None)[0]

    return la.norm(A@x)
