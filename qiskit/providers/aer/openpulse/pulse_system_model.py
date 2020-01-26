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

"System Model class for system specification for the PulseSimulator"

from warnings import warn
from collections import OrderedDict
from qiskit.providers import BaseBackend
from .hamiltonian_model import HamiltonianModel
from ..aererror import AerError


class PulseSystemModel():
    """PulseSystemModel containing all model parameters necessary for simulation.
    """
    def __init__(self,
                 hamiltonian=None,
                 qubit_freq_est=None,
                 meas_freq_est=None,
                 u_channel_lo=None,
                 control_channel_labels=None,
                 qubit_list=None,
                 dt=None):
        """Basic constructor.

        Raises:
            AerError: if hamiltonian is not None or a HamiltonianModel
        """

        # default type values
        self._qubit_freq_est = qubit_freq_est
        self._meas_freq_est = meas_freq_est

        # necessary values
        if hamiltonian is not None and not isinstance(hamiltonian, HamiltonianModel):
            raise AerError("hamiltonian must be a HamiltonianModel object")
        self.hamiltonian = hamiltonian
        self.u_channel_lo = u_channel_lo
        self.control_channel_labels = control_channel_labels or []
        self.qubit_list = qubit_list
        self.dt = dt

    @classmethod
    def from_backend(cls, backend, qubit_list=None):
        """Returns a PulseSystemModel constructed from a backend object.

        Args:
            backend (Backend): backend object to draw information from.
            qubit_list (list): a list of ints for which qubits to include in the model.

        Returns:
            PulseSystemModel: the PulseSystemModel constructed from the backend.

        Raises:
            AerError: If channel or u_channel_lo are invalid.
        """

        if not isinstance(backend, BaseBackend):
            raise AerError("{} is not a Qiskit backend".format(backend))

        # get relevant information from backend
        defaults = backend.defaults().to_dict()
        config = backend.configuration().to_dict()

        if not config['open_pulse']:
            raise AerError('{} is not an open pulse backend'.format(backend))

        # draw defaults
        qubit_freq_est = defaults.get('qubit_freq_est', None)
        meas_freq_est = defaults.get('meas_freq_est', None)

        # draw from configuration
        # if no qubit_list, use all for device
        qubit_list = qubit_list or list(range(config['n_qubits']))
        ham_string = config['hamiltonian']
        hamiltonian = HamiltonianModel.from_dict(ham_string, qubit_list)
        u_channel_lo = config.get('u_channel_lo', None)
        dt = config.get('dt', None)

        control_channel_labels = [None] * len(u_channel_lo)
        # populate control_channel_dict
        # for now it assumes the u channel drives a single qubit, and we return the index
        # of the driven qubit, along with the frequency description
        if u_channel_lo is not None:
            for u_idx, u_lo in enumerate(u_channel_lo):
                # find drive index
                drive_idx = None
                while drive_idx is None:
                    u_str_label = 'U{0}'.format(str(u_idx))
                    for h_term_str in ham_string['h_str']:
                        # check if this string corresponds to this u channel
                        if u_str_label in h_term_str:
                            # get index of X operator drive term
                            x_idx = h_term_str.find('X')
                            # if 'X' is found, and is not at the end of the string, drive_idx
                            # is the subsequent character
                            if x_idx != -1 and x_idx + 1 < len(h_term_str):
                                drive_idx = int(h_term_str[x_idx + 1])

                if drive_idx is not None:
                    # construct string for u channel
                    u_string = ''
                    for u_term_dict in u_lo:
                        scale = u_term_dict.get('scale', [1.0, 0])
                        q_idx = u_term_dict.get('q')
                        if len(u_string) > 0:
                            u_string += ' + '
                        u_string += str(scale[0] + scale[1] * 1j) + 'q' + str(q_idx)
                    control_channel_labels[u_idx] = {'driven_q': drive_idx, 'freq': u_string}

        return cls(hamiltonian=hamiltonian,
                   qubit_freq_est=qubit_freq_est,
                   meas_freq_est=meas_freq_est,
                   u_channel_lo=u_channel_lo,
                   control_channel_labels=control_channel_labels,
                   qubit_list=qubit_list,
                   dt=dt)

    def control_channel_index(self, label):
        """Return the index of the control channel with identifying label.

        Args:
            label (Any): label that identifies a control channel

        Returns:
            int or None: index of the ControlChannel
        """
        if label not in self.control_channel_labels:
            warn('There is no listed ControlChannel matching the provided label.')
            return None
        else:
            return self.control_channel_labels.index(label)

    def calculate_channel_frequencies(self, qubit_lo_freq=None):
        """Calculate frequencies for each channel.

        Args:
            qubit_lo_freq (list or None): list of qubit linear
               oscillator drive frequencies. If None these will be calculated
               using self._qubit_freq_est.

        Returns:
            OrderedDict: a dictionary of channel frequencies.

        Raises:
            ValueError: If channel or u_channel_lo are invalid.
        """
        if not qubit_lo_freq:
            if not self._qubit_freq_est:
                raise ValueError("No qubit_lo_freq to use.")

            qubit_lo_freq = self._qubit_freq_est

        if self.u_channel_lo is None:
            raise ValueError("{} has no u_channel_lo.".format(self.__class__.__name__))

        # Setup freqs for the channels
        freqs = OrderedDict()
        for key in self.hamiltonian._channels:
            chidx = int(key[1:])
            if key[0] == 'D':
                freqs[key] = qubit_lo_freq[chidx]
            elif key[0] == 'U':
                freqs[key] = 0
                for u_lo_idx in self.u_channel_lo[chidx]:
                    if u_lo_idx['q'] < len(qubit_lo_freq):
                        qfreq = qubit_lo_freq[u_lo_idx['q']]
                        qscale = u_lo_idx['scale'][0]
                        freqs[key] += qfreq * qscale
            else:
                raise ValueError("Channel is not D or U")
        return freqs
