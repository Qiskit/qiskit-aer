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

"System Model class for system specification for the PulseSimulator"

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
        hamiltonian = HamiltonianModel.from_dict(config['hamiltonian'], qubit_list)
        u_channel_lo = config.get('u_channel_lo', None)
        dt = config.get('dt', None)

        return cls(hamiltonian=hamiltonian,
                   qubit_freq_est=qubit_freq_est,
                   meas_freq_est=meas_freq_est,
                   u_channel_lo=u_channel_lo,
                   qubit_list=qubit_list,
                   dt=dt)

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
