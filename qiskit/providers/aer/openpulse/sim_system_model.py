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

from .hamiltonian_model import HamiltonianModel

class SimSystemModel():

    def __init__(self,
                 hamiltonian=None,
                 qubit_freq_est=None,
                 meas_freq_est=None,
                 u_channel_lo=None,
                 qubit_list=None,
                 dt=None):

        self.hamiltonian = hamiltonian
        self.qubit_freq_est = qubit_freq_est
        self.meas_freq_est = meas_freq_est
        self.u_channel_lo = u_channel_lo
        self.qubit_list = qubit_list
        self.dt = dt

    @classmethod
    def from_backend(cls, backend, qubit_list=None):
        # get relevant information from backend
        defaults = backend.defaults().to_dict()
        config = backend.configuration().to_dict()

        # draw from defaults
        qubit_freq_est = defaults.get('qubit_freq_est', None)
        meas_freq_est = defaults.get('meas_freq_est', None)

        #draw from configuration
        qubit_list = qubit_list or range(len(config['n_qubits']))
        hamiltonian = HamiltonianModel(config['hamiltonian'], qubit_list)
        u_channel_lo = config.get('u_channel_lo', None)
        dt = config.get('dt', None)

        return cls(hamiltonian=hamiltonian,
                   qubit_freq_est=qubit_freq_est,
                   meas_freq_est=meas_freq_est,
                   u_channel_lo=u_channel_lo,
                   qubit_list=qubit_list,
                   dt=dt)

   def compute_channel_frequencies(self, qubit_lo_freq=None):
       """Calulate frequencies for each channel.

       Args:
           qubit_lo_freq (list or None): list of qubit linear
               oscillator drive frequencies. If None these will be calculated
               using self.qubit_freq_est.

       Returns:
           OrderedDict: a dictionary of channel frequencies.

       Raises:
           ValueError: If channel or u_channel_lo are invalid.
       """
       # TODO: Update docstring with description of what qubit_lo_freq and
       # u_channel_lo are

       # Setup freqs for the channels
       freqs = OrderedDict()

       # TODO: set u_channel_lo from hamiltonian
       if not self.u_channel_lo:
           raise ValueError("SimSystemModel has no u_channel_lo.")

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
