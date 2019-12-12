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
                 qubit_lo_freq=None,
                 u_channel_lo=None,
                 dt=None):

        self.hamiltonian = hamiltonian
        self.qubit_lo_freq = qubit_lo_freq
        self.u_channel_lo = u_channel_lo
        self.dt = dt

    @classmethod
    def from_backend(cls, backend, qubit_list=None):
        # get relevant information from backend
        defaults_dict = backend.defaults().to_dict()
        config_dict = backend.configuration().to_dict()

        # draw from defaults
        qubit_lo_freq = defaults.get('qubit_freq_est', None)

        #draw from configuration
        hamiltonian = HamiltonianModel(config_dict['hamiltonian'], qubit_list)
        u_channel_lo = config_dict.get('u_channel_lo', None)
        dt = config_dict.get('dt', None)

        return cls(hamiltonian, qubit_lo_freq, u_channel_lo, dt)

    def set_qubit_lo_freq(cls, qubit_lo_freq):

        if qubit_lo_freq == 'from_hamiltonian':
