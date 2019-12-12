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
