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
# pylint: disable=eval-used, exec-used, invalid-name

"System Model class for system specification for the PulseSimulator"

import numpy as np
from copy import deepcopy
from warnings import warn
from collections import OrderedDict
from qiskit.providers import BaseBackend
from qiskit.providers.aer.aererror import AerError
from .hamiltonian_model import HamiltonianModel
from .string_model_parser.string_model_parser import NoiseParser
from ..controllers.pulse_utils import get_ode_rhs_functor


class PulseSystemModel():
    r"""Physical model object for pulse simulator.

    This class contains model information required by the
    :class:`~qiskit.providers.aer.PulseSimulator`. It contains:

        * ``"hamiltonian"``: a :class:`HamiltonianModel` object representing the
          Hamiltonian of the system.
        * ``"qubit_freq_est"`` and ``"meas_freq_est"``: optional default values for
          qubit and measurement frequencies.
        * ``"u_channel_lo"``: A description of :class:`ControlChannel` local oscillator
          frequencies in terms of qubit local oscillator frequencies.
        * ``"control_channel_labels"``: Optional list of identifying information for
          each :class:`ControlChannel` that the model supports.
        * ``"subsystem_list"``: List of subsystems in the model.
        * ``"dt"``: Sample width size for OpenPulse instructions.

    A model can be instantiated from the helper function :func:`duffing_system_model`,
    or using the :meth:`PulseSystemModel.from_backend` constructor.

    **Example**

    Constructing from a backend:

    .. code-block: python

        provider = IBMQ.load_account()
        armonk_backend = provider.get_backend('ibmq_armonk')

        system_model = PulseSystemModel.from_backend(armonk_backend)
    """
    def __init__(self,
                 hamiltonian=None,
                 qubit_freq_est=None,
                 meas_freq_est=None,
                 u_channel_lo=None,
                 control_channel_labels=None,
                 subsystem_list=None,
                 dt=None):
        """Initialize a PulseSystemModel.

        Args:
            hamiltonian (HamiltonianModel): The Hamiltonian of the system.
            qubit_freq_est (list): list of qubit lo frequencies defaults to be used in simulation
                                   if none are specified in the PulseQobj.
            meas_freq_est (list): list of qubit meas frequencies defaults to be used in simulation
                                  if none are specified in the PulseQobj.
            u_channel_lo (list): list of ControlChannel frequency specifications.
            control_channel_labels (list): list of labels for control channels, which can be of
                                           any type.
            subsystem_list (list): list of valid qubit indicies for the model.
            dt (float): pixel size for pulse Instructions.
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
        self.subsystem_list = subsystem_list
        self.dt = dt

        self._noise = None
        self._rhs_dict = None

    @classmethod
    def from_backend(cls, backend, subsystem_list=None):
        """Returns a PulseSystemModel constructed from an OpenPulse enabled backend object.

        Args:
            backend (Backend): backend object to draw information from.
            subsystem_list (list): a list of ints for which qubits to include in the model.

        Returns:
            PulseSystemModel: the PulseSystemModel constructed from the backend.

        Raises:
            AerError: If channel or u_channel_lo are invalid.
        """

        if not isinstance(backend, BaseBackend):
            raise AerError("{} is not a Qiskit backend".format(backend))

        # get relevant information from backend
        defaults = backend.defaults()
        config = backend.configuration()

        if not config.open_pulse:
            raise AerError('{} is not an open pulse backend'.format(backend))

        # draw defaults
        qubit_freq_est = getattr(defaults, 'qubit_freq_est', None)
        meas_freq_est = getattr(defaults, 'meas_freq_est', None)

        # draw from configuration
        # if no subsystem_list, use all for device
        subsystem_list = subsystem_list or list(range(config.n_qubits))
        ham_string = config.hamiltonian
        hamiltonian = HamiltonianModel.from_dict(ham_string, subsystem_list)
        u_channel_lo = getattr(config, 'u_channel_lo', None)
        dt = getattr(config, 'dt', None)

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
                        scale = getattr(u_term_dict, 'scale', [1.0, 0])
                        q_idx = getattr(u_term_dict, 'q')
                        if len(u_string) > 0:
                            u_string += ' + '
                        if isinstance(scale, complex):
                            u_string += str(scale) + 'q' + str(q_idx)
                        else:
                            u_string += str(scale[0] + scale[1] * 1j) + 'q' + str(q_idx)
                    control_channel_labels[u_idx] = {'driven_q': drive_idx, 'freq': u_string}

        return cls(hamiltonian=hamiltonian,
                   qubit_freq_est=qubit_freq_est,
                   meas_freq_est=meas_freq_est,
                   u_channel_lo=u_channel_lo,
                   control_channel_labels=control_channel_labels,
                   subsystem_list=subsystem_list,
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
        """Calculate frequencies for each channel given qubit_lo_freq.

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
                    if u_lo_idx.q < len(qubit_lo_freq):
                        qfreq = qubit_lo_freq[u_lo_idx.q]
                        qscale = u_lo_idx.scale.real
                        freqs[key] += qfreq * qscale
            else:
                raise ValueError("Channel is not D or U")
        return freqs

    def add_noise_from_dict(self, noise_dict):
        if noise_dict:
            dim_qub = self.hamiltonian._subsystem_dims

            noise = NoiseParser(noise_dict=noise_dict, dim_osc={}, dim_qub=dim_qub)
            noise.parse()

            self._noise = noise.compiled


    def _config_internal_data(self):
        """Preps internal data into format required by RHS function.
        """
        ham_model = self.hamiltonian

        vars = list(ham_model._variables.values())
        # Need this info for evaluating the hamiltonian vars in the c++ rhs
        vars_names = list(ham_model._variables.keys())

        num_h_terms = len(ham_model._system)
        H = [hpart[0] for hpart in ham_model._system]

        # take care of collapse operators, if any
        self._c_num = 0
        if self._noise is not None:
            self._c_num = len(self._noise)
            num_h_terms += 1

        self._c_ops_data = []
        self._c_ops_ind = []
        self._c_ops_ptr = []
        self._n_ops_data = []
        self._n_ops_ind = []
        self._n_ops_ptr = []

        # if there are any collapse operators
        H_noise = 0
        for kk in range(self._c_num):
            c_op = self._noise[kk]
            n_op = c_op.dag() * c_op
            # collapse ops
            self._c_ops_data.append(c_op.data.data)
            self._c_ops_ind.append(c_op.data.indices)
            self._c_ops_ptr.append(c_op.data.indptr)
            # norm ops
            self._n_ops_data.append(n_op.data.data)
            self._n_ops_ind.append(n_op.data.indices)
            self._n_ops_ptr.append(n_op.data.indptr)
            # Norm ops added to time-independent part of
            # Hamiltonian to decrease norm
            H_noise -= 0.5j * n_op

        if H_noise:
            H = H + [H_noise]

        # construct data sets
        h_ops_data = [-1.0j * hpart.data.data for hpart in H]
        h_ops_ind = [hpart.data.indices for hpart in H]
        h_ops_ptr = [hpart.data.indptr for hpart in H]

        self._rhs_dict = {'freqs': list(self._freqs.values()),
                          'pulse_array': self._pulse_array,
                          'pulse_indices': self._pulse_indices,
                          'vars': vars,
                          'vars_names': vars_names,
                          'num_h_terms': num_h_terms,
                          'h_ops_data': h_ops_data,
                          'h_ops_ind': h_ops_ind,
                          'h_ops_ptr': h_ops_ptr,
                          'h_diag_elems': ham_model._h_diag}

    def init_rhs(self, exp):
        """Set up and return rhs function corresponding to this model for a given
        experiment exp
        """

        # if _rhs_dict has not been set up, config the internal data
        if self._rhs_dict is None:
            self._config_internal_data()

        channels = dict(self.hamiltonian._channels)

        # Init register
        register = np.ones(self._n_registers, dtype=np.uint8)

        ode_rhs_obj = get_ode_rhs_functor(self._rhs_dict, exp, self.hamiltonian._system, channels, register)

        def rhs(t, y):
            return ode_rhs_obj(t, y)

        return rhs

    def copy(self):
        return deepcopy(self)
