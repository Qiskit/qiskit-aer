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
# pylint: disable=invalid-name, missing-return-type-doc

""" This file should ultimately disappear into different pieces.


Need separate:
    - interpretation of qobj/pulse schedules (this is the only thing that
      should be called "pulse digest"). This needs to be further separated into:
        - construction of signals
        - construction of "simulation description"
    - Simulation preparation
        - e.g. frame transformations, computing eigenvalues, setting solver steps ...
        - These are all specific to, and should be contained in, a solver

- actual qobj digest has now been separated, though unfortunately it depends on
  3 parameters from the model, which needs to be changed/figured out
"""

from warnings import warn
import numpy as np
from qiskit.providers.aer.aererror import AerError
from ..qobj.op_system import OPSystem
from .string_model_parser import NoiseParser
from ..qobj.operators import qubit_occ_oper_dressed
from ..solver.options import OPoptions
from ..qobj import op_qobj as op
from .pulse_qobj_digest0 import digest_pulse_qobj


def full_digest(qobj, system_model, backend_options=None):
    """Convert specification of a simulation in the pulse language into the format accepted
    by the simulator.

    Args:
        qobj (PulseQobj): experiment specification
        system_model (PulseSystemModel): object representing system model
        backend_options (dict): dictionary of simulation options
    Returns:
        out (OPSystem): object understandable by the pulse simulator
    Raises:
        ValueError: When necessary parameters are missing
        Exception: For invalid ode options
    """

    out = OPSystem()

    if backend_options is None:
        backend_options = {}

    """ Note: the overriding behaviour of backend_options is currently
    broken """
    # override anything in qobj_config that is present in backend_options
    #for key in backend_options.keys():
    #    qobj_config[key] = backend_options[key]

    noise_model = backend_options.get('noise_model', None)

    # post warnings for unsupported features
    _unsupported_warnings(noise_model)

    # ###############################
    # ### Extract model parameters
    # ###############################

    # Get qubit list and number
    qubit_list = system_model.subsystem_list
    if qubit_list is None:
        raise ValueError('Model must have a qubit list to simulate.')
    n_qubits = len(qubit_list)

    # get Hamiltonian
    if system_model.hamiltonian is None:
        raise ValueError('Model must have a Hamiltonian to simulate.')
    ham_model = system_model.hamiltonian

    # For now we dump this into OpSystem, though that should be refactored
    out.system = ham_model._system
    out.vars = ham_model._variables
    out.channels = ham_model._channels
    out.h_diag = ham_model._h_diag
    out.evals = ham_model._evals
    out.estates = ham_model._estates
    dim_qub = ham_model._subsystem_dims
    dim_osc = {}
    # convert estates into a Qutip qobj
    estates = [op.state(state) for state in ham_model._estates.T[:]]
    out.initial_state = estates[0]
    out.global_data['vars'] = list(out.vars.values())
    # Need this info for evaluating the hamiltonian vars in the c++ solver
    out.global_data['vars_names'] = list(out.vars.keys())

    # Get dt
    if system_model.dt is None:
        raise ValueError('Qobj must have a dt value to simulate.')
    out.dt = system_model.dt

    # Parse noise
    noise_dict = noise_model or {}
    if noise_dict:
        noise = NoiseParser(noise_dict=noise_dict, dim_osc=dim_osc, dim_qub=dim_qub)
        noise.parse()

        out.noise = noise.compiled
        if any(out.noise):
            out.can_sample = False
            out.global_data['c_num'] = len(out.noise)
    else:
        out.noise = None

    # ###############################
    # ### Parse qobj_config settings
    # ###############################

    # This should just depend on the qobj, or at most, also on dt
    digested_qobj = digest_pulse_qobj(qobj, out.channels, out.dt, qubit_list)

    out.global_data['shots'] = digested_qobj.shots
    out.global_data['meas_level'] = digested_qobj.meas_level
    out.global_data['meas_return'] = digested_qobj.meas_return
    out.global_data['memory_slots'] = digested_qobj.memory_slots
    out.global_data['memory'] = digested_qobj.memory
    out.global_data['n_registers'] = digested_qobj.n_registers

    out.global_data['pulse_array'] = digested_qobj.pulse_array
    out.global_data['pulse_indices'] = digested_qobj.pulse_indices
    out.pulse_to_int = digested_qobj.pulse_to_int

    out.experiments = digested_qobj.experiments

    # ###############################
    # ### Handle qubit_lo_freq
    # ###############################

    # First, get it from the qobj (if it wasn't specified in qobj,
    # this will be None)
    qubit_lo_freq = digested_qobj.qubit_lo_freq

    # if it wasn't specified in the PulseQobj, draw from system_model
    if qubit_lo_freq is None:
        qubit_lo_freq = system_model._qubit_freq_est

    # if still None draw from the Hamiltonian
    if qubit_lo_freq is None:
        qubit_lo_freq = system_model.hamiltonian.get_qubit_lo_from_drift()
        warn('Warning: qubit_lo_freq was not specified in PulseQobj or in PulseSystemModel, ' +
             'so it is beign automatically determined from the drift Hamiltonian.')

    out.freqs = system_model.calculate_channel_frequencies(qubit_lo_freq=qubit_lo_freq)
    out.global_data['freqs'] = list(out.freqs.values())


    # ###############################
    # ### Parse backend_options
    # ###############################
    if 'seed' in backend_options:
        out.global_data['seed'] = int(backend_options.get('seed'))
    else:
        out.global_data['seed'] = None
    out.global_data['q_level_meas'] = int(backend_options.get('q_level_meas', 1))

    # solver options
    allowed_ode_options = ['atol', 'rtol', 'nsteps', 'max_step',
                           'num_cpus', 'norm_tol', 'norm_steps',
                           'rhs_reuse', 'rhs_filename']
    ode_options = backend_options.get('ode_options', {})
    for key in ode_options:
        if key not in allowed_ode_options:
            raise Exception('Invalid ode_option: {}'.format(key))
    out.ode_options = OPoptions(**ode_options)

    # Set the ODE solver max step to be the half the
    # width of the smallest pulse
    min_width = np.iinfo(np.int32).max
    for key, val in out.pulse_to_int.items():
        if key != 'pv':
            stop = out.global_data['pulse_indices'][val + 1]
            start = out.global_data['pulse_indices'][val]
            min_width = min(min_width, stop - start)
    out.ode_options.max_step = min_width / 2 * out.dt

    # ###############################
    # ### Further interpretation of experiments
    # ###############################
    out.global_data['measurement_ops'] = [None] * n_qubits


    for exp in out.experiments:

        # Add in measurement operators
        # Not sure if this will work for multiple measurements
        # Note: the extraction of multiple measurements works, but the simulator itself
        # implicitly assumes there is only one measurement at the end
        if any(exp['acquire']):
            for acq in exp['acquire']:
                for jj in acq[1]:
                    if jj > qubit_list[-1]:
                        continue
                    if not out.global_data['measurement_ops'][jj]:
                        out.global_data['measurement_ops'][jj] = \
                            qubit_occ_oper_dressed(jj,
                                                   estates,
                                                   h_osc=dim_osc,
                                                   h_qub=dim_qub,
                                                   level=out.global_data['q_level_meas']
                                                   )

        if not exp['can_sample']:
            out.can_sample = False

    # This is a temporary flag while stabilizing cpp func ODE solver
    out.use_cpp_ode_func = True
    return out


def _unsupported_warnings(noise_model):
    """ Warns the user about untested/unsupported features.

    Parameters:
        noise_model (dict): backend_options for simulation
    Returns:
    Raises:
        AerError: for unsupported features
    """

    # Warnings that don't stop execution
    warning_str = '{} are an untested feature, and therefore may not behave as expected.'
    if noise_model is not None:
        warn(warning_str.format('Noise models'))
