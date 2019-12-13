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
# pylint: disable=invalid-name

"""A module of routines for digesting a PULSE qobj
into something we can actually use.
"""

from warnings import warn
from collections import OrderedDict
import numpy as np
from .op_system import OPSystem
from .opparse import NoiseParser
from .operators import qubit_occ_oper_dressed
from ..solver.options import OPoptions
from ..hamiltonian_model import HamiltonianModel
# pylint: disable=no-name-in-module,import-error
from ..cy.utils import oplist_to_array
from . import op_qobj as op


def digest_pulse_obj(qobj_input, backend_options, noise_model):
    """Takes an input PULSE obj an disgests it into
    things we can actually make use of.

    Args:
        qobj_input (Qobj): Qobj of PULSE type.
        backend_options (dict): backend simulation options
        noise_model (dict): currently not supported

    Returns:
        OPSystem: The parsed qobj.

    Raises:
        Exception: Invalid options
        ValueError: Invalid channel selection.
    """
    # Output data object
    out = OPSystem()

    # take inputs and format into a single dictionary
    qobj = _format_qobj_dict(qobj_input, backend_options, noise_model)

    # post warnings for unsupported features
    _unsupported_warnings(qobj)

    # Get the config settings from the qobj
    config_dict = qobj['config']
    if 'backend_options' not in config_dict:
        raise ValueError('Pulse Qobj must have "backend_options".')
    config_dict_sim = config_dict['backend_options']

    # Parse config settings
    if 'memory_slots' not in config_dict:
        raise ValueError('Number of memory_slots must be specific in Qobj config')
    out.global_data['shots'] = int(config_dict.get('shots', 1024))
    out.global_data['meas_level'] = int(config_dict.get('meas_level', 1))
    out.global_data['meas_return'] = config_dict.get('meas_return', 'avg')
    out.global_data['seed'] = config_dict_sim.get('seed', None)
    out.global_data['memory_slots'] = config_dict.get('memory_slots', 0)
    out.global_data['memory'] = config_dict.get('memory', False)
    out.global_data['n_registers'] = config_dict.get('n_registers', 0)
    out.global_data['q_level_meas'] = int(config_dict_sim.get('q_level_meas', 1))

    # Attach the ODE options
    allowed_ode_options = ['atol', 'rtol', 'nsteps', 'max_step',
                           'num_cpus', 'norm_tol', 'norm_steps',
                           'rhs_reuse', 'rhs_filename']
    ode_options = config_dict_sim.get('ode_options', {})
    for key in ode_options:
        if key not in allowed_ode_options:
            raise Exception('Invalid ode_option: {}'.format(key))
    out.ode_options = OPoptions(**ode_options)

    # Parse the hamiltonian
    if 'hamiltonian' not in config_dict_sim:
        raise ValueError('Qobj must have hamiltonian in config to simulate.')
    hamiltonian = config_dict_sim['hamiltonian']
    # Get qubit number
    qubit_list = config_dict_sim.get('qubit_list', None)
    if qubit_list is None:
        qubit_list = list(range(config_dict_sim['n_qubits']))
    else:
        config_dict_sim['n_qubits'] = len(qubit_list)
    qubit_lo_freq = config_dict_sim.get('qubit_lo_freq',
                                        config_dict['qubit_lo_freq'])
    u_channel_lo = config_dict_sim.get('u_channel_lo')

    ham_model = HamiltonianModel(hamiltonian, qubit_list)
    # For now we dump this into OpSystem, though that should be refactored
    out.system = ham_model._system
    out.vars = ham_model._vars
    out.channels = ham_model._channels
    out.freqs = ham_model.calculate_frequencies(qubit_lo_freq=qubit_lo_freq,
                                                u_channel_lo=u_channel_lo)
    out.h_diag = ham_model._h_diag
    out.evals = ham_model._evals
    out.estates = ham_model._estates
    dim_qub = ham_model._dim_qub
    dim_osc = ham_model._dim_osc
    # convert estates into a Qutip qobj
    estates = [op.state(state) for state in ham_model._estates.T[:]]
    out.initial_state = estates[0]
    out.global_data['vars'] = list(out.vars.values())
    out.global_data['freqs'] = list(out.freqs.values())

    # Parse noise
    noise_dict = config_dict_sim.get('noise_model', {})
    if noise_dict:
        noise = NoiseParser(noise_dict=noise_dict,
                            dim_osc=dim_osc, dim_qub=dim_qub)
        noise.parse()

        out.noise = noise.compiled
        if any(out.noise):
            out.can_sample = False
            out.global_data['c_num'] = len(out.noise)
    else:
        out.noise = None

    # Build pulse arrays
    pulses, pulses_idx, pulse_dict = build_pulse_arrays(qobj)

    out.global_data['pulse_array'] = pulses
    out.global_data['pulse_indices'] = pulses_idx
    out.pulse_to_int = pulse_dict

    # Get dt
    if 'dt' not in config_dict_sim.keys():
        raise ValueError('Qobj must have a dt value to simulate.')
    out.dt = config_dict_sim['dt']

    # Set the ODE solver max step to be the half the
    # width of the smallest pulse
    min_width = np.iinfo(np.int32).max
    for key, val in out.pulse_to_int.items():
        if key != 'pv':
            stop = out.global_data['pulse_indices'][val + 1]
            start = out.global_data['pulse_indices'][val]
            min_width = min(min_width, stop - start)
    out.ode_options.max_step = min_width / 2 * out.dt

    # Convert experiments to data structures.
    # convert estates into a qobj

    out.global_data['measurement_ops'] = [None] * config_dict_sim['n_qubits']

    for exp in qobj['experiments']:
        exp_struct = experiment_to_structs(exp,
                                           out.channels,
                                           out.global_data['pulse_indices'],
                                           out.pulse_to_int,
                                           out.dt, qubit_list)

        # Add in measurement operators
        # Not sure if this will work for multiple measurements
        if any(exp_struct['acquire']):
            for acq in exp_struct['acquire']:
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

        out.experiments.append(exp_struct)
        if not exp_struct['can_sample']:
            out.can_sample = False
    return out


def _format_qobj_dict(qobj, backend_options, noise_model):
    """Add additional fields to qobj dictionary"""
    # Convert qobj to dict and add additional fields
    qobj_dict = qobj.to_dict()
    if 'backend_options' not in qobj_dict['config']:
        qobj_dict['config']['backend_options'] = {}

    # Temp backwards compatibility
    if 'sim_config' in qobj_dict['config']:
        for key, val in qobj_dict['config']['sim_config'].items():
            qobj_dict['config']['backend_options'][key] = val
        qobj_dict['config'].pop('sim_config')

    # Add additional backend options
    if backend_options is not None:
        for key, val in backend_options.items():
            qobj_dict['config']['backend_options'][key] = val
    # Add noise model
    if noise_model is not None:
        qobj_dict['config']['backend_options']['noise_model'] = noise_model
    return qobj_dict


def _unsupported_warnings(qobj_dict):
    """ Warns the user about untested/unsupported features.

    Parameters:
        qobj_dict (dict): Formatted qobj_dict from _format_qobj_dict
    Returns:
    Raises:
    """

    # Warnings that don't stop execution
    warning_str = '{} are an untested feature, and therefore may not behave as expected.'
    if 'osc' in qobj_dict['config']['backend_options']['hamiltonian'].keys():
        warn(warning_str.format('Oscillator-type systems'))
    if 'noise_model' in qobj_dict['config']['backend_options']:
        warn(warning_str.format('Noise models'))
    if _contains_pv_instruction(qobj_dict['experiments']):
        warn(warning_str.format('PersistentValue instructions'))


def _contains_pv_instruction(experiments):
    """ Return True if the list of experiments from the output of _format_qobj_dict contains
    a PersistentValue instruction

    Parameters:
        experiments (list): list of schedules
    Returns:
        True or False: whether or not the schedules contain a PersistentValue command
    Raises:
    """
    for exp in experiments:
        for inst in exp['instructions']:
            if inst['name'] == 'pv':
                return True
    return False


def build_pulse_arrays(qobj):
    """ Build pulses and pulse_idx arrays, and a pulse_dict
    used in simulations and mapping of experimental pulse
    sequencies to pulse_idx sequencies and timings.

    Parameters:
        qobj (Qobj): A pulse-qobj instance.

    Returns:
        tuple: Returns all pulses in one array,
        an array of start indices for pulses, and dict that
        maps pulses to the index at which the pulses start.
    """
    qobj_pulses = qobj['config']['pulse_library']
    pulse_dict = {}
    total_pulse_length = 0

    num_pulse = 0
    for pulse in qobj_pulses:
        pulse_dict[pulse['name']] = num_pulse
        total_pulse_length += len(pulse['samples'])
        num_pulse += 1

    idx = num_pulse + 1
    # now go through experiments looking for PV gates
    pv_pulses = []
    for exp in qobj['experiments']:
        for pulse in exp['instructions']:
            if pulse['name'] == 'pv':
                if pulse['val'] not in [pval[1] for pval in pv_pulses] and pulse['val'] != 0:
                    pv_pulses.append((pulse['val'], idx))
                    idx += 1
                    total_pulse_length += 1

    pulse_dict['pv'] = pv_pulses

    pulses = np.empty(total_pulse_length, dtype=complex)
    pulses_idx = np.zeros(idx + 1, dtype=np.uint32)

    stop = 0
    ind = 1
    for _, pulse in enumerate(qobj_pulses):
        stop = pulses_idx[ind - 1] + len(pulse['samples'])
        pulses_idx[ind] = stop
        oplist_to_array(pulse['samples'], pulses, pulses_idx[ind - 1])
        ind += 1

    for pv in pv_pulses:
        stop = pulses_idx[ind - 1] + 1
        pulses_idx[ind] = stop
        oplist_to_array([pv[0]], pulses, pulses_idx[ind - 1])
        ind += 1

    return pulses, pulses_idx, pulse_dict


def experiment_to_structs(experiment, ham_chans, pulse_inds,
                          pulse_to_int, dt, qubit_list=None):
    """Converts an experiment to a better formatted structure

    Args:
        experiment (dict): An experiment.
        ham_chans (dict): The channels in the Hamiltonian.
        pulse_inds (array): Array of pulse indices.
        pulse_to_int (array): Qobj pulses labeled by ints.
        dt (float): Pulse time resolution.
        qubit_list (list): List of qubits.

    Returns:
        dict: The output formatted structure.

    Raises:
        ValueError: Channel not in Hamiltonian.
        TypeError: Incorrect snapshot type.
    """
    # TO DO: Error check that operations are restricted to qubit list
    max_time = 0
    structs = {}
    structs['header'] = experiment['header']
    structs['channels'] = OrderedDict()
    for chan_name in ham_chans:
        structs['channels'][chan_name] = [[], []]
    structs['acquire'] = []
    structs['cond'] = []
    structs['snapshot'] = []
    structs['tlist'] = []
    structs['can_sample'] = True
    # This is a list that tells us whether
    # the last PV pulse on a channel needs to
    # be assigned a final time based on the next pulse on that channel
    pv_needs_tf = [0] * len(ham_chans)

    # The instructions are time-ordered so just loop through them.
    for inst in experiment['instructions']:
        # Do D and U channels
        if 'ch' in inst.keys() and inst['ch'][0] in ['d', 'u']:
            chan_name = inst['ch'].upper()
            if chan_name not in ham_chans.keys():
                raise ValueError('Channel {} is not in Hamiltonian model'.format(inst['ch']))

            # If last pulse on channel was a PV then need to set
            # its final time to be start time of current pulse
            if pv_needs_tf[ham_chans[chan_name]]:
                structs['channels'][chan_name][0][-3] = inst['t0'] * dt
                pv_needs_tf[ham_chans[chan_name]] = 0

            # Get condtional info
            if 'conditional' in inst.keys():
                cond = inst['conditional']
            else:
                cond = -1
            # PV's
            if inst['name'] == 'pv':
                # Get PV index
                for pv in pulse_to_int['pv']:
                    if pv[0] == inst['val']:
                        index = pv[1]
                        break
                structs['channels'][chan_name][0].extend([inst['t0'] * dt, None, index, cond])
                pv_needs_tf[ham_chans[chan_name]] = 1

            # Frame changes
            elif inst['name'] == 'fc':
                structs['channels'][chan_name][1].extend([inst['t0'] * dt, inst['phase'], cond])

            # A standard pulse
            else:
                start = inst['t0'] * dt
                pulse_int = pulse_to_int[inst['name']]
                pulse_width = (pulse_inds[pulse_int + 1] - pulse_inds[pulse_int]) * dt
                stop = start + pulse_width
                structs['channels'][chan_name][0].extend([start, stop, pulse_int, cond])

                max_time = max(max_time, stop)

        # Take care of acquires and snapshots (bfuncs added )
        else:
            # measurements
            if inst['name'] == 'acquire':

                # Better way??
                qlist2 = []
                mlist2 = []
                if qubit_list is None:
                    qlist2 = inst['qubits']
                    mlist2 = inst['memory_slot']
                else:
                    for qind, qb in enumerate(inst['qubits']):
                        if qb in qubit_list:
                            qlist2.append(qb)
                            mlist2.append(inst['memory_slot'][qind])

                acq_vals = [inst['t0'] * dt,
                            np.asarray(qlist2, dtype=np.uint32),
                            np.asarray(mlist2, dtype=np.uint32)
                            ]
                if 'register_slot' in inst.keys():
                    acq_vals.append(np.asarray(inst['register_slot'],
                                               dtype=np.uint32))
                else:
                    acq_vals.append(None)
                structs['acquire'].append(acq_vals)

                # update max_time
                max_time = max(max_time, (inst['t0'] + inst['duration']) * dt)

                # Add time to tlist
                if inst['t0'] * dt not in structs['tlist']:
                    structs['tlist'].append(inst['t0'] * dt)

            # conditionals
            elif inst['name'] == 'bfunc':
                bfun_vals = [inst['t0'] * dt, inst['mask'], inst['relation'],
                             inst['val'], inst['register']]
                if 'memory' in inst.keys():
                    bfun_vals.append(inst['memory'])
                else:
                    bfun_vals.append(None)

                structs['cond'].append(acq_vals)

                # update max_time
                max_time = max(max_time, inst['t0'] * dt)

                # Add time to tlist
                if inst['t0'] * dt not in structs['tlist']:
                    structs['tlist'].append(inst['t0'] * dt)

            # snapshots
            elif inst['name'] == 'snapshot':
                if inst['type'] != 'state':
                    raise TypeError("Snapshots must be of type 'state'")
                structs['snapshot'].append([inst['t0'] * dt, inst['label']])

                # Add time to tlist
                if inst['t0'] * dt not in structs['tlist']:
                    structs['tlist'].append(inst['t0'] * dt)

                # update max_time
                max_time = max(max_time, inst['t0'] * dt)

    # If any PVs still need time then they are at the end
    # and should just go til final time
    ham_keys = list(ham_chans.keys())
    for idx, pp in enumerate(pv_needs_tf):
        if pp:
            structs['channels'][ham_keys[idx]][0][-3] = max_time
            pv_needs_tf[idx] = 0

    # Convert lists to numpy arrays
    for key in structs['channels'].keys():
        structs['channels'][key][0] = np.asarray(structs['channels'][key][0],
                                                 dtype=float)
        structs['channels'][key][1] = np.asarray(structs['channels'][key][1],
                                                 dtype=float)

    structs['tlist'] = np.asarray([0] + structs['tlist'], dtype=float)

    if len(structs['acquire']) > 1 or structs['tlist'][-1] > structs['acquire'][-1][0]:
        structs['can_sample'] = False

    return structs
