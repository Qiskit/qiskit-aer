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
# pylint: disable=invalid-name, import-error

"""Interpretation and storage of PulseQobj information for pulse simulation
"""

from collections import OrderedDict
import numpy as np
from qiskit.providers.aer.aererror import AerError
# pylint: disable=no-name-in-module
from ..de_solvers.pulse_utils import oplist_to_array


class DigestedPulseQobj:
    """Container class for information extracted from PulseQobj
    """

    def __init__(self):

        # ####################################
        # Some "Simulation description"
        # ####################################

        # parameters related to memory/measurements
        self.shots = None
        self.meas_level = None
        self.meas_return = None
        self.memory_slots = None
        self.memory = None
        self.n_registers = None

        # ####################################
        # Signal portion
        # ####################################

        # these contain a particular undocumented data structure
        self.pulse_array = None
        self.pulse_indices = None
        self.pulse_to_int = None

        self.qubit_lo_freq = None

        # #############################################
        # Mix of both signal and simulation description
        # #############################################

        # These should be turned into an internal "simulation events"
        # structure

        # "experiments" contains a combination of signal information and
        # other experiment descriptions, which should be separated
        self.experiments = None


def digest_pulse_qobj(qobj, channels, dt, qubit_list, backend_options=None):
    """ Given a PulseQobj (and other parameters), returns a DigestedPulseQobj
    containing relevant extracted information

    Parameters:
        qobj (qobj): the PulseQobj
        channels (OrderedDict): channel dictionary
        dt (float): pulse sample width
        qubit_list (list): list of qubits to include
        backend_options (dict): dict with options that can override all other parameters

    Returns:
        DigestedPulseQobj: digested pulse qobj

    Raises:
        ValueError: for missing parameters
        AerError: for unsupported features or invalid qobj
        TypeError: for arguments of invalid type
    """

    if backend_options is None:
        backend_options = {}

    digested_qobj = DigestedPulseQobj()

    qobj_dict = qobj.to_dict()
    qobj_config = qobj_dict['config']

    # raises errors for unsupported features
    _unsupported_errors(qobj_dict)

    # override anything in qobj_config that is present in backend_options
    for key in backend_options.keys():
        qobj_config[key] = backend_options[key]

    if 'memory_slots' not in qobj_config:
        raise ValueError('Number of memory_slots must be specific in Qobj config')

    # set memory and measurement details
    digested_qobj.shots = int(qobj_config.get('shots', 1024))
    digested_qobj.meas_level = int(qobj_config.get('meas_level', 2))
    digested_qobj.meas_return = qobj_config.get('meas_return', 'avg')
    digested_qobj.memory_slots = qobj_config.get('memory_slots', 0)
    digested_qobj.memory = qobj_config.get('memory', False)
    digested_qobj.n_registers = qobj_config.get('n_registers', 0)

    # set qubit_lo_freq as given in qobj
    if 'qubit_lo_freq' in qobj_config and qobj_config['qubit_lo_freq'] != [np.inf]:
        # qobj frequencies are divided by 1e9, so multiply back
        digested_qobj.qubit_lo_freq = [freq * 1e9 for freq in qobj_config['qubit_lo_freq']]

    # build pulse arrays from qobj
    pulses, pulses_idx, pulse_dict = build_pulse_arrays(qobj_dict['experiments'],
                                                        qobj_config['pulse_library'])

    digested_qobj.pulse_array = pulses
    digested_qobj.pulse_indices = pulses_idx
    digested_qobj.pulse_to_int = pulse_dict

    experiments = []

    for exp in qobj_dict['experiments']:
        exp_struct = experiment_to_structs(exp,
                                           channels,
                                           pulses_idx,
                                           pulse_dict,
                                           dt,
                                           qubit_list)
        experiments.append(exp_struct)

    digested_qobj.experiments = experiments

    return digested_qobj


def _unsupported_errors(qobj_dict):
    """ Raises errors for untested/unsupported features.

    Parameters:
        qobj_dict (dict): qobj in dictionary form
    Returns:
    Raises:
        AerError: for unsupported features
    """

    # Warnings that don't stop execution
    warning_str = '{} are an untested feature, and therefore may not behave as expected.'
    if _contains_pv_instruction(qobj_dict['experiments']):
        raise AerError(warning_str.format('PersistentValue instructions'))

    required_str = '{} are required for simulation, and none were specified.'
    if not _contains_acquire_instruction(qobj_dict['experiments']):
        raise AerError(required_str.format('Acquire instructions'))


def _contains_acquire_instruction(experiments):
    """ Return True if the list of experiments contains an Acquire instruction
    Parameters:
        experiments (list): list of schedules
    Returns:
        True or False: whether or not the schedules contain an Acquire command
    Raises:
    """

    for exp in experiments:
        for inst in exp['instructions']:
            if inst['name'] == 'acquire':
                return True
    return False


def _contains_pv_instruction(experiments):
    """ Return True if the list of experiments contains a PersistentValue instruction

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


def build_pulse_arrays(experiments, pulse_library):
    """ Build pulses and pulse_idx arrays, and a pulse_dict
    used in simulations and mapping of experimental pulse
    sequencies to pulse_idx sequencies and timings.

    Parameters:
        experiments (list): list of experiments
        pulse_library (list): list of pulses

    Returns:
        tuple: Returns all pulses in one array,
        an array of start indices for pulses, and dict that
        maps pulses to the index at which the pulses start.
    """
    pulse_dict = {}
    total_pulse_length = 0

    num_pulse = 0
    for pulse in pulse_library:
        pulse_dict[pulse['name']] = num_pulse
        total_pulse_length += len(pulse['samples'])
        num_pulse += 1

    idx = num_pulse + 1
    # now go through experiments looking for PV gates
    pv_pulses = []
    for exp in experiments:
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
    for _, pulse in enumerate(pulse_library):
        stop = pulses_idx[ind - 1] + len(pulse['samples'])
        pulses_idx[ind] = stop
        oplist_to_array(format_pulse_samples(pulse['samples']), pulses, pulses_idx[ind - 1])
        ind += 1

    for pv in pv_pulses:
        stop = pulses_idx[ind - 1] + 1
        pulses_idx[ind] = stop
        oplist_to_array(format_pulse_samples([pv[0]]), pulses, pulses_idx[ind - 1])
        ind += 1

    return pulses, pulses_idx, pulse_dict


def format_pulse_samples(pulse_samples):
    """Converts input into a list of complex numbers, where each complex numbers is
    given as a list of length 2. If it is already of this format, it simply returns it.

    This function assumes the input is either an ndarray, a list of numpy complex number types,
    or a list already in the desired format.

    Args:
        pulse_samples (list): An ndarray of complex numbers or a list

    Returns:
        list: list of the required format
    """

    new_samples = list(pulse_samples)

    if not np.iscomplexobj(new_samples[0]):
        return new_samples

    return [[samp.real, samp.imag] for samp in new_samples]


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

    if structs['tlist'][-1] > structs['acquire'][-1][0]:
        structs['can_sample'] = False

    return structs
