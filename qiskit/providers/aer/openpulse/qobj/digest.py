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
# pylint: disable=eval-used, exec-used, invalid-name, consider-using-enumerate

"""A module of routines for digesting a PULSE qobj
into something we can actually use.
"""

from collections import OrderedDict
import numpy as np
import numpy.linalg as la
from .op_system import OPSystem
from .opparse import HamiltonianParser, NoiseParser
from .operators import qubit_occ_oper_dressed
from ..solver.options import OPoptions
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

    # Get the config settings from the qobj
    config_dict = qobj['config']
    if 'backend_options' not in config_dict:
        raise ValueError('Pulse Qobj must have "sim_config".')
    config_dict_sim = config_dict['backend_options']
    noise_dict = config_dict_sim.get('noise_model', {})
    if 'hamiltonian' not in config_dict_sim:
        raise ValueError('Qobj must have hamiltonian in config to simulate.')
    hamiltonian = config_dict_sim['hamiltonian']

    # Warnings for untested features
    warning_str = '{} are an untested feature, and therefore may not behave as expected.'
    if 'osc' in hamiltonian.keys():
        raise Warning(warning_str.format('Oscillator-type systems'))
    if noise_model is not None:
        raise Warning(warning_str.format('Noise models'))
    # check for persistent value pulses; if there is one, raise a Warning
    contains_pv_inst = False
    for experiment in qobj['experiments']:
        for inst in exp['instructions']:
            if inst['name'] == 'pv':
                contains_pv_inst = True
                break
        if contains_pv_inst:
             break
    if contains_pv_inst:
        raise Warning(warning_str.format('PersistentValue instructions'))

    # Get qubit number
    qubit_list = config_dict_sim.get('qubit_list', None)
    if qubit_list is None:
        qubit_list = list(range(config_dict_sim['n_qubits']))
    else:
        config_dict_sim['n_qubits'] = len(qubit_list)

    config_keys_sim = config_dict_sim.keys()
    config_keys = config_dict.keys()

    # Look for config keys
    out.global_data['shots'] = 1024
    if 'shots' in config_keys:
        out.global_data['shots'] = int(config_dict['shots'])

    out.global_data['meas_level'] = 1
    if 'meas_level' in config_keys:
        out.global_data['meas_level'] = int(config_dict['meas_level'])

    out.global_data['meas_return'] = 'avg'
    if 'meas_return' in config_keys:
        out.global_data['meas_return'] = config_dict['meas_return']

    out.global_data['seed'] = None
    if 'seed' in config_keys:
        out.global_data['seed'] = int(config_dict['seed'])

    if 'memory_slots' in config_keys:
        out.global_data['memory_slots'] = config_dict['memory_slots']
    else:
        err_str = 'Number of memory_slots must be specific in Qobj config'
        raise ValueError(err_str)

    if 'memory' in config_keys:
        out.global_data['memory'] = config_dict['memory']
    else:
        out.global_data['memory'] = False

    out.global_data['n_registers'] = 0
    if 'n_registers' in config_keys:
        out.global_data['n_registers'] = config_dict['n_registers']

    # which level to measure
    out.global_data['q_level_meas'] = 1
    if 'q_level_meas' in config_keys_sim:
        out.global_data['q_level_meas'] = int(config_dict_sim['q_level_meas'])

    # Attach the ODE options
    allowed_ode_options = ['atol', 'rtol', 'nsteps', 'max_step',
                           'num_cpus', 'norm_tol', 'norm_steps',
                           'rhs_reuse', 'rhs_filename']
    user_set_ode_options = {}
    if 'ode_options' in config_keys_sim:
        for key, val in config_dict_sim['ode_options'].items():
            if key not in allowed_ode_options:
                raise Exception('Invalid ode_option: {}'.format(key))
            user_set_ode_options[key] = val
    out.ode_options = OPoptions(**user_set_ode_options)

    # Step #1: Parse hamiltonian representation
    out.vars = OrderedDict(hamiltonian['vars'])
    out.global_data['vars'] = list(out.vars.values())

    # Get qubit subspace dimensions
    if 'qub' in hamiltonian.keys():
        dim_qub = hamiltonian['qub']
        _dim_qub = {}
        # Convert str keys to int keys
        for key, val in hamiltonian['qub'].items():
            _dim_qub[int(key)] = val
        dim_qub = _dim_qub
    else:
        dim_qub = {}.fromkeys(range(config_dict_sim['n_qubits']), 2)

    # Get oscillator subspace dimensions
    if 'osc' in hamiltonian.keys():
        dim_osc = hamiltonian['osc']
        _dim_osc = {}
        # Convert str keys to int keys
        for key, val in dim_osc.items():
            _dim_osc[int(key)] = val
        dim_osc = _dim_osc
    else:
        dim_osc = {}

    # Parse the Hamiltonian
    system = HamiltonianParser(h_str=hamiltonian['h_str'],
                               dim_osc=dim_osc,
                               dim_qub=dim_qub)
    system.parse(qubit_list)
    out.system = system.compiled

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

    # Step #2: Get Hamiltonian channels
    out.channels = get_hamiltonian_channels(out.system)

    h_diag, evals, estates = get_diag_hamiltonian(out.system,
                                                  out.vars, out.channels)

    # convert estates into a qobj
    estates_qobj = []
    for kk in range(len(estates[:, ])):
        estates_qobj.append(op.state(estates[:, kk]))

    out.h_diag = np.ascontiguousarray(h_diag.real)
    out.evals = evals
    out.estates = estates

    # Set initial state
    out.initial_state = 0 * op.basis(len(evals), 1)
    for idx, estate_coef in enumerate(estates[:, 0]):
        out.initial_state += estate_coef * op.basis(len(evals), idx)
    # init_fock_state(dim_osc, dim_qub)

    # Setup freqs for the channels
    out.freqs = OrderedDict()

    # determine whether to compute qubit_lo_freq from hamiltonian
    qubit_lo_from_ham = (('qubit_lo_freq' in config_dict_sim) and
                         (config_dict_sim['qubit_lo_freq'] == 'from_hamiltonian') and
                         (len(dim_osc) == 0)) or not config_dict['qubit_lo_freq']

    # set frequencies based on qubit_lo_from_ham value
    q_lo_freq = None
    if qubit_lo_from_ham:
        q_lo_freq = np.zeros(len(dim_qub))
        min_eval = np.min(evals)
        for q_idx in range(len(dim_qub)):
            single_excite = _first_excited_state(q_idx, dim_qub)
            dressed_eval = _eval_for_max_espace_overlap(single_excite, evals, estates)
            q_lo_freq[q_idx] = (dressed_eval - min_eval) / (2 * np.pi)
    else:
        q_lo_freq = config_dict['qubit_lo_freq']

    # set freqs
    for key in out.channels.keys():
        chidx = int(key[1:])
        if key[0] == 'D':
            out.freqs[key] = q_lo_freq[chidx]
        elif key[0] == 'U':
            out.freqs[key] = 0
            for u_lo_idx in config_dict_sim['u_channel_lo'][chidx]:
                if u_lo_idx['q'] < len(q_lo_freq):
                    qfreq = q_lo_freq[u_lo_idx['q']]
                    qscale = u_lo_idx['scale'][0]
                    out.freqs[key] += qfreq * qscale
        else:
            raise ValueError("Channel is not D or U")

    out.global_data['freqs'] = list(out.freqs.values())

    # Step #3: Build pulse arrays
    pulses, pulses_idx, pulse_dict = build_pulse_arrays(qobj)

    out.global_data['pulse_array'] = pulses
    out.global_data['pulse_indices'] = pulses_idx
    out.pulse_to_int = pulse_dict

    # Step #4: Get dt
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

    # Step #6: Convert experiments to data structures.

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
                                                   estates_qobj,
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


def get_diag_hamiltonian(parsed_ham, ham_vars, channels):
    """ Get the diagonal elements of the hamiltonian and get the
    dressed frequencies and eigenstates

    Parameters:
        parsed_ham (list): A list holding ops and strings from the Hamiltonian
        of a specific quantum system.

        ham_vars (dict): dictionary of variables

        channels (dict): drive channels (set to 0)

    Returns:
        h_diag: diagonal elements of the hamiltonian
        h_evals: eigenvalues of the hamiltonian with no time-dep terms
        h_estates: eigenstates of the hamiltonian with no time-dep terms

    Raises:
        Exception: Missing index on channel.
    """
    # Get the diagonal elements of the hamiltonian with all the
    # drive terms set to zero
    for chan in channels:
        exec('%s=0' % chan)

    # might be a better solution to replace the 'var' in the hamiltonian
    # string with 'op_system.vars[var]'
    for var in ham_vars:
        exec('%s=%f' % (var, ham_vars[var]))

    H_full = np.zeros(np.shape(parsed_ham[0][0].full()), dtype=complex)

    for hpart in parsed_ham:
        H_full += hpart[0].full() * eval(hpart[1])

    h_diag = np.diag(H_full)

    evals, estates = la.eigh(H_full)

    eval_mapping = []
    for ii in range(len(evals)):
        eval_mapping.append(np.argmax(np.abs(estates[:, ii])))

    evals2 = evals.copy()
    estates2 = estates.copy()

    for ii, val in enumerate(eval_mapping):
        evals2[val] = evals[ii]
        estates2[:, val] = estates[:, ii]

    return h_diag, evals2, estates2


def get_hamiltonian_channels(parsed_ham):
    """ Get all the qubit channels D_i and U_i in the string
    representation of a system Hamiltonian.

    Parameters:
        parsed_ham (list): A list holding ops and strings from the Hamiltonian
        of a specific quantum system.

    Returns:
        list: A list of all channels in Hamiltonian string.

    Raises:
        Exception: Missing index on channel.
    """
    out_channels = []
    for _, ham_str in parsed_ham:
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
            if temp_chan not in out_channels:
                out_channels.append(temp_chan)
    out_channels.sort(key=lambda x: (int(x[1:]), x[0]))

    out_dict = OrderedDict()
    for idx, val in enumerate(out_channels):
        out_dict[val] = idx

    return out_dict


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
                structs['channels'][chan_name][0][-3] = inst['t0']
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
                structs['channels'][chan_name][0].extend([inst['t0'], None, index, cond])
                pv_needs_tf[ham_chans[chan_name]] = 1

            # Frame changes
            elif inst['name'] == 'fc':
                structs['channels'][chan_name][1].extend([inst['t0'], inst['phase'], cond])

            # A standard pulse
            else:
                start = inst['t0']
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

                acq_vals = [inst['t0'],
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
                max_time = max(max_time, inst['t0'] + dt * inst['duration'])

                # Add time to tlist
                if inst['t0'] not in structs['tlist']:
                    structs['tlist'].append(inst['t0'])

            # conditionals
            elif inst['name'] == 'bfunc':
                bfun_vals = [inst['t0'], inst['mask'], inst['relation'],
                             inst['val'], inst['register']]
                if 'memory' in inst.keys():
                    bfun_vals.append(inst['memory'])
                else:
                    bfun_vals.append(None)

                structs['cond'].append(acq_vals)

                # update max_time
                max_time = max(max_time, inst['t0'])

                # Add time to tlist
                if inst['t0'] not in structs['tlist']:
                    structs['tlist'].append(inst['t0'])

            # snapshots
            elif inst['name'] == 'snapshot':
                if inst['type'] != 'state':
                    raise TypeError("Snapshots must be of type 'state'")
                structs['snapshot'].append([inst['t0'], inst['label']])

                # Add time to tlist
                if inst['t0'] not in structs['tlist']:
                    structs['tlist'].append(inst['t0'])

                # update max_time
                max_time = max(max_time, inst['t0'])

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
    for idx in range(len(dim_qub)):
        new_vec = np.zeros(dim_qub[idx])
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
