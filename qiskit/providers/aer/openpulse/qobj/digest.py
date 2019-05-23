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

import numpy as np
from collections import OrderedDict
from qiskit.qiskiterror import QiskitError
from openpulse.qobj.op_system import OPSystem
from openpulse.solver.options import OPoptions
from openpulse.qobj.opparse import HamiltonianParser, NoiseParser
from openpulse.qobj.operators import init_fock_state, qubit_occ_oper
# pylint: disable=no-name-in-module,import-error
from openpulse.cython.utils import oplist_to_array

"""A module of routines for digesting a PULSE qobj
into something we can actually use.
"""

def digest_pulse_obj(qobj):
    """Takes an input PULSE obj an disgests it into
    things we can actually make use of.
    
    Args:
        qobj (Qobj): Qobj of PULSE type.

    Returns:
        OPSystem: The parsed qobj.
    """
    # Output data object
    out = OPSystem()

    config_keys = qobj['config'].keys()

    # Look for config keys
    out.global_data['shots'] = 1024
    if 'shots' in config_keys:
        out.global_data['shots'] = qobj['config']['shots']

    out.global_data['seed'] = None
    if 'seed' in config_keys:
        out.global_data['seed'] = int(qobj['config']['seed'])

    if 'memory_slots' in config_keys:
        out.global_data['memory_slots'] = qobj['config']['memory_slots']
    else:
        err_str = 'Number of memory_slots must be specific in Qobj config'
        raise QiskitError(err_str)
    
    if 'memory' in config_keys:
        out.global_data['memory'] = qobj['config']['memory']
    else:
         out.global_data['memory'] = False

    out.global_data['n_registers'] = 0
    if 'n_registers' in config_keys:
        out.global_data['n_registers'] = qobj['config']['n_registers']
    
    # Attach the ODE options
    allowed_ode_options = ['atol', 'rtol', 'nsteps', 'max_step', 
                       'num_cpus', 'norm_tol', 'norm_steps',
                       'rhs_reuse', 'rhs_filename']
    user_set_ode_options = {}
    if 'ode_options' in config_keys:
        for key, val in qobj['config']['ode_options'].items():
            if key not in allowed_ode_options:
                raise Exception('Invalid ode_option: {}'.format(key))
            else:
                user_set_ode_options[key] = val
    out.ode_options = OPoptions(**user_set_ode_options)


    # Step #1: Parse hamiltonian representation
    if 'hamiltonian' not in config_keys:
        raise QiskitError('Qobj must have hamiltonian in config to simulate.')
    else:
        ham = qobj['config']['hamiltonian']
        
        out.vars = OrderedDict(ham['vars'])
        out.global_data['vars'] = list(out.vars.values())
        
        # Get qubit subspace dimensions
        if 'qub' in ham.keys():
            dim_qub = ham['qub']
            _dim_qub = {}
            # Convert str keys to int keys
            for key, val in ham['qub'].items():
                _dim_qub[int(key)] = val
            dim_qub = _dim_qub
        else:
            dim_qub = {}.fromkeys(range(qobj['config']['n_qubits']), 2)
            
        # Get oscillator subspace dimensions
        if 'osc' in ham.keys():
            dim_osc = ham['osc']
            _dim_osc = {}
            # Convert str keys to int keys
            for key, val in dim_osc.items():
                _dim_osc[int(key)] = val
            dim_osc = _dim_osc 
        else:
            dim_osc = {}
        
        # Parse the Hamiltonian
        system = HamiltonianParser(h_str=ham['h_str'],
                                   dim_osc=dim_osc,
                                   dim_qub=dim_qub)
        system.parse()
        out.system = system.compiled
        
        if 'noise' in qobj['config'].keys():
            noise = NoiseParser(noise_dict=qobj['config']['noise'],
                                dim_osc=dim_osc, dim_qub=dim_qub)
            noise.parse()
        
            out.noise = noise.compiled
            out.can_sample = False
        else:
            out.noise = None

    # Set initial state
    out.initial_state = init_fock_state(dim_osc, dim_qub)
    
    # Step #2: Get Hamiltonian channels
    out.channels = get_hamiltonian_channels(ham['h_str'])
    
    # Step #3: Build pulse arrays
    pulses, pulses_idx, pulse_dict = build_pulse_arrays(qobj)

    out.global_data['pulse_array'] = pulses
    out.global_data['pulse_indices'] = pulses_idx
    out.pulse_to_int = pulse_dict
    
    # Step #4: Get dt
    if 'dt' not in qobj['config'].keys():
        raise QiskitError('Qobj must have a dt value to simulate.')
    else:
        out.dt = qobj['config']['dt']

    
    # Set the ODE solver max step to be the half the width of the smallest pulse
    min_width = np.iinfo(np.int32).max
    for key, val in out.pulse_to_int.items():
        if key != 'pv':
            stop = out.global_data['pulse_indices'][val+1]
            start = out.global_data['pulse_indices'][val]
            min_width = min(min_width, stop-start)
    out.ode_options.max_step = min_width/2 * out.dt
    
    # Step #6: Convert experiments to data structures.

    out.global_data['measurement_ops'] = [None]*qobj['config']['n_qubits']

    for exp in qobj['experiments']:
        exp_struct = experiment_to_structs(exp, 
                                           out.channels,
                                           out.global_data['pulse_indices'],
                                           out.pulse_to_int,
                                           out.dt)
        if len(exp_struct['acquire']) > 0:
            for acq in exp_struct['acquire']:
                for jj in acq[1]:
                    if not out.global_data['measurement_ops'][jj]:
                        out.global_data['measurement_ops'][jj] = \
                            qubit_occ_oper(jj,
                                           h_osc=dim_osc,
                                           h_qub=dim_qub,
                                           level=1
                                          )
        
        out.experiments.append(exp_struct)
        if not exp_struct['can_sample']:
            out.can_sample = False
    return out

def get_hamiltonian_channels(hamstring):
    """ Get all the qubit channels D_i and U_i in the string
    representation of a system Hamiltonian.
    
    Parameters:
        hamstring (list): A list holding strings giving the Hamiltonian
        of a specific quantum system.
        
    Returns:
        list: A list of all channels in Hamiltonian string.
        
    Raises:
        Exception: Missing index on channel.
    """
    out_channels = []
    for ham_str in hamstring:
        chan_idx = [i for i, letter in enumerate(ham_str) if letter in ['D', 'U']]
        for ch in chan_idx:
            if (ch+1) == len(ham_str) or not ham_str[ch+1].isdigit():
                raise Exception('Channel name must include an integer labeling the qubit.')
        for kk in chan_idx:
            done = False
            offset = 0
            while not done:
                offset += 1
                if not ham_str[kk+offset].isdigit():
                    done = True
                # In case we hit the end of the string
                elif (kk+offset+1) == len(ham_str):
                    done = True
                    offset += 1
            temp_chan = ham_str[kk:kk+offset]
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
        ndarray, ndarray, dict: Returns all pulses in one array,
        an array of start indices for pulses, and dict that
        maps pulses to the index at which the pulses start.
    """
    qobj_pulses = qobj['config']['pulse_library']
    pulse_dict = {}
    total_pulse_length = 0
    for kk, pulse in enumerate(qobj_pulses):
        pulse_dict[pulse['name']] = kk
        total_pulse_length += len(pulse['samples'])
    
    idx = kk+1
    #now go through experiments looking for PV gates
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
    pulses_idx = np.zeros(idx+1, dtype=np.uint32)
    
    stop = 0
    ind = 1
    for kk, pulse in enumerate(qobj_pulses):
        stop = pulses_idx[ind-1] + len(pulse['samples'])
        pulses_idx[ind] = stop
        oplist_to_array(pulse['samples'], pulses, pulses_idx[ind-1])
        ind += 1
        
    for pv in pv_pulses:
        stop = pulses_idx[ind-1] + 1
        pulses_idx[ind] = stop
        oplist_to_array([pv[0]], pulses, pulses_idx[ind-1])
        ind += 1
    
    return pulses, pulses_idx, pulse_dict

def experiment_to_structs(experiment, ham_chans, pulse_inds, pulse_to_int, dt):
    max_time = 0
    structs = {}
    structs['channels'] = OrderedDict()
    for chan_name in ham_chans:
        structs['channels'][chan_name] = [[], []]
    structs['acquire'] = []
    structs['cond'] = []
    structs['snapshot'] = []
    structs['tlist'] = []
    structs['can_sample'] = True
    # This is a list that tells us whether the last PV pulse on a channel needs to 
    # be assigned a final time based on the next pulse on that channel
    pv_needs_tf = [0]*len(ham_chans)
    
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
                structs['channels'][chan_name][1].extend([inst['t0'],inst['phase'], cond])
            
            # A standard pulse
            else:
                start = inst['t0']
                pulse_int = pulse_to_int[inst['name']]
                pulse_width = (pulse_inds[pulse_int+1]-pulse_inds[pulse_int])*dt
                stop = start + pulse_width
                structs['channels'][chan_name][0].extend([start, stop, pulse_int, cond])
                
                max_time = max(max_time, stop)
                
        # Take care of acquires and snapshots (bfuncs added )
        else:
            # measurements
            if inst['name'] == 'acquire':
                acq_vals = [inst['t0'],
                            np.asarray(inst['qubits'], dtype=np.uint32),
                            np.asarray(inst['memory_slot'], dtype=np.uint32)
                           ]
                if 'register_slot' in inst.keys():
                    acq_vals.append(np.asarray(inst['register_slot'], dtype=np.uint32))
                else:
                    acq_vals.append(None)
                structs['acquire'].append(acq_vals)
                
                #update max_time
                max_time = max(max_time, inst['t0']+dt*inst['duration'])
                
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
                
                #update max_time
                max_time = max(max_time, inst['t0'])
                
                # Add time to tlist
                if inst['t0'] not in structs['tlist']:
                    structs['tlist'].append(inst['t0'])
                 
            # snapshots
            elif inst['name'] == 'snapshot':
                if inst['type'] != 'state':
                    raise QiskitError("Snapshots must be of type 'state'")
                structs['snapshot'].append([inst['t0'], inst['label']])
                
                # Add time to tlist
                if inst['t0'] not in structs['tlist']:
                    structs['tlist'].append(inst['t0'])
                
                #update max_time
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
        structs['channels'][key][0] = np.asarray(structs['channels'][key][0], dtype=float)
        structs['channels'][key][1] = np.asarray(structs['channels'][key][1], dtype=float)

    structs['tlist'] = np.asarray([0]+structs['tlist'], dtype=float)

    if len(structs['acquire']) > 1 or structs['tlist'][-1] > structs['acquire'][-1][0]:
        structs['can_sample'] = False

    return structs