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

"Helper functions for creating HamiltonianModel and PulseSystemModel objects"

from .hamiltonian_model import HamiltonianModel
from .pulse_system_model import PulseSystemModel


'''
functions for hamiltonian strings
'''

def _full_single_q_terms(freq_symbol, anharm_symbol, num_qubits):

    qubit_list = list(range(num_qubits))
    freq_str_list = [freq_symbol + str(idx) for idx in qubit_list]
    anharm_str_list = [anharm_symbol + str(idx) for idx in qubit_list]

    harm_terms = _harmonic_oscillator_str_list(freq_str_list, anharm_str_list, qubit_list)
    anharm_terms = _anharmonic_oscillator_str_list(anharm_str_list, qubit_list)

    return harm_terms + anharm_terms

'''
functions for types of terms
'''

def _harmonic_oscillator_str_list(freq_str_list, anharm_str_list, qubit_idx_list):
    return _str_list_generator('np.pi*(2*{0}-{1})*O{2}',
                               freq_str_list,
                               anharm_str_list,
                               qubit_idx_list)

def _anharmonic_oscillator_str_list(anharm_str_list, qubit_idx_list):
    return _str_list_generator('np.pi*{0}*O{1}*O{1}',
                               anharm_str_list,
                               qubit_idx_list)

def _qubit_drive_str_list(drive_str_list, qubit_idx_list):
    return _str_list_generator('2*np.pi*{0}*X{1}||D{1}',
                               drive_str_list,
                               qubit_idx_list)

def _u_drive_str_list(drive_str_list, qubit_idx_list ,u_idx_list):
    """
    Should qubit_idx be changed to target_idx??? (Ask Dave)
    """
    return _str_list_generator('2*np.pi*{0}*X{1}||U{2}',
                               drive_str_list,
                               qubit_idx_list,
                               u_idx_list)

def _exchange_coupling_str_list(coupling_str_list, q1_idx_list, q2_idx_list):
    return _str_list_generator('2*np.pi*{0}*(Sp{1}*Sm{2}+Sm{1}*Sp{2})',
                               coupling_str_list,
                               q1_idx_list,
                               q2_idx_list)

def _str_list_generator(str_template, *args):
    """Given a string template, returns a list where each entry is the template formatted by the
    zip of args. It is assumed that either args is a tuple of lists each of the same length, or
    is a tuple with each entry beign either an str or int.
    E.g.
    1. _str_list_generator('First: {0}, Second: {1}', 'a0', 'b0') returns ['First: a0, Second: b0']
    2. _str_list_generator('First: {0}, Second: {1}', ['a0', 'a1'], ['b0', 'b1']) returns
       ['First: a0, Second: b0', 'First: a1, Second: b1']

    Args:
        str_template (str): string template
        args (tuple): assumed to be either tuple of lists of the same length, or a tuple with
                      entries that are either type str or int

    Returns:
        list of strings

    Raises:
    """

    args = [_arg_to_list(arg) for arg in args]
    return [str_template.format(*zipped_arg) for zipped_arg in zip(*args)]

def _arg_to_list(arg):
    '''
    check if arg is a list, if not put it into a list
    '''
    if isinstance(arg, list):
        return arg

    return [arg]
