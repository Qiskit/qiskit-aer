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

from collections.abc import Iterable
from .hamiltonian_model import HamiltonianModel
from .pulse_system_model import PulseSystemModel

# line for complete graph edges
# edges = [(i,j) for j in range(num_transmons) for i in range(num_transmons) if j > i]

"""
Functions for constructing entire PulseSystemModel objects from a system level description
"""

def uniform_transmon_system_complete_graph(num_transmons,
                                           dim_transmons,
                                           base_freq,
                                           freq_offset,
                                           anharm,
                                           drive_strength,
                                           coupling_strength,
                                           dt):

    # specify edges of the complete graph
    edges = [(i,j) for j in range(num_transmons) for i in range(num_transmons) if j > i]

    return uniform_transmon_system_from_graph(num_transmons,
                                              dim_transmons,
                                              edges,
                                              base_freq,
                                              freq_offset,
                                              anharm,
                                              drive_strength,
                                              coupling_strength,
                                              dt)

# should output also a list for the u channel orderings
def uniform_transmon_system_from_graph(num_transmons,
                                       dim_transmons,
                                       edges,
                                       base_freq,
                                       freq_offset,
                                       anharm,
                                       drive_strength,
                                       coupling_strength,
                                       dt):

    hamiltonian_dict = _uniform_transmon_hamiltonian(num_transmons,
                                                     dim_transmons,
                                                     edges,
                                                     base_freq,
                                                     freq_offset,
                                                     anharm,
                                                     drive_strength,
                                                     coupling_strength)
    ham_model = HamiltonianModel.from_dict(hamiltonian_dict)

    ############
    # do this
    ###########
    u_channel_lo, u_channel_labels = []



    system_model = PulseSystemModel(hamiltonian=ham_model,
                                    u_channel_lo=u_channel_lo,
                                    qubit_list=list(range(num_transmons)),
                                    dt=dt)

    return system_model, u_channel_labels

"""
u_channel_lo specification generators
"""

# Do this, need to return description of u_channel labels
def _u_channel_lo_from_graph(edges):
    return [], {}

"""
Functions for constructing entire transmon Hamiltonian dicts from system level descriptions
"""

def _uniform_transmon_hamiltonian_dict(num_transmons,
                                       dim_transmons,
                                       edges,
                                       base_freq,
                                       freq_offset,
                                       anharm,
                                       drive_strength,
                                       coupling_strength,
                                       freq_symbol='v',
                                       anharm_symbol='alpha',
                                       drive_symbol='r',
                                       coupling_symbol='j'):


    # construct individual qubit terms
    hamiltonian_str = _single_transmon_drift_terms(freq_symbol, anharm_symbol, num_transmons)
    hamiltonian_str += _drive_terms(drive_symbol, num_transmons, all_same_drive=True)

    # construct two transmon terms
    hamiltonian_str += _exchange_coupling_terms(coupling_symbol, edges, all_same_coupling=True)
    hamiltonian_str += _u_channel_terms(drive_symbol, edges, all_same_drive=True)

    # construct variable dictionary
    var_dict = {}
    for transmon_idx in range(num_transmons):
        var_dict[freq_symbol + str(transmon_idx)] = base_freq + transmon_idx * freq_offset

    for transmon_idx in range(num_transmons):
        var_dict[anharm_symbol + str(transmon_idx)] = anharm

    var_dict[drive_symbol] = drive_strength
    var_dict[coupling_symbol] = coupling_strength

    # construct transmon dimension dictionary
    dim_dict = {str(idx) : dim_transmons for idx in range(num_transmons)}

    return {'h_str': hamiltonian_str, 'vars': var_dict, 'qub': dim_dict}

def _full_transmon_hamiltonian_dict_old(num_transmons,
                                    dim_transmons,
                                    transmon_freqs,
                                    anharm_freqs,
                                    drive_strengths,
                                    edges,
                                    coupling_strengths,
                                    coupling_edges,
                                    freq_symbol='v',
                                    anharm_symbol='alpha',
                                    drive_symbol='r',
                                    coupling_symbol='j'):

    # construct individual qubit terms
    hamiltonian_str = _single_transmon_drift_terms(freq_symbol, anharm_symbol, num_transmons)
    hamiltonian_str += _drive_terms(drive_symbol, num_transmons, all_same_drive=True)

    # construct two transmon terms
    hamiltonian_str += _exchange_coupling_terms(coupling_symbol, edges, all_same_coupling=True)
    hamiltonian_str += _u_channel_terms(drive_symbol, edges, all_same_drive=True)

    # construct variable dictionary
    var_dict = {freq_symbol + str(idx) : transmon_freqs[idx] for idx in range(num_transmons)}
    var_dict.update({anharm_symbol + str(idx) : anharm_freqs[idx] for idx in range(num_transmons)})
    var_dict.update({drive_symbol + str(idx) : drive_strengths[idx] for idx in range(num_transmons)})
    var_dict.update({coupling_symbol + str(idx) : coupling_strengths[idx] for idx in range(len(edges))})


    for transmon_idx in range(num_transmons):
        var_dict[freq_symbol + str(transmon_idx)] = transmon_freqs[transmon_idx]

    for transmon_idx in range(num_transmons):
        var_dict[anharm_symbol + str(transmon_idx)] = anharm

    var_dict[drive_symbol] = drive_strength
    var_dict[coupling_symbol] = coupling_strength


def _edges_to_graph(edges):
    return {frozenset({idx1, idx2}) for idx1, idx2 in edges}

def _graph_to_cr_dict(graph):
    cr_dict = {}
    for idx1, idx2 in graph:
        cr_dict[(idx1, idx2)] = len(cr_dict)
        cr_dict[(idx2, idx1)] = len(cr_dict)

    return cr_dict


"""
Functions for constructing Hamiltonian strings for particular types of terms from a system
level description
"""

def _single_transmon_drift_terms(freq_symbol, anharm_symbol, num_transmons):
    """Returns a complete list of single transmon Hamiltonian strings, using freq_symbol and
    anharm_symbol as the base string for frequency and anharmonicity symbols.

    E.g. _full_single_transmon_terms('v', 'alpha', 2) returns
    ['np.pi*(2*v0-alpha0)*O0', 'np.pi*(2*v1-alpha1)*O1', 'np.pi*alpha0*O0*O0', 'np.pi*alpha1*O1*O1']

    Args:
        freq_symbol (str): string to use for frequency symbols
        anharm_symbol (str): string to use for anharmonicity symbols
        num_transmons (int): number of transmons

    Returns:
        list of strings of single transmon terms

    Raises:
    """

    transmon_list = list(range(num_transmons))
    freq_str_list = [freq_symbol + str(idx) for idx in transmon_list]
    anharm_str_list = [anharm_symbol + str(idx) for idx in transmon_list]

    harm_terms = _harmonic_oscillator_str_list(freq_str_list, anharm_str_list, transmon_list)
    anharm_terms = _anharmonic_oscillator_str_list(anharm_str_list, transmon_list)

    return harm_terms + anharm_terms

def _drive_terms(drive_symbol, num_systems, all_same_drive=False):
    """Returns a list of single system drive terms, using drive_symbol as the base string for
    drive strengths

    Args:
        drive_symbol (str): base string to use for drive strengths
        num_systems (int): number of systems
        all_same_drive (bool): if False, each drive strength is given a unique symbol,
                               e.g. 'r0', 'r1', ... . If True drive_symbol is used as the strength
                               for all drive terms

    Returns:
        list of strings of system drive terms

    Raises:
    """

    system_list = list(range(num_systems))

    if all_same_drive:
        drive_str_list = [drive_symbol] * num_systems
    else:
        drive_str_list = [drive_symbol + str(idx) for idx in system_list]

    return _qubit_drive_str_list(drive_str_list, system_list)

def _exchange_coupling_terms(coupling_symbol, coupling_graph, all_same_coupling=False):
    """Returns a list of exchange coupling terms, according to the coupling_graph

    Args:
        coupling_symbol (str): base string to use for coupling strengths
        coupling_graph (set): a set containing the edges of the graph, represented using frozensets
        all_same_coupling (bool): if False, each coupling term is given a unique symbol, if True,
                               all are given the symbol coupling_symbol

    Returns:
        list of strings of coupling terms

    Raises:
    """

    edges = [(idx1, idx2) for idx1, idx2 in coupling_graph]

    q1_idx_list, q2_idx_list = zip(*edges)

    if all_same_coupling:
        coupling_str_list = [coupling_symbol] * len(edges)
    else:
        coupling_str_list = [coupling_symbol + str(idx1) + str(idx2) for idx1, idx2 in edges]

    return _exchange_coupling_str_list(coupling_str_list, q1_idx_list, q2_idx_list)


def _u_channel_cr_terms(drive_symbol, cr_dict, all_same_drive=False):
    """Returns a list of u channel terms for a given cr_dict

    Args:
        drive_symbol (str): base string to use for drive strengths
        cr_dict (dict): dict with keys assumed to be tuples containing two integers (representing
                        the driven qubit index, and the target qubit index of a cr drive), and
                        values being integers for the u channel index
        all_same_drive (bool): if False, each coupling term is given a unique symbol, if True,
                               all are given the symbol drive_symbol

    Returns:
        list of strings of u channel drive terms

    Raises:
    """

    # populate indices for correspondence between driven qubit and u channel index
    driven_qubit_idx = []
    u_channel_idx = []

    for key, val in cr_dict.items():
        driven_qubit_idx.append(key[0])
        u_channel_idx.append(val)

    if all_same_drive:
        drive_str_list = [drive_symbol] * len(cr_dict)
    else:
        drive_str_list = [drive_symbol + str(idx) for idx in cr_dict.values()]

    return _u_drive_str_list(drive_str_list, driven_qubit_idx, u_channel_idx)


"""
Low level functions for constructing basic hamiltonian strings directly from templates and
specified lists of parameters
"""

def _harmonic_oscillator_str_list(freq_str_list, anharm_str_list, qubit_idx_list):
    """Construct list of Hamiltonian strings for harmonic oscillator term

    Args:
        freq_str_list (list): list of frequency symbols
        anharm_str_list (list): list of anharmonicity symbols
        qubit_idx_list (list): list of system indices

    Returns:
        list of strings

    Raises:
    """

    return _str_list_generator('np.pi*(2*{0}-{1})*O{2}',
                               freq_str_list,
                               anharm_str_list,
                               qubit_idx_list)

def _anharmonic_oscillator_str_list(anharm_str_list, qubit_idx_list):
    """Construct list of Hamiltonian strings for anharmonic oscillator term

    Args:
        anharm_str_list (list): list of anharmonicity symbols
        qubit_idx_list (list): list of system indices

    Returns:
        list of strings

    Raises:
    """

    return _str_list_generator('np.pi*{0}*O{1}*O{1}',
                               anharm_str_list,
                               qubit_idx_list)

def _qubit_drive_str_list(drive_str_list, qubit_idx_list):
    """Construct list of Hamiltonian strings for qubit drive term

    Args:
        drive_str_list (list): list of drive strength symbols
        qubit_idx_list (list): list of system indices

    Returns:
        list of strings

    Raises:
    """

    return _str_list_generator('2*np.pi*{0}*X{1}||D{1}',
                               drive_str_list,
                               qubit_idx_list)

def _u_drive_str_list(drive_str_list, qubit_idx_list, u_idx_list):
    """Construct list of Hamiltonian strings for u channel drive term

    Args:
        drive_str_list (list): list of drive strength symbols
        qubit_idx_list (list): list of system indices
        u_idx_list (list): list of u channel indices

    Returns:
        list of strings

    Raises:
    """

    return _str_list_generator('2*np.pi*{0}*X{1}||U{2}',
                               drive_str_list,
                               qubit_idx_list,
                               u_idx_list)

def _exchange_coupling_str_list(coupling_str_list, q1_idx_list, q2_idx_list):
    """Construct list of Hamiltonian strings for exchange coupling

    Args:
        coupling_str_list (list): list of coupling strength symbols
        q1_idx_list (list): list of indicies for the first qubit in the coupling
        q2_idx_list (list): list of indicies for the second qubit in the coupling

    Returns:
        list of strings

    Raises:
    """

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
        args (tuple): assumed to be either tuple of iterables of the same length, or a tuple with
                      entries that are either type str or int

    Returns:
        list of strings

    Raises:
    """

    args = [_arg_to_iterable(arg) for arg in args]
    return [str_template.format(*zipped_arg) for zipped_arg in zip(*args)]

def _arg_to_iterable(arg):
    """Check if arg is an iterable, if not put it into a list.

    Args:
        arg (Iterable): argument to be checked and turned into an interable if necessary

    Returns:
        Iterable

    Raises:
    """
    if isinstance(arg, Iterable):
        return arg

    return [arg]
