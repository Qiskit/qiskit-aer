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


"""
Helper functions for creating pieces necessary to construct transmon system models
"""

def _transmon_hamiltonian_dict(transmons,
                               transmon_dims,
                               transmon_freqs,
                               freq_symbols,
                               anharm_freqs,
                               anharm_symbols,
                               drive_strengths,
                               drive_symbols,
                               sorted_coupling_graph,
                               coupling_strengths,
                               coupling_symbols,
                               cr_idx_dict):
    """
    assumptions:
    transmons, transmon_freqs, anharm_freqs, drive_strengths, freq_symbols, anharm_symbols,
    drive_symbols all are same length and are in corresponding order

    sorted_coupling_graph, coupling_strength, and coupling_symbols are all same length and in
    corresponding order

    cr_idx_dict can be of any length
    """

    # single transmon terms
    hamiltonian_str = _single_transmon_drift_terms(freq_symbols, anharm_symbols, transmons)
    hamiltonian_str += _drive_terms(drive_symbols, transmons)

    # two transmon terms
    hamiltonian_str += _exchange_coupling_terms(coupling_symbols, sorted_coupling_graph)
    driven_transmon_indices = [key[0] for key in cr_idx_dict.keys()]
    cr_channel_idx = cr_idx_dict.values()
    hamiltonian_str += _cr_terms(drive_symbols, driven_transmon_indices, cr_channel_idx)

    # construct vars dictionary
    var_dict = {}
    for idx in transmons:
        var_dict[freq_symbols[idx]] = transmon_freqs[idx]
        var_dict[drive_symbols[idx]] = drive_strengths[idx]

    for symbol, strength in zip(coupling_symbols, coupling_strengths):
        var_dict[symbol] = strength

    dim_dict = {str(transmon) : dim for transmon, dim in zip(transmons, transmon_dims)}

    return {'h_str': hamiltonian_str, 'vars': var_dict, 'qub': dim_dict}

def _cr_lo_list(cr_idx_dict):

    # populate list of u channel lo for cr gates
    lo_list = [0]*len(cr_idx_dict)
    for qubit_pair, u_idx in cr_idx_dict.items():
        lo_list[u_idx] = [{'scale' : [1.0, 0.0], 'q' : qubit_pair[0]}]

    return lo_list


"""
Functions for creating Hamiltonian strings for certain types of terms
"""

def _single_transmon_drift_terms(freq_symbols, anharm_symbols, transmon_list):


    harm_terms = _str_list_generator('np.pi*(2*{0}-{1})*O{2}',
                                     freq_symbols,
                                     anharm_symbols,
                                     transmon_list)
    anharm_terms = _str_list_generator('np.pi*{0}*O{1}*O{1}',
                                       anharm_symbols,
                                       transmon_list)

    return harm_terms + anharm_terms

def _drive_terms(drive_symbols, transmon_list):

    return _str_list_generator('2*np.pi*{0}*X{1}||D{1}',
                               drive_symbols,
                               transmon_list)

def _exchange_coupling_terms(coupling_symbols, ordered_edges):

    idx1_list, idx2_list = zip(*list(ordered_edges))

    return _str_list_generator('2*np.pi*{0}*(Sp{1}*Sm{2}+Sm{1}*Sp{2})',
                               coupling_symbols,
                               idx1_list,
                               idx2_list)


def _cr_terms(drive_symbols, driven_transmon_indices, u_channel_indices):

    return _str_list_generator('2*np.pi*{0}*X{1}||U{2}',
                               drive_symbols,
                               driven_transmon_indices,
                               u_channel_indices)


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


"""
Helper classes
"""

class _coupling_graph:
    """
    Helper class containing some functionality for representing coupling graphs. The main points
    are: having a representation that is set-like (self.graph), having a fixed ordering of edges
    (self.sorted_graph), and having a fixed ordering for bi-directional edges
    (self.sorted_two_way_graph) for setting up CR channels.
    """

    def __init__(self, edges):

        self.graph = {frozenset({idx1, idx2}) for idx1, idx2 in edges}

        graph_list = []
        for edge in self.graph:
            edge_list = list(edge)
            edge_list.sort()
            graph_list.append(tuple(edge_list))

        graph_list.sort()

        self.sorted_graph = graph_list

        two_way_graph_list = []
        for edge in self.sorted_graph:
            two_way_graph_list.append(edge)
            two_way_graph_list.append((edge[1], edge[0]))

        self.sorted_two_way_graph = two_way_graph_list

        self.two_way_graph_dict = {self.sorted_two_way_graph[k] : k for k in range(len(self.sorted_two_way_graph))}

    def sorted_edge_index(self, edge):
        edge_list = list(edge)
        edge_list.sort()
        return self.sorted_graph.index(tuple(edge))

    def two_way_edge_index(self, edge):
        return self.two_way_graph_dict[tuple(edge)]
