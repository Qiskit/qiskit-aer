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

"Helper functions for creating HamiltonianModel and PulseSystemModel objects"

from warnings import warn
from collections.abc import Iterable
from .hamiltonian_model import HamiltonianModel
from .pulse_system_model import PulseSystemModel


def transmon_system_model(num_transmons,
                          dim_transmons,
                          transmon_freqs,
                          anharm_freqs,
                          drive_strengths,
                          coupling_dict,
                          dt,
                          freq_symbol='v',
                          anharm_symbol='alpha',
                          drive_symbol='r',
                          coupling_symbol='j'):
    """Creates a PulseSystemModel for a transmon system with a specified number of transmons
    num_transmons, each truncated to the dimension dim_transmons. To perform cross resonance drives
    on coupled qubits, use the ControlChannel indices contained in the returned cr_idx_dict.

    Single transmons are specified by the parameters:
        - transmon frequency v,
        - anharmonicity frequency alpha, and
        - drive strength r,
    which enter into the Hamiltonian model via the terms:
        pi*(2*v-alpha)*O + pi*alpha*O*O + 2*pi*r*X*D(t),
    where
        - O is the number operator for the transmon,
        - X = A + C, where A and C are, respectively, the annihilation and creation operators, and
        - D(t) is the drive signal for the transmon.

    Couplings between transmons are specified in terms of a coupling map, implemented as a dict
    passed in coupling_dict. E.g. the coupling_dict
        {(0, 1): 0.02, (1,3): 0.01}
    specifies that qubits (0, 1) are coupled with strength 0.02, and the qubits (1, 3) are
    coupled with strength 0.01. Couplings enter the Hamiltonian model via the exchange coupling
    term:
        2*pi*j*(A0*C1+C0*A1),
    where
        - j is the coupling strength,
        - A0 and C0 are the annihilation and creation operators of the first transmon, and
        - A1 and C1 are the annihilation and creation operators of the second transmon.

    For each pair of coupled transmons, pulse ControlChannels are prepared for doing cross
    resonance (CR) drives between qubits. A CR drive on transmon 0 with target transmon 1 is modeled
    with the Hamiltonian term:
        2*pi*r0*X0*U(t),
    where
        - r0 is the drive strength for transmon 0,
        - X0 is the drive operator for transmon 0, and
        - U(t) is the signal for the CR drive, which will be set to have carrier frequency equal to
          the user-specified carrier frequency of the target transmon.

    Indices for ControlChannels corresponding to CR drives are provided in the returned cr_idx_dict.
    To access the control channel index for a CR drive on transmons (drive, target) use:
        cr_idx_dict[(drive, target)]
    E.g. For the coupling_dict above, cr_idx_dict will be automatically generated as:
        {(0,1): 0, (1,0): 1, (1,3): 2, (3,1): 3}.
    Hence, the ControlChannel to use for a CR drive on transmon 1 with target transmon 3 is:
        cr_idx_dict[(1,3)] == 2.

    Args:
        num_transmons (int): Number of transmons in the model
        dim_transmons (int): Dimension of truncation for each transmon
        transmon_freqs (list): Transmon frequencies, assumed to be of length num_transmons
        anharm_freqs (list): Anharmonicity values, assumed to be of length num_transmons
        drive_strengths (list): Drive strength values, assumed to be of length num_transmons
        coupling_dict (dict): Specification of the coupling graph with keys being edges, and values
                              the strength of the coupling.
        dt (float): Pixel size for pulse instructions
        freq_symbol (str): Frequency symbol for internal HamiltonianModel
        anharm_symbol (str): Anharmonicity symbol for internal HamiltonianModel
        drive_symbol (str): Drive symbol for internal HamiltonianModel
        coupling_symbol (str): Coupling symbol for internal HamiltonianModel

    Returns:
        tuple[PulseSystemModel, dict]: The generated transmon system model, and a dict specifying
                                       u channel index to use for a specified CR drive;
                                       I.e. to specify a CR drive with qubits (drive, target),
                                       retrieve the u channel index via:
                                            cr_idx_dict[(drive, target)].
                                       For example, for the coupling_dict:
                                            {(0, 1): 0.02, (1,3): 0.01},
                                       the corresponding cr_idx_dict will be:
                                            {(0,1): 0, (1,0): 1, (1,3): 2, (3,1): 3}
    """

    coupling_edges = coupling_dict.keys()

    # construct coupling graph, and raise warning if coupling_edges contains duplicate edges
    coupling_graph = _coupling_graph(coupling_edges)
    if len(coupling_graph.graph) < len(coupling_edges):
        warn('Warning: The coupling_dict contains diplicate edges, and the second appearance of \
              the same edge will be ignored.')

    # construct the HamiltonianModel
    transmons = list(range(num_transmons))
    transmon_dims = [dim_transmons] * num_transmons
    freq_symbols = _str_list_generator(freq_symbol + '{0}', transmons)
    anharm_symbols = _str_list_generator(anharm_symbol + '{0}', transmons)
    drive_symbols = _str_list_generator(drive_symbol + '{0}', transmons)
    sorted_coupling_edges = coupling_graph.sorted_graph
    # populate coupling strengths in order (careful wtih vertices in edges now being sorted)
    coupling_strengths = []
    for edge in sorted_coupling_edges:
        weight = coupling_dict.get(edge, None)
        if weight is None:
            weight = coupling_dict[(edge[1], edge[0])]
        coupling_strengths.append(weight)
    coupling_symbols = _str_list_generator(coupling_symbol + '{0}{1}', *zip(*sorted_coupling_edges))
    cr_idx_dict = coupling_graph.two_way_graph_dict

    hamiltonian_dict = _transmon_hamiltonian_dict(transmons=transmons,
                                                  transmon_dims=transmon_dims,
                                                  transmon_freqs=transmon_freqs,
                                                  freq_symbols=freq_symbols,
                                                  anharm_freqs=anharm_freqs,
                                                  anharm_symbols=anharm_symbols,
                                                  drive_strengths=drive_strengths,
                                                  drive_symbols=drive_symbols,
                                                  ordered_coupling_edges=sorted_coupling_edges,
                                                  coupling_strengths=coupling_strengths,
                                                  coupling_symbols=coupling_symbols,
                                                  cr_idx_dict=cr_idx_dict)

    hamiltonian_model = HamiltonianModel.from_dict(hamiltonian_dict)

    # construct the u_channel_lo list
    u_channel_lo = _cr_lo_list(cr_idx_dict)

    system_model = PulseSystemModel(hamiltonian=hamiltonian_model,
                                    u_channel_lo=u_channel_lo,
                                    qubit_list=transmons,
                                    dt=dt)

    return system_model, cr_idx_dict


# Helper functions for creating pieces necessary to construct transmon system models


def _transmon_hamiltonian_dict(transmons,
                               transmon_dims,
                               transmon_freqs,
                               freq_symbols,
                               anharm_freqs,
                               anharm_symbols,
                               drive_strengths,
                               drive_symbols,
                               ordered_coupling_edges,
                               coupling_strengths,
                               coupling_symbols,
                               cr_idx_dict):
    """Creates a hamiltonian string dict based on the arguments.

    Note, this function makes the following assumptions:
        - transmons, transmon_dims, transmon_freqs, freq_symbols, anharm_freqs, anharm_symbols,
          drive_strengths, and drive_symbols are all lists of the same length (i.e. the total
          transmon number)
        - ordered_coupling_edges, coupling_strengths, and coupling_symbols are lists of the same
          length

    Args:
        transmons (list): ints for transmon labels
        transmon_dims (list): ints for transmon dimensions
        transmon_freqs (list): transmon frequencies
        freq_symbols (list): symbols to be used for transmon frequencies
        anharm_freqs (list): anharmonicity values
        anharm_symbols (list): symbols to be used for anharmonicity terms
        drive_strengths (list): drive strength coefficients
        drive_symbols (list): symbols for drive coefficients
        ordered_coupling_edges (list): tuples of two ints specifying transmon couplings. Order
                                       corresponds to order of coupling_strengths and
                                       coupling_symbols
        coupling_strengths (list): strength of each coupling term (corresponds to ordering of
                                   ordered_coupling_edges)
        coupling_symbols (list): symbols for coupling coefficients
        cr_idx_dict (dict): A dict with keys given by tuples containing two ints, and value an int,
                            representing cross resonance drive channels. E.g. an entry {(0,1) : 1}
                            specifies a CR drive on qubit 0 with qubit 1 as target, with u_channel
                            index 1.

    Returns:
        dict: hamiltonian string format
    """

    # single transmon terms
    hamiltonian_str = _single_transmon_drift_terms(freq_symbols, anharm_symbols, transmons)
    hamiltonian_str += _drive_terms(drive_symbols, transmons)

    # exchange terms
    hamiltonian_str += _exchange_coupling_terms(coupling_symbols, ordered_coupling_edges)

    # cr terms
    driven_transmon_indices = [key[0] for key in cr_idx_dict.keys()]
    cr_drive_symbols = [drive_symbols[idx] for idx in driven_transmon_indices]
    cr_channel_idx = cr_idx_dict.values()
    hamiltonian_str += _cr_terms(cr_drive_symbols, driven_transmon_indices, cr_channel_idx)

    # construct vars dictionary
    var_dict = {}
    for idx in transmons:
        var_dict[freq_symbols[idx]] = transmon_freqs[idx]
        var_dict[anharm_symbols[idx]] = anharm_freqs[idx]
        var_dict[drive_symbols[idx]] = drive_strengths[idx]

    for symbol, strength in zip(coupling_symbols, coupling_strengths):
        var_dict[symbol] = strength

    dim_dict = {str(transmon): dim for transmon, dim in zip(transmons, transmon_dims)}

    return {'h_str': hamiltonian_str, 'vars': var_dict, 'qub': dim_dict}


def _cr_lo_list(cr_idx_dict):
    """Generates u_channel_lo list for a PulseSystemModel from a cr_idx_dict.

    Args:
        cr_idx_dict (dict): A dictionary with keys given by tuples of ints with int values. A key,
                            e.g. (0,1), signifies CR drive on qubit 0 with target 1, and the value
                            is the u channel index corresponding to that drive.
                            Note: this function assumes that
                            cr_idx_dict.values() == range(len(cr_idx_dict)).

    Returns:
        list: u_channel_lo format required by the simulator
    """

    # populate list of u channel lo for cr gates
    lo_list = [0] * len(cr_idx_dict)
    for qubit_pair, u_idx in cr_idx_dict.items():
        lo_list[u_idx] = [{'scale': [1.0, 0.0], 'q': qubit_pair[1]}]

    return lo_list


# Functions for creating Hamiltonian strings for various types of terms


def _single_transmon_drift_terms(freq_symbols, anharm_symbols, transmon_list):
    """Harmonic and anharmonic drift terms

    Args:
        freq_symbols (list): coefficients for harmonic part
        anharm_symbols (list): coefficients for anharmonic part
        transmon_list (list): list of transmon indices
    Returns:
        list: drift term strings
    """

    harm_terms = _str_list_generator('np.pi*(2*{0}-{1})*O{2}',
                                     freq_symbols,
                                     anharm_symbols,
                                     transmon_list)
    anharm_terms = _str_list_generator('np.pi*{0}*O{1}*O{1}',
                                       anharm_symbols,
                                       transmon_list)

    return harm_terms + anharm_terms


def _drive_terms(drive_symbols, transmon_list):
    """Drive terms for single transmon

    Args:
        drive_symbols (list): coefficients of drive terms
        transmon_list (list): list of transmon indices
    Returns:
        list: drive term strings
    """

    return _str_list_generator('2*np.pi*{0}*X{1}||D{1}',
                               drive_symbols,
                               transmon_list)


def _exchange_coupling_terms(coupling_symbols, ordered_edges):
    """Exchange coupling between transmons

    Args:
        coupling_symbols (list): coefficients of exchange couplings
        ordered_edges (list): list tuples of transmon indices for the couplings
    Returns:
        list: exchange coupling strings
    """

    idx1_list, idx2_list = zip(*list(ordered_edges))

    return _str_list_generator('2*np.pi*{0}*(Sp{1}*Sm{2}+Sm{1}*Sp{2})',
                               coupling_symbols,
                               idx1_list,
                               idx2_list)


def _cr_terms(drive_symbols, driven_transmon_indices, u_channel_indices):
    """Cross resonance drive terms

    Args:
        drive_symbols (list): coefficients for drive terms
        driven_transmon_indices (list): list of indices for transmons that drive is applied to
        u_channel_indices (list): indicies for the u_channels corresponding to each term
    Returns:
        list: cr term strings
    """

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
        list: list of str_template formated by args lists
    """

    args = [_arg_to_iterable(arg) for arg in args]
    return [str_template.format(*zipped_arg) for zipped_arg in zip(*args)]


def _arg_to_iterable(arg):
    """Check if arg is an iterable, if not put it into a list. The purpose is to allow arguments
    of functions to be either lists or singletons, e.g. instead of having to pass ['a'], 'a' can be
    passed directly.

    Args:
        arg (Iterable): argument to be checked and turned into an interable if necessary

    Returns:
        Iterable: either arg, or arg transformed into a list
    """
    # catch expected types (issue is str is iterable)
    if isinstance(arg, (int, str)):
        return [arg]

    if isinstance(arg, Iterable):
        return arg

    return [arg]


# Helper classes


class _coupling_graph:
    """
    Helper class containing functionality for representing coupling graphs, with the main goal to
    construct different representations for different purposes:
        - self.graph: graph as a set of edges stored as frozen sets, e.g.
                    {frozenset({0,1}), frozenset({1,2}), frozenset({2,3})}
        - self.sorted_graph: graph as a list of tuples in lexicographic order, e.g.
                    [(0,1), (1,2), (2,3)]
          Note: these are actively ordered by the object, as the point is to have a canonical
          ordering of edges. The integers in the tuples are also ordered.
        - self.sorted_two_way_graph: list of tuples where each edge is repeated with the vertices
          reversed. The ordering is the same as in sorted_graph, with the duplicate appearing
          immediately after the original, e.g.
                    [(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)]
        - self.two_way_graph_dict: same as above, but in dict form, e.g.
                    {(0,1) : 0, (1,0) : 1, (1,2) : 2, (2,1) : 3, (2,3) : 4, (3,2) : 5}
    """

    def __init__(self, edges):
        """returns _coupling_graph object

        Args:
            edges (Iterable): An iterable of iterables, where the inner interables are assumed to
                              contain two elements, e.g. [(0,1), (2,3)], or ((0,1), (2,3))

        Returns:
            _coupling_graph: coupling graph specified by edges
        """

        # create the set representation of the graph
        self.graph = {frozenset({idx1, idx2}) for idx1, idx2 in edges}

        # created the sorted list representation
        graph_list = []
        for edge in self.graph:
            edge_list = list(edge)
            edge_list.sort()
            graph_list.append(tuple(edge_list))

        graph_list.sort()
        self.sorted_graph = graph_list

        # create the sorted_two_way_graph
        two_way_graph_list = []
        for edge in self.sorted_graph:
            two_way_graph_list.append(edge)
            two_way_graph_list.append((edge[1], edge[0]))

        self.sorted_two_way_graph = two_way_graph_list

        # create the dictionary version
        self.two_way_graph_dict = {self.sorted_two_way_graph[k]: k
                                   for k in range(len(self.sorted_two_way_graph))}

    def sorted_edge_index(self, edge):
        """Given an edge, returns the index in self.sorted_graph. Order in edge does not matter.

        Args:
            edge (Iterable): an iterable containing two integers

        Returns:
            int: index of edge
        """
        edge_list = list(edge)
        edge_list.sort()
        return self.sorted_graph.index(tuple(edge_list))

    def two_way_edge_index(self, directed_edge):
        """Given a directed edge, returns the index in self.sorted_two_way_graph

        Args:
            directed_edge (Iterable): an iterable containing two integers

        Returns:
            int: index of directed_edge
        """
        return self.two_way_graph_dict[tuple(directed_edge)]
