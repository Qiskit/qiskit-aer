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


def duffing_system_model(dim_oscillators,
                         oscillator_freqs,
                         anharm_freqs,
                         drive_strengths,
                         coupling_dict,
                         dt):
    r"""Returns a :class:`PulseSystemModel` representing a physical model for a
    collection of Duffing oscillators.

    In the model, each individual oscillator is specified by the parameters:

        * Frequency: :math:`\nu`, specified in the list ``oscillator_freqs``
        * Anharmonicity: :math:`\alpha`, specified in the list ``anharm_freqs``, and
        * Drive strength: :math:`r`, specified in the list ``drive_strengths``.

    For each oscillator, the above parameters enter into the Hamiltonian via the terms:

    .. math::
        \pi(2 \nu - \alpha)a^\dagger a +
        \pi \alpha (a^\dagger a)^2 + 2 \pi r D(t) (a + a^\dagger),

    where :math:`a^\dagger` and :math:`a` are, respectively, the creation and annihilation
    operators for the oscillator, and :math:`D(t)` is the drive signal for the oscillator.

    Each coupling term between a pair of oscillators is specified by:

        * Oscillator pair: :math:`(i,k)`, and
        * Coupling strength: :math:`j`,

    which are passed in the argument ``coupling_dict``, which is a ``dict`` with keys
    being the ``tuple`` ``(i,k)``, and values the strength ``j``. Specifying a coupling
    results in the Hamiltonian term:

    .. math::
        2 \pi j (a_i^\dagger a_k + a_i a_k^\dagger).

    Finally, the returned :class:`PulseSystemModel` is setup for performing cross-resonance
    drives between coupled qubits. The index for the :class:`ControlChannel` corresponding
    to a particular cross-resonance drive channel is retreived by calling
    :meth:`PulseSystemModel.control_channel_index` with the tuple ``(drive_idx, target_idx)``,
    where ``drive_idx`` is the index of the oscillator being driven, and ``target_idx`` is
    the target oscillator (see example below).

    Note: In this model, all frequencies are in frequency units (as opposed to radial).

    **Example**

    Constructing a three Duffing Oscillator :class:``PulseSystemModel``.

    .. code-block:: python

        # cutoff dimensions
        dim_oscillators = 3

        # single oscillator drift parameters
        oscillator_freqs = [5.0e9, 5.1e9, 5.2e9]
        anharm_freqs = [-0.33e9, -0.33e9, -0.33e9]

        # drive strengths
        drive_strengths = [0.02e9, 0.02e9, 0.02e9]

        # specify coupling as a dictionary; here the qubit pair (0,1) is coupled with
        # strength 0.002e9, and the qubit pair (1,2) is coupled with strength 0.001e9
        coupling_dict = {(0,1): 0.002e9, (1,2): 0.001e9}

        # time
        dt = 1e-9

        # create the model
        three_qubit_model = duffing_system_model(dim_oscillators=dim_oscillators,
                                                 oscillator_freqs=oscillator_freqs,
                                                 anharm_freqs=anharm_freqs,
                                                 drive_strengths=drive_strengths,
                                                 coupling_dict=coupling_dict,
                                                 dt=dt)

    In the above model, qubit pairs (0,1) and (1,2) are coupled. To perform a
    cross-resonance drive on qubit 1 with target 0, use the :class:`ControlChannel`
    with index:

    .. code-block:: python

        three_qubit_model.control_channel_index((1,0))

    Args:
        dim_oscillators (int): Dimension of truncation for each oscillator.
        oscillator_freqs (list): Oscillator frequencies in frequency units.
        anharm_freqs (list): Anharmonicity values in frequency units.
        drive_strengths (list): Drive strength values in frequency units.
        coupling_dict (dict): Coupling graph with keys being edges, and values
                              the coupling strengths in frequency units.
        dt (float): Sample width for pulse instructions.

    Returns:
        PulseSystemModel: The generated Duffing system model
    """

    # set symbols for string generation
    freq_symbol = 'v'
    anharm_symbol = 'alpha'
    drive_symbol = 'r'
    coupling_symbol = 'j'

    coupling_edges = coupling_dict.keys()

    # construct coupling graph, and raise warning if coupling_edges contains duplicate edges
    coupling_graph = CouplingGraph(coupling_edges)
    if len(coupling_graph.graph) < len(coupling_edges):
        warn('Warning: The coupling_dict contains diplicate edges, and the second appearance of \
              the same edge will be ignored.')

    # construct the HamiltonianModel
    num_oscillators = len(oscillator_freqs)
    oscillators = list(range(num_oscillators))
    oscillator_dims = [dim_oscillators] * num_oscillators
    freq_symbols = _str_list_generator(freq_symbol + '{0}', oscillators)
    anharm_symbols = _str_list_generator(anharm_symbol + '{0}', oscillators)
    drive_symbols = _str_list_generator(drive_symbol + '{0}', oscillators)
    sorted_coupling_edges = coupling_graph.sorted_graph
    # populate coupling strengths in sorted order (vertex indices are now also sorted within edges,
    # so this needs to be accounted for when retrieving weights from coupling_dict)
    coupling_strengths = [coupling_dict.get(edge) or coupling_dict.get((edge[1], edge[0])) for
                          edge in sorted_coupling_edges]
    coupling_symbols = _str_list_generator(coupling_symbol + '{0}{1}', *zip(*sorted_coupling_edges))
    cr_idx_dict = coupling_graph.two_way_graph_dict

    hamiltonian_dict = _duffing_hamiltonian_dict(oscillators=oscillators,
                                                 oscillator_dims=oscillator_dims,
                                                 oscillator_freqs=oscillator_freqs,
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

    # construct and return the PulseSystemModel
    return PulseSystemModel(hamiltonian=hamiltonian_model,
                            u_channel_lo=u_channel_lo,
                            control_channel_labels=coupling_graph.sorted_two_way_graph,
                            subsystem_list=oscillators,
                            dt=dt)


# Helper functions for creating pieces necessary to construct oscillator system models


def _duffing_hamiltonian_dict(oscillators,
                              oscillator_dims,
                              oscillator_freqs,
                              freq_symbols,
                              anharm_freqs,
                              anharm_symbols,
                              drive_strengths,
                              drive_symbols,
                              ordered_coupling_edges,
                              coupling_strengths,
                              coupling_symbols,
                              cr_idx_dict):
    """Creates a hamiltonian string dict for a duffing oscillator model

    Note, this function makes the following assumptions:
        - oscillators, oscillator_dims, oscillator_freqs, freq_symbols, anharm_freqs,
          anharm_symbols, drive_strengths, and drive_symbols are all lists of the same length
          (i.e. the total oscillator number)
        - ordered_coupling_edges, coupling_strengths, and coupling_symbols are lists of the same
          length

    Args:
        oscillators (list): ints for oscillator labels
        oscillator_dims (list): ints for oscillator dimensions
        oscillator_freqs (list): oscillator frequencies
        freq_symbols (list): symbols to be used for oscillator frequencies
        anharm_freqs (list): anharmonicity values
        anharm_symbols (list): symbols to be used for anharmonicity terms
        drive_strengths (list): drive strength coefficients
        drive_symbols (list): symbols for drive coefficients
        ordered_coupling_edges (list): tuples of two ints specifying oscillator couplings. Order
                                       corresponds to order of coupling_strengths and
                                       coupling_symbols
        coupling_strengths (list): strength of each coupling term (corresponds to ordering of
                                   ordered_coupling_edges)
        coupling_symbols (list): symbols for coupling coefficients
        cr_idx_dict (dict): A dict with keys given by tuples containing two ints, and value an int,
                            representing cross resonance drive channels. E.g. an entry {(0,1) : 1}
                            specifies a CR drive on oscillator 0 with oscillator 1 as target, with
                            u_channel index 1.

    Returns:
        dict: hamiltonian string format
    """

    # single oscillator terms
    hamiltonian_str = _single_duffing_drift_terms(freq_symbols, anharm_symbols, oscillators)
    hamiltonian_str += _drive_terms(drive_symbols, oscillators)

    # exchange terms
    if len(ordered_coupling_edges) > 0:
        hamiltonian_str += _exchange_coupling_terms(coupling_symbols, ordered_coupling_edges)

    # cr terms
    if len(cr_idx_dict) > 0:
        driven_system_indices = [key[0] for key in cr_idx_dict.keys()]
        cr_drive_symbols = [drive_symbols[idx] for idx in driven_system_indices]
        cr_channel_idx = cr_idx_dict.values()
        hamiltonian_str += _cr_terms(cr_drive_symbols, driven_system_indices, cr_channel_idx)

    # construct vars dictionary
    var_dict = {}
    for idx in oscillators:
        var_dict[freq_symbols[idx]] = oscillator_freqs[idx]
        var_dict[anharm_symbols[idx]] = anharm_freqs[idx]
        var_dict[drive_symbols[idx]] = drive_strengths[idx]

    if len(coupling_symbols) > 0:
        for symbol, strength in zip(coupling_symbols, coupling_strengths):
            var_dict[symbol] = strength

    dim_dict = {str(oscillator): dim for oscillator, dim in zip(oscillators, oscillator_dims)}

    return {'h_str': hamiltonian_str, 'vars': var_dict, 'qub': dim_dict}


def _cr_lo_list(cr_idx_dict):
    """Generates u_channel_lo list for a PulseSystemModel from a cr_idx_dict.

    Args:
        cr_idx_dict (dict): A dictionary with keys given by tuples of ints with int values. A key,
                            e.g. (0,1), signifies CR drive on system 0 with target 1, and the
                            value is the u channel index corresponding to that drive.
                            Note: this function assumes that
                            cr_idx_dict.values() == range(len(cr_idx_dict)).

    Returns:
        list: u_channel_lo format required by the simulator
    """

    # populate list of u channel lo for cr gates
    lo_list = [0] * len(cr_idx_dict)
    for system_pair, u_idx in cr_idx_dict.items():
        lo_list[u_idx] = [{'scale': [1.0, 0.0], 'q': system_pair[1]}]

    return lo_list


# Functions for creating Hamiltonian strings for various types of terms


def _single_duffing_drift_terms(freq_symbols, anharm_symbols, system_list):
    """Harmonic and anharmonic drift terms

    Args:
        freq_symbols (list): coefficients for harmonic part
        anharm_symbols (list): coefficients for anharmonic part
        system_list (list): list of system indices
    Returns:
        list: drift term strings
    """

    harm_terms = _str_list_generator('np.pi*(2*{0}-{1})*O{2}',
                                     freq_symbols,
                                     anharm_symbols,
                                     system_list)
    anharm_terms = _str_list_generator('np.pi*{0}*O{1}*O{1}',
                                       anharm_symbols,
                                       system_list)

    return harm_terms + anharm_terms


def _drive_terms(drive_symbols, system_list):
    """Drive terms for single oscillator

    Args:
        drive_symbols (list): coefficients of drive terms
        system_list (list): list of system indices
    Returns:
        list: drive term strings
    """

    return _str_list_generator('2*np.pi*{0}*X{1}||D{1}',
                               drive_symbols,
                               system_list)


def _exchange_coupling_terms(coupling_symbols, ordered_edges):
    """Exchange coupling terms between systems

    Args:
        coupling_symbols (list): coefficients of exchange couplings
        ordered_edges (list): list tuples of system indices for the couplings
    Returns:
        list: exchange coupling strings
    """

    idx1_list, idx2_list = zip(*list(ordered_edges))

    return _str_list_generator('2*np.pi*{0}*(Sp{1}*Sm{2}+Sm{1}*Sp{2})',
                               coupling_symbols,
                               idx1_list,
                               idx2_list)


def _cr_terms(drive_symbols, driven_system_indices, u_channel_indices):
    """Cross resonance drive terms

    Args:
        drive_symbols (list): coefficients for drive terms
        driven_system_indices (list): list of indices for systems that drive is applied to
        u_channel_indices (list): indicies for the u_channels corresponding to each term
    Returns:
        list: cr term strings
    """

    return _str_list_generator('2*np.pi*{0}*X{1}||U{2}',
                               drive_symbols,
                               driven_system_indices,
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


class CouplingGraph:
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
        """returns CouplingGraph object

        Args:
            edges (Iterable): An iterable of iterables, where the inner interables are assumed to
                              contain two elements, e.g. [(0,1), (2,3)], or ((0,1), (2,3))

        Returns:
            CouplingGraph: coupling graph specified by edges
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
