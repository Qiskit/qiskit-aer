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
# pylint: disable=invalid-name

"""Module for creating quantum operators."""

from ...qutip_extra_lite import qobj_generators


def gen_oper(opname, index, h_osc, h_qub, states=None):
    """Generate quantum operators.

    Args:
        opname (str): Name of the operator to be returned.
        index (int): Index of operator.
        h_osc (dict): Dimension of oscillator subspace
        h_qub (dict): Dimension of qubit subspace
        states (tuple): State indices of projection operator.

    Returns:
        Qobj: quantum operator for target qubit.
    """

    opr_tmp = None

    # get number of levels in Hilbert space
    if opname in ['X', 'Y', 'Z', 'Sp', 'Sm', 'I', 'O', 'P']:
        is_qubit = True
        dim = h_qub.get(index, 2)

        if opname in ['X', 'Y', 'Z'] and dim > 2:
            if opname == 'X':
                opr_tmp = qobj_generators.get_oper('A', dim) + qobj_generators.get_oper('C', dim)
            elif opname == 'Y':
                opr_tmp = (-1j * qobj_generators.get_oper('A', dim) +
                           1j * qobj_generators.get_oper('C', dim))
            else:
                opr_tmp = (qobj_generators.get_oper('I', dim) -
                           2 * qobj_generators.get_oper('N', dim))

    else:
        is_qubit = False
        dim = h_osc.get(index, 5)

    if opname == 'P':
        opr_tmp = qobj_generators.get_oper(opname, dim, states)
    else:
        if opr_tmp is None:
            opr_tmp = qobj_generators.get_oper(opname, dim)

    # reverse sort by index
    rev_h_osc = sorted(h_osc.items(), key=lambda x: x[0])[::-1]
    rev_h_qub = sorted(h_qub.items(), key=lambda x: x[0])[::-1]

    # osc_n * … * osc_0 * qubit_n * … * qubit_0
    opers = []
    for ii, dd in rev_h_osc:
        if ii == index and not is_qubit:
            opers.append(opr_tmp)
        else:
            opers.append(qobj_generators.qeye(dd))
    for ii, dd in rev_h_qub:
        if ii == index and is_qubit:
            opers.append(opr_tmp)
        else:
            opers.append(qobj_generators.qeye(dd))

    return qobj_generators.tensor(opers)
