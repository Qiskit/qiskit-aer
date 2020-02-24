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
# pylint: disable=invalid-name, no-name-in-module, import-error

"""Operators to use in simulator"""

import numpy as np
from ..qutip_lite import operators as ops
from ..qutip_lite import states as st
from ..qutip_lite import tensor as ten
from ..qutip_lite.qobj import Qobj


def sigmax(dim=2):
    """Qiskit wrapper of sigma-X operator.
    """
    if dim == 2:
        return ops.sigmax()
    else:
        raise Exception('Invalid level specification of the qubit subspace')


def sigmay(dim=2):
    """Qiskit wrapper of sigma-Y operator.
    """
    if dim == 2:
        return ops.sigmay()
    else:
        raise Exception('Invalid level specification of the qubit subspace')


def sigmaz(dim=2):
    """Qiskit wrapper of sigma-Z operator.
    """
    if dim == 2:
        return ops.sigmaz()
    else:
        raise Exception('Invalid level specification of the qubit subspace')


def sigmap(dim=2):
    """Qiskit wrapper of sigma-plus operator.
    """
    return ops.create(dim)


def sigmam(dim=2):
    """Qiskit wrapper of sigma-minus operator.
    """
    return ops.destroy(dim)


def create(dim):
    """Qiskit wrapper of creation operator.
    """
    return ops.create(dim)


def destroy(dim):
    """Qiskit wrapper of annihilation operator.
    """
    return ops.destroy(dim)


def num(dim):
    """Qiskit wrapper of number operator.
    """
    return ops.num(dim)


def qeye(dim):
    """Qiskit wrapper of identity operator.
    """
    return ops.qeye(dim)


def project(dim, states):
    """Qiskit wrapper of projection operator.
    """
    ket, bra = states
    if ket in range(dim) and bra in range(dim):
        return st.basis(dim, ket) * st.basis(dim, bra).dag()
    else:
        raise Exception('States are specified on the outside of Hilbert space %s' % states)


def tensor(list_qobj):
    """ Qiskit wrapper of tensor product
    """
    return ten.tensor(list_qobj)



def basis(level, pos):
    """ Qiskit wrapper of basis
    """
    return st.basis(level, pos)


def state(state_vec):
    """ Qiskit wrapper of qobj
    """
    return Qobj(state_vec)


def fock_dm(level, eigv):
    """ Qiskit wrapper of fock_dm
    """
    return st.fock_dm(level, eigv)


def get_oper(name, *args):
    """ Return quantum operator of given name
    """
    return __operdict.get(name, qeye)(*args)


__operdict = {'X': sigmax, 'Y': sigmay, 'Z': sigmaz,
              'Sp': create, 'Sm': destroy, 'I': qeye,
              'O': num, 'P': project, 'A': destroy,
              'C': create, 'N': num}
