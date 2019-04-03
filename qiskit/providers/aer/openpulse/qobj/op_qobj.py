# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
# pylint: disable=invalid-name

import numpy as np
import qutip as qt

from qutip.cy.spmatfuncs import (spmv_csr, cy_expect_psi_csr)
from qutip.fastsparse import fast_csr_matrix


def sigmax(dim=2):
    """Qiskit wrapper of sigma-X operator.
    """
    if dim == 2:
        return qt.sigmax()
    else:
        raise Exception('Invalid level specification of the qubit subspace')


def sigmay(dim=2):
    """Qiskit wrapper of sigma-Y operator.
    """
    if dim == 2:
        return qt.sigmay()
    else:
        raise Exception('Invalid level specification of the qubit subspace')


def sigmaz(dim=2):
    """Qiskit wrapper of sigma-Z operator.
    """
    if dim == 2:
        return qt.sigmaz()
    else:
        raise Exception('Invalid level specification of the qubit subspace')


def sigmap(dim=2):
    """Qiskit wrapper of sigma-plus operator.
    """
    return qt.create(dim)


def sigmam(dim=2):
    """Qiskit wrapper of sigma-minus operator.
    """
    return qt.destroy(dim)


def create(dim):
    """Qiskit wrapper of creation operator.
    """
    return qt.create(dim)


def destroy(dim):
    """Qiskit wrapper of annihilation operator.
    """
    return qt.destroy(dim)


def num(dim):
    """Qiskit wrapper of number operator.
    """
    return qt.num(dim)


def qeye(dim):
    """Qiskit wrapper of identity operator.
    """
    return qt.qeye(dim)


def project(dim, states):
    """Qiskit wrapper of projection operator.
    """
    ket, bra = states
    if ket in range(dim) and bra in range(dim):
        return qt.basis(dim, ket) * qt.basis(dim, bra).dag()
    else:
        raise Exception('States are specified on the outside of Hilbert space %s', states)


def tensor(list_qobj):
    """ Qiskit wrapper of tensor product
    """
    return qt.tensor(list_qobj)


def conj(val):
    """ Qiskit wrapper of conjugate
    """
    if isinstance(val, qt.qobj.Qobj):
        return val.conj()
    else:
        return np.conj(val)


def sin(val):
    """ Qiskit wrapper of sine function
    """
    if isinstance(val, qt.qobj.Qobj):
        return val.sinm()
    else:
        return np.sin(val)


def cos(val):
    """ Qiskit wrapper of cosine function
    """
    if isinstance(val, qt.qobj.Qobj):
        return val.cosm()
    else:
        return np.cos(val)


def exp(val):
    """ Qiskit wrapper of exponential function
    """
    if isinstance(val, qt.qobj.Qobj):
        return val.expm()
    else:
        return np.exp(val)


def sqrt(val):
    """ Qiskit wrapper of square root
    """
    if isinstance(val, qt.qobj.Qobj):
        return val.sqrtm()
    else:
        return np.sqrt(val)


def dag(qobj):
    """ Qiskit wrapper of adjoint
    """
    return qobj.dag()


def dammy(qobj):
    """ Return given quantum object
    """
    return qobj


def basis(level, pos):
    """ Qiskit wrapper of basis
    """
    return qt.basis(level, pos)


def fock_dm(level, eigv):
    """ Qiskit wrapper of fock_dm
    """
    return qt.fock_dm(level, eigv)


def opr_prob(opr, state_vec):
    """ Measure probability of operator in given quantum state
    """
    return cy_expect_psi_csr(opr.data.data,
                             opr.data.indices,
                             opr.data.indptr,
                             state_vec, 1)


def opr_apply(opr, state_vec):
    """ Apply operator to given quantum state
    """
    return spmv_csr(opr.data.data,
                    opr.data.indices,
                    opr.data.indptr,
                    state_vec)


def get_oper(name, *args):
    """ Return quantum operator of given name
    """
    return __operdict.get(name, qeye)(*args)


def get_func(name, qobj):
    """ Apply function of given name
    """
    return __funcdict.get(name, dammy)(qobj)


__operdict = {'X': sigmax, 'Y': sigmay, 'Z': sigmaz,
              'Sp': sigmap, 'Sm': sigmam, 'I': qeye,
              'O': num, 'P': project, 'A': destroy,
              'C': create, 'N': num}

__funcdict = {'cos': cos, 'sin': sin, 'exp': exp,
              'sqrt': sqrt, 'conj': conj, 'dag': dag}
