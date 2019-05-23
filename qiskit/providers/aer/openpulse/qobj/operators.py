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
import scipy.linalg as la
from openpulse.qobj import op_qobj as op


def gen_oper(opname, index, h_osc, h_qub, states=None):
    """Generate quantum operators.

    Parameters
    ----------
    opname (str): Name of the operator to be returned.
    index (int): Index of operator.
    h_osc (dict): Dimension of oscillator subspace
    h_qub (dict): Dimension of qubit subspace
    states (tuple): State indices of projection operator.

    Returns
    -------
    out_oper (qutip.Qobj): quantum operator for target qubit.
    """

    # get number of levels in Hilbert space
    if opname in ['X', 'Y', 'Z', 'Sp', 'Sm', 'I', 'O', 'P']:
        is_qubit = True
        dim = h_qub.get(index, 2)
    else:
        is_qubit = False
        dim = h_osc.get(index, 5)

    if opname == 'P':
        opr_tmp = op.get_oper(opname, dim, states)
    else:
        opr_tmp = op.get_oper(opname, dim)

    # reverse sort by index
    rev_h_osc = sorted(h_osc.items(), key=lambda x: x[0])[::-1]
    rev_h_qub = sorted(h_qub.items(), key=lambda x: x[0])[::-1]

    # osc_n * … * osc_0 * qubit_n * … * qubit_0
    opers = []
    for ii, dd in rev_h_osc:
        if ii == index and not is_qubit:
            opers.append(opr_tmp)
        else:
            opers.append(op.qeye(dd))
    for ii, dd in rev_h_qub:
        if ii == index and is_qubit:
            opers.append(opr_tmp)
        else:
            opers.append(op.qeye(dd))

    return op.tensor(opers)


def qubit_occ_oper(target_qubit, h_osc, h_qub, level=1):
    """Builds the occupation number operator for a target qubit
    in a qubit oscillator system, where the oscillator are the first
    subsystems, and the qubit last.
    
    Parameters
    ----------
    target_qubit (int): Qubit for which operator is built.
    h_osc (dict): Dict of number of levels in each oscillator.
    h_qub (dict): Dict of number of levels in each qubit system.
    level (int): Level of qubit system to be measured.

    Returns
    -------
    out_oper (qutip.Qobj): Occupation number operator for target qubit.
    """
    # reverse sort by index
    rev_h_osc = sorted(h_osc.items(), key=lambda x: x[0])[::-1]
    rev_h_qub = sorted(h_qub.items(), key=lambda x: x[0])[::-1]

    # osc_n * … * osc_0 * qubit_n * … * qubit_0
    opers = []
    for ii, dd in rev_h_osc:
        opers.append(op.qeye(dd))
    for ii, dd in rev_h_qub:
        if ii == target_qubit:
            opers.append(op.fock_dm(h_qub.get(target_qubit, 2), level))
        else:
            opers.append(op.qeye(dd))

    return op.tensor(opers)


def measure_outcomes(measured_qubits, state_vector, measure_ops,
                     seed=None):
    """Generate measurement outcomes for a given set of qubits and state vector.
    
    Parameters:
        measured_qubits (array_like): Qubits to be measured.
        state_vector(ndarray): State vector.
        measure_ops (list): List of measurement operator
        seed (int): Optional seed to RandomState for reproducibility.
        
    Returns:
        outcomes (str): String of binaries representing measured qubit values.
    """
    outcome_len = max(measured_qubits)+1
    # Create random generator with given seed (if any).
    rng_gen = np.random.RandomState(seed)
    rnds = rng_gen.rand(outcome_len)
    outcomes = ''
    for kk in range(outcome_len):
        if kk in measured_qubits:
            excited_prob = op.opr_prob(measure_ops[kk], state_vector)
            if excited_prob >= rnds[kk]:
                outcomes += '1'
            else:
                outcomes += '0' 
        else:
            # We need a string for all the qubits up to the last
            # one being measured for the mask operation to work
            # Here, we default to unmeasured qubits in the grnd state.
            outcomes += '0'
    return outcomes


def apply_projector(measured_qubits, results, h_qub, h_osc, state_vector):
    """Builds and applies the projection operator associated
    with a given qubit measurement result onto a state vector.
    
    Parameters
    ----------
    measured_qubits (list): measured qubit indices.
    results (list): results of qubit measurements.
    h_qub (dict): Dict of number of levels in each qubit system.
    h_osc (dict): Dict of number of levels in each oscillator.
    state_vector (ndarray): State vector.

    Returns:
    ----------
    proj_state (qutip.Qobj): State vector after projector applied, and normalized.
    """

    # reverse sort by index
    rev_h_osc = sorted(h_osc.items(), key=lambda x: x[0])[::-1]
    rev_h_qub = sorted(h_qub.items(), key=lambda x: x[0])[::-1]

    # osc_n * … * osc_0 * qubit_n * … * qubit_0
    opers = []
    for ii, dd in rev_h_osc:
        opers.append(op.qeye(dd))
    for ii, dd in rev_h_qub:
        if ii in measured_qubits:
            opers.append(op.fock_dm(dd, results[ii]))
        else:
            opers.append(op.qeye(dd))

    proj_oper = op.tensor(opers)
    psi = op.opr_apply(proj_oper, state_vector)
    psi /= la.norm(psi)

    return psi


def init_fock_state(h_osc, h_qub, noise_dict={}):
    """ Generate initial Fock state, in the number state
    basis, for an oscillator in a thermal state defined
    by the expectation value of number of particles.

    Parameters:
        h_osc (dict): Dimension of oscillator subspace
        h_qub (dict): Dimension of qubit subspace
        noise_dict (dict): Dictionary of thermal particles for each oscillator subspace
    Returns:
        qutip.Qobj: State vector
    """
    # reverse sort by index
    rev_h_osc = sorted(h_osc.items(), key=lambda x: x[0])[::-1]
    rev_h_qub = sorted(h_qub.items(), key=lambda x: x[0])[::-1]

    # osc_n * … * osc_0 * qubit_n * … * qubit_0
    sub_state_vecs = []
    for ii, dd in rev_h_osc:
        n_thermal = noise_dict['oscillator']['n_th'].get(str(ii), 0)
        if n_thermal == 0:
            # no thermal particles
            idx = 0
        else:
            # consider finite thermal excitation
            levels = np.arange(dd)
            beta = np.log(1.0 / n_thermal + 1.0)
            diags = np.exp(-beta * levels)
            diags /= np.sum(diags)
            cum_sum = np.cumsum(diags)
            idx = np.where(np.random.random() < cum_sum)[0][0]
        sub_state_vecs.append(op.basis(dd, idx))
    for ii, dd in rev_h_qub:
        sub_state_vecs.append(op.basis(dd, 0))

    return op.tensor(sub_state_vecs)
