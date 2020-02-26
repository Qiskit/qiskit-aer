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

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
# pylint: disable=invalid-name

"""OpenPulse runtime code generator"""

import os
import sys
from ..qutip_lite import cy
from . import settings

_cython_path = os.path.abspath(cy.__file__).replace('__init__.py', '')
_cython_path = _cython_path.replace("\\", "/")
_include_string = "'" + _cython_path + "complex_math.pxi'"


class OPCodegen():
    """
    Class for generating cython code files at runtime.
    """
    def __init__(self, op_system):

        sys.path.append(os.getcwd())

        # Hamiltonian time-depdendent pieces
        self.op_system = op_system
        self.dt = op_system.dt

        self.num_ham_terms = self.op_system.global_data['num_h_terms']

        # Code generator properties
        self._file = None
        self.code = []  # strings to be written to file
        self.level = 0  # indent level
        self.spline_count = 0

    def write(self, string):
        """write lines of code to self.code"""
        self.code.append("    " * self.level + string + "\n")

    def file(self, filename):
        """open file called filename for writing"""
        self._file = open(filename, "w")

    def generate(self, filename="rhs.pyx"):
        """generate the file"""

        for line in cython_preamble():
            self.write(line)

        # write function for Hamiltonian terms (there is always at least one
        # term)
        for line in cython_checks() + self.ODE_func_header():
            self.write(line)
        self.indent()
        for line in func_header(self.op_system):
            self.write(line)
        for line in self.channels():
            self.write(line)
        for line in self.func_vars():
            self.write(line)
        for line in self.func_end():
            self.write(line)
        self.dedent()

        self.file(filename)
        self._file.writelines(self.code)
        self._file.close()
        settings.CGEN_NUM += 1

    def indent(self):
        """increase indention level by one"""
        self.level += 1

    def dedent(self):
        """decrease indention level by one"""
        if self.level == 0:
            raise SyntaxError("Error in code generator")
        self.level -= 1

    def ODE_func_header(self):
        """Creates function header for time-dependent ODE RHS."""
        func_name = "def cy_td_ode_rhs("
        # strings for time and vector variables
        input_vars = ("\n        double t" +
                      ",\n        complex[::1] vec")

        # add diagonal hamiltonian terms
        input_vars += (",\n        double[::1] energ")

        for k in range(self.num_ham_terms):
            input_vars += (",\n        " +
                           "complex[::1] data%d, " % k +
                           "int[::1] idx%d, " % k +
                           "int[::1] ptr%d" % k)

        # Add global vaiables
        input_vars += (",\n        " + "complex[::1] pulse_array")
        input_vars += (",\n        " + "unsigned int[::1] pulse_indices")

        # Add per experiment variables
        for key in self.op_system.channels.keys():
            input_vars += (",\n        " + "double[::1] %s_pulses" % key)
            input_vars += (",\n        " + "double[::1] %s_fc" % key)

        # add Hamiltonian variables
        for key in self.op_system.vars.keys():
            input_vars += (",\n        " + "complex %s" % key)

        # add Freq variables
        for key in self.op_system.freqs.keys():
            input_vars += (",\n        " + "double %s_freq" % key)

        # register
        input_vars += (",\n        " + "unsigned char[::1] register")

        func_end = "):"
        return [func_name + input_vars + func_end]

    def channels(self):
        """Write out the channels
        """
        channel_lines = [""]

        channel_lines.append("# Compute complex channel values at time `t`")
        for chan, idx in self.op_system.channels.items():
            chan_str = "%s = chan_value(t, %s, %s_freq, " % (chan, idx, chan) + \
                       "%s_pulses,  pulse_array, pulse_indices, " % chan + \
                       "%s_fc, register)" % (chan)
            channel_lines.append(chan_str)
        channel_lines.append('')
        return channel_lines

    def func_vars(self):
        """Writes the variables and spmv parts"""
        func_vars = []
        sp1 = "    "
        sp2 = sp1 + sp1

        func_vars.append("# Eval the time-dependent terms and do SPMV.")
        for idx in range(len(self.op_system.system) + 1):

            if (idx == len(self.op_system.system) and
                    (len(self.op_system.system) < self.num_ham_terms)):
                # this is the noise term
                term = [1.0, 1.0]
            elif idx < len(self.op_system.system):
                term = self.op_system.system[idx]
            else:
                continue

            if isinstance(term, list) or term[1]:
                func_vars.append("td%s = %s" % (idx, term[1]))
            else:
                func_vars.append("td%s = 1.0" % (idx))

            func_vars.append("if abs(td%s) > 1e-15:" % idx)

            func_vars.append(sp1 + "for row in range(num_rows):")
            func_vars.append(sp2 + "dot = 0;")
            func_vars.append(sp2 + "row_start = ptr%d[row];" % idx)
            func_vars.append(sp2 + "row_end = ptr%d[row+1];" % idx)
            func_vars.append(sp2 + "for jj in range(row_start,row_end):")
            func_vars.append(sp1 +
                             sp2 +
                             "osc_term = exp(1j*(energ[row]-energ[idx%d[jj]])*t)" % idx)
            func_vars.append(sp1 + sp2 + "if row<idx%d[jj]:" % idx)
            func_vars.append(sp2 + sp2 + "coef = conj(td%d)" % idx)
            func_vars.append(sp1 + sp2 + "else:")
            func_vars.append(sp2 + sp2 + "coef = td%d" % idx)
            func_vars.append(sp1 + sp2 +
                             "dot += coef*osc_term*data%d[jj]*vec[idx%d[jj]];" % (idx, idx))
            func_vars.append(sp2 + "out[row] += dot;")

        # remove the diagonal terms
        func_vars.append("for row in range(num_rows):")
        func_vars.append(sp1 + "out[row] += 1j*energ[row]*vec[row];")

        return func_vars

    def func_end(self):
        """End of the RHS function.
        """
        end_str = [""]
        end_str.append("# Convert to NumPy array, grab ownership, and return.")
        end_str.append("cdef np.npy_intp dims = num_rows")

        temp_str = "cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = "
        temp_str += "np.PyArray_SimpleNewFromData(1, &dims, np.NPY_COMPLEX128, out)"
        end_str.append(temp_str)
        end_str.append("PyArray_ENABLEFLAGS(arr_out, np.NPY_OWNDATA)")
        end_str.append("return arr_out")
        return end_str


def func_header(op_system):
    """Header for the RHS function.
    """
    func_vars = ["", 'cdef size_t row', 'cdef unsigned int row_start, row_end',
                 'cdef unsigned int num_rows = vec.shape[0]',
                 'cdef double complex dot, osc_term, coef',
                 "cdef double complex * " +
                 'out = <complex *>PyDataMem_NEW(num_rows * sizeof(complex))',
                 'memset(&out[0],0,num_rows * sizeof(complex))'
                 ]
    func_vars.append("")

    for val in op_system.channels:
        func_vars.append("cdef double complex %s" % val)

    for kk in range(len(op_system.system) + 1):
        func_vars.append("cdef double complex td%s" % kk)

    return func_vars


def cython_preamble():
    """
    Returns list of code segments for Cython preamble.
    """
    preamble = ["""\
#!python
#cython: language_level=3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW(size_t size)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

from qiskit.providers.aer.pulse.qutip_lite.cy.spmatfuncs cimport spmvpy
from libc.math cimport pi

from qiskit.providers.aer.pulse.cy.channel_value cimport chan_value

from libc.string cimport memset

include """ + _include_string + """
"""]
    return preamble


def cython_checks():
    """
    List of strings that turn off Cython checks.
    """
    return ["""@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)"""]
