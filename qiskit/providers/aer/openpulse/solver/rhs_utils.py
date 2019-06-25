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

import os
from openpulse.solver.codegen import OPCodegen
import openpulse.solver.settings as op_set


def _op_generate_rhs(op_system):
    """ Generates the RHS Cython file for solving the sytem
    described in op_system

    Args:
        op_system (OPSystem): An OpenPulse system object.
    """
    name = "rhs" + str(os.getpid()) + str(op_set.CGEN_NUM)+'_op'
    op_system.global_data['rhs_file_name'] = name
    cgen = OPCodegen(op_system)
    cgen.generate(name + ".pyx")

def _op_func_load(op_system):
    """Loads the Cython function defined in the file
    `rhs_file_name.pyx` where `rhs_file_name` is
    stored in the op_system.

    Args:
        op_system (OPSystem): An OpenPulse system object.
    """
    code = compile('from ' + op_system.global_data['rhs_file_name'] +
                   ' import cy_td_ode_rhs', '<string>', 'exec')
    exec(code, globals())
    # pylint: disable=undefined-variable
    op_system.global_data['rhs_func'] = cy_td_ode_rhs
