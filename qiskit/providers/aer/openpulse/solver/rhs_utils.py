# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
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
    global _cy_rhs_func
    code = compile('from ' + op_system.global_data['rhs_file_name'] +
                   ' import cy_td_ode_rhs', '<string>', 'exec')
    exec(code, globals())
    # pylint: disable=undefined-variable
    op_system.global_data['rhs_func'] = cy_td_ode_rhs
