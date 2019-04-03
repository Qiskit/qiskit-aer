# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
# pylint: disable=no-value-for-parameter

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.

from scipy.integrate._ode import zvode


class qiskit_zvode(zvode):
    """Modifies the stepper for ZVODE so that
    it always stops at a given time in tlist.

    By default, it over shoots the time, which
    of course is just stupid.
    """
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r
