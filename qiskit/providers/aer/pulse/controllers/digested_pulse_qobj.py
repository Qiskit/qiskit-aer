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
# pylint: disable=invalid-name, missing-return-type-doc

"""A temporary placeholder object for storing what is extracted from a PulseQobj.

This class should eventually disolve into the more general structure of the
simulator package
"""

class DigestedPulseQobj:

    def __init__(self):

        # ####################################
        # Some "Simulation description"
        # ####################################

        # stuff related to memory/measurements
        self.shots = None
        self.meas_level = None
        self.meas_return = None
        self.memory_slots = None
        self.memory = None
        self.n_registers = None

        # ####################################
        # Some Signal portion
        # ####################################

        # I think these should ultimately construct and return "Signal" objects

        # specific data struct being used
        self.pulse_array = None
        self.pulse_indices = None
        self.pulse_dict = None

        self.qubit_lo_freq = None


        # #############################################
        # Mix of both signal and simulation description
        # #############################################

        # These should be turned into an internal "simulation events"
        # structure

        # "experiments" contains a combination of signal information and
        # other experiment descriptions, which should be separated
        self.experiments = None
