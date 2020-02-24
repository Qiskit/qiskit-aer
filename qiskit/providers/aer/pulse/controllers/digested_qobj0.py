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

Not 100% sure but I think this class should disolve into the more general structure of the
simulator package. I.e. the point of "pulse digest" should ideally be to converted into the
pulse-independent description of this simulation package.
"""

class DigestedPulseQobj:

    def __init__(self):

        # stuff related to memory/measurements
        self.shots = None
        self.meas_level = None
        self.meas_return = None
        self.memory_slots = None
        self.memory = None
        self.n_registers = None


        # signal stuff, to be put into pulse/signals
        self.pulse_array = None
        self.pulse_indices = None
        self.pulse_dict = None

        # maybe to be put into signal when we have a "mixedsignal"
        self.qubit_lo_freq = None

        # the digested experiments
        self.experiments = None
