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
# pylint: disable=invalid-name

"The OpenPulse simulator system class"


class OPSystem():
    """ A Class that holds all the information
    needed to simulate a given PULSE qobj
    """
    def __init__(self):
        # The system Hamiltonian in numerical format
        self.system = None
        # The noise (if any) in numerical format
        self.noise = None
        # System variables
        self.vars = None
        # The initial state of the system
        self.initial_state = None
        # Channels in the Hamiltonian string
        # these tell the order in which the channels
        # are evaluated in the RHS solver.
        self.channels = None
        # options of the ODE solver
        self.ode_options = None
        # time between pulse sample points.
        self.dt = None
        # Array containing all pulse samples
        self.pulse_array = None
        # Array of indices indicating where a pulse starts in the self.pulse_array
        self.pulse_indices = None
        # A dict that translates pulse names to integers for use in self.pulse_indices
        self.pulse_to_int = None
        # Holds the parsed experiments
        self.experiments = []
        # Can experiments be simulated once then sampled
        self.can_sample = True
        # holds global data
        self.global_data = {}
        # holds frequencies for the channels
        self.freqs = {}
        # diagonal elements of the hamiltonian
        self.h_diag = None
        # eigenvalues of the time-independent hamiltonian
        self.evals = None
        # Use C++ version of the function to pass to the ODE solver or the Cython one
        self.use_cpp_ode_func = False
        # eigenstates of the time-independent hamiltonian
        self.estates = None
