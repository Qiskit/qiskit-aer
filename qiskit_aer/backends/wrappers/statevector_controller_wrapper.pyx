# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Cython wrapper for Aer C++ qubit vector simulator.
"""

# Import C++ Classes
from libcpp.string cimport string

# QubitVector State class
cdef extern from "simulators/qubitvector/statevector_controller.hpp" namespace "AER::Simulator":
    cdef cppclass StatevectorController:
        StatevectorController() except +
        string execute_string(string &qobj) except +

        void set_config_string(string &qobj) except +
        void clear_config()

        void set_max_threads(int threads)
        void set_max_threads_circuit(int threads)
        void set_max_threads_shot(int threads)
        void set_max_threads_state(int threads)

        int get_max_threads()
        int get_max_threads_circuit()
        int get_max_threads_shot()
        int get_max_threads_state()


cdef class StatevectorControllerWrapper:

    cdef StatevectorController iface

    def __reduce__(self):
        return (self.__class__,())

    def execute(self, qobj, options, noise_model):
        # Note: noise model is not used for this controller
        # Convert input to C++ string
        cdef string qobj_enc = str(qobj).encode('UTF-8')
        cdef string options_enc = str(options).encode('UTF-8')
        # Load options
        self.iface.set_config_string(options_enc)
        # Execute simulation
        cdef string output = self.iface.execute_string(qobj_enc)
        # Clear options
        self.iface.clear_config()
        # Return output
        return output

    def set_max_threads(self, threads):
        self.iface.set_max_threads(int(threads))

    def set_max_threads_circuit(self, threads):
        self.iface.set_max_threads_circuit(int(threads))

    def set_max_threads_shot(self, threads):
        self.iface.set_max_threads_shot(int(threads))

    def set_max_threads_state(self, threads):
        self.iface.set_max_threads_state(int(threads))

    def get_max_threads(self):
        return self.iface.get_max_threads()

    def get_max_threads_circuit(self):
        return self.iface.get_max_threads_circuit()

    def get_max_threads_shot(self):
        return self.iface.get_max_threads_shot()

    def get_max_threads_state(self):
        return self.iface.get_max_threads_state()
