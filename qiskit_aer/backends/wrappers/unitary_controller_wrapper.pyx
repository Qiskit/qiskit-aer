# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Cython wrapper for Aer UnitaryController
"""

# Import C++ Classes
from libcpp.string cimport string


# QubitVector State class
cdef extern from "simulators/qubitunitary/unitary_controller.hpp" namespace "AER::Simulator":
    cdef cppclass UnitaryController:
        UnitaryController() except +
        string execute(string &qobj) except +

        void set_data_config(string &qobj) except +
        void set_state_config(string &qobj) except +

        void clear_data_config()
        void clear_state_config()

        void set_max_threads(int threads)
        void set_max_threads_circuit(int threads)
        void set_max_threads_shot(int threads)
        void set_max_threads_state(int threads)

        int get_max_threads()
        int get_max_threads_circuit()
        int get_max_threads_shot()
        int get_max_threads_state()


cdef class UnitaryControllerWrapper:

    cdef UnitaryController iface

    def __reduce__(self):
        return (self.__class__,())

    def execute(self, qobj):
        # Convert input to C++ string
        cdef string qobj_enc = str(qobj).encode('UTF-8')
        return self.iface.execute(qobj_enc)
        # Execute

    def set_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.iface.set_state_config(config_enc)
        self.iface.set_data_config(config_enc)

    def clear_config(self):
        self.iface.clear_state_config()
        self.iface.clear_data_config()

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
