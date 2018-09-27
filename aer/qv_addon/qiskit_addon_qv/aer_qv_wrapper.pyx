"""
Cython interface to C++ quantum circuit simulator.
"""

# Import C++ Classes
from libcpp.string cimport string


# QubitVector State class
cdef extern from "simulators/qubitvector/qv_state.hpp" namespace "AER::QubitVector":
    cdef cppclass QubitVector:
        QubitVector() except +
    cdef cppclass State[QubitVector]:
        State() except +


# Import C++ simulator Interface class
cdef extern from "framework/interface.hpp" namespace "AER":
    cdef cppclass Interface:
        Interface() except+
        string execute[STATE, STATECLASS](string &qobj) except +

        void load_noise_model(string &qobj) except +
        void load_engine_config(string &qobj) except +
        void load_state_config(string &qobj) except +

        void clear_noise_model()
        void clear_engine_config()
        void clear_state_config()

        void set_max_threads(int threads)
        void set_max_threads_circuit(int threads)
        void set_max_threads_shot(int threads)
        void set_max_threads_state(int threads)

        int get_max_threads()
        int get_max_threads_circuit()
        int get_max_threads_shot()
        int get_max_threads_state()


cdef class AerQvSimulatorWrapper:

    cdef Interface iface

    def __reduce__(self):
        return (self.__class__,())

    def execute(self, qobj):
        # Convert input to C++ string
        cdef string qobj_enc = str(qobj).encode('UTF-8')
        return self.iface.execute[QubitVector, State[QubitVector]](qobj_enc)
        # Execute

    def load_noise_model(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.iface.load_noise_model(config_enc)

    def load_state_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.iface.load_state_config(config_enc)

    def load_engine_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.iface.load_engine_config(config_enc)

    def clear_noise_model(self):
        self.iface.clear_noise_model()

    def clear_state_config(self):
        self.iface.clear_state_config()

    def clear_engine_config(self):
        self.iface.clear_engine_config()

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
