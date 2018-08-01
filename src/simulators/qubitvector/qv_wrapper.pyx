"""
Cython interface to C++ quantum circuit simulator.
"""

# Import C++ Classes
from libcpp.string cimport string

# Import C++ simulator Interface class
cdef extern from "framework/interface.hpp" namespace "AER":
    cdef cppclass Interface[CONTROLLER]:
        Interface() except+
        string execute(string &qobj) except +
        void load_engine_config(string &qobj) except +
        void load_state_config(string &qobj) except +
        void set_num_threads(int threads)
        int get_num_threads()

# Import C++ simulator Interface class
cdef extern from "base/controller.hpp" namespace "AER::Base":
    cdef cppclass Controller[STATE, ENGINE]:
        Controller() except +

# QubitVector State class
cdef extern from "simulators/qubitvector/qubitvector.hpp" namespace "QV":
    cdef cppclass QubitVector:
        State() except +

# QubitVector State class
cdef extern from "simulators/qubitvector/qv_state.hpp" namespace "AER::QubitVector":
    cdef cppclass State:
        State() except +

# QubitVector QasmEngine Class
cdef extern from "simulators/qubitvector/qv_qasm_engine.hpp" namespace "AER::QubitVector":
    cdef cppclass QasmEngine:
        QasmEngine() except +

# QubitVector FinalStateEngine class
cdef extern from "engines/finalstate_engine.hpp" namespace "AER::Engines":
    cdef cppclass FinalStateEngine[STATE]:
        FinalStateEngine() except +

# QubitVector FinalStateEngine class
cdef extern from "engines/observables_engine.hpp" namespace "AER::Engines":
    cdef cppclass ObservablesEngine[STATE]:
        ObservablesEngine() except +


cdef class QasmSimulatorWrapper:

    cdef Interface[Controller[QasmEngine, State]] *thisptr

    def __cinit__(self):
        self.thisptr = new Interface[Controller[QasmEngine, State]]()

    def __dealloc__(self):
        del self.thisptr

    def load_state_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.thisptr.load_state_config(config_enc)

    def load_engine_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.thisptr.load_engine_config(config_enc)

    def set_num_threads(self, threads):
        self.thisptr.set_num_threads(int(threads))

    def execute(self, qobj):
        # Convert input to C++ string
        cdef string qobj_enc = str(qobj).encode('UTF-8')
        # Execute
        return self.thisptr.execute(qobj_enc)


cdef class ObservablesSimulatorWrapper:

    cdef Interface[Controller[ObservablesEngine[QubitVector], State]] *thisptr

    def __cinit__(self):
        self.thisptr = new Interface[Controller[ObservablesEngine[QubitVector], State]]()

    def __dealloc__(self):
        del self.thisptr

    def load_state_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.thisptr.load_state_config(config_enc)

    def load_engine_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.thisptr.load_engine_config(config_enc)

    def set_num_threads(self, threads):
        self.thisptr.set_num_threads(int(threads))

    def execute(self, qobj):
        # Convert input to C++ string
        cdef string qobj_enc = str(qobj).encode('UTF-8')
        # Execute
        return self.thisptr.execute(qobj_enc)


cdef class StatevectorSimulatorWrapper:

    cdef Interface[Controller[FinalStateEngine[QubitVector], State]] *thisptr

    def __cinit__(self):
        self.thisptr = new Interface[Controller[FinalStateEngine[QubitVector], State]]()
        cdef string default_config = '{"finalstate_label": "statevector"}'.encode('UTF-8')
        self.thisptr.load_engine_config(default_config)

    def __dealloc__(self):
        del self.thisptr

    def load_state_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.thisptr.load_state_config(config_enc)

    def load_engine_config(self, config):
        # Convert input to C++ string
        cdef string config_enc = str(config).encode('UTF-8')
        self.thisptr.load_engine_config(config_enc)

    def set_num_threads(self, threads):
        self.thisptr.set_num_threads(int(threads))

    def execute(self, qobj):
        # Convert input to C++ string
        cdef string qobj_enc = str(qobj).encode('UTF-8')
        # Execute
        return self.thisptr.execute(qobj_enc)
