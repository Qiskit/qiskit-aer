#include <iostream>

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP
#if defined(_MSC_VER)
    #undef snprintf
#endif

#include "aer_controller_binding.hpp"
#include "aer_state_binding.hpp"
#include "aer_circuit_binding.hpp"

using namespace AER;

Operations::Op create_aer_operation(const py::handle &qasmQobjInst) {
  return Operations::input_to_op(qasmQobjInst);
}

PYBIND11_MODULE(controller_wrappers, m) {

#ifdef AER_MPI
  int prov;
  MPI_Init_thread(nullptr,nullptr,MPI_THREAD_MULTIPLE,&prov);
#endif
    m.def("create_aer_operation", &create_aer_operation);
    bind_aer_controller(m);
    bind_aer_state(m);
    bind_aer_circuit(m);
}
