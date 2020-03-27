#ifndef PULSE_UTILS_H
#define PULSE_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PyObject* expect_psi_csr(const py::array_t<std::complex<double>, py::array::c_style>& data,
                         const py::array_t<int, py::array::c_style>& ind,
                         const py::array_t<int, py::array::c_style>& ptr,
                         const py::array_t<std::complex<double>, py::array::c_style>& vec,
                         bool isherm);

py::array_t<double> occ_probabilities(py::array_t<int> qubits,
                                      py::array_t<std::complex<double>> state,
                                      py::list meas_ops);

void write_shots_memory(py::array_t<unsigned char> mem,
                        py::array_t<unsigned int> mem_slots,
                        py::array_t<double> probs,
                        py::array_t<double> rand_vals);

void oplist_to_array(py::list A, py::array_t<std::complex<double>> B, int start_idx);

py::array_t<double> spmv_csr(const py::array_t<std::complex<double>, py::array::c_style>& data,
                             const py::array_t<int, py::array::c_style>& ind,
                             const py::array_t<int, py::array::c_style>& ptr,
                             const py::array_t<std::complex<double>, py::array::c_style>& vec);

PYBIND11_MODULE(pulse_utils, m) {
    m.doc() = "Utility functions for pulse simulator"; // optional module docstring

    m.def("cy_expect_psi_csr", &expect_psi_csr, "Expected value for a operator");
    m.def("occ_probabilities", &occ_probabilities, "");
    m.def("write_shots_memory", &write_shots_memory, "");
    m.def("oplist_to_array", &oplist_to_array, "");
    m.def("spmv_csr", &spmv_csr, "");
}



#endif //PULSE_UTILS_H
