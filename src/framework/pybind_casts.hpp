/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2021.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_pybind_casts_hpp_
#define _aer_framework_pybind_casts_hpp_

#include "../simulators/stabilizer/clifford.hpp"

namespace py = pybind11;

namespace pybind11 {
namespace detail {
template <typename T> struct type_caster<matrix<T>>{
  using base = type_caster_base<matrix<T>>;
public:
  PYBIND11_TYPE_CASTER(matrix<T>, _("matrix_t"));
  // Conversion part 1 (Python->C++):
  bool load(py::handle src, bool convert){
      // TODO: Check if make sense have to flavors of matrix: F-style and C-style
      auto py_matrix = py::cast<py::array_t<T>>(src);
      auto c_order = py_matrix.attr("flags").attr("carray").template cast<bool>();
      if(py_matrix.ndim() != 2){
          throw std::invalid_argument(std::string("Python: invalid matrix (empty array)."));
      }
      size_t nrows = py_matrix.shape(0);
      size_t ncols = py_matrix.shape(1);
      // Matrix looks ok, now we parse it
      auto raw_mat = py_matrix.template unchecked<2>();
      if(c_order){
          value = matrix<T>(nrows, ncols, false);
          for (size_t r = 0; r < nrows; r++) {
              for (size_t c = 0; c < ncols; c++) {
                  value(r, c) = raw_mat(r, c);
              }
          }
      } else {
          value = matrix<T>::copy_from_buffer(nrows, ncols, static_cast<T *>(py_matrix.request().ptr));
      }
      return true;
  }
  // Conversion part 2 (C++ -> Python):
  static py::handle cast(matrix<T>, py::return_value_policy policy, py::handle parent){
      throw std::runtime_error("Casting from matrix to python not supported.");
  }
};

template <> struct type_caster<Clifford::Clifford>{
    using base = type_caster_base<Clifford::Clifford>;
public:
    PYBIND11_TYPE_CASTER(Clifford::Clifford, _("clifford"));
    // Conversion part 1 (Python->C++):
    bool load(py::handle src, bool convert){
        Clifford::build_from(src, value);
        return true;
    }
    // Conversion part 2 (C++ -> Python):
    static py::handle cast(Clifford::Clifford, py::return_value_policy policy, py::handle parent){
        throw std::runtime_error("Casting from Clifford to python not supported.");
    }
};
}
}

#endif // _aer_framework_pybind_casts_hpp_