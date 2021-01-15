/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_pybind_basics_hpp_
#define _aer_framework_pybind_basics_hpp_

#include <complex>
#include <vector>

#include "framework/linalg/vector.hpp"
#include "framework/matrix.hpp"

#include "framework/pybind_json.hpp"

namespace AerToPy {

//============================================================================
// Pybind11 move conversion of basic types
//============================================================================

// Move an arbitrary object to Python by calling Pybind11 cast with move
// Template specialization is used with this function for adding custom
// conversion for other types
// NOTE: Can this function be replaced by overload py::cast for custom types?
template <typename T> py::object to_python(T &&obj);

// Move a matrix to Python via conversion to Numpy array
template <typename T> py::object to_python(matrix<T> &&obj);

// Move a Vector to Python via conversion to Numpy array
template <typename T> py::object to_python(AER::Vector<T> &&obj);

// Move a Vector to Python via recusivly calling to_python on elements
template <typename T> py::object to_python(std::vector<T> &&obj);

// Move an Unordered string map to Python object by calling to_python on elements
template <typename T> py::object to_python(std::unordered_map<std::string, T> &&obj);

// Move an Unordered string map into an existing Python dict
template <typename T>
void add_to_python(py::dict &pydata, std::unordered_map<std::string, T> &&obj);


// Template specialization for moving numeric std::vectors to Numpy arrays
template <> py::object to_python(std::vector<AER::int_t> &&obj);
template <> py::object to_python(std::vector<AER::uint_t> &&obj);
template <> py::object to_python(std::vector<float> &&obj);
template <> py::object to_python(std::vector<double> &&obj);
template <> py::object to_python(std::vector<std::complex<double>> &&obj);
template <> py::object to_python(std::vector<std::complex<float>> &&obj);

// Template specialization for JSON
// NOTE: this copies rather than moves
template <> py::object to_python(json_t &&obj);

//------------------------------------------------------------------------------
// Convert To Numpy Arrays
//------------------------------------------------------------------------------

// Convert a matrix to a 2D Numpy array in Fortan order
template <typename T>
py::array_t<T, py::array::f_style> to_numpy(matrix<T> &&obj);

// Convert a Vector to a 1D Numpy array
template <typename T>
py::array_t<T> to_numpy(AER::Vector<T> &&obj);

// Convert a vector to a 1D Numpy array
template <typename T>
py::array_t<T> to_numpy(std::vector<T> &&obj);

//============================================================================
// Implementation
//============================================================================

//------------------------------------------------------------------------------
// Basic Types
//------------------------------------------------------------------------------

template <typename T>
py::object to_python(T &&obj) {
  return py::cast(obj, py::return_value_policy::move);
}

template <>
py::object to_python(json_t &&obj) {
  py::object pydata;
  from_json(obj, pydata);
  return pydata;
}

template <typename T>
py::object to_python(std::unordered_map<std::string, T> &&obj) {
  py::dict pydata;
  add_to_python(pydata, std::move(obj));
  return std::move(pydata);
}

template <typename T>
void add_to_python(py::dict &pydata, std::unordered_map<std::string, T> &&obj) {
  for(auto& elt : obj) {
    pydata[elt.first.data()] = to_python(std::move(elt.second));
  }
}

template <typename T>
py::object to_python(std::vector<T> &&obj) {
  py::list pydata;
  for(auto& elt : obj) {
    pydata.append(to_python(std::move(elt)));
  }
  return std::move(pydata);
}

template <typename T>
py::object to_python(matrix<T> &&obj) {
  return to_numpy(std::move(obj));
}

template <typename T>
py::object to_python(AER::Vector<T> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<AER::int_t> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<AER::uint_t> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<double> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<float> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<std::complex<double>> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<std::complex<float>> &&obj) {
  return to_numpy(std::move(obj));
}

//------------------------------------------------------------------------------
// Array Types
//------------------------------------------------------------------------------

template <typename T>
py::array_t<T, py::array::f_style> to_numpy(matrix<T> &&src) {
  std::array<py::ssize_t, 2> shape {static_cast<py::ssize_t>(src.GetRows()),
                                    static_cast<py::ssize_t>(src.GetColumns())};
  matrix<T>* src_ptr = new matrix<T>(std::move(src));
  auto capsule = py::capsule(src_ptr, [](void* p) { delete reinterpret_cast<matrix<T>*>(p); });
  return py::array_t<T, py::array::f_style>(shape, src_ptr->data(), capsule);
}

template <typename T>
py::array_t<T> to_numpy(AER::Vector<T> &&src) {
  AER::Vector<T>* src_ptr = new AER::Vector<T>(std::move(src));
  auto capsule = py::capsule(src_ptr, [](void* p) { delete reinterpret_cast<AER::Vector<T>*>(p); });
  return py::array_t<T>(
    src_ptr->size(),  // shape of array
    src_ptr->data(),  // c-style contiguous strides for vector
    capsule           // numpy array references this parent
  );
}


template <typename T>
py::array_t<T> to_numpy(std::vector<T> &&src) {
  std::vector<T>* src_ptr = new std::vector<T>(std::move(src));
  auto capsule = py::capsule(src_ptr, [](void* p) { delete reinterpret_cast<std::vector<T>*>(p); });
  return py::array_t<T>(
    src_ptr->size(),  // shape of array
    src_ptr->data(),  // c-style contiguous strides for vector
    capsule           // numpy array references this parent
  );
}

//------------------------------------------------------------------------------
}  // end namespace AerToPy
//------------------------------------------------------------------------------
#endif
