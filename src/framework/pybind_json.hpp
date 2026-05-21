/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_pybind_json_hpp_
#define _aer_framework_pybind_json_hpp_

#if defined(_MSC_VER) && _MSC_VER < 1500 // VC++ 8.0 and below
#define snprintf _snprintf
#else
#undef snprintf
#endif

#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#pragma GCC diagnostic ignored "-Wfloat-equal"

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
DISABLE_WARNING_POP
#pragma GCC diagnostic warning "-Wfloat-equal"

#include "framework/json.hpp"

namespace py = pybind11;
namespace nl = nlohmann;
using namespace pybind11::literals;
using json_t = nlohmann::json;

//------------------------------------------------------------------------------
// Nlohman JSON <--> Python Conversion
//------------------------------------------------------------------------------

namespace JSON {

/**
 * Convert a python object to a json.
 * @param js a json_t object to contain converted type.
 * @param obj is a python object to convert.
 */
void py_to_json(json_t &js, const py::handle &obj);

/**
 * Create a python object from a json.
 * @param js a json_t object.
 * @param o is a reference to a python object to populate.
 */
void py_from_json(const json_t &js, py::object &o);

} // end namespace JSON.

namespace nlohmann {

/**
 * adl_serializer specialization for pybind11::handle.
 *
 * Provides version-stable Python -> JSON conversion through nlohmann's
 * documented extension point. Avoids relying on free-function ADL of
 * to_json, which became non-discoverable starting with nlohmann_json 3.11.
 */
template <>
struct adl_serializer<pybind11::handle> {
  static void to_json(nlohmann::json &js, const pybind11::handle &obj) {
    JSON::py_to_json(js, obj);
  }
};

/**
 * adl_serializer specialization for pybind11::object.
 *
 * Delegates to_json to the handle version (object derives from handle), and
 * provides JSON -> Python conversion.
 */
template <>
struct adl_serializer<pybind11::object> {
  static void to_json(nlohmann::json &js, const pybind11::object &obj) {
    JSON::py_to_json(js, obj);
  }
  static void from_json(const nlohmann::json &js, pybind11::object &o) {
    JSON::py_from_json(js, o);
  }
};

} // end namespace nlohmann.
//------------------------------------------------------------------------------
// Python->JSON Conversion
//------------------------------------------------------------------------------

namespace JSON {

/**
 * Convert a numpy array to a json object
 * @param arr is a numpy array
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a 1-d numpy array to a json object
 * @param arr is a numpy array
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json_1d(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a 2-d numpy array to a json object
 * @param arr is a numpy array
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json_2d(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a 3-d numpy array to a json object
 * @param arr is a numpy array
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json_3d(py::array_t<T, py::array::c_style> arr);

json_t iterable_to_json_list(const py::handle &obj);

} // end namespace JSON

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Conversion
//------------------------------------------------------------------------------

template <typename T>
json_t JSON::numpy_to_json_1d(py::array_t<T, py::array::c_style> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dims must be 1");
  }

  T *ptr = (T *)buf.ptr;
  size_t D0 = buf.shape[0];

  std::vector<T> tbr; // to be returned
  for (size_t n0 = 0; n0 < D0; n0++)
    tbr.push_back(ptr[n0]);

  return std::move(tbr);
}

template <typename T>
json_t JSON::numpy_to_json_2d(py::array_t<T, py::array::c_style> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Number of dims must be 2");
  }

  T *ptr = (T *)buf.ptr;
  size_t D0 = buf.shape[0];
  size_t D1 = buf.shape[1];

  std::vector<std::vector<T>> tbr; // to be returned
  for (size_t n0 = 0; n0 < D0; n0++) {
    std::vector<T> tbr1;
    for (size_t n1 = 0; n1 < D1; n1++) {
      tbr1.push_back(ptr[n1 + D1 * n0]);
    }
    tbr.push_back(tbr1);
  }

  return std::move(tbr);
}

template <typename T>
json_t JSON::numpy_to_json_3d(py::array_t<T, py::array::c_style> arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 3) {
    throw std::runtime_error("Number of dims must be 3");
  }
  T *ptr = (T *)buf.ptr;
  size_t D0 = buf.shape[0];
  size_t D1 = buf.shape[1];
  size_t D2 = buf.shape[2];

  // to be returned
  std::vector<std::vector<std::vector<T>>> tbr;
  for (size_t n0 = 0; n0 < D0; n0++) {
    std::vector<std::vector<T>> tbr1;
    for (size_t n1 = 0; n1 < D1; n1++) {
      std::vector<T> tbr2;
      for (size_t n2 = 0; n2 < D2; n2++) {
        tbr2.push_back(ptr[n2 + D2 * (n1 + D1 * n0)]);
      }
      tbr1.push_back(tbr2);
    }
    tbr.push_back(tbr1);
  }

  return std::move(tbr);
}

template <typename T>
json_t JSON::numpy_to_json(py::array_t<T, py::array::c_style> arr) {
  py::buffer_info buf = arr.request();

  if (buf.ndim == 1) {
    return JSON::numpy_to_json_1d(arr);
  } else if (buf.ndim == 2) {
    return JSON::numpy_to_json_2d(arr);
  } else if (buf.ndim == 3) {
    return JSON::numpy_to_json_3d(arr);
  } else {
    throw std::runtime_error("Invalid number of dimensions!");
  }
  json_t tbr;
  return tbr;
}

json_t JSON::iterable_to_json_list(const py::handle &obj) {
  json_t js = nl::json::array();
  for (py::handle value : obj) {
    js.push_back(value);
  }
  return js;
}

void JSON::py_to_json(json_t &js, const py::handle &obj) {
  static py::object PyNoiseModel =
      py::module::import("qiskit_aer.noise.noise_model").attr("NoiseModel");
  static py::object PyCircuitHeader =
      py::module::import("qiskit_aer.backends.backend_utils")
          .attr("CircuitHeader");
  if (py::isinstance<py::float_>(obj)) {
    js = obj.cast<nl::json::number_float_t>();
  } else if (py::isinstance<py::bool_>(obj)) {
    js = obj.cast<nl::json::boolean_t>();
  } else if (py::isinstance<py::int_>(obj)) {
    js = obj.cast<nl::json::number_integer_t>();
  } else if (py::isinstance<py::str>(obj)) {
    js = obj.cast<nl::json::string_t>();
  } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    js = JSON::iterable_to_json_list(obj);
  } else if (py::isinstance<py::dict>(obj)) {
    for (auto item : py::cast<py::dict>(obj)) {
      json_t tmp;
      JSON::py_to_json(tmp, item.second);
      js[item.first.cast<nl::json::string_t>()] = std::move(tmp);
    }
  } else if (py::isinstance<py::array_t<double>>(obj)) {
    js = JSON::numpy_to_json(
        obj.cast<py::array_t<double, py::array::c_style>>());
  } else if (py::isinstance<py::array_t<std::complex<double>>>(obj)) {
    js = JSON::numpy_to_json(
        obj.cast<py::array_t<std::complex<double>, py::array::c_style>>());
  } else if (obj.is_none()) {
    return;
  } else if (py::isinstance(obj, PyNoiseModel)) {
    JSON::py_to_json(js, obj.attr("to_dict")());
  } else if (py::isinstance(obj, PyCircuitHeader)) {
    JSON::py_to_json(js, obj.attr("to_dict")());
  } else {
    auto type_str = std::string(py::str(obj.get_type()));
    if (type_str == "<class \'complex\'>" ||
        type_str == "<class \'numpy.complex64\'>" ||
        type_str == "<class \'numpy.complex128\'>" ||
        type_str == "<class \'numpy.complex_\'>") {
      auto tmp = obj.cast<std::complex<double>>();
      js.push_back(tmp.real());
      js.push_back(tmp.imag());
    } else if (type_str == "<class \'numpy.uint32\'>" ||
               type_str == "<class \'numpy.uint64\'>" ||
               type_str == "<class \'numpy.int32\'>" ||
               type_str == "<class \'numpy.int64\'>") {
      js = obj.cast<nl::json::number_integer_t>();
    } else if (type_str == "<class \'numpy.float32\'>" ||
               type_str == "<class \'numpy.float64\'>") {
      js = obj.cast<nl::json::number_float_t>();
    } else if (py::isinstance<py::iterable>(
                   obj)) { // last one to avoid intercepting numpy arrays, etc
      js = JSON::iterable_to_json_list(obj);
    } else {
      throw std::runtime_error(
          "to_json not implemented for this type of object: " +
          std::string(py::str(obj.get_type())));
    }
  }
}

void JSON::py_from_json(const json_t &js, py::object &o) {
  if (js.is_boolean()) {
    o = py::bool_(js.get<nl::json::boolean_t>());
  } else if (js.is_number()) {
    if (js.is_number_float()) {
      o = py::float_(js.get<nl::json::number_float_t>());
    } else if (js.is_number_unsigned()) {
      o = py::int_(js.get<nl::json::number_unsigned_t>());
    } else {
      o = py::int_(js.get<nl::json::number_integer_t>());
    }
  } else if (js.is_string()) {
    o = py::str(js.get<nl::json::string_t>());
  } else if (js.is_array()) {
    std::vector<py::object> obj(js.size());
    for (size_t i = 0; i < js.size(); i++) {
      py::object tmp;
      JSON::py_from_json(js[i], tmp);
      obj[i] = tmp;
    }
    o = py::cast(obj);
  } else if (js.is_object()) {
    py::dict obj;
    for (json_t::const_iterator it = js.cbegin(); it != js.cend(); ++it) {
      py::object tmp;
      JSON::py_from_json(it.value(), tmp);
      obj[py::str(it.key())] = tmp;
    }
    o = std::move(obj);
  } else if (js.is_null()) {
    o = py::none();
  } else {
    throw std::runtime_error("from_json not implemented for this json::type: " +
                             js.dump());
  }
}
//------------------------------------------------------------------------------

#endif
