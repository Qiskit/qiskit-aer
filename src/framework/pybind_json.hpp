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

#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <iostream>
#include <type_traits>

#include <nlohmann_json.hpp>
#include "framework/matrix.hpp"

namespace py = pybind11;
namespace nl = nlohmann;
using namespace pybind11::literals;
using json_t = nlohmann::json;

//============================================================================
// JSON Conversion for pybind types
//============================================================================

namespace std {

/**
 * Convert a python object to a json.
 * @param js a json_t object to contain converted type.
 * @param o is a python object to convert.
 */
void to_json(json_t &js, const py::handle &o);

/**
 * Create a python object from a json
 * @param js a json_t object 
 * @param o is a reference to an existing (empty) python object
 */
void from_json(const json_t &js, py::object &o);

} // end namespace std.

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

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Conversion
//------------------------------------------------------------------------------

template <typename T>
json_t numpy_to_json_1d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dims must be 1");
    }

    T *ptr = (T *) buf.ptr;
    size_t D0 = buf.shape[0];

    std::vector<T> tbr; // to be returned 
    for (size_t n0 = 0; n0 < D0; n0++)
        tbr.push_back(ptr[n0]);

    return std::move(tbr);
}

template <typename T>
json_t numpy_to_json_2d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dims must be 2");
    }

    T *ptr = (T *) buf.ptr;
    size_t D0 = buf.shape[0];
    size_t D1 = buf.shape[1];

    std::vector<std::vector<T > > tbr; // to be returned
    for (size_t n0 = 0; n0 < D0; n0++) {
        std::vector<T> tbr1;
        for (size_t n1 = 0; n1 < D1; n1++) {
            tbr1.push_back(ptr[n1 + D1*n0]);
        }
        tbr.push_back(tbr1);
    }

    return std::move(tbr);

}

template <typename T>
json_t numpy_to_json_3d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Number of dims must be 3");
    }
    T *ptr = (T *) buf.ptr;
    size_t D0 = buf.shape[0];
    size_t D1 = buf.shape[1];
    size_t D2 = buf.shape[2];

    // to be returned
    std::vector<std::vector<std::vector<T > > > tbr;
    for (size_t n0 = 0; n0 < D0; n0++) {
        std::vector<std::vector<T> > tbr1;
        for (size_t n1 = 0; n1 < D1; n1++) {
            std::vector<T> tbr2;
            for (size_t n2 = 0; n2 < D2; n2++) {
                tbr2.push_back(ptr[n2 + D2*(n1 + D1*n0)]);
            }
            tbr1.push_back(tbr2);
        }
        tbr.push_back(tbr1);
    }

    return std::move(tbr);

}

template <typename T>
json_t numpy_to_json(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    //std::cout << "buff dim: " << buf.ndim << std::endl;

    if (buf.ndim == 1) {
        return numpy_to_json_1d(arr);
    } else if (buf.ndim == 2) {
        return numpy_to_json_2d(arr);
    } else if (buf.ndim == 3) {
        return numpy_to_json_3d(arr);
    } else {
        throw std::runtime_error("Invalid number of dimensions!");
    }
    json_t tbr;
    return tbr;
}

void std::to_json(json_t &js, const py::handle &obj) {
    if (py::isinstance<py::bool_>(obj)) {
        js = obj.cast<nl::json::boolean_t>();
        //js = obj.cast<bool>();
    } else if (py::isinstance<py::int_>(obj)) {
        js = obj.cast<nl::json::number_integer_t>();
    } else if (py::isinstance<py::float_>(obj)) {
        js = obj.cast<nl::json::number_float_t>();
    } else if (py::isinstance<py::str>(obj)) {
        js = obj.cast<nl::json::string_t>();
    } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        js = nl::json::array();
        for (py::handle value: obj)
        {
            js.push_back(value);
        }
    } else if (py::isinstance<py::dict>(obj)) {
        for (auto item : py::cast<py::dict>(obj))
        {
            js[item.first.cast<nl::json::string_t>()] = item.second;
        }
    } else if (py::isinstance<py::array_t<double> >(obj)) {
        js = numpy_to_json(obj.cast<py::array_t<double, py::array::c_style> >());
    } else if (py::isinstance<py::array_t<std::complex<double> > >(obj)) {
        js = numpy_to_json(obj.cast<py::array_t<std::complex<double>, py::array::c_style> >());
    } else if (std::string(py::str(obj.get_type())) == "<class \'complex\'>") {
        auto tmp = obj.cast<std::complex<double>>();
        js.push_back(tmp.real());
        js.push_back(tmp.imag());
    } else if (obj.is_none()) {
        return;
    } else {
        throw std::runtime_error("to_json not implemented for this type of object: " + obj.cast<std::string>());
    }
}

void std::from_json(const json_t &js, py::object &o) {
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
        for (auto i = 0; i < js.size(); i++)
        {
            py::object tmp;
            from_json(js[i], tmp);
            obj[i] = tmp;
        }
        o = py::cast(obj);
    } else if (js.is_object()) {
        py::dict obj;
        for (json_t::const_iterator it = js.cbegin(); it != js.cend(); ++it)
        {
            py::object tmp;
            from_json(it.value(), tmp);
            obj[py::str(it.key())] = tmp;
        }
        o = std::move(obj);
    } else if (js.is_null()) {
        o = py::none();
    } else {
        throw std::runtime_error("from_json not implemented for this json::type: " + js.dump());
    }
}

//------------------------------------------------------------------------------
#endif
