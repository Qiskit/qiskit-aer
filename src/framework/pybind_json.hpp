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
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>


#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <nlohmann/json.hpp>
#include "framework/json.hpp"

namespace py = pybind11;
namespace nl = nlohmann;
using namespace pybind11::literals;
using json_t = nlohmann::json;

#include "framework/results/result.hpp"
#include "framework/results/data/average_data.hpp"

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

/**
 * Convert a std::vector into a numpy array
 * @param src is a vector
 * @returns a python object (py::array_t<T>)
 */
template<typename T>
py::object array_from_vector(const std::vector<T> &src);


/**
 * Convert a Matrix into a numpy array
 * @param mat is a Matrix
 * @returns a python object (py::array_t<T>)
 */
template<typename T>
py::object array_from_matrix(const matrix<T> &mat);

/**
 * Convert a AverageData to a python object
 * @param avg_data is an AverageData
 * @returns a py::dict
 */
template<typename T>
py::object from_avg_data(const AER::AverageData<T> &avg_data);

/**
 * Convert a AverageData to a python object
 * @param avg_data is an AverageData
 * @returns a py::dict
 */
template<typename T>
py::object from_avg_data(const AER::AverageData<matrix<T>> &avg_data);

/**
 * Convert a AverageData to a python object
 * @param avg_data is an AverageData
 * @returns a py::dict
 */
template<typename T>
py::object from_avg_data(const AER::AverageData<std::vector<T>> &avg_data);

/**
 * Convert a AverageSnapshot to a python object
 * @param avg_snap is an AverageSnapshot
 * @returns a py::dict
 */
template<typename T>
py::object from_avg_snap(const AER::AverageSnapshot<T> &avg_snap);

/**
 * Convert an ExperimentData to a python object
 * @param result is an ExperimentData
 * @returns a py::dict
 */
py::object from_data(const AER::ExperimentData &result);

/**
 * Convert an ExperimentResult to a python object
 * @param result is an ExperimentResult
 * @returns a py::dict
 */
py::object from_experiment(const AER::ExperimentResult &result);

/**
 * Convert a Result to a python object
 * @param result is a Result
 * @returns a py::dict
 */
py::object from_result(const AER::Result &result);

} //end namespace AerToPy

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

} //end namespace JSON

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

    T *ptr = (T *) buf.ptr;
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
json_t JSON::numpy_to_json_3d(py::array_t<T, py::array::c_style> arr) {
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
json_t JSON::numpy_to_json(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    //std::cout << "buff dim: " << buf.ndim << std::endl;

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

void std::to_json(json_t &js, const py::handle &obj) {
    if (py::isinstance<py::float_>(obj)) {
        js = obj.cast<nl::json::number_float_t>();
    } else if (py::isinstance<py::bool_>(obj)) {
        js = obj.cast<nl::json::boolean_t>();
    } else if (py::isinstance<py::int_>(obj)) {
        js = obj.cast<nl::json::number_integer_t>();
    } else if (py::isinstance<py::str>(obj)) {
        js = obj.cast<nl::json::string_t>();
    } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        js = nl::json::array();
        for (py::handle value: obj) {
            js.push_back(value);
        }
    } else if (py::isinstance<py::dict>(obj)) {
        for (auto item : py::cast<py::dict>(obj)) {
            js[item.first.cast<nl::json::string_t>()] = item.second;
        }
    } else if (py::isinstance<py::array_t<double> >(obj)) {
        js = JSON::numpy_to_json(obj.cast<py::array_t<double, py::array::c_style> >());
    } else if (py::isinstance<py::array_t<std::complex<double> > >(obj)) {
        js = JSON::numpy_to_json(obj.cast<py::array_t<std::complex<double>, py::array::c_style> >());
    } else if (obj.is_none()) {
        return;
    } else {
        auto type_str = std::string(py::str(obj.get_type()));
        if ( type_str == "<class \'complex\'>"
             || type_str == "<class \'numpy.complex64\'>"
             || type_str == "<class \'numpy.complex128\'>"
             || type_str == "<class \'numpy.complex_\'>" ) {
            auto tmp = obj.cast<std::complex<double>>();
            js.push_back(tmp.real());
            js.push_back(tmp.imag());
        } else if ( type_str == "<class \'numpy.uint32\'>"
                    || type_str == "<class \'numpy.uint64\'>"
                    || type_str == "<class \'numpy.int32\'>"
                    || type_str == "<class \'numpy.int64\'>" ) {
            js = obj.cast<nl::json::number_integer_t>();
        } else if ( type_str == "<class \'numpy.float32\'>"
                    || type_str == "<class \'numpy.float64\'>" ) {
            js = obj.cast<nl::json::number_float_t>();
        } else {
            throw std::runtime_error("to_json not implemented for this type of object: " + std::string(py::str(obj.get_type())));
        }
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

//============================================================================
// Pybind Conversion for Simulator types
//============================================================================

template<typename T>
py::object AerToPy::array_from_vector(const std::vector<T> &src) {
    std::array<py::ssize_t, 1> shape { static_cast<py::ssize_t>(src.size()) };
    auto tbr = py::array_t<T, py::array::f_style>(shape);
    auto buf = tbr.template mutable_unchecked<1>();
    for (size_t i = 0; i < src.size(); i++) {
        buf(i) = src[i];
    }
    return std::move(tbr);
}

template<typename T>
py::object AerToPy::array_from_matrix(const matrix<T> &src) {
    std::array<py::ssize_t, 2> shape { static_cast<py::ssize_t>(src.GetRows()), static_cast<py::ssize_t>(src.GetColumns()) };
    auto tbr = py::array_t<T, py::array::f_style>(shape);
    auto buf = tbr.template mutable_unchecked<2>();
    for (size_t r = 0; r < src.GetRows(); r++) {
        for (size_t c = 0; c < src.GetColumns(); c++)
            buf(r, c) = src(r, c);
    }
    return std::move(tbr);
}

template<typename T> 
py::object AerToPy::from_avg_data(const AER::AverageData<T> &avg_data) {
  py::dict d;
  d["value"] = avg_data.mean();
  if (avg_data.has_variance()) {
    d["variance"] = avg_data.variance();
  }
  return std::move(d);
}

template<typename T> 
py::object AerToPy::from_avg_data(const AER::AverageData<matrix<T>> &avg_data) {
  py::dict d;
  d["value"] = AerToPy::array_from_matrix(avg_data.mean());
  if (avg_data.has_variance()) {
    d["variance"] = AerToPy::array_from_matrix(avg_data.variance());
  }
  return std::move(d);
}

template<typename T> 
py::object AerToPy::from_avg_data(const AER::AverageData<std::vector<T>> &avg_data) {
  py::dict d;
  d["value"] = AerToPy::array_from_vector(avg_data.mean());
  if (avg_data.has_variance()) {
    d["variance"] = AerToPy::array_from_vector(avg_data.variance());
  }
  return std::move(d);
}
template<typename T> 
py::object AerToPy::from_avg_snap(const AER::AverageSnapshot<T> &avg_snap) {
  py::dict d;
  for (const auto &outer_pair : avg_snap.data()) {
    py::list d1;
    for (const auto &inner_pair : outer_pair.second) {
      // Store mean and variance for snapshot
      py::dict datum = AerToPy::from_avg_data(inner_pair.second);
      // Add memory key if there are classical registers
      auto memory = inner_pair.first;
      if ( ! memory.empty()) {
        datum["memory"] = inner_pair.first;
      }
      // Add to list of output
      d1.append(datum);
    }
    d[outer_pair.first.data()] = d1;
  }
  return std::move(d);
}

py::object AerToPy::from_data(const AER::ExperimentData &datum) {
  py::dict pydata;

  // Measure data
  if (datum.return_counts_ && ! datum.counts_.empty()) {
    pydata["counts"] = datum.counts_;
  }
  if (datum.return_memory_ && ! datum.memory_.empty()) {
    pydata["memory"] = datum.memory_;
  }
  if (datum.return_register_ && ! datum.register_.empty()) {
    pydata["register"] = datum.register_;
  }

  // Add additional data
  for (const auto &pair : datum.additional_json_data_) {
    py::object tmp;
    from_json(pair.second, tmp);
    pydata[pair.first.data()] = tmp;
  }
  for (const auto &pair : datum.additional_cvector_data_) {
    pydata[pair.first.data()] = AerToPy::array_from_vector(pair.second);
  }
  for (const auto &pair : datum.additional_cmatrix_data_) {
    pydata[pair.first.data()] = AerToPy::array_from_matrix(pair.second);
  }

  // Snapshot data
  if (datum.return_snapshots_) {
    py::dict snapshots;
    // Average snapshots
    for (const auto &pair : datum.average_json_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : datum.average_complex_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : datum.average_cvector_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : datum.average_cmatrix_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : datum.average_cmap_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : datum.average_rmap_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    // Singleshot snapshot data
    // Note these will override the average snapshots
    // if they share the same type string
    for (const auto &pair : datum.pershot_json_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : datum.pershot_complex_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : datum.pershot_cvector_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data()) {
        py::list l;
        for (auto &matr : per_pair.second.data())
          l.append(AerToPy::array_from_vector(matr));
        d[per_pair.first.data()] = l;
      }
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : datum.pershot_cmatrix_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data()) {
        py::list l;
        for (auto &matr : per_pair.second.data())
          l.append(AerToPy::array_from_matrix(matr));
        d[per_pair.first.data()] = l;
      }
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : datum.pershot_cmap_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : datum.pershot_rmap_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }

    if ( py::len(snapshots) != 0 )
        pydata["snapshots"] = snapshots;
  }
  //for (auto item : pydatum)
  //  py::print("    {}:, {}"_s.format(item.first, item.second));
  return std::move(pydata);
}

py::object AerToPy::from_experiment(const AER::ExperimentResult &result) {
  py::dict pyexperiment;

  pyexperiment["shots"] = result.shots;
  pyexperiment["seed_simulator"] = result.seed;

  pyexperiment["data"] = AerToPy::from_data(result.data);

  pyexperiment["success"] = (result.status == AER::ExperimentResult::Status::completed);
  switch (result.status) {
    case AER::ExperimentResult::Status::completed:
      pyexperiment["status"] = "DONE";
      break;
    case AER::ExperimentResult::Status::error:
      pyexperiment["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::ExperimentResult::Status::empty:
      pyexperiment["status"] = "EMPTY";
  }
  pyexperiment["time_taken"] = result.time_taken;
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pyexperiment["header"] = tmp;
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pyexperiment["metadata"] = tmp;
  }
  return std::move(pyexperiment);
}

py::object AerToPy::from_result(const AER::Result &result) {
  py::dict pyresult;
  pyresult["qobj_id"] = result.qobj_id;

  pyresult["backend_name"] = result.backend_name;
  pyresult["backend_version"] = result.backend_version;
  pyresult["date"] = result.date;
  pyresult["job_id"] = result.job_id;

  py::list exp_results;
  for( const AER::ExperimentResult& exp : result.results)
    exp_results.append(AerToPy::from_experiment(exp));
  pyresult["results"] = exp_results;

  // For header and metadata we continue using the json->pyobject casting
  //   bc these are assumed to be small relative to the ExperimentResults
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pyresult["header"] = tmp;
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pyresult["metadata"] = tmp;
  }
  pyresult["success"] = (result.status == AER::Result::Status::completed);
  switch (result.status) {
    case AER::Result::Status::completed:
      pyresult["status"] = "COMPLETED";
      break;
    case AER::Result::Status::partial_completed:
      pyresult["status"] = "PARTIAL COMPLETED";
      break;
    case AER::Result::Status::error:
      pyresult["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::Result::Status::empty:
      pyresult["status"] = "EMPTY";
  }
  return std::move(pyresult);

}

#endif
