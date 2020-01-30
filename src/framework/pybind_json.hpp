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

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <iostream>
#include <type_traits>

#include <nlohmann_json.hpp>
#include "framework/json.hpp"

namespace py = pybind11;
namespace nl = nlohmann;
using namespace pybind11::literals;
using json_t = nlohmann::json;

#include "framework/results/result.hpp"
#include "framework/results/data/average_data.hpp"

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

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

/**
 * Convert a Matrix to a python object
 * @param mat is a Matrix
 * @returns a python object (py::list of lists)
 */
template<typename T>
py::object from_matrix(const matrix<T> &mat);

/**
 * Convert a AverageData to a python object
 * @param avg_data is an AverageData
 * @returns a py::dict
 */
template<typename T>
py::dict from_avg_data(const AER::AverageData<T> &avg_data);

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
py::object from_exp_data(const AER::ExperimentData &result);

/**
 * Convert an ExperimentResult to a python object
 * @param result is an ExperimentResult
 * @returns a py::dict
 */
py::object from_exp_result(const AER::ExperimentResult &result);

/**
 * Convert a Result to a python object
 * @param result is a Result
 * @returns a py::dict
 */
py::object from_result(const AER::Result &result);

} //end namespace AerToPy

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
        js = JSON::numpy_to_json(obj.cast<py::array_t<double, py::array::c_style> >());
    } else if (py::isinstance<py::array_t<std::complex<double> > >(obj)) {
        js = JSON::numpy_to_json(obj.cast<py::array_t<std::complex<double>, py::array::c_style> >());
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

//============================================================================
// Pybind Conversion for Simulator types
//============================================================================

template<typename T> 
py::object AerToPy::from_matrix(const matrix<T> &mat) {
  // THIS SHOULD RETURN A py::array_t but the author was la...
  size_t rows = mat.GetRows();
  size_t cols = mat.GetColumns();
  std::vector<std::vector<T> > tbr;
  tbr.reserve(rows);
  for (size_t r = 0; r < rows; r++) {
    std::vector<T> mrow;
    mrow.reserve(cols);
    for (size_t c = 0; c < cols; c++)
      mrow.emplace_back(mat(r, c));
    tbr.emplace_back(mrow);
  }
  return py::cast(tbr);
}

template<typename T> 
py::dict AerToPy::from_avg_data(const AER::AverageData<T> &avg_data) {
  py::dict d;
  d["value"] = avg_data.mean();
  if (avg_data.has_variance()) {
    d["variance"] = avg_data.variance();
  }
  return d;
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
      if ( ! memory.empty()) datum["memory"] = inner_pair.first;
        // Add to list of output
      d1.append(datum);
    }
    d[outer_pair.first.data()] = d1;
  }
  return d;
}

py::object AerToPy::from_exp_data(const AER::ExperimentData &result) {
  py::dict pyresult;

  // Measure data
  if (result.return_counts_ && ! result.counts_.empty())
    pyresult["counts"] = result.counts_;
  if (result.return_memory_ && ! result.memory_.empty())
    pyresult["memory"] = result.memory_;
  if (result.return_register_ && ! result.register_.empty())
    pyresult["register"] = result.register_;

  // Add additional data
  for (const auto &pair : result.additional_json_data_) {
    py::object tmp;
    from_json(pair.second, tmp);
    pyresult[pair.first.data()] = tmp;
  }
  for (const auto &pair : result.additional_cvector_data_) {
    pyresult[pair.first.data()] = pair.second;
  }
  for (const auto &pair : result.additional_cmatrix_data_) {
    pyresult[pair.first.data()] = AerToPy::from_matrix(pair.second);    
  }

  // Snapshot data
  if (result.return_snapshots_) {
    py::dict snapshots;
    // Average snapshots
    for (const auto &pair : result.average_json_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.average_complex_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : result.average_cvector_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : result.average_cmatrix_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : result.average_cmap_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    for (auto &pair : result.average_rmap_snapshots_) {
      snapshots[pair.first.data()] = AerToPy::from_avg_snap(pair.second);
    }
    // Singleshot snapshot data
    // Note these will override the average snapshots
    // if they share the same type string
    for (const auto &pair : result.pershot_json_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.pershot_complex_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : result.pershot_cvector_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : result.pershot_cmatrix_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : result.pershot_cmap_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    for (auto &pair : result.pershot_rmap_snapshots_) {
      py::dict d;
      // string PershotData
      for (auto &per_pair : pair.second.data())
        d[per_pair.first.data()] = per_pair.second.data();
      snapshots[pair.first.data()] = d;
    }
    if ( py::len(snapshots) != 0 )
        pyresult["snapshots"] = snapshots;
  }
  //for (auto item : pyresult)
  //  py::print("    {}:, {}"_s.format(item.first, item.second));
  return pyresult;
}

py::object AerToPy::from_exp_result(const AER::ExperimentResult &result) {
  py::dict pyresult;

  pyresult["shots"] = result.shots;
  pyresult["seed_simulator"] = result.seed;

  pyresult["data"] = AerToPy::from_exp_data(result.data);

  pyresult["success"] = (result.status == AER::ExperimentResult::Status::completed);
  switch (result.status) {
    case AER::ExperimentResult::Status::completed:
      pyresult["status"] = std::string("DONE");
      break;
    case AER::ExperimentResult::Status::error:
      pyresult["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::ExperimentResult::Status::empty:
      pyresult["status"] = std::string("EMPTY");
  }
  pyresult["time_taken"] = result.time_taken;
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
  return pyresult;

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
    exp_results.append(AerToPy::from_exp_result(exp));
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
      pyresult["status"] = std::string("COMPLETED");
      break;
    case AER::Result::Status::partial_completed:
      pyresult["status"] = std::string("PARTIAL COMPLETED");
      break;
    case AER::Result::Status::error:
      pyresult["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::Result::Status::empty:
      pyresult["status"] = std::string("EMPTY");
  }
  return pyresult;

}

#endif
