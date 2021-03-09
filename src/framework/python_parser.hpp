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

#ifndef _aer_framework_python_parser_hpp_
#define _aer_framework_python_parser_hpp_

#include "json.hpp"

#include "pybind_json.hpp"

namespace py = pybind11;

namespace pybind11 {
    namespace detail {
        template <typename T> struct type_caster<matrix<T>>{
            using base = type_caster_base<matrix<T>>;
        public:
        PYBIND11_TYPE_CASTER(matrix<T>, _("matrix_complex_t"));
            // Conversion part 1 (Python->C++):
            bool load(py::handle src, bool convert){
                // TODO: Check if make sense have to flavors of matrix: F-style and C-style
//                auto py_matrix = py::cast<py::array_t<T, py::array::f_style | py::array::forcecast>>(src);
                auto py_matrix = py::cast<py::array_t<T>>(src);
                auto flags = py_matrix.attr("flags").attr("carray").template cast<bool>();
                if(py_matrix.ndim() != 2){
                    throw std::invalid_argument(std::string("Python: invalid matrix (empty array)."));
                }
                size_t ncols = py_matrix.shape(0);
                size_t nrows = py_matrix.shape(1);
                // Matrix looks ok, now we parse it
                auto raw_mat = py_matrix.template unchecked<2>();
//                auto raw_mat = static_cast<T *>(py_matrix.request().ptr);
//                value = matrix<T>::copy_from_buffer(nrows, ncols, raw_mat);
                value = matrix<T>(nrows, ncols, false);
                for (size_t r = 0; r < nrows; r++) {
                    for (size_t c = 0; c < ncols; c++) {
                        value(r, c) = raw_mat(r, c);
                    }
                }
                return true;
            }
            // Conversion part 2 (C++ -> Python):
            static py::handle cast(matrix<T>, py::return_value_policy policy, py::handle parent){
                throw std::runtime_error("Casting from matrix to python not supported.");
            }
        };
    }
}

namespace AER{

namespace Parser {
    bool check_key(const std::string& key, const py::handle& po){
        return py::hasattr(po, key.c_str());
    }

    py::object get_py_value(const std::string& key, const py::handle& po){
        return po.attr(key.c_str());
    }

    bool get_value(py::object& var, const std::string& key, const py::handle& po) {
        if(check_key(key, po)) {
            var = get_py_value(key, po);
            return true;
        } else {
            return false;
        }
    }

    template <typename T> bool get_value(T &var, const std::string& key, const py::handle& po){
        if(check_key(key, po)) {
            var = get_py_value(key, po).cast<T>();
            return true;
        } else {
            return false;
        }
    }

    void convert_to_json(json_t &var, const py::handle& po){
        if(py::hasattr(po, "to_dict")){
            std::to_json(var, po.attr("to_dict")());
        }else if(py::isinstance<py::list>(po)){
            var = nl::json::array();
            for(auto item: po){
                json_t item_js;
                convert_to_json(item_js, item);
                var.push_back(item_js);
            }
        }else{
            std::to_json(var, po);
        }
    }
    template <> bool get_value(json_t &var, const std::string& key, const py::handle& po){
        py::object ret_po;
        auto success = get_value<py::object>(ret_po, key, po);
        if(success){
            convert_to_json(var, ret_po);
        }
        return success;
    }
    py::object get_value(const std::string& key, const py::handle& po){
        return get_py_value(key, po);
    }

    bool is_array(const py::handle& po){
        return py::isinstance<py::list>(po) || py::isinstance<py::array>(po);
    }

    bool is_array(const std::string& key, const py::handle& po) {
        py::object the_list = get_py_value(key, po);
        return is_array(the_list);
    }

    py::list get_as_list(const py::handle& po){
        if(!is_array(po)){
            throw std::runtime_error("Object is not a list!");
        }
        return  py::cast<py::list>(po);
    }

    py::list get_list(const std::string& key, const py::handle& po){
        py::object the_list = get_py_value(key, po);
        if(!is_array(the_list)){
            throw std::runtime_error("Object " + key + "is not a list!");
        }
        return py::list(the_list);
    }

    bool is_number(const py::handle& po){
        return py::isinstance<py::int_>(po) || py::isinstance<py::float_>(po);
    }

    bool is_number(const std::string& key, const py::handle& po) {
        py::object key_po = get_py_value(key, po);
        return is_number(key_po);
    }

    template <typename T>
    T get_list_elem(const py::list& po, unsigned int i){
        return py::cast<py::object>(po[i]).cast<T>();
    }

    std::string dump(const py::handle& po){
        json_t js;
        convert_to_json(js, po);
        return js.dump();
    }
}
}

#endif // _aer_framework_python_parser_hpp_
