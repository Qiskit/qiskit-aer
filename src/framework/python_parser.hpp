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
#include "json_parser.hpp"
#include "pybind_json.hpp"

namespace AER{

template <>
struct Parser<py::handle> {
    Parser() = delete;

    static bool check_key(const std::string& key, const py::handle& po){
        if(py::isinstance<py::dict>(po)){
            return !py::cast<py::dict>(po)[key.c_str()].is_none();
        }
        return py::hasattr(po, key.c_str());
    }

    static bool check_keys(const std::vector<std::string>& keys, const py::handle& po) {
        bool pass = true;
        for (const auto &s : keys){
            pass &= check_key(s, po);
        }
        return pass;
    }

    static py::object get_py_value(const std::string& key, const py::handle& po){
        if(py::isinstance<py::dict>(po)){
            return py::cast<py::dict>(po)[key.c_str()];
        }
        return po.attr(key.c_str());
    }

    static bool get_value(py::object& var, const std::string& key, const py::handle& po) {
        if(check_key(key, po)) {
            var = get_py_value(key, po);
            return true;
        } else {
            return false;
        }
    }

    template <typename T>
    static bool get_value(T &var, const std::string& key, const py::handle& po){
        if(check_key(key, po)) {
            var = get_py_value(key, po).cast<T>();
            return true;
        } else {
            return false;
        }
    }

    static void convert_to_json(json_t &var, const py::handle& po){
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

    static py::object get_value(const std::string& key, const py::handle& po){
        return get_py_value(key, po);
    }

    static bool is_array(const py::handle& po){
        return py::isinstance<py::list>(po) || py::isinstance<py::array>(po);
    }

    static bool is_array(const std::string& key, const py::handle& po) {
        py::object the_list = get_py_value(key, po);
        return is_array(the_list);
    }

    static bool is_list_like(const py::handle& po){
        return is_array(po) || py::isinstance<py::tuple>(po);
    }

    static py::list get_as_list(const py::handle& po){
        if(!is_list_like(po)){
            throw std::runtime_error("Object is not list like!");
        }
        return  py::cast<py::list>(po);
    }

    static py::list get_list(const std::string& key, const py::handle& po){
        py::object the_list = get_py_value(key, po);
        if(!is_array(the_list)){
            throw std::runtime_error("Object " + key + "is not a list!");
        }
        return py::cast<py::list>(the_list);
    }

    static bool is_number(const py::handle& po){
        return py::isinstance<py::int_>(po) || py::isinstance<py::float_>(po);
    }

    static bool is_number(const std::string& key, const py::handle& po) {
        py::object key_po = get_py_value(key, po);
        return is_number(key_po);
    }

    template <typename T>
    static T get_list_elem(const py::list& po, unsigned int i){
        return py::cast<py::object>(po[i]).cast<T>();
    }

    template <typename T>
    static T get_list_elem(const py::handle& po, unsigned int i){
        auto py_list = get_as_list(po);
        return get_list_elem<T>(py_list, i);
    }

    static std::string dump(const py::handle& po){
        json_t js;
        convert_to_json(js, po);
        return js.dump();
    }
};

template <>
bool Parser<py::handle>::get_value<json_t>(json_t &var, const std::string& key, const py::handle& po){
    py::object ret_po;
    auto success = get_value<py::object>(ret_po, key, po);
    if(success){
        convert_to_json(var, ret_po);
    }
    return success;
}
}

#endif // _aer_framework_python_parser_hpp_
