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

#ifndef _aer_framework_json_parser_hpp_
#define _aer_framework_json_parser_hpp_

#include "json.hpp"


namespace AER{

namespace Parser {
    template <typename T> bool get_value(T &var, const std::string& key, const json_t &js){
        return JSON::get_value(var, key, js);
    }

    bool check_key(const std::string& key, const json_t &js){
        return JSON::check_key(key, js);
    }

    const json_t& get_value(const std::string& key, const json_t &js){
        return JSON::get_value(key, js);
    }

    bool is_array(const json_t &js){
        return js.is_array();
    }

    bool is_array(const std::string& key, const json_t &js){
        return js[key].is_array();
    }

    const json_t& get_as_list(const json_t& js){
        if(!is_array(js)){
            throw std::runtime_error("Object is not a list!");
        }
        return js;
    }

    const json_t& get_list(const std::string& key, const json_t &js){
        if(!is_array(key, js)){
            throw std::runtime_error("Object " + key + "is not a list!");
        }
        return JSON::get_value(key, js);
    }


    bool is_number(const std::string& key, const json_t &js){
        return js[key].is_number();
    }

    // ************** TO DELETE *************************
    void convert_to_json(json_t &var, const json_t& js){
        var = js;
    }
    // **************************************************

    std::string dump(const json_t& js){
        return js.dump();
    }

    template <typename T>
    T get_list_elem(const json_t& js, unsigned int i){
        return js[i];
    }
}
}

#endif // _aer_framework_json_parser_hpp_
