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

namespace AER {
// This structure is to avoid overload resolving to the wron function,
// as py::objects can always be implicitly converted to json, though
// can break at runtime, or even worse trasnform to json and then to c++
// without notice.
template <typename inputdata_t>
struct Parser {};

template <>
struct Parser<json_t> {
  Parser() = delete;

  template <typename T>
  static bool get_value(T &var, const std::string &key, const json_t &js) {
    return JSON::get_value(var, key, js);
  }

  static bool check_key(const std::string &key, const json_t &js) {
    return JSON::check_key(key, js);
  }

  static const json_t &get_value(const std::string &key, const json_t &js) {
    return JSON::get_value(key, js);
  }

  static bool check_keys(const std::vector<std::string> &keys,
                         const json_t &js) {
    return JSON::check_keys(keys, js);
  }

  static bool is_array(const json_t &js) { return js.is_array(); }

  static bool is_array(const std::string &key, const json_t &js) {
    return js[key].is_array();
  }

  static const json_t &get_as_list(const json_t &js) {
    if (!is_array(js)) {
      throw std::runtime_error("Object is not a list!");
    }
    return js;
  }

  static const json_t &get_list(const std::string &key, const json_t &js) {
    if (!is_array(key, js)) {
      throw std::runtime_error("Object " + key + "is not a list!");
    }
    return JSON::get_value(key, js);
  }

  static bool is_number(const std::string &key, const json_t &js) {
    return js[key].is_number();
  }

  static std::string dump(const json_t &js) { return js.dump(); }

  template <typename T>
  static T get_list_elem(const json_t &js, unsigned int i) {
    return js[i];
  }
};
} // namespace AER

#endif // _aer_framework_json_parser_hpp_
