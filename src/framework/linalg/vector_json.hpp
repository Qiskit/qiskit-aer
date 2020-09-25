/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_linalg_vector_json_hpp
#define _aer_framework_linalg_vector_json_hpp

#include <nlohmann/json.hpp>

#include "framework/linalg/vector.hpp"

namespace AER {
//------------------------------------------------------------------------------
// Implementation: JSON Conversion
//------------------------------------------------------------------------------

  
template <typename T> void to_json(nlohmann::json &js, const Vector<T> &vec) {
  js = nlohmann::json();
  for (size_t i = 0; i < vec.size(); i++) {
    js.push_back(vec[i]);
  }
}


template <typename T> void from_json(const nlohmann::json &js, Vector<T> &vec) {
  // Check JSON is an array
  if(!js.is_array()) {
    throw std::invalid_argument(
        std::string("JSON: invalid Vector (not array)."));
  }
  // Check if JSON is empty
  if(js.empty()) {
    return;
  }

  // Initialize empty vector of correct size
  const size_t size = js.size();
  vec = Vector<T>(size, false);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = js[i].get<T>();
  }
}

//------------------------------------------------------------------------------
} // end Namespace AER
//------------------------------------------------------------------------------
#endif
