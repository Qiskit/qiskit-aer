/**
 * Copyright 2017, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    qobj.hpp
 * @brief   Qobj class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_framework_qobj_hpp_
#define _aer_framework_qobj_hpp_

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/circuit.hpp"

namespace AER {

//============================================================================
// Qobj data structure
//============================================================================

class Qobj {
public:

  Qobj() = default;
  virtual ~Qobj() = default;

  // JSON deserialization constructor
  inline Qobj(const json_t &js) {load_qobj_from_json(js);};
  

  //----------------------------------------------------------------
  // Data
  //----------------------------------------------------------------

  std::string id;                 // qobj identifier passed to result
  std::string type = "QASM";      // currently we only support QASM      
  std::vector<Circuit> circuits;  // List of circuits
  json_t header;                  // (optional) passed through to result;
  json_t config;                  // (optional) not currently used?

  //----------------------------------------------------------------
  // Loading Functions
  //----------------------------------------------------------------
  
  void load_qobj_from_json(const json_t &js);
  void load_qobj_from_file(const std::string file);
  inline void load_qobj_from_string(const std::string &input);
};

inline void from_json(const json_t &js, Qobj &qobj) {qobj = Qobj(js);}

//============================================================================
// Implementation: Qobj methods
//============================================================================

void Qobj::load_qobj_from_file(const std::string file) {
  json_t js = JSON::load(file);
  load_qobj_from_json(js);
}


void Qobj::load_qobj_from_string(const std::string &input) {
  json_t js = json_t::parse(input);
  load_qobj_from_json(js);
}


void Qobj::load_qobj_from_json(const json_t &js) {

  // Get qobj id
  if (JSON::get_value(id, "id", js) == false) {
    throw std::invalid_argument("Invalid qobj: no \"id\" field");
  };
  // Get header and config;
  JSON::get_value(config, "config", js);
  JSON::get_value(header, "header", js);
  // Get type
  JSON::get_value(type, "type", js);
  if (type != "QASM") {
    throw std::invalid_argument("Invalid qobj: currently only \"type\" = \"QASM\" is supported.");
  };
  // Get circuits
  if (JSON::check_key("experiments", js) == false) {
    throw std::invalid_argument("Invalid qobj: no \"experiments\" field.");
  }
  const json_t &circs = js["experiments"];
  for (auto it = circs.cbegin(); it != circs.cend(); ++it) {
    circuits.emplace_back(*it);
  }
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
//------------------------------------------------------------------------------
#endif