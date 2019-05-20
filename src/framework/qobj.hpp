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
  int_t seed = -1;                // Seed for qobj (-1 for random)

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
  if (JSON::get_value(id, "qobj_id", js) == false) {
    throw std::invalid_argument("Invalid qobj: no \"qobj_id\" field");
  };
  // Get header and config;
  JSON::get_value(config, "config", js);
  JSON::get_value(header, "header", js);
  // Check for fixed seed
  JSON::get_value(seed, "seed", config); // DEPRECIATED: Remove in 0.3.
  JSON::get_value(seed, "seed_simulator", config);
  // Get type
  JSON::get_value(type, "type", js);
  if (type != "QASM") {
    throw std::invalid_argument("Invalid qobj: currently only \"type\" = \"QASM\" is supported.");
  };
  // Get circuits
  if (JSON::check_key("experiments", js) == false) {
    throw std::invalid_argument("Invalid qobj: no \"experiments\" field.");
  }
  // Parse experiments
  const json_t &circs = js["experiments"];
  uint_t seed_shift = 0;
  for (const auto &circ : circs) {
    Circuit circuit(circ, config);
    // override random seed with fixed seed if set
    // We shift the seed for each successive experiment
    // So that results aren't correlated between experiments
    if (seed >= 0) {
      circuit.set_seed(seed + seed_shift);
      seed_shift += 2113; // Shift the seed
    }
    circuits.push_back(circuit);
  }
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
//------------------------------------------------------------------------------
#endif