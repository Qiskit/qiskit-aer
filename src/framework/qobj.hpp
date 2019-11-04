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
  //----------------------------------------------------------------
  // Constructors
  //----------------------------------------------------------------

  // Default constructor and destructors
  Qobj() = default;
  virtual ~Qobj() = default;

  // JSON deserialization constructor
  Qobj(const json_t &js);

  //----------------------------------------------------------------
  // Data
  //----------------------------------------------------------------
  std::string id;                 // qobj identifier passed to result
  std::string type = "QASM";      // currently we only support QASM
  std::vector<Circuit> circuits;  // List of circuits
  json_t header;                  // (optional) passed through to result
  json_t config;                  // (optional) qobj level config data
};


//============================================================================
// JSON initialization and deserialization
//============================================================================

// JSON deserialization
inline void from_json(const json_t &js, Qobj &qobj) {qobj = Qobj(js);}

Qobj::Qobj(const json_t &js) {
  // Check required fields
  if (JSON::get_value(id, "qobj_id", js) == false) {
    throw std::invalid_argument(R"(Invalid qobj: no "qobj_id" field)");
  };
  JSON::get_value(type, "type", js);
  if (type != "QASM") {
    throw std::invalid_argument(R"(Invalid qobj: "type" != "QASM".)");
  };
  if (JSON::check_key("experiments", js) == false) {
    throw std::invalid_argument(R"(Invalid qobj: no "experiments" field.)");
  }

  // Get header and config;
  JSON::get_value(config, "config", js);
  JSON::get_value(header, "header", js);

  // Check for fixed simulator seed
  // If seed is negative a random seed will be chosen for each
  // experiment. Otherwise each experiment will be set to a fixed
  // (but different) seed.
  int_t seed = -1;
  uint_t seed_shift = 0;
  JSON::get_value(seed, "seed_simulator", config);

  // Parse experiments
  const json_t &circs = js["experiments"];
  for (const auto &circ : circs) {
    Circuit circuit(circ, config);
    // Override random seed with fixed seed if set
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