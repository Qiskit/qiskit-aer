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

//#define DEBUG // Uncomment for verbose debugging output
#include <cstdio>
#include <iostream>
#include <string>

#include "version.hpp"
// Simulator
#include "controllers/qasm_controller.hpp"

/*******************************************************************************
 *
 * EXIT CODES:
 * 
 * 0: The Qobj was succesfully executed.
 *    Returns full result JSON.
 * 
 * 1: Command line invalid or Qobj JSON cannot be loaded.
 *    Returns JSON:
 *    {"success": false, "status": "ERROR: Invalid input (error msg)"}
 * 
 * 2: Qobj failed to load or execute.
 *    Returns JSON:
 *    {"success": false, "status": "ERROR: Failed to execute qobj (error msg)"}
 * 
 * 3: At least one experiment in Qobj failed to execute successfully.
 *    Returns parial result JSON with failed experiments returning:
 *    "{"success": false, "status": "ERROR: error msg"}
 *
 ******************************************************************************/

enum class CmdArguments {
  SHOW_VERSION,
  INPUT_CONFIG,
  INPUT_DATA
};

inline CmdArguments parse_cmd_options(const std::string& argv){
  if(argv == "-v" || argv == "--version")
    return CmdArguments::SHOW_VERSION;

  if (argv == "-c" || argv == "--config")
    return CmdArguments::INPUT_CONFIG;

  return CmdArguments::INPUT_DATA;
}

inline void show_version(){
  std::cout << "Qiskit Aer: "
  << AER_MAJOR_VERSION << "."
  << AER_MINOR_VERSION << "."
  << AER_PATCH_VERSION << "\n";
}

inline void failed(const std::string &msg, std::ostream &o = std::cout,
            int indent = -1){
  json_t ret;
  ret["success"] = false;
  ret["status"] = std::string("ERROR: ") + msg;
  o << ret.dump(indent) << std::endl;
}

inline void usage(const std::string& command, std::ostream &out){
  failed("Invalid command line", out);
  // Print usage message
  std::cerr << "\n\n";
  show_version();
  std::cerr << "\n";
  std::cerr << "Usage: \n";
  std::cerr << command << " [-v] [-c <config>] <file>\n";
  std::cerr << "    -v          : Show version\n";
  std::cerr << "    -c <config> : Configuration file\n";;
  std::cerr << "    file        : qobj file\n";
}

int main(int argc, char **argv) {

  std::ostream &out = std::cout; // output stream
  int indent = 4;
  json_t qobj;
  json_t config;

  if(argc == 1){ // NOLINT
    usage(std::string(argv[0]), out); // NOLINT
    return 1;
  }

  // Parse command line options
  for(auto pos = 1UL; pos < static_cast<unsigned int>(argc); ++pos){ // NOLINT
    auto option = parse_cmd_options(std::string(argv[pos])); // NOLINT
    switch(option){
      case CmdArguments::SHOW_VERSION:
        show_version();
        return 0;
      case CmdArguments::INPUT_CONFIG:
        if (++pos == static_cast<unsigned int>(argc)) {
          failed("Invalid config (no file is specified.)", out, indent);
          return 1;
        }
        try {
          config = JSON::load(std::string(argv[pos]));
        }catch(std::exception &e){
          std::string msg = "Invalid config (" +  std::string(e.what()) + ")";
          failed(msg, out, indent);
          return 1;
        }
        break;
      case CmdArguments::INPUT_DATA:
        try {
          qobj = JSON::load(std::string(argv[pos])); // NOLINT
          pos = argc; //Exit from the loop
        }catch(std::exception &e){
          std::string msg = "Invalid input (" +  std::string(e.what()) + ")";
          failed(msg, out, indent);
          return 1;
        }
        break;
    }
  }

  // Execute simulation
  try {

    // Check for command line config
    // and if present add to qobj config
    json_t& config_all = qobj["config"];
    if (!config.empty()) // NOLINT
      config_all.update(config.begin(), config.end());

    // Initialize simulator
    AER::Simulator::QasmController sim;
    auto result = sim.execute(qobj);
    out << result.json().dump(4) << std::endl;

    // Check if execution was successful.
    bool success = false;
    std::string status;
    JSON::get_value(success, "success", result);
    JSON::get_value(status, "status", result);
    if (!success) {
      if(status == "COMPLETED")
        return 3; // The simulation was was completed unsuccesfully.
      return 2; // Failed to execute the Qobj
    }
  } catch (std::exception &e) {
    std::stringstream msg;
    msg << "Failed to execute qobj (" << e.what() << ")";
    failed(msg.str(), out, indent);
    return 2;
  }

  return 0;
} // end main
