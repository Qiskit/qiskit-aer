/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

//#define DEBUG // Uncomment for verbose debugging output
#include <cstdio>
#include <iostream>
#include <string>

#include "version.hpp"
// Simulator
#include "simulators/qasm/qasm_controller.hpp"

/*******************************************************************************
 *
 * Main
 *
 ******************************************************************************/

enum class CmdArguments {
  SHOW_VERSION,
  INPUT_DATA
};

CmdArguments parse_cmd_options(const std::string& argv){
  if(argv == "-v" || argv == "--version"){
       return CmdArguments::SHOW_VERSION;
  }
  return CmdArguments::INPUT_DATA;
}

void show_version(){
  std::cout << "Qiskit Aer: "
  << MAJOR_VERSION << "."
  << MINOR_VERSION << "."
  << PATCH_VERSION << "\n";
}

void failed(const std::string &msg, std::ostream &o = std::cout,
            int indent = -1){
  json_t ret;
  ret["success"] = false;
  ret["status"] = std::string("ERROR: ") + msg;
  o << ret.dump(indent) << std::endl;
}

void usage(const std::string& command, std::ostream &out){
  failed("Invalid command line", out);
  // Print usage message
  std::cerr << "\n\n";
  show_version();
  std::cerr << "\n";
  std::cerr << "Usage: \n";
  std::cerr << command << " [-v] <file>\n";
  std::cerr << "    -v    Show version\n";
  std::cerr << "    file : qobj file\n";
}

int main(int argc, char **argv) {

  std::ostream &out = std::cout; // output stream
  int indent = 4;
  json_t qobj;

  if(argc == 1){
    usage(std::string(argv[0]), out);
    return 1;
  }

  for(auto pos = 1ul; pos < static_cast<unsigned int>(argc); ++pos){
    auto option = parse_cmd_options(std::string(argv[pos]));
    switch(option){
      case CmdArguments::SHOW_VERSION:
        show_version();
        return 0;
      case CmdArguments::INPUT_DATA:
        try {
          qobj = JSON::load(std::string(argv[1]));
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

    // Initialize simulator
    AER::Simulator::QasmController sim;
    out << sim.execute(qobj).dump(4) << std::endl;

    return 0;
  } catch (std::exception &e) {
    std::stringstream msg;
    msg << "Failed to execute qobj (" << e.what() << ")";
    failed(msg.str(), out, indent);
    return 1;
  }

} // end main
