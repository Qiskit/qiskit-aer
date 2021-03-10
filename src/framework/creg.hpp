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

#ifndef _aer_framework_creg_hpp_
#define _aer_framework_creg_hpp_

#include "framework/operations.hpp"
#include "framework/utils.hpp"
#include "framework/rng.hpp"

namespace AER {

//============================================================================
// ClassicalRegister base class for Qiskit-Aer
//============================================================================

// ClassicalRegister class
class ClassicalRegister {

public:

  // Return the current value of the memory as little-endian hex-string
  inline std::string memory_hex() const {return Utils::bin2hex(creg_memory_);}

  // Return the current value of the memory as little-endian bit-string
  inline std::string memory_bin() const {return "0b" + creg_memory_;}

  // Return the current value of the memory as little-endian hex-string
  inline std::string register_hex() const {return Utils::bin2hex(creg_register_);}

  // Return the current value of the memory as little-endian bit-string
  inline std::string register_bin() const {return "0b" + creg_register_;}

  // Return the size of the memory bits
  size_t memory_size() const {return creg_memory_.size();}

  // Return the size of the register bits
  size_t register_size() const {return creg_register_.size();}

  // Return a reference to the current value of the memory
  // this is a bit-string without the "0b" prefix.
  inline auto& creg_memory() {return creg_memory_;}

  // Return a reference to the current value of the memory
  // this is a bit-string without the "0b" prefix.
  inline auto& creg_register() {return creg_register_;}

  // Initialize the memory and register bits to default values (all 0)
  void initialize(size_t num_memory, size_t num_registers);

  // Initialize the memory and register bits to specific values
  void initialize(size_t num_memory,
                  size_t num_registers,
                  const std::string &memory_hex,
                  const std::string &register_hex);

  // Return true if a conditional op test passes based on the current
  // register bits values.
  // If the op is not a conditional op this will return true.
  bool check_conditional(const Operations::Op &op) const;

  // Apply a boolean function Op
  void apply_bfunc(const Operations::Op &op);

  // Apply readout error instruction to classical registers
  void apply_roerror(const Operations::Op &op, RngEngine &rng);

  // Store a measurement outcome in the specified memory and register bit locations
  void store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers);

protected:

  // Classical registers
  std::string creg_memory_;   // standard classical bit memory
  std::string creg_register_; // optional classical bit register

  // Measurement config settings
  bool return_hex_strings_ = true;       // Set to false for bit-string output
};

//============================================================================
// Implementations
//============================================================================

void ClassicalRegister::initialize(size_t num_memory, size_t num_register) {
  // Set registers to the all 0 bit state
  creg_memory_ = std::string(num_memory, '0');
  creg_register_ = std::string(num_register, '0');
}


void ClassicalRegister::initialize(size_t num_memory,
                                   size_t num_register,
                                   const std::string &memory_hex,
                                   const std::string &register_hex) {
  // Convert to bit-string for internal storage
  std::string memory_bin = Utils::hex2bin(memory_hex, false);
  creg_memory_ = std::move(Utils::padleft_inplace(memory_bin, '0', num_memory));

  std::string register_bin = Utils::hex2bin(register_hex, false);
  creg_register_ = std::move(Utils::padleft_inplace(memory_bin, '0', num_register));
}


void ClassicalRegister::store_measure(const reg_t &outcome,
                                      const reg_t &memory,
                                      const reg_t &registers) {
  // Assumes memory and registers are either empty or same size as outcome!
  bool use_mem = !memory.empty();
  bool use_reg = !registers.empty();
  for (size_t j=0; j < outcome.size(); j++) {
    if (use_mem) {
      // least significant bit first ordering
      const size_t pos = creg_memory_.size() - memory[j] - 1; 
      creg_memory_[pos] = std::to_string(outcome[j])[0]; // int->string->char
    }
    if (use_reg) {
      // least significant bit first ordering
      const size_t pos = creg_register_.size() - registers[j] - 1; 
      creg_register_[pos] = std::to_string(outcome[j])[0];  // int->string->char
    }
  }
}


bool ClassicalRegister::check_conditional(const Operations::Op &op) const {
  // Check if op is conditional
  if (op.conditional)
    return (creg_register_[creg_register_.size() - op.conditional_reg - 1] == '1');

  // Op is not conditional
  return true;
}


void ClassicalRegister::apply_bfunc(const Operations::Op &op) {

  // Check input is boolean function op
  if (op.type != Operations::OpType::bfunc) {
    throw std::invalid_argument("ClassicalRegister::apply_bfunc: Input is not a bfunc op.");
  }

  const std::string &mask = op.string_params[0];
  const std::string &target_val = op.string_params[1];
  int_t compared; // if equal this should be 0, if less than -1, if greater than +1

  // Check if register size fits into a 64-bit integer
  if (creg_register_.size() <= 64) {
    uint_t reg_int = std::stoull(creg_register_, nullptr, 2); // stored as bitstring
    uint_t mask_int = std::stoull(mask, nullptr, 16); // stored as hexstring
    uint_t target_int = std::stoull(target_val, nullptr, 16); // stored as hexstring
    compared = (reg_int & mask_int) - target_int;
  } else {
    // We need to use big ints so we implement the bit-mask via the binary string
    // representation rather than using a big integer class
    std::string mask_bin = Utils::hex2bin(mask, false);
    size_t length = std::min(mask_bin.size(), creg_register_.size());
    std::string masked_val = std::string(length, '0');
    for (size_t rev_pos = 0; rev_pos < length; rev_pos++) {
      masked_val[length - 1 - rev_pos] = (mask_bin[mask_bin.size() - 1 - rev_pos] 
                                          & creg_register_[creg_register_.size() - 1 - rev_pos]);
    }
    // remove leading 0's
    size_t end_i = masked_val.find('1');
    if (end_i == std::string::npos)
        masked_val = "0";
    else
        masked_val.erase(0, end_i);

    masked_val = Utils::bin2hex(masked_val); // convert to hex string
    // Using string comparison to compare to target value
    compared = masked_val.compare(target_val);
  }
  // check value of compared integer for different comparison operations
  bool outcome;
  switch (op.bfunc) {
    case Operations::RegComparison::Equal:
      outcome = (compared == 0);
      break;
    case Operations::RegComparison::NotEqual:
      outcome = (compared != 0);
      break;
    case Operations::RegComparison::Less:
      outcome = (compared < 0);
      break;
    case Operations::RegComparison::LessEqual:
      outcome = (compared <= 0);
      break;
    case Operations::RegComparison::Greater:
      outcome = (compared > 0);
      break;
    case Operations::RegComparison::GreaterEqual:
      outcome = (compared >= 0);
      break;
    default:
      // we shouldn't ever get here
      throw std::invalid_argument("Invalid boolean function relation.");
  }
  // Store outcome in register
  if (op.registers.size() > 0) {
    const size_t pos = creg_register_.size() - op.registers[0] - 1; 
    creg_register_[pos] = (outcome) ? '1' : '0';
  }
  // Optionally store outcome in memory
  if (op.memory.size() > 0) {
    const size_t pos = creg_memory_.size() - op.memory[0] - 1; 
    creg_memory_[pos] = (outcome) ? '1' : '0';
  }
}

// Apply readout error instruction to classical registers
void ClassicalRegister::apply_roerror(const Operations::Op &op, RngEngine &rng) {
  
  // Check input is readout error op
  if (op.type != Operations::OpType::roerror) {
    throw std::invalid_argument("ClassicalRegister::apply_roerror Input is not a readout error op.");
  }
  
  // Get current classical bit (and optionally register bit) values
  std::string mem_str;
  
  // Get values of bits as binary string
  // We iterate from the end of the list of memory bits
  for (auto it = op.memory.rbegin(); it < op.memory.rend(); ++it) {
    auto bit = *it;
    mem_str.push_back(creg_memory_[creg_memory_.size() - 1 - bit]);
  }
  auto mem_val = std::stoull(mem_str, nullptr, 2);
  auto outcome = rng.rand_int(op.probs[mem_val]);
  auto noise_str = Utils::int2string(outcome, 2, op.memory.size());
  for (size_t pos = 0; pos < op.memory.size(); ++pos) {
    auto bit = op.memory[pos];
    creg_memory_[creg_memory_.size() - 1 - bit] = noise_str[noise_str.size() - 1 - pos];
  }
  // and the same error to register classical bits if they are used
  for (size_t pos = 0; pos < op.registers.size(); ++pos) {
    auto bit = op.registers[pos];
    creg_register_[creg_register_.size() - 1 - bit] = noise_str[noise_str.size() - 1 - pos];
  }
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
