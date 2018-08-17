/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    abstract_error.hpp
 * @brief   Abstract Error base class for Noise model errors
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_noise_abstract_error_hpp_
#define _aer_noise_abstract_error_hpp_

#include "framework/operations.hpp"
#include "framework/types.hpp"
#include "framework/rng.hpp"

namespace AER {
namespace Noise {

//=========================================================================
// Error abstract base class
//=========================================================================

class AbstractError {
public:

  // Alias for return type
  using NoiseOps = std::vector<Operations::Op>;

  // Sample an realization of the error from the error model using the passed
  // in RNG engine.
  virtual NoiseOps sample_noise(const reg_t &qubits,
                                RngEngine &rng) = 0;

  // Check that the parameters of the Error class are valid
  // The output is a pair of a bool (true if valid, false if not)
  // and a string containing an error message for the false case.
  virtual std::pair<bool, std::string> validate() const = 0;

  // Load from a JSON file
  virtual void load_from_json(const json_t &js) = 0;
  
  // Set the sampled errors to be applied after the original operation
  inline void set_errors_after() {errors_after_op_ = true;}

  // Set the sampled errors to be applied before the original operation
  inline void set_errors_before() {errors_after_op_ = false;}

  // Returns true if the errors are to be applied after the operation
  inline bool errors_after() const {return errors_after_op_;}

private:
  // flag for where errors should be applied relative to the sampled op
  bool errors_after_op_ = true;
};


//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif