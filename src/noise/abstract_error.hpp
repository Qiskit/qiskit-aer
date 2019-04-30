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

  // Constructors
  AbstractError() = default;
  virtual ~AbstractError() = default;

  // Alias for return type
  using NoiseOps = std::vector<Operations::Op>;

  // Sample an realization of the error from the error model using the passed
  // in RNG engine.
  virtual NoiseOps sample_noise(const reg_t &qubits,
                                RngEngine &rng) const = 0;

  // Load from a JSON file
  virtual void load_from_json(const json_t &js) = 0;
  
  // Set number of qubits or memory bits for error
  inline void set_num_qubits(uint_t num_qubits) {num_qubits_ = num_qubits;}

  // Get number of qubits or memory bits for error
  inline uint_t get_num_qubits() const {return num_qubits_;}

  // Set the sampled errors to be applied after the original operation
  inline void set_errors_after() {errors_after_op_ = true;}

  // Set the sampled errors to be applied before the original operation
  inline void set_errors_before() {errors_after_op_ = false;}

  // Returns true if the errors are to be applied after the operation
  inline bool errors_after() const {return errors_after_op_;}

private:
  // flag for where errors should be applied relative to the sampled op
  bool errors_after_op_ = true;

  // Number of qubits / memory bits the error applies to
  uint_t num_qubits_ = 0;
};


//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif