/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    noise.hpp
 * @brief   Noise model base class for qiskit-aer simulator engines
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_base_noise_hpp_
#define _aer_base_noise_hpp_

#include "framework/operations.hpp"
#include "framework/types.hpp"
#include "framework/rng.hpp"

// Work in progress

/* Schema idea?

{
  "type": "gate_error",   // string
  "operations": ["x", "y", "z"], // list string
  "qubits": [[0], [1], ...], // if null or missing gate applies to all qubits
  "targets": [[qs,..], ...], // if null or missing gate targets are gate qubits
  "params": [mat1, mat2, mat3,...] // kraus CPTP ops
}

*/

namespace AER {
namespace Noise {

using NoiseOps = std::vector<Operations::Op>;
using NoisePair = std::pair<double, NoiseOps>;

//=========================================================================
// Error abstract base class
//=========================================================================

class Error {
public:
  // Sample an realization of the error from the error model using the passed
  // in RNG engine.
  virtual NoiseOps sample_noise(const Operations::Op &op,
                                const reg_t &qubits,
                                RngEngine &rng) = 0;

  // Return the number of error terms
  inline size_t num_qubits() const {return num_qubits_;}

  // Return the number of error terms
  inline void set_num_qubits(size_t nq) {num_qubits_ = nq;}

protected:
  size_t num_qubits_;
};

//=========================================================================
// Noise Model abstract base class
//=========================================================================

class Model {
public:
  // Sample noise for the current operation
  // Base class returns the input operation with no noise (ideal)
  // derived classes should implement this method
  virtual NoiseOps sample_noise(const Operations::Op &op) {return {op};}

  // Set the RngEngine seed to a fixed value
  inline void set_rng_seed(uint_t seed) { rng_ = RngEngine(seed);}

protected:
  RngEngine rng_; // initialized with random seed
};

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif