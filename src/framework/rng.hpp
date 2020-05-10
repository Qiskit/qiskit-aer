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

#ifndef _aer_framework_rng_hpp_
#define _aer_framework_rng_hpp_

#include <cstdint>
#include <random>

namespace AER {

//============================================================================
// RngEngine Class
//
// Objects of this class are used to generate random numbers for backends.
// These are used to decide outcomes of measurements and resets, and for
// implementing noise.
//
//============================================================================

class RngEngine {
public:

  //-----------------------------------------------------------------------
  // Constructors
  //-----------------------------------------------------------------------

  // Default constructor initialize RNG engine with a random seed
  RngEngine() { set_random_seed(); }

  // Seeded constructor initialize RNG engine with a fixed seed
  explicit RngEngine(size_t seed) { set_seed(seed); }

  //-----------------------------------------------------------------------
  // Set fixed or random seed for RNG
  //-----------------------------------------------------------------------

  // Set random seed for the RNG engine
  void set_random_seed() { 
    std::random_device rd;
    set_seed(rd());
  }

  // Set a fixed seed for the RNG engine
  void set_seed(size_t seed) { rng.seed(seed); }
  
  //-----------------------------------------------------------------------
  // Sampling methods
  //-----------------------------------------------------------------------

  // Generate a uniformly distributed pseudo random real in the half-open
  // interval [a,b)
  double rand(double a, double b) {
    return std::uniform_real_distribution<double>(a, b)(rng);
  }

  // Generate a uniformly distributed pseudo random real in the half-open
  // interval [0,b)
  double rand(double b) {
    return rand(double(0), b);
  };

  // Generate a uniformly distributed pseudo random real in the half-open
  // interval [0,1)
  double rand() {return rand(0, 1); };

  // Generate a uniformly distributed pseudo random integer in the closed
  // interval [a,b]
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral<Integer>::value>>
  Integer rand_int(Integer a, Integer b) {
    return std::uniform_int_distribution<Integer>(a, b)(rng);
  }

  // Generate a pseudo random integer from a a discrete distribution
  // constructed from an input vector of probabilities for [0,..,n-1]
  // where n is the lenght of the vector. If this vector is not normalized
  // it will be scaled when it is converted to a discrete_distribution
  template <typename Float,
            typename = std::enable_if_t<std::is_floating_point<Float>::value>>
  size_t rand_int(const std::vector<Float> &probs) {
    return std::discrete_distribution<size_t>(probs.begin(), probs.end())(rng);
  }

private:
  std::mt19937_64 rng; // Mersenne twister rng engine
};

//------------------------------------------------------------------------------
} // End namespace AER
#endif
