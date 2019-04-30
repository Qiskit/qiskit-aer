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

#include "framework/types.hpp"

namespace AER {

/***************************************************************************/ /**
  *
  * RngEngine Class
  *
  * Objects of this class are used to generate random numbers for backends.
  *These
  * are used to decide outcomes of measurements and resets, and for implementing
  * noise.
  *
  ******************************************************************************/

class RngEngine {
public:
  /**
   * Generate a uniformly distributed pseudo random real in the half-open
   * interval [a,b)
   * @param a closed lower bound on interval
   * @param b open upper bound on interval
   * @return the generated double
   */
  double rand(double a, double b);

  /**
   * Generate a uniformly distributed pseudo random real in the half-open
   * interval [0,b)
   * @param b open upper bound on interval
   * @return the generated double
   */
  inline double rand(double b) { return rand(0, b); };

  /**
   * Generate a uniformly distributed pseudo random real in the half-open
   * interval [0,1)
   * @return the generated double
   */
  inline double rand() { return rand(0, 1); };

  /**
   * Generate a uniformly distributed pseudo random integer in the closed
   * interval [a,b]
   * @param a lower bound on interval
   * @param b upper bound on interval
   * @return the generated integer
   */
  int_t rand_int(int_t a, int_t b);
  uint_t rand_int(uint_t a, uint_t b);

  /**
   * Generate a pseudo random integer from a a discrete distribution
   * constructed from an input vector of probabilities for [0,..,n-1]
   * where n is the lenght of the vector. If this vector is not normalized
   * it will be scaled when it is converted to a discrete_distribution
   * @param probs the vector of probabilities
   * @return the generated integer
   */
  uint_t rand_int(const std::vector<double> &probs);

  /**
   * Default constructor initialize RNG engine with a random seed
   */
  RngEngine() {
    std::random_device rd;
    rng.seed(rd());
  };

  /**
   * Seeded constructor initialize RNG engine with a fixed seed
   * @param seed integer to use as seed for mt19937 engine
   */
  explicit RngEngine(uint_t seed) { rng.seed(seed); };


  // Set a fixed seed for the RNG engine
  void set_seed(uint_t seed) { rng.seed(seed); };

private:
  std::mt19937 rng; // Mersenne twister rng engine
};

/*******************************************************************************
 *
 * RngEngine Methods
 *
 ******************************************************************************/

double RngEngine::rand(double a, double b) {
  double p = std::uniform_real_distribution<double>(a, b)(rng);
  return p;
}

// randomly distributed integers in [a,b]
int_t RngEngine::rand_int(int_t a, int_t b) {
  int_t n = std::uniform_int_distribution<int_t>(a, b)(rng);
  return n;
}

uint_t RngEngine::rand_int(uint_t a, uint_t b) {
  int_t n = std::uniform_int_distribution<uint_t>(a, b)(rng);
  return n;
}

// randomly distributed integers from vector
uint_t RngEngine::rand_int(const std::vector<double> &probs) {
  uint_t n = std::discrete_distribution<uint_t>(probs.begin(), probs.end())(rng);
  return n;
}

//------------------------------------------------------------------------------
} // End namespace QISKIT
#endif
