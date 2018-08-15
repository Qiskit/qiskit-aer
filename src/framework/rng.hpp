/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file rng.hpp
 * @brief RngEngine use by the BaseBackend simulator class
 * @author Christopher J. Wood <cjwood@us.ibm.com>
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

  /**
   * Generate a pseudo random integer from an input discrete distribution
   * @param probs the discrete distribution to sample from
   * @return the generated integer
   */
  template<class IntType> 
  IntType rand_int(std::discrete_distribution<IntType> probs);

  /**
   * Generate a pseudo random integer from a a discrete distribution
   * constructed from an input vector of probabilities for [0,..,n-1]
   * where n is the lenght of the vector. If this vector is not normalized
   * it will be scaled when it is converted to a discrete_distribution
   * @param probs the vector of probabilities
   * @return the generated integer
   */
  int_t rand_int(const std::vector<double> &probs);

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

// randomly distributed integers from discrete distribution
template <class IntType>
IntType RngEngine::rand_int(std::discrete_distribution<IntType> probs) {
  IntType n = probs(rng);
  return n;
}

// randomly distributed integers from vector
int_t RngEngine::rand_int(const std::vector<double> &probs) {
  int_t n = std::discrete_distribution<int_t>(probs.begin(), probs.end())(rng);
  return n;
}

//------------------------------------------------------------------------------
} // End namespace QISKIT
#endif
