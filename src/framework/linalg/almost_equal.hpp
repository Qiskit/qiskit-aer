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

#ifndef _aer_framework_linalg_almost_equal_hpp_
#define _aer_framework_linalg_almost_equal_hpp_

#include <complex>
#include <limits>
#include <type_traits>

#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// No silver bullet for floating point comparison techniques.
// With this function the user can at least specify the precision
// If we have numbers closer to 0, then max_diff can be set to a value
// way smaller than epsilon. For numbers larger than 1.0, epsilon will
// scale (the bigger the number, the bigger the epsilon).
template <typename T, typename = enable_if_scalar_t<T>>
bool almost_equal(T f1, T f2,
                  T max_diff = std::numeric_limits<T>::epsilon(),
                  T max_relative_diff = std::numeric_limits<T>::epsilon()) {
  T diff = std::abs(f1 - f2);
  if (diff <= max_diff) return true;
  return diff <=
         max_relative_diff * std::max(std::abs(f1), std::abs(f2));
}

// Complex numbers are probably not considered scalar, so we need another
// implementation to deal with them
template <typename complex_t, typename = enable_if_complex_t<complex_t>>
bool complex_almost_equal(complex_t f1, complex_t f2,
                  decltype(f1.real()) max_diff = std::numeric_limits<decltype(f1.real())>::epsilon(),
                  decltype(f1.real()) max_relative_diff = std::numeric_limits<decltype(f1.real())>::epsilon()) {
  complex_t real_diff = std::abs(f1.real() - f2.real());
  complex_t imag_diff = std::abs(f1.imag() - f2.imag());
  if (real_diff <= max_diff && imag_diff <= max_diff)
    return true;

  return real_diff <=
         max_relative_diff * std::max(std::abs(f1.real()), std::abs(f2.real()))
         &&
         imag_diff <=
         max_relative_diff * std::max(std::abs(f1.imag()), std::abs(f2.imag()));
}



//------------------------------------------------------------------------------
}  // namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif