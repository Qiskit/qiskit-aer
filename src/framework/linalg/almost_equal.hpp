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
template <typename T1, typename T2, typename = enable_if_numeric<T1>,
          typename = enable_if_numeric<T2>>
bool almost_equal(T1 f1, T2 f2,
                  T1 max_diff = std::numeric_limits<T1>::epsilon(),
                  T1 max_relative_diff = std::numeric_limits<T1>::epsilon()) {
  T1 diff = std::abs<T1>(f1 - f2);
  if (diff <= max_diff) return true;

  return diff <=
         max_relative_diff * std::max<T1>(std::abs<T1>(f1), std::abs<T2>(f2));
}

//------------------------------------------------------------------------------
}  // namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif