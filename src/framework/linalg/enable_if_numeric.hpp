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

#ifndef _aer_framework_linalg_enable_if_numeric_hpp_
#define _aer_framework_linalg_enable_if_numeric_hpp_

#include <complex>
#include <type_traits>

// Type check template to enable functions if type is a numeric scalar
// (integer, floating points)
template <class T>
struct is_numeric_scalar
    : std::integral_constant<
          bool, std::is_arithmetic<T>::value> {};

// Type check template to enable functions if type is a numeric complex
template <class float_t>
struct is_numeric_complex
    : std::integral_constant<
          bool, std::is_same<std::complex<float>,
                    typename std::remove_cv<float_t>::type>::value ||
                std::is_same<std::complex<double>,
                    typename std::remove_cv<float_t>::type>::value ||
                std::is_same<std::complex<long double>,
                    typename std::remove_cv<float_t>::type>::value> {};

template <class T>
using enable_if_scalar_t = std::enable_if_t<is_numeric_scalar<T>::value>;
template <class float_t>
using enable_if_complex_t = std::enable_if_t<is_numeric_complex<float_t>::value>;

//------------------------------------------------------------------------------
#endif