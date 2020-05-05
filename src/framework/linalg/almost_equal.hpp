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

#include "framework/matrix.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

template<typename T>
struct epsilon {
    static constexpr auto value = std::numeric_limits<T>::epsilon();
};

/**
 * The epsilon of a complex is the epsilon from the value type of the complex.
 * eg: std::complex<double> = double
 *     std::complex<float> = float
 */
template<typename T>
struct epsilon<std::complex<T>> {
    static constexpr auto value = epsilon<T>::value;
};


/*
 * We use function overloads to dispatch to the correct library functions 
 *   wherever necessary - mainly for container types (vector, matrix) and/or
 *   structured data types (std::complex).
 * read is_tt as is-double-template (I think)
 * See here:
 *   https://stackoverflow.com/questions/41438493/how-to-identifying-whether-a-template-argument-is-stdcomplex
 *
 */
template <template <class...> class TT, class... Args>
std::true_type is_tt_impl(TT<Args...>);
template <template <class...> class TT>
std::false_type is_tt_impl(...);

template <template <class...> class TT, class T>
using is_tt = decltype(is_tt_impl<TT>(std::declval<typename std::decay<T>::type>()));

template <class T>
typename std::enable_if<is_tt<std::complex, T>::value>::type print(T t) {
    std::cout << "(" << t.real() << "," << t.imag() << ")" << std::endl;
}

template <class T>
typename std::enable_if<!is_tt<std::complex, T>::value>::type print(T t) {
    std::cout << t << std::endl;
}

template <typename T> //, typename = enable_if_numeric_t<T>>
bool almost_equal(std::complex<T> f1, std::complex<T> f2,
                  T max_diff = epsilon<T>::value,
                  T max_relative_diff = epsilon<T>::value) {
  return almost_equal<T>(f1.real(), f2.real(), max_diff, max_relative_diff)
         && almost_equal<T>(f1.imag(), f2.imag(), max_diff, max_relative_diff);
}



template <typename T> //, typename = enable_if_numeric_t<T>>
bool almost_equal(matrix<T> mat1, matrix<T> mat2,
                  T max_diff = epsilon<T>::value,
                  T max_relative_diff = epsilon<T>::value) {
  if(mat1.size() != mat2.size())
       return false;

  for(auto i = 0; i < mat1.size(); ++i){
    if(almost_equal<T>(mat1[i], mat2[i], max_diff, max_relative_diff) == false)
      return false;
  }
  return true;
}


// No silver bullet for floating point comparison techniques.
// With this function the user can at least specify the precision
// If we have numbers closer to 0, then max_diff can be set to a value
// way smaller than epsilon. For numbers larger than 1.0, epsilon will
// scale (the bigger the number, the bigger the epsilon).
template <typename T> //, typename = enable_if_numeric_t<T>>
bool almost_equal(T f1, T f2,
                  T max_diff = epsilon<T>::value,
                  T max_relative_diff = epsilon<T>::value) {
  T diff = std::abs(f1 - f2);
  if (diff <= max_diff) return true;
  return diff <=
         max_relative_diff * std::max(std::abs(f1), std::abs(f2));
}

//------------------------------------------------------------------------------
}  // namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
