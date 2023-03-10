/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "framework/json.hpp"
#include <string>

#include <catch2/catch.hpp>

namespace Catch {
template <typename T>
std::string convertMyTypeToString(const std::vector<T> &value) {
  std::stringstream oss;
  oss << value;
  return oss.str();
}

template <typename T>
struct StringMaker<std::vector<T>> {
  static std::string convert(const std::vector<AER::complex_t> &value) {
    return convertMyTypeToString(value);
  }
};
} // namespace Catch

namespace AER {
namespace Test {
namespace Utilities {
inline json_t load_qobj(const std::string &filename) {
  return JSON::load(filename);
}

template <typename T>
T calculate_floats(T start, T decrement, int count) {
  for (int i = 0; i < count; ++i)
    start -= decrement;
  return start;
}

template <typename T, typename U>
bool _compare(const matrix<T> &lhs, const matrix<T> &rhs,
              U max_diff = std::numeric_limits<U>::epsilon(),
              U max_relative_diff = std::numeric_limits<U>::epsilon()) {
  bool res = true;
  std::ostringstream message;
  if (lhs.size() != rhs.size()) {
    res = false;
  }

  for (size_t i = 0; i < lhs.GetRows(); ++i) {
    for (size_t j = 0; j < lhs.GetColumns(); ++j) {
      if (!(AER::Linalg::almost_equal(lhs(i, j), rhs(i, j), max_diff,
                                      max_relative_diff))) {
        message << "Matrices differ at element: (" << i << ", " << j << ")"
                << std::setprecision(22) << ". [" << lhs(i, j) << "] != ["
                << rhs(i, j) << "]\n";
        res = false;
      }
    }
  }
  if (!res) {
    message << "Matrices differ: " << lhs << " != " << rhs << std::endl;
    std::cout << message.str();
  }
  return res;
}

template <typename T>
bool compare(const matrix<T> &lhs, const matrix<T> &rhs,
             T max_diff = std::numeric_limits<T>::epsilon(),
             T max_relative_diff = std::numeric_limits<T>::epsilon()) {
  return _compare(lhs, rhs, max_diff, max_relative_diff);
}

template <typename T>
bool compare(const matrix<std::complex<T>> &lhs,
             const matrix<std::complex<T>> &rhs,
             T max_diff = std::numeric_limits<T>::epsilon(),
             T max_relative_diff = std::numeric_limits<T>::epsilon()) {
  return _compare(lhs, rhs, max_diff, max_relative_diff);
}

template <typename T, typename U>
bool _compare(const std::vector<T> &lhs, const std::vector<T> &rhs,
              U max_diff = std::numeric_limits<U>::epsilon(),
              U max_relative_diff = std::numeric_limits<U>::epsilon()) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(AER::Linalg::almost_equal(lhs[i], rhs[i], max_diff,
                                    max_relative_diff))) {
      std::cout << "Vectors differ at element: " << i << std::setprecision(22)
                << ". [" << lhs[i] << "] != [" << rhs[i] << "]\n";
      std::cout << "Vectors differ: " << Catch::convertMyTypeToString(lhs)
                << " != " << Catch::convertMyTypeToString(rhs) << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
bool compare(const std::vector<T> &lhs, const std::vector<T> &rhs,
             T max_diff = std::numeric_limits<T>::epsilon(),
             T max_relative_diff = std::numeric_limits<T>::epsilon()) {
  return _compare(lhs, rhs, max_diff, max_relative_diff);
}

template <typename T>
bool compare(const std::vector<std::complex<T>> &lhs,
             const std::vector<std::complex<T>> &rhs,
             T max_diff = std::numeric_limits<T>::epsilon(),
             T max_relative_diff = std::numeric_limits<T>::epsilon()) {
  return _compare(lhs, rhs, max_diff, max_relative_diff);
}
} // namespace Utilities
} // namespace Test
} // namespace AER