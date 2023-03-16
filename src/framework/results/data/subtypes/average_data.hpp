/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2021.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_results_data_subtypes_average_hpp_
#define _aer_framework_results_data_subtypes_average_hpp_

#include "framework/results/data/subtypes/accum_data.hpp"

namespace AER {

template <typename T>
class AverageData : public AccumData<T> {
  using Base = AccumData<T>;

public:
  // Access data
  T &value();

  // Add data (copy)
  void add(const T &data);

  // Add data (move)
  void add(T &&data);

  // Add data
  void combine(AverageData<T> &&other);

  // Clear all stored data
  void clear();

  // Divide accum by counts to convert to the normalized mean
  void normalize();

  // Multiply accum by counts to convert to the un-normalized mean
  void denormalize();

protected:
  // Number of datum that have been accumulated
  size_t count_ = 0;

  // Flag for whether the accumulated data has been divided
  // by the count
  bool normalized_ = false;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
void AverageData<T>::add(const T &data) {
  denormalize();
  Base::add(data);
  count_ += 1;
}

template <typename T>
void AverageData<T>::add(T &&data) {
  denormalize();
  Base::add(std::move(data));
  count_ += 1;
}

template <typename T>
void AverageData<T>::combine(AverageData<T> &&other) {
  denormalize();
  other.denormalize();
  Base::combine(std::move(other));
  count_ += other.count_;
}

template <typename T>
void AverageData<T>::clear() {
  Base::clear();
  count_ = 0;
  normalized_ = false;
}

template <typename T>
void AverageData<T>::normalize() {
  if (normalized_)
    return;
  Linalg::idiv(Base::data_, double(count_));
  normalized_ = true;
}

template <typename T>
void AverageData<T>::denormalize() {
  if (!normalized_)
    return;
  Linalg::imul(Base::data_, double(count_));
  normalized_ = false;
}

template <typename T>
T &AverageData<T>::value() {
  normalize();
  return Base::data_;
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
