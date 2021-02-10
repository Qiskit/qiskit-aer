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

#ifndef _aer_framework_results_data_subtypes_accum_hpp_
#define _aer_framework_results_data_subtypes_accum_hpp_

#include "framework/linalg/linalg.hpp"
#include "framework/results/data/subtypes/single_data.hpp"

namespace AER {

template <typename T>
class AccumData : public SingleData<T> {
using Base = SingleData<T>;
public:
  // Add data (copy)
  void add(const T& data);

  // Add data (move)
  void add(T&& data);

  // Combine data (move)
  void combine(AccumData<T>&& other);

  // Clear all stored data
  void clear();

protected:
  bool empty_ = true;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
void AccumData<T>::add(const T& data) {
  if (empty_) {
    Base::data_ = data;
    empty_ = false;
  } else {
    Linalg::iadd(Base::data_, data);
  }
}

template <typename T>
void AccumData<T>::add(T&& data) {
  if (empty_) {
    Base::data_ = std::move(data);
    empty_ = false;
  } else {
    Linalg::iadd(Base::data_, std::move(data));
  }
}

template <typename T>
void AccumData<T>::combine(AccumData<T>&& other) {
  add(std::move(other.data_));
}

template <typename T>
void AccumData<T>::clear() {
  Base::clear();
  empty_ = true;
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
