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

#ifndef _aer_framework_results_data_subtypes_list_hpp_
#define _aer_framework_results_data_subtypes_list_hpp_

#include <vector>

#include "framework/results/data/subtypes/single_data.hpp"

namespace AER {

template <typename T>
class ListData : public SingleData<std::vector<T>> {
using Base = SingleData<std::vector<T>>;
public:
  // Add data (copy)
  void add(const T& data);

  // Add data (move)
  void add(T&& data);

  // Add data
  void combine(ListData<T>&& other);

  // Combine with another data object
  void clear();
};


//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
void ListData<T>::add(const T& data) {
  Base::data_.push_back(data);
}

template <typename T>
void ListData<T>::add(T&& data) {
  Base::data_.push_back(std::move(data));
}

template <typename T>
void ListData<T>::combine(ListData<T>&& other) {
  Base::data_.insert(Base::data_.end(),
                     std::make_move_iterator(other.data_.begin()),
                     std::make_move_iterator(other.data_.end()));
}

template <typename T>
void ListData<T>::clear() {
  Base::data_.clear();
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
