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

#ifndef _aer_framework_results_data_map_hpp_
#define _aer_framework_results_data_map_hpp_

#include "framework/json.hpp"
#include "framework/types.hpp"

namespace AER {

// Recursive nested data template class
template <template <class> class Data, class T, size_t N = 1>
class DataMap {
public:
  // Access data
  auto &value() { return data_; }

  // The following protected functions are designed to be called via
  // a mixin class that inherits this class.

  // Add data (copy)
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == N - 1>::type>
  void add(const T &data, const std::string &key, const Args &...inner_keys);

  // Add data (move)
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == N - 1>::type>
  void add(T &&data, const std::string &key, const Args &...inner_keys);

  // Combine with another data object
  void combine(DataMap<Data, T, N> &&other);

  // copy from another data onject
  void copy(DataMap<Data, T, N> &other);

  // Clear all stored data
  void clear();

  // Convert to JSON
  json_t to_json();

  // Add to existing JSON
  void add_to_json(json_t &js);

  // Enable or disable storing datamap
  bool enabled = true;

protected:
  stringmap_t<DataMap<Data, T, N - 1>> data_;
};

// Template specialization for N=1 case
template <template <class> class Data, class T>
class DataMap<Data, T, 1> {
public:
  // Access data
  auto &value() { return data_; }

  // Add data (copy)
  void add(const T &data, const std::string &key);

  // Add data (move)
  void add(T &&data, const std::string &key);

  // Combine with another data object
  void combine(DataMap<Data, T, 1> &&other);

  // copy from another data onject
  void copy(DataMap<Data, T, 1> &other);

  // Clear all stored data
  void clear();

  // Convert to JSON
  json_t to_json();

  // Add to existing JSON
  void add_to_json(json_t &js);

  // Enable or disable storing datamap
  bool enabled = true;

protected:
  stringmap_t<Data<T>> data_;
};

//------------------------------------------------------------------------------
// Implementation N
//------------------------------------------------------------------------------

template <template <class> class Data, class T, size_t N>
template <typename... Args, typename>
void DataMap<Data, T, N>::add(const T &data, const std::string &key,
                              const Args &...inner_keys) {
  if (enabled) {
    data_[key].add(data, inner_keys...);
  }
}

template <template <class> class Data, class T, size_t N>
template <typename... Args, typename>
void DataMap<Data, T, N>::add(T &&data, const std::string &key,
                              const Args &...inner_keys) {
  if (enabled) {
    data_[key].add(std::move(data), inner_keys...);
  }
}

template <template <class> class Data, class T, size_t N>
void DataMap<Data, T, N>::combine(DataMap<Data, T, N> &&other) {
  if (enabled) {
    for (auto &pair : other.data_) {
      const auto &key = pair.first;
      // If empty we copy data without accumulating
      if (data_.find(key) == data_.end()) {
        data_[key] = std::move(pair.second);
      } else {
        data_[key].combine(std::move(pair.second));
      }
    }
  }
}

template <template <class> class Data, class T, size_t N>
void DataMap<Data, T, N>::copy(DataMap<Data, T, N> &other) {
  if (enabled) {
    for (auto &pair : other.data_) {
      const auto &key = pair.first;
      // If empty we copy data without accumulating
      if (data_.find(key) == data_.end()) {
        data_[key] = pair.second;
      } else {
        auto t = pair.second;
        data_[key].combine(std::move(t));
      }
    }
  }
}

template <template <class> class Data, class T, size_t N>
void DataMap<Data, T, N>::clear() {
  data_.clear();
}

template <template <class> class Data, class T, size_t N>
json_t DataMap<Data, T, N>::to_json() {
  json_t jsdata = json_t::object();
  if (enabled) {
    for (auto &pair : data_) {
      jsdata[pair.first] = pair.second.to_json();
    }
  }
  return jsdata;
}

template <template <class> class Data, class T, size_t N>
void DataMap<Data, T, N>::add_to_json(json_t &jsdata) {
  if (enabled) {
    for (auto &pair : data_) {
      pair.second.add_to_json(jsdata[pair.first]);
    }
  }
}

//------------------------------------------------------------------------------
// Implementation N=1 specialization
//------------------------------------------------------------------------------

template <template <class> class Data, class T>
void DataMap<Data, T, 1>::add(const T &data, const std::string &key) {
  if (enabled) {
    data_[key].add(data);
  }
}

template <template <class> class Data, class T>
void DataMap<Data, T, 1>::add(T &&data, const std::string &key) {
  if (enabled) {
    data_[key].add(std::move(data));
  }
}

template <template <class> class Data, class T>
void DataMap<Data, T, 1>::combine(DataMap<Data, T, 1> &&other) {
  if (enabled) {
    for (auto &pair : other.data_) {
      const auto &key = pair.first;
      // If empty we copy data without accumulating
      if (data_.find(pair.first) == data_.end()) {
        data_[key] = std::move(pair.second);
      } else {
        data_[key].combine(std::move(pair.second));
      }
    }
  }
}

template <template <class> class Data, class T>
void DataMap<Data, T, 1>::copy(DataMap<Data, T, 1> &other) {
  if (enabled) {
    for (auto &pair : other.data_) {
      const auto &key = pair.first;
      // If empty we copy data without accumulating
      if (data_.find(key) == data_.end()) {
        data_[key] = pair.second;
      } else {
        auto t = pair.second;
        data_[key].combine(std::move(t));
      }
    }
  }
}

template <template <class> class Data, class T>
void DataMap<Data, T, 1>::clear() {
  data_.clear();
}

template <template <class> class Data, class T>
json_t DataMap<Data, T, 1>::to_json() {
  json_t jsdata = json_t::object();
  if (enabled) {
    for (auto &pair : data_) {
      jsdata[pair.first] = pair.second.to_json();
    }
  }
  return jsdata;
}

template <template <class> class Data, class T>
void DataMap<Data, T, 1>::add_to_json(json_t &jsdata) {
  if (enabled) {
    for (auto &pair : data_) {
      jsdata[pair.first] = pair.second.to_json();
    }
  }
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
