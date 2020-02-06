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

#ifndef _aer_framework_results_data_average_data_hpp_
#define _aer_framework_results_data_average_data_hpp_

#include "framework/json.hpp"
#include "framework/linalg/linalg.hpp"
#include "framework/types.hpp"

namespace AER {

template <typename T>
class AverageData {
 public:
  // Return the mean of the accumulated data:
  // mean = accum / count
  T mean() const;

  // Return the unbiased sample variance of the accumulated data:
  // var = (1 / (n - 1)) * (sum_i (data[i]^2 / n) - mean^2) for n > 1
  T variance() const;

  // Add a new datum to the snapshot at the specified key
  // Uses copy semantics
  void add_data(const T &datum, bool compute_variance = false);

  // Add a new datum to the snapshot at the specified key
  // Uses move semantics
  void add_data(T &&datum, bool compute_variance = false) noexcept;

  // Combine with another snapshot container
  // Uses copy semantics
  void combine(const AverageData<T> &other);

  // Combine with another snapshot container
  // Uses move semantics
  void combine(AverageData<T> &&other) noexcept;

  // Access data
  T &data() { return accum_; }

  // Const access data
  const T &data() const { return accum_; }

  // Get number of datum accumulated
  size_t size() const { return count_; }

  // Change the size to a new value for normalization
  // This will effect value of mean and variance
  void resize(size_t sz) { count_ = sz; }

  // Clear all stored data
  void clear();

  // Return true if data is empty
  bool empty() const { return count_ == 0; }

  // Return bool for in the container can compute variance
  bool has_variance() const { return variance_; }

 protected:
  // Accumulated data
  T accum_;

  // Store the square of accumulated data for computing sample variance.
  T accum_squared_;

  // Flag for storing accum squared for computing variance
  bool variance_ = true;

  // Number of datum that have been accumulated
  size_t count_ = 0;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
T AverageData<T>::mean() const {
  return (count_ > 1) ? Linalg::div(T(accum_), double(count_)) : accum_;
}

template <typename T>
T AverageData<T>::variance() const {
  // If no counts zero or we haven't been computing variance
  // we return default value
  if (count_ == 0 || variance_ == false) {
    return T();
  }

  // If counts is 1 variance is zero
  if (count_ == 1) {
    return Linalg::mul(accum_squared_, 0);
  }

  // var = (count / (count - 1)) * (accum_squared ** 2 / count - mean ** 2) for
  // n > 1
  T mean_squared = mean();
  Linalg::isquare(mean_squared);                // mean_squared
  T var = Linalg::div(accum_squared_, double(count_));  // squared mean
  Linalg::isub(var, mean_squared);              // squared mean - mean squared
  // Apply sample bias correction of counts / (counts - 1)
  Linalg::idiv(var, double(count_) / double(count_ - 1));
  return var;
}

template <typename T>
void AverageData<T>::combine(const AverageData<T> &other) {
  // If empty we copy data without accumulating
  if (empty()) {
    count_ = other.count_;
    accum_ = other.accum_;
    variance_ = other.has_variance();
    if (variance_) {
      accum_squared_ = other.accum_squared_;
    }
  } else {
    // Otherwise we accumulate
    count_ += other.count_;
    Linalg::iadd(accum_, other.accum_);
    // If either container doesn't have variance we disable variance
    // of the current container
    variance_ &= other.has_variance();
    if (variance_) {
      Linalg::iadd(accum_squared_, other.accum_squared_);
    }
  }
}

template <typename T>
void AverageData<T>::combine(AverageData<T> &&other) noexcept {
  // If empty we copy data without accumulating
  if (empty()) {
    count_ = other.count_;
    accum_ = std::move(other.accum_);
    variance_ = other.has_variance();
    if (variance_) {
      accum_squared_ = std::move(other.accum_squared_);
    }
  } else {
    // Otherwise we accumulate
    count_ += other.count_;
    Linalg::iadd(accum_, std::move(other.accum_));
    // If either container doesn't have variance we disable variance
    // of the current container
    variance_ &= other.has_variance();
    if (variance_) {
      Linalg::iadd(accum_squared_, std::move(other.accum_squared_));
    }
  }
  // Now that we have moved we clear the other to initial state.
  other.clear();
}

template <typename T>
void AverageData<T>::clear() {
  // Clear stored data using default constructor of data type
  accum_ = T();
  accum_squared_ = T();
  count_ = 0;
  variance_ = true;
}

template <typename T>
void AverageData<T>::add_data(const T &datum, bool compute_variance) {
  // If we add a single datum without variance we
  // disable variance for the container
  variance_ &= compute_variance;

  // For initial datum we set the accumulators to the data
  if (count_ == 0) {
    accum_ = datum;
    if (has_variance()) {
      accum_squared_ = Linalg::square(accum_);
    }
  } else {
    // We use Linalg library to accumulate standard types
    Linalg::iadd(accum_, datum);
    if (has_variance()) {
      Linalg::iadd(accum_squared_, Linalg::square(datum));
    }
  }
  // Increment the count size of the accumulated data
  count_ += 1;
}

template <typename T>
void AverageData<T>::add_data(T &&datum, bool compute_variance) noexcept {
  variance_ &= compute_variance;
  if (count_ == 0) {
    accum_ = std::move(datum);
    if (has_variance()) {
      accum_squared_ = Linalg::square(accum_);
    }
  } else {
    Linalg::iadd(accum_, datum);
    if (has_variance()) {
      T tmp = std::move(datum);
      Linalg::iadd(accum_squared_, Linalg::isquare(tmp));
    }
  }
  count_ += 1;
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
template <typename T>
void to_json(json_t &js, const AverageData<T> &data) {
  js = json_t::object();
  js["value"] = data.mean();
  if (data.has_variance()) {
    js["variance"] = data.variance();
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
