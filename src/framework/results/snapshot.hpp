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

#ifndef _aer_framework_results_snapshot_hpp_
#define _aer_framework_results_snapshot_hpp_

#include "framework/types.hpp"

namespace AER {

//------------------------------------------------------------------------------
// Snapshot data storage class
//------------------------------------------------------------------------------

class SingleShotSnapshot {

// Inner snapshot data map type

public:

  // Add a new datum to the snapshot at the specified key
  // This uses the `to_json` function for convertion
  template <typename T>
  inline void add_data(const std::string &key, T &datum) {
    json_t tmp = datum;
    data_[key].push_back(tmp);
  }

  // Combine with another snapshot object clearing each inner map
  // as it is copied, and then clearing the resulting object.
  void combine(SingleShotSnapshot &snapshot);

  // Clear all data from current snapshot
  inline void clear() {data_.clear();}

  // Clear all snapshot data for a given label
  inline void erase(const std::string &label) {data_.erase(label);}

  // Dump all snapshots to JSON;
  json_t json() const;

  // Return true if snapshot container is empty
  inline bool empty() const {return data_.empty();}

private:

  // Internal Storage
  // Map key is the snapshot label string
  stringmap_t<std::vector<json_t>> data_;
};


//------------------------------------------------------------------------------
// AverageData class for storage of averaged quantities
//------------------------------------------------------------------------------

// Data class for storing snapshots that are averaged over each shot
// 'accum' stores the data type as an accumualted sum of each recorded shots data type
// 'count' keeps track of how many datum have been accumulated to return the average.
// the 'average' function returns the average of the data as accum / counts.
class AverageData {
public:

  // Return the mean of the accumulated data:
  // mean = accum / count
  inline json_t mean() const {return (count_ > 1) ? divide_helper(accum_, count_)
                                                  : accum_;}

  // Return the unbiased sample variance of the accumulated data:
  // var = (1 / (n - 1)) * (sum_i (data[i]^2 / n) - mean^2) for n > 1
  json_t variance() const;

  // Add another datum by adding to accum and incrementing count by 1.
  // If variance is set to true the square of the datum will also be accumulated
  // and can be used to compute the sample variance.
  void add(json_t &datum, bool variance = false);

  // Combine with another AverageData class by combining accum and count members
  // This clears the values of the combined rhs argument
  void combine(AverageData &rhs);

protected:

  json_t accum_; // stores the accumulated data for multiple datum
  json_t accum_squared_; // store the square of accumulated data for computing sample variance.
  uint_t count_ = 0; // stores number of datum that have been accumulatede

  // Recursively adds the rhs JSON to the lhs JSON.
  // if subtract is false: lhs = lhs + rhs
  // if subtract is true: lhs = lhs - rhs
  static void accum_helper(json_t &lhs, json_t &rhs, bool subtract = false);
  
  // Recursively squares a json object
  static json_t square_helper(const json_t &data);

  // Recursively divide a json object
  static json_t divide_helper(const json_t &data, double val);

};


//------------------------------------------------------------------------------
// Snapshot data storage class
//------------------------------------------------------------------------------

class AverageSnapshot {

// Inner snapshot data map type

public:

  // Add a new datum to the snapshot at the specified key
  // This uses the `to_json` function for convertion
  template <typename T>
  inline void add_data(const std::string &key,
                       const std::string &memory,
                       T &datum,
                       bool variance = false) {
    json_t tmp = datum;
    data_[key][memory].add(datum, variance);
  }

  // Combine with another snapshot object clearing each inner map
  // as it is copied, and then clearing the resulting object.
  void combine(AverageSnapshot &snapshot);

  // Clear all data from current snapshot
  inline void clear() {data_.clear();}

  // Clear all snapshot data for a given label
  inline void erase(const std::string &label) {data_.erase(label);}

  // Dump all snapshots to JSON;
  json_t json() const;

  // Return true if snapshot container is empty
  inline bool empty() const {return data_.empty();}

protected:

  // Internal Storage
  // Outer map key is the snapshot label string
  // Inner map key is the memory value string
  stringmap_t<stringmap_t<AverageData>> data_;
};


//------------------------------------------------------------------------------
// Implementation: SingleShotSnapshot class methods
//------------------------------------------------------------------------------

void SingleShotSnapshot::combine(SingleShotSnapshot &snapshot) {
  for (auto &data : snapshot.data_) {
    auto &slot = data_[data.first];
    auto &new_data = data.second;
    slot.insert(slot.end(), std::make_move_iterator(new_data.begin()), 
                            std::make_move_iterator(new_data.end()));
    new_data.clear();
  }
  snapshot.clear(); // clear added snapshot
}


json_t SingleShotSnapshot::json() const {
  json_t result;
  for (const auto &pair : data_) {
    result[pair.first] = pair.second;
  }
  return result;
}


//------------------------------------------------------------------------------
// Implementation: AverageSnapshot class methods
//------------------------------------------------------------------------------

void AverageSnapshot::combine(AverageSnapshot &snapshot) {
  for (auto &data : snapshot.data_) {
    for (auto &ave_data : data.second) {
      data_[data.first][ave_data.first].combine(ave_data.second);
    }
  }
  snapshot.clear(); // clear added snapshot
}


json_t AverageSnapshot::json() const {
  json_t result;
  for (const auto &outer_pair : data_) {
    for (const auto &inner_pair : outer_pair.second) {
      json_t datum;
      // Add mean value of the snapshot
      datum["value"] = inner_pair.second.mean();
      // Add conditional memory if creg is present
      auto memory = inner_pair.first;
      if (memory.empty() == false)
        datum["memory"] = inner_pair.first;
      // Add variance if it was computed
      json_t variance = inner_pair.second.variance();
      if (variance.is_null() == false)
        datum["variance"] = variance;
      // Add to list of output
      result[outer_pair.first].push_back(datum);
    }
  }
  return result;
}


//------------------------------------------------------------------------------
// Implementation: AverageData class methods
//------------------------------------------------------------------------------


void AverageData::add(json_t &datum, bool variance) {
  count_ += 1;
  accum_helper(accum_, datum);
  if (variance) {
    json_t squared = square_helper(datum);
    accum_helper(accum_squared_, squared);
  }
}


void AverageData::combine(AverageData &rhs) {
  accum_helper(accum_, rhs.accum_);
  accum_helper(accum_squared_, rhs.accum_squared_);
  count_ += rhs.count_;
  // zero rhs data
  rhs.accum_ = json_t();
  rhs.accum_squared_ = json_t();
  rhs.count_ = 0;
}


json_t AverageData::variance() const {
  if (count_ == 0 || count_ == 1 ||
      accum_squared_.size() != accum_.size())
    return nullptr;

  json_t mean_squared = square_helper(mean());
  json_t result = divide_helper(accum_squared_, count_); // Squared mean
  // subtract mean_squared from squared_mean / counts
  accum_helper(result, mean_squared, true);
  // Apply sample bias correction of 1 / (counts - 1)
  return (count_ > 1) ? divide_helper(result, count_ - 1.0)
                      : result;
}


json_t AverageData::square_helper(const json_t &data) {
  json_t squared;
  if (data.is_number()) {
    double tmp = data;
    squared = tmp * tmp;
  } else if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++)
      squared.push_back(square_helper(data[pos]));
  } else if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it)
      squared[it.key()] = square_helper(it.value());
  } else {
    throw std::invalid_argument("Input JSON data cannot be squared.");
  }
  return squared;
}


json_t AverageData::divide_helper(const json_t &data, double val) {
  json_t mult;
  if (data.is_number()) {
    double tmp = data;
    tmp /= val;
    mult = tmp;
  } else if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++)
      mult.push_back(divide_helper(data[pos], val));
  } else if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it)
      mult[it.key()] = divide_helper(it.value(), val);
  } else {
    throw std::invalid_argument("Input JSON data cannot be multiplied.");
  }
  return mult;
}


void AverageData::accum_helper(json_t &lhs, json_t &rhs, bool subtract) {
  if (lhs.is_null()) {
    lhs = rhs;
  } else if (lhs.is_number() && rhs.is_number()) {
    if (subtract)
      lhs = double(lhs) - double(rhs);
    else
      lhs = double(lhs) + double(rhs);
  } else if (lhs.is_array() && rhs.is_array() && lhs.size() == rhs.size()) {
    for (size_t pos = 0; pos < lhs.size(); pos++)
      accum_helper(lhs[pos], rhs[pos], subtract);
  } else if (lhs.is_object() && rhs.is_object()) {
    for (auto it = rhs.begin(); it != rhs.end(); ++it)
      accum_helper(lhs[it.key()], it.value(), subtract);
  } else {
    throw std::invalid_argument("Input JSON data cannot be accumulated.");
  }
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
