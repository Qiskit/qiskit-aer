/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_framework_snapshot_hpp_
#define _aer_framework_snapshot_hpp_

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
  
  // Return the averaged data: accum / count
  json_t data() const;

  // Add another datum by adding to accum and incrementing count by 1.
  void add(json_t &datum);

  // Combine with another AverageData class by combining accum and count members
  // This clears the values of the combined rhs argument
  void combine(AverageData &rhs);

private:

  json_t accum_; // stores the accumulated data for multiple datum
  uint_t count_; // stores number of datum that have been accumulatede

  // Adds the rhs JSON to the lhs JSON: lhs = lhs + rhs
  static void accum_helper(json_t &lhs, json_t &rhs);
  
  // Average the input json in place: accum = accum / count
  static void average_helper(json_t &accum, uint_t count);

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
                       T &datum) {
    json_t tmp = datum;
    data_[key][memory].add(datum);
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

private:

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
      datum["memory"] = inner_pair.first;
      datum["value"] = inner_pair.second.data();
      result[outer_pair.first].push_back(datum);
    }
  }
  return result;
}


//------------------------------------------------------------------------------
// Implementation: AverageData class methods
//------------------------------------------------------------------------------

void AverageData::add(json_t &datum) {
  accum_helper(accum_, datum);
  count_++;
}


void AverageData::combine(AverageData &rhs) {
  accum_helper(accum_, rhs.accum_);
  count_ += rhs.count_;
  // zero rhs data
  rhs.accum_ = json_t();
  rhs.count_ = 0;
}
  

json_t AverageData::data() const {
  json_t result = accum_;
  average_helper(result, count_);
  return result;
}
  

void AverageData::accum_helper(json_t &lhs, json_t &rhs) {
  if (lhs.is_null()) {
    lhs = rhs;
  } else if (lhs.is_number() && rhs.is_number()) {
    lhs = double(lhs) + double(rhs);
  } else if (lhs.is_array() && rhs.is_array() && lhs.size() == rhs.size()) {
    for (size_t pos = 0; pos < lhs.size(); pos++)
      accum_helper(lhs[pos], rhs[pos]);
  } else if (lhs.is_object() && rhs.is_object()) {
    for (auto it = rhs.begin(); it != rhs.end(); ++it)
      accum_helper(lhs[it.key()], it.value());
  } else {
    throw std::invalid_argument("Input JSON data cannot be accumulated.");
  }
}


void AverageData::average_helper(json_t &js, uint_t count) {
  if (js.is_number())
    js = double(js) / count;
  else if (js.is_array()) {
    for (auto &item : js)
      average_helper(item, count);
  } else if (js.is_object()) {
    for (auto it = js.begin(); it != js.end(); ++it)
      average_helper(it.value(), count);
  } else
    throw std::invalid_argument("Input JSON data type cannot be averaged.");
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
