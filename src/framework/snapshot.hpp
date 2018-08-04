/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    snapshot.hpp
 * @brief   Snapshot data class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_framework_snapshot_hpp_
#define _aer_framework_snapshot_hpp_

#include "framework/types.hpp"

namespace AER {
namespace Snapshots {

//------------------------------------------------------------------------------
// Snapshot data storage class
//------------------------------------------------------------------------------

template <typename Key, typename Data, template<typename> class DataClass>
class Snapshot {

// Inner snapshot data map type
using SlotData = std::map<Key, DataClass<Data>>;

public:

  // Add a new datum to the snapshot at the specified slot and key value
  inline void add_data(std::string slot, const Key &key, const Data& datum) {
    data_[slot][key].add(datum);
  };

  // Return a const reference copy of a DataClass object for a given slot and
  // key value. If the slot or key do not exist an error will be thrown
  const DataClass<Data>& get_data(std::string slot, const Key &key) const;

  // return occupied snapshot slots
  std::set<std::string> slots() const;

  // return snapshot keys for a given slot if the slot does not exist
  // this returns an empty set (and does not initialize the slot)
  std::set<Key> slot_keys(std::string slot) const;

  // Return the inner map for a snapshot slot. If the slot does not exist
  // this will initialize a new empty map for that slot
  inline SlotData& slot(std::string slot) {return data_[slot];};

  // Combine with another snapshot object clearing each inner map
  // as it is copied, and then clearing the resulting object.
  void combine(Snapshot<Key, Data, DataClass> &snapshot) ;

  // Clear all data from current snapshot
  inline void clear() {data_.clear();};

private:

  // Internal Storage
  std::map<std::string, SlotData> data_;
};


//------------------------------------------------------------------------------
// ShotData class for storage each shot of snapshot quantities
//------------------------------------------------------------------------------

// Data class for storing snapshots of for individual shots
template<class data_t>
class ShotData {

public:
  // Return the shot data vector
  inline const std::vector<data_t>& data() const {return data_;};
  inline std::vector<data_t>& data() {return data_;};

  // Add another datum by appending it to the data vector
  inline void add(const data_t &datum) {data_.push_back(datum);};

  // Add another ShotData class by appending its data vector to the current 
  // data vector. This is done by moving the contents to the second ShotData
  // object is empty after the operation.
  void combine(ShotData<data_t> &rhs);
  
private:
  std::vector<data_t> data_;
};


//------------------------------------------------------------------------------
// AverageData class for storage of averaged quantities
//------------------------------------------------------------------------------

// Data class for storing snapshots that are averaged over each shot
// 'accum' stores the data type as an accumualted sum of each recorded shots data type
// 'count' keeps track of how many datum have been accumulated to return the average.
// the 'average' function returns the average of the data as accum / counts.
template<class data_t>
class AverageData {
public:
  
  // Return the averaged data: accum / count
  data_t data() const;

  // Add another datum by adding to accum and incrementing count by 1.
  void add(const data_t &datum);

  // Combine with another AverageData class by combining accum and count members
  // This clears the values of the combined rhs argument
  void combine(AverageData<data_t> &rhs);

private:

  data_t accum_; // stores the accumulated data for multiple datum
  uint_t count_; // stores number of datum that have been accumulatede

  // Define helper functions for averaging several sorts of data types
  // There is probably a better way to implement this...

  template <class T>
  void accum_helper(std::vector<T> &lhs, const std::vector<T> &rhs) const;

  template <class T1, class T2>
  void accum_helper(std::map<T1, T2> &lhs, const std::map<T1, T2> &rhs) const;

  template <class T>
  void accum_helper(T &lhs, const T &rhs) const;
  
  template <class T>
  inline std::vector<T> average_helper(const std::vector<T> &accum) const;

  template <class T1, class T2>
  inline std::map<T1, T2> average_helper(const std::map<T1, T2> &accum) const;

  template <class T>
  inline T average_helper(const T &accum) const;
};


//------------------------------------------------------------------------------
// Implementation: Snapshot class methods
//------------------------------------------------------------------------------

template <typename Key, typename Data, template<typename> class DataClass>
std::set<std::string> Snapshot<Key, Data, DataClass>::slots() const {
    std::set<std::string> ret;
  for (const auto &pair : data_) {
    ret.insert(pair.first);
  }
  return ret;
}

template <typename Key, typename Data, template<typename> class DataClass>
std::set<Key> Snapshot<Key, Data, DataClass>::slot_keys(std::string slot) const {
  std::set<Key> ret;
  auto it = data_.find(slot);
  if (it != data_.end()) {
    for (const auto &pair : it->second) {
      ret.insert(pair.first);
    }
  }
  return ret;
}

template <typename Key, typename Data, template<typename> class DataClass>
const DataClass<Data>&
Snapshot<Key, Data, DataClass>::get_data(std::string slot, const Key &key) const {
  auto islot = data_.find(slot);
  if (islot == data_.end()) {
    throw std::invalid_argument("Snapshot slot does not exist.");
  }
  auto ikey = islot->second.find(key);
  if (ikey == islot->second.end()) {
    throw std::invalid_argument("Snapshot key does not exist.");
  }
  return ikey->second;
}

template <typename Key, typename Data, template<typename> class DataClass>
void Snapshot<Key, Data, DataClass>::combine(Snapshot<Key, Data, DataClass> &snapshot) {
  for (auto &slot_data : snapshot.data_) {
    for (auto &key_data : slot_data.second) {
      data_[slot_data.first][key_data.first].combine(key_data.second);
    }
  }
  snapshot.clear(); // clear added snapshot
}

//------------------------------------------------------------------------------
// Implementation: ShotData class methods
//------------------------------------------------------------------------------

template <class data_t>
void ShotData<data_t>::combine(ShotData<data_t> &rhs) {
  data_.insert(data_.end(), std::make_move_iterator(rhs.data_.begin()), 
                            std::make_move_iterator(rhs.data_.end()));
  rhs.data_.clear();
}

//------------------------------------------------------------------------------
// Implementation: AverageData class methods
//------------------------------------------------------------------------------

template<class data_t>
void AverageData<data_t>::add(const data_t &datum) {
  accum_helper(accum_, datum);
  count_++;
}

template<class data_t>
void AverageData<data_t>::combine(AverageData<data_t> &rhs) {
  accum_helper(accum_, rhs.accum_);
  count_ += rhs.count_;
  rhs.accum_ = data_t();
  rhs.count_ = 0;
}
  
template<class data_t>
data_t AverageData<data_t>::data() const {
  return average_helper(accum_);
};
  
template <class data_t>
template <class T>
void AverageData<data_t>::accum_helper(std::vector<T> &lhs, const std::vector<T> &rhs) const {
  if (lhs.empty()) {
    lhs = rhs;
  } else if (!rhs.empty()) {
    if (lhs.size() != rhs.size())
      throw std::invalid_argument("Snapshots::AverageData::add (vectors are not equal.)");
    for (size_t pos = 0; pos < lhs.size(); ++ pos)
      lhs[pos] += rhs[pos];
  }
};

template <class data_t>
template <class T1, class T2>
void AverageData<data_t>::accum_helper(std::map<T1, T2> &lhs, const std::map<T1, T2> &rhs) const {
  for (const auto &pair : rhs)
    lhs[pair.first] += pair.second;
}

template <class data_t>
template <class T>
void AverageData<data_t>::accum_helper(T &lhs, const T &rhs) const {
  try {lhs += rhs;}
  catch (std::exception) {
    throw std::invalid_argument("Snapshots::AverageData::add (cannot combine data types)");
  };
}

template <class data_t>
template <class T>
inline std::vector<T> AverageData<data_t>::average_helper(const std::vector<T> &accum) const {
  double renorm = 1.0 / count_;
  std::vector<T> ret;
  ret.reserve(accum.size());
  for (const auto &elt : accum) {
    ret.push_back(elt * renorm);
  }
  return ret;
}

template <class data_t>
template <class T1, class T2>
inline std::map<T1, T2> AverageData<data_t>::average_helper(const std::map<T1, T2> &accum) const {
  double renorm = 1.0 / count_;
  std::map<T1, T2> ret;
  for (const auto &pair : accum) {
    ret[pair.first] = renorm * pair.second;
  }
  return ret;
}

template <class data_t>
template <class T>
inline T AverageData<data_t>::average_helper(const T &accum) const {
  try { 
    double renorm = 1.0 / count_;
    return renorm * accum;
  } catch (std::exception) {
    throw std::invalid_argument("Snapshots::AverageData::add (Cannot average data type)");
  };
}

//------------------------------------------------------------------------------
} // end namespace Snapshot
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
