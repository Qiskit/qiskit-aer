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

//------------------------------------------------------------------------------
// AccumulatedData class for average snapshot storage
//------------------------------------------------------------------------------

template<class Data>
class AccumulatedData {
  public:
    Data accum;
    uint_t count;

    inline void add(const Data &data) {
      accum_helper(accum, data);
      count++;
    };

    inline void add(const AccumulatedData<Data> &rhs) {
      accum_helper(accum, rhs.accum);
      count += rhs.count;
    };
  
    inline Data average() const {
      return average_helper(accum);
    };
  
  private:

    // Define helper functions for averaging several sorts of data types
    // There is probably a better way to implement this...
  
    template <class T>
    void accum_helper(std::vector<T> &lhs, const std::vector<T> &rhs) const {
      if (lhs.empty()) {
        lhs = rhs;
      } else if (!rhs.empty()) {
        if (lhs.size() != rhs.size())
          throw std::invalid_argument("AccumulatedData::add (vectors are not equal.)");
        for (size_t pos = 0; pos < lhs.size(); ++ pos)
          lhs[pos] += rhs[pos];
      }
    };

    template <class T1, class T2>
    void accum_helper(std::map<T1, T2> &lhs, const std::map<T1, T2> &rhs) const {
      for (const auto &pair : rhs)
        lhs[pair.first] += pair.second;
    };

    template <class T>
    void accum_helper(T &lhs, const T &rhs) const {
      try {lhs += rhs;}
      catch (std::exception) {
        throw std::invalid_argument("AccumulatedData::add (cannot combine data types)");
      };
    };
    
    template <class T>
    inline std::vector<T> average_helper(const std::vector<T> &_accum) const {
      double renorm = 1.0 / count;
      std::vector<T> ret;
      ret.reserve(_accum.size());
      for (const auto &elt : _accum) {
        ret.push_back(elt * renorm);
      }
      return ret;
    };

    template <class T1, class T2>
    inline std::map<T1, T2> average_helper(const std::map<T1, T2> &_accum) const {
      double renorm = 1.0 / count;
      std::map<T1, T2> ret;
      for (const auto &pair : _accum) {
        ret[pair.first] = renorm * pair.second;
      }
      return ret;
    };

    template <class T>
    inline T average_helper(const T &_accum) const {
      try { 
        double renorm = 1.0 / count;
        return renorm * _accum;
      } catch (std::exception) {
        throw std::invalid_argument("AccumulatedData::add (Cannot average data type)");
      };
    };
};


//------------------------------------------------------------------------------
// AverageSnapshot class
//------------------------------------------------------------------------------

template <class Key, class Data>
class AveragedSnapshot {

using SlotData = std::map<Key, AccumulatedData<Data>>;

public:

  // Add data to current data
  inline void add_data(std::string slot, const Key &key, const Data& dat) {
    data(slot, key).add(dat);
  };

  inline Data averaged_data(std::string slot, const Key &key) const {
    auto islot = data_.find(slot);
    if (islot == data_.end()) {
      throw std::invalid_argument("Snapshot slot does not exist.");
    }
    auto ikey = islot->second.find(key);
    if (ikey == islot->second.end()) {
      throw std::invalid_argument("Snapshot key does not exist.");
    }
    return ikey->second.average();
  };
  
  // return occupied snapshot slots
  inline std::set<std::string> slots() const {
    std::set<std::string> ret;
    for (const auto &pair : data_) {
      ret.insert(pair.first);
    }
    return ret;
  };

  // return snapshot keys for a given slot if the slot does not exist
  // this returns an empty set (and does not create the slot)
  inline std::set<Key> slot_data_keys(std::string slot) const {
    std::set<Key> ret;
    auto it = data_.find(slot);
    if (it != data_.end()) {
      for (const auto &pair : it->second) {
        ret.insert(pair.first);
      }
    }
    return ret;
  }

  // Combine with another snapshot object clearing as pieces are copied
  inline void combine(AveragedSnapshot<Key, Data> &snapshot) {
    for (auto &slot_data : snapshot.data_) {
      for (auto &key_data : slot_data.second) {
        add_data(slot_data.first, key_data.first, key_data.second);
      }
      slot_data.second = SlotData(); // reset with default constructor
    }
    snapshot = AveragedSnapshot<Key, Data>(); // cleared
  };

  // Return a pair of the Data and number of counts in the accumulation
  // If the slot/key arguments aren't currently in the data this will
  // initialize them with default values and counts = 0
  inline AccumulatedData<Data>& data(std::string slot, const Key &key) {
    return data_[slot][key];
  }

  // Creates a new member if not present
  inline SlotData& slot_data(std::string slot) {
    return data_[slot];
  };

private:

  // Internal Storage
  std::map<std::string, SlotData> data_;
};

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
