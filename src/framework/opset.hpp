/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_opset_hpp_
#define _aer_framework_opset_hpp_

#include "framework/operations.hpp"

namespace AER {
namespace Operations {

//=========================================================================
// OpSet Class
//=========================================================================

// This class is used to store type information about a set of operations.
class OpSet {
private:
  // Hash function so that we can use an enum class as a std::unordered_set
  // key on older C++11 compilers like GCC 5.
  struct EnumClassHash {
    template <typename T> size_t operator()(T t) const {
      return static_cast<size_t>(t);
    }
  };

public:
  // Alias for set of OpTypes
  using optypeset_t = std::unordered_set<Operations::OpType, EnumClassHash>;

  // Public data members
  optypeset_t optypes;     // A set of op types
  stringset_t gates;      // A set of names for OpType::gates
  stringset_t snapshots;  // set of types for OpType::snapshot

  OpSet() = default;
  OpSet(const std::vector<Op> &ops) {insert(ops);}

  //-----------------------------------------------------------------------
  // Insert operations to the OpSet
  //-----------------------------------------------------------------------

  // Add another opset to the current one
  void insert(const OpSet& opset);

  // Add additional op to the opset
  void insert(const Op &op);
  
  // Add additional ops to the opset
  void insert(const std::vector<Op> &ops);

  //-----------------------------------------------------------------------
  // Check if operations are in the OpSet
  //-----------------------------------------------------------------------

  // Return true if an operation is contained in the current OpSet
  bool contains(const OpType &optype) const;

  //-----------------------------------------------------------------------
  // Validate OpSet against sets of allowed operations
  //-----------------------------------------------------------------------

  // Return True if opset ops are all contained in the other opset.
  bool validate(const OpSet &other_opset) const;

  // Return True if opset ops are contained in allowed_ops
  bool validate_optypes(const optypeset_t &allowed_ops) const;

  // Return True if opset gates are contained in allowed_gate
  bool validate_gates(const stringset_t &allowed_gates) const;

  // Return True if opset snapshots are contained in allowed_snapshots
  bool validate_snapshots(const stringset_t &allowed_snapshots) const;

  //-----------------------------------------------------------------------
  // Return OpSet operations invalid for a set of allowed operations
  //-----------------------------------------------------------------------

  // Return a set of all invalid circuit op names
  optypeset_t invalid_optypes(const optypeset_t &allowed_ops) const;

  // Return a set of all invalid circuit op names
  stringset_t invalid_gates(const stringset_t &allowed_gates) const;
  
  // Return a set of all invalid circuit op names
  stringset_t invalid_snapshots(const stringset_t &allowed_snapshots) const;
};

inline std::ostream& operator<<(std::ostream& s, const OpSet& opset) {
  s << "optypes={";
  bool first = true;
  for (OpType optype: opset.optypes) {
    if (first)
      first = false;
    else
      s << ",";
    s << optype;
  }
  s << "}, gates={";
  first = true;
  for (const std::string& gate: opset.gates) {
    if (first)
      first = false;
    else
      s << ",";
    s << gate;
  }
  s << "}, snapshots={";
  first = true;
  for (const std::string& snapshot: opset.snapshots) {
    if (first)
      first = false;
    else
      s << ",";
    s << snapshot;
  }
  s << "}";
  return s;
}

//------------------------------------------------------------------------------
// OpSet class methods
//------------------------------------------------------------------------------

void OpSet::insert(const Op &op) {
  optypes.insert(op.type);
  if (op.type == OpType::gate)
    gates.insert(op.name);
  if (op.type == OpType::snapshot)
    snapshots.insert(op.name);
}

void OpSet::insert(const std::vector<Op> &ops) {
  for (const auto &op : ops)
    insert(op);
}


void OpSet::insert(const OpSet &opset) {
  optypes.insert(opset.optypes.begin(),
                  opset.optypes.end());
  gates.insert(opset.gates.begin(),
                opset.gates.end());
  snapshots.insert(opset.snapshots.begin(),
                    opset.snapshots.end());
}

bool OpSet::contains(const OpType &optype) const {
  if (optypes.find(optype) == optypes.end())
    return false;
  return true;
}

bool OpSet::validate(const OpSet &other_opset) const {
  return validate_optypes(other_opset.optypes) &&
         validate_gates(other_opset.gates) &&
         validate_snapshots(other_opset.snapshots);
}

bool OpSet::validate_optypes(const optypeset_t &allowed_ops) const {
  for (const auto &op : optypes) {
    if (allowed_ops.find(op) == allowed_ops.end())
      return false;
  }
  return true;
}

bool OpSet::validate_gates(const stringset_t &allowed_gates) const {
  for (const auto &gate : gates) {
    if (allowed_gates.find(gate) == allowed_gates.end())
      return false;
  }
  return true;
}

bool OpSet::validate_snapshots(const stringset_t &allowed_snapshots) const {
  for (const auto &snap : snapshots) {
    if (allowed_snapshots.find(snap) == allowed_snapshots.end())
      return false;
  }
  return true;
}

// Return a set of all invalid circuit op names
OpSet::optypeset_t OpSet::invalid_optypes(const optypeset_t &allowed_ops) const {
  optypeset_t invalid;
  for (const auto &op : optypes) {
    if (allowed_ops.find(op) == allowed_ops.end())
      invalid.insert(op);
  }
  return invalid;                    
}

stringset_t OpSet::invalid_gates(const stringset_t &allowed_gates) const {
  stringset_t invalid;
  for (const auto &gate : gates) {
     if (allowed_gates.find(gate) == allowed_gates.end())
      invalid.insert(gate);
  }
  return invalid;
}

stringset_t OpSet::invalid_snapshots(const stringset_t &allowed_snapshots) const {
  stringset_t invalid;
  for (const auto &snap : snapshots) {
     if (allowed_snapshots.find(snap) == allowed_snapshots.end())
      invalid.insert(snap);
  }
  return invalid;
}

//------------------------------------------------------------------------------
} // end namespace Operations
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
