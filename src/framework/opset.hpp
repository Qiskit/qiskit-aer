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
  optypeset_t optypes;    // A set of op types
  stringset_t gates;      // A set of names for OpType::gates
  stringset_t snapshots;  // set of types for OpType::snapshot

  OpSet() = default;
  OpSet(const optypeset_t & _optypes,
        const stringset_t &_gates,
        const stringset_t &_snapshots)
    : optypes(_optypes), gates(_gates), snapshots(_snapshots) {};
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

  // Return true if all instructions in an OpSet are contained in
  // the current OpSet
  bool contains(const OpSet &opset) const;

  // Return true if all optypes are contained in the current Opset
  bool contains_optypes(const optypeset_t &_optypes) const;

  // Return true if all gates are contained in the current OpSet
  bool contains_gates(const stringset_t &_gates) const;

  // Return true if all snapshots are contained in the current OpSet
  bool contains_snapshots(const stringset_t &_snapshots) const;

  // Return true if an optype is contained in the current OpSet
  bool contains_optype(const OpType &optype) const;

  // Return true if a gate is contained in the current OpSet
  bool contains_gate(const std::string &gate) const;

  // Return true if a snapshot is contained in the current OpSet
  bool contains_snapshot(const std::string &snapshot) const;

  //-----------------------------------------------------------------------
  // Validate OpSet against sets of allowed operations
  //-----------------------------------------------------------------------

  // Return true if all instructions in an OpSet are contained in
  // the current OpSet
  bool validate(const OpSet &opset) const;

  // Return true if all instructions are contained in the current OpSet
  bool validate(const optypeset_t &allowed_ops,
                const stringset_t &allowed_gates,
                const stringset_t &allowed_snapshots) const;

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

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Insertion
//-------------------------------------------------------------------------

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

//-------------------------------------------------------------------------
// Contains
//-------------------------------------------------------------------------

bool OpSet::contains(const OpSet &opset) const {
  return contains_optypes(opset.optypes) &&
         contains_gates(opset.gates) &&
         contains_snapshots(opset.snapshots);
}

bool OpSet::contains_optypes(const optypeset_t &_optypes) const {
  for (const auto &optype : _optypes) {
    if (optypes.find(optype) == optypes.end())
      return false;
  }
  return true;
}

bool OpSet::contains_gates(const stringset_t &_gates) const {
  for (const auto &gate : _gates) {
    if (gates.find(gate) == gates.end())
      return false;
  }
  return true;
}


bool OpSet::contains_snapshots(const stringset_t &_snapshots) const {
  for (const auto &snapshot : _snapshots) {
    if (snapshots.find(snapshot) == snapshots.end())
      return false;
  }
  return true;
}

bool OpSet::contains_optype(const OpType &optype) const {
  if (optypes.find(optype) == optypes.end())
    return false;
  return true;
}

bool OpSet::contains_gate(const std::string &gate) const {
  if (gates.find(gate) == gates.end())
    return false;
  return true;
}

bool OpSet::contains_snapshot(const std::string &snapshot) const {
  if (snapshots.find(snapshot) == snapshots.end())
    return false;
  return true;
}

//-------------------------------------------------------------------------
// Validate
//-------------------------------------------------------------------------

bool OpSet::validate(const OpSet &opset) const {
  return validate_optypes(opset.optypes) &&
         validate_gates(opset.gates) &&
         validate_snapshots(opset.snapshots);
}

bool OpSet::validate(const optypeset_t &allowed_ops,
                     const stringset_t &allowed_gates,
                     const stringset_t &allowed_snapshots) const {
  return validate_optypes(allowed_ops) &&
         validate_gates(allowed_gates) &&
         validate_snapshots(allowed_snapshots);
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
