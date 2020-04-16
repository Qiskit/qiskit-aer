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

#include <algorithm>
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

  OpSet(const optypeset_t &_optypes,
        const stringset_t &_gates,
        const stringset_t &_snapshots)
    : optypes(_optypes), gates(_gates), snapshots(_snapshots) {}
  
  OpSet(const std::vector<Op> &ops) { for (const auto& op : ops) {insert(op);} }

  //-----------------------------------------------------------------------
  // Insert operations to the OpSet
  //-----------------------------------------------------------------------

  // Add another opset to the current one
  void insert(const OpSet &_opset);

  // Add additional op to the opset
  void insert(const Op &_op);
  
  //-----------------------------------------------------------------------
  // Check if operations are in the OpSet
  //-----------------------------------------------------------------------

  // Return true if another OpSet is contained in the current OpSet
  bool contains(const OpSet &_opset) const;

  // Return true if an all operations are contained in the current OpSet
  bool contains(const std::vector<Op> &_ops) const;

  // Return true if an operation is contained in the current OpSet
  bool contains(const Op &_op) const;

  // Return true if all operation types are contained in the current OpSet
  bool contains(const optypeset_t &_optypes) const;

  // Return true if an operation type is contained in the current OpSet
  bool contains(const OpType &_optype) const;

  // Return true if all gates are contained in the current OpSet
  bool contains_gates(const stringset_t &_gates) const;

  // Return true if gate is contained in the current OpSet
  bool contains_gates(const std::string &_gate) const;

  // Return true if all snapshots are contained in the current OpSet
  bool contains_snapshots(const stringset_t &_snapshots) const;

  // Return true if snapshot is contained in the current OpSet
  bool contains_snapshots(const std::string &_snapshot) const;

  //-----------------------------------------------------------------------
  // Return set difference with another OpSet
  //-----------------------------------------------------------------------

  // Return an OpSet of all ops in another opset not contained in the OpSet
  OpSet difference(const OpSet &_opset) const;

  // Return an Opset of all ops in a vector not contained in the OpSet
  OpSet difference(const std::vector<Op> &_ops) const;

  // Return a set of all optypes in set not contained in the OpSet
  optypeset_t difference(const optypeset_t &_optypes) const;

  // Return a set of all gates in a set not contained in the OpSet
  stringset_t difference_gates(const stringset_t &_gates) const;
  
  // Return a set of all snapshots in a set not contained in the OpSet
  stringset_t difference_snapshots(const stringset_t &_snapshots) const;

};


//------------------------------------------------------------------------------
// OpSet class methods
//------------------------------------------------------------------------------

void OpSet::insert(const Op &op) {
  optypes.insert(op.type);
  if (op.type == OpType::gate)
    gates.insert(op.name);
  else if (op.type == OpType::snapshot)
    snapshots.insert(op.name);
}

void OpSet::insert(const OpSet &opset) {
  optypes.insert(opset.optypes.begin(),
                  opset.optypes.end());
  gates.insert(opset.gates.begin(),
                opset.gates.end());
  snapshots.insert(opset.snapshots.begin(),
                    opset.snapshots.end());
}

bool OpSet::contains(const OpSet &_opset) const {
  return (contains(_opset.optypes)
          && contains_gates(_opset.gates)
          && contains_snapshots(_opset.snapshots));
}

bool OpSet::contains(const Op &_op) const {
  if (contains(_op.type)) {
    if (_op.type == OpType::gate)
      return contains_gates(_op.name);
    else if (_op.type == OpType::snapshot)
      return contains_snapshots(_op.name);
    return true;
  }
  return false;
}

bool OpSet::contains(const std::vector<Op> &_ops) const {
  for (const auto& op: _ops) {
    if (!contains(op))
      return false;
  }
  return true;
}

bool OpSet::contains(const OpType &_optype) const {
  return !(optypes.find(_optype) == optypes.end());
}

bool OpSet::contains(const optypeset_t &_optypes) const {
  for (const auto &optype : _optypes) {
    if (!contains(optype))
      return false;
  }
  return true;
}

bool OpSet::contains_gates(const std::string &_gate) const {
  return !(gates.find(_gate) == gates.end());
}

bool OpSet::contains_gates(const stringset_t &_gates) const {
  for (const auto &gate : _gates) {
    if (!contains_gates(gate))
      return false;
  }
  return true;
}

bool OpSet::contains_snapshots(const std::string &_snapshot) const {
  return !(snapshots.find(_snapshot) == snapshots.end());
}

bool OpSet::contains_snapshots(const stringset_t &_snapshots) const {
  for (const auto &snapshot : _snapshots) {
    if (!contains_snapshots(snapshot))
      return false;
  }
  return true;
}

//-----------------------------------------------------------------------
// Return set difference with another OpSet
//-----------------------------------------------------------------------

// Return an OpSet of all ops in another opset not contained in the OpSet
OpSet OpSet::difference(const OpSet &_opset) const {
  OpSet ret;
  ret.optypes = difference(_opset.optypes);
  ret.gates = difference_gates(_opset.gates);
  ret.snapshots = difference_gates(_opset.snapshots);
  return ret;
}

// Return a set of all optypes in set not contained in the OpSet
OpSet::optypeset_t OpSet::difference(const optypeset_t &_optypes) const {
  optypeset_t ret;
  std::set_difference(_optypes.begin(), _optypes.end(),
                      optypes.begin(), optypes.end(),
                      std::inserter(ret, ret.begin()));
  return ret;
}

// Return a set of all gates in a set not contained in the OpSet
stringset_t OpSet::difference_gates(const stringset_t &_gates) const {
  stringset_t ret;
  std::set_difference(_gates.begin(), _gates.end(),
                      gates.begin(), gates.end(),
                      std::inserter(ret, ret.begin()));
  return ret;
}

// Return a set of all snapshots in a set not contained in the OpSet
stringset_t OpSet::difference_snapshots(const stringset_t &_snapshots) const {
  stringset_t ret;
  std::set_difference(_snapshots.begin(), _snapshots.end(),
                      snapshots.begin(), snapshots.end(),
                      std::inserter(ret, ret.begin()));
  return ret;
}

//------------------------------------------------------------------------------
} // end namespace Operations
} // end namespace AER
//------------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Ostream overload for opset
//-------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& out,
                                const AER::Operations::OpSet& opset) {
  bool first = true;
  out << "{";
  if (!opset.optypes.empty()) {
    out << "\"optypes\": " << opset.optypes;
    first = false;
  }
  if (!opset.gates.empty()) {
    if (!first)
      out << ", ";
    out << "\"gates\": " << opset.gates;
    first = false;
  }
  if (!opset.snapshots.empty()) {
    if (!first)
      out << ", ";
    out << "\"snapshots\": " << opset.snapshots;
    first = false;
  }
  out << "}";
  return out;
}

//------------------------------------------------------------------------------
#endif
