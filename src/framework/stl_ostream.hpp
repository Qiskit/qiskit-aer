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

#ifndef _aer_framework_stl_ostream_hpp_
#define _aer_framework_stl_ostream_hpp_

#include <array>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//=============================================================================
// STL container ostream overloads
//
// This includes overloads for:
// * std::vector          v => "[v[0], ..., v[N-1]]"
// * std::array           a => "[a[0], ..., a[N-1]]"
// * std::set             s => "{s[0], ..., s[N-1]}"
// * std::unordered_set   s => "{s[0], ..., s[N-1]}"
// * std::map             m => "{m[0].first: m[0].second, ..., m[N-1].first: m[N-1].second}"
// * std::unordered_map   m => "{m[0].first: m[0].second, ..., m[N-1].first: m[N-1].second}"
// * std::pair            p => "p.first:p.second"
//=============================================================================

template <class T, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::vector<T, Allocator> &v);

template <class T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &a);

template <class Key, class Compare, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::set<Key, Compare, Allocator> &s);

template <class Key, class Hash, class KeyEqual, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::unordered_set<Key, Hash, KeyEqual, Allocator> &s);

template <class Key, class T, class Compare, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::map<Key, T, Compare, Allocator> &m);

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::unordered_map<Key, T, Hash, KeyEqual, Allocator> &m);

//=============================================================================
// Implementations
//=============================================================================
namespace { // private namespace

  // Private overload for std::pair used by map and unordered_map
  template <class T1, class T2>
  std::ostream &operator<<(std::ostream &out, const std::pair<T1, T2> &p) {
    out << p.first << ":" << p.second;
    return out;
  };

  // Helper function for printing containers
  template <typename container_t>
  std::ostream& container_to_stream(std::ostream& out,
                                    const container_t& container,
                                    const std::string &delim_left,
                                    const std::string &delim_right,
                                    const std::string &seperator = ", "){
      out << delim_left;
      size_t pos = 0, last = container.size() - 1;
      for (auto const &p : container) {
          out << p;
          if (pos != last)
              out << seperator;
          pos++;
      }
      out << delim_right;
      return out;
  };
}

template <class T, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::vector<T, Allocator> &v) {
  return container_to_stream(out, v, "[", "]");
}

template <class T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &a) {
  return container_to_stream(out, a, "[", "]");
}

template <class Key, class Compare, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::set<Key, Compare, Allocator> &s) {
  return container_to_stream(out, s, "{", "}");
}

template <class Key, class Hash, class KeyEqual, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::unordered_set<Key, Hash, KeyEqual, Allocator> &s) {
  return container_to_stream(out, s, "{", "}");
}

template <class Key, class T, class Compare, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::map<Key, T, Compare, Allocator> &m) {
  return container_to_stream(out, m, "{", "}");
}

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
std::ostream &operator<<(std::ostream &out, const std::unordered_map<Key, T, Hash, KeyEqual, Allocator> &m) {
  return container_to_stream(out, m, "{", "}");
}

//-----------------------------------------------------------------------------
#endif
