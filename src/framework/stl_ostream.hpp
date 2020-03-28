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

#include <unordered_map>
#include <unordered_set>
#include <vector>

//=============================================================================
// STL container ostream overloads
//
// This includes overloads for:
// * std::par
// * std::vector
// * std::array
// * std::map
// * std::unordered_map
// * std::set
// * std::unordered_set
//=============================================================================

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &out, const std::pair<T1, T2> &p);

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v);

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &v);

template <typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &out, const std::map<T1, T2, T3> &m);

template <typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &out,
                         const std::unordered_map<T1, T2, T3> &m);

template <typename T1>
std::ostream &operator<<(std::ostream &out, const std::unordered_set<T1> &s);

template <typename T1>
std::ostream &operator<<(std::ostream &out, const std::set<T1> &s);

//=============================================================================
// Implementations
//=============================================================================

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &out, const std::pair<T1, T2> &p) {
  out << "(" << p.first << ", " << p.second << ")";
  return out;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << "[";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &v) {
  out << "[";
  for (size_t i = 0; i < N; ++i) {
    out << v[i];
    if (i != N - 1)
      out << ", ";
  }
  out << "]";
  return out;
}

template <typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &out, const std::map<T1, T2, T3> &m) {
  out << "{";
  size_t pos = 0, last = m.size() - 1;
  for (auto const &p : m) {
    out << p.first << ":" << p.second;
    if (pos != last)
      out << ", ";
    pos++;
  }
  out << "}";
  return out;
}

template <typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &out,
                         const std::unordered_map<T1, T2, T3> &m) {
  out << "{";
  size_t pos = 0, last = m.size() - 1;
  for (auto const &p : m) {
    out << p.first << ":" << p.second;
    if (pos != last)
      out << ", ";
    pos++;
  }
  out << "}";
  return out;
}

template <typename T1>
std::ostream &operator<<(std::ostream &out, const std::unordered_set<T1> &s) {
  out << "{";
  size_t pos = 0, last = s.size() - 1;
  for (auto const &elt : s) {
    out << elt;
    if (pos != last)
      out << ", ";
    pos++;
  }
  out << "}";
  return out;
}

template <typename T1>
std::ostream &operator<<(std::ostream &out, const std::set<T1> &s) {
  out << "{";
  size_t pos = 0, last = s.size() - 1;
  for (auto const &elt : s) {
    out << elt;
    if (pos != last)
      out << ", ";
    pos++;
  }
  out << "}";
  return out;
}

//-----------------------------------------------------------------------------
#endif
