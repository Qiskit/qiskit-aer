/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _indexing_hpp_
#define _indexing_hpp_

#include <array>
#include <vector>
#include <cstdint>
#include <vector>  // teset
#include <complex> // test
#include <functional> // test

// This should be merged with the QubitVector class

namespace Indexing {

using uint_t = uint64_t;
using int_t = int64_t;

namespace Qubit {

/*
# Auto generate these values with following python snippet
import json

def cpp_init_list(int_lst):
    ret = json.dumps([str(i) + 'ULL' for i in int_lst])
    return ret.replace('"', '').replace('[','{{').replace(']','}};')

print('const std::array<uint_t, 64> BITS = ' + cpp_init_list([(1 << i) for i in range(64)]) + '\n')
print('const std::array<uint_t, 64> MASKS = ' + cpp_init_list([(1 << i) - 1 for i in range(64)]))
*/

const std::array<uint_t, 64> BITS {{
  1ULL, 2ULL, 4ULL, 8ULL,
  16ULL, 32ULL, 64ULL, 128ULL,
  256ULL, 512ULL, 1024ULL, 2048ULL,
  4096ULL, 8192ULL, 16384ULL, 32768ULL,
  65536ULL, 131072ULL, 262144ULL, 524288ULL,
  1048576ULL, 2097152ULL, 4194304ULL, 8388608ULL,
  16777216ULL, 33554432ULL, 67108864ULL, 134217728ULL,
  268435456ULL, 536870912ULL, 1073741824ULL, 2147483648ULL,
  4294967296ULL, 8589934592ULL, 17179869184ULL, 34359738368ULL, 
  68719476736ULL, 137438953472ULL, 274877906944ULL, 549755813888ULL,
  1099511627776ULL, 2199023255552ULL, 4398046511104ULL, 8796093022208ULL,
  17592186044416ULL, 35184372088832ULL, 70368744177664ULL, 140737488355328ULL, 
  281474976710656ULL, 562949953421312ULL, 1125899906842624ULL, 2251799813685248ULL,
  4503599627370496ULL, 9007199254740992ULL, 18014398509481984ULL, 36028797018963968ULL,
  72057594037927936ULL, 144115188075855872ULL, 288230376151711744ULL, 576460752303423488ULL,
  1152921504606846976ULL, 2305843009213693952ULL, 4611686018427387904ULL, 9223372036854775808ULL
}};


const std::array<uint_t, 64> MASKS {{
  0ULL, 1ULL, 3ULL, 7ULL,
  15ULL, 31ULL, 63ULL, 127ULL,
  255ULL, 511ULL, 1023ULL, 2047ULL,
  4095ULL, 8191ULL, 16383ULL, 32767ULL,
  65535ULL, 131071ULL, 262143ULL, 524287ULL,
  1048575ULL, 2097151ULL, 4194303ULL, 8388607ULL,
  16777215ULL, 33554431ULL, 67108863ULL, 134217727ULL,
  268435455ULL, 536870911ULL, 1073741823ULL, 2147483647ULL,
  4294967295ULL, 8589934591ULL, 17179869183ULL, 34359738367ULL,
  68719476735ULL, 137438953471ULL, 274877906943ULL, 549755813887ULL,
  1099511627775ULL, 2199023255551ULL, 4398046511103ULL, 8796093022207ULL,
  17592186044415ULL, 35184372088831ULL, 70368744177663ULL, 140737488355327ULL,
  281474976710655ULL, 562949953421311ULL, 1125899906842623ULL, 2251799813685247ULL,
  4503599627370495ULL, 9007199254740991ULL, 18014398509481983ULL, 36028797018963967ULL,
  72057594037927935ULL, 144115188075855871ULL, 288230376151711743ULL, 576460752303423487ULL,
  1152921504606846975ULL, 2305843009213693951ULL, 4611686018427387903ULL, 9223372036854775807ULL
}};


//------------------------------------------------------------------------------
// Function Definitions
//------------------------------------------------------------------------------

// Return as an int an N qubit bitstring for M-N qubit bit k with 0s inserted
// for N qubits at the locations specified by qubits_sorted.
// qubits_sorted must be sorted lowest to highest. Eg. {0, 1}.
template <size_t N>
uint_t index0(const std::array<uint_t, N> &qubits_sorted, const uint_t k);


// Return a vector of of 2^N in ints, where each int corresponds to an N qubit
// bitstring for M-N qubit bits in state k, and the specified N qubits in states
// [0, ..., 2^N - 1] qubits_sorted must be sorted lowest to highest. Eg. {0, 1}.
// qubits specifies the location of the qubits in the retured strings.
template <size_t N>
std::array<uint_t, 1ULL << N> indexes(const std::array<uint_t, N> &qubits,
                                      const std::array<uint_t, N> &qubits_sorted,
                                      const uint_t k);


// Dynamic (slower) version of index0 that allows the qubit number to be
// determined at runtime instead of compile time
uint_t index0_dynamic(const std::vector<uint_t> &qubits_sorted,
                      const size_t N,
                      const uint_t k);


// Dynamic (slower) version of indexes that allows the qubit number to be
// determined at runtime instead of compile time
std::vector<uint_t> indexes_dynamic(const std::vector<uint_t> &qubits,
                                    const std::vector<uint_t> &qubits_sorted,
                                    const size_t N,
                                    const uint_t k);


//------------------------------------------------------------------------------
// Static Indexing
//------------------------------------------------------------------------------

template <size_t N>
uint_t index0(const std::array<uint_t, N> &qubits_sorted, const uint_t k) {
  uint_t lowbits, retval = k;
  for (size_t j = 0; j < N; j++) {
    lowbits = retval & MASKS[qubits_sorted[j]];
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}


template <size_t N>
std::array<uint_t, 1ULL << N> indexes(const std::array<uint_t, N> &qs,
                                      const std::array<uint_t, N> &qubits_sorted,
                                      const uint_t k) {
  std::array<uint_t, 1ULL << N> ret;
  ret[0] = index0<N>(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = 1ULL << i;
    const auto bit = BITS[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}


// Specialized case of template function for N=1 qubits
template <>
std::array<uint_t, 2> indexes(const std::array<uint_t, 1> &qubits,
                              const std::array<uint_t, 1> &qubits_sorted,
                              const uint_t k) {
  std::array<uint_t, 2> ret;
  ret[0] = index0(qubits_sorted, k);
  ret[1] = ret[0] | BITS[qubits[0]];
  return ret;
}


// Specialized case of template function for N=2 qubits
template <>
std::array<uint_t, 4> indexes(const std::array<uint_t, 2> &qubits,
                              const std::array<uint_t, 2> &qubits_sorted,
                              const uint_t k) {
  std::array<uint_t, 4> ret;
  ret[0] = index0(qubits_sorted, k);
  ret[1] = ret[0] | BITS[qubits[0]];
  ret[2] = ret[0] | BITS[qubits[1]];
  ret[3] = ret[1] | BITS[qubits[1]];
  return ret;
}


//------------------------------------------------------------------------------
// Dynamic Indexing
//------------------------------------------------------------------------------

uint_t index0_dynamic(const std::vector<uint_t> &qubits_sorted,
                      const size_t N,
                      const uint_t k) {
  uint_t lowbits, retval = k;
  for (size_t j = 0; j < N; j++) {
    lowbits = retval & MASKS[qubits_sorted[j]];
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}


std::vector<uint_t> indexes_dynamic(const std::vector<uint_t> &qubits,
                                    const std::vector<uint_t> &qubits_sorted,
                                    const size_t N,
                                    const uint_t k) {
  std::vector<uint_t> ret(1ULL << N);
  ret[0] = index0_dynamic(qubits_sorted, N, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = 1ULL << i;
    const auto bit = BITS[qubits[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}


//------------------------------------------------------------------------------
} // end namespace Qubit
//------------------------------------------------------------------------------
} // end namespace Indexing
//------------------------------------------------------------------------------
#endif // end module
