/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2024.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_bitvector_hpp_
#define _aer_framework_bitvector_hpp_

#include "framework/rng.hpp"
#include "framework/types.hpp"
#include "framework/utils.hpp"

namespace AER {

//============================================================================
// Bit vestor class
//============================================================================

class BitVector {
protected:
  reg_t bits_;
  uint_t num_bits_;
  const static size_t REG_SIZE = 64;
  const static size_t REG_BITS = 6;
  const static size_t REG_MASK = (1ull << REG_BITS) - 1;

public:
  BitVector() { num_bits_ = 0; }
  BitVector(uint_t nbits, bool init = false) { allocate(nbits, init); }
  BitVector(const BitVector &src) {
    bits_ = src.bits_;
    num_bits_ = src.num_bits_;
  }

  uint_t num_bits() { return num_bits_; }
  uint_t length() { return bits_.size(); }

  void allocate(uint_t n, bool init = false) {
    uint_t size = n >> REG_BITS;
    if (size == 0)
      size = 1;
    if (init)
      bits_.resize(size, 0ull);
    else
      bits_.resize(size);
    num_bits_ = n;
  }

  BitVector &operator=(const BitVector &src) {
    bits_ = src.bits_;
    num_bits_ = src.num_bits_;
    return *this;
  }
  BitVector &operator=(const std::string &src) {
    from_string(src);
    return *this;
  }
  BitVector &operator=(const reg_t &src) {
    from_vector(src);
    return *this;
  }

  // copy with swap
  void map(const BitVector &src, const reg_t map);

  // bit access
  inline bool get(const uint_t idx) const {
    uint_t pos = idx >> REG_BITS;
    uint_t bit = idx & REG_MASK;
    return (((bits_[pos] >> bit) & 1ull) == 1ull);
  }
  inline bool operator[](const uint_t idx) const { return get(idx); }
  inline uint_t &operator()(const uint_t pos) { return bits_[pos]; }
  inline uint_t operator()(const uint_t pos) const { return bits_[pos]; }

  void set(const uint_t idx, const bool val) {
    uint_t pos = idx >> REG_BITS;
    uint_t bit = idx & REG_MASK;
    uint_t mask = ~(1ull << bit);
    bits_[pos] &= mask;
    bits_[pos] |= (((uint_t)val) << bit);
  }

  void zero(void);
  void rand(RngEngine &rng);

  bool operator==(const BitVector &src) const;
  bool operator!=(const BitVector &src) const;
  bool is_zero(void);

  // logical operators
  BitVector &operator|=(const BitVector &src);
  BitVector &operator&=(const BitVector &src);
  BitVector &operator^=(const BitVector &src);

  friend BitVector operator|(const BitVector &v0, const BitVector &v1);
  friend BitVector operator&(const BitVector &v0, const BitVector &v1);
  friend BitVector operator^(const BitVector &v0, const BitVector &v1);
  friend BitVector operator~(const BitVector &v);

  void set_or(const uint_t idx, const bool b) {
    uint_t pos = idx >> REG_BITS;
    uint_t bit = idx & REG_MASK;
    uint_t val = (uint_t)b << bit;
    bits_[pos] |= val;
  }

  void set_and(const uint_t idx, const bool b) {
    uint_t pos = idx >> REG_BITS;
    uint_t bit = idx & REG_MASK;
    uint_t val = (~(1ull << bit)) | ((uint_t)b << bit);
    bits_[pos] &= val;
  }

  void set_xor(const uint_t idx, const bool b) {
    uint_t pos = idx >> REG_BITS;
    uint_t bit = idx & REG_MASK;
    uint_t val = (uint_t)b << bit;
    bits_[pos] ^= val;
  }

  void set_not(const uint_t idx) {
    uint_t pos = idx >> REG_BITS;
    uint_t bit = idx & REG_MASK;
    uint_t val = 1ull << bit;
    bits_[pos] ^= val;
  }

  // convert from other data
  void from_uint(const uint_t nbits, const uint_t src);
  void from_string(const std::string &src);
  void from_vector(const reg_t &src);
  void from_vector_with_map(const reg_t &src, const reg_t &map);

  // convert to other data types
  std::string to_string();
  std::string to_hex_string(bool prefix = true);
  reg_t to_vector();

  bool parity(void);
  uint_t popcount(void);
};

void BitVector::map(const BitVector &src, const reg_t map) {
  allocate(map.size());

  for (uint_t i = 0; i < map.size(); i++) {
    set(i, src[map[i]]);
  }
}

void BitVector::from_uint(const uint_t nbits, const uint_t src) {
  allocate(nbits);
  bits_[0] = src;
}

void BitVector::from_string(const std::string &src) {
  allocate(src.size());

  uint_t pos = 0;
  for (uint_t i = 0; i < bits_.size(); i++) {
    uint_t n = REG_SIZE;
    uint_t val = 0;
    if (n > num_bits_ - pos)
      n = num_bits_ - pos;
    for (uint_t j = 0; j < n; j++) {
      val |= (((uint_t)(src[num_bits_ - 1 - pos] == '1')) << j);
      pos++;
    }
    bits_[i] = val;
  }
}

void BitVector::from_vector(const reg_t &src) {
  allocate(src.size());

  uint_t pos = 0;
  for (uint_t i = 0; i < bits_.size(); i++) {
    uint_t n = REG_SIZE;
    uint_t val = 0;
    if (n > num_bits_ - pos)
      n = num_bits_ - pos;
    for (uint_t j = 0; j < n; j++) {
      val |= ((src[pos++] & 1ull) << j);
    }
    bits_[i] = val;
  }
}

void BitVector::from_vector_with_map(const reg_t &src, const reg_t &map) {
  allocate(src.size());

  uint_t pos = 0;
  for (uint_t i = 0; i < bits_.size(); i++) {
    uint_t n = REG_SIZE;
    uint_t val = 0;
    if (n > num_bits_ - pos)
      n = num_bits_ - pos;
    for (uint_t j = 0; j < n; j++) {
      val |= ((src[map[pos++]] & 1ull) << j);
    }
    bits_[i] = val;
  }
}

std::string BitVector::to_string(void) {
  std::string str;
  for (uint_t i = 0; i < num_bits_; i++) {
    if (get(num_bits_ - 1 - i))
      str += '1';
    else
      str += '0';
  }
  return str;
}

std::string BitVector::to_hex_string(bool prefix) {
  // initialize output string
  std::string hex = (prefix) ? "0x" : "";

  for (uint_t i = 0; i < bits_.size(); i++) {
    if (i == 0) {
      uint_t n = num_bits_ & (REG_SIZE - 1);
      uint_t val = bits_[bits_.size() - 1] & ((1ull << n) - 1);

      std::stringstream ss;
      ss << std::hex << val;
      hex += ss.str();
    } else {
      std::stringstream ss;
      ss << std::hex << bits_[bits_.size() - 1 - i];
      std::string part = ss.str();
      part.insert(0, (REG_SIZE / 4) - part.size(), '0');
      hex += part;
    }
  }
  return hex;
}

reg_t BitVector::to_vector(void) {
  reg_t ret(num_bits_);
  for (uint_t i = 0; i < num_bits_; i++) {
    ret[i] = (uint_t)get(i);
  }
  return ret;
}

void BitVector::zero(void) {
  for (uint_t i = 0; i < bits_.size(); i++) {
    bits_[i] = 0;
  }
}

void BitVector::rand(RngEngine &rng) {
  if (bits_.size() == 1) {
    bits_[0] = rng.rand_int(0ull, (1ull << num_bits_) - 1ull);
  } else {
    double r = rng.rand();
    double c = 0.5;

    for (int_t i = num_bits_ - 1; i >= 0; i--) {
      if (r >= c) {
        set(i, true);
        r -= c;
      } else
        set(i, false);
      c /= 2.0;
    }
  }
}

bool BitVector::operator==(const BitVector &src) const {
  if (num_bits_ != src.num_bits_)
    return false;
  for (uint_t i = 0; i < bits_.size(); i++) {
    if (bits_[i] != src.bits_[i])
      return false;
  }
  return true;
}

bool BitVector::operator!=(const BitVector &src) const {
  if (num_bits_ != src.num_bits_)
    return true;
  for (uint_t i = 0; i < bits_.size(); i++) {
    if (bits_[i] != src.bits_[i])
      return true;
  }
  return false;
}

bool BitVector::is_zero(void) {
  if (num_bits_ == 0)
    return true;
  for (uint_t i = 0; i < bits_.size(); i++) {
    if (bits_[i] != 0)
      return false;
  }
  return true;
}

BitVector &BitVector::operator|=(const BitVector &src) {
  uint_t size = std::min(src.bits_.size(), bits_.size());
  for (uint_t i = 0; i < size; i++)
    bits_[i] |= src.bits_[i];
  return *this;
}

BitVector &BitVector::operator&=(const BitVector &src) {
  uint_t size = std::min(src.bits_.size(), bits_.size());
  for (uint_t i = 0; i < size; i++)
    bits_[i] &= src.bits_[i];
  return *this;
}

BitVector &BitVector::operator^=(const BitVector &src) {
  uint_t size = std::min(src.bits_.size(), bits_.size());
  for (uint_t i = 0; i < size; i++)
    bits_[i] ^= src.bits_[i];
  return *this;
}

BitVector operator|(const BitVector &v0, const BitVector &v1) {
  BitVector ret(v0);
  return ret |= v1;
}

BitVector operator&(const BitVector &v0, const BitVector &v1) {
  BitVector ret(v0);
  return ret &= v1;
}

BitVector operator^(const BitVector &v0, const BitVector &v1) {
  BitVector ret(v0);
  return ret ^= v1;
}

BitVector operator~(const BitVector &v) {
  BitVector ret(v);
  for (uint_t i = 0; i < ret.bits_.size(); i++) {
    ret.bits_[i] = ~ret.bits_[i];
  }
  return ret;
}

bool BitVector::parity(void) {
  uint_t ret = 0;
  for (uint_t i = 0; i < bits_.size(); i++) {
    ret ^= AER::Utils::hamming_parity(bits_[i]);
  }
  return ret != 0;
}

uint_t BitVector::popcount(void) {
  uint_t ret = 0;
  for (uint_t i = 0; i < bits_.size(); i++) {
    ret += AER::Utils::popcount(bits_[i]);
  }
  return ret;
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif // _aer_framework_bitvector_hpp_
