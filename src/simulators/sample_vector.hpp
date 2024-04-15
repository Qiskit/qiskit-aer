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

#ifndef _aer_simulator_sample_vector_hpp_
#define _aer_simulator_sample_vector_hpp_

#include "framework/types.hpp"

namespace AER {

//============================================================================
// stroage for sampling measure results
//============================================================================

class SampleVector {
protected:
  reg_t bits_;
  uint_t size_;
  uint_t base_;
  uint_t elem_shift_bits_;
  uint_t elem_mask_;
  uint_t vec_shift_bits_;
  uint_t vec_mask_;
  const static size_t REG_SIZE = 64;

public:
  SampleVector() {
    base_ = 2;
    size_ = 0;
  }
  SampleVector(uint_t nbits, uint_t base = 2) { allocate(nbits, base); }
  SampleVector(const SampleVector &src) {
    bits_ = src.bits_;
    size_ = src.size_;
    base_ = src.base_;
    elem_shift_bits_ = src.elem_shift_bits_;
    elem_mask_ = src.elem_mask_;
    vec_shift_bits_ = src.vec_shift_bits_;
    vec_mask_ = src.vec_mask_;
  }

  uint_t size() { return size_; }
  uint_t length() { return bits_.size(); }

  void allocate(uint_t n, uint_t base);

  SampleVector &operator=(const SampleVector &src) {
    bits_ = src.bits_;
    size_ = src.size_;
    base_ = src.base_;
    elem_shift_bits_ = src.elem_shift_bits_;
    elem_mask_ = src.elem_mask_;
    vec_shift_bits_ = src.vec_shift_bits_;
    vec_mask_ = src.vec_mask_;
    return *this;
  }
  SampleVector &operator=(const std::string &src) {
    from_string(src);
    return *this;
  }
  SampleVector &operator=(const reg_t &src) {
    from_vector(src);
    return *this;
  }

  // copy with swap
  void map(const SampleVector &src, const reg_t map);

  // bit access
  inline uint_t get(const uint_t idx) const {
    uint_t vpos = idx >> vec_shift_bits_;
    uint_t bpos = (idx & vec_mask_) << elem_shift_bits_;
    return ((bits_[vpos] >> bpos) & elem_mask_);
  }
  inline uint_t operator[](const uint_t idx) const { return get(idx); }
  inline uint_t &operator()(const uint_t pos) { return bits_[pos]; }
  inline uint_t operator()(const uint_t pos) const { return bits_[pos]; }

  inline void set(const uint_t idx, const uint_t val) {
    uint_t vpos = idx >> vec_shift_bits_;
    uint_t bpos = (idx & vec_mask_) << elem_shift_bits_;

    uint_t mask = ~(elem_mask_ << bpos);
    bits_[vpos] &= mask;
    bits_[vpos] |= ((val & elem_mask_) << bpos);
  }

  // convert from other data
  void from_uint(const uint_t src, const uint_t n, const uint_t base = 2);
  void from_string(const std::string &src, const uint_t base = 2);
  void from_vector(const reg_t &src, const uint_t base = 2);
  void from_vector_with_map(const reg_t &src, const reg_t &map,
                            const uint_t base = 2);

  // convert to other data types
  std::string to_string();
  reg_t to_vector();
};

void SampleVector::allocate(uint_t n, uint_t base) {
  vec_shift_bits_ = 6;
  uint_t t = 1;
  elem_shift_bits_ = 0;
  for (uint_t i = 0; i < 6; i++) {
    t <<= 1;
    if (t >= base) {
      break;
    }
    vec_shift_bits_--;
    elem_shift_bits_++;
  }
  elem_mask_ = (1ull << (elem_shift_bits_ + 1)) - 1;
  vec_mask_ = (1ull << vec_shift_bits_) - 1;

  uint_t size = (n + (REG_SIZE >> elem_shift_bits_) - 1) >> vec_shift_bits_;
  bits_.resize(size, 0ull);
  size_ = n;
}

void SampleVector::map(const SampleVector &src, const reg_t map) {
  allocate(map.size(), src.base_);

  for (uint_t i = 0; i < map.size(); i++) {
    set(i, src[map[i]]);
  }
}

void SampleVector::from_uint(const uint_t src, const uint_t n,
                             const uint_t base) {
  allocate(n, base);
  bits_[0] = src;
}

void SampleVector::from_string(const std::string &src, const uint_t base) {
  allocate(src.size(), base);

  uint_t pos = 0;
  uint_t n = REG_SIZE >> elem_shift_bits_;
  for (uint_t i = 0; i < bits_.size(); i++) {
    uint_t val = 0;
    if (n > size_ - pos)
      n = size_ - pos;
    for (uint_t j = 0; j < n; j++) {
      val |= (((uint_t)(src[size_ - 1 - pos] - '0') & elem_mask_)
              << (j << elem_shift_bits_));
      pos++;
    }
    bits_[i] = val;
  }
}

void SampleVector::from_vector(const reg_t &src, const uint_t base) {
  allocate(src.size(), base);

  uint_t pos = 0;
  uint_t n = REG_SIZE >> elem_shift_bits_;
  for (uint_t i = 0; i < bits_.size(); i++) {
    uint_t val = 0;
    if (n > size_ - pos)
      n = size_ - pos;
    for (uint_t j = 0; j < n; j++) {
      val |= ((src[pos++] & elem_mask_) << (j << elem_shift_bits_));
    }
    bits_[i] = val;
  }
}

void SampleVector::from_vector_with_map(const reg_t &src, const reg_t &map,
                                        const uint_t base) {
  allocate(src.size(), base);

  uint_t pos = 0;
  uint_t n = REG_SIZE >> elem_shift_bits_;
  for (uint_t i = 0; i < bits_.size(); i++) {
    uint_t val = 0;
    if (n > size_ - pos)
      n = size_ - pos;
    for (uint_t j = 0; j < n; j++) {
      val |= ((src[map[pos++]] & elem_mask_) << (j << elem_shift_bits_));
    }
    bits_[i] = val;
  }
}

std::string SampleVector::to_string(void) {
  std::string str;
  for (uint_t i = 0; i < size_; i++) {
    uint_t val = get(size_ - 1 - i);
    str += std::to_string(val);
  }
  return str;
}

reg_t SampleVector::to_vector(void) {
  reg_t ret(size_);
  for (uint_t i = 0; i < size_; i++) {
    ret[i] = get(i);
  }
  return ret;
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif // _aer_simulator_sample_vector_hpp_
