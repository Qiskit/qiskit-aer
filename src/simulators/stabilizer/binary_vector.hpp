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


#ifndef _binary_vector_hpp_
#define _binary_vector_hpp_

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

namespace BV {

/*******************************************************************************
 *
 * BinaryVector Class
 *
 ******************************************************************************/

class BinaryVector {
public:
  const static size_t BLOCK_SIZE = 64;

  BinaryVector() : m_length(0), m_data(0){};

  explicit BinaryVector(uint64_t length)
      : m_length(length), m_data((length - 1) / BLOCK_SIZE + 1, ZERO_){};

  BinaryVector(std::vector<uint64_t> mdata)
      : m_length(mdata.size()), m_data(mdata){};

  explicit BinaryVector(std::string);

  void setLength(uint64_t length);

  void setVector(std::string);
  void setValue(bool value, uint64_t pos);

  void set0(uint64_t pos) { setValue(ZERO_, pos); };
  void set1(uint64_t pos) { setValue(ONE_, pos); };

  void flipAt(uint64_t pos);

  BinaryVector &operator+=(const BinaryVector &rhs);

  bool operator[](const uint64_t pos) const;

  void swap(BinaryVector &rhs);

  uint64_t getLength() const { return m_length; };

  void makeZero() { m_data.assign((m_length - 1) / BLOCK_SIZE + 1, ZERO_); }

  bool isZero() const;

  bool isSame(const BinaryVector &rhs) const;
  bool isSame(const BinaryVector &rhs, bool pad) const;

  std::vector<uint64_t> nonzeroIndices() const;
  std::vector<uint64_t> getData() const { return m_data; };

private:
  uint64_t m_length;
  std::vector<uint64_t> m_data;
  static const uint64_t ZERO_;
  static const uint64_t ONE_;
};


/*******************************************************************************
 *
 * Related Functions
 *
 ******************************************************************************/

inline bool operator==(const BinaryVector &lhs, const BinaryVector &rhs) {
  return lhs.isSame(rhs, true);
}


inline int64_t gauss_eliminate(std::vector<BinaryVector> &M,
                               const int64_t start_col = 0)
// returns the rank of M.
// M[] has length nrows.
// each M[i] must have the same length ncols.
{
  const int64_t nrows = M.size();
  const int64_t ncols = M.front().getLength();
  int64_t rank = 0;
  int64_t k, r, i;
  for (k = start_col; k < ncols; k++) {
    i = -1;
    for (r = rank; r < nrows; r++) {
      if (M[r][k] == 0)
        continue;
      if (i == -1) {
        i = r;
        rank++;
      } else {
        M[r] += M[i];
      }
    }
    if (i >= rank) {
      M[i].swap(M[rank - 1]);
    }
  }
  return rank;
}


inline std::vector<uint64_t> string_to_bignum(std::string val,
                                              uint64_t blockSize,
                                              uint64_t base) {
  std::vector<uint64_t> ret;
  if (blockSize * log2(base) > 64) {
    throw std::runtime_error(
        std::string("block size is greater than 64-bits for current case"));
  }
  auto n = val.size();
  auto blocks = n / blockSize;
  auto tail = n % blockSize;
  for (uint64_t j = 0; j != blocks; ++j)
    ret.push_back(
        stoull(val.substr(n - (j + 1) * blockSize, blockSize), 0, blockSize));
  if (tail > 0)
    ret.push_back(stoull(val.substr(0, tail), 0, blockSize));
  return ret;
}


inline std::vector<uint64_t> string_to_bignum(std::string val) {
  std::string type = val.substr(0, 2);
  if (type == "0b" || type == "0B")
    // Binary string
    return string_to_bignum(val.substr(2, val.size() - 2), 64, 2);
  else if (type == "0x" || type == "0X")
    // Hexidecimal string
    return string_to_bignum(val.substr(2, val.size() - 2), 16, 16);
  else {
    // Decimal string
    throw std::runtime_error(
        std::string("string must be binary (0b) or hex (0x)"));
  }
}


/*******************************************************************************
 *
 * BinaryVector Class Methods
 *
 ******************************************************************************/

const uint64_t BinaryVector::ZERO_ = 0ULL;
const uint64_t BinaryVector::ONE_ = 1ULL;

BinaryVector::BinaryVector(std::string val) {
  m_data = string_to_bignum(val);
  m_length = m_data.size();
}


void BinaryVector::setLength(uint64_t length) {
  if (length == 0 || m_length > 0)
    return;

  m_length = length;
  m_data.assign((length - 1) / BLOCK_SIZE + 1, ZERO_);
}


void BinaryVector::setValue(bool value, uint64_t pos) {
  auto q = pos / BLOCK_SIZE;
  auto r = pos % BLOCK_SIZE;
  if (value)
    m_data[q] |= (ONE_ << r);
  else
    m_data[q] &= ~(ONE_ << r);
}


void BinaryVector::flipAt(const uint64_t pos) {
  auto q = pos / BLOCK_SIZE;
  auto r = pos % BLOCK_SIZE;
  m_data[q] ^= (ONE_ << r);
}


BinaryVector &BinaryVector::operator+=(const BinaryVector &rhs) {
  const auto size = m_data.size();
  for (size_t i = 0; i < size; i++)
    m_data[i] ^= rhs.m_data[i];
  return (*this);
}


bool BinaryVector::operator[](const uint64_t pos) const {
  auto q = pos / BLOCK_SIZE;
  auto r = pos % BLOCK_SIZE;
  return ((m_data[q] & (ONE_ << r)) != 0);
}


void BinaryVector::swap(BinaryVector &rhs) {
  uint64_t tmp;
  tmp = rhs.m_length;
  rhs.m_length = m_length;
  m_length = tmp;

  m_data.swap(rhs.m_data);
}


bool BinaryVector::isZero() const {
  const size_t size = m_data.size();
  for (size_t i = 0; i < size; i++)
    if (m_data[i])
      return false;
  return true;
}


bool BinaryVector::isSame(const BinaryVector &rhs) const {
  if (m_length != rhs.m_length)
    return false;
  const size_t size = m_data.size();
  for (size_t q = 0; q < size; q++) {
    if (m_data[q] != rhs.m_data[q])
      return false;
  }
  return true;
}


bool BinaryVector::isSame(const BinaryVector &rhs, bool pad) const {
  if (!pad)
    return isSame(rhs);

  const size_t sz0 = m_data.size();
  const size_t sz1 = rhs.m_data.size();
  const size_t sz = (sz0 > sz1) ? sz1 : sz0;

  // Check vectors agree on overlap
  for (size_t q = 0; q < sz; q++)
    if (m_data[q] != rhs.m_data[q])
      return false;
  // Check padding of larger vector is trivial
  for (size_t q = sz; q < sz0; q++)
    if (m_data[q] != 0)
      return false;
  for (size_t q = sz; q < sz1; q++)
    if (rhs.m_data[q] != 0)
      return false;

  return true;
}


std::vector<uint64_t> BinaryVector::nonzeroIndices() const {
  std::vector<uint64_t> result;
  size_t i = 0;
  while (i < m_data.size()) {
    while (m_data[i] == 0) {
      i++;
      if (i == m_data.size())
        return result; // empty
    }
    auto m = m_data[i];
    size_t r = 0;
    while (r < BLOCK_SIZE) {
      while ((m & (ONE_ << r)) == 0) {
        r++;
      }
      if (r >= BLOCK_SIZE)
        break;
      result.push_back(static_cast<uint64_t>((i * BLOCK_SIZE) + r));
      r++;
    }
    i++;
  }
  return result;
}


//------------------------------------------------------------------------------
} // end namespace BV
//------------------------------------------------------------------------------
#endif