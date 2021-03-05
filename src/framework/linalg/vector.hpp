/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_linalg_vector_hpp
#define _aer_framework_linalg_vector_hpp

#include <algorithm>
#include <cstddef>
#include <iostream>

#include "framework/linalg/enable_if_numeric.hpp"

/*******************************************************************************
 *
 * Numeric Vector class
 *
 ******************************************************************************/

namespace AER {

template <class T> T *malloc_array(size_t size) {
  return reinterpret_cast<T *>(malloc(sizeof(T) * size));
}

template <class T> T *calloc_array(size_t size) {
  return reinterpret_cast<T *>(calloc(size, sizeof(T)));
}

template <class T> class Vector {

public:
  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  // Construct an empty vector
  Vector() = default;

  // Construct a vector
  // If `fill=True` the vector will be initialized with all values in zero
  // if `fill=False` the vector entries will be in an indeterminate state
  // and should have their values assigned before use.
  Vector(size_t sz, bool fill = true);

  // Copy construct a vector
  Vector(const Vector<T> &other);

  // Move construct a vector
  Vector(Vector<T> &&other) noexcept;

  // Destructor
  virtual ~Vector() { free(data_); }

  //-----------------------------------------------------------------------
  // Assignment
  //-----------------------------------------------------------------------

  // Copy assignment
  Vector<T> &operator=(const Vector<T> &other);

  // Move assignment
  Vector<T> &operator=(Vector<T> &&other) noexcept;

  // Copy and cast assignment
  template <class S> Vector<T> &operator=(const Vector<S> &other);

  //-----------------------------------------------------------------------
  // Buffer conversion
  //-----------------------------------------------------------------------

  // Copy construct a vector from C-array buffer
  static Vector<T> copy_from_buffer(size_t sz, const T *buffer);

  // Move construct a vector rom C-array buffer
  static Vector<T> move_from_buffer(size_t sz, T *buffer);

  // Copy vector to a new C-array
  T *copy_to_buffer() const;

  // Move vector to a C-array
  T *move_to_buffer();

  //-----------------------------------------------------------------------
  // Element access
  //-----------------------------------------------------------------------

  // Addressing elements by vector representation
  T &operator[](size_t i) noexcept { return data_[i]; };
  const T &operator[](size_t i) const noexcept { return data_[i]; };

  // Access the array data pointer
  const T *data() const noexcept { return data_; }
  T *data() noexcept { return data_; }

  //-----------------------------------------------------------------------
  // Capacity
  //-----------------------------------------------------------------------

  // Check if vector is size 0
  bool empty() const noexcept { return size_ == 0; }

  // Return the size of the vector
  size_t size() const noexcept { return size_; };

  //-----------------------------------------------------------------------
  // Operations
  //-----------------------------------------------------------------------

  // Fill array with constant value
  void fill(const T &val);

  // Swap contents of two vectors
  void swap(Vector<T> &other);

  // Empty the vector to size size
  void clear() noexcept;

  // Resize container
  // Value of additional entries will depend on allocator type
  void resize(size_t sz);

  //-----------------------------------------------------------------------
  // Linear Algebra
  //-----------------------------------------------------------------------
  // TODO: replace these with BLAS implementations

  // Entry wise addition
  Vector<T> operator+(const Vector<T> &other) const;
  Vector<T> &operator+=(const Vector<T> &other);

  // Entry wise subtraction
  Vector<T> operator-(const Vector<T> &other) const;
  Vector<T> &operator-=(const Vector<T> &other);

  // Scalar multiplication
  Vector<T> operator*(const T &other) const;
  Vector<T> &operator*=(const T &other);

  // Scalar multiplication with casting
  template <class Scalar, typename = enable_if_numeric_t<Scalar>>
  Vector<T> operator*(const Scalar &other) const {
    operator*(T{other});
  }
  template <class Scalar, typename = enable_if_numeric_t<Scalar>>
  Vector<T> &operator*=(const Scalar &other) {
    operator*=(T{other});
  }

  // Scalar division
  Vector<T> operator/(const T &other) const;
  Vector<T> &operator/=(const T &other);

  // Scalar division with casting
  template <class Scalar, typename = enable_if_numeric_t<Scalar>>
  Vector<T> operator/(const Scalar &other) const {
    operator/(T{other});
  }
  template <class Scalar, typename = enable_if_numeric_t<Scalar>>
  Vector<T> &operator/=(const Scalar &other) {
    operator/=(T{other});
  }

protected:
  // Vector size
  size_t size_ = 0;

  // Vector data pointer
  T *data_ = nullptr;
};

/*******************************************************************************
 *
 * Vector class: methods
 *
 ******************************************************************************/

//-----------------------------------------------------------------------
// Constructors
//-----------------------------------------------------------------------

template <class T>
Vector<T>::Vector(size_t sz, bool fill)
    : size_(sz), data_((fill) ? calloc_array<T>(size_) : malloc_array<T>(size_)) {}

template <class T>
Vector<T>::Vector(const Vector<T> &other)
    : size_(other.size_), data_(malloc_array<T>(other.size_)) {
  std::copy(other.data_, other.data_ + other.size_, data_);
}

template <class T>
Vector<T>::Vector(Vector<T> &&other) noexcept
    : size_(other.size_), data_(other.data_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

//-----------------------------------------------------------------------
// Assignment
//-----------------------------------------------------------------------

template <class T> Vector<T> &Vector<T>::operator=(Vector<T> &&other) noexcept {
  free(data_);
  size_ = other.size_;
  data_ = other.data_;
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}

template <class T> Vector<T> &Vector<T>::operator=(const Vector<T> &other) {
  if (size_ != other.size_) {
    free(data_);
    size_ = other.size_;
    data_ = malloc_array<T>(size_);
  }
  std::copy(other.data_, other.data_ + size_, data_);
  return *this;
}

template <class T>
template <class S>
inline Vector<T> &Vector<T>::operator=(const Vector<S> &other) {

  if (size_ != other.size_) {
    free(data_);
    size_ = other.size_;
    data_ = malloc_array<T>(size_);
  }
  std::transform(other.data_, other.data_ + size_, data_,
                 [](const S &i) { return T{i}; });
  return *this;
}

//-----------------------------------------------------------------------
// Buffer conversion
//-----------------------------------------------------------------------

template <class T>
Vector<T> Vector<T>::copy_from_buffer(size_t sz, const T *buffer) {
  Vector<T> ret;
  ret.size_ = sz;
  ret.data_ = malloc_array<T>(ret.size_);
  std::copy(buffer, buffer + ret.size_, ret.data_);
  return ret;
}

template <class T>
Vector<T> Vector<T>::move_from_buffer(size_t sz, T *buffer) {
  Vector<T> ret;
  ret.size_ = sz;
  ret.data_ = buffer;
  return ret;
}

template <class T> T *Vector<T>::copy_to_buffer() const {
  T *buffer = malloc_array<T>(size_);
  std::copy(data_, data_ + size_, buffer);
  return buffer;
}

template <class T> T *Vector<T>::move_to_buffer() {
  T *buffer = data_;
  data_ = nullptr;
  size_ = 0;
  return buffer;
}

//-----------------------------------------------------------------------
// Operations
//-----------------------------------------------------------------------

template <class T> void Vector<T>::clear() noexcept {
  free(data_);
  size_ = 0;
}

template <class T> void Vector<T>::swap(Vector<T> &other) {
  std::swap(size_, other.size_);
  std::swap(data_, other.data_);
}

template <class T> void Vector<T>::resize(size_t sz) {
  if (size_ == sz)
    return;
  T *tmp = calloc_array<T>(sz);
  std::move(data_, data_ + size_, tmp);
  free(data_);
  size_ = sz;
  data_ = tmp;
}

template <class T> void Vector<T>::fill(const T &val) {
  std::fill(data_, data_ + size_, val);
}

//-----------------------------------------------------------------------
// Linear Algebra
//-----------------------------------------------------------------------

template <class T>
Vector<T> Vector<T>::operator+(const Vector<T> &other) const {
  if (size_ != other.size_) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  Vector<T> result;
  result.size_ = size_;
  result.data_ = malloc_array<T>(size_);
  std::transform(data_, data_ + size_, other.data_, result.data_,
                 [](const T &a, const T &b) -> T { return a + b; });
  return result;
}

template <class T> Vector<T> &Vector<T>::operator+=(const Vector<T> &other) {
  if (size_ != other.size_) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  std::transform(data_, data_ + size_, other.data_, data_,
                 [](const T &a, const T &b) -> T { return a + b; });
  return *this;
}

template <class T>
Vector<T> Vector<T>::operator-(const Vector<T> &other) const {
  if (size_ != other.size_) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  Vector<T> result;
  result.size_ = size_;
  result.data_ = malloc_array<T>(size_);
  std::transform(data_, data_ + size_, other.data_, result.data_,
                 [](const T &a, const T &b) -> T { return a - b; });
  return result;
}

template <class T> Vector<T> &Vector<T>::operator-=(const Vector<T> &other) {
  if (size_ != other.size_) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  std::transform(data_, data_ + size_, other.data_, data_,
                 [](const T &a, const T &b) -> T { return a - b; });
  return *this;
}

template <class T> Vector<T> Vector<T>::operator*(const T &other) const {
  Vector<T> ret;
  ret.size_ = size_;
  ret.data_ = malloc_array<T>(size_);
  std::transform(data_, data_ + size_, ret.data_,
                 [&other](const T &a) -> T { return a * other; });
  return ret;
}

template <class T> Vector<T> &Vector<T>::operator*=(const T &other) {
  std::for_each(data_, data_ + size_,
                 [&other](T &a) { a *= other; });
  return *this;
}

template <class T> Vector<T> Vector<T>::operator/(const T &other) const {
  Vector<T> ret;
  ret.size_ = size_;
  ret.data_ = malloc_array<T>(size_);
  std::transform(data_, data_ + size_, ret.data_,
                 [&other](const T &a) -> T { return a / other; });
  return ret;
}

template <class T> Vector<T> &Vector<T>::operator/=(const T &other) {
  std::for_each(data_, data_ + size_, [&other](T &a) { a /= other; });
  return *this;
}


//------------------------------------------------------------------------------
} // end Namespace AER
//------------------------------------------------------------------------------

//-----------------------------------------------------------------------
// ostream
//-----------------------------------------------------------------------
template <class T>
std::ostream &operator<<(std::ostream &out, const AER::Vector<T> &v) {
  out << "[";
  size_t last = v.size() - 1;
  const auto &data = v.data();
  for (size_t i = 0; i < v.size(); ++i) {
    out << data[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

//------------------------------------------------------------------------------
#endif
