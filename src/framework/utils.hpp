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

#ifndef _aer_framework_utils_hpp_
#define _aer_framework_utils_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "framework/avx2_detect.hpp"
#include "framework/types.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN64) || defined(_WIN32)
// This is needed because windows.h redefine min()/max() so interferes with
// std::min/max
#define NOMINMAX
#include <windows.h>
#endif

namespace AER {
namespace Utils {

//------------------------------------------------------------------------------
// Matrix Functions
//------------------------------------------------------------------------------

// Construct a matrix from a vector of matrix-row vectors
template <class T>
matrix<T> make_matrix(const std::vector<std::vector<T>> &mat);

// Reshape a length column-major vectorized matrix into a square matrix
template <class T>
matrix<T> devectorize_matrix(const std::vector<T> &vec);

// Vectorize a matrix by stacking matrix columns (column-major vectorization)
template <class T>
std::vector<T> vectorize_matrix(const matrix<T> &mat);

// Return the transpose a matrix
template <class T>
matrix<T> transpose(const matrix<T> &A);

// Return the adjoing (Hermitian-conjugate) of a matrix
template <class T>
matrix<std::complex<T>> dagger(const matrix<std::complex<T>> &A);

// Return the complex conjugate of a matrix
template <class T>
matrix<std::complex<T>> conjugate(const matrix<std::complex<T>> &A);

// Given a list of matrices for a multiplexer stacks and packs them 0/1/2/...
// into a single 2^control x (2^target x 2^target) cmatrix_t)
// Equivalent to a 2^qubits x 2^target "flat" matrix
template <class T>
matrix<T> stacked_matrix(const std::vector<matrix<T>> &mmat);

// Return a vector containing the diagonal of a matrix
template <class T>
std::vector<T> matrix_diagonal(const matrix<T> &mat);

// Inplace transformations
template <class T>
matrix<T> &transpose_inplace(matrix<T> &A);
template <class T>
matrix<std::complex<T>> &dagger_inplace(matrix<std::complex<T>> &A);
template <class T>
matrix<std::complex<T>> &conjugate_inplace(matrix<std::complex<T>> &A);

// Tracing
template <class T>
T trace(const matrix<T> &A);
template <class T>
matrix<T> partial_trace_a(const matrix<T> &rho, size_t dimA);
template <class T>
matrix<T> partial_trace_b(const matrix<T> &rho, size_t dimB);

// Tensor product
template <class T>
matrix<T> tensor_product(const matrix<T> &A, const matrix<T> &B);
template <class T>
matrix<T> unitary_superop(const matrix<T> &mat);
template <class T>
matrix<T> kraus_superop(const std::vector<matrix<T>> &kmats);

// concatenate
// Returns a matrix that is the concatenation of two matrices A, B
// The matrices must have the same dimensions
// If axis == 0, place rows of B after rows of A (vertical extension)
// If axis == 1, place columns of B after columns of A (horizontal extension)
template <class T>
matrix<T> concatenate(const matrix<T> &A, const matrix<T> &B, uint_t axis);

// split
// Splits A into 2 matrices B and C equal in dimensions
// If axis == 0, split A by rows. A must have an even number of rows.
// If axis == 1, split A by columns. A must have an even number of columns.
template <class T>
void split(const matrix<T> &A, matrix<T> &B, matrix<T> &C, uint_t axis);

// Elementwise matrix multiplication
template <class T>
matrix<T> elementwise_multiplication(const matrix<T> &A, const matrix<T> &B);

// Matrix sum of elements
template <class T>
T sum(const matrix<T> &A);

// Matrix comparison
template <class T>
bool is_square(const matrix<T> &mat);

template <class T>
bool is_diagonal(const matrix<T> &mat);

template <class T>
bool is_equal(const matrix<T> &mat1, const matrix<T> &mat2, double threshold);

template <class T>
bool is_diagonal(const matrix<T> &mat, double threshold);

template <class T>
std::pair<bool, double> is_identity_phase(const matrix<T> &mat,
                                          double threshold);

template <class T>
bool is_identity(const matrix<T> &mat, double threshold);

template <class T>
bool is_diagonal_identity(const matrix<T> &mat, double threshold);

template <class T>
bool is_unitary(const matrix<T> &mat, double threshold);

template <class T>
bool is_hermitian(const matrix<T> &mat, double threshold);

template <class T>
bool is_symmetrix(const matrix<T> &mat, double threshold);

template <class T>
bool is_cptp_kraus(const std::vector<matrix<T>> &kraus, double threshold);

//------------------------------------------------------------------------------
// Vector functions
//------------------------------------------------------------------------------

// Return true of the vector has norm-1.
template <typename T>
double is_unit_vector(const std::vector<T> &vec);

// Conjugate a vector
template <typename T>
std::vector<std::complex<T>> conjugate(const std::vector<std::complex<T>> &v);
template <class T>
Vector<std::complex<T>> conjugate(const Vector<std::complex<T>> &v);

// Compute the Euclidean 2-norm of a vector
template <typename T>
double norm(const std::vector<T> &vec);

// Return the matrix formed by taking the outproduct of two vector |ket><bra|
template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra);

template <typename T>
inline matrix<T> projector(const std::vector<T> &ket) {
  return outer_product(ket, ket);
}

// Tensor product vector
template <typename T>
std::vector<T> tensor_product(const std::vector<T> &v, const std::vector<T> &w);
template <typename T>
Vector<T> tensor_product(const Vector<T> &v, const Vector<T> &w);

// Return a new vector formed by multiplying each element of the input vector
// with a scalar. The product of types T1 * T2 must be valid.
template <typename T1, typename T2>
std::vector<T1> scalar_multiply(const std::vector<T1> &vec, T2 val);

// Inplace multiply each entry in a vector by a scalar and returns a reference
// to the input vector argument. The product of types T1 * T2 must be valid.
template <typename T1, typename T2>
std::vector<T1> &scalar_multiply_inplace(std::vector<T1> &vec, T2 scalar);

// Truncate the first argument its absolute value is less than epsilon
// this function returns a refernce to the chopped first argument
double &chop_inplace(double &val, double epsilon);
std::complex<double> &chop_inplace(std::complex<double> &val, double epsilon);

double chop(double val, double epsilon);
float chop(float val, double epsilon);

// As above for complex first arguments
template <typename T>
std::complex<T> chop(std::complex<T> val, double epsilon);
// Truncate each element in a vector if its absolute value is less than epsilon
// This function returns a reference to the chopped input vector
template <typename T>
std::vector<T> &chop_inplace(std::vector<T> &vec, double epsilon);

template <typename T>
std::vector<T> chop(const std::vector<T> &vec, double epsilon);

// Add rhs vector to lhs using move semantics.
// rhs should not be used after this operation.
template <class T>
void combine(std::vector<T> &lhs, const std::vector<T> &rhs);

// Convert a dense vector into sparse ket form.
// epsilon determins the threshold for which small values will be removed from
// the output. The base of the ket (2-10 for qudits, or 16 for hexadecimal)
// specifies the subsystem dimension and the base of the dit-string labels.
template <typename T>
std::map<std::string, T> vec2ket(const std::vector<T> &vec, double epsilon,
                                 uint_t base = 2);

template <typename T>
std::map<std::string, T> vec2ket(const T *const vec, uint_t dim, double epsilon,
                                 uint_t base = 2);

//------------------------------------------------------------------------------
// Bit Conversions
//------------------------------------------------------------------------------

// Format a hex string so that it has a prefix "0x", abcdef chars are lowercase
// and leading zeros are removed
// Example: 0010A -> 0x10a
std::string format_hex(const std::string &hex);
std::string &format_hex_inplace(std::string &hex);

// Pad string with a char if it is less
std::string padleft(const std::string &s, char c, size_t min_length);
std::string &padleft_inplace(std::string &s, char c, size_t min_length);

// Convert integers and hexadecimals to register vectors
reg_t int2reg(uint_t n, uint_t base = 2);
reg_t int2reg(uint_t n, uint_t base, uint_t minlen);
reg_t hex2reg(std::string str);

// Convert bit-strings to hex-strings
// if prefix is true "0x" will prepend the output string
std::string bin2hex(const std::string bin, bool prefix = true);

// Convert hex-strings to bit-strings
// if prefix is true "0b" will prepend the output string
std::string hex2bin(const std::string bs, bool prefix = true);

// Convert 64-bit unsigned integers to dit-string (dit base = 2 to 10)
std::string int2string(uint_t n, uint_t base = 2);
std::string int2string(uint_t n, uint_t base, uint_t length);

// Convert integers to bit-strings
inline std::string int2bin(uint_t n) { return int2string(n, 2); }
inline std::string int2bin(uint_t n, uint_t length) {
  return int2string(n, 2, length);
}

// Convert integers to hex-strings
inline std::string int2hex(uint_t n) { return bin2hex(int2bin(n)); }

// Convert reg to int
uint_t reg2int(const reg_t &reg, uint_t base);

// Count number of 1's in bitstring representation of an integer
const uint_t zer = 0U;
const uint_t one = 1U;

//==============================================================================
// Implementations: Matrix functions
//==============================================================================

template <class T>
matrix<T> devectorize_matrix(const std::vector<T> &vec) {
  size_t dim = std::sqrt(vec.size());
  matrix<T> mat(dim, dim);
  for (size_t col = 0; col < dim; col++)
    for (size_t row = 0; row < dim; row++) {
      mat(row, col) = vec[dim * col + row];
    }
  return mat;
}

template <class T>
std::vector<T> vectorize_matrix(const matrix<T> &mat) {
  std::vector<T> vec;
  vec.resize(mat.size(), 0.);
  size_t nrows = mat.GetRows();
  size_t ncols = mat.GetColumns();
  for (size_t col = 0; col < ncols; col++)
    for (size_t row = 0; row < nrows; row++) {
      vec[nrows * col + row] = mat(row, col);
    }
  return vec;
}

template <class T>
matrix<T> make_matrix(const std::vector<std::vector<T>> &mat) {
  size_t nrows = mat.size();
  size_t ncols = mat[0].size();
  matrix<T> ret(nrows, ncols);
  for (size_t row = 0; row < nrows; row++)
    for (size_t col = 0; col < nrows; col++) {
      ret(row, col) = mat[row][col];
    }
  return ret;
}

template <class T>
matrix<T> transpose(const matrix<T> &A) {
  // Transposes a Matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<T> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = A(i, j);
    }
  }
  return temp;
}

template <class T>
matrix<std::complex<T>> dagger(const matrix<std::complex<T>> &A) {
  // Take the Hermitian conjugate of a complex matrix
  const size_t cols = A.GetColumns(), rows = A.GetRows();
  matrix<std::complex<T>> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = std::conj(A(i, j));
    }
  }
  return temp;
}

template <class T>
matrix<std::complex<T>> conjugate(const matrix<std::complex<T>> &A) {
  // Take the complex conjugate of a complex matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<std::complex<T>> temp(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(i, j) = std::conj(A(i, j));
    }
  }
  return temp;
}

template <class T>
matrix<T> stacked_matrix(const std::vector<matrix<T>> &mmat) {
  size_t size_of_controls =
      mmat[0]
          .GetRows(); // or GetColumns, as these matrices are (should be) square
  size_t number_of_controls = mmat.size();

  // Pack vector of matrices into single (stacked) matrix ... note: matrix dims:
  // rows = (stacked_rows x size_of_controls) where:
  //     stacked_rows is the number of control matrices * the size (#rows or
  //     #columns) of each control matrix size_of_controls is the #rows (or
  //     #columns) of each control matrix
  uint_t stacked_rows =
      number_of_controls *
      size_of_controls; // Used only for clarity in allocating the matrix

  cmatrix_t stacked_matrix(stacked_rows, size_of_controls);
  for (uint_t row = 0; row < stacked_rows; row++)
    for (uint_t col = 0; col < size_of_controls; col++)
      stacked_matrix(row, col) = {0.0, 0.0};

  for (uint_t mmat_number = 0; mmat_number < mmat.size(); mmat_number++) {
    for (uint_t row = 0; row < size_of_controls; row++) {
      for (uint_t col = 0; col < size_of_controls; col++) {
        stacked_matrix(mmat_number * size_of_controls + row, col) =
            mmat[mmat_number](row, col);
      }
    }
  }
  return stacked_matrix;
}

template <class T>
std::vector<T> matrix_diagonal(const matrix<T> &mat) {
  std::vector<T> vec;
  size_t size = std::min(mat.GetRows(), mat.GetColumns());
  vec.resize(size, 0.);
  for (size_t i = 0; i < size; i++)
    vec[i] = mat(i, i);
  return vec;
}

template <class T>
matrix<T> &transpose_inplace(matrix<T> &A) {
  // Transposes a Matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = i + 1; j < cols; j++) {
      const auto tmp = A(i, j);
      A(i, j) = A(j, i);
      A(j, i) = tmp;
    }
  }
  return A;
}

template <class T>
matrix<std::complex<T>> &dagger_inplace(matrix<std::complex<T>> &A) {
  // Take the Hermitian conjugate of a complex matrix
  const size_t cols = A.GetColumns(), rows = A.GetRows();
  matrix<std::complex<T>> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    A(i, i) = conj(A(i, i));
    for (size_t j = i + 1; j < cols; j++) {
      const auto tmp = conj(A(i, j));
      A(i, j) = conj(A(j, i));
      A(j, i) = tmp;
    }
  }
  return A;
}

template <class T>
matrix<std::complex<T>> &conj_inplace(matrix<std::complex<T>> &A) {
  // Take the complex conjugate of a complex matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      A(i, j) = conj(A(i, j));
    }
  }
  return A;
}

template <class T>
T trace(const matrix<T> &A) {
  // Finds the trace of a matrix
  size_t rows = A.GetRows(), cols = A.GetColumns();
  if (rows != cols) {
    throw std::invalid_argument("MU::trace: matrix is not square");
  }
  T temp = 0.0;
  for (size_t i = 0; i < rows; i++) {
    temp = temp + A(i, i);
  }
  return temp;
}

template <class T>
matrix<T> partial_trace_a(const matrix<T> &rho, size_t dimA) {
  // Traces out first system (dimension dimA) of composite Hilbert space
  size_t rows = rho.GetRows(), cols = rho.GetColumns();
  if (rows != cols) {
    throw std::invalid_argument("MU::partial_trace_a: matrix is not square");
  }
  if (rows % dimA != 0) {
    throw std::invalid_argument(
        "MU::partial_trace_a: dim(rho)/dim(system b) is not an integer");
  }
  size_t dimB = rows / dimA;
  matrix<T> rhoB(dimB, dimB);
  T temp = 0.0;
  for (size_t i = 0; i < dimB; i++) {
    for (size_t j = 0; j < dimB; j++) {
      for (size_t k = 0; k < dimA; k++) {
        temp = temp + rho(i + dimB * k, j + dimB * k);
      }
      rhoB(i, j) = temp;
      temp = 0.0;
    }
  }
  return rhoB;
}

template <class T>
matrix<T> partial_trace_b(const matrix<T> &rho, size_t dimB) {
  // Traces out second system (dimension dimB) of composite Hilbert space
  size_t rows = rho.GetRows(), cols = rho.GetColumns();
  if (rows != cols) {
    throw std::invalid_argument("MU::partial_trace_b: matrix is not square");
  }
  if (rows % dimB != 0) {
    throw std::invalid_argument(
        "MU::partial_trace_b: dim(rho)/dim(system a) is not an integer");
  }
  size_t dimA = rows / dimB;
  matrix<T> rhoA(dimA, dimA);
  T temp = 0.0;
  for (size_t i = 0; i < dimA; i++) {
    size_t offsetX = i * dimB;
    for (size_t j = 0; j < dimA; j++) {
      size_t offsetY = j * dimB;
      for (size_t k = 0; k < dimB; k++) {
        temp = temp + rho(offsetX + k, offsetY + k);
      }
      rhoA(i, j) = temp;
      temp = 0.0;
    }
  }
  return rhoA;
}

template <class T>
matrix<T> tensor_product(const matrix<T> &A, const matrix<T> &B) {
  // Works out the TensorProduct of two matricies A tensor B
  // Note that if A is i x j and B is p x q then A \otimes B is an ip x jq
  // rmatrix

  // If A or B is empty it will return the other matrix
  if (A.size() == 0)
    return B;
  if (B.size() == 0)
    return A;

  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  size_t rows_new = rows1 * rows2, cols_new = cols1 * cols2, n, m;
  matrix<T> temp(rows_new, cols_new);
  // a11 B, a12 B ... a1j B
  // ai1 B, ai2 B ... aij B
  for (size_t i = 0; i < rows1; i++) {
    for (size_t j = 0; j < cols1; j++) {
      for (size_t p = 0; p < rows2; p++) {
        for (size_t q = 0; q < cols2; q++) {
          n = i * rows2 + p;
          m = j * cols2 + q; //  0 (0 + 1)  + 1*dimb=2 + (0 + 1 )  (j*dimb+q)
          temp(n, m) = A(i, j) * B(p, q);
        }
      }
    }
  }
  return temp;
}

template <class T>
matrix<T> unitary_superop(const matrix<T> &mat) {
  return tensor_product(conjugate(mat), mat);
}

template <class T>
matrix<T> kraus_superop(const std::vector<matrix<T>> &kmats) {
  if (kmats.empty())
    return matrix<T>();
  matrix<T> mat = unitary_superop(kmats[0]);
  for (size_t i = 1; i < kmats.size(); ++i) {
    mat += unitary_superop(kmats[i]);
  }
  return mat;
}

template <class T>
matrix<T> concatenate(const matrix<T> &A, const matrix<T> &B, uint_t axis) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("Utils::concatenate: axis must be 0 or 1");
  }
  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  matrix<T> temp = A;
  if (axis == 0) {
    if (cols1 != cols2) {
      throw std::invalid_argument("Utils::concatenate: axis must be 0 or 1");
    }
    temp.resize(rows1 + rows2, cols1);
    for (size_t i = 0; i < rows2; i++)
      for (size_t j = 0; j < cols1; j++)
        temp(rows1 + i, j) = B(i, j);
  } else if (axis == 1) {
    if (rows1 != rows2) {
      throw std::invalid_argument(
          "Utils::concatenate: the 2 matrices have a different number of rows");
    }
    temp.resize(rows1, cols1 + cols2);
    for (size_t i = 0; i < rows1; i++)
      for (size_t j = 0; j < cols2; j++)
        temp(i, cols1 + j) = B(i, j);
  }
  return temp;
}

template <class T>
void split(const matrix<T> &A, matrix<T> &B, matrix<T> &C, uint_t axis) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("Utils::split: axis must be 0 or 1");
  }
  size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<T> temp = A;
  if (axis == 0) {
    if (rows % 2 != 0) {
      throw std::invalid_argument("Utils::split: can't split matrix A by rows");
    }
    B.resize(rows / 2, cols);
    C.resize(rows / 2, cols);
    for (size_t i = 0; i < rows / 2; i++) {
      for (size_t j = 0; j < cols; j++) {
        B(i, j) = A(i, j);
        C(i, j) = A(i + rows / 2, j);
      }
    }
  } else if (axis == 1) {
    if (cols % 2 != 0) {
      throw std::invalid_argument(
          "Utils::split: can't split matrix A by columns");
    }
    B.resize(rows, cols / 2);
    C.resize(rows, cols / 2);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols / 2; j++) {
        B(i, j) = A(i, j);
        C(i, j) = A(i, j + cols / 2);
      }
    }
  }
}

template <class T>
matrix<T> elementwise_multiplication(const matrix<T> &A, const matrix<T> &B) {
  // Works out an elementwise multiplication of two matrices A, B
  // If A or B is empty it will return the other matrix
  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  if (rows1 != rows2 || cols1 != cols2) {
    throw std::invalid_argument(
        "Utils::elementwise_multiplication: matrices have different sizes");
  }
  matrix<T> temp(rows1, cols1);
  for (size_t i = 0; i < rows1; i++)
    for (size_t j = 0; j < cols1; j++)
      temp(i, j) = A(i, j) * B(i, j);
  return temp;
}

template <class T>
T sum(const matrix<T> &A) {
  T temp = 0;
  for (uint_t i = 0; i < A.size(); i++)
    temp += A[i];
  return temp;
}

template <class T>
bool is_square(const matrix<T> &mat) {
  if (mat.GetRows() != mat.GetColumns())
    return false;
  return true;
}

template <class T>
bool is_diagonal(const matrix<T> &mat) {
  // Check if row-matrix for diagonal
  if (mat.GetRows() == 1 && mat.GetColumns() > 0)
    return true;
  return false;
}

template <class T>
bool is_equal(const matrix<T> &mat1, const matrix<T> &mat2, double threshold) {

  // Check matrices are same shape
  const auto nrows = mat1.GetRows();
  const auto ncols = mat1.GetColumns();
  if (nrows != mat2.GetRows() || ncols != mat2.GetColumns())
    return false;

  // Check matrices are equal on an entry by entry basis
  double delta = 0;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      delta += std::real(std::abs(mat1(i, j) - mat2(i, j)));
    }
  }
  return (delta < threshold);
}

template <class T>
bool is_diagonal(const matrix<T> &mat, double threshold) {
  // Check U matrix is identity
  const auto nrows = mat.GetRows();
  const auto ncols = mat.GetColumns();
  if (nrows != ncols)
    return false;
  for (size_t i = 0; i < nrows; i++)
    for (size_t j = 0; j < ncols; j++)
      if (i != j && std::real(std::abs(mat(i, j))) > threshold)
        return false;
  return true;
}

template <class T>
std::pair<bool, double> is_identity_phase(const matrix<T> &mat,
                                          double threshold) {

  // To check if identity we first check we check that:
  // 1. U(0,0) = exp(i * theta)
  // 2. U(i, i) = U(0, 0)
  // 3. U(i, j) = 0 for j != i
  auto failed = std::make_pair(false, 0.0);

  // Check condition 1.
  const auto u00 = mat(0, 0);
  // if (std::norm(std::abs(u00) - 1.0) > threshold)
  //   return failed;
  if (std::norm(std::abs(u00) - 1.0) > threshold) {
    return failed;
  }
  const auto theta = std::arg(u00);

  // Check conditions 2 and 3
  double delta = 0.;
  const auto nrows = mat.GetRows();
  const auto ncols = mat.GetColumns();
  if (nrows != ncols)
    return failed;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      auto val = (i == j) ? std::norm(mat(i, j) - u00) : std::norm(mat(i, j));
      if (val > threshold) {
        return failed; // fail fast if single entry differs
      } else
        delta += val; // accumulate difference
    }
  }
  // Check small errors didn't accumulate
  if (delta > threshold) {
    return failed;
  }
  // Otherwise we pass
  return std::make_pair(true, theta);
}

template <class T>
bool is_identity(const matrix<T> &mat, double threshold) {
  // Check mat(0, 0) == 1
  if (std::norm(mat(0, 0) - T(1)) > threshold)
    return false;
  // If this passes now use is_identity_phase (and we know
  // phase will be zero).
  return is_identity_phase(mat, threshold).first;
}

template <class T>
bool is_diagonal_identity(const matrix<T> &mat, double threshold) {
  // Check U matrix is identity
  if (is_diagonal(mat, threshold) == false)
    return false;
  double delta = 0.;
  const auto ncols = mat.GetColumns();
  for (size_t j = 0; j < ncols; j++) {
    delta += std::real(std::abs(mat(0, j) - 1.0));
  }
  return (delta < threshold);
}

template <class T>
bool is_unitary(const matrix<T> &mat, double threshold) {
  size_t nrows = mat.GetRows();
  size_t ncols = mat.GetColumns();
  // Check if diagonal row-matrix
  if (nrows == 1) {
    for (size_t j = 0; j < ncols; j++) {
      double delta = std::abs(1.0 - std::real(std::abs(mat(0, j))));
      if (delta > threshold)
        return false;
    }
    return true;
  }
  // Check U matrix is square
  if (nrows != ncols)
    return false;
  // Check U matrix is unitary
  const matrix<T> check = mat * dagger(mat);
  return is_identity(check, threshold);
}

template <class T>
bool is_hermitian_matrix(const matrix<T> &mat, double threshold) {
  return is_equal(mat, dagger(mat), threshold);
}

template <class T>
bool is_symmetrix(const matrix<T> &mat, double threshold) {
  return is_equal(mat, transpose(mat), threshold);
}

template <class T>
bool is_cptp_kraus(const std::vector<matrix<T>> &mats, double threshold) {
  const auto dim = mats[0].GetRows();
  matrix<T> cptp(dim, dim);
  for (const auto &mat : mats) {
    cptp += dagger(mat) * mat;
  }
  return is_identity(cptp, threshold);
}

//==============================================================================
// Implementations: Vector functions
//==============================================================================

template <class T>
bool is_unit_vector(const std::vector<T> &vec, double threshold) {
  return (std::abs(norm<T>(vec) - 1.0) < threshold);
}

template <typename T>
std::vector<std::complex<T>> conjugate(const std::vector<std::complex<T>> &v) {
  std::vector<std::complex<T>> ret;
  std::transform(
      v.cbegin(), v.cend(), std::back_inserter(ret),
      [](const std::complex<T> &c) -> std::complex<T> { return std::conj(c); });
  return ret;
}

template <typename T>
Vector<std::complex<T>> conjugate(const Vector<std::complex<T>> &v) {
  Vector<std::complex<T>> ret(v.size(), false);
  std::transform(
      v.data(), v.data() + v.size(), ret.data(),
      [](const std::complex<T> &c) -> std::complex<T> { return std::conj(c); });
  return ret;
}

template <typename T>
double norm(const std::vector<T> &vec) {
  double val = 0.0;
  for (const auto &v : vec) {
    val += std::real(v * std::conj(v));
  }
  return std::sqrt(val);
}

template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra) {
  const uint_t d1 = ket.size();
  const uint_t d2 = bra.size();
  matrix<T> ret(d1, d2);
  for (uint_t i = 0; i < d1; i++)
    for (uint_t j = 0; j < d2; j++) {
      ret(i, j) = ket[i] * std::conj(bra[j]);
    }
  return ret;
}

template <typename T>
std::vector<T> tensor_product(const std::vector<T> &vec1,
                              const std::vector<T> &vec2) {
  std::vector<T> ret;
  ret.reserve(vec1.size() * vec2.size());
  for (const auto &a : vec1)
    for (const auto &b : vec2) {
      ret.push_back(a * b);
    }
  return ret;
}

template <typename T>
Vector<T> tensor_product(const Vector<T> &vec1, const Vector<T> &vec2) {
  const auto SZ1 = vec1.size();
  const auto SZ2 = vec2.size();
  Vector<T> ret(SZ1 * SZ2, false);
  for (size_t i = 0; i < SZ1; ++i)
    for (size_t j = 0; j < SZ2; ++j) {
      ret[SZ2 * i + j] = vec1[i] * vec2[j];
    }
  return ret;
}

template <typename T1, typename T2>
std::vector<T1> scalar_multiply(const std::vector<T1> &vec, T2 val) {
  std::vector<T1> ret;
  ret.reserve(vec.size());
  for (const auto &elt : vec) {
    ret.push_back(val * elt);
  }
  return ret;
}

template <typename T1, typename T2>
std::vector<T1> &scalar_multiply_inplace(std::vector<T1> &vec, T2 val) {
  for (auto &elt : vec) {
    elt = val * elt; // use * incase T1 doesn't have *= method
  }
  return vec;
}

double &chop_inplace(double &val, double epsilon) {
  if (std::abs(val) < epsilon)
    val = 0.;
  return val;
}

std::complex<double> &chop_inplace(std::complex<double> &val, double epsilon) {
  val.real(chop(val.real(), epsilon));
  val.imag(chop(val.imag(), epsilon));
  return val;
}

template <typename T>
std::vector<T> &chop_inplace(std::vector<T> &vec, double epsilon) {
  if (epsilon > 0.)
    for (auto &v : vec)
      chop_inplace(v, epsilon);
  return vec;
}

double chop(double val, double epsilon) {
  return (std::abs(val) < epsilon) ? 0. : val;
}

float chop(float val, double epsilon) {
  return (std::abs(val) < epsilon) ? 0. : val;
}

template <typename T>
std::complex<T> chop(std::complex<T> val, double epsilon) {
  return {chop(val.real(), epsilon), chop(val.imag(), epsilon)};
}

template <typename T>
std::vector<T> chop(const std::vector<T> &vec, double epsilon) {
  std::vector<T> tmp;
  tmp.reserve(vec.size());
  for (const auto &v : vec)
    tmp.push_back(chop(v, epsilon));
  return tmp;
}

template <class T>
void combine(std::vector<T> &lhs, const std::vector<T> &rhs) {
  // if lhs is empty, set it to be rhs vector
  if (lhs.size() == 0) {
    lhs = rhs;
    return;
  }
  // if lhs is not empty rhs must be same size
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument(
        "Utils::combine (vectors are not same length.)");
  }
  for (size_t j = 0; j < lhs.size(); ++j) {
    lhs[j] += rhs[j];
  }
}

template <typename T>
std::map<std::string, T> vec2ket(const std::vector<T> &vec, double epsilon,
                                 uint_t base) {

  bool hex_output = false;
  if (base == 16) {
    hex_output = true;
    base = 2; // If hexadecimal strings we convert to bin first
  }
  // check vector length
  size_t dim = vec.size();
  double n = std::log(dim) / std::log(base);
  uint_t nint = std::trunc(n);
  if (std::abs(nint - n) > 1e-5) {
    std::stringstream ss;
    ss << "vec2ket (vector dimension " << dim << " is not of size " << base
       << "^n)";
    throw std::invalid_argument(ss.str());
  }
  std::map<std::string, T> ketmap;
  for (size_t k = 0; k < dim; ++k) {
    T val = chop(vec[k], epsilon);
    if (std::abs(val) > epsilon) {
      std::string key =
          (hex_output) ? Utils::int2hex(k) : Utils::int2string(k, base, nint);
      ketmap.insert({key, val});
    }
  }
  return ketmap;
}

template <typename T>
std::map<std::string, T> vec2ket(const T *const vec, uint_t dim, double epsilon,
                                 uint_t base) {
  bool hex_output = false;
  if (base == 16) {
    hex_output = true;
    base = 2; // If hexadecimal strings we convert to bin first
  }

  double n = std::log(dim) / std::log(base);
  uint_t nint = std::trunc(n);
  if (std::abs(nint - n) > 1e-5) {
    std::stringstream ss;
    ss << "vec2ket (vector dimension " << dim << " is not of size " << base
       << "^n)";
    throw std::invalid_argument(ss.str());
  }

  std::map<std::string, T> ketmap;
  for (size_t k = 0; k < dim; ++k) {
    T val = chop(vec[k], epsilon);
    if (std::abs(val) > epsilon) {
      std::string key =
          (hex_output) ? Utils::int2hex(k) : Utils::int2string(k, base, nint);
      ketmap.insert({key, val});
    }
  }
  return ketmap;
}

//==============================================================================
// Implementations: Bit conversions
//==============================================================================

std::string &format_hex_inplace(std::string &hex) {
  // make abcdef and x lower case
  std::transform(hex.begin(), hex.end(), hex.begin(), ::tolower);
  // check if 0x prefix is present, add if it isn't
  std::string prefix = hex.substr(0, 2);
  if (prefix != "0x")
    hex = "0x" + hex;
  // delete leading zeros Eg 0x001 -> 0x1
  hex.erase(2, std::min(hex.find_first_not_of("0", 2) - 2, hex.size() - 3));
  return hex;
}

std::string format_hex(const std::string &hex) {
  std::string tmp = hex;
  format_hex_inplace(tmp);
  return tmp;
}

std::string &padleft_inplace(std::string &s, char c, size_t min_length) {
  auto l = s.size();
  if (l < min_length)
    s = std::string(min_length - l, c) + s;
  return s;
}

std::string padleft(const std::string &s, char c, size_t min_length) {
  std::string tmp = s;
  return padleft_inplace(tmp, c, min_length);
}

reg_t int2reg(uint_t n, uint_t base) {
  reg_t ret;
  while (n >= base) {
    ret.push_back(n % base);
    n /= base;
  }
  ret.push_back(n); // last case n < base;
  return ret;
}

reg_t int2reg(uint_t n, uint_t base, uint_t minlen) {
  reg_t ret = int2reg(n, base);
  if (ret.size() < minlen) // pad vector with zeros
    ret.resize(minlen);
  return ret;
}

reg_t hex2reg(std::string str) {
  reg_t reg;
  std::string prefix = str.substr(0, 2);
  if (prefix == "0x" || prefix == "0X") { // Hexadecimal
    str.erase(0, 2);                      // remove '0x';
    size_t length = (str.size() % 8) + 32 * (str.size() / 8);
    reg.reserve(length);
    while (str.size() > 8) {
      unsigned long hex = stoull(str.substr(str.size() - 8), nullptr, 16);
      reg_t tmp = int2reg(hex, 2, 32);
      std::move(tmp.begin(), tmp.end(), back_inserter(reg));
      str.erase(str.size() - 8);
    }
    if (str.size() > 0) {
      reg_t tmp = int2reg(stoul(str, nullptr, 16), 2, 0);
      std::move(tmp.begin(), tmp.end(), back_inserter(reg));
    }
    return reg;
  } else {
    throw std::runtime_error(std::string("invalid hexadecimal"));
  }
}

std::string hex2bin(std::string str, bool prefix) {
  // empty case
  if (str.empty())
    return std::string();

  // If string starts with 0b prob prefix
  if (str.size() > 1 && str.substr(0, 2) == "0x") {
    str.erase(0, 2);
  }

  // We go via long integer conversion, so we process 64-bit chunks at
  // a time
  const size_t block = 16;
  const size_t len = str.size();
  const size_t chunks = len / block;
  const size_t remain = len % block;

  // Initialize output string
  std::string bin = (prefix) ? "0b" : "";

  // Start with remain
  if (remain != 0)
    bin += int2string(std::stoull(str.substr(0, remain), nullptr, 16), 2);

  for (size_t j = 0; j < chunks; ++j) {
    std::string part = int2string(
        std::stoull(str.substr(remain + j * block, block), nullptr, 16), 2, 64);
    bin += part;
  }
  return bin;
}

std::string bin2hex(std::string str, bool prefix) {
  // empty case
  if (str.empty())
    return std::string();

  // If string starts with 0b prob prefix
  if (str.size() > 1 && str.substr(0, 2) == "0b") {
    str.erase(0, 2);
  }

  // We go via long integer conversion, so we process 64-bit chunks at
  // a time
  const size_t bin_block = 64;
  const size_t hex_block = bin_block / 4;
  const size_t len = str.size();
  const size_t chunks = len / bin_block;
  const size_t remain = len % bin_block;

  // initialize output string
  std::string hex = (prefix) ? "0x" : "";

  // Add remainder
  if (remain > 0) {
    // Add remainder
    std::stringstream ss;
    ss << std::hex << std::stoull(str.substr(0, remain), nullptr, 2);
    hex += ss.str();
  }

  // Add > 64 bit chunks
  if (chunks > 0) {
    std::string part;
    // Add last 64-bit chunk
    {
      std::stringstream ss;
      ss << std::hex << std::stoull(str.substr(remain, bin_block), nullptr, 2);
      part = ss.str();
      if (remain > 0) {
        part.insert(0, hex_block - part.size(), '0'); // pad out zeros
      }
      hex += part;
    }
    // Add any additional chunks
    for (size_t j = 1; j < chunks; ++j) {
      std::stringstream ss; // clear string stream
      ss << std::hex
         << std::stoull(str.substr(remain + j * bin_block, bin_block), nullptr,
                        2);
      part = ss.str();
      part.insert(0, hex_block - part.size(), '0');
      hex += part;
    }
  }
  return hex;
}

uint_t reg2int(const reg_t &reg, uint_t base) {
  uint_t ret = 0;
  if (base == 2) {
    // For base-2 use bit-shifting
    for (size_t j = 0; j < reg.size(); j++)
      if (reg[j])
        ret += (1ULL << j);
  } else {
    // For other bases use exponentiation
    for (size_t j = 0; j < reg.size(); j++)
      if (reg[j] > 0)
        ret += reg[j] * static_cast<uint_t>(pow(base, j));
  }
  return ret;
}

std::string int2string(uint_t n, uint_t base) {
  if (base < 2 || base > 10) {
    throw std::invalid_argument(
        "Utils::int2string base must be between 2 and 10.");
  }
  if (n < base)
    return std::to_string(n);
  else
    return int2string(n / base, base) + std::to_string(n % base);
}

std::string int2string(uint_t n, uint_t base, uint_t minlen) {
  std::string tmp = int2string(n, base);
  return padleft_inplace(tmp, '0', minlen);
}

#ifdef _MSC_VER
#ifdef _WIN64
#define POPCNT __popcnt64
#else
#define POPCNT __popcnt
#endif
#define INTRINSIC_PARITY 1
inline bool _intrinsic_parity(uint_t x) { return (POPCNT(x) & one); }
inline uint_t _instrinsic_weight(uint_t x) { return (POPCNT(x)); }
#endif
#ifdef __GNUC__
#define INTRINSIC_PARITY 1
inline bool _intrinsic_parity(uint_t x) {
  return (__builtin_popcountll(x) & one);
}
inline uint_t _instrinsic_weight(uint_t x) { return (__builtin_popcountll(x)); }
#endif
#ifdef _CLANG_
#if __has__builtin(__builtin_popcount)
#define INTRINSIC_PARITY 1
inline bool _intrinsic_parity(uint_t x) {
  return (__builtin_popcountll(x) & one);
}
inline uint_t _instrinsic_weight(uint_t x) { return (__builtin_popcountll(x)); }
#endif
#endif
// Implementation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
static bool _naive_parity(uint_t x) {
  uint_t c; // c accumulates the total bits set in x
  for (c = 0; x; c++) {
    x &= (x - 1); // clear the least significant bit set
  }
  return (c & one);
}
static uint_t _naive_weight(uint_t x) {
  auto count = x;
  count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
  count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
  count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
  count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
  count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
  count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
  return count;
}

#ifdef INTRINSIC_PARITY
bool (*hamming_parity)(uint_t) = is_avx2_supported() ? &_intrinsic_parity
                                                     : &_naive_parity;
uint_t (*popcount)(uint_t) = is_avx2_supported() ? &_instrinsic_weight
                                                 : &_naive_weight;
#else
bool (*hamming_parity)(uint_t) = &_naive_parity;
uint_t (*popcount)(uint_t) = &_naive_weight;
#endif
size_t get_system_memory_mb();
size_t get_system_memory_mb() {
  size_t total_physical_memory = 0;
#if defined(__linux__)
  size_t pages = (size_t)sysconf(_SC_PHYS_PAGES);
  size_t page_size = (size_t)sysconf(_SC_PAGE_SIZE);
  total_physical_memory = pages * page_size;
#elif defined(__APPLE__)
  size_t len = sizeof(total_physical_memory);
  sysctlbyname("hw.memsize", &total_physical_memory, &len, NULL, 0);
#elif defined(_WIN64) || defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  total_physical_memory = status.ullTotalPhys;
#endif
  return total_physical_memory >> 20;
}

// apply OpenMP parallel loop to lambda function if enabled
template <typename Lambda>
void apply_omp_parallel_for(bool enabled, int_t i_begin, int_t i_end,
                            Lambda &func, int nthreads = 0) {
  if (enabled) {
    if (nthreads > 0) {
#pragma omp parallel for num_threads(nthreads)
      for (int_t i = i_begin; i < i_end; i++)
        func(i);
    } else {
#pragma omp parallel for
      for (int_t i = i_begin; i < i_end; i++)
        func(i);
    }
  } else {
    for (int_t i = i_begin; i < i_end; i++)
      func(i);
  }
}

// apply OpenMP parallel loop to lambda function and return reduced double if
// enabled
template <typename Lambda>
double apply_omp_parallel_for_reduction(bool enabled, int_t i_begin,
                                        int_t i_end, Lambda &func,
                                        int nthreads = 0) {
  double val = 0.0;
  if (enabled) {
    if (nthreads > 0) {
#pragma omp parallel for reduction(+ : val) num_threads(nthreads)
      for (int_t i = i_begin; i < i_end; i++)
        val += func(i);
    } else {
#pragma omp parallel for reduction(+ : val)
      for (int_t i = i_begin; i < i_end; i++)
        val += func(i);
    }
  } else {
    for (int_t i = i_begin; i < i_end; i++)
      val += func(i);
  }
  return val;
}

// apply OpenMP parallel loop to lambda function and return reduced integer if
// enabled
template <typename Lambda>
int apply_omp_parallel_for_reduction_int(bool enabled, int_t i_begin,
                                         int_t i_end, Lambda &func,
                                         int nthreads = 0) {
  int val = 0;
  if (enabled) {
    if (nthreads > 0) {
#pragma omp parallel for reduction(+ : val) num_threads(nthreads)
      for (int_t i = i_begin; i < i_end; i++)
        val += func(i);
    } else {
#pragma omp parallel for reduction(+ : val)
      for (int_t i = i_begin; i < i_end; i++)
        val += func(i);
    }
  } else {
    for (int_t i = i_begin; i < i_end; i++)
      val += func(i);
  }
  return val;
}

//------------------------------------------------------------------------------
} // end namespace Utils
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
