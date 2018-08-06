/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    utils.hpp
 * @brief   Utility functions that didn't below anywhere else
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_framework_utils_hpp_
#define _aer_framework_utils_hpp_

#include <sstream>

#include "framework/types.hpp"

namespace AER {
namespace Utils {

//------------------------------------------------------------------------------
// Static Matrices
//------------------------------------------------------------------------------

class Matrix {
  public:
    // Single-qubit gates
    const static cmatrix_t I;
    const static cmatrix_t X;
    const static cmatrix_t Y;
    const static cmatrix_t Z;
    const static cmatrix_t H;
    const static cmatrix_t S;
    const static cmatrix_t T;
    const static cmatrix_t X90;

    // Two-qubit gates
    const static cmatrix_t CX;
    const static cmatrix_t CZ;
    const static cmatrix_t SWAP;
    const static cmatrix_t CR; // TODO
    const static cmatrix_t CR90; // TODO
    
    // Identity Matrix
    static cmatrix_t Identity(size_t dim);

    // Single-qubit waltz gates
    static cmatrix_t U1(double lam);
    static cmatrix_t U2(double phi, double lam);
    static cmatrix_t U3(double theta, double phi, double lam);
  };

//------------------------------------------------------------------------------
// Matrix Functions
//------------------------------------------------------------------------------

// Vector conversion
template<class T> matrix<T> make_matrix(const std::vector<std::vector<T>> &mat);
template<class T> matrix<T> devectorize_matrix(const std::vector<T> &vec);
template<class T> std::vector<T> vectorize_matrix(const matrix<T> &mat);

// Transformations
template <class T> matrix<T> transpose(const matrix<T> &A); // TOO
template <class T>
matrix<std::complex<T>> dagger(const matrix<std::complex<T>> &A);
template <class T>
matrix<std::complex<T>> conjugate(const matrix<std::complex<T>> &A);

// Tracing
template <class T> T trace(const matrix<T> &A);
template <class T> matrix<T> partial_trace_a(const matrix<T> &rho, size_t dimA);
template <class T> matrix<T> partial_trace_b(const matrix<T> &rho, size_t dimB);

// Tensor product
template <class T> matrix<T> tensor_product(const matrix<T> &A, const matrix<T> &B);

//------------------------------------------------------------------------------
// Vector functions
//------------------------------------------------------------------------------

// Return the matrix formed by taking the outproduct of two vector |ket><bra|
template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra);

template <typename T>
inline matrix<T> projector(const std::vector<T> &ket) {return outer_product(ket, ket);};

// Truncate the first argument its absolute value is less than epsilon
// this function returns a refernce to the chopped first argument
template <typename T>
std::vector<T> multiply(const std::vector<T> &vec, T val);

// Truncate the first argument its absolute value is less than epsilon
// this function returns a refernce to the chopped first argument
double &chop_inplace(double &val, double epsilon);

double chop(double val, double epsilon);
// As above for complex first arguments

template <typename T>
std::complex<T> &chop_inplace(std::complex<T> &val, double epsilon);

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
// epsilon determins the threshold for which small values will be removed from the output
// The base of the ket (2 to 10) specifies the subsystem dimension and the base
// of the dit-string labels.
template <typename T>
std::map<std::string, T> vec2ket(const std::vector<T> &vec, double epsilon, uint_t base = 2);

//------------------------------------------------------------------------------
// Bit Conversions
//------------------------------------------------------------------------------

// Format a hex string so that it has a prefix "0x", abcdef chars are lowercase
// and leading zeros are removed
// Example: 0010A -> 0x10a
std::string format_hex(const std::string &hex);
std::string& format_hex_inplace(std::string &hex);

// Convert integers and hexadecimals to register vectors
reg_t int2reg(uint_t n, uint_t base = 2);
reg_t int2reg(uint_t n, uint_t base, uint_t minlen);
reg_t hex2reg(std::string str);

// Convert bit-strings to hex-strings
// TODO: add prefix case 0b for input and 0x for output
std::string bin2hex(const std::string bin);

// Convert hex-strings to bit-strings
// TODO: add prefix case 0x for input and 0b for output
std::string hex2bin(const std::string bs);

// Convert 64-bit unsigned integers to dit-string (dit base = 2 to 10)
std::string int2string(uint_t n, uint_t base = 2);
std::string int2string(uint_t n, uint_t base, uint_t length);

// Convert integers to bit-strings
inline std::string int2bin(uint_t n) {return int2string(n, 2);};
inline std::string int2bin(uint_t n, uint_t length) {return int2string(n, 2, length);};

// Convert integers to hex-strings
inline std::string int2hex(uint_t n) {return bin2hex(int2bin(n));};

//==============================================================================
// Implementations: Static Matrices
//==============================================================================

const cmatrix_t Matrix::I = make_matrix<complex_t>({{{1, 0}, {0, 0}},
                                                    {{0, 0}, {1, 0}}});
  
const cmatrix_t Matrix::X = make_matrix<complex_t>({{{0, 0}, {1, 0}},
                                                    {{1, 0}, {0, 0}}});

const cmatrix_t Matrix::Y = make_matrix<complex_t>({{{0, 0}, {0, -1}},
                                                    {{0, 1}, {0, 0}}});

const cmatrix_t Matrix::Z = make_matrix<complex_t>({{{1, 0}, {0, 0}}, 
                                                    {{0, 0}, {-1, 0}}});

const cmatrix_t Matrix::S = make_matrix<complex_t>({{{1, 0}, {0, 0}},
                                                    {{0, 0}, {0, 1}}});

const cmatrix_t Matrix::T = make_matrix<complex_t>({{{1, 0}, {0, 0}},
                                                    {{0, 0}, {1 / std::sqrt(2), 1 / std::sqrt(2)}}});

const cmatrix_t Matrix::H = make_matrix<complex_t>({{{1 / std::sqrt(2.), 0}, {1 / std::sqrt(2.), 0}},
                                                    {{1 / std::sqrt(2.), 0}, {-1 / std::sqrt(2.), 0}}});

const cmatrix_t Matrix::X90 = make_matrix<complex_t>({{{1. / std::sqrt(2.), 0}, {0, -1. / std::sqrt(2.)}},
                                                      {{0, -1. / std::sqrt(2.)}, {1. / std::sqrt(2.), 0}}});

const cmatrix_t Matrix::CX = make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                      {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
                                                      {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                                      {{0, 0}, {1, 0}, {0, 0}, {0, 0}}});

const cmatrix_t Matrix::CZ = make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                      {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                                      {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                                      {{0, 0}, {0, 0}, {0, 0}, {-1, 0}}});

const cmatrix_t Matrix::SWAP = make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                                        {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                                        {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                                        {{0, 0}, {0, 0}, {0, 0}, {1, 0}}});

// TODO const cmatrix_t Matrix::CR = ...
// TODO const cmatrix_t Matrix::CR90 = ...

cmatrix_t Matrix::Identity(size_t dim) {
  cmatrix_t mat(dim, dim);
  for (size_t j=0; j<dim; j++)
    mat(j, j) = {1.0, 0.0};
  return mat;
}


cmatrix_t Matrix::U1(double lambda) {
  cmatrix_t mat(2, 2);
  mat(0, 0) = {1., 0.};
  mat(1, 1) = std::exp(complex_t(0., lambda));
  return mat;
}


cmatrix_t Matrix::U2(double phi, double lambda) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  const complex_t invsqrt2(1. / std::sqrt(2), 0.);
  mat(0, 0) = invsqrt2;
  mat(0, 1) = -std::exp(i * lambda) * invsqrt2;
  mat(1, 0) = std::exp(i * phi) * invsqrt2;
  mat(1, 1) = std::exp(i * (phi + lambda)) * invsqrt2;
  return mat;
}


cmatrix_t Matrix::U3(double theta, double phi, double lambda) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  mat(0, 0) = std::cos(theta / 2.);
  mat(0, 1) = -std::exp(i * lambda) * std::sin(theta / 2.);
  mat(1, 0) = std::exp(i * phi) * std::sin(theta / 2.);
  mat(1, 1) = std::exp(i * (phi + lambda)) * std::cos(theta / 2.);
  return mat;
}


//==============================================================================
// Implementations: Matrix functions
//==============================================================================

template<class T>
matrix<T> devectorize_matrix(const std::vector<T>& vec) {
  size_t dim = std::sqrt(vec.size());
  matrix<T> mat(dim, dim);
  for (size_t col=0; col < dim; col++)
    for (size_t row=0; row < dim; row++) {
      mat(row, col) = vec[dim * col + row];
    }
  return mat;
}


template<class T>
std::vector<T> vectorize_matrix(const matrix<T>& mat) {
  std::vector<T> vec;
  vec.resize(mat.size(), 0.);
  size_t nrows = mat.GetRows();
  size_t ncols = mat.GetColumns();
  for (size_t col=0; col < ncols; col++)
    for (size_t row=0; row < nrows; row++) {
      vec[nrows * col + row] = mat(row, col);
    }
  return vec;
}


template <class T>
matrix<T> make_matrix(const std::vector<std::vector<T>> & mat) {
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
  size_t rows = A.GetRows(), cols = A.GetColumns();
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
  size_t cols = A.GetColumns(), rows = A.GetRows();
  matrix<std::complex<T>> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = conj(A(i, j));
    }
  }
  return temp;
}


template <class T>
matrix<std::complex<T>> conj(const matrix<std::complex<T>> &A) {
  // Take the complex conjugate of a complex matrix
  size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<std::complex<T>> temp(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(i, j) = conj(A(i, j));
    }
  }
  return temp;
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
    throw std::invalid_argument("MU::partial_trace_a: dim(rho)/dim(system b) is not an integer");
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
    throw std::invalid_argument("MU::partial_trace_b: dim(rho)/dim(system a) is not an integer");
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


//==============================================================================
// Implementations: Vector functions
//==============================================================================

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
std::vector<T> multiply(const std::vector<T> &vec, T val) {
  std::vector<T> ret;
  ret.reserve(vec.size());
  for (const auto &elt : vec) {
    ret.push_back(val * elt);
  }
  return ret;
}


double &chop_inplace(double &val, double epsilon) {
  if (std::abs(val) < epsilon)
    val = 0.;
  return val;
}


template <typename T>
std::complex<T> &chop_inplace(std::complex<T> &val, double epsilon) {
  val.real(chop_inplace(val.real(), epsilon));
  val.imag(chop_inplace(val.imag(), epsilon));
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
    throw std::invalid_argument("Utils::combine (vectors are not same length.)");
  }
  for (size_t j=0; j < lhs.size(); ++j) {
    lhs[j] += rhs[j];
  }
}


template <typename T>
std::map<std::string, T> vec2ket(const std::vector<T> &vec, double epsilon, uint_t base) {
  // check vector length
  size_t dim = vec.size();
  double n = std::log(dim) / std::log(base);
  uint_t nint = std::trunc(n);
  if (std::abs(nint - n) > 1e-5) {
    std::stringstream ss;
    ss << "vec2ket (vector dimension " << dim << " is not of size " << base << "^n)";
    throw std::invalid_argument(ss.str());
  }
  std::map<std::string, T> ketmap;
  for (size_t k = 0; k < dim; ++k) {
    T tmp = chop(vec[k], epsilon);
    if (std::abs(tmp) > epsilon) { 
      ketmap.insert({Utils::int2string(k, base, nint), tmp});
    }
  }
  return ketmap;
}


//==============================================================================
// Implementations: Bit conversions
//==============================================================================

std::string& format_hex_inplace(std::string &hex) {
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
    str.erase(0, 2); // remove '0x';
    size_t length = (str.size() % 8) + 32 * (str.size() / 8);
    reg.reserve(length);
    while (str.size() > 8) {
      unsigned long hex = stoull(str.substr(str.size() - 8), 0, 16);
      reg_t tmp = int2reg(hex, 2, 32);
      std::move(tmp.begin(), tmp.end(), back_inserter(reg));
      str.erase(str.size() - 8);
    }
    if (str.size() > 0) {
      reg_t tmp = int2reg(stoul(str, 0, 16), 2, 0);
      std::move(tmp.begin(), tmp.end(), back_inserter(reg));
    }
    return reg;
  } else {
    throw std::runtime_error(std::string("invalid hexadecimal"));
  }
}


std::string hex2bin(std::string str) {
  // empty case
  if (str.empty())
    return std::string();

  // If string starts with 0b prob prefix
  if (str.size() > 1 && str.substr(0, 2) == "0x") {
    str.erase(0, 2);
  }

  // We go via long integer conversion, so we process 64-bit chunks at
  // a time
  const size_t block = 8;
  const size_t len = str.size();
  const size_t chunks = len / block;
  const size_t remain = len % block;

  // Start with remain
  std::string bin = "0b" + int2string(std::stoull(str.substr(0, remain), nullptr, 16), 2);
  for (size_t j=0; j < chunks; ++j) {
    std::string part = int2string(std::stoull(str.substr(remain + j * block, block), nullptr, 16), 2, 64);
    bin += part;
  }
  return bin;
}


std::string bin2hex(std::string str) {
  // empty case
  if (str.empty())
    return std::string();

  // If string starts with 0b prob prefix
  if (str.size() > 1 && str.substr(0, 2) == "0b") {
    str.erase(0, 2);
  }

  // We go via long integer conversion, so we process 64-bit chunks at
  // a time
  const size_t block = 64;
  const size_t len = str.size();
  const size_t chunks = len / block;
  const size_t remain = len % block;

  // Start with remain
  std::stringstream ss;
  ss << std::hex << std::stoull(str.substr(0, remain), nullptr, 2);
  std::string hex = "0x" + ss.str(); // the return string
  for (size_t j=0; j < chunks; ++j) {
    ss.str(std::string()); // clear string stream
    ss << std::hex << std::stoull(str.substr(remain + j * block, block), nullptr, 2);
    std::string part = ss.str();
    part.insert(0, block - part.size(), '0'); // pad out zeros
    hex += part;
  }
  return hex;
}


std::string int2string(uint_t n, uint_t base) {
  if (base < 2 || base > 10) {
    throw std::invalid_argument("Utils::int2string base must be between 2 and 10.");
  }
  if (n < base)
    return std::to_string(n);
  else
    return int2string(n / base, base) + std::to_string(n % base);
}

std::string int2string(uint_t n, uint_t base, uint_t minlen) {
  std::string s = int2string(n, base);
  auto l = s.size();
  if (l < minlen)
    s = std::string(minlen - l, '0') + s;
  return s;
}


//------------------------------------------------------------------------------
} // end namespace Utils
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
