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

/*
Dependences: BLAS
Brief Discription: This is my Matrix class. It works only with real/complex
matrices and stores the entries in  column-major-order. In column-major storage
the columns are stored one after the other. The linear offset p from the
beginning of the array to any given element A(i,j) can then be computed as:
   p = j*Nrows+i
where Nrows is the number of rows in the matrix. Hence, one scrolls down
rows and moves to a new column once the last row is reached. More precisely, if
i wanted to know the i and j associtated with a given p then i would use
  i=p %Nrows
  j= floor(p/Nrows)

Multiplication is done with the C wrapper of the fortran blas library.
*/

#ifndef _aer_framework_matrix_hpp
#define _aer_framework_matrix_hpp

#include <complex>
#include <iostream>
#include <vector>
#include <array>

/*******************************************************************************
 *
 * BLAS headers
 *
 ******************************************************************************/

const std::array<char, 3> Trans = {'N', 'T', 'C'};
/*  Trans (input) CHARACTER*1.
                On entry, TRANSA specifies the form of op( A ) to be used in the
   matrix multiplication as follows:
                        = 'N' no transpose;
                        = 'T' transpose of A;
                        = 'C' hermitian conjugate of A.
*/

#ifdef __cplusplus
extern "C" {
#endif

//===========================================================================
// Prototypes for level 3 BLAS
//===========================================================================

// Single-Precison Real Matrix-Vector Multiplcation
void sgemv_(const char *TransA, const size_t *M, const size_t *N,
            const float *alpha, const float *A, const size_t *lda,
            const float *x, const size_t *incx, const float *beta, float *y,
            const size_t *lincy);
// Double-Precison Real Matrix-Vector Multiplcation
void dgemv_(const char *TransA, const size_t *M, const size_t *N,
            const double *alpha, const double *A, const size_t *lda,
            const double *x, const size_t *incx, const double *beta, double *y,
            const size_t *lincy);
// Single-Precison Complex Matrix-Vector Multiplcation
void cgemv_(const char *TransA, const size_t *M, const size_t *N,
            const std::complex<float> *alpha, const std::complex<float> *A,
            const size_t *lda, const std::complex<float> *x, const size_t *incx,
            const std::complex<float> *beta, std::complex<float> *y,
            const size_t *lincy);
// Double-Precison Real Matrix-Vector Multiplcation
void zgemv_(const char *TransA, const size_t *M, const size_t *N,
            const std::complex<double> *alpha, const std::complex<double> *A,
            const size_t *lda, const std::complex<double> *x,
            const size_t *incx, const std::complex<double> *beta,
            std::complex<double> *y, const size_t *lincy);
// Single-Precison Real Matrix-Matrix Multiplcation
void sgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const float *alpha,
            const float *A, const size_t *lda, const float *B,
            const size_t *lba, const float *beta, float *C, size_t *ldc);
// Double-Precison Real Matrix-Matrix Multiplcation
void dgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const double *alpha,
            const double *A, const size_t *lda, const double *B,
            const size_t *lba, const double *beta, double *C, size_t *ldc);
// Single-Precison Complex Matrix-Matrix Multiplcation
void cgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const size_t *lda,
            const std::complex<float> *B, const size_t *ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            size_t *ldc);
// Double-Precison Complex Matrix-Matrix Multiplcation
void zgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const size_t *lda,
            const std::complex<double> *B, const size_t *ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            size_t *ldc);
#ifdef __cplusplus
}
#endif

/*******************************************************************************
 *
 * Matrix Class
 *
 ******************************************************************************/

template <class T>
T* malloc_array(size_t size) {
  return reinterpret_cast<T*>(malloc(sizeof(T) * size));
}

template <class T>
T* calloc_array(size_t size) {
  return reinterpret_cast<T*>(calloc(size, sizeof(T)));
}


template <class T> // define a class template
class matrix {
  // friend functions get to use the private varibles of the class as well as
  // have different classes as inputs
  template <class S>
  friend std::ostream &
  operator<<(std::ostream &output,
             const matrix<S> &A); // overloading << to output a matrix
  template <class S>
  friend std::istream &
  operator>>(std::istream &input,
             const matrix<S> &A); // overloading >> to read in a matrix

  // Multiplication (does not catch an error of a S1 = real and S2 being
  // complex)
  template <class S1, class S2>
  friend matrix<S1>
  operator*(const S2 &beta,
            const matrix<S1> &A); // multiplication by a scalar beta*A
  template <class S1, class S2>
  friend matrix<S1>
  operator*(const matrix<S1> &A,
            const S2 &beta); // multiplication by a scalar A*beta
  // Single-Precison Matrix Multiplication
  friend matrix<float>
  operator*(const matrix<float> &A,
            const matrix<float> &B); // real matrix multiplication A*B
  friend matrix<std::complex<float>> operator*(
      const matrix<std::complex<float>> &A,
      const matrix<std::complex<float>> &B); // complex matrix multplication A*B
  friend matrix<std::complex<float>>
  operator*(const matrix<float> &A,
            const matrix<std::complex<float>>
                &B); // real-complex matrix multplication A*B
  friend matrix<std::complex<float>>
  operator*(const matrix<std::complex<float>> &A,
            const matrix<float> &B); // real-complex matrix multplication A*B
  // Double-Precision Matrix Multiplication
  friend matrix<double>
  operator*(const matrix<double> &A,
            const matrix<double> &B); // real matrix multiplication A*B
  friend matrix<std::complex<double>>
  operator*(const matrix<std::complex<double>> &A,
            const matrix<std::complex<double>>
                &B); // complex matrix multplication A*B
  friend matrix<std::complex<double>>
  operator*(const matrix<double> &A,
            const matrix<std::complex<double>>
                &B); // real-complex matrix multplication A*B
  friend matrix<std::complex<double>>
  operator*(const matrix<std::complex<double>> &A,
            const matrix<double> &B); // real-complex matrix multplication A*B
  // Single-Precision Matrix-Vector Multiplication
  friend std::vector<float> operator*(const matrix<float> &A,
                                      const std::vector<float> &v);
  friend std::vector<std::complex<float>>
  operator*(const matrix<std::complex<float>> &A,
            const std::vector<std::complex<float>> &v);
  // Double-Precision Matrix-Vector Multiplication
  friend std::vector<double> operator*(const matrix<double> &A,
                                       const std::vector<double> &v);
  friend std::vector<std::complex<double>>
  operator*(const matrix<std::complex<double>> &A,
            const std::vector<std::complex<double>> &v);

public:
  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  // Construct an empty matrix
  matrix() = default;

  // Construct a matrix of specified size
  // If `fill=True` the matrix will be initialized with all values in zero
  // if `fill=False` the matrix entries will be in an indeterminant state
  // and should have their values assigned before use.
  matrix(size_t rows, size_t cols, bool fill = true);

  // Copy construct a matrix
  matrix(const matrix<T> &other);

  // Move construct a matrix
  matrix(matrix<T>&& other) noexcept;

  // Destructor
  virtual ~matrix() { free(data_); }

  //-----------------------------------------------------------------------
  // Assignment
  //-----------------------------------------------------------------------

  // Copy assignment
  matrix<T> &operator=(const matrix<T> &other);

  // Move assignment
  matrix<T> &operator=(matrix<T> &&other) noexcept;

  // Copy and cast assignment
  template <class S>
  matrix<T> &operator=(const matrix<S> &other);

  //-----------------------------------------------------------------------
  // Buffer conversion
  //-----------------------------------------------------------------------

  // Copy construct a matrix from C-array buffer
  // The buffer should have size = rows * cols.
  static matrix<T> copy_from_buffer(size_t rows, size_t cols, const T* buffer);

  // Move construct a matrix from C-array buffer
  // The buffer should have size = rows * cols.
  static matrix<T> move_from_buffer(size_t rows, size_t cols, T* buffer);

  // Copy matrix to a new C-array
  T* copy_to_buffer() const;

  // Move matrix to a C-array
  T* move_to_buffer();

  //-----------------------------------------------------------------------
  // Element access
  //-----------------------------------------------------------------------

  // Addressing elements by vector representation
  T& operator[](size_t element);
  const T& operator[](size_t element) const;

  // Addressing elements by matrix representation
  T& operator()(size_t row, size_t col);
  const T& operator()(size_t row, size_t col) const;

  // Access the array data pointer
  const T* data() const noexcept { return data_; }
  T* data() noexcept { return data_; }

  //-----------------------------------------------------------------------
  // Other methods
  //-----------------------------------------------------------------------

  // Return the size of the underlying array
  size_t size() const { return size_; }
  
  // Return True if size == 0
  bool empty() const { return size_ == 0; }

  // Clear used memory
  void clear();

  // Fill with constant value
  void fill(const T& val);

  // Resize the matrix and reset to zero if different size
  void initialize(size_t row, size_t col); 

  // Resize the matrix keeping current values
  void resize(size_t row, size_t col); 

  // overloading functions.
  matrix<T> operator+(const matrix<T> &A);
  matrix<T> operator-(const matrix<T> &A);
  matrix<T> operator+(const matrix<T> &A) const;
  matrix<T> operator-(const matrix<T> &A) const;
  matrix<T> &operator+=(const matrix<T> &A);
  matrix<T> &operator-=(const matrix<T> &A);

  //-----------------------------------------------------------------------
  // Legacy methods
  //-----------------------------------------------------------------------

  // Member Functions
  size_t GetColumns() const; // gives the number of columns
  size_t GetRows() const;    // gives the number of rows
  size_t GetLD() const;      // gives the leading dimension -- number of rows

protected:
  size_t rows_ = 0, cols_ = 0, size_ = 0, LD_ = 0;
  // rows_ and cols_ are the rows and columns of the matrix
  // size_ = rows*colums dimensions of the vector representation
  // LD is the leading dimeonsion and for Column major order is in general eqaul
  // to rows
  
  // the ptr to the vector containing the matrix
  T* data_ = nullptr;
};

/*******************************************************************************
 *
 * Matrix class: methods
 *
 ******************************************************************************/

//-----------------------------------------------------------------------
// Constructors
//-----------------------------------------------------------------------

template <class T>
matrix<T>::matrix(size_t rows, size_t cols, bool fill)
    : rows_(rows), cols_(cols), size_(rows * cols), LD_(rows),
      data_((fill) ? calloc_array<T>(size_) : malloc_array<T>(size_)) {}

template <class T>
matrix<T>::matrix(const matrix<T> &other) : matrix(other.rows_, other.cols_, false) {
  std::copy(other.data_, other.data_ + other.size_, data_);
}

template <class T>
matrix<T>::matrix(matrix<T>&& other) noexcept
  : rows_(other.rows_), cols_(other.cols_), size_(other.size_), LD_(rows_),
    data_(other.data_) {
  other.data_ = nullptr;
}

//-----------------------------------------------------------------------
// Assignment
//-----------------------------------------------------------------------

template <class T>
matrix<T>& matrix<T>::operator=(matrix<T>&& other) noexcept {
  free(data_);
  rows_ = other.rows_;
  cols_ = other.cols_;
  size_ = rows_ * cols_;
  LD_ = other.LD_;
  data_ = other.data_;
  other.data_ = nullptr;
  return *this;
}

template <class T>
matrix<T> &matrix<T>::operator=(const matrix<T> &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) { 
    // size delete re-construct
    // the matrix
    free(data_);
    rows_ = other.rows_;
    cols_ = other.cols_;
    size_ = rows_ * cols_;
    LD_ = other.LD_;
    data_ = malloc_array<T>(size_);
  }
  std::copy(other.data_, other.data_ + size_, data_);
  return *this;
}

template <class T>
template <class S>
inline matrix<T> &matrix<T>::operator=(const matrix<S> &other) {

  if (rows_ != other.GetRows() ||
      cols_ != other.GetColumns()) {
    free(data_);
    rows_ = other.GetRows();
    cols_ = other.GetColumns();
    size_ = rows_ * cols_;
    LD_ = other.GetLD();
    data_ = malloc_array<T>(size_);
  }
  for (size_t p = 0; p < size_; p++) {
    data_[p] = T(other[p]);
  }
  return *this;
}

//-----------------------------------------------------------------------
// Buffer conversion
//-----------------------------------------------------------------------

template <class T>
matrix<T> matrix<T>::copy_from_buffer(size_t rows, size_t cols, const T* buffer) {
  matrix<T> ret;
  ret.size_ = rows * cols;
  ret.rows_ = rows;
  ret.cols_ = cols;
  ret.LD_ = rows;
  ret.data_ = calloc_array<T>(ret.size_);
  std::copy(buffer, buffer + ret.size_, ret.data_);
  return ret;
}

template <class T>
matrix<T> matrix<T>::move_from_buffer(size_t rows, size_t cols, T* buffer) {
  matrix<T> ret;
  ret.size_ = rows * cols;
  ret.rows_ = rows;
  ret.cols_ = cols;
  ret.LD_ = rows;
  ret.data_ = buffer;
  return ret;
}

template <class T>
T* matrix<T>::copy_to_buffer() const {
  T* buffer = malloc_array<T>(size_);
  std::copy(data_, data_ + size_, buffer);
  return buffer;
}

template <class T>
T* matrix<T>::move_to_buffer() {
  T* buffer = data_;
  data_ = nullptr;
  size_ = 0;
  rows_ = 0;
  cols_ = 0;
  return buffer;
}

//-----------------------------------------------------------------------
// Element access
//-----------------------------------------------------------------------

template <class T>
T& matrix<T>::operator[](size_t p) {
#ifdef DEBUG
  if (p >= size_) {
    std::cerr
        << "error: matrix class operator []: Matrix subscript out of bounds"
        << std::endl;
    exit(1);
  }
#endif
  return data_[p];
}
template <class T>
const T& matrix<T>::operator[](size_t p) const {
#ifdef DEBUG
  if (p >= size_) {
    std::cerr << "Error: matrix class operator [] const: Matrix subscript out "
                 "of bounds"
              << std::endl;
    exit(1);
  }
#endif
  return data_[p];
}

template <class T>
T& matrix<T>::operator()(size_t i, size_t j) {
#ifdef DEBUG
  if (i >= rows_ || j >= cols_) {
    std::cerr
        << "Error: matrix class operator (): Matrices subscript out of bounds"
        << std::endl;
    exit(1);
  }
#endif
  return data_[j * rows_ + i];
}

template <class T>
const T& matrix<T>::operator()(size_t i, size_t j) const {
#ifdef DEBUG
  if (i >= rows_ || j >= cols_) {
    std::cerr << "Error: matrix class operator ()   const: Matrices subscript "
                 "out of bounds"
              << std::endl;
    exit(1);
  }
#endif
  return data_[j * rows_ + i];
}

template <class T>
void matrix<T>::clear() {
  if (!data_ || !size_)
    return;
  rows_ = cols_ = size_ = 0;
  free(data_);
}

template <class T> inline void matrix<T>::initialize(size_t rows, size_t cols) {
  if (rows_ != rows || cols_ != cols) {
    free(data_);
    rows_ = rows;
    cols_ = cols;
    size_ = rows_ * cols_;
    LD_ = rows;
    data_ = calloc_array<T>(size_);
  }
}

template <class T>
void matrix<T>::fill(const T& val) {
  std::fill(data_, data_ + size_, val);
}

template <class T>
void matrix<T>::resize(size_t rows, size_t cols) {
  if (rows_ == rows && cols_ == cols)
    return;
  size_ = rows * cols;
  T *tempmat = malloc_array<T>(size_);
  for (size_t j = 0; j < cols; j++)
    for (size_t i = 0; i < rows; i++)
      if (i < rows_ && j < cols_)
        tempmat[j * rows + i] = data_[j * rows_ + i];
      else
        tempmat[j * rows + i] = 0.0;
  free(data_);
  LD_ = rows_ = rows;
  cols_ = cols;
  data_ = tempmat;
}

template <class T> inline size_t matrix<T>::GetRows() const {
  // returns the rows of the matrix
  return rows_;
}
template <class T> inline size_t matrix<T>::GetColumns() const {
  // returns the colums of the matrix
  return cols_;
}
template <class T> inline size_t matrix<T>::GetLD() const {
  // returns the leading dimension
  return LD_;
}

template <class T> inline matrix<T> matrix<T>::operator+(const matrix<T> &A) {
// overloads the + for matrix addition, can this be more efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator +: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.data_[p] = data_[p] + A.data_[p];
  }
  return temp;
}
template <class T> inline matrix<T> matrix<T>::operator-(const matrix<T> &A) {
// overloads the - for matrix substraction, can this be more efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator -: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.data_[p] = data_[p] - A.data_[p];
  }
  return temp;
}
template <class T>
inline matrix<T> matrix<T>::operator+(const matrix<T> &A) const {
// overloads the + for matrix addition if it is a const matrix, can this be more
// efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr << "Error: matrix class operator + const: Matrices are not the "
                 "same size"
              << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.data_[p] = data_[p] + A.data_[p];
  }
  return temp;
}
template <class T>
inline matrix<T> matrix<T>::operator-(const matrix<T> &A) const {
// overloads the - for matrix substraction, can this be more efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr << "Error: matrix class operator - const: Matrices are not the "
                 "same size"
              << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.data_[p] = data_[p] - A.data_[p];
  }
  return temp;
}
template <class T> inline matrix<T> &matrix<T>::operator+=(const matrix<T> &A) {
// overloads the += for matrix addition and assignment, can this be more
// efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator +=: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  for (size_t p = 0; p < size_; p++) {
    data_[p] += A.data_[p];
  }
  return *this;
}
template <class T> inline matrix<T> &matrix<T>::operator-=(const matrix<T> &A) {
// overloads the -= for matrix subtraction and assignement, can this be more
// efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator -=: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  for (size_t p = 0; p < size_; p++) {
    data_[p] -= A.data_[p];
  }
  return *this;
}

/*******************************************************************************
 *
 * Matrix class: Friend Functions
 *
 ******************************************************************************/
template <class T>
std::ostream &operator<<(std::ostream &out, const matrix<T> &A) {
  out << "[";
  size_t last_row = A.rows_ - 1;
  size_t last_col = A.cols_ - 1;
  for (size_t i = 0; i < A.rows_; ++i) {
    out << "[";
    for (size_t j = 0; j < A.cols_; ++j) {
      out << A.data_[i + A.rows_ * j];
      if (j != last_col)
        out << ", ";
    }
    out << "]";
    if (i != last_row)
      out << ", ";
  }
  out << "]";
  return out;
}

template <class T>
std::istream &operator>>(std::istream &input, const matrix<T> &A) {
  // overloads the >> to read in a row into column format
  for (size_t j = 0; j < A.cols_; j++) {
    for (size_t i = 0; i < A.rows_; i++) {
      input >> A.data_[j * A.rows_ + i];
    }
  }
  return input;
}
template <class S1, class S2>
matrix<S1> operator*(const matrix<S1> &A, const S2 &beta) {
  // overloads A*beta
  size_t rows = A.rows_, cols = A.cols_;
  matrix<S1> temp(rows, cols);
  for (size_t j = 0; j < cols; j++) {
    for (size_t i = 0; i < rows; i++) {
      temp(i, j) = beta * A(i, j);
    }
  }
  return temp;
}
template <class S1, class S2>
matrix<S1> operator*(const S2 &beta, const matrix<S1> &A) {
  // overloads beta*A
  size_t rows = A.rows_, cols = A.cols_;
  matrix<S1> temp(rows, cols);
  for (size_t j = 0; j < cols; j++) {
    for (size_t i = 0; i < rows; i++) {
      temp(i, j) = beta * A(i, j);
    }
  }
  return temp;
}

// Operator overloading with BLAS functions
inline matrix<double> operator*(const matrix<double> &A,
                                const matrix<double> &B) {
  // overloads A*B for real matricies and uses the blas dgemm routine
  // cblas_dgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<double> C(A.rows_, B.cols_);
  double alpha = 1.0, beta = 0.0;
  dgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.data_,
         &A.LD_, B.data_, &B.LD_, &beta, C.data_, &C.LD_);
  // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, 1.0, A.data_, A.LD_, B.data_, B.LD_, 0.0, C.data_, C.LD_);
  return C;
}
inline matrix<float> operator*(const matrix<float> &A, const matrix<float> &B) {
  // overloads A*B for real matricies and uses the blas sgemm routine
  // cblas_sgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<float> C(A.rows_, B.cols_);
  float alpha = 1.0, beta = 0.0;
  sgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.data_,
         &A.LD_, B.data_, &B.LD_, &beta, C.data_, &C.LD_);
  // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, 1.0, A.data_, A.LD_, B.data_, B.LD_, 0.0, C.data_, C.LD_);
  return C;
}
inline matrix<std::complex<float>>
operator*(const matrix<std::complex<float>> &A,
          const matrix<std::complex<float>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<float>> C(A.rows_, B.cols_);
  std::complex<float> alpha = 1.0, beta = 0.0;
  cgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.data_,
         &A.LD_, B.data_, &B.LD_, &beta, C.data_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, &alpha, A.data_, A.LD_, B.data_, B.LD_, &beta, C.data_, C.LD_);
  return C;
}
inline matrix<std::complex<double>>
operator*(const matrix<std::complex<double>> &A,
          const matrix<std::complex<double>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<double>> C(A.rows_, B.cols_);
  std::complex<double> alpha = 1.0, beta = 0.0;
  zgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.data_,
         &A.LD_, B.data_, &B.LD_, &beta, C.data_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, &alpha, A.data_, A.LD_, B.data_, B.LD_, &beta, C.data_, C.LD_);
  return C;
}
inline matrix<std::complex<float>>
operator*(const matrix<float> &A, const matrix<std::complex<float>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<float>> C(A.rows_, B.cols_), Ac(A.rows_, A.cols_);
  Ac = A;
  std::complex<float> alpha = 1.0, beta = 0.0;
  cgemm_(&Trans[0], &Trans[0], &Ac.rows_, &B.cols_, &Ac.cols_, &alpha, Ac.data_,
         &Ac.LD_, B.data_, &B.LD_, &beta, C.data_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ac.rows_, B.cols_,
  // Ac.cols_, &alpha, Ac.data_, Ac.LD_, B.data_, B.LD_, &beta, C.data_, C.LD_);
  return C;
}
inline matrix<std::complex<double>>
operator*(const matrix<double> &A, const matrix<std::complex<double>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<double>> C(A.rows_, B.cols_), Ac(A.rows_, A.cols_);
  Ac = A;
  std::complex<double> alpha = 1.0, beta = 0.0;
  zgemm_(&Trans[0], &Trans[0], &Ac.rows_, &B.cols_, &Ac.cols_, &alpha, Ac.data_,
         &Ac.LD_, B.data_, &B.LD_, &beta, C.data_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ac.rows_, B.cols_,
  // Ac.cols_, &alpha, Ac.data_, Ac.LD_, B.data_, B.LD_, &beta, C.data_, C.LD_);
  return C;
}
inline matrix<std::complex<float>>
operator*(const matrix<std::complex<float>> &A, const matrix<float> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<float>> C(A.rows_, B.cols_), Bc(B.rows_, B.cols_);
  Bc = B;
  std::complex<float> alpha = 1.0, beta = 0.0;
  cgemm_(&Trans[0], &Trans[0], &A.rows_, &Bc.cols_, &A.cols_, &alpha, A.data_,
         &A.LD_, Bc.data_, &Bc.LD_, &beta, C.data_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, Bc.cols_,
  // A.cols_, &alpha, A.data_, A.LD_, Bc.data_, Bc.LD_, &beta, C.data_, C.LD_);
  return C;
}
inline matrix<std::complex<double>>
operator*(const matrix<std::complex<double>> &A, const matrix<double> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<double>> C(A.rows_, B.cols_), Bc(B.rows_, B.cols_);
  Bc = B;
  std::complex<double> alpha = 1.0, beta = 0.0;
  zgemm_(&Trans[0], &Trans[0], &A.rows_, &Bc.cols_, &A.cols_, &alpha, A.data_,
         &A.LD_, Bc.data_, &Bc.LD_, &beta, C.data_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, Bc.cols_,
  // A.cols_, &alpha, A.data_, A.LD_, Bc.data_, Bc.LD_, &beta, C.data_, C.LD_);
  return C;
}

// Single-Precision Real
inline std::vector<float> operator*(const matrix<float> &A,
                                    const std::vector<float> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<float> y(A.rows_);
  float alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  sgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.data_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}
// Double-Precision Real
inline std::vector<double> operator*(const matrix<double> &A,
                                     const std::vector<double> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<double> y(A.rows_);
  double alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  dgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.data_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}
// Single-Precision Complex
inline std::vector<std::complex<float>>
operator*(const matrix<std::complex<float>> &A,
          const std::vector<std::complex<float>> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<std::complex<float>> y(A.rows_);
  std::complex<float> alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  cgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.data_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}
// Double-Precision Complex
inline std::vector<std::complex<double>>
operator*(const matrix<std::complex<double>> &A,
          const std::vector<std::complex<double>> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<std::complex<double>> y(A.rows_);
  std::complex<double> alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  zgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.data_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}

//------------------------------------------------------------------------------
// end _matrix_h_
//------------------------------------------------------------------------------
#endif
