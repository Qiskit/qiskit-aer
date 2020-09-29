/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */



#ifndef _qv_transformer_
#define _qv_transformer_

#include "framework/utils.hpp"
#include "simulators/statevector/indexes.hpp"


namespace AER {
namespace QV {

template <typename T> using cvector_t = std::vector<std::complex<T>>;


template <typename Container, typename data_t = double, typename Derived = void>
class Transformer {

// TODO: This class should have the indexes.hpp moved inside it

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  Transformer(Container& data, size_t data_size, size_t omp_threads = 1) :
    data_(data),
    data_size_(data_size),
    omp_threads_(omp_threads) {}

  // Prevent initializing with temporary since container is stored as a reference
  Transformer(Container&& data, size_t data_size, size_t omp_threads) = delete;

  virtual ~Transformer() = default;

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits,
                    const cvector_t<double> &mat);
  
  // Apply a N-qubit diagonal matrix to a array container
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits,
                             const cvector_t<double> &diag);

protected:

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  template <size_t N>
  void apply_matrix_n(const reg_t &qubits,
                      const cvector_t<double> &mat);

  // Specialized single qubit apply matrix function
  void apply_matrix_1(const uint_t qubit,
                      const cvector_t<double> &mat);

  // Specialized single qubit apply matrix function
  void apply_diagonal_matrix_1(const uint_t qubit,
                               const cvector_t<double> &mat);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------
  Container& data_;
  size_t data_size_;
  uint_t omp_threads_ = 1;             // Number of threads used by OpenMP

  // Convert a matrix to a different type
  // TODO: this makes an unnecessary copy when data_t = double.
  inline cvector_t<data_t> convert(const cvector_t<double>& v) const {
    cvector_t<data_t> ret(v.size());
    for (size_t i = 0; i < v.size(); ++i)
      ret[i] = v[i];
    return ret;
  }



};


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/


template <typename Container, typename data_t, typename Derived>
void Transformer<Container, data_t, Derived>::apply_matrix(const reg_t &qubits,
                                                           const cvector_t<double> &mat) {
  // Static array optimized lambda functions
  switch (qubits.size()) {
    case 1: return apply_matrix_1(qubits[0], mat);
    case 2: return apply_matrix_n<2>(qubits, mat);
    case 3: return apply_matrix_n<3>(qubits, mat);
    case 4: return apply_matrix_n<4>(qubits, mat);
    case 5: return apply_matrix_n<5>(qubits, mat);
    case 6: return apply_matrix_n<6>(qubits, mat);
    case 7: return apply_matrix_n<7>(qubits, mat);
    case 8: return apply_matrix_n<8>(qubits, mat);
    case 9: return apply_matrix_n<9>(qubits, mat);
    case 10: return apply_matrix_n<10>(qubits, mat);
    case 11: return apply_matrix_n<11>(qubits, mat);
    case 12: return apply_matrix_n<12>(qubits, mat);
    case 13: return apply_matrix_n<13>(qubits, mat);
    case 14: return apply_matrix_n<14>(qubits, mat);
    case 15: return apply_matrix_n<15>(qubits, mat);
    case 16: return apply_matrix_n<16>(qubits, mat);
    case 17: return apply_matrix_n<17>(qubits, mat);
    case 18: return apply_matrix_n<18>(qubits, mat);
    case 19: return apply_matrix_n<19>(qubits, mat);
    case 20: return apply_matrix_n<20>(qubits, mat);
    default: {
      throw std::runtime_error(
          "Maximum size of apply matrix is a 20-qubit matrix.");
    }
  }
}

template <typename Container, typename data_t, typename Derived>
template <size_t N>
void Transformer<Container, data_t, Derived>::apply_matrix_n(const reg_t &qs,
                                                             const cvector_t<double> &mat) {  
  const size_t DIM = 1ULL << N;
  auto func = [&](const areg_t<1UL << N> &inds, const cvector_t<data_t> &_mat)->void {
    std::array<std::complex<data_t>, 1ULL<<N> cache;
    for (size_t i = 0; i < DIM; i++) {
      const auto ii = inds[i];
      cache[i] = data_[ii];
      data_[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++)
        data_[inds[i]] += _mat[i + DIM * j] * cache[j];
  };
  areg_t<N> qubits;
  std::copy_n(qs.begin(), N, qubits.begin());
  apply_lambda(0, data_size_, omp_threads_, func, qubits, convert(mat));
}


template <typename Container, typename data_t, typename Derived>
void Transformer<Container, data_t, Derived>::apply_matrix_1(const uint_t qubit,
                                                             const cvector_t<double>& mat) {

  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix_1(qubit, diag);
    return;
  }

  // Convert qubit to array register for lambda functions
  areg_t<1> qubits = {{qubit}};

  // Check if anti-diagonal matrix and if so use optimized lambda
  if(mat[0] == 0.0 && mat[3] == 0.0) {
    if (mat[1] == 1.0 && mat[2] == 1.0) {
      // X-matrix
      auto func = [&](const areg_t<2> &inds)->void {
        std::swap(data_[inds[0]], data_[inds[1]]);
      };
      apply_lambda(0, data_size_, omp_threads_, func, qubits);
      return;
    }
    if (mat[2] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto func = [&](const areg_t<2> &inds, const cvector_t<data_t> &_mat)->void {
        data_[inds[1]] = _mat[1] * data_[inds[0]];
        data_[inds[0]] = 0.0;
      };
      apply_lambda(0, data_size_, omp_threads_, func, qubits, convert(mat));
      return;
    }
    if (mat[1] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto func = [&](const areg_t<2> &inds, const cvector_t<data_t> &_mat)->void {
        data_[inds[0]] = _mat[2] * data_[inds[1]];
        data_[inds[1]] = 0.0;
      };
      apply_lambda(0, data_size_, omp_threads_, func, qubits, convert(mat));
      return;
    }
    // else we have a general anti-diagonal matrix
    auto func = [&](const areg_t<2> &inds, const cvector_t<data_t> &_mat)->void {
      const std::complex<data_t> cache = data_[inds[0]];
      data_[inds[0]] = _mat[2] * data_[inds[1]];
      data_[inds[1]] = _mat[1] * cache;
    };
    apply_lambda(0, data_size_, omp_threads_, func, qubits, convert(mat));
    return;
  }

  auto func = [&](const areg_t<2> &inds, const cvector_t<data_t> &_mat)->void {
    const auto cache = data_[inds[0]];
    data_[inds[0]] = _mat[0] * cache + _mat[2] * data_[inds[1]];
    data_[inds[1]] = _mat[1] * cache + _mat[3] * data_[inds[1]];
  };

  apply_lambda(0, data_size_, omp_threads_, func, qubits, convert(mat));
}


template <typename Container, typename data_t, typename Derived>
void Transformer<Container, data_t, Derived>::apply_diagonal_matrix(const reg_t &qubits,
                                                                    const cvector_t<double> &diag) {
  if (qubits.size() == 1) {
    apply_diagonal_matrix_1(qubits[0], diag);
    return;
  }

  const size_t N = qubits.size();
  auto func = [&](const areg_t<2> &inds, const cvector_t<data_t> &_diag)->void {
    for (int_t i = 0; i < 2; ++i) {
      const int_t k = inds[i];
      int_t iv = 0;
      for (int_t j = 0; j < N; j++)
        if ((k & (1ULL << qubits[j])) != 0)
          iv += (1ULL << j);
      if (_diag[iv] != (data_t) 1.0)
        data_[k] *= _diag[iv];
    }
  };
  apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubits[0]}}), convert(diag));
}

template <typename Container, typename data_t, typename Derived>
void Transformer<Container, data_t, Derived>::apply_diagonal_matrix_1(const uint_t qubit,
                                                  const cvector_t<double>& diag) {
  // TODO: This should be changed so it isn't checking doubles with ==
  if (diag[0] == 1.0) {  // [[1, 0], [0, z]] matrix
    if (diag[1] == 1.0)
      return; // Identity

    if (diag[1] == std::complex<double>(0., -1.)) { // [[1, 0], [0, -i]]
      auto func = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[1] == std::complex<double>(0., 1.)) {
      // [[1, 0], [0, i]]
      auto func = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[0] == 0.0) {
      // [[1, 0], [0, 0]]
      auto func = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        data_[inds[1]] = 0.0;
      };
     apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    // general [[1, 0], [0, z]]
    auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const auto k = inds[1];
      data_[k] *= _mat[1];
    };
    apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
    return;
  } else if (diag[1] == 1.0) {
    // [[z, 0], [0, 1]] matrix
    if (diag[0] == std::complex<double>(0., -1.)) {
      // [[-i, 0], [0, 1]]
      auto func = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[0] == std::complex<double>(0., 1.)) {
      // [[i, 0], [0, 1]]
      auto func = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[0] == 0.0) {
      // [[0, 0], [0, 1]]
      auto func = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        data_[inds[0]] = 0.0;
      };
      apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    // general [[z, 0], [0, 1]]
    auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const auto k = inds[0];
      data_[k] *= _mat[0];
    };
    apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
    return;
  } else {
    // Lambda function for diagonal matrix multiplication
    auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const auto k0 = inds[0];
      const auto k1 = inds[1];
      data_[k0] *= _mat[0];
      data_[k1] *= _mat[1];
    };
    apply_lambda(0, data_size_, omp_threads_, func, areg_t<1>({{qubit}}), convert(diag));
  }
}

// Declare used template arguments
template class Transformer<std::complex<double>*, double>;
template class Transformer<std::complex<float>*, float>;

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
