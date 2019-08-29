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

#ifndef _kernel_hpp
#define _kernel_hpp
namespace Kernel{
template<typename data_t> using data_t = double;
using uint_t = uint64_t;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;


template <typename data_t>
inline void kernel1(std::complex<data_t> *data_, const areg_t<2> &inds,const cvector_t<data_t> &_mat){
    const auto cache = data_[inds[0]];
    data_[inds[0]] = _mat[0] * cache + _mat[2] * data_[inds[1]];
    data_[inds[1]] = _mat[1] * cache + _mat[3] * data_[inds[1]];
}

template <typename data_t>
inline void kernel2(std::complex<data_t> *data_, const areg_t<4> &inds,const cvector_t<data_t> &_mat){
    std::array<std::complex<data_t>, 4> cache;
    for (size_t i = 0; i < 4; i++) {
        const auto ii = inds[i];
        cache[i] = data_[ii];
        data_[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++)
        data_[inds[i]] += _mat[i + 4 * j] * cache[j];
}

template <typename data_t>
inline void kernel3(std::complex<data_t> *data_, const areg_t<8> &inds,const cvector_t<data_t> &_mat){
    std::array<std::complex<data_t>, 8> cache;
    for (size_t i = 0; i < 8; i++) {
        const auto ii = inds[i];
        cache[i] = data_[ii];
        data_[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < 8; i++)
        for (size_t j = 0; j < 8; j++)
        data_[inds[i]] += _mat[i + 8 * j] * cache[j];

}

}//namespace Kernel
#endif