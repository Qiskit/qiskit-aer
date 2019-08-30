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

#ifndef _kernel_simd_hpp
#define _kernel_simd_hpp

#include <x86intrin.h>
#include <complex>

namespace KernelSimd{
//template<typename data_t> using data_t = double;
using uint_t = uint64_t;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;

#ifndef _mm256_set_m128d
#define _mm256_set_m128d(hi,lo) _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), (hi), 0x1)
#endif
#ifndef _mm256_storeu2_m128d
#define _mm256_storeu2_m128d(hiaddr,loaddr,a) \
        do { __m256d _a = (a); _mm_storeu_pd((loaddr), \
             _mm256_castpd256_pd128(_a)); \
             _mm_storeu_pd((hiaddr), \
             _mm256_extractf128_pd(_a, 0x1));\
            } while (0)
#endif
#ifndef _mm256_loadu2_m128d
#define _mm256_loadu2_m128d(hiaddr,loaddr) _mm256_set_m128d(_mm_loadu_pd(hiaddr), _mm_loadu_pd(loaddr))
#endif

const __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

inline __m256d mul(__m256d const& c1, __m256d const& c2, __m256d const& c2_perm){
    auto acbd = _mm256_mul_pd(c1, c2);
    auto acbd_perm = _mm256_mul_pd(c1, c2_perm);
    return _mm256_hsub_pd(acbd, acbd_perm);
}

inline __m256d loadu2(double const*p1, double const*p2){
    return _mm256_loadu2_m128d((double const*)p2, (double const*)p1);
}

inline __m256d loadbc(double *p){
    auto tmp = _mm_load_pd((const double*)p);
    return _mm256_broadcast_pd(&tmp);
}

inline void kernel1_init(__m256d *mat_vec, __m256d *mat_perm, const cvector_t<double> &mat){ 
    mat_vec[0] = loadu2((double *)&mat[0], (double *)&mat[1]); 
    mat_vec[1] = loadu2((double *)&mat[2], (double *)&mat[3]); 

    for (unsigned i = 0; i < 2; ++i){
      auto badc = _mm256_permute_pd(mat_vec[i], 5);
      mat_perm[i] = _mm256_mul_pd(badc, neg);
    }
}

template<typename data_t=double>
inline void kernel1(std::complex<data_t> *data_, const areg_t<2> &inds, __m256d *mat_vec, __m256d *mat_perm ){
    __m256d vec[2];
    vec[0] = loadbc((double *)&data_[inds[0]]);
    vec[1] = loadbc((double *)&data_[inds[1]]);


    _mm256_storeu2_m128d((double *)&data_[inds[1]], (double *)&data_[inds[0]], 
        _mm256_add_pd(mul(vec[0], mat_vec[0], mat_perm[0]), mul(vec[1], mat_vec[1], mat_perm[1])));
}

inline void kernel2_init(__m256d *mat_vec, __m256d *mat_perm, const cvector_t<double> &mat){ 

    mat_vec[0]  = loadu2((double *)&mat[0], (double *)&mat[1]);
    mat_vec[1]  = loadu2((double *)&mat[4], (double *)&mat[5]);
    mat_vec[2]  = loadu2((double *)&mat[8], (double *)&mat[9]);
    mat_vec[3]  = loadu2((double *)&mat[12], (double *)&mat[13]);
    mat_vec[4]  = loadu2((double *)&mat[2], (double *)&mat[3]);
    mat_vec[5]  = loadu2((double *)&mat[6], (double *)&mat[7]);
    mat_vec[6]  = loadu2((double *)&mat[10], (double *)&mat[11]);
    mat_vec[7]  = loadu2((double *)&mat[14], (double *)&mat[15]);

    for (unsigned i = 0; i < 8; ++i){
      auto badc = _mm256_permute_pd(mat_vec[i], 5);
      mat_perm[i] = _mm256_mul_pd(badc, neg);
    }
}

template<typename data_t=double>
inline void kernel2(std::complex<data_t> *data_, const areg_t<4> &inds, __m256d *mat_vec, __m256d *mat_perm ){
    __m256d vec[4];

    vec[0] = loadbc((double *)&data_[inds[0]]);
    vec[1] = loadbc((double *)&data_[inds[1]]);
    vec[2] = loadbc((double *)&data_[inds[2]]);
    vec[3] = loadbc((double *)&data_[inds[3]]);

    _mm256_storeu2_m128d((double*)&data_[inds[1]], (double*)&data_[inds[0]], _mm256_add_pd(mul(vec[0], mat_vec[0], mat_perm[0]), _mm256_add_pd(mul(vec[1], mat_vec[1], mat_perm[1]), _mm256_add_pd(mul(vec[2], mat_vec[2], mat_perm[2]), mul(vec[3], mat_vec[3], mat_perm[3])))));
    _mm256_storeu2_m128d((double*)&data_[inds[3]], (double*)&data_[inds[2]], _mm256_add_pd(mul(vec[0], mat_vec[4], mat_perm[4]), _mm256_add_pd(mul(vec[1], mat_vec[5], mat_perm[5]), _mm256_add_pd(mul(vec[2], mat_vec[6], mat_perm[6]), mul(vec[3], mat_vec[7], mat_perm[7])))));

}

inline void kernel3_init(__m256d *mat_vec, __m256d *mat_perm, const cvector_t<double> &mat){

    mat_vec[0] = loadu2((double *)&mat[0], (double *)&mat[1]);
    mat_vec[1] = loadu2((double *)&mat[8], (double *)&mat[9]);
    mat_vec[2] = loadu2((double *)&mat[16], (double *)&mat[17]);
    mat_vec[3] = loadu2((double *)&mat[24], (double *)&mat[25]);
    mat_vec[4] = loadu2((double *)&mat[32], (double *)&mat[33]);
    mat_vec[5] = loadu2((double *)&mat[40], (double *)&mat[41]);
    mat_vec[6] = loadu2((double *)&mat[48], (double *)&mat[49]);
    mat_vec[7] = loadu2((double *)&mat[56], (double *)&mat[57]);
    mat_vec[8] = loadu2((double *)&mat[2], (double *)&mat[3]);
    mat_vec[9] = loadu2((double *)&mat[10], (double *)&mat[11]);
    mat_vec[10] = loadu2((double *)&mat[18], (double *)&mat[19]);
    mat_vec[11] = loadu2((double *)&mat[26], (double *)&mat[27]);
    mat_vec[12] = loadu2((double *)&mat[34], (double *)&mat[35]);
    mat_vec[13] = loadu2((double *)&mat[42], (double *)&mat[43]);
    mat_vec[14] = loadu2((double *)&mat[50], (double *)&mat[51]);
    mat_vec[15] = loadu2((double *)&mat[58], (double *)&mat[59]);
    mat_vec[16] = loadu2((double *)&mat[4], (double *)&mat[5]);
    mat_vec[17] = loadu2((double *)&mat[12], (double *)&mat[13]);
    mat_vec[18] = loadu2((double *)&mat[20], (double *)&mat[21]);
    mat_vec[19] = loadu2((double *)&mat[28], (double *)&mat[29]);
    mat_vec[20] = loadu2((double *)&mat[36], (double *)&mat[37]);
    mat_vec[21] = loadu2((double *)&mat[44], (double *)&mat[45]);
    mat_vec[22] = loadu2((double *)&mat[52], (double *)&mat[53]);
    mat_vec[23] = loadu2((double *)&mat[60], (double *)&mat[61]);
    mat_vec[24] = loadu2((double *)&mat[6], (double *)&mat[7]);
    mat_vec[25] = loadu2((double *)&mat[14], (double *)&mat[15]);
    mat_vec[26] = loadu2((double *)&mat[22], (double *)&mat[23]);
    mat_vec[27] = loadu2((double *)&mat[30], (double *)&mat[31]);
    mat_vec[28] = loadu2((double *)&mat[38], (double *)&mat[39]);
    mat_vec[29] = loadu2((double *)&mat[46], (double *)&mat[47]);
    mat_vec[30] = loadu2((double *)&mat[54], (double *)&mat[55]);
    mat_vec[31] = loadu2((double *)&mat[62], (double *)&mat[63]);

    for (unsigned i = 0; i < 32; ++i) {
        auto mat_tmp = _mm256_permute_pd(mat_vec[i], 5);
        mat_perm[i] = _mm256_mul_pd(mat_tmp, neg);
    }
}

template<typename data_t=double>
inline void kernel3(std::complex<data_t> *data_, const areg_t<8> &inds, __m256d *mat_vec, __m256d *mat_perm ){
    __m256d vec[4];
    __m256d temp[4];

    vec[0] = loadbc((double *)&data_[inds[0]]);
    vec[1] = loadbc((double *)&data_[inds[1]]);
    vec[2] = loadbc((double *)&data_[inds[2]]);
    vec[3] = loadbc((double *)&data_[inds[3]]);

    temp[0] = _mm256_add_pd(mul(vec[0], mat_vec[0], mat_perm[0]), _mm256_add_pd(mul(vec[1], mat_vec[1], mat_perm[1]), _mm256_add_pd(mul(vec[2], mat_vec[2], mat_perm[2]), mul(vec[3], mat_vec[3], mat_perm[3]))));
    temp[1] = _mm256_add_pd(mul(vec[0], mat_vec[8], mat_perm[8]), _mm256_add_pd(mul(vec[1], mat_vec[9], mat_perm[9]), _mm256_add_pd(mul(vec[2], mat_vec[10], mat_perm[10]), mul(vec[3], mat_vec[11], mat_perm[11]))));
    temp[2] = _mm256_add_pd(mul(vec[0], mat_vec[16], mat_perm[16]), _mm256_add_pd(mul(vec[1], mat_vec[17], mat_perm[17]), _mm256_add_pd(mul(vec[2], mat_vec[18], mat_perm[18]), mul(vec[3], mat_vec[19], mat_perm[19]))));
    temp[3] = _mm256_add_pd(mul(vec[0], mat_vec[24], mat_perm[24]), _mm256_add_pd(mul(vec[1], mat_vec[25], mat_perm[25]), _mm256_add_pd(mul(vec[2], mat_vec[26], mat_perm[26]), mul(vec[3], mat_vec[27], mat_perm[27]))));

    vec[0] = loadbc((double *)&data_[inds[4]]);
    vec[1] = loadbc((double *)&data_[inds[5]]);
    vec[2] = loadbc((double *)&data_[inds[6]]);
    vec[3] = loadbc((double *)&data_[inds[7]]);

    _mm256_storeu2_m128d((double *)&data_[inds[1]], (double *)&data_[inds[0]], _mm256_add_pd(temp[0], _mm256_add_pd(mul(vec[0], mat_vec[4], mat_perm[4]), _mm256_add_pd(mul(vec[1], mat_vec[5], mat_perm[5]), _mm256_add_pd(mul(vec[2], mat_vec[6], mat_perm[6]), mul(vec[3], mat_vec[7], mat_perm[7]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[3]], (double *)&data_[inds[2]], _mm256_add_pd(temp[1], _mm256_add_pd(mul(vec[0], mat_vec[12], mat_perm[12]), _mm256_add_pd(mul(vec[1], mat_vec[13], mat_perm[13]), _mm256_add_pd(mul(vec[2], mat_vec[14], mat_perm[14]), mul(vec[3], mat_vec[15], mat_perm[15]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[5]], (double *)&data_[inds[4]], _mm256_add_pd(temp[2], _mm256_add_pd(mul(vec[0], mat_vec[20], mat_perm[20]), _mm256_add_pd(mul(vec[1], mat_vec[21], mat_perm[21]), _mm256_add_pd(mul(vec[2], mat_vec[22], mat_perm[22]), mul(vec[3], mat_vec[23], mat_perm[23]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[7]], (double *)&data_[inds[6]], _mm256_add_pd(temp[3], _mm256_add_pd(mul(vec[0], mat_vec[28], mat_perm[28]), _mm256_add_pd(mul(vec[1], mat_vec[29], mat_perm[29]), _mm256_add_pd(mul(vec[2], mat_vec[30], mat_perm[30]), mul(vec[3], mat_vec[31], mat_perm[31]))))));
}

} // namespace Kernel
#endif
