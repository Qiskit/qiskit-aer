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

    int mat_vec_index = 0;
    for (int i = 0; i < 2; i++) {
	for (int j = 0; j < 4; j++){
          mat_vec[mat_vec_index]  = loadu2((double *)&mat[i*2+j*4], (double *)&mat[i*2+j*4+1]);
          mat_vec_index++;
        }
    }

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

    int mat_vec_index = 0;
    for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 8; j++){
          mat_vec[mat_vec_index]  = loadu2((double *)&mat[i*2+j*8], (double *)&mat[i*2+j*8+1]);
          mat_vec_index++;
        }
    }

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

inline void kernel4_init(__m256d *mat_vec, __m256d *mat_perm, const cvector_t<double> &mat)
{
    int mat_vec_index = 0;
    for (int i = 0; i < 8; i++) {
	for (int j = 0; j < 16; j++){
          mat_vec[mat_vec_index]  = loadu2((double *)&mat[i*2+j*16], (double *)&mat[i*2+j*16+1]);
          mat_vec_index++;
        }
    }

    for (unsigned i = 0; i < 128; ++i) {
        auto mat_tmp = _mm256_permute_pd(mat_vec[i], 5);
        mat_perm[i] = _mm256_mul_pd(mat_tmp, neg);
    }
}


template<typename data_t=double>
inline void kernel4(std::complex<data_t> *data_, const areg_t<16> &inds, __m256d *mat_vec, __m256d *mat_perm ){
    __m256d vec[4];
    __m256d temp[8];

    vec[0] = loadbc((double *)&data_[inds[0]]);
    vec[1] = loadbc((double *)&data_[inds[1]]);
    vec[2] = loadbc((double *)&data_[inds[2]]);
    vec[3] = loadbc((double *)&data_[inds[3]]);

    temp[0] = _mm256_set_pd(0,0,0,0);
    temp[1] = _mm256_set_pd(0,0,0,0);
    temp[2] = _mm256_set_pd(0,0,0,0);
    temp[3] = _mm256_set_pd(0,0,0,0);
    temp[4] = _mm256_set_pd(0,0,0,0);
    temp[5] = _mm256_set_pd(0,0,0,0);
    temp[6] = _mm256_set_pd(0,0,0,0);
    temp[7] = _mm256_set_pd(0,0,0,0);

    temp[0] = _mm256_add_pd(temp[0], _mm256_add_pd(mul(vec[0], mat_vec[0], mat_perm[0]), _mm256_add_pd(mul(vec[1], mat_vec[1], mat_perm[1]), _mm256_add_pd(mul(vec[2], mat_vec[2], mat_perm[2]), mul(vec[3], mat_vec[3], mat_perm[3])))));
    temp[1] = _mm256_add_pd(temp[1], _mm256_add_pd(mul(vec[0], mat_vec[16], mat_perm[16]), _mm256_add_pd(mul(vec[1], mat_vec[17], mat_perm[17]), _mm256_add_pd(mul(vec[2], mat_vec[18], mat_perm[18]), mul(vec[3], mat_vec[19], mat_perm[19])))));
    temp[2] = _mm256_add_pd(temp[2], _mm256_add_pd(mul(vec[0], mat_vec[32], mat_perm[32]), _mm256_add_pd(mul(vec[1], mat_vec[33], mat_perm[33]), _mm256_add_pd(mul(vec[2], mat_vec[34], mat_perm[34]), mul(vec[3], mat_vec[35], mat_perm[35])))));
    temp[3] = _mm256_add_pd(temp[3], _mm256_add_pd(mul(vec[0], mat_vec[48], mat_perm[48]), _mm256_add_pd(mul(vec[1], mat_vec[49], mat_perm[49]), _mm256_add_pd(mul(vec[2], mat_vec[50], mat_perm[50]), mul(vec[3], mat_vec[51], mat_perm[51])))));
    temp[4] = _mm256_add_pd(temp[4], _mm256_add_pd(mul(vec[0], mat_vec[64], mat_perm[64]), _mm256_add_pd(mul(vec[1], mat_vec[65], mat_perm[65]), _mm256_add_pd(mul(vec[2], mat_vec[66], mat_perm[66]), mul(vec[3], mat_vec[67], mat_perm[67])))));
    temp[5] = _mm256_add_pd(temp[5], _mm256_add_pd(mul(vec[0], mat_vec[80], mat_perm[80]), _mm256_add_pd(mul(vec[1], mat_vec[81], mat_perm[81]), _mm256_add_pd(mul(vec[2], mat_vec[82], mat_perm[82]), mul(vec[3], mat_vec[83], mat_perm[83])))));
    temp[6] = _mm256_add_pd(temp[6], _mm256_add_pd(mul(vec[0], mat_vec[96], mat_perm[96]), _mm256_add_pd(mul(vec[1], mat_vec[97], mat_perm[97]), _mm256_add_pd(mul(vec[2], mat_vec[98], mat_perm[98]), mul(vec[3], mat_vec[99], mat_perm[99])))));
    temp[7] = _mm256_add_pd(temp[7], _mm256_add_pd(mul(vec[0], mat_vec[112], mat_perm[112]), _mm256_add_pd(mul(vec[1], mat_vec[113], mat_perm[113]), _mm256_add_pd(mul(vec[2], mat_vec[114], mat_perm[114]), mul(vec[3], mat_vec[115], mat_perm[115])))));

    vec[0] = loadbc((double *)&data_[inds[4]]);
    vec[1] = loadbc((double *)&data_[inds[5]]);
    vec[2] = loadbc((double *)&data_[inds[6]]);
    vec[3] = loadbc((double *)&data_[inds[7]]);

    temp[0] = _mm256_add_pd(temp[0], _mm256_add_pd(mul(vec[0], mat_vec[4], mat_perm[4]), _mm256_add_pd(mul(vec[1], mat_vec[5], mat_perm[5]), _mm256_add_pd(mul(vec[2], mat_vec[6], mat_perm[6]), mul(vec[3], mat_vec[7], mat_perm[7])))));
    temp[1] = _mm256_add_pd(temp[1], _mm256_add_pd(mul(vec[0], mat_vec[20], mat_perm[20]), _mm256_add_pd(mul(vec[1], mat_vec[21], mat_perm[21]), _mm256_add_pd(mul(vec[2], mat_vec[22], mat_perm[22]), mul(vec[3], mat_vec[23], mat_perm[23])))));
    temp[2] = _mm256_add_pd(temp[2], _mm256_add_pd(mul(vec[0], mat_vec[36], mat_perm[36]), _mm256_add_pd(mul(vec[1], mat_vec[37], mat_perm[37]), _mm256_add_pd(mul(vec[2], mat_vec[38], mat_perm[38]), mul(vec[3], mat_vec[39], mat_perm[39])))));
    temp[3] = _mm256_add_pd(temp[3], _mm256_add_pd(mul(vec[0], mat_vec[52], mat_perm[52]), _mm256_add_pd(mul(vec[1], mat_vec[53], mat_perm[53]), _mm256_add_pd(mul(vec[2], mat_vec[54], mat_perm[54]), mul(vec[3], mat_vec[55], mat_perm[55])))));
    temp[4] = _mm256_add_pd(temp[4], _mm256_add_pd(mul(vec[0], mat_vec[68], mat_perm[68]), _mm256_add_pd(mul(vec[1], mat_vec[69], mat_perm[69]), _mm256_add_pd(mul(vec[2], mat_vec[70], mat_perm[70]), mul(vec[3], mat_vec[71], mat_perm[71])))));
    temp[5] = _mm256_add_pd(temp[5], _mm256_add_pd(mul(vec[0], mat_vec[84], mat_perm[84]), _mm256_add_pd(mul(vec[1], mat_vec[85], mat_perm[85]), _mm256_add_pd(mul(vec[2], mat_vec[86], mat_perm[86]), mul(vec[3], mat_vec[87], mat_perm[87])))));
    temp[6] = _mm256_add_pd(temp[6], _mm256_add_pd(mul(vec[0], mat_vec[100], mat_perm[100]), _mm256_add_pd(mul(vec[1], mat_vec[101], mat_perm[101]), _mm256_add_pd(mul(vec[2], mat_vec[102], mat_perm[102]), mul(vec[3], mat_vec[103], mat_perm[103])))));
    temp[7] = _mm256_add_pd(temp[7], _mm256_add_pd(mul(vec[0], mat_vec[116], mat_perm[116]), _mm256_add_pd(mul(vec[1], mat_vec[117], mat_perm[117]), _mm256_add_pd(mul(vec[2], mat_vec[118], mat_perm[118]), mul(vec[3], mat_vec[119], mat_perm[119])))));

    vec[0] = loadbc((double *)&data_[inds[8]]);
    vec[1] = loadbc((double *)&data_[inds[9]]);
    vec[2] = loadbc((double *)&data_[inds[10]]);
    vec[3] = loadbc((double *)&data_[inds[11]]);

    temp[0] = _mm256_add_pd(temp[0], _mm256_add_pd(mul(vec[0], mat_vec[8], mat_perm[8]), _mm256_add_pd(mul(vec[1], mat_vec[9], mat_perm[9]), _mm256_add_pd(mul(vec[2], mat_vec[10], mat_perm[10]), mul(vec[3], mat_vec[11], mat_perm[11])))));
    temp[1] = _mm256_add_pd(temp[1], _mm256_add_pd(mul(vec[0], mat_vec[24], mat_perm[24]), _mm256_add_pd(mul(vec[1], mat_vec[25], mat_perm[25]), _mm256_add_pd(mul(vec[2], mat_vec[26], mat_perm[26]), mul(vec[3], mat_vec[27], mat_perm[27])))));
    temp[2] = _mm256_add_pd(temp[2], _mm256_add_pd(mul(vec[0], mat_vec[40], mat_perm[40]), _mm256_add_pd(mul(vec[1], mat_vec[41], mat_perm[41]), _mm256_add_pd(mul(vec[2], mat_vec[42], mat_perm[42]), mul(vec[3], mat_vec[43], mat_perm[43])))));
    temp[3] = _mm256_add_pd(temp[3], _mm256_add_pd(mul(vec[0], mat_vec[56], mat_perm[56]), _mm256_add_pd(mul(vec[1], mat_vec[57], mat_perm[57]), _mm256_add_pd(mul(vec[2], mat_vec[58], mat_perm[58]), mul(vec[3], mat_vec[59], mat_perm[59])))));
    temp[4] = _mm256_add_pd(temp[4], _mm256_add_pd(mul(vec[0], mat_vec[72], mat_perm[72]), _mm256_add_pd(mul(vec[1], mat_vec[73], mat_perm[73]), _mm256_add_pd(mul(vec[2], mat_vec[74], mat_perm[74]), mul(vec[3], mat_vec[75], mat_perm[75])))));
    temp[5] = _mm256_add_pd(temp[5], _mm256_add_pd(mul(vec[0], mat_vec[88], mat_perm[88]), _mm256_add_pd(mul(vec[1], mat_vec[89], mat_perm[89]), _mm256_add_pd(mul(vec[2], mat_vec[90], mat_perm[90]), mul(vec[3], mat_vec[91], mat_perm[91])))));
    temp[6] = _mm256_add_pd(temp[6], _mm256_add_pd(mul(vec[0], mat_vec[104], mat_perm[104]), _mm256_add_pd(mul(vec[1], mat_vec[105], mat_perm[105]), _mm256_add_pd(mul(vec[2], mat_vec[106], mat_perm[106]), mul(vec[3], mat_vec[107], mat_perm[107])))));
    temp[7] = _mm256_add_pd(temp[7], _mm256_add_pd(mul(vec[0], mat_vec[120], mat_perm[120]), _mm256_add_pd(mul(vec[1], mat_vec[121], mat_perm[121]), _mm256_add_pd(mul(vec[2], mat_vec[122], mat_perm[122]), mul(vec[3], mat_vec[123], mat_perm[123])))));

    vec[0] = loadbc((double *)&data_[inds[12]]);
    vec[1] = loadbc((double *)&data_[inds[13]]);
    vec[2] = loadbc((double *)&data_[inds[14]]);
    vec[3] = loadbc((double *)&data_[inds[15]]);

    _mm256_storeu2_m128d((double *)&data_[inds[1]], (double *)&data_[inds[0]], _mm256_add_pd(temp[0], _mm256_add_pd(mul(vec[0], mat_vec[12], mat_perm[12]), _mm256_add_pd(mul(vec[1], mat_vec[13], mat_perm[13]), _mm256_add_pd(mul(vec[2], mat_vec[14], mat_perm[14]), mul(vec[3], mat_vec[15], mat_perm[15]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[3]], (double *)&data_[inds[2]], _mm256_add_pd(temp[1], _mm256_add_pd(mul(vec[0], mat_vec[28], mat_perm[28]), _mm256_add_pd(mul(vec[1], mat_vec[29], mat_perm[29]), _mm256_add_pd(mul(vec[2], mat_vec[30], mat_perm[30]), mul(vec[3], mat_vec[31], mat_perm[31]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[5]], (double *)&data_[inds[4]], _mm256_add_pd(temp[2], _mm256_add_pd(mul(vec[0], mat_vec[44], mat_perm[44]), _mm256_add_pd(mul(vec[1], mat_vec[45], mat_perm[45]), _mm256_add_pd(mul(vec[2], mat_vec[46], mat_perm[46]), mul(vec[3], mat_vec[47], mat_perm[47]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[7]], (double *)&data_[inds[6]], _mm256_add_pd(temp[3], _mm256_add_pd(mul(vec[0], mat_vec[60], mat_perm[60]), _mm256_add_pd(mul(vec[1], mat_vec[61], mat_perm[61]), _mm256_add_pd(mul(vec[2], mat_vec[62], mat_perm[62]), mul(vec[3], mat_vec[63], mat_perm[63]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[9]], (double *)&data_[inds[8]], _mm256_add_pd(temp[4], _mm256_add_pd(mul(vec[0], mat_vec[76], mat_perm[76]), _mm256_add_pd(mul(vec[1], mat_vec[77], mat_perm[77]), _mm256_add_pd(mul(vec[2], mat_vec[78], mat_perm[78]), mul(vec[3], mat_vec[79], mat_perm[79]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[11]], (double *)&data_[inds[10]], _mm256_add_pd(temp[5], _mm256_add_pd(mul(vec[0], mat_vec[92], mat_perm[92]), _mm256_add_pd(mul(vec[1], mat_vec[93], mat_perm[93]), _mm256_add_pd(mul(vec[2], mat_vec[94], mat_perm[94]), mul(vec[3], mat_vec[95], mat_perm[95]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[13]], (double *)&data_[inds[12]], _mm256_add_pd(temp[6], _mm256_add_pd(mul(vec[0], mat_vec[108], mat_perm[108]), _mm256_add_pd(mul(vec[1], mat_vec[109], mat_perm[109]), _mm256_add_pd(mul(vec[2], mat_vec[110], mat_perm[110]), mul(vec[3], mat_vec[111], mat_perm[111]))))));
    _mm256_storeu2_m128d((double *)&data_[inds[15]], (double *)&data_[inds[14]], _mm256_add_pd(temp[7], _mm256_add_pd(mul(vec[0], mat_vec[124], mat_perm[124]), _mm256_add_pd(mul(vec[1], mat_vec[125], mat_perm[125]), _mm256_add_pd(mul(vec[2], mat_vec[126], mat_perm[126]), mul(vec[3], mat_vec[127], mat_perm[127]))))));
}

} // namespace Kernel
#endif
