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

inline void kernel4_init(__m256d *mat_vec, __m256d *mat_perm, const cvector_t<double> &mat)
{
    mat_vec[0] = loadu2((double *)&mat[0], (double *)&mat[1]);
    mat_vec[1] = loadu2((double *)&mat[16], (double *)&mat[17]);
    mat_vec[2] = loadu2((double *)&mat[32], (double *)&mat[33]);
    mat_vec[3] = loadu2((double *)&mat[48], (double *)&mat[49]);
    mat_vec[4] = loadu2((double *)&mat[64], (double *)&mat[65]);
    mat_vec[5] = loadu2((double *)&mat[80], (double *)&mat[81]);
    mat_vec[6] = loadu2((double *)&mat[96], (double *)&mat[97]);
    mat_vec[7] = loadu2((double *)&mat[112], (double *)&mat[113]);
    mat_vec[8] = loadu2((double *)&mat[128], (double *)&mat[129]);
    mat_vec[9] = loadu2((double *)&mat[144], (double *)&mat[145]);
    mat_vec[10] = loadu2((double *)&mat[160], (double *)&mat[161]);
    mat_vec[11] = loadu2((double *)&mat[176], (double *)&mat[177]);
    mat_vec[12] = loadu2((double *)&mat[192], (double *)&mat[193]);
    mat_vec[13] = loadu2((double *)&mat[208], (double *)&mat[209]);
    mat_vec[14] = loadu2((double *)&mat[224], (double *)&mat[225]);
    mat_vec[15] = loadu2((double *)&mat[240], (double *)&mat[241]);
    mat_vec[16] = loadu2((double *)&mat[2], (double *)&mat[3]);
    mat_vec[17] = loadu2((double *)&mat[18], (double *)&mat[19]);
    mat_vec[18] = loadu2((double *)&mat[34], (double *)&mat[35]);
    mat_vec[19] = loadu2((double *)&mat[50], (double *)&mat[51]);
    mat_vec[20] = loadu2((double *)&mat[66], (double *)&mat[67]);
    mat_vec[21] = loadu2((double *)&mat[82], (double *)&mat[83]);
    mat_vec[22] = loadu2((double *)&mat[98], (double *)&mat[99]);
    mat_vec[23] = loadu2((double *)&mat[114], (double *)&mat[115]);
    mat_vec[24] = loadu2((double *)&mat[130], (double *)&mat[131]);
    mat_vec[25] = loadu2((double *)&mat[146], (double *)&mat[147]);
    mat_vec[26] = loadu2((double *)&mat[162], (double *)&mat[163]);
    mat_vec[27] = loadu2((double *)&mat[178], (double *)&mat[179]);
    mat_vec[28] = loadu2((double *)&mat[194], (double *)&mat[195]);
    mat_vec[29] = loadu2((double *)&mat[210], (double *)&mat[211]);
    mat_vec[30] = loadu2((double *)&mat[226], (double *)&mat[227]);
    mat_vec[31] = loadu2((double *)&mat[242], (double *)&mat[243]);
    mat_vec[32] = loadu2((double *)&mat[4], (double *)&mat[5]);
    mat_vec[33] = loadu2((double *)&mat[20], (double *)&mat[21]);
    mat_vec[34] = loadu2((double *)&mat[36], (double *)&mat[37]);
    mat_vec[35] = loadu2((double *)&mat[52], (double *)&mat[53]);
    mat_vec[36] = loadu2((double *)&mat[68], (double *)&mat[69]);
    mat_vec[37] = loadu2((double *)&mat[84], (double *)&mat[85]);
    mat_vec[38] = loadu2((double *)&mat[100], (double *)&mat[101]);
    mat_vec[39] = loadu2((double *)&mat[116], (double *)&mat[117]);
    mat_vec[40] = loadu2((double *)&mat[132], (double *)&mat[133]);
    mat_vec[41] = loadu2((double *)&mat[148], (double *)&mat[149]);
    mat_vec[42] = loadu2((double *)&mat[164], (double *)&mat[165]);
    mat_vec[43] = loadu2((double *)&mat[180], (double *)&mat[181]);
    mat_vec[44] = loadu2((double *)&mat[196], (double *)&mat[197]);
    mat_vec[45] = loadu2((double *)&mat[212], (double *)&mat[213]);
    mat_vec[46] = loadu2((double *)&mat[228], (double *)&mat[229]);
    mat_vec[47] = loadu2((double *)&mat[244], (double *)&mat[245]);
    mat_vec[48] = loadu2((double *)&mat[6], (double *)&mat[7]);
    mat_vec[49] = loadu2((double *)&mat[22], (double *)&mat[23]);
    mat_vec[50] = loadu2((double *)&mat[38], (double *)&mat[39]);
    mat_vec[51] = loadu2((double *)&mat[54], (double *)&mat[55]);
    mat_vec[52] = loadu2((double *)&mat[70], (double *)&mat[71]);
    mat_vec[53] = loadu2((double *)&mat[86], (double *)&mat[87]);
    mat_vec[54] = loadu2((double *)&mat[102], (double *)&mat[103]);
    mat_vec[55] = loadu2((double *)&mat[118], (double *)&mat[119]);
    mat_vec[56] = loadu2((double *)&mat[134], (double *)&mat[135]);
    mat_vec[57] = loadu2((double *)&mat[150], (double *)&mat[151]);
    mat_vec[58] = loadu2((double *)&mat[166], (double *)&mat[167]);
    mat_vec[59] = loadu2((double *)&mat[182], (double *)&mat[183]);
    mat_vec[60] = loadu2((double *)&mat[198], (double *)&mat[199]);
    mat_vec[61] = loadu2((double *)&mat[214], (double *)&mat[215]);
    mat_vec[62] = loadu2((double *)&mat[230], (double *)&mat[231]);
    mat_vec[63] = loadu2((double *)&mat[246], (double *)&mat[247]);
    mat_vec[64] = loadu2((double *)&mat[8], (double *)&mat[9]);
    mat_vec[65] = loadu2((double *)&mat[24], (double *)&mat[25]);
    mat_vec[66] = loadu2((double *)&mat[40], (double *)&mat[41]);
    mat_vec[67] = loadu2((double *)&mat[56], (double *)&mat[57]);
    mat_vec[68] = loadu2((double *)&mat[72], (double *)&mat[73]);
    mat_vec[69] = loadu2((double *)&mat[88], (double *)&mat[89]);
    mat_vec[70] = loadu2((double *)&mat[104], (double *)&mat[105]);
    mat_vec[71] = loadu2((double *)&mat[120], (double *)&mat[121]);
    mat_vec[72] = loadu2((double *)&mat[136], (double *)&mat[137]);
    mat_vec[73] = loadu2((double *)&mat[152], (double *)&mat[153]);
    mat_vec[74] = loadu2((double *)&mat[168], (double *)&mat[169]);
    mat_vec[75] = loadu2((double *)&mat[184], (double *)&mat[185]);
    mat_vec[76] = loadu2((double *)&mat[200], (double *)&mat[201]);
    mat_vec[77] = loadu2((double *)&mat[216], (double *)&mat[217]);
    mat_vec[78] = loadu2((double *)&mat[232], (double *)&mat[233]);
    mat_vec[79] = loadu2((double *)&mat[248], (double *)&mat[249]);
    mat_vec[80] = loadu2((double *)&mat[10], (double *)&mat[11]);
    mat_vec[81] = loadu2((double *)&mat[26], (double *)&mat[27]);
    mat_vec[82] = loadu2((double *)&mat[42], (double *)&mat[43]);
    mat_vec[83] = loadu2((double *)&mat[58], (double *)&mat[59]);
    mat_vec[84] = loadu2((double *)&mat[74], (double *)&mat[75]);
    mat_vec[85] = loadu2((double *)&mat[90], (double *)&mat[91]);
    mat_vec[86] = loadu2((double *)&mat[106], (double *)&mat[107]);
    mat_vec[87] = loadu2((double *)&mat[122], (double *)&mat[123]);
    mat_vec[88] = loadu2((double *)&mat[138], (double *)&mat[139]);
    mat_vec[89] = loadu2((double *)&mat[154], (double *)&mat[155]);
    mat_vec[90] = loadu2((double *)&mat[170], (double *)&mat[171]);
    mat_vec[91] = loadu2((double *)&mat[186], (double *)&mat[187]);
    mat_vec[92] = loadu2((double *)&mat[202], (double *)&mat[203]);
    mat_vec[93] = loadu2((double *)&mat[218], (double *)&mat[219]);
    mat_vec[94] = loadu2((double *)&mat[234], (double *)&mat[235]);
    mat_vec[95] = loadu2((double *)&mat[250], (double *)&mat[251]);
    mat_vec[96] = loadu2((double *)&mat[12], (double *)&mat[13]);
    mat_vec[97] = loadu2((double *)&mat[28], (double *)&mat[29]);
    mat_vec[98] = loadu2((double *)&mat[44], (double *)&mat[45]);
    mat_vec[99] = loadu2((double *)&mat[60], (double *)&mat[61]);
    mat_vec[100] = loadu2((double *)&mat[76], (double *)&mat[77]);
    mat_vec[101] = loadu2((double *)&mat[92], (double *)&mat[93]);
    mat_vec[102] = loadu2((double *)&mat[108], (double *)&mat[109]);
    mat_vec[103] = loadu2((double *)&mat[124], (double *)&mat[125]);
    mat_vec[104] = loadu2((double *)&mat[140], (double *)&mat[141]);
    mat_vec[105] = loadu2((double *)&mat[156], (double *)&mat[157]);
    mat_vec[106] = loadu2((double *)&mat[172], (double *)&mat[173]);
    mat_vec[107] = loadu2((double *)&mat[188], (double *)&mat[189]);
    mat_vec[108] = loadu2((double *)&mat[204], (double *)&mat[205]);
    mat_vec[109] = loadu2((double *)&mat[220], (double *)&mat[221]);
    mat_vec[110] = loadu2((double *)&mat[236], (double *)&mat[237]);
    mat_vec[111] = loadu2((double *)&mat[252], (double *)&mat[253]);
    mat_vec[112] = loadu2((double *)&mat[14], (double *)&mat[15]);
    mat_vec[113] = loadu2((double *)&mat[30], (double *)&mat[31]);
    mat_vec[114] = loadu2((double *)&mat[46], (double *)&mat[47]);
    mat_vec[115] = loadu2((double *)&mat[62], (double *)&mat[63]);
    mat_vec[116] = loadu2((double *)&mat[78], (double *)&mat[79]);
    mat_vec[117] = loadu2((double *)&mat[94], (double *)&mat[95]);
    mat_vec[118] = loadu2((double *)&mat[110], (double *)&mat[111]);
    mat_vec[119] = loadu2((double *)&mat[126], (double *)&mat[127]);
    mat_vec[120] = loadu2((double *)&mat[142], (double *)&mat[143]);
    mat_vec[121] = loadu2((double *)&mat[158], (double *)&mat[159]);
    mat_vec[122] = loadu2((double *)&mat[174], (double *)&mat[175]);
    mat_vec[123] = loadu2((double *)&mat[190], (double *)&mat[191]);
    mat_vec[124] = loadu2((double *)&mat[206], (double *)&mat[207]);
    mat_vec[125] = loadu2((double *)&mat[222], (double *)&mat[223]);
    mat_vec[126] = loadu2((double *)&mat[238], (double *)&mat[239]);
    mat_vec[127] = loadu2((double *)&mat[254], (double *)&mat[255]);

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
