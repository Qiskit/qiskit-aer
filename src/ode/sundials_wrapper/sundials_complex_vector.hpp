#ifndef QASM_SIMULATOR_SUNDIALS_COMPLEX_VECTOR_H
#define QASM_SIMULATOR_SUNDIALS_COMPLEX_VECTOR_H

#include <sundials/sundials_nvector.h>
#include <complex>
#include <vector>

#include "framework/types.hpp"

namespace AER {
  template <typename container_t>
  struct SundialsOps;

  template <typename container_t>
  struct SundialsComplexContent {
    container_t data;

    static N_Vector new_vector(int vec_length) {
      return SundialsOps<SundialsComplexContent>::SundialsComplexContent_New(vec_length);
    };

    template<typename T, typename = std::enable_if_t<!std::is_integral<T>::value>>
    static N_Vector new_vector(const T &container) {
      N_Vector y = SundialsOps<SundialsComplexContent>::SundialsComplexContent_New(container.size());
      auto &raw = get_data(y);
      for (size_t i = 0; i < container.size(); ++i) {
        raw[i] = container[i];
      }
      return y;
    }

    static void prepare_data(N_Vector v, int length){
      auto content = static_cast<SundialsComplexContent *>(v->content);
      content->data.resize(length);
    }

    static SundialsComplexContent* get_content(N_Vector v) {
      return static_cast<SundialsComplexContent *>(v->content);
    }

    static typename container_t::value_type* get_raw_data(N_Vector v) {
      return get_content(v)->data.data();
    }

    static container_t& get_data(N_Vector v) {
      return get_content(v)->data;
    }

    static sunindextype get_size(N_Vector v) {
      return get_data(v).size();
    }

    static void set_data(N_Vector v, const container_t& y0) {
      auto raw_y = SundialsComplexContent::get_raw_data(v);
      for (size_t i = 0; i < y0.size(); ++i) {
        raw_y[i] = y0[i];
      }
    }
  };

  template<typename content_t>
  struct SundialsOps{
    static N_Vector SundialsComplexContent_CloneEmpty(N_Vector w) {
      if (w == nullptr) return nullptr;

      /* Create vector */
      N_Vector v = N_VNewEmpty();
      if (v == nullptr) return nullptr;

      /* Attach operations */
      if (N_VCopyOps(w, v)) {
        N_VDestroy(v);
        return nullptr;
      }

      /* Create content */
      v->content = new content_t;
      return v;
    }

    static N_Vector SundialsComplexContent_Clone(N_Vector w) {
      if (w == nullptr) return nullptr;
      N_Vector v = SundialsComplexContent_CloneEmpty(w);
      if (v == nullptr) return nullptr;

      sunindextype length = content_t::get_size(w);
      /* Prepare data */
      if (length > 0) {
        content_t::prepare_data(v, length);
      }

      return v;
    }

    static void SundialsComplexContent_Destroy(N_Vector v) {
      if (v == nullptr) return;
      /* free content */
      if (v->content != nullptr) {
        delete content_t::get_content(v);
        v->content = nullptr;
      }
      /* free ops and vector */
      if (v->ops != nullptr) {
        free(v->ops);
        v->ops = nullptr;
      }
      free(v);
      return;
    }

    static void SundialsComplexContent_Space(N_Vector v, sunindextype *lrw, sunindextype *liw) {
      *lrw = content_t::get_size(v);
      *liw = 1;

      return;
    }

  static sunindextype SundialsComplexContent_GetLength(N_Vector v) {
      return content_t::get_size(v);
    }

    static N_Vector_ID SundialsComplexContent_GetVectorID(N_Vector v) {
      return SUNDIALS_NVEC_CUSTOM;
    }

    static void SundialsComplexContent_LinearSum(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto y_raw = content_t::get_raw_data(y);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = (a * x_raw[i]) + (b * y_raw[i]);
      }
    }

    static void SundialsComplexContent_Const(realtype c, N_Vector z) {
      auto len = content_t::get_size(z);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = c;
      }
    }

  static void SundialsComplexContent_Prod(N_Vector x, N_Vector y, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto y_raw = content_t::get_raw_data(x);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = x_raw[i] * y_raw[i];
      }
    }

  static void SundialsComplexContent_Div(N_Vector x, N_Vector y, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto y_raw = content_t::get_raw_data(x);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = x_raw[i] / y_raw[i];
      }
    }

  static void SundialsComplexContent_Scale(realtype c, N_Vector x, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = c * x_raw[i];
      }
    }

  static void SundialsComplexContent_Abs(N_Vector x, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = std::abs(x_raw[i]);
      }
    }

  static void SundialsComplexContent_Inv(N_Vector x, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = 1. / x_raw[i];
      }
    }

  static void SundialsComplexContent_AddConst(N_Vector x, realtype b, N_Vector z) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto z_raw = content_t::get_raw_data(z);
      for (int i = 0; i < len; i++) {
        z_raw[i] = x_raw[i] + b;
      }
    }

  static realtype SundialsComplexContent_MaxNorm(N_Vector x) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      double max = 0.0;
      double temp;
      for (int i = 0; i < len; i++) {
        temp = std::abs(x_raw[i]);
        max = temp > max ? temp : max;
      }
      return max;
    }

  static realtype SundialsComplexContent_WrmsNorm(N_Vector x, N_Vector w) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      auto w_raw = content_t::get_raw_data(w);
      double ret{0.0};
      for (int i = 0; i < len; i++) {
        //ret += std::norm(w_raw[i]) * std::norm(x_raw[i]);
        ret += std::norm(w_raw[i] * x_raw[i]);
      }
      return std::sqrt(ret / len);
    }

  static realtype SundialsComplexContent_Min(N_Vector x) {
      auto len = content_t::get_size(x);
      auto x_raw = content_t::get_raw_data(x);
      double min = std::numeric_limits<double>::max();
      double temp;
      for (int i = 0; i < len; i++) {
        temp = std::real(x_raw[i]);
        min = temp < min ? temp : min;
      }
      return min;
    }


  static realtype SundialsComplex_DotProd(N_Vector u, N_Vector v){
      throw std::runtime_error("DotProduct cannot be implemeted. Defined only as it is required (but not used)!");
    }

  static N_Vector SundialsComplexContent_NewEmpty(sunindextype vec_length) {
      N_Vector v = N_VNewEmpty();
      if (v == nullptr) return nullptr;

      // Attach operations
      // constructors, destructors, and utility operations
      v->ops->nvgetvectorid = SundialsOps::SundialsComplexContent_GetVectorID;
      v->ops->nvclone = SundialsOps::SundialsComplexContent_Clone;
      v->ops->nvdestroy = SundialsOps::SundialsComplexContent_Destroy;
      v->ops->nvspace = SundialsOps::SundialsComplexContent_Space;
      v->ops->nvcloneempty = SundialsOps::SundialsComplexContent_CloneEmpty;
      v->ops->nvgetlength = SundialsOps::SundialsComplexContent_GetLength;

      // standard vector operations
      v->ops->nvlinearsum = SundialsOps::SundialsComplexContent_LinearSum;
      v->ops->nvconst = SundialsOps::SundialsComplexContent_Const;
      v->ops->nvprod = SundialsOps::SundialsComplexContent_Prod;
      v->ops->nvdiv = SundialsOps::SundialsComplexContent_Div;
      v->ops->nvscale = SundialsOps::SundialsComplexContent_Scale;
      v->ops->nvabs = SundialsOps::SundialsComplexContent_Abs;
      v->ops->nvinv = SundialsOps::SundialsComplexContent_Inv;
      v->ops->nvmin = SundialsOps::SundialsComplexContent_Min;
      v->ops->nvaddconst = SundialsOps::SundialsComplexContent_AddConst;
      v->ops->nvmaxnorm = SundialsOps::SundialsComplexContent_MaxNorm;
      v->ops->nvwrmsnorm = SundialsOps::SundialsComplexContent_WrmsNorm;

      // Operations required but not needed
      v->ops->nvdotprod = SundialsOps::SundialsComplex_DotProd;

      // Create content
      v->content = new content_t;
      return v;
    }

  static N_Vector SundialsComplexContent_New(sunindextype vec_length) {
      N_Vector v = SundialsComplexContent_NewEmpty(vec_length);
      if (v == nullptr) return nullptr;

      /* Create data */
      if (vec_length > 0) {
        content_t::prepare_data(v, vec_length);
      }
      return v;
    }
  };
}

#endif //QASM_SIMULATOR_SUNDIALS_COMPLEX_VECTOR_H
