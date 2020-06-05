#ifndef QASM_SIMULATOR_CVODE_WRAPPER_HPP
#define QASM_SIMULATOR_CVODE_WRAPPER_HPP

#include <cvode/cvode.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <functional>

#include "sundials_complex_vector.hpp"
#include "ode/ode.h"

namespace AER{

  template <typename T>
  class CvodeWrapper : Ode<T> {
  public:
    using rhsFuncType = std::function<void(double, const T&, T&)>;

    CvodeWrapper(OdeMethod method, rhsFuncType f, const T& y0, double t0, double reltol, double abstol);
    CvodeWrapper(OdeMethod method, rhsFuncType f, unsigned int n, double t0, double reltol, double abstol);

    CvodeWrapper(CvodeWrapper&& rhs);


    ~CvodeWrapper();

    const T& get_solution() const;

    void set_solution(const T& y0);
    void set_intial_value(const T& y0, double t0);

    void set_step_limits(double max_step, double min_step, double first_step_size);
    void set_tolerances(double abstol, double reltol);
    void set_maximum_order(int order);
    void set_max_nsteps(int max_step);

    void integrate_n_steps(double at, int n_steps);

    void integrate(double t, bool one_step=false);

    //void integrate(double t, bool one_step=false, int (*rootFinding)());

    N_Vector y_;
    double t_;

  private:
    void init(OdeMethod method, rhsFuncType f, double t0, double reltol, double abstol);

    struct user_data_func{
      rhsFuncType rhs;
    };


    static int rhs_wrapper(realtype t, N_Vector y, N_Vector ydot, void *user_data){
      auto data_wrapper = static_cast<user_data_func *>(user_data);

      data_wrapper->rhs(t, SundialsComplexContent<T>::get_data(y), SundialsComplexContent<T>::get_data(ydot));
      return 0;
    };

    static int check_retval(void *returnvalue, const char *funcname, int opt)
    {
      int *retval;

      if (opt == 0 && returnvalue == NULL) {
        // Check if SUNDIALS function returned NULL pointer - no memory allocated
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
      } else if (opt == 1) {
        // Check if retval < 0
        retval = (int *) returnvalue;
        if (*retval < 0) {
          fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                  funcname, *retval);
  //            throw std::runtime_error("No more!");
          return 1;
        }
      } else if (opt == 2 && returnvalue == NULL) {
        // Check if function returned NULL pointer - no memory allocated
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1; }

      return 0;
    }

    void* cvode_mem_;
    SUNNonlinearSolver NLS_;

    user_data_func udf_;
  };

  template<typename T>
  CvodeWrapper<T>::CvodeWrapper(OdeMethod method, rhsFuncType f, unsigned int n, double t0, double reltol, double abstol)
  {
    y_ = SundialsComplexContent<T>::new_vector(n);
    init(method, f, t0, reltol, abstol);
  }

  template<typename T>
  CvodeWrapper<T>::CvodeWrapper(OdeMethod method, rhsFuncType f, const T& y0, double t0, double reltol, double abstol) {
    y_ = SundialsComplexContent<T>::new_vector(y0);
    init(method, f, t0, reltol, abstol);
  }

  template<typename T>
  CvodeWrapper<T>::CvodeWrapper(CvodeWrapper&& rhs){
    cvode_mem_ = rhs.cvode_mem_;
    rhs.cvode_mem_ = nullptr;

    NLS_ = rhs.NLS_;
    rhs.NLS_ = nullptr;

    udf_ = rhs.udf_;
    CVodeSetUserData(cvode_mem_, &udf_);

    y_ = rhs.y_;
    rhs.y_ = nullptr;

    t_ = rhs.t_;
  }

  template<typename T>
  void CvodeWrapper<T>::init(OdeMethod method, rhsFuncType f, double t0, double reltol, double abstol){
    if(method == OdeMethod::BDF){
      throw std::runtime_error("BDF not supported yet in Sundials Wrapper");
    }

    t_ = t0;

    cvode_mem_ = CVodeCreate(CV_ADAMS);
    auto retval = CVodeInit(cvode_mem_, rhs_wrapper, t0, y_);
  //    if (check_retval(&retval, "CVodeInit", 1)) return 1;
    check_retval(&retval, "CVodeInit", 1);

    retval = CVodeSStolerances(cvode_mem_, reltol, abstol);
  //    if (check_retval(&retval, "CVodeSVtolerances", 1)) return 1;
    check_retval(&retval, "CVodeSVtolerances", 1);
    NLS_ = SUNNonlinSol_FixedPoint(y_, 0);
    check_retval((void *)NLS_, "SUNNonlinSol_FixedPoint", 0);

    /* attach nonlinear solver object to CVode */
    retval = CVodeSetNonlinearSolver(cvode_mem_, NLS_);
    //    if(check_retval(&retval, "CVodeSetNonlinearSolver", 1)) return(1);
    check_retval(&retval, "CVodeSetNonlinearSolver", 1);

    udf_.rhs = f;
    CVodeSetUserData(cvode_mem_, &udf_);

    CVodeSetMaxConvFails(cvode_mem_, 10000);
  }

  template<typename T>
  CvodeWrapper<T>::~CvodeWrapper() {
    // Free supplied vectors
    N_VDestroy(y_);

    // Free integrator memory
    CVodeFree(&cvode_mem_);

    //Free SUNLinSolver
    SUNNonlinSolFree(NLS_);
  }

  template<typename T>
  const T& CvodeWrapper<T>::get_solution() const {
    return SundialsComplexContent<T>::get_data(y_);
  }

  template<typename T>
  void CvodeWrapper<T>::set_solution(const T& y0)
  {
    SundialsComplexContent<T>::set_data(y_, y0);
  }

  template<typename T>
  void CvodeWrapper<T>::set_intial_value(const T& y0, double t0) {
    t_ = t0;
    set_solution(y0);
  }

  template<typename T>
  void CvodeWrapper<T>::integrate_n_steps(double at, int n_steps) {
    auto tout = t_ + at;
    for(int i = 0; i < n_steps; i++) {
      auto retval = CVode(cvode_mem_, tout, y_, &t_, CV_NORMAL);
      if (check_retval(&retval, "CVode", 1)) break;
      tout += at;
    }
  }

  template<typename T>
  void CvodeWrapper<T>::integrate(double t, bool one_step) {
    auto retval = CVode(cvode_mem_, t, y_, &t_, one_step ? CV_ONE_STEP : CV_NORMAL);
    check_retval(&retval, "CVode", 1);
    if(retval != 0) throw std::runtime_error("Error " + std::to_string(retval));
  }

  template<typename T>
  void CvodeWrapper<T>::set_tolerances(double abstol, double reltol) {
    auto retval = CVodeSStolerances(cvode_mem_, reltol, abstol);
    check_retval(&retval, "CVodeSVtolerances", 1);
  }

  template<typename T>
  void CvodeWrapper<T>::set_step_limits(double max_step, double min_step, double first_step) {
    auto retval = CVodeSetMaxStep(cvode_mem_, max_step);
    check_retval(&retval, "CVodeSetMaxStep", 1);
    retval = CVodeSetMinStep(cvode_mem_, min_step);
    check_retval(&retval, "CVodeSetMinStep", 1);
    retval = CVodeSetInitStep(cvode_mem_, first_step);
    check_retval(&retval, "CVodeSetInitStep", 1);
  }

  template<typename T>
  void CvodeWrapper<T>::set_maximum_order(int order) {
    auto retval = CVodeSetMaxOrd(cvode_mem_, order);
    check_retval(&retval, "CVodeSetMaxStep", 1);
  }

  template<typename T>
  void CvodeWrapper<T>::set_max_nsteps(int max_steps) {
    auto retval = CVodeSetMaxNumSteps(cvode_mem_, max_steps);
    check_retval(&retval, "CVodeSetMaxStep", 1);
  }
}

#endif //QASM_SIMULATOR_CVODE_WRAPPER_HPP
