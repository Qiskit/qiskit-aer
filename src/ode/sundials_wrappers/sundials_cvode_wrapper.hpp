#ifndef QASM_SIMULATOR_CVODE_WRAPPER_HPP
#define QASM_SIMULATOR_CVODE_WRAPPER_HPP

#include <cvodes/cvodes.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <functional>

#include "sundials_complex_vector.hpp"
#include "ode/ode.h"

namespace AER{
  int check_retval(void *returnvalue, const char *funcname, int opt);

  template <typename T>
  class CvodeWrapper : public Ode<T> {
    // TODO: I guess we don't want to allow copies. Change shared_ptr to unique_ptr
  public:
    using rhsFuncType = std::function<void(double, const T&, T&)>;
    using perturbFuncType = std::function<void(const std::vector<double>&)>;

    template <typename U>
    CvodeWrapper(OdeMethod method, rhsFuncType f, const U& y0, double t0);
    CvodeWrapper(OdeMethod method, rhsFuncType f, unsigned int n, double t0);

    template <typename U>
    CvodeWrapper(OdeMethod method, rhsFuncType f, const U& y0, perturbFuncType pf, const std::vector<double>& p, double t0);

    CvodeWrapper(const CvodeWrapper&) = delete;
    CvodeWrapper operator=(const CvodeWrapper&) = delete;
    CvodeWrapper(CvodeWrapper&&) = default;

    double get_t(){ return t_; };
    const T& get_solution() const;
    const T& get_sens_solution(uint i);

    void set_t(double t);
    void set_solution(const T& y0);
    void set_intial_value(const T& y0, double t0);

    void set_step_limits(double max_step, double min_step, double first_step_size);
    void set_tolerances(double abstol, double reltol);
    void set_maximum_order(int order);
    void set_max_nsteps(int max_step);

    void integrate_n_steps(double at, int n_steps);

    void integrate(double t, bool one_step=false);

    bool succesful(){
      return !retval_;
    }
    //void integrate(double t, bool one_step=false, int (*rootFinding)());

  private:
    void init(OdeMethod method, rhsFuncType f, double t0);

    void init_sens(perturbFuncType pf, const std::vector<double>& p);

    struct user_data_func{
      rhsFuncType rhs;
      std::vector<double> p;
      perturbFuncType pf;
    };

    static constexpr double ATOL = 1e-12;
    static constexpr double RTOL = 1e-6;

    static int rhs_wrapper(realtype t, N_Vector y, N_Vector ydot, void *user_data){
      auto data_wrapper = static_cast<user_data_func *>(user_data);
      if(data_wrapper->pf){
        data_wrapper->pf(data_wrapper->p);
      }
      data_wrapper->rhs(t, SundialsComplexContent<T>::get_data(y), SundialsComplexContent<T>::get_data(ydot));
      return 0;
    };

    std::shared_ptr<void> cvode_mem_;

    std::shared_ptr<_generic_N_Vector> y_;
    double t_;

    std::shared_ptr<_generic_SUNNonlinearSolver> NLS_;
    std::shared_ptr<_generic_N_Vector*> sens_;
    std::shared_ptr<_generic_SUNNonlinearSolver> NLS_sens_;

    std::shared_ptr<user_data_func> udf_;
    int retval_;
  };

  template<typename T>
  CvodeWrapper<T>::CvodeWrapper(OdeMethod method, rhsFuncType f, unsigned int n, double t0)
  {
    y_ = std::shared_ptr<_generic_N_Vector>(SundialsComplexContent<T>::new_vector(n), N_VDestroy);
    init(method, f, t0);
  }

  template<typename T>
  template <typename U>
  CvodeWrapper<T>::CvodeWrapper(OdeMethod method, rhsFuncType f, const U& y0, double t0) {
    y_ = std::shared_ptr<_generic_N_Vector>(SundialsComplexContent<T>::new_vector(y0), N_VDestroy);
    init(method, f, t0);
  }

  template<typename T>
  template <typename U>
  CvodeWrapper<T>::CvodeWrapper(OdeMethod method, rhsFuncType f, const U& y0, perturbFuncType pf, const std::vector<double>& p, double t0)
  : CvodeWrapper(method, f, y0, t0) {
    init_sens(pf, p);
  }

  template <typename T>
  void CvodeWrapper<T>::init(OdeMethod method, rhsFuncType f, double t0) {
    if (method == OdeMethod::BDF) {
      throw std::runtime_error("BDF not supported yet in Sundials Wrapper");
    }

    t_ = t0;

    cvode_mem_ = std::shared_ptr<void>(CVodeCreate(CV_ADAMS),
                                       [](void *cvode_mem) { CVodeFree(&cvode_mem); });
    auto retval = CVodeInit(cvode_mem_.get(), rhs_wrapper, t0, y_.get());
    //    if (check_retval(&retval, "CVodeInit", 1)) return 1;
    check_retval(&retval, "CVodeInit", 1);

    retval = CVodeSStolerances(cvode_mem_.get(), RTOL, ATOL);
    check_retval(&retval, "CVodeSVtolerances", 1);
    NLS_ = std::shared_ptr<_generic_SUNNonlinearSolver>(SUNNonlinSol_FixedPoint(y_.get(), 0), SUNNonlinSolFree);
    check_retval((void *)NLS_.get(), "SUNNonlinSol_FixedPoint", 0);

    /* attach nonlinear solver object to CVode */
    retval = CVodeSetNonlinearSolver(cvode_mem_.get(), NLS_.get());
    check_retval(&retval, "CVodeSetNonlinearSolver", 1);
    udf_ = std::make_shared<user_data_func>();
    udf_->rhs = f;
    CVodeSetUserData(cvode_mem_.get(), udf_.get());

    // TODO: Hardcoded value??
    CVodeSetMaxConvFails(cvode_mem_.get(), 10000);
  }

  template<typename T>
  void CvodeWrapper<T>::init_sens(perturbFuncType pf, const std::vector<double>& p){
    udf_->p = p;
    udf_->pf = pf;
    auto num_sens = p.size();
    sens_ = std::shared_ptr<_generic_N_Vector*>(N_VCloneVectorArray(num_sens, y_.get()),
        [num_sens](_generic_N_Vector** sens){N_VDestroyVectorArray(sens, num_sens);});

    // TODO: Check if we need this
    for(int is=0; is<num_sens; is++){
      N_VConst(0.0, sens_.get()[is]);
    }

    auto retval = CVodeSensInit1(cvode_mem_.get(), num_sens, CV_SIMULTANEOUS, nullptr, sens_.get());
    check_retval(&retval, "CVodeSensInit1", 1);

    retval = CVodeSensEEtolerances(cvode_mem_.get());
    check_retval(&retval, "CVodeSensEEtolerances", 1);

    // TODO: Error control hardcoded to true
    retval = CVodeSetSensErrCon(cvode_mem_.get(), true);
    check_retval(&retval, "CVodeSetSensErrCon", 1);

    retval = CVodeSetSensDQMethod(cvode_mem_.get(), CV_CENTERED, 0.0);
    check_retval(&retval, "CVodeSetSensDQMethod", 1);

    retval = CVodeSetSensParams(cvode_mem_.get(), udf_->p.data(), nullptr, nullptr);
    check_retval(&retval, "CVodeSetSensParams", 1);

    // Assuming SIMULTANEOUS so far
    NLS_sens_ = std::shared_ptr<_generic_SUNNonlinearSolver>(SUNNonlinSol_FixedPointSens(num_sens + 1, y_.get(), 0), SUNNonlinSolFree);
    retval = CVodeSetNonlinearSolverSensSim(cvode_mem_.get(), NLS_sens_.get());
    check_retval(&retval, "CVodeSetSensParams", 1);

  }

  template<typename T>
  const T& CvodeWrapper<T>::get_solution() const {
    return SundialsComplexContent<T>::get_data(y_.get());
  }

  template<typename T>
  const T& CvodeWrapper<T>::get_sens_solution(uint i) {
    if(i >= udf_->p.size()){
      throw std::runtime_error("Trying to get sensitivity componente outside limits");
    }
    auto retval = CVodeGetSens(cvode_mem_.get(), &t_, sens_.get());
    check_retval(&retval, "CVodeSetSensParams", 1);
    return SundialsComplexContent<T>::get_data(sens_.get()[i]);
  }

  template<typename T>
  void CvodeWrapper<T>::set_t(double t)
  {
      t_ = t;
      CVodeReInit(cvode_mem_.get(), t_, y_.get());
  }

  template<typename T>
  void CvodeWrapper<T>::set_solution(const T& y0)
  {
    SundialsComplexContent<T>::set_data(y_.get(), y0);
    CVodeReInit(cvode_mem_.get(), t_, y_.get());
  }

  template<typename T>
  void CvodeWrapper<T>::set_intial_value(const T& y0, double t0) {
    t_ = t0;
    SundialsComplexContent<T>::set_data(y_.get(), y0);
    CVodeReInit(cvode_mem_.get(), t_, y_.get());
  }

  template<typename T>
  void CvodeWrapper<T>::integrate_n_steps(double at, int n_steps) {
    auto tout = t_ + at;
    for(int i = 0; i < n_steps; i++) {
      retval_ = CVode(cvode_mem_.get(), tout, y_.get(), &t_, CV_NORMAL);
      if (check_retval(&retval_, "CVode", 1)) break;
      tout += at;
    }
  }

  template<typename T>
  void CvodeWrapper<T>::integrate(double t, bool one_step) {
    retval_ = CVode(cvode_mem_.get(), t, y_.get(), &t_, one_step ? CV_ONE_STEP : CV_NORMAL);
    check_retval(&retval_, "CVode", 1);
    if(retval_ != 0) throw std::runtime_error("Error " + std::to_string(retval_));
  }

  template<typename T>
  void CvodeWrapper<T>::set_tolerances(double abstol, double reltol) {
    auto retval = CVodeSStolerances(cvode_mem_.get(), reltol, abstol);
    check_retval(&retval, "CVodeSVtolerances", 1);
  }

  template<typename T>
  void CvodeWrapper<T>::set_step_limits(double max_step, double min_step, double first_step) {
    auto retval = CVodeSetMaxStep(cvode_mem_.get(), max_step);
    check_retval(&retval, "CVodeSetMaxStep", 1);
    retval = CVodeSetMinStep(cvode_mem_.get(), min_step);
    check_retval(&retval, "CVodeSetMinStep", 1);
    retval = CVodeSetInitStep(cvode_mem_.get(), first_step);
    check_retval(&retval, "CVodeSetInitStep", 1);
  }

  template<typename T>
  void CvodeWrapper<T>::set_maximum_order(int order) {
    auto retval = CVodeSetMaxOrd(cvode_mem_.get(), order);
    check_retval(&retval, "CVodeSetMaxStep", 1);
  }

  template<typename T>
  void CvodeWrapper<T>::set_max_nsteps(int max_steps) {
    auto retval = CVodeSetMaxNumSteps(cvode_mem_.get(), max_steps);
    check_retval(&retval, "CVodeSetMaxStep", 1);
  }

  int check_retval(void *returnvalue, const char *funcname, int opt) {
    int *retval;

    if (opt == 0 && returnvalue == NULL) {
      // Check if SUNDIALS function returned NULL pointer - no memory allocated
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
      return 1;
    } else if (opt == 1) {
      // Check if retval < 0
      retval = (int *)returnvalue;
      if (*retval < 0) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n", funcname, *retval);
        //            throw std::runtime_error("No more!");
        return 1;
      }
    } else if (opt == 2 && returnvalue == NULL) {
      // Check if function returned NULL pointer - no memory allocated
      fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
      return 1;
    }
    return 0;
  }
}

#endif //QASM_SIMULATOR_CVODE_WRAPPER_HPP
