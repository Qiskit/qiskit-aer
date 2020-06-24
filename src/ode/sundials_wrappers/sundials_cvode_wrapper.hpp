#ifndef QASM_SIMULATOR_CVODE_WRAPPER_HPP
#define QASM_SIMULATOR_CVODE_WRAPPER_HPP

#include <cvodes/cvodes.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <functional>

#include "sundials_complex_vector.hpp"
#include "ode/ode.hpp"

namespace AER {
  namespace ODE {
    void error_handler(int error_code, const char *module, const char *function, char *msg,
                       void *eh_data) {
      if (error_code <= 0) {
        throw std::runtime_error(std::string(msg) + " Sundials error code: " + std::to_string(error_code));
      }
    }

    template<typename T>
    class CvodeWrapper : public Ode<T> {
      // TODO: I guess we don't want to allow copies. Change shared_ptr to unique_ptr
    public:
      static const std::string ID;

      CvodeWrapper(rhsFuncType<T> f, const T &y0, double t0);

      CvodeWrapper(rhsFuncType<T> f, T&& y0, double t0);

      void setup_sens(perturbFuncType pf, const std::vector<double> &p);

      CvodeWrapper(const CvodeWrapper &) = delete;

      CvodeWrapper operator=(const CvodeWrapper &) = delete;

      CvodeWrapper(CvodeWrapper &&) = default;

      double get_t() { return t_; };

      const T &get_solution() const;

      const T &get_sens_solution(uint i);

      void set_t(double t);

      void set_solution(const T& y0);
      void set_solution(T&& y0);

      void set_intial_value(const T& y0, double t0);
      void set_intial_value(T&& y0, double t0);

      void set_step_limits(double max_step, double min_step, double first_step_size);

      void set_tolerances(double abstol, double reltol);

      void set_maximum_order(int order);

      void set_max_nsteps(int max_step);

      void integrate_n_steps(double at, int n_steps);

      void integrate(double t, bool one_step = false);

      bool succesful() {
        return !retval_;
      }

    private:
      void init(rhsFuncType<T> f, double t0);

      void init_sens(perturbFuncType pf, const std::vector<double> &p);

      void reinit();
      void reinit_if_run();

      struct user_data_func {
        rhsFuncType<T> rhs;
        std::vector<double> p;
        perturbFuncType pf;
      };

      static constexpr double ATOL = 1e-12;
      static constexpr double RTOL = 1e-12;

      static int rhs_wrapper(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
        auto data_wrapper = static_cast<user_data_func *>(user_data);
        if (data_wrapper->pf) {
          data_wrapper->pf(data_wrapper->p);
        }
        data_wrapper->rhs(t, SundialsComplexContent<T>::get_data(y), SundialsComplexContent<T>::get_data(ydot));
        return 0;
      };

      using CVodeMemDeleter = std::function<void(void *)>;
      using NLSDeleter = std::function<int(SUNNonlinearSolver)>;
      using NVectorDeleter = std::function<void(N_Vector)>;
      using NVectorArrayDeleter =std::function<void(N_Vector*)>;

      std::unique_ptr<void, CVodeMemDeleter> cvode_mem_ =
          std::unique_ptr<void, CVodeMemDeleter>(nullptr,
                                                 [](void *cvode_mem) { CVodeFree(&cvode_mem);});

      std::unique_ptr<_generic_N_Vector, NVectorDeleter> y_ =
          std::unique_ptr<_generic_N_Vector, NVectorDeleter>(nullptr, N_VDestroy);
      double t_;

      std::unique_ptr<_generic_SUNNonlinearSolver, NLSDeleter> NLS_ =
          std::unique_ptr<_generic_SUNNonlinearSolver, NLSDeleter>(nullptr, SUNNonlinSolFree);
      std::unique_ptr<_generic_N_Vector *, NVectorArrayDeleter> sens_ =
          std::unique_ptr<_generic_N_Vector *, NVectorArrayDeleter>(nullptr, [](N_Vector*){});
      std::unique_ptr<_generic_SUNNonlinearSolver, NLSDeleter> NLS_sens_ =
          std::unique_ptr<_generic_SUNNonlinearSolver, NLSDeleter>(nullptr, SUNNonlinSolFree);


      std::shared_ptr<user_data_func> udf_;
      int retval_;

      bool already_run_ = false;
    };

    template<typename T>
    const std::string CvodeWrapper<T>::ID = "cvodes-adams";

    template<typename T>
    CvodeWrapper<T>::CvodeWrapper(rhsFuncType<T> f, const T& y0, double t0)
    {
      y_.reset(SundialsComplexContent<T>::new_vector(y0));
      init(f, t0);
    }

    template<typename T>
    CvodeWrapper<T>::CvodeWrapper(rhsFuncType<T> f, T&& y0, double t0) {
      y_.reset(SundialsComplexContent<T>::new_vector(std::move(y0)));
      init(f, t0);
    }

    template<typename T>
    void CvodeWrapper<T>::setup_sens(perturbFuncType pf, const std::vector<double> &p) {
      if (already_run_) {
        throw std::runtime_error("Setup sensitivities before any integration step!");
      }
      init_sens(pf, p);
    }

    template<typename T>
    void CvodeWrapper<T>::init(rhsFuncType<T> f, double t0) {
      t_ = t0;

      cvode_mem_.reset(CVodeCreate(CV_ADAMS));
      CVodeInit(cvode_mem_.get(), rhs_wrapper, t0, y_.get());

      CVodeSetErrHandlerFn(cvode_mem_.get(), error_handler, nullptr);

      CVodeSStolerances(cvode_mem_.get(), RTOL, ATOL);
      NLS_.reset(SUNNonlinSol_FixedPoint(y_.get(), 0));

      /* attach nonlinear solver object to CVode */
      CVodeSetNonlinearSolver(cvode_mem_.get(), NLS_.get());
      udf_ = std::make_shared<user_data_func>();
      udf_->rhs = f;
      CVodeSetUserData(cvode_mem_.get(), udf_.get());

      // TODO: Hardcoded value??
      CVodeSetMaxConvFails(cvode_mem_.get(), 10000);
    }

    template<typename T>
    void CvodeWrapper<T>::init_sens(perturbFuncType pf, const std::vector<double> &p) {
      udf_->p = p;
      udf_->pf = pf;
      auto num_sens = p.size();
      sens_ = std::unique_ptr<_generic_N_Vector *, NVectorArrayDeleter>(
          N_VCloneVectorArray(num_sens, y_.get()),
          [num_sens](N_Vector *sens) { N_VDestroyVectorArray(sens, num_sens); });

      for (int is = 0; is < num_sens; is++) {
        N_VConst(0.0, sens_.get()[is]);
      }

      CVodeSensInit1(cvode_mem_.get(), num_sens, CV_SIMULTANEOUS, nullptr, sens_.get());

      CVodeSensEEtolerances(cvode_mem_.get());

      // TODO: Error control hardcoded to true
      CVodeSetSensErrCon(cvode_mem_.get(), true);
      CVodeSetSensDQMethod(cvode_mem_.get(), CV_CENTERED, 0.0);
      CVodeSetSensParams(cvode_mem_.get(), udf_->p.data(), nullptr, nullptr);

      // Assuming SIMULTANEOUS so far
      NLS_sens_.reset(SUNNonlinSol_FixedPointSens(num_sens + 1, y_.get(), 0));
      CVodeSetNonlinearSolverSensSim(cvode_mem_.get(), NLS_sens_.get());
    }

    template<typename T>
    void CvodeWrapper<T>::reinit() {
      CVodeReInit(cvode_mem_.get(), t_, y_.get());
    }

    template <typename T>
    void CvodeWrapper<T>::reinit_if_run() {
      if(already_run_) reinit();
    }

    template<typename T>
    const T &CvodeWrapper<T>::get_solution() const {
      return SundialsComplexContent<T>::get_data(y_.get());
    }

    template<typename T>
    const T &CvodeWrapper<T>::get_sens_solution(uint i) {

      if (i >= udf_->p.size()) {
        throw std::range_error("Trying to get sensitivity component outside limits");
      }
      CVodeGetSens(cvode_mem_.get(), &t_, sens_.get());
      return SundialsComplexContent<T>::get_data(sens_.get()[i]);
    }

    template<typename T>
    void CvodeWrapper<T>::set_t(double t) {
      t_ = t;
      reinit();
    }

    template<typename T>
    void CvodeWrapper<T>::set_solution(const T &y0) {
      SundialsComplexContent<T>::set_data(y_.get(), y0);
      reinit();
    }

    template <typename T>
    void CvodeWrapper<T>::set_solution(T &&y0) {
      SundialsComplexContent<T>::set_data(y_.get(), std::move(y0));
      reinit();
    }

    template<typename T>
    void CvodeWrapper<T>::set_intial_value(const T &y0, double t0) {
      t_ = t0;
      SundialsComplexContent<T>::set_data(y_.get(), y0);
      reinit();
    }

    template <typename T>
    void CvodeWrapper<T>::set_intial_value(T &&y0, double t0) {
      t_ = t0;
      SundialsComplexContent<T>::set_data(y_.get(), std::move(y0));
      reinit();
    }

    template<typename T>
    void CvodeWrapper<T>::integrate_n_steps(double at, int n_steps) {
      auto tout = t_ + at;
      for (int i = 0; i < n_steps; i++) {
        retval_ = CVode(cvode_mem_.get(), tout, y_.get(), &t_, CV_NORMAL);
        if (retval_ < 0) break;
        tout += at;
      }
    }

    template<typename T>
    void CvodeWrapper<T>::integrate(double t, bool one_step) {
      already_run_ = true;
      retval_ = CVode(cvode_mem_.get(), t, y_.get(), &t_, one_step ? CV_ONE_STEP : CV_NORMAL);
    }

    template<typename T>
    void CvodeWrapper<T>::set_tolerances(double abstol, double reltol) {
      CVodeSStolerances(cvode_mem_.get(), reltol, abstol);
      reinit_if_run();
    }

    template<typename T>
    void CvodeWrapper<T>::set_step_limits(double max_step, double min_step, double first_step) {
      CVodeSetMaxStep(cvode_mem_.get(), max_step);
      CVodeSetMinStep(cvode_mem_.get(), min_step);
      CVodeSetInitStep(cvode_mem_.get(), first_step);
    }

    template<typename T>
    void CvodeWrapper<T>::set_maximum_order(int order) {
      CVodeSetMaxOrd(cvode_mem_.get(), order);
    }

    template<typename T>
    void CvodeWrapper<T>::set_max_nsteps(int max_steps) {
      CVodeSetMaxNumSteps(cvode_mem_.get(), max_steps);
    }

  }
}

#endif //QASM_SIMULATOR_CVODE_WRAPPER_HPP
