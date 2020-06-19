#ifndef QASM_SIMULATOR_ABM_WRAPPER_HPP
#define QASM_SIMULATOR_ABM_WRAPPER_HPP

#include "ode/ode.h"
// TODO: Specific header to include instead of all odeint
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

namespace AER {

  template <typename T>
  class ABMWrapper : Ode<T> {
  public:
    using rhsFuncType = std::function<void(double, const T&, T&)>;

    ABMWrapper(rhsFuncType f, const T& y0, double t0) : t_(t0), y_(y0) {
      rhs_ = [f](const T& y, T& ydot, double t){ f(t, y, ydot);};
    }

    void integrate(double t, bool one_step=false){
      // TODO: Ho do we compute step size?
      double step = std::max(min_step_, std::min(max_step_, (t - t_) / 1500.));
      integrate_adaptive( abm_ ,rhs_ ,y_ ,t_ ,t , step);
      t_ = t;
      //integrate_adaptive(make_controlled(abstol_, reltol_, max_step_, abm_) ,rhs_ ,y_ ,t_ ,t , step);
    }

    const T& get_solution() const{
      return y_;
    }

    void set_solution(const T& y0){
      y_ = y0;
    }
    void set_intial_value(const T& y0, double t0){
      y_ = y0;
      t_ = t0;
    }

    void set_step_limits(double max_step, double min_step, double first_step_size){
      max_step_ = max_step;
      min_step_ = min_step;
      first_step_size_ = first_step_size;
    }

    void set_tolerances(double abstol, double reltol){
      abstol_ = abstol;
      reltol_ = reltol;
    };

    void set_maximum_order(int order){};
    void set_max_nsteps(int max_step){};

    double t_;
    T y_;

  private:
    std::function<void(const T&, T&, double)> rhs_;
    double max_step_ = 1e6;
    double min_step_ = 1e-12;
    double first_step_size_ = 1e-12;
    double abstol_ = 1e-6;
    double reltol_ = 1e-8;
    adams_bashforth_moulton<8, T> abm_;
  };

}

#endif //QASM_SIMULATOR_ABM_WRAPPER_HPP
