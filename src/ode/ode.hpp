#ifndef QASM_SIMULATOR_ODE_H
#define QASM_SIMULATOR_ODE_H

namespace AER {

  namespace ODE {
    template<typename T>
    using rhsFuncType = std::function<void(double, const T&, T&)>;
    using perturbFuncType = std::function<void(const std::vector<double>&)>;

    template<typename T>
    class Ode {
    public:
      virtual ~Ode(){};

      virtual void integrate(double t, bool one_step = false) = 0;

      virtual void setup_sens(perturbFuncType pf, const std::vector<double> &p){
        throw std::runtime_error("This ODE solver lacks sensitivities computation feature.");
      };

      virtual double get_t() = 0;
      virtual const T& get_solution() const = 0;
      virtual const T& get_sens_solution(uint i)  {
        throw std::runtime_error("This ODE solver lacks sensitivities computation feature.");
      };

      virtual void set_t(double t) = 0;
      virtual void set_solution(const T& y0) = 0;
      virtual void set_intial_value(const T& y0, double t0) = 0;

      virtual void set_step_limits(double max_step, double min_step, double first_step_size) = 0;
      virtual void set_tolerances(double abstol, double reltol) = 0;
      virtual void set_maximum_order(int order) = 0;
      virtual void set_max_nsteps(int max_step) = 0;

      virtual bool succesful() = 0;
    };
  }
}
#endif //QASM_SIMULATOR_ODE_H
