#ifndef QASM_SIMULATOR_ODE_H
#define QASM_SIMULATOR_ODE_H

namespace AER {

  enum class OdeMethod {
    BDF, ADAMS
  };

  template<typename T>
  class Ode {
  public:
    virtual void integrate(double t, bool one_step = false) = 0;

    virtual const T &get_solution() const = 0;

    virtual void set_intial_value(const T &y0, double t0) = 0;
  };
}
#endif //QASM_SIMULATOR_ODE_H
