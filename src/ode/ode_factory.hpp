#ifndef QASM_SIMULATOR_ODE_FACTORY_H
#define QASM_SIMULATOR_ODE_FACTORY_H

#include "ode.hpp"
#include "sundials_wrappers/sundials_cvode_wrapper.hpp"
#include "odeint_wrappers/abm_wrapper.hpp"

namespace AER {
  namespace ODE {
    template<typename T, typename ...Params>
    std::unique_ptr<Ode<T>> create_ode(const std::string &ode_type, Params &&... params) {
      if (ode_type == CvodeWrapper<T>::ID) {
        return std::unique_ptr<Ode<T>>(new CvodeWrapper<T>(std::forward<Params>(params)...));
      } else if (ode_type == ABMWrapper<T>::ID) {
        return std::unique_ptr<Ode<T>>(new ABMWrapper<T>(std::forward<Params>(params)...));
      } else {
        throw std::invalid_argument("No suitable constructor for ODE type: " + ode_type);
      }
    }

    template<typename T>
    std::vector<std::string> get_cpp_wrappers_list() {
      static std::vector<std::string> cpp_wrappers{{CvodeWrapper<T>::ID, ABMWrapper<T>::ID}};
      return cpp_wrappers;
    }
  }
}

#endif //QASM_SIMULATOR_ODE_FACTORY_H
