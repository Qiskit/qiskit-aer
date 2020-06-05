#ifndef QASM_SIMULATOR_ODE_FACTORY_H
#define QASM_SIMULATOR_ODE_FACTORY_H

#include "ode.h"

template <typename T>
Ode<T> createOde(const std::string& type, OdeMethod ode_method, unsigned int dimension);

#endif //QASM_SIMULATOR_ODE_FACTORY_H
