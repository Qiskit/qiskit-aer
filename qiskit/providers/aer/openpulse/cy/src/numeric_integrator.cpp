#include <vector>
#include <complex>
#include <iostream>


#include "numeric_integrator.hpp"
#include "helpers.hpp"


rhs_ret_t td_ode_rhs(
    pyhton_op_dict_t global_data, pyhton_op_dict_t exp, unsigned char _register
){
    std::cout << "Entering the function\n";
    if(global_data == nullptr || exp == nullptr)
        return nullptr; // TODO: Throw a Python exception

    // 1. Get vec from global_data['h_ops_data'][0]
    auto vec = get_vec_from_dict_item<std::complex<double>>(global_data, "h_ops_data");
    unsigned int num_rows = vec.size();


    // unsigned int num_rows = vec.shape[0]
    // double complex dot, osc_term, coef
    // double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))
    return nullptr;
}
