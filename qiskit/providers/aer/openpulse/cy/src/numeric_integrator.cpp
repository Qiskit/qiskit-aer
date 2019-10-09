#include <vector>
#include <complex>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "numeric_integrator.hpp"
#include "helpers.hpp"

using complex_t = std::complex<double>;


complex_t chan_value(
    double t,
    unsigned int chan_num,
    double freq_ch,
    std::vector<double>& chan_pulse_times,
    std::vector<complex_t>& pulse_array,
    std::vector<unsigned int>& pulse_ints,
    std::vector<double>& fc_array,
    std::vector<unsigned char>& reg){

    return complex_t(0.0, 0.0);
}

PyObject * td_ode_rhs(
    PyObject * global_data,
    PyObject * channels,
    PyObject * vars,
    PyObject * freqs,
    PyObject * exp,
    unsigned char _register){

    std::cout << "Entering the function\n";
    if(global_data == nullptr || exp == nullptr)
        return nullptr; // TODO: Throw a Python exception

    // 1. Get vec from global_data['h_ops_data'][0]
    auto vec = get_vec_from_dict_item<complex_t>(global_data, "h_ops_data");
    // TODO: Not quite sure about vec.size()= shape?
    // unsigned int num_rows = vec.shape[0]
    unsigned int num_rows = vec.size();
    std::cout << "vec.size(): " << vec.size() << "\n";

    // 2. double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))
    auto out = std::make_unique<complex_t>(
        PyDataMem_NEW_ZEROED(num_rows, sizeof(complex_t))
    );

    // 3. Compute complex channel values at time `t`
    // D0 = chan_value(t, 0, D0_freq, D0_pulses,  pulse_array, pulse_indices, D0_fc, register)
    // U0 = chan_value(t, 1, U0_freq, U0_pulses,  pulse_array, pulse_indices, U0_fc, register)
    // D1 = chan_value(t, 2, D1_freq, D1_pulses,  pulse_array, pulse_indices, D1_fc, register)
    // U1 = chan_value(t, 3, U1_freq, U1_pulses,  pulse_array, pulse_indices, U1_fc, register)
    ////
    // for chan, idx in self.op_system.channels.items():
    // chan_str = "%s = chan_value(t, %s, %s_freq, " % (chan, idx, chan) + \
    //            "%s_pulses,  pulse_array, pulse_indices, " % chan + \
    //            "%s_fc, register)" % (chan)

    auto channels_map = get_map_from_py_dict<std::string, double>(channels);
    for(auto& channel : channels_map) {
        
    }




    return nullptr;
}
