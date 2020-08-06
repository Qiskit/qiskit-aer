
#ifndef QASM_SIMULATOR_COMMON_MACROS_HPP
#define QASM_SIMULATOR_COMMON_MACROS_HPP

#if defined(__GNUC__) && defined(__x86_64__)
 #define GNUC_AVX 1
#else
 #define GNUC_AVX 0
#endif

#endif //QASM_SIMULATOR_COMMON_MACROS_HPP
