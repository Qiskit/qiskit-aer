#ifndef _aer_misc_hacks_gcc_symbols_
#define _aer_misc_hacks_gcc_symbols_

#include <iostream>
#include <dlfcn.h>
#include <sys/types.h>

// Define undefined symbols
extern "C" {

    using GOMP_atomic_start_t = void(*)();
    GOMP_atomic_start_t _hook_GOMP_atomic_start;
    void GOMP_atomic_start (void){
        _hook_GOMP_atomic_start();
    }

    using GOMP_atomic_end_t = void(*)();
    GOMP_atomic_end_t _hook_GOMP_atomic_end;
    void GOMP_atomic_end (void){
        _hook_GOMP_atomic_end();
    }
    
    using GOMP_barrier_t = void(*)();
    GOMP_barrier_t _hook_GOMP_barrier;
    void GOMP_barrier(void){
        _hook_GOMP_barrier();
    }

    using GOMP_parallel_t = void(*)(void (*)(void *), void *, unsigned, unsigned);
    GOMP_parallel_t _hook_GOMP_parallel;
    void GOMP_parallel(void (*fn) (void *), void *data, unsigned num_threads, unsigned flags){
        _hook_GOMP_parallel(fn, data, num_threads, flags);
    }

    #define __KAI_KMPC_CONVENTION
    using omp_get_max_threads_t = int(*)(void);
    omp_get_max_threads_t _hook_omp_get_max_threads;
    int __KAI_KMPC_CONVENTION omp_get_max_threads(void) {
        return _hook_omp_get_max_threads();
    }
    using omp_set_nested_t = void(*)(int);
    omp_set_nested_t _hook_omp_set_nested;
    void __KAI_KMPC_CONVENTION omp_set_nested(int foo){
        _hook_omp_set_nested(foo);
    }
    using omp_get_num_threads_t = int(*)(void);
    omp_get_num_threads_t _hook_omp_get_num_threads;
    int __KAI_KMPC_CONVENTION omp_get_num_threads(void) {
        return _hook_omp_get_num_threads();
    }
    using omp_get_thread_num_t = int(*)(void);
    omp_get_thread_num_t _hook_omp_get_thread_num;
    int __KAI_KMPC_CONVENTION omp_get_thread_num(void) {
        return _hook_omp_get_thread_num();
    }

    void populate_hooks(void * handle){
        _hook_GOMP_atomic_end = reinterpret_cast<decltype(&GOMP_atomic_end)>(dlsym(handle, "GOMP_atomic_end"));
        _hook_GOMP_atomic_start = reinterpret_cast<decltype(&GOMP_atomic_start)>(dlsym(handle, "GOMP_atomic_start"));
        _hook_GOMP_barrier = reinterpret_cast<decltype(&GOMP_barrier)>(dlsym(handle, "GOMP_barrier"));
        _hook_GOMP_parallel = reinterpret_cast<decltype(&GOMP_parallel)>(dlsym(handle, "GOMP_parallel"));
        _hook_omp_get_num_threads = reinterpret_cast<decltype(&omp_get_num_threads)>(dlsym(handle, "omp_get_num_threads"));
        _hook_omp_get_max_threads = reinterpret_cast<decltype(&omp_get_max_threads)>(dlsym(handle, "omp_get_max_threads"));
        _hook_omp_set_nested = reinterpret_cast<decltype(&omp_set_nested)>(dlsym(handle, "omp_set_nested"));
    }

}

#endif