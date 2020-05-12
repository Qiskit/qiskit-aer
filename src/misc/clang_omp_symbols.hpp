/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_misc_hacks_clang_symbols_
#define _aer_misc_hacks_clang_symbols_

/*
 * This is some sort of "black magic" to solve a problem we have with OpenMP libraries on Mac.
 * The problem is actually in the library itself, but it's out of our control, so we had to
 * fix it this way.
 * Symbol signatures are taken from: https://github.com/llvm/llvm-project/blob/master/openmp/runtime/src/kmp.h
 */

#include <dlfcn.h>
#include <sys/types.h>

// Define undefined symbols
extern "C" {

    typedef struct {
        int reserved_1;
        int flags;
        int reserved_2;
        int reserved_3;
        char *psource;
    }id;

    typedef int kmp_int32;
    typedef struct ident {
        kmp_int32 reserved_1; /**<  might be used in Fortran; see above  */
        kmp_int32 flags; /**<  also f.flags; KMP_IDENT_xxx flags; KMP_IDENT_KMPC
                            identifies this union member  */
        kmp_int32 reserved_2; /**<  not really used in Fortran any more; see above */
        kmp_int32 reserved_3; /**<  source[4] in Fortran, do not use for C++  */
        char const *psource; /**<  String describing the source location.
                            The string is composed of semi-colon separated fields
                            which describe the source file, the function and a pair
                            of line numbers that delimit the construct. */
    } ident_t;
    typedef ident_t kmp_Ident;

    using __kmpc_barrier_t = void(*)(id*, int);
    __kmpc_barrier_t _hook__kmpc_barrier;
    void __kmpc_barrier(id* pId, int gtid){
        _hook__kmpc_barrier(pId, gtid);
    }

    using __kmpc_for_static_fini_t =  void(*)(kmp_Ident *, int32_t);
    __kmpc_for_static_fini_t _hook__kmpc_for_static_fini;
    void __kmpc_for_static_fini(kmp_Ident *loc, int32_t global_tid){
        _hook__kmpc_for_static_fini(loc, global_tid);
    }

    using __kmpc_for_static_init_4_t = void(*)(kmp_Ident *, int32_t, int32_t, int32_t *,
                                     int32_t *, int32_t *, int32_t *, int32_t, int32_t);
    __kmpc_for_static_init_4_t _hook__kmpc_for_static_init_4;
    void __kmpc_for_static_init_4(kmp_Ident *loc, int32_t global_tid,
                                     int32_t sched, int32_t *plastiter,
                                     int32_t *plower, int32_t *pupper,
                                     int32_t *pstride, int32_t incr,
                                     int32_t chunk){
        _hook__kmpc_for_static_init_4(loc, global_tid, sched, plastiter, plower, pupper, pstride,
                                       incr, chunk);
    }

    using __kmpc_for_static_init_8_t = void(*)(kmp_Ident *, int32_t, int32_t, int32_t *,
                                     int64_t *, int64_t *, int64_t *, int64_t, int64_t);
    __kmpc_for_static_init_8_t _hook__kmpc_for_static_init_8;
    void __kmpc_for_static_init_8(kmp_Ident *loc, int32_t global_tid,
                                     int32_t sched, int32_t *plastiter,
                                     int64_t *plower, int64_t *pupper,
                                     int64_t *pstride, int64_t incr,
                                     int64_t chunk){
        _hook__kmpc_for_static_init_8(loc, global_tid, sched, plastiter, plower, pupper, pstride,
                                       incr, chunk);
    }

    using __kmpc_for_static_init_8u_t = void(*)(kmp_Ident *, int32_t, int32_t, int32_t *, uint64_t *, uint64_t *,
                                      int64_t *, int64_t, int64_t);
    __kmpc_for_static_init_8u_t _hook__kmpc_for_static_init_8u;
    void __kmpc_for_static_init_8u(kmp_Ident *loc, int32_t global_tid,
                                      int32_t sched, int32_t *plastiter1,
                                      uint64_t *plower, uint64_t *pupper,
                                      int64_t *pstride, int64_t incr,
                                      int64_t chunk){
        _hook__kmpc_for_static_init_8u(loc, global_tid, sched, plastiter1, plower, pupper, pstride,
                                       incr, chunk);
    }

    using kpm_int32 = int;
    typedef void (*kmp_ParFctPtr)(kpm_int32 *global_tid, kpm_int32 *bound_tid, ...);
    using __kmpc_fork_call_t = void(*)(kmp_Ident *, kpm_int32, kmp_ParFctPtr, ...);
    __kmpc_fork_call_t _hook__kmpc_fork_call;
    #include <stdarg.h>
    void __kmpc_fork_call(kmp_Ident *loc, kpm_int32 argc, kmp_ParFctPtr microtask, ...){
        va_list argptr;
        va_start(argptr, microtask);
        // The types are always pointer to void (from llvm kmp_runtime.cpp)
        void * arg1 = va_arg(argptr, void *);
        void * arg2 = va_arg(argptr, void *);
        void * arg3 = va_arg(argptr, void *);
        void * arg4 = va_arg(argptr, void *);
        void * arg5 = va_arg(argptr, void *);
        void * arg6 = va_arg(argptr, void *);
        void * arg7 = va_arg(argptr, void *);
        void * arg8 = va_arg(argptr, void *);
        void * arg9 = va_arg(argptr, void *);
        _hook__kmpc_fork_call(loc, argc, microtask, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
        va_end(argptr);
    }

    using __kmpc_push_num_threads_t = void(*)(kmp_Ident *, int32_t, int32_t);
    __kmpc_push_num_threads_t _hook__kmpc_push_num_threads;
    void __kmpc_push_num_threads(kmp_Ident *loc, int32_t global_tid, int32_t num_threads){
        _hook__kmpc_push_num_threads(loc, global_tid, num_threads);
    }

    typedef kmp_int32 kmp_critical_name[8];
    using __kmpc_reduce_nowait_t = kmp_int32(*)(ident_t *, kmp_int32, kmp_int32, size_t,
                                void *, void (*)(void *lhs_data, void *rhs_data),kmp_critical_name *);
    __kmpc_reduce_nowait_t _hook__kmpc_reduce_nowait;
    kmp_int32 __kmpc_reduce_nowait(
                                ident_t *loc, kmp_int32 global_tid, kmp_int32 num_vars, size_t reduce_size,
                                void *reduce_data, void (*reduce_func)(void *lhs_data, void *rhs_data),
                                kmp_critical_name *lck){
        return _hook__kmpc_reduce_nowait(loc, global_tid, num_vars, reduce_size, reduce_data, reduce_func, lck);
    }


    using __kmpc_for_static_init_4u_t = void(*)(kmp_Ident *, int32_t, int32_t, int32_t *, uint32_t *, uint32_t *,
                                                int32_t *, int32_t, int32_t);
    __kmpc_for_static_init_4u_t _hook__kmpc_for_static_init_4u;
    void __kmpc_for_static_init_4u(kmp_Ident *loc, int32_t global_tid,
                                      int32_t sched, int32_t *plastiter,
                                      uint32_t *plower, uint32_t *pupper,
                                      int32_t *pstride, int32_t incr,
                                      int32_t chunk){
        _hook__kmpc_for_static_init_4u(loc, global_tid, sched, plastiter, plower, pupper, pstride, incr, chunk);
    }

    using __kmpc_end_reduce_nowait_t = void(*)(ident_t *, kmp_int32,   kmp_critical_name *);
    __kmpc_end_reduce_nowait_t _hook__kmpc_end_reduce_nowait;
    void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid, kmp_critical_name *lck){
        _hook__kmpc_end_reduce_nowait(loc, global_tid, lck);
    }

    using __kmpc_serialized_parallel_t = void(*)(kmp_Ident *loc, uint32_t global_tid);
    __kmpc_serialized_parallel_t _hook__kmpc_serialized_parallel;
    void __kmpc_serialized_parallel(kmp_Ident *loc, uint32_t global_tid){
        _hook__kmpc_serialized_parallel(loc, global_tid);
    }

    using __kmpc_end_serialized_parallel_t = void(*)(ident_t *loc, kmp_int32 global_tid);
    __kmpc_end_serialized_parallel_t _hook__kmpc_end_serialized_parallel;
    void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 global_tid){
        _hook__kmpc_end_serialized_parallel(loc, global_tid);
    }

    using __kmpc_global_thread_num_t = int(*)(id*);
    __kmpc_global_thread_num_t _hook__kmpc_global_thread_num;
    int __kmpc_global_thread_num(id* pId){
        return _hook__kmpc_global_thread_num(pId);
    }

    using __kmpc_critical_t = void(*)(ident_t *, kmp_int32, kmp_critical_name *);
    __kmpc_critical_t _hook__kmpc_critical;
    void __kmpc_critical(ident_t *id, kmp_int32 global_tid, kmp_critical_name *lck){
        return _hook__kmpc_critical(id, global_tid, lck);
    }

    using __kmpc_end_critical_t = void(*)(ident_t *, kmp_int32, kmp_critical_name *);
    __kmpc_end_critical_t _hook__kmpc_end_critical;
    void __kmpc_end_critical(ident_t *id, kmp_int32 global_tid, kmp_critical_name *lck){
        return _hook__kmpc_end_critical(id, global_tid, lck);
    }

    using __kmpc_master_t = kmp_int32(*)(ident_t *, kmp_int32);
    __kmpc_master_t _hook__kmpc_master;
    kmp_int32 __kmpc_master(ident_t *id, kmp_int32 global_tid){
        return _hook__kmpc_master(id, global_tid);
    }

    using __kmpc_end_master_t = void(*)(ident_t *, kmp_int32);
    __kmpc_end_master_t _hook__kmpc_end_master;
    void __kmpc_end_master(ident_t *id, kmp_int32 global_tid){
        return _hook__kmpc_end_master(id, global_tid);
    }

    #define __KAI_KMPC_CONVENTION
    using omp_get_max_threads_t = int(*)(void);
    omp_get_max_threads_t _hook_omp_get_max_threads;
    int __KAI_KMPC_CONVENTION omp_get_max_threads(void) {
        return _hook_omp_get_max_threads();
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
    using omp_set_nested_t = void(*)(int);
    omp_set_nested_t _hook_omp_set_nested;
    void __KAI_KMPC_CONVENTION omp_set_nested(int foo){
        _hook_omp_set_nested(foo);
    }
    using omp_get_num_procs_t = int(*)(void);
    omp_get_num_procs_t _hook_omp_get_num_procs;
    int __KAI_KMPC_CONVENTION omp_get_num_procs(void) {
        return _hook_omp_get_num_procs();
    }


    // Symbols above this line would be needed in a future, if clang changes
    // the OpenMP implementation. So I'll keep them here just in case I need
    // them in the future
    void __kmpc_dispatch_init_4(kmp_Ident *loc, int32_t global_tid,
                                   int32_t sched, int32_t lower, int32_t upper,
                                   int32_t incr, int32_t chunk); //{}
    void __kmpc_dispatch_init_4u(kmp_Ident *loc, int32_t global_tid,
                                    int32_t sched, uint32_t lower,
                                    uint32_t upper, int32_t incr,
                                    int32_t chunk); //{}
    void __kmpc_dispatch_init_8(kmp_Ident *loc, int32_t global_tid,
                                   int32_t sched, int64_t lower, int64_t upper,
                                   int64_t incr, int64_t chunk); //{}
    void __kmpc_dispatch_init_8u(kmp_Ident *loc, int32_t global_tid,
                                    int32_t sched, uint64_t lower,
                                    uint64_t upper, int64_t incr,
                                    int64_t chunk); //{}
    int __kmpc_dispatch_next_4(kmp_Ident *loc, int32_t global_tid,
                                  int32_t *plastiter, int32_t *plower,
                                  int32_t *pupper, int32_t *pstride); //{}
    int __kmpc_dispatch_next_4u(kmp_Ident *loc, int32_t global_tid,
                                   int32_t *plastiter, uint32_t *plower,
                                   uint32_t *pupper, int32_t *pstride); //{}
    int __kmpc_dispatch_next_8(kmp_Ident *loc, int32_t global_tid,
                                  int32_t *plastiter, int64_t *plower,
                                  int64_t *pupper, int64_t *pstride); //{}
    int __kmpc_dispatch_next_8u(kmp_Ident *loc, int32_t global_tid,
                                   int32_t *plastiter, uint64_t *plower,
                                   uint64_t *pupper, int64_t *pstride); //{}
    void __kmpc_dispatch_fini_4(kmp_Ident *loc, int32_t global_tid); //{}
    void __kmpc_dispatch_fini_4u(kmp_Ident *loc, int32_t global_tid); //{}
    void __kmpc_dispatch_fini_8(kmp_Ident *loc, int32_t global_tid); //{}
    void __kmpc_dispatch_fini_8u(kmp_Ident *loc, int32_t global_tid); //{}
    void __kmpc_for_static_init_4_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                          int32_t sched, int32_t *plastiter,
                                          int32_t *plower, int32_t *pupper,
                                          int32_t *pstride, int32_t incr,
                                          int32_t chunk); //{}
    void __kmpc_for_static_init_4u_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                           int32_t sched, int32_t *plastiter,
                                           uint32_t *plower, uint32_t *pupper,
                                           int32_t *pstride, int32_t incr,
                                           int32_t chunk); //{}
    void __kmpc_for_static_init_8_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                          int32_t sched, int32_t *plastiter,
                                          int64_t *plower, int64_t *pupper,
                                          int64_t *pstride, int64_t incr,
                                          int64_t chunk); //{}
    void __kmpc_for_static_init_8u_simple_spmd(kmp_Ident *loc, int32_t global_tid,
                                           int32_t sched, int32_t *plastiter1,
                                           uint64_t *plower, uint64_t *pupper,
                                           int64_t *pstride, int64_t incr,
                                           int64_t chunk); //{}
    void __kmpc_for_static_init_4_simple_generic(kmp_Ident *loc,
                                             int32_t global_tid, int32_t sched,
                                             int32_t *plastiter,
                                             int32_t *plower, int32_t *pupper,
                                             int32_t *pstride, int32_t incr,
                                             int32_t chunk); //{}
    void __kmpc_for_static_init_4u_simple_generic(
                                            kmp_Ident *loc, int32_t global_tid, int32_t sched, int32_t *plastiter,
                                            uint32_t *plower, uint32_t *pupper, int32_t *pstride, int32_t incr,
                                            int32_t chunk); //{}
    void __kmpc_for_static_init_8_simple_generic(kmp_Ident *loc,
                                             int32_t global_tid, int32_t sched,
                                             int32_t *plastiter,
                                             int64_t *plower, int64_t *pupper,
                                             int64_t *pstride, int64_t incr,
                                             int64_t chunk); //{}
    void __kmpc_for_static_init_8u_simple_generic(
                                            kmp_Ident *loc, int32_t global_tid, int32_t sched, int32_t *plastiter1,
                                            uint64_t *plower, uint64_t *pupper, int64_t *pstride, int64_t incr,
                                            int64_t chunk); //{}
}


namespace AER {
namespace Hacks {

void populate_hooks(void * handle){
    _hook__kmpc_barrier = reinterpret_cast<decltype(&__kmpc_barrier)>(dlsym(handle, "__kmpc_barrier"));
    _hook__kmpc_for_static_fini = reinterpret_cast<decltype(&__kmpc_for_static_fini)>(dlsym(handle, "__kmpc_for_static_fini"));
    _hook__kmpc_end_reduce_nowait = reinterpret_cast<decltype(&__kmpc_end_reduce_nowait)>(dlsym(handle, "__kmpc_end_reduce_nowait"));
    _hook__kmpc_reduce_nowait = reinterpret_cast<decltype(&__kmpc_reduce_nowait)>(dlsym(handle, "__kmpc_reduce_nowait"));
    _hook__kmpc_for_static_init_4 = reinterpret_cast<decltype(&__kmpc_for_static_init_4)>(dlsym(handle, "__kmpc_for_static_init_4"));
    _hook__kmpc_for_static_init_8 = reinterpret_cast<decltype(&__kmpc_for_static_init_8)>(dlsym(handle, "__kmpc_for_static_init_8"));
    _hook__kmpc_for_static_init_8u = reinterpret_cast<decltype(&__kmpc_for_static_init_8u)>(dlsym(handle, "__kmpc_for_static_init_8u"));
    _hook__kmpc_fork_call = reinterpret_cast<decltype(&__kmpc_fork_call)>(dlsym(handle, "__kmpc_fork_call"));
    _hook__kmpc_push_num_threads = reinterpret_cast<decltype(&__kmpc_push_num_threads)>(dlsym(handle, "__kmpc_push_num_threads"));
    _hook__kmpc_serialized_parallel = reinterpret_cast<decltype(&__kmpc_serialized_parallel)>(dlsym(handle, "__kmpc_serialized_parallel"));
    _hook__kmpc_end_serialized_parallel = reinterpret_cast<decltype(&__kmpc_end_serialized_parallel)>(dlsym(handle, "__kmpc_end_serialized_parallel"));
    _hook__kmpc_global_thread_num = reinterpret_cast<decltype(&__kmpc_global_thread_num)>(dlsym(handle, "__kmpc_global_thread_num"));
    _hook__kmpc_critical = reinterpret_cast<decltype(&__kmpc_critical)>(dlsym(handle, "__kmpc_critical"));
    _hook__kmpc_end_critical = reinterpret_cast<decltype(&__kmpc_end_critical)>(dlsym(handle, "__kmpc_end_critical"));
    _hook__kmpc_master = reinterpret_cast<decltype(&__kmpc_master)>(dlsym(handle, "__kmpc_master"));
    _hook__kmpc_end_master = reinterpret_cast<decltype(&__kmpc_end_master)>(dlsym(handle, "__kmpc_end_master"));
    _hook_omp_get_max_threads = reinterpret_cast<decltype(&omp_get_max_threads)>(dlsym(handle, "omp_get_max_threads"));
    _hook_omp_get_num_threads = reinterpret_cast<decltype(&omp_get_num_threads)>(dlsym(handle, "omp_get_num_threads"));
    _hook_omp_get_thread_num = reinterpret_cast<decltype(&omp_get_thread_num)>(dlsym(handle, "omp_get_thread_num"));
    _hook_omp_set_nested = reinterpret_cast<decltype(&omp_set_nested)>(dlsym(handle, "omp_set_nested"));
    _hook_omp_get_num_procs = reinterpret_cast<decltype(&omp_get_num_procs)>(dlsym(handle, "omp_get_num_procs"));
}

}
}

#endif
