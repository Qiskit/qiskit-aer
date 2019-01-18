#ifndef _aer_misc_hacks_gcc_symbols_
#define _aer_misc_hacks_gcc_symbols_

#include <iostream>
#include <sys/types.h>

// Define undefined symbols
extern "C" {
    #define WEAK __attribute__((weak_import))

    WEAK void GOMP_critical_start (void);
    WEAK void GOMP_critical_end (void);
    WEAK void GOMP_critical_name_start (void **);
    WEAK void GOMP_critical_name_end (void **);
    WEAK void GOMP_atomic_start (void);
    void GOMP_atomic_end (void){
        fprintf(stderr, "GOMP_atomic_end()\n");
    }
}

#endif