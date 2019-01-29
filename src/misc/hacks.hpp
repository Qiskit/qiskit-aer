#ifndef _aer_misc_hacks_hpp_
#define _aer_misc_hacks_hpp_

#include <mach-o/dyld.h>
#include <mach-o/nlist.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <functional>

#if defined(__clang__)
    #include "clang_omp_symbols.hpp"
#elif defined(__GNUC__) || defined(__GNUG__)
    #include "gcc_omp_symbols.hpp"
#endif

namespace AER {
namespace Hacks {

    // TODO: We may need to remove this function, because it's not used anymore. I want to keep it here
    // for a while just in case we'll need it in a future.
    bool _is_openmp_loaded(){
        // Iterate through all images currently in memory
        for (int32_t i = _dyld_image_count(); i >= 0 ; i--) {
            // dlopen() each image
            const char *image_name = _dyld_get_image_name(i);
            if(image_name) {
                // These are the only libraries we know that implement OpenMP on Mac
                // and that clash each other 
                if(strstr(image_name, "libomp.dylib") ||
                   strstr(image_name, "libiomp5.dylib") ||
                   strstr(image_name, "libgomp.dylib")) {
                       return true;
                   }
            }
        }
        return false;
    }
    /**
     * I hope this is a temporary hack, for fixing Issue: 
     * https://github.com/Qiskit/qiskit-aer/issues/1
     */
    void maybe_load_openmp(){
        void * handle = dlopen("libomp.dylib", RTLD_LAZY);
        if(handle == NULL){
            fprintf(stderr, "WARNING: Couldn't load libomp.dylib but we needed to. Error: %s\n", dlerror());
            fprintf(stderr, "Make sure you have libomp.dylib or libiomp5.dylib accesible to the program\n");
        }

        populate_hooks(handle);
    }
}
}


#endif
