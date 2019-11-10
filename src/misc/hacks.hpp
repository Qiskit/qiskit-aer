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

#ifndef _aer_misc_hacks_hpp_
#define _aer_misc_hacks_hpp_

// We only need this hack for MacOS builds,
// on Linux and Windows everything works fine
#ifdef __APPLE__
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

namespace {
/*
 * Returns the path to the already loaded OpenMP library, if there's no
 *  library loaded, then load our default one.
 */

auto _apple_get_loaded_openmp_library = [](const std::string& default_path) -> std::string {
    // Iterate through all images currently in memory
    for (int32_t i = _dyld_image_count(); i >= 0 ; i--) {
        const char *image_name = _dyld_get_image_name(i);
        if(image_name) {
            // These are the only libraries we know that implement OpenMP on Mac
            // and that clash each other
            if(strstr(image_name, "libomp.dylib") ||
                strstr(image_name, "libiomp5.dylib") ||
                strstr(image_name, "libgomp.dylib")) {
                    return std::string(image_name);
                }
        }
    }
    return default_path + "/libomp.dylib";
};

/*
 * I hope this is a temporary hack, for fixing Issue:
 * https://github.com/Qiskit/qiskit-aer/issues/1
 */
auto _apple_maybe_load_openmp = [](const std::string& library_path) -> void {
    // dlopen() will return a handle to the library if this is already loaded
    void * handle = dlopen(_apple_get_loaded_openmp_library(library_path).c_str(), RTLD_LAZY);
    if(handle == nullptr){
        fprintf(stderr, "WARNING: Couldn't load libomp.dylib but we needed to. Error: %s\n", dlerror());
        fprintf(stderr, "Make sure you have libomp.dylib or libiomp5.dylib accesible to the program\n");
    }
    AER::Hacks::populate_hooks(handle);
};
}

#endif // endif __APPLE__

namespace AER {
namespace Hacks {
    #ifdef __APPLE__
        const auto maybe_load_openmp = ::_apple_maybe_load_openmp;
    #else
        const auto maybe_load_openmp = [](const std::string& dummy) -> void {};
    #endif
}
}

#endif
