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

#ifndef _LOG_HPP
#define _LOG_HPP

#include <unordered_map>
#include <vector>
#include <complex>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "types.hpp"
#include "python_to_cpp.hpp"
#include "nparray.hpp"

template<typename T>
void jlog(const std::string& msg, const T& value){
    spdlog::debug("{}: {}", msg, value);
}

template<>
void jlog(const std::string& msg, const complex_t& values){
    spdlog::debug("{}: [{},{}i]", msg, values.real(), values.imag());
}

template<typename T>
void jlog(const std::string& msg, const NpArray<T>& values){
    spdlog::debug("{}", msg);
    spdlog::debug(".shape: ");
    for(const auto& shape : values.shape)
        spdlog::debug("{} ", shape);

    spdlog::debug("\n.data: ");
    for(const auto& val : values.data){
        jlog("", val);
    }
}

template<typename T>
void jlog(const std::string& msg, const std::vector<T>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        jlog("", val);
    }
}

template<>
void jlog(const std::string& msg, const std::unordered_map<std::string, std::vector<std::vector<double>>>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        for(const auto& inner: val.second){
            for(const auto& inner2: inner){
                spdlog::debug("{}:{} ", val.first, inner2);
            }
        }
    }
}

template<>
void jlog(const std::string& msg, const std::unordered_map<std::string, double>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        spdlog::debug("{}:{} ", val.first, val.second);
    }
}

template<>
void jlog(const std::string& msg, const std::unordered_map<std::string, std::vector<NpArray<double>>>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        for(const auto& inner: val.second){
            jlog(val.first, inner);
        }
    }
}

template<>
void jlog(const std::string& msg, const ordered_map<std::string, std::vector<NpArray<double>>>& values){
    spdlog::debug("{}", msg);
    using order_map_t = ordered_map<std::string, std::vector<NpArray<double>>>;
    for(const auto& val : const_cast<order_map_t&>(values)){
        for(const auto& inner: val.second){
            jlog(val.first, inner);
        }
    }
}


#endif //_LOG_HPP