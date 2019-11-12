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

#ifndef _EVAL_HAMILTONIAN_HPP
#define _EVAL_HAMILTONIAN_HPP

#include <unordered_map>
#include <vector>
#include <complex>
#include <numpy/arrayobject.h>
#include <muparserx/mpParser.h>

// TODO: Document
const complex_t evaluate_hamiltonian_expression(const std::string& expr_string,
                                  const std::vector<double>& vars,
                                  const std::vector<std::string>& vars_names,
                                  const std::unordered_map<std::string, complex_t>& chan_values){
    using namespace mup;
    ParserX parser;
    Value pi(M_PI);
    parser.DefineVar("npi", Variable(&pi));

    std::vector<Value> values;
    values.reserve(vars.size() + chan_values.size());
    for(const auto& idx_var : enumerate(vars)){
        auto index = idx_var.first;
        auto var = static_cast<complex_t>(idx_var.second);
        values.emplace_back(Value(var));
        parser.DefineVar(vars_names[index], Variable(&values[values.size()-1]));
    }

    for(const auto& idx_channel : chan_values){
        auto channel = idx_channel.first; // The string of the channel
        auto var = idx_channel.second; // The complex_t of the map
        values.emplace_back(Value(var));
        parser.DefineVar(channel, Variable(&values[values.size()-1]));
    }

    const auto replace = [](const std::string& from, const std::string& to, std::string where) -> std::string {
        size_t start_pos = 0;
        while((start_pos = where.find(from, start_pos)) != std::string::npos) {
            where.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
        return where;
    };

    // This is needed because muparserx doesn't support . as part of the var name
    auto filtered_expr = replace("np.pi", "npi", expr_string);
    parser.SetExpr(filtered_expr);
    Value result = parser.Eval();

    return result.GetComplex();
}

#endif //_EVAL_HAMILTONIAN_HPP
