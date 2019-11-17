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
#include <string>
#include <numpy/arrayobject.h>
#include <muparserx/mpParser.h>

using namespace mup;

// TODO: Document
complex_t evaluate_hamiltonian_expression(const std::string& expr_string,
                                  const std::vector<double>& vars,
                                  const std::vector<std::string>& vars_names,
                                  const std::unordered_map<std::string, complex_t>& chan_values){


    static std::unordered_map<std::string, std::unique_ptr<ParserX>> expr_strings_parser;
    auto parser_iter = expr_strings_parser.find(expr_string);
    if(parser_iter == expr_strings_parser.end()){
        auto parser = std::make_unique<ParserX>();
        //Value pi(M_PI);
        //parser->DefineVar("npi", Variable(&pi));

        std::cout << "Creating parser: " << std::hex << parser << " for expr: " << expr_string << "\n";

        const auto replace = [](const std::string& from, const std::string& to, std::string where) -> std::string {
            size_t start_pos = 0;
            while((start_pos = where.find(from, start_pos)) != std::string::npos) {
                where.replace(start_pos, from.length(), to);
                start_pos += to.length();
            }
            return where;
        };
        parser->SetExpr(replace("np.pi", "pi", expr_string));
        expr_strings_parser.emplace(expr_string, std::move(parser));
    }
    auto& parser = expr_strings_parser[expr_string];

    std::cout << "Getting parser " << std::hex << parser << "\n";

    static std::unordered_map<std::string, std::unique_ptr<Value>> var_values;
    auto maybe_update_value = [&parser](const std::string& var_name, const complex_t& var_value){
        if(var_values.find(var_name) == var_values.end()){
            var_values.emplace(var_name, std::make_unique<Value>(var_value));
            parser->DefineVar(var_name, Variable(&(*var_values[var_name])));
        }else{
            std::cout << var_name << " is now: " << std::to_string(var_value.real()) << "," << std::to_string(var_value.imag()) << "\n";
            auto& ref = var_values[var_name];
            ref = var_value;
        }
    };

    for(const auto& idx_var : enumerate(vars)){
        size_t index = idx_var.first;
        auto var_value = static_cast<complex_t>(idx_var.second);
        maybe_update_value(vars_names[index], var_value);
    }

    for(const auto& idx_channel : chan_values){
        auto channel = idx_channel.first; // The string of the channel
        auto var_value = idx_channel.second; // The complex_t of the map
        maybe_update_value(channel, var_value);
    }

    try{
        Value result = parser->Eval();
        return result.GetComplex();
    }catch(std::exception ex){
        std::cout << ex.what();
    }
    return 0.;
}

#endif //_EVAL_HAMILTONIAN_HPP
