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
#include <muparserx/parser/mpParser.h>

struct ParserValues {
    ParserValues(std::unique_ptr<mup::ParserX> parser, const std::string& expr):
        parser(std::move(parser)), expr(expr) {
    }
    std::unique_ptr<mup::ParserX> parser;
    std::string expr;
    std::unordered_map<std::string, std::unique_ptr<mup::Value>> var_values;
};

namespace std {
  template <>
  struct hash<ParserValues>{
    std::size_t operator()(const ParserValues& p) const {
      return std::hash<std::string>()(p.expr);
    }
  };
}

// TODO: Document
complex_t evaluate_hamiltonian_expression(const std::string& expr_string,
                                  const std::vector<double>& vars,
                                  const std::vector<std::string>& vars_names,
                                  const std::unordered_map<std::string, complex_t>& chan_values){


    static std::unordered_map<std::string, std::unique_ptr<ParserValues>> parser_expr;
    auto parser_iter = parser_expr.find(expr_string);
    if(parser_iter == parser_expr.end()){
        auto parserx = std::make_unique<mup::ParserX>();
        //Value pi(M_PI);
        //parser->DefineVar("npi", Variable(&pi));
        const auto replace = [](const std::string& from, const std::string& to, std::string where) -> std::string {
            size_t start_pos = 0;
            while((start_pos = where.find(from, start_pos)) != std::string::npos) {
                where.replace(start_pos, from.length(), to);
                start_pos += to.length();
            }
            return where;
        };
        parserx->SetExpr(replace("np.pi", "pi", expr_string));
        auto parser = std::make_unique<ParserValues>(std::move(parserx), expr_string);
        //std::cout << "Creating parser: " << std::hex << parser.get() << " for expr: " << expr_string << "\n";
        parser_expr.emplace(expr_string, std::move(parser));
    }
    auto * parser = parser_expr[expr_string].get();

    //std::cout << "Getting parser " << std::hex << parser << "\n";

    auto maybe_update_value = [parser](const std::string& var_name, const complex_t& var_value){
        if(parser->var_values.find(var_name) == parser->var_values.end()){
            parser->var_values.emplace(var_name, std::make_unique<mup::Value>(var_value));
            parser->parser->DefineVar(var_name, mup::Variable(parser->var_values[var_name].get()));
        }else{ // There's already a variable defined for this expresion
            //std::cout << var_name << " is now: " << std::to_string(var_value.real()) << "," << std::to_string(var_value.imag()) << "\n";
            auto * ref = parser->var_values[var_name].get();
            // Update the value from the container
            *ref = var_value;
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
        mup::Value result = parser->parser->Eval();
        return result.GetComplex();
    }catch(std::exception ex){
        std::cout << ex.what();
    }
    return 0.;
}

#endif //_EVAL_HAMILTONIAN_HPP
