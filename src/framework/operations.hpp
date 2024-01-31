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

#ifndef _aer_framework_operations_hpp_
#define _aer_framework_operations_hpp_

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "framework/json_parser.hpp"
#include "framework/linalg/almost_equal.hpp"
#include "framework/types.hpp"
#include "framework/utils.hpp"
#include "simulators/stabilizer/clifford.hpp"

namespace AER {

class ClassicalRegister;

namespace Operations {

// Operator enum class used for unary classical expression.
enum class UnaryOp { BitNot, LogicNot };

// Operator enum class used for binary classical expression or boolean
// function operation.
enum class BinaryOp {
  BitAnd,
  BitOr,
  BitXor,
  LogicAnd,
  LogicOr,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual
};

bool isBoolBinaryOp(const BinaryOp binary_op);
bool isBoolBinaryOp(const BinaryOp binary_op) {
  return binary_op != BinaryOp::BitAnd && binary_op != BinaryOp::BitOr &&
         binary_op != BinaryOp::BitXor;
}

uint_t truncate(const uint_t val, const size_t width);
uint_t truncate(const uint_t val, const size_t width) {
  size_t shift = 64 - width;
  return (val << shift) >> shift;
}

enum class CExprType { Expr, Var, Value, Cast, Unary, Binary, Nop };

enum class ValueType { Bool, Uint };

class ScalarType {
public:
  ScalarType(const ValueType _type, const size_t width_)
      : type(_type), width(width_) {}

public:
  const ValueType type;
  const size_t width;
};

template <typename T>
inline std::shared_ptr<T> get_wider_type(std::shared_ptr<T> left,
                                         std::shared_ptr<T> right) {
  if (left->width > right->width)
    return left;
  else
    return right;
}

class Uint : public ScalarType {
public:
  Uint(const size_t size) : ScalarType(ValueType::Uint, size) {}
};

class Bool : public ScalarType {
public:
  Bool() : ScalarType(ValueType::Bool, 1) {}
};

class CExpr {
public:
  CExpr(const CExprType _expr_type, const std::shared_ptr<ScalarType> _type)
      : expr_type(_expr_type), type(_type) {}
  virtual bool eval_bool(const std::string &memory) { return false; };
  virtual uint_t eval_uint(const std::string &memory) { return 0ul; };

public:
  const CExprType expr_type;
  const std::shared_ptr<ScalarType> type;
};

class CastExpr : public CExpr {
public:
  CastExpr(std::shared_ptr<ScalarType> _type,
           const std::shared_ptr<CExpr> operand_)
      : CExpr(CExprType::Cast, _type), operand(operand_) {}

  virtual bool eval_bool(const std::string &memory) {
    if (type->type != ValueType::Bool)
      throw std::invalid_argument(
          R"(eval_bool is called for non-bool expression.)");
    if (operand->type->type == ValueType::Bool)
      return operand->eval_bool(memory);
    else if (operand->type->type == ValueType::Uint)
      return operand->eval_uint(memory) == 0ul;
    else
      throw std::invalid_argument(R"(invalid cast: from unknown type.)");
  }

  virtual uint_t eval_uint(const std::string &memory) {
    if (type->type != ValueType::Uint)
      throw std::invalid_argument(
          R"(eval_uint is called for non-uint expression.)");
    if (operand->type->type == ValueType::Bool)
      return operand->eval_bool(memory) ? 1ul : 0ul;
    else if (operand->type->type == ValueType::Uint)
      return truncate(operand->eval_uint(memory), type->width);
    else
      throw std::invalid_argument(R"(invalid cast: from unknown type.)");
  }

public:
  const std::shared_ptr<CExpr> operand;
};

class VarExpr : public CExpr {
public:
  VarExpr(std::shared_ptr<ScalarType> _type,
          const std::vector<uint_t> &_cbit_idxs)
      : CExpr(CExprType::Var, _type), cbit_idxs(_cbit_idxs) {}

  virtual bool eval_bool(const std::string &memory) {
    if (type->type != ValueType::Bool)
      throw std::invalid_argument(
          R"(eval_bool is called for non-bool expression.)");
    return eval_uint_(memory) != 0ul;
  }

  virtual uint_t eval_uint(const std::string &memory) {
    if (type->type != ValueType::Uint)
      throw std::invalid_argument(
          R"(eval_uint is called for non-uint expression.)");
    return eval_uint_(memory);
  }

private:
  uint_t eval_uint_(const std::string &memory) {
    uint_t val = 0ul;
    uint_t shift = 0;
    for (const uint_t cbit_idx : cbit_idxs) {
      if (memory.size() <= cbit_idx)
        throw std::invalid_argument(R"(invalid cbit index.)");
      if (memory[memory.size() - cbit_idx - 1] == '1')
        val |= (1 << shift);
      ++shift;
    }
    return truncate(val, type->width);
  }

public:
  const std::vector<uint_t> cbit_idxs;
};

class ValueExpr : public CExpr {
public:
  ValueExpr(std::shared_ptr<ScalarType> _type)
      : CExpr(CExprType::Value, _type) {}
};

class UintValue : public ValueExpr {
public:
  UintValue(size_t width, const uint_t value_)
      : ValueExpr(std::make_shared<Uint>(width)), value(value_) {}

  virtual bool eval_bool(const std::string &memory) {
    throw std::invalid_argument(
        R"(eval_bool is called for Uint value without cast.)");
  }

  virtual uint_t eval_uint(const std::string &memory) { return value; }

public:
  const uint_t value;
};

class BoolValue : public ValueExpr {
public:
  BoolValue(const bool value_)
      : ValueExpr(std::make_shared<Bool>()), value(value_) {}

  virtual bool eval_bool(const std::string &memory) { return value != 0ul; }

  virtual uint_t eval_uint(const std::string &memory) {
    throw std::invalid_argument(
        R"(eval_uint is called for Bool value without cast.)");
  }

public:
  const bool value;
};

class UnaryExpr : public CExpr {
public:
  UnaryExpr(const UnaryOp op_, const std::shared_ptr<CExpr> operand_)
      : CExpr(CExprType::Unary, operand_->type), op(op_), operand(operand_) {
    if (op == UnaryOp::LogicNot && operand_->type->type != ValueType::Bool)
      throw std::invalid_argument(
          R"(LogicNot unary expression must has Bool expression as its operand.)");

    if (op == UnaryOp::BitNot && operand_->type->type != ValueType::Uint)
      throw std::invalid_argument(
          R"(BitNot unary expression must has Uint expression as its operand.)");
  }

  virtual bool eval_bool(const std::string &memory) {
    if (op == UnaryOp::BitNot)
      throw std::invalid_argument(
          R"(eval_bool is called for BitNot unary expression.)");
    else // LogicNot
      return !operand->eval_bool(memory);
  }

  virtual uint_t eval_uint(const std::string &memory) {
    if (op == UnaryOp::BitNot)
      return truncate(~operand->eval_uint(memory), type->width);
    else // LogicNot
      throw std::invalid_argument(
          R"(eval_uint is called for LogicNot unary expression.)");
  }

public:
  const UnaryOp op;
  const std::shared_ptr<CExpr> operand;
};

class BinaryExpr : public CExpr {
public:
  BinaryExpr(const BinaryOp op_, const std::shared_ptr<CExpr> left_,
             const std::shared_ptr<CExpr> right_)
      : CExpr(CExprType::Binary,
              isBoolBinaryOp(op_) ? std::make_shared<Bool>()
                                  : get_wider_type(left_->type, right_->type)),
        op(op_), left(left_), right(right_) {

    if (left->type->type != right->type->type)
      throw std::invalid_argument(
          R"(binary expression does not support different types in child expressions.)");

    switch (op) {
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      break;
    case BinaryOp::LogicAnd:
    case BinaryOp::LogicOr:
      if (left->type->type != ValueType::Bool)
        throw std::invalid_argument(
            R"(logic operation allows only for bool expressions.)");
      break;
    case BinaryOp::Equal:
    case BinaryOp::NotEqual:
      break;
    case BinaryOp::Less:
    case BinaryOp::LessEqual:
    case BinaryOp::Greater:
    case BinaryOp::GreaterEqual:
      if (left->type->type != ValueType::Uint)
        throw std::invalid_argument(
            R"(comparison operation allows only for uint expressions.)");
      break;
    default:
      throw std::invalid_argument(R"(must not reach here.)");
    }
  }

  virtual bool eval_bool(const std::string &memory) {
    switch (op) {
    case BinaryOp::BitAnd:
      if (left->type->type == ValueType::Uint)
        return eval_uint(memory) != 0;
      else
        return left->eval_bool(memory) && right->eval_bool(memory);
    case BinaryOp::BitOr:
      if (left->type->type == ValueType::Uint)
        return eval_uint(memory) != 0;
      else
        return left->eval_bool(memory) || right->eval_bool(memory);
    case BinaryOp::BitXor:
      if (left->type->type == ValueType::Uint)
        return eval_uint(memory) != 0;
      else
        return left->eval_bool(memory) ^ right->eval_bool(memory);
    case BinaryOp::LogicAnd:
      return left->eval_bool(memory) && right->eval_bool(memory);
    case BinaryOp::LogicOr:
      return left->eval_bool(memory) || right->eval_bool(memory);
    case BinaryOp::Equal:
      if (left->type->type == ValueType::Bool)
        return left->eval_bool(memory) == right->eval_bool(memory);
      else
        return left->eval_uint(memory) == right->eval_uint(memory);
    case BinaryOp::NotEqual:
      if (left->type->type == ValueType::Bool)
        return left->eval_bool(memory) != right->eval_bool(memory);
      else
        return left->eval_uint(memory) != right->eval_uint(memory);
    case BinaryOp::Less:
      return left->eval_uint(memory) < right->eval_uint(memory);
    case BinaryOp::LessEqual:
      return left->eval_uint(memory) <= right->eval_uint(memory);
    case BinaryOp::Greater:
      return left->eval_uint(memory) > right->eval_uint(memory);
    case BinaryOp::GreaterEqual:
      return left->eval_uint(memory) >= right->eval_uint(memory);
    default:
      throw std::invalid_argument(R"(must not reach here.)");
    }
  }

  virtual uint_t eval_uint(const std::string &memory) {
    switch (op) {
    case BinaryOp::BitAnd:
      return left->eval_uint(memory) & right->eval_uint(memory);
    case BinaryOp::BitOr:
      return left->eval_uint(memory) | right->eval_uint(memory);
    case BinaryOp::BitXor:
      return left->eval_uint(memory) ^ right->eval_uint(memory);
    case BinaryOp::LogicAnd:
    case BinaryOp::LogicOr:
    case BinaryOp::Equal:
    case BinaryOp::NotEqual:
    case BinaryOp::Less:
    case BinaryOp::LessEqual:
    case BinaryOp::Greater:
    case BinaryOp::GreaterEqual:
      throw std::invalid_argument(
          R"(eval_uint is called for binary expression that returns bool.)");
    default:
      throw std::invalid_argument(R"(must not reach here.)");
    }
  }

public:
  const BinaryOp op;
  const std::shared_ptr<CExpr> left;
  const std::shared_ptr<CExpr> right;
};

// Enum class for operation types
enum class OpType {
  gate,
  measure,
  reset,
  bfunc,
  barrier,
  qerror_loc,
  matrix,
  diagonal_matrix,
  multiplexer,
  initialize,
  sim_op,
  nop,
  // Noise instructions
  kraus,
  superop,
  roerror,
  noise_switch,
  sample_noise,
  // Save instructions
  save_state,
  save_expval,
  save_expval_var,
  save_statevec,
  save_statevec_dict,
  save_densmat,
  save_probs,
  save_probs_ket,
  save_amps,
  save_amps_sq,
  save_stabilizer,
  save_clifford,
  save_unitary,
  save_mps,
  save_superop,
  // Set instructions
  set_statevec,
  set_densmat,
  set_unitary,
  set_superop,
  set_stabilizer,
  set_mps,
  // Control Flow
  jump,
  mark,
  unary_expr,
  binary_expr
};

enum class DataSubType {
  single,
  c_single,
  list,
  c_list,
  accum,
  c_accum,
  average,
  c_average
};

static const std::unordered_set<OpType> SAVE_TYPES = {
    OpType::save_state,    OpType::save_expval,        OpType::save_expval_var,
    OpType::save_statevec, OpType::save_statevec_dict, OpType::save_densmat,
    OpType::save_probs,    OpType::save_probs_ket,     OpType::save_amps,
    OpType::save_amps_sq,  OpType::save_stabilizer,    OpType::save_clifford,
    OpType::save_unitary,  OpType::save_mps,           OpType::save_superop};

inline std::ostream &operator<<(std::ostream &stream, const OpType &type) {
  switch (type) {
  case OpType::gate:
    stream << "gate";
    break;
  case OpType::measure:
    stream << "measure";
    break;
  case OpType::reset:
    stream << "reset";
    break;
  case OpType::bfunc:
    stream << "bfunc";
    break;
  case OpType::barrier:
    stream << "barrier";
    break;
  case OpType::save_state:
    stream << "save_state";
    break;
  case OpType::save_expval:
    stream << "save_expval";
    break;
  case OpType::save_expval_var:
    stream << "save_expval_var";
  case OpType::save_statevec:
    stream << "save_statevector";
    break;
  case OpType::save_statevec_dict:
    stream << "save_statevector_dict";
    break;
  case OpType::save_mps:
    stream << "save_matrix_product_state";
    break;
  case OpType::save_densmat:
    stream << "save_density_matrix";
    break;
  case OpType::save_probs:
    stream << "save_probabilities";
    break;
  case OpType::save_probs_ket:
    stream << "save_probabilities_dict";
    break;
  case OpType::save_amps:
    stream << "save_amplitudes";
    break;
  case OpType::save_amps_sq:
    stream << "save_amplitudes_sq";
    break;
  case OpType::save_stabilizer:
    stream << "save_stabilizer";
    break;
  case OpType::save_clifford:
    stream << "save_clifford";
    break;
  case OpType::save_unitary:
    stream << "save_unitary";
    break;
  case OpType::save_superop:
    stream << "save_superop";
    break;
  case OpType::set_statevec:
    stream << "set_statevector";
    break;
  case OpType::set_densmat:
    stream << "set_density_matrix";
    break;
  case OpType::set_unitary:
    stream << "set_unitary";
    break;
  case OpType::set_superop:
    stream << "set_superop";
    break;
  case OpType::set_stabilizer:
    stream << "set_stabilizer";
    break;
  case OpType::set_mps:
    stream << "set_matrix_product_state";
    break;
  case OpType::matrix:
    stream << "unitary";
    break;
  case OpType::diagonal_matrix:
    stream << "diagonal";
    break;
  case OpType::multiplexer:
    stream << "multiplexer";
    break;
  case OpType::kraus:
    stream << "kraus";
    break;
  case OpType::superop:
    stream << "superop";
    break;
  case OpType::roerror:
    stream << "roerror";
    break;
  case OpType::qerror_loc:
    stream << "qerror_loc";
    break;
  case OpType::sample_noise:
    stream << "sample_noise";
    break;
  case OpType::noise_switch:
    stream << "noise_switch";
    break;
  case OpType::initialize:
    stream << "initialize";
    break;
  case OpType::sim_op:
    stream << "sim_op";
    break;
  case OpType::nop:
    stream << "nop";
    break;
  case OpType::mark:
    stream << "mark";
    break;
  case OpType::jump:
    stream << "jump";
    break;
  case OpType::unary_expr:
    stream << "unary_expr";
    break;
  case OpType::binary_expr:
    stream << "binary_expr";
    break;
  default:
    stream << "unknown";
  }
  return stream;
}

inline std::ostream &operator<<(std::ostream &stream,
                                const DataSubType &subtype) {
  switch (subtype) {
  case DataSubType::single:
    stream << "single";
    break;
  case DataSubType::c_single:
    stream << "c_single";
    break;
  case DataSubType::list:
    stream << "list";
    break;
  case DataSubType::c_list:
    stream << "c_list";
    break;
  case DataSubType::accum:
    stream << "accum";
    break;
  case DataSubType::c_accum:
    stream << "c_accum";
    break;
  case DataSubType::average:
    stream << "average";
    break;
  case DataSubType::c_average:
    stream << "c_average";
    break;
  default:
    stream << "unknown";
  }
  return stream;
}

//------------------------------------------------------------------------------
// Op Class
//------------------------------------------------------------------------------

struct Op {
  // General Operations
  OpType type;                    // operation type identifier
  std::string name;               // operation name
  reg_t qubits;                   //  qubits operation acts on
  std::vector<reg_t> regs;        //  list of qubits for matrixes
  std::vector<complex_t> params;  // real or complex params for gates
  std::vector<uint_t> int_params; // integer parameters
  std::vector<std::string>
      string_params; // used for label, control-flow, and boolean functions

  // Conditional Operations
  bool conditional = false; // is gate conditional gate
  uint_t conditional_reg; // (opt) the (single) register location to look up for
                          // conditional
  BinaryOp binary_op;     // (opt) boolean function relation
  std::shared_ptr<CExpr> expr; // (opt) classical expression

  // Measurement
  reg_t memory;    // (opt) register operation it acts on (measure)
  reg_t registers; // (opt) register locations it acts on (measure, conditional)

  // Mat and Kraus
  std::vector<cmatrix_t> mats;

  // Readout error
  std::vector<rvector_t> probs;

  // Expvals
  std::vector<std::tuple<std::string, double, double>> expval_params;

  // Set states
  Clifford::Clifford clifford;
  mps_container_t mps;

  // Save
  DataSubType save_type = DataSubType::single;

  // runtime parameter bind
  bool has_bind_params = false;
};

inline std::ostream &operator<<(std::ostream &s, const Op &op) {
  s << op.name << "[";
  bool first = true;
  for (size_t qubit : op.qubits) {
    if (!first)
      s << ",";
    s << qubit;
    first = false;
  }
  s << "],[";
  first = true;
  for (reg_t reg : op.regs) {
    if (!first)
      s << ",";
    s << "[";
    bool first0 = true;
    for (size_t qubit : reg) {
      if (!first0)
        s << ",";
      s << qubit;
      first0 = false;
    }
    s << "]";
    first = false;
  }
  s << "]";
  return s;
}

//------------------------------------------------------------------------------
// Error Checking
//------------------------------------------------------------------------------

// Raise an exception if name string is empty
inline void check_empty_name(const Op &op) {
  if (op.name.empty())
    throw std::invalid_argument(
        R"(Invalid qobj instruction ("name" is empty).)");
}

// Raise an exception if qubits list is empty
inline void check_empty_qubits(const Op &op) {
  if (op.qubits.empty())
    throw std::invalid_argument(R"(Invalid operation ")" + op.name +
                                R"(" ("qubits" is empty).)");
}

// Raise an exception if params is empty
inline void check_empty_params(const Op &op) {
  if (op.params.empty())
    throw std::invalid_argument(R"(Invalid operation ")" + op.name +
                                R"(" ("params" is empty).)");
}

// Raise an exception if qubits is more than expected
inline void check_length_qubits(const Op &op, const size_t size) {
  if (op.qubits.size() < size)
    throw std::invalid_argument(R"(Invalid operation ")" + op.name +
                                R"(" ("qubits" is incorrect length).)");
}

// Raise an exception if params is empty
inline void check_length_params(const Op &op, const size_t size) {
  if (op.params.size() < size)
    throw std::invalid_argument(R"(Invalid operation ")" + op.name +
                                R"(" ("params" is incorrect length).)");
}

// Raise an exception if qubits list contains duplications
inline void check_duplicate_qubits(const Op &op) {
  auto cpy = op.qubits;
  std::unique(cpy.begin(), cpy.end());
  if (cpy != op.qubits)
    throw std::invalid_argument(R"(Invalid operation ")" + op.name +
                                R"(" ("qubits" are not unique).)");
}

inline void check_gate_params(const Op &op) {
  const stringmap_t<std::tuple<int_t, int_t>> param_tables(
      {{"u1", {1, 1}},       {"u2", {1, 2}},     {"u3", {1, 3}},
       {"u", {1, 3}},        {"U", {1, 3}},      {"CX", {2, 0}},
       {"cx", {2, 0}},       {"cz", {2, 0}},     {"cy", {2, 0}},
       {"cp", {2, 1}},       {"cu1", {2, 1}},    {"cu2", {2, 2}},
       {"cu3", {2, 3}},      {"swap", {2, 0}},   {"id", {0, 0}},
       {"p", {1, 1}},        {"x", {1, 0}},      {"y", {1, 0}},
       {"z", {1, 0}},        {"h", {1, 0}},      {"s", {1, 0}},
       {"sdg", {1, 0}},      {"t", {1, 0}},      {"tdg", {1, 0}},
       {"r", {1, 2}},        {"rx", {1, 1}},     {"ry", {1, 1}},
       {"rz", {1, 1}},       {"rxx", {2, 1}},    {"ryy", {2, 1}},
       {"rzz", {2, 1}},      {"rzx", {2, 1}},    {"ccx", {3, 0}},
       {"cswap", {3, 0}},    {"mcx", {1, 0}},    {"mcy", {1, 0}},
       {"mcz", {1, 0}},      {"mcu1", {1, 1}},   {"mcu2", {1, 2}},
       {"mcu3", {1, 3}},     {"mcswap", {2, 0}}, {"mcphase", {1, 1}},
       {"mcr", {1, 1}},      {"mcrx", {1, 1}},   {"mcry", {1, 1}},
       {"mcrz", {1, 1}},     {"sx", {1, 0}},     {"sxdg", {1, 0}},
       {"csx", {2, 0}},      {"mcsx", {1, 0}},   {"csxdg", {2, 0}},
       {"mcsxdg", {1, 0}},   {"delay", {1, 0}},  {"pauli", {1, 0}},
       {"mcx_gray", {1, 0}}, {"cu", {2, 4}},     {"mcu", {1, 4}},
       {"mcp", {1, 1}},      {"ecr", {2, 0}}});

  auto it = param_tables.find(op.name);
  if (it == param_tables.end()) {
    std::stringstream msg;
    msg << "Invalid gate name :\"" << op.name << "\"." << std::endl;
    throw std::invalid_argument(msg.str());
  } else {
    check_length_qubits(op, std::get<0>(it->second));
    check_length_params(op, std::get<1>(it->second));
  }
}

//------------------------------------------------------------------------------
// Generator functions
//------------------------------------------------------------------------------

inline Op make_initialize(const reg_t &qubits,
                          const std::vector<complex_t> &init_data) {
  Op op;
  op.type = OpType::initialize;
  op.name = "initialize";
  op.qubits = qubits;
  op.params = init_data;
  return op;
}

inline Op make_unitary(const reg_t &qubits, const cmatrix_t &mat,
                       const int_t conditional = -1,
                       const std::shared_ptr<CExpr> expr = nullptr,
                       std::string label = "") {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  op.qubits = qubits;
  op.mats = {mat};
  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }
  op.expr = expr;
  if (label != "")
    op.string_params = {label};
  return op;
}

inline Op make_unitary(const reg_t &qubits, cmatrix_t &&mat,
                       std::string label = "") {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  op.qubits = qubits;
  op.mats.resize(1);
  op.mats[0] = std::move(mat);
  if (label != "")
    op.string_params = {label};
  return op;
}

inline Op make_diagonal(const reg_t &qubits, const cvector_t &vec,
                        const int_t conditional = -1,
                        const std::string label = "") {
  Op op;
  op.type = OpType::diagonal_matrix;
  op.name = "diagonal";
  op.qubits = qubits;
  op.params = vec;

  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }

  if (label != "")
    op.string_params = {label};

  return op;
}

inline Op make_diagonal(const reg_t &qubits, cvector_t &&vec,
                        const int_t conditional = -1,
                        const std::string label = "") {
  Op op;
  op.type = OpType::diagonal_matrix;
  op.name = "diagonal";
  op.qubits = qubits;
  op.params = std::move(vec);

  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }

  if (label != "")
    op.string_params = {label};

  return op;
}

inline Op make_superop(const reg_t &qubits, const cmatrix_t &mat,
                       const int_t conditional = -1,
                       const std::shared_ptr<CExpr> expr = nullptr) {
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  op.qubits = qubits;
  op.mats = {mat};
  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }
  op.expr = expr;
  return op;
}

inline Op make_superop(const reg_t &qubits, cmatrix_t &&mat) {
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  op.qubits = qubits;
  op.mats.resize(1);
  op.mats[0] = std::move(mat);
  return op;
}

inline Op make_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &mats,
                     const int_t conditional = -1,
                     const std::shared_ptr<CExpr> expr = nullptr) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  op.qubits = qubits;
  op.mats = mats;
  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }
  op.expr = expr;
  return op;
}

inline Op make_kraus(const reg_t &qubits, std::vector<cmatrix_t> &&mats) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  op.qubits = qubits;
  op.mats = std::move(mats);
  return op;
}

inline Op make_roerror(const reg_t &memory,
                       const std::vector<rvector_t> &probs) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  op.memory = memory;
  op.probs = probs;
  return op;
}

inline Op make_roerror(const reg_t &memory, std::vector<rvector_t> &&probs) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  op.memory = memory;
  op.probs = std::move(probs);
  return op;
}

inline Op make_bfunc(const std::string &mask, const std::string &val,
                     const std::string &relation, const uint_t regidx) {
  Op op;
  op.type = OpType::bfunc;
  op.name = "bfunc";

  op.string_params.resize(2);
  op.string_params[0] = mask;
  op.string_params[1] = val;

  // Load single register
  op.registers.push_back(regidx);

  // Format hex strings
  Utils::format_hex_inplace(op.string_params[0]);
  Utils::format_hex_inplace(op.string_params[1]);

  const stringmap_t<BinaryOp> comp_table({
      {"==", BinaryOp::Equal},
      {"!=", BinaryOp::NotEqual},
      {"<", BinaryOp::Less},
      {"<=", BinaryOp::LessEqual},
      {">", BinaryOp::Greater},
      {">=", BinaryOp::GreaterEqual},
  });

  auto it = comp_table.find(relation);
  if (it == comp_table.end()) {
    std::stringstream msg;
    msg << "Invalid bfunc relation string :\"" << it->first << "\"."
        << std::endl;
    throw std::invalid_argument(msg.str());
  } else {
    op.binary_op = it->second;
  }

  return op;
}

Op make_gate(const std::string &name, const reg_t &qubits,
             const std::vector<complex_t> &params,
             const std::vector<std::string> &string_params,
             const int_t conditional, const std::shared_ptr<CExpr> expr,
             const std::string &label);
Op make_gate(const std::string &name, const reg_t &qubits,
             const std::vector<complex_t> &params,
             const std::vector<std::string> &string_params,
             const int_t conditional, const std::shared_ptr<CExpr> expr,
             const std::string &label) {
  Op op;
  op.type = OpType::gate;
  op.name = name;
  op.qubits = qubits;
  op.params = params;

  if (string_params.size() > 0)
    op.string_params = string_params;
  else if (label != "")
    op.string_params = {label};
  else
    op.string_params = {op.name};
  op.expr = expr;

  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }

  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u1(uint_t qubit, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u1";
  op.qubits = {qubit};
  op.params = {lam};
  op.string_params = {op.name};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u2(uint_t qubit, T phi, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u2";
  op.qubits = {qubit};
  op.params = {phi, lam};
  op.string_params = {op.name};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u3(uint_t qubit, T theta, T phi, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u3";
  op.qubits = {qubit};
  op.params = {theta, phi, lam};
  op.string_params = {op.name};
  return op;
}

inline Op make_reset(const reg_t &qubits, const int_t conditional) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  op.qubits = qubits;

  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }

  return op;
}

inline Op make_multiplexer(const reg_t &qubits,
                           const std::vector<cmatrix_t> &mats,
                           const int_t conditional = -1,
                           const std::shared_ptr<CExpr> expr = nullptr,
                           std::string label = "") {

  // Check matrices are N-qubit
  auto dim = mats[0].GetRows();
  auto num_targets = static_cast<uint_t>(std::log2(dim));
  if (1ULL << num_targets != dim) {
    throw std::invalid_argument("invalid multiplexer matrix dimension.");
  }
  // Check number of matrix compents is power of 2.
  size_t num_mats = mats.size();
  auto num_controls = static_cast<uint_t>(std::log2(num_mats));
  if (1ULL << num_controls != num_mats) {
    throw std::invalid_argument("invalid number of multiplexer matrices.");
  }
  if (num_controls == 0) { // mats.size() must be 1
    return make_unitary(qubits, mats[0]);
  }
  // Check number of targets and controls matches qubits
  if (num_controls + num_targets != qubits.size()) {
    throw std::invalid_argument("multiplexer qubits don't match parameters.");
  }
  // Check each matrix component is unitary and same size
  for (const auto &mat : mats) {
    if (!Utils::is_unitary(mat, 1e-7))
      throw std::invalid_argument("multiplexer matrix is not unitary.");
    if (mat.GetRows() != dim) {
      throw std::invalid_argument("multiplexer matrices are different size.");
    }
  }
  // Get lists of controls and targets
  reg_t controls(num_controls), targets(num_targets);
  std::copy_n(qubits.begin(), num_targets, targets.begin());
  std::copy_n(qubits.begin() + num_targets, num_controls, controls.begin());

  // Construct the Op
  Op op;
  op.type = OpType::multiplexer;
  op.name = "multiplexer";
  op.qubits = qubits;
  op.mats = mats;
  op.regs = std::vector<reg_t>({controls, targets});
  if (label != "")
    op.string_params = {label};

  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }
  op.expr = expr;

  // Validate qubits are unique.
  check_empty_qubits(op);
  check_duplicate_qubits(op);

  return op;
}

inline Op make_save_state(const reg_t &qubits, const std::string &name,
                          const std::string &snapshot_type,
                          const std::string &label) {
  Op op;
  op.name = name;

  // Get subtype
  static const std::unordered_map<std::string, OpType> types{
      {"save_state", OpType::save_state},
      {"save_statevector", OpType::save_statevec},
      {"save_statevector_dict", OpType::save_statevec_dict},
      {"save_amplitudes", OpType::save_amps},
      {"save_amplitudes_sq", OpType::save_amps_sq},
      {"save_clifford", OpType::save_clifford},
      {"save_probabilities", OpType::save_probs},
      {"save_probabilities_dict", OpType::save_probs_ket},
      {"save_matrix_product_state", OpType::save_mps},
      {"save_unitary", OpType::save_unitary},
      {"save_superop", OpType::save_superop},
      {"save_density_matrix", OpType::save_densmat},
      {"save_stabilizer", OpType::save_stabilizer},
      {"save_expval", OpType::save_expval},
      {"save_expval_var", OpType::save_expval_var}};

  auto type_it = types.find(name);
  if (type_it == types.end()) {
    throw std::runtime_error("Invalid data type \"" + name +
                             "\" in save data instruction.");
  }
  op.type = type_it->second;

  // Get subtype
  static const std::unordered_map<std::string, DataSubType> subtypes{
      {"single", DataSubType::single},   {"c_single", DataSubType::c_single},
      {"average", DataSubType::average}, {"c_average", DataSubType::c_average},
      {"list", DataSubType::list},       {"c_list", DataSubType::c_list},
      {"accum", DataSubType::accum},     {"c_accum", DataSubType::c_accum},
  };

  auto subtype_it = subtypes.find(snapshot_type);
  if (subtype_it == subtypes.end()) {
    throw std::runtime_error("Invalid data subtype \"" + snapshot_type +
                             "\" in save data instruction.");
  }
  op.save_type = subtype_it->second;

  op.string_params.emplace_back(label);

  op.qubits = qubits;

  return op;
}

inline Op make_save_amplitudes(const reg_t &qubits, const std::string &name,
                               const std::vector<uint_t> &base_type,
                               const std::string &snapshot_type,
                               const std::string &label) {
  auto op = make_save_state(qubits, name, snapshot_type, label);
  op.int_params = base_type;
  return op;
}

inline Op make_save_expval(const reg_t &qubits, const std::string &name,
                           const std::vector<std::string> pauli_strings,
                           const std::vector<double> coeff_reals,
                           const std::vector<double> coeff_imags,
                           const std::string &snapshot_type,
                           const std::string &label) {

  assert(pauli_strings.size() == coeff_reals.size());
  assert(pauli_strings.size() == coeff_imags.size());

  auto op = make_save_state(qubits, name, snapshot_type, label);

  for (uint_t i = 0; i < pauli_strings.size(); ++i)
    op.expval_params.emplace_back(pauli_strings[i], coeff_reals[i],
                                  coeff_imags[i]);

  if (op.expval_params.empty()) {
    std::string pauli(op.qubits.size(), 'I');
    op.expval_params.emplace_back(pauli, 0., 0.);
  }
  return op;
}

template <typename inputdata_t>
inline Op make_set_vector(const reg_t &qubits, const std::string &name,
                          const inputdata_t &params) {
  Op op;
  // Get type
  static const std::unordered_map<std::string, OpType> types{
      {"set_statevector", OpType::set_statevec},
  };
  auto type_it = types.find(name);
  if (type_it == types.end()) {
    throw std::runtime_error("Invalid data type \"" + name +
                             "\" in set data instruction.");
  }
  op.type = type_it->second;
  op.name = name;
  op.qubits = qubits;
  op.params =
      Parser<inputdata_t>::template get_list_elem<std::vector<complex_t>>(
          params, 0);
  return op;
}

template <typename inputdata_t>
inline Op make_set_matrix(const reg_t &qubits, const std::string &name,
                          const inputdata_t &params) {
  Op op;
  // Get type
  static const std::unordered_map<std::string, OpType> types{
      {"set_density_matrix", OpType::set_densmat},
      {"set_unitary", OpType::set_unitary},
      {"set_superop", OpType::set_superop}};
  auto type_it = types.find(name);
  if (type_it == types.end()) {
    throw std::runtime_error("Invalid data type \"" + name +
                             "\" in set data instruction.");
  }
  op.type = type_it->second;
  op.name = name;
  op.qubits = qubits;
  op.mats.push_back(
      Parser<inputdata_t>::template get_list_elem<cmatrix_t>(params, 0));
  return op;
}

template <typename inputdata_t>
inline Op make_set_mps(const reg_t &qubits, const std::string &name,
                       const inputdata_t &params) {
  Op op;
  op.type = OpType::set_mps;
  op.name = name;
  op.qubits = qubits;
  op.mps =
      Parser<inputdata_t>::template get_list_elem<mps_container_t>(params, 0);
  return op;
}

template <typename inputdata_t>
inline Op make_set_clifford(const reg_t &qubits, const std::string &name,
                            const inputdata_t &params) {
  Op op;
  op.type = OpType::set_stabilizer;
  op.name = name;
  op.qubits = qubits;
  op.clifford = Parser<inputdata_t>::template get_list_elem<Clifford::Clifford>(
      params, 0);
  return op;
}

inline Op make_jump(const reg_t &qubits, const std::vector<std::string> &params,
                    const int_t conditional,
                    const std::shared_ptr<CExpr> expr = nullptr) {
  Op op;
  op.type = OpType::jump;
  op.name = "jump";
  op.qubits = qubits;
  op.string_params = params;
  if (op.string_params.empty())
    throw std::invalid_argument(
        std::string("Invalid jump (\"params\" field missing)."));

  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }
  op.expr = expr;

  return op;
}

inline Op make_mark(const reg_t &qubits,
                    const std::vector<std::string> &params) {
  Op op;
  op.type = OpType::mark;
  op.name = "mark";
  op.qubits = qubits;
  op.string_params = params;
  if (op.string_params.empty())
    throw std::invalid_argument(
        std::string("Invalid mark (\"params\" field missing)."));

  return op;
}

inline Op make_barrier(const reg_t &qubits) {
  Op op;
  op.type = OpType::barrier;
  op.name = "barrier";
  op.qubits = qubits;
  return op;
}

inline Op make_measure(const reg_t &qubits, const reg_t &memory,
                       const reg_t &registers) {
  Op op;
  op.type = OpType::measure;
  op.name = "measure";
  op.qubits = qubits;
  op.memory = memory;
  op.registers = registers;
  return op;
}

inline Op make_qerror_loc(const reg_t &qubits, const std::string &label,
                          const int_t conditional = -1,
                          const std::shared_ptr<CExpr> expr = nullptr) {
  Op op;
  op.type = OpType::qerror_loc;
  op.name = label;
  op.qubits = qubits;
  if (conditional >= 0) {
    op.conditional = true;
    op.conditional_reg = conditional;
  }
  op.expr = expr;
  return op;
}

// make new op by parameter binding
inline Op bind_parameter(const Op &src, const uint_t iparam,
                         const uint_t num_params) {
  Op op;
  op.type = src.type;
  op.name = src.name;
  op.qubits = src.qubits;
  op.conditional = src.conditional;
  op.conditional_reg = src.conditional_reg;

  if (src.params.size() > 0) {
    uint_t stride = src.params.size() / num_params;
    op.params.resize(stride);
    for (uint_t i = 0; i < stride; i++)
      op.params[i] = src.params[iparam * stride + i];
  } else if (src.mats.size() > 0) {
    uint_t stride = src.mats.size() / num_params;
    op.mats.resize(stride);
    for (uint_t i = 0; i < stride; i++)
      op.mats[i] = src.mats[iparam * stride + i];
  }
  return op;
}

//------------------------------------------------------------------------------
// JSON conversion
//------------------------------------------------------------------------------

// Main deserialization functions
template <typename inputdata_t>
Op input_to_op(const inputdata_t &input); // Partial TODO
json_t op_to_json(const Op &op);          // Partial TODO

inline void from_json(const json_t &js, Op &op) { op = input_to_op(js); }

inline void to_json(json_t &js, const Op &op) { js = op_to_json(op); }

void to_json(json_t &js, const DataSubType &type);

// Standard operations
template <typename inputdata_t>
Op input_to_op_gate(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_barrier(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_measure(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_reset(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_bfunc(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_initialize(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_pauli(const inputdata_t &input);

// Set state
template <typename inputdata_t>
Op input_to_op_set_vector(const inputdata_t &input, OpType op_type);

template <typename inputdata_t>
Op input_to_op_set_matrix(const inputdata_t &input, OpType op_type);

template <typename inputdata_t>
Op input_to_op_set_clifford(const inputdata_t &input, OpType op_type);

template <typename inputdata_t>
Op input_to_op_set_mps(const inputdata_t &input, OpType op_type);

// Save data
template <typename inputdata_t>
Op input_to_op_save_default(const inputdata_t &input, OpType op_type);
template <typename inputdata_t>
Op input_to_op_save_expval(const inputdata_t &input, bool variance);
template <typename inputdata_t>
Op input_to_op_save_amps(const inputdata_t &input, bool squared);

// Control-Flow
template <typename inputdata_t>
Op input_to_op_jump(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_mark(const inputdata_t &input);

// Matrices
template <typename inputdata_t>
Op input_to_op_unitary(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_diagonal(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_superop(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_multiplexer(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_kraus(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_noise_switch(const inputdata_t &input);
template <typename inputdata_t>
Op input_to_op_qerror_loc(const inputdata_t &input);

// Classical bits
template <typename inputdata_t>
Op input_to_op_roerror(const inputdata_t &input);

// Optional instruction parameters
enum class Allowed { Yes, No };

template <typename inputdata_t>
void add_conditional(const Allowed val, Op &op, const inputdata_t &input);

//------------------------------------------------------------------------------
// Implementation: JSON deserialization
//------------------------------------------------------------------------------

// TODO: convert if-else to switch
template <typename inputdata_t>
Op input_to_op(const inputdata_t &input) {
  // load operation identifier
  std::string name;
  Parser<inputdata_t>::get_value(name, "name", input);
  // Barrier
  if (name == "barrier")
    return input_to_op_barrier(input);
  // Measure & Reset
  if (name == "measure")
    return input_to_op_measure(input);
  if (name == "reset")
    return input_to_op_reset(input);
  if (name == "initialize")
    return input_to_op_initialize(input);
  // Arbitrary matrix gates
  if (name == "unitary")
    return input_to_op_unitary(input);
  if (name == "diagonal" || name == "diag")
    return input_to_op_diagonal(input);
  if (name == "superop")
    return input_to_op_superop(input);
  // Save
  if (name == "save_state")
    return input_to_op_save_default(input, OpType::save_state);
  if (name == "save_expval")
    return input_to_op_save_expval(input, false);
  if (name == "save_expval_var")
    return input_to_op_save_expval(input, true);
  if (name == "save_statevector")
    return input_to_op_save_default(input, OpType::save_statevec);
  if (name == "save_statevector_dict")
    return input_to_op_save_default(input, OpType::save_statevec_dict);
  if (name == "save_stabilizer")
    return input_to_op_save_default(input, OpType::save_stabilizer);
  if (name == "save_clifford")
    return input_to_op_save_default(input, OpType::save_clifford);
  if (name == "save_unitary")
    return input_to_op_save_default(input, OpType::save_unitary);
  if (name == "save_superop")
    return input_to_op_save_default(input, OpType::save_superop);
  if (name == "save_density_matrix")
    return input_to_op_save_default(input, OpType::save_densmat);
  if (name == "save_probabilities")
    return input_to_op_save_default(input, OpType::save_probs);
  if (name == "save_matrix_product_state")
    return input_to_op_save_default(input, OpType::save_mps);
  if (name == "save_probabilities_dict")
    return input_to_op_save_default(input, OpType::save_probs_ket);
  if (name == "save_amplitudes")
    return input_to_op_save_amps(input, false);
  if (name == "save_amplitudes_sq")
    return input_to_op_save_amps(input, true);
  // Set
  if (name == "set_statevector")
    return input_to_op_set_vector(input, OpType::set_statevec);
  if (name == "set_density_matrix")
    return input_to_op_set_matrix(input, OpType::set_densmat);
  if (name == "set_unitary")
    return input_to_op_set_matrix(input, OpType::set_unitary);
  if (name == "set_superop")
    return input_to_op_set_matrix(input, OpType::set_superop);
  if (name == "set_stabilizer")
    return input_to_op_set_clifford(input, OpType::set_stabilizer);
  if (name == "set_matrix_product_state")
    return input_to_op_set_mps(input, OpType::set_mps);

  // Bit functions
  if (name == "bfunc")
    return input_to_op_bfunc(input);
  // Noise functions
  if (name == "noise_switch")
    return input_to_op_noise_switch(input);
  if (name == "qerror_loc")
    return input_to_op_qerror_loc(input);
  if (name == "multiplexer")
    return input_to_op_multiplexer(input);
  if (name == "kraus")
    return input_to_op_kraus(input);
  if (name == "roerror")
    return input_to_op_roerror(input);
  if (name == "pauli")
    return input_to_op_pauli(input);

  // Control-flow
  if (name == "jump")
    return input_to_op_jump(input);
  if (name == "mark")
    return input_to_op_mark(input);
  // Default assume gate
  return input_to_op_gate(input);
}

json_t op_to_json(const Op &op) {
  json_t ret;
  ret["name"] = op.name;
  if (!op.qubits.empty())
    ret["qubits"] = op.qubits;
  if (!op.regs.empty())
    ret["regs"] = op.regs;
  if (!op.params.empty())
    ret["params"] = op.params;
  else if (!op.int_params.empty())
    ret["params"] = op.int_params;
  if (op.conditional)
    ret["conditional"] = op.conditional_reg;
  if (!op.memory.empty())
    ret["memory"] = op.memory;
  if (!op.registers.empty())
    ret["register"] = op.registers;
  if (!op.mats.empty())
    ret["mats"] = op.mats;
  return ret;
}

void to_json(json_t &js, const OpType &type);
void to_json(json_t &js, const OpType &type) {
  std::stringstream ss;
  ss << type;
  js = ss.str();
}

void to_json(json_t &js, const DataSubType &subtype) {
  std::stringstream ss;
  ss << subtype;
  js = ss.str();
}

//------------------------------------------------------------------------------
// Implementation: Gates, measure, reset deserialization
//------------------------------------------------------------------------------

template <typename inputdata_t>
void add_conditional(const Allowed allowed, Op &op, const inputdata_t &input) {
  // Check conditional
  if (Parser<inputdata_t>::check_key("conditional", input)) {
    // If instruction isn't allow to be conditional throw an exception
    if (allowed == Allowed::No) {
      throw std::invalid_argument("Invalid instruction: \"" + op.name +
                                  "\" cannot be conditional.");
    }
    // If instruction is allowed to be conditional add parameters
    Parser<inputdata_t>::get_value(op.conditional_reg, "conditional", input);
    op.conditional = true;
  }
}

template <typename inputdata_t>
Op input_to_op_gate(const inputdata_t &input) {
  Op op;
  op.type = OpType::gate;
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.params, "params", input);

  // Check for optional label
  // If label is not specified record the gate name as the label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  if (label != "")
    op.string_params = {label};
  else
    op.string_params = {op.name};

  // Conditional
  add_conditional(Allowed::Yes, op, input);

  // Validation
  check_empty_name(op);
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  check_gate_params(op);

  return op;
}

template <typename inputdata_t>
Op input_to_op_qerror_loc(const inputdata_t &input) {
  Op op;
  op.type = OpType::qerror_loc;
  Parser<inputdata_t>::get_value(op.name, "label", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::Yes, op, input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_barrier(const inputdata_t &input) {
  Op op;
  op.type = OpType::barrier;
  op.name = "barrier";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  // Check conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_measure(const inputdata_t &input) {
  Op op;
  op.type = OpType::measure;
  op.name = "measure";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.memory, "memory", input);
  Parser<inputdata_t>::get_value(op.registers, "register", input);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.memory.empty() == false && op.memory.size() != op.qubits.size()) {
    throw std::invalid_argument(
        R"(Invalid measure operation: "memory" and "qubits" are different lengths.)");
  }
  if (op.registers.empty() == false &&
      op.registers.size() != op.qubits.size()) {
    throw std::invalid_argument(
        R"(Invalid measure operation: "register" and "qubits" are different lengths.)");
  }
  return op;
}

template <typename inputdata_t>
Op input_to_op_reset(const inputdata_t &input) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  return op;
}

template <typename inputdata_t>
Op input_to_op_initialize(const inputdata_t &input) {
  Op op;
  op.type = OpType::initialize;
  op.name = "initialize";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.params, "params", input);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  check_length_params(op, 1ULL << op.qubits.size());
  return op;
}
template <typename inputdata_t>
Op input_to_op_pauli(const inputdata_t &input) {
  Op op;
  op.type = OpType::gate;
  op.name = "pauli";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.string_params, "params", input);

  // Check for optional label
  // If label is not specified record the gate name as the label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  if (label != "")
    op.string_params.push_back(label);
  else
    op.string_params.push_back(op.name);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);

  return op;
}

//------------------------------------------------------------------------------
// Implementation: Boolean Functions
//------------------------------------------------------------------------------
template <typename inputdata_t>
Op input_to_op_bfunc(const inputdata_t &input) {
  Op op;
  op.type = OpType::bfunc;
  op.name = "bfunc";
  op.string_params.resize(2);
  std::string relation;
  Parser<inputdata_t>::get_value(op.string_params[0], "mask",
                                 input); // mask hexadecimal string
  Parser<inputdata_t>::get_value(op.string_params[1], "val",
                                 input); // value hexadecimal string
  Parser<inputdata_t>::get_value(relation, "relation",
                                 input); // relation string
  // Load single register / memory bit for storing result
  uint_t tmp;
  if (Parser<inputdata_t>::get_value(tmp, "register", input)) {
    op.registers.push_back(tmp);
  }
  if (Parser<inputdata_t>::get_value(tmp, "memory", input)) {
    op.memory.push_back(tmp);
  }

  // Format hex strings
  Utils::format_hex_inplace(op.string_params[0]);
  Utils::format_hex_inplace(op.string_params[1]);

  const stringmap_t<BinaryOp> comp_table({
      {"==", BinaryOp::Equal},
      {"!=", BinaryOp::NotEqual},
      {"<", BinaryOp::Less},
      {"<=", BinaryOp::LessEqual},
      {">", BinaryOp::Greater},
      {">=", BinaryOp::GreaterEqual},
  });

  auto it = comp_table.find(relation);
  if (it == comp_table.end()) {
    std::stringstream msg;
    msg << "Invalid bfunc relation string :\"" << it->first << "\"."
        << std::endl;
    throw std::invalid_argument(msg.str());
  } else {
    op.binary_op = it->second;
  }

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  if (op.registers.empty()) {
    throw std::invalid_argument(
        "Invalid measure operation: \"register\" is empty.");
  }
  return op;
}

template <typename inputdata_t>
Op input_to_op_roerror(const inputdata_t &input) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  Parser<inputdata_t>::get_value(op.memory, "memory", input);
  Parser<inputdata_t>::get_value(op.registers, "register", input);
  Parser<inputdata_t>::get_value(op.probs, "params", input);
  // Conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Matrix and Kraus deserialization
//------------------------------------------------------------------------------
template <typename inputdata_t>
Op input_to_op_unitary(const inputdata_t &input) {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.mats, "params", input);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.mats.size() != 1) {
    throw std::invalid_argument("\"unitary\" params must be a single matrix.");
  }
  for (const auto &mat : op.mats) {
    if (!Utils::is_unitary(mat, 1e-7)) {
      throw std::invalid_argument("\"unitary\" matrix is not unitary.");
    }
  }
  // Check for a label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  op.string_params.push_back(label);

  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}
template <typename inputdata_t>
Op input_to_op_diagonal(const inputdata_t &input) {
  Op op;
  op.type = OpType::diagonal_matrix;
  op.name = "diagonal";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.params, "params", input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.params.size() != 1ULL << op.qubits.size()) {
    throw std::invalid_argument("\"diagonal\" matrix is wrong size.");
  }
  for (const auto &val : op.params) {
    if (!Linalg::almost_equal(std::abs(val), 1.0, 1e-7)) {
      throw std::invalid_argument("\"diagonal\" matrix is not unitary.");
    }
  }

  // Check for a label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  op.string_params.push_back(label);

  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}
template <typename inputdata_t>
Op input_to_op_superop(const inputdata_t &input) {
  // Warning: we don't check superoperator is valid!
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.mats, "params", input);
  // Check conditional
  add_conditional(Allowed::Yes, op, input);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.mats.size() != 1) {
    throw std::invalid_argument("\"superop\" params must be a single matrix.");
  }
  return op;
}
template <typename inputdata_t>
Op input_to_op_multiplexer(const inputdata_t &input) {
  // Parse parameters
  reg_t qubits;
  std::vector<cmatrix_t> mats;
  std::string label;
  Parser<inputdata_t>::get_value(qubits, "qubits", input);
  Parser<inputdata_t>::get_value(mats, "params", input);
  Parser<inputdata_t>::get_value(label, "label", input);
  // Construct op
  auto op = make_multiplexer(qubits, mats, -1, nullptr, label);
  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}
template <typename inputdata_t>
Op input_to_op_kraus(const inputdata_t &input) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.mats, "params", input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_noise_switch(const inputdata_t &input) {
  Op op;
  op.type = OpType::noise_switch;
  op.name = "noise_switch";
  Parser<inputdata_t>::get_value(op.params, "params", input);
  // Conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Set state
//------------------------------------------------------------------------------
template <typename inputdata_t>
Op input_to_op_set_vector(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t &params = Parser<inputdata_t>::get_value("params", input);
  op.params =
      Parser<inputdata_t>::template get_list_elem<std::vector<complex_t>>(
          params, 0);
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_set_matrix(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t &params = Parser<inputdata_t>::get_value("params", input);
  op.mats.push_back(
      Parser<inputdata_t>::template get_list_elem<cmatrix_t>(params, 0));
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_set_clifford(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t &params = Parser<inputdata_t>::get_value("params", input);
  op.clifford = Parser<inputdata_t>::template get_list_elem<Clifford::Clifford>(
      params, 0);
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_set_mps(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t &params = Parser<inputdata_t>::get_value("params", input);
  op.mps =
      Parser<inputdata_t>::template get_list_elem<mps_container_t>(params, 0);

  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Save data deserialization
//------------------------------------------------------------------------------
template <typename inputdata_t>
Op input_to_op_save_default(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  Parser<inputdata_t>::get_value(op.name, "name", input);

  // Get subtype
  static const std::unordered_map<std::string, DataSubType> subtypes{
      {"single", DataSubType::single},   {"c_single", DataSubType::c_single},
      {"average", DataSubType::average}, {"c_average", DataSubType::c_average},
      {"list", DataSubType::list},       {"c_list", DataSubType::c_list},
      {"accum", DataSubType::accum},     {"c_accum", DataSubType::c_accum},
  };
  std::string subtype;
  Parser<inputdata_t>::get_value(subtype, "snapshot_type", input);
  auto subtype_it = subtypes.find(subtype);
  if (subtype_it == subtypes.end()) {
    throw std::runtime_error("Invalid data subtype \"" + subtype +
                             "\" in save data instruction.");
  }
  op.save_type = subtype_it->second;

  // Get data key
  op.string_params.emplace_back("");
  Parser<inputdata_t>::get_value(op.string_params[0], "label", input);

  // Add optional qubits field
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  return op;
}
template <typename inputdata_t>
Op input_to_op_save_expval(const inputdata_t &input, bool variance) {
  // Initialized default save instruction params
  auto op_type = (variance) ? OpType::save_expval_var : OpType::save_expval;
  Op op = input_to_op_save_default(input, op_type);

  // Parse Pauli operator components
  const auto threshold = 1e-12; // drop small components
  // Get components
  if (Parser<inputdata_t>::check_key("params", input) &&
      Parser<inputdata_t>::is_array("params", input)) {
    for (const auto &comp_ : Parser<inputdata_t>::get_value("params", input)) {
      const auto &comp = Parser<inputdata_t>::get_as_list(comp_);
      // Get complex coefficient
      std::vector<double> coeffs =
          Parser<inputdata_t>::template get_list_elem<std::vector<double>>(comp,
                                                                           1);
      if (std::abs(coeffs[0]) > threshold || std::abs(coeffs[1]) > threshold) {
        std::string pauli =
            Parser<inputdata_t>::template get_list_elem<std::string>(comp, 0);
        if (pauli.size() != op.qubits.size()) {
          throw std::invalid_argument(
              std::string("Invalid expectation value save instruction ") +
              "(Pauli label does not match qubit number.).");
        }
        op.expval_params.emplace_back(pauli, coeffs[0], coeffs[1]);
      }
    }
  } else {
    throw std::invalid_argument("Invalid save expectation value \"params\".");
  }

  // Check edge case of all coefficients being empty
  // In this case the operator had all coefficients zero, or sufficiently
  // close to zero that they were all truncated.
  if (op.expval_params.empty()) {
    std::string pauli(op.qubits.size(), 'I');
    op.expval_params.emplace_back(pauli, 0., 0.);
  }

  return op;
}
template <typename inputdata_t>
Op input_to_op_save_amps(const inputdata_t &input, bool squared) {
  // Initialized default save instruction params
  auto op_type = (squared) ? OpType::save_amps_sq : OpType::save_amps;
  Op op = input_to_op_save_default(input, op_type);
  Parser<inputdata_t>::get_value(op.int_params, "params", input);
  return op;
}

template <typename inputdata_t>
Op input_to_op_jump(const inputdata_t &input) {
  Op op;
  op.type = OpType::jump;
  op.name = "jump";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.string_params, "params", input);
  if (op.string_params.empty())
    throw std::invalid_argument(
        std::string("Invalid jump (\"params\" field missing)."));

  // Conditional
  add_conditional(Allowed::Yes, op, input);

  return op;
}

template <typename inputdata_t>
Op input_to_op_mark(const inputdata_t &input) {
  Op op;
  op.type = OpType::mark;
  op.name = "mark";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.string_params, "params", input);
  if (op.string_params.empty())
    throw std::invalid_argument(
        std::string("Invalid mark (\"params\" field missing)."));

  // Conditional
  add_conditional(Allowed::No, op, input);

  return op;
}

//------------------------------------------------------------------------------
} // end namespace Operations
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
