/**
 * Copyright 2019, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_tensor_state_hpp_
#define _aer_tensor_state_hpp_

#include "framework/json.hpp"
#include "framework/types.hpp"
#include "framework/utils.hpp"
#include "tensor.hpp"

//namespace AER {
namespace TensorState {

class TensorState{
public:
	TensorState(size_t size = 0);
	~TensorState();
    uint num_qubits() const{return size_;}
    void set_num_qubits(size_t size);
	void apply_h(uint index){q_reg_[index].apply_h();}
	void apply_x(uint index){q_reg_[index].apply_x();}
	void apply_y(uint index){q_reg_[index].apply_y();}
	void apply_z(uint index){q_reg_[index].apply_z();}
    void apply_s(uint index){q_reg_[index].apply_s();}
	void apply_sdg(uint index){q_reg_[index].apply_sdg();}
	void apply_t(uint index){q_reg_[index].apply_t();}
	void apply_tdg(uint index){q_reg_[index].apply_tdg();}
	void U1(uint index, double lambda){q_reg_[index].apply_u1(lambda);}
	void U2(uint index, double phi, double lambda){q_reg_[index].apply_u2(phi,lambda);}
	void U3(uint index, double theta, double phi, double lambda){q_reg_[index].apply_u3(theta,phi,lambda);}
	void apply_cnot(uint index_A, uint index_B);
	void apply_swap(uint index_A, uint index_B);
	void apply_cz(uint index_A, uint index_B);


  void apply_matrix(const AER::reg_t &qubits, const cvector_t &vmat) 
                      {cout << "apply_matrix not supported yet" <<endl;}
  void apply_diagonal_matrix(const AER::reg_t &qubits, const cvector_t &vmat) 
                      {cout << "apply_diagonalmatrix not supported yet" <<endl;}
  double Expectation_value(vector<uint> indexes, string matrices);
  void initialize();
  void initialize(const TensorState &other);
  void initialize(uint num_qubits, const cvector_t &vecState);
  //void initialize(complex_t alpha[], complex_t beta[]);
  void printTN();
  Tensor state_vec(uint first_index, uint last_index);

  //methods from qasm_controller that are not supported yet
  void set_omp_threads(int threads) {
           cout << "set_omp_threads not supported yet" <<endl;}
  void set_omp_threshold(int omp_qubit_threshold) {
           cout << "set_omp_threadshold not supported yet" <<endl;}
  void set_json_chop_threshold(double json_chop_threshold) {
           cout << "set_json_chop_threshold not supported yet" <<endl;}
  void set_sample_measure_index_size(int index_size){
           cout << "set_sample_measure_index_size not supported yet" <<endl;}
  void enable_gate_opt() {
           cout << "enable_gate_opt not supported yet" <<endl;}
  rvector_t probabilities(const AER::reg_t &qubits) const{
           cout << "probabilities not supported yet" <<endl;
           return rvector_t();}
  void store_measure(const AER::reg_t outcome, const AER::reg_t &cmemory, const AER::reg_t &cregister) const{
           cout << " store_measure not supported yet" <<endl;}
  double norm(const AER::reg_t &reg_qubits, cvector_t &vmat) const {
           cout << "norm not supported yet" <<endl;
           return 0;}
  auto sample_measure(std::vector<double> &rnds) {
    cout << "sample_measure not supported yet" <<endl;
    return 0;}
    

protected:
    uint size_;
	/*
	The data structure of a MPS- a vector of Gamma tensors and a vector of Lambda vectors.
	*/
	vector<Tensor> q_reg_;
	vector<rvector_t> lambda_reg_;
};



//-------------------------------------------------------------------------
} // end namespace TensorState
//-------------------------------------------------------------------------
//} // end namespace AER
//-------------------------------------------------------------------------
#endif /* _aer_tensor_state_hpp_ */
