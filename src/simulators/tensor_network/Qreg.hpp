/*
 * Qreg.hpp
 *
 *  Created on: Aug 23, 2018
 *      Author: eladgold
 */

#ifndef QREG_HPP_
#define QREG_HPP_
//#define SQR_HALF 0.707106781186547572737
#define SQR_HALF sqrt(0.5)
//#define MAX 100
#define MPS true

#define DEBUG false

#define SHOW_SVD false
#define TRY true
#include "Tensor.hpp"
#include <nlohmann/json.hpp>
//#include "json_nlohmann.hpp"
using json_t = nlohmann::json;

int parseLine(char* line);
int getValue();

class Qreg
{
public:
	Qreg(int size);
	~Qreg();
	int get_size(){return size_;}
	void I(int index);
	void H(int index);
	void X(int index);
	void Y(int index);
	void Z(int index);
	void S(int index);
	void S_D(int index);
	void T(int index);
	void T_D(int index);
	void U1(int index, double lambda);
	void U2(int index, double lambda, double phi);
	void U3(int index, double lambda, double phi, double theta);
	void CNOT(int index_A, int index_B);
	void SWAP(int index_A, int index_B);
	void Density_Matrix(int index, double** rho);
	double Expectation_value_X(int index);
	void Initialize();
	void Initialize(complex_t alpha[], complex_t beta[]);
	void Update_entangled_qubits(int index_A, int SV_num);
	void Update_entangled_qubits_matrix(int index_A,int index_B, int SV_num);
	void printTN();
	void print_entangled_matrix();
	void state_vec(json_t& j);
	void state_vec_TRY(json_t& json_result);
	int Compose_TRY(complex_t** new_data, complex_t** temp, int index_B);
	int Compose(complex_t** new_data, int index_A, int index_B); //returns new data width
	void DeCompose(complex_t** new_data, int index_A, int index_B);
        int GetDim(uint index)
           {return q_reg[index]->get_dim();}

protected:
	Tensor** q_reg;
	Tensor** lambda_reg;
	int size_;
	int*** entangled_qubits_matrix; // 3rd deg is to remember the entangled index in the tensor, counting from msb(=1)
	int* entangled_dim_between_qubits;
};

#endif /* QREG_HPP_ */
