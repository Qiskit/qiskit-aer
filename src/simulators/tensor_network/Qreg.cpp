/*
 * Qreg.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: eladgold
 */

#include "Qreg.hpp"
#include <bitset>
#include <math.h>

// for analysis of memory consumption
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "sys/types.h"
#include "sys/sysinfo.h"

struct sysinfo memInfo;
long long totalPhysMem = memInfo.totalram;


int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

template <class T>
void printMatrix(T **matrix, uint rows, uint columns){
  for(uint i = 0; i < rows; ++i)
    {
      for(uint j = 0; j < columns; ++j)
	{
	  if( norm(matrix[i][j]) > 0.000000001)
	    cout << fixed << matrix[i][j] << ' ';
	  else
	    cout << 0 << ' ';
	}
      cout << endl;
    }
}

int getPhysValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

void swap(complex_t &a, complex_t &b)
{
	complex_t temp = a;
	a = b;
	b = temp;
}

Qreg::Qreg(int size)
{

    size_ = size;
	q_reg = new Tensor*[size];
	lambda_reg = new Tensor*[size-1];
	entangled_dim_between_qubits = new int[size];
	for(int i = 0; i < size; ++i)
	{
		entangled_dim_between_qubits[i] = 1;
	}
	complex_t alpha = 1.0f;
	complex_t beta = 0.0f;
	for(int i = 0; i < size_; i++)
		q_reg[i] = new Tensor(alpha,beta);
	for(int i = 0; i < size_-1; i++)
		lambda_reg[i] = new Tensor(alpha,beta);
}

Qreg::~Qreg()
{
	for(int i = 0; i < size_; ++i)
	{
		delete q_reg[i];
	}
	for(int i = 0; i < size_-1; i++)
		delete lambda_reg[i];
	delete [] lambda_reg;
	delete [] q_reg;
	delete [] entangled_dim_between_qubits;
}

void Qreg::Initialize()
{
	complex_t alpha = 1.0f;
	complex_t beta = 0.0f;
	for(int i = 0; i < size_; i++)
		q_reg[i] = new Tensor(alpha,beta);
}

void Qreg::Initialize(complex_t alpha[], complex_t beta[])
{
	for(int i = 0; i < size_; i++)
		q_reg[i] = new Tensor(alpha[i],beta[i]);
}

void Qreg::I(int index)
{
	Tensor* pA = q_reg[index-1];
	complex_t temp[2];
	for (int i = 0; i < NN; i++)
	{
		temp[0] = pA->get_data(0,i);
		temp[1] = pA->get_data(1,i);
		pA->insert_data(0,i,temp[0]);
		pA->insert_data(1,i,temp[1]);
	}
}

void Qreg::H(int index)
{
	Tensor* pA = q_reg[index-1];
	complex_t temp[2];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		temp[0] = SQR_HALF*pA->get_data(0,i) + SQR_HALF*pA->get_data(1,i);
		temp[1] = SQR_HALF*pA->get_data(0,i) - SQR_HALF*pA->get_data(1,i);
		pA->insert_data(0,i,temp[0]);
		pA->insert_data(1,i,temp[1]);
	}
}

void Qreg::X(int index)
{
	Tensor* pA = q_reg[index-1];
	complex_t temp[2];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		temp[0] = pA->get_data(0,i);
		temp[1] = pA->get_data(1,i);
		pA->insert_data(0,i,temp[1]);
		pA->insert_data(1,i,temp[0]);
	}
}

void Qreg::Y(int index)
{
	Tensor* pA = q_reg[index-1];
	complex_t temp[2];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		temp[0] = pA->get_data(0,i);
		temp[1] = pA->get_data(1,i);
		pA->insert_data(0,i,temp[1] * complex_t(0,-1));
		pA->insert_data(1,i,temp[0] * complex_t(0, 1));
	}
}

void Qreg::Z(int index)
{
	Tensor* pA = q_reg[index-1];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		pA->insert_data(0,i,pA->get_data(0,i));
		pA->insert_data(1,i, pA->get_data(1,i) * (-1.0));
	}
}

void Qreg::S(int index)
{
	Tensor* pA = q_reg[index-1];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		pA->insert_data(0,i,pA->get_data(0,i));
		pA->insert_data(1,i, pA->get_data(1,i) * complex_t(0, 1));
	}
}

void Qreg::S_D(int index)
{
	Tensor* pA = q_reg[index-1];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		pA->insert_data(0,i,pA->get_data(0,i));
		pA->insert_data(1,i, pA->get_data(1,i) * complex_t(0, -1));
	}
}

void Qreg::T(int index)
{
	Tensor* pA = q_reg[index-1];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		pA->insert_data(0,i,pA->get_data(0,i));
		pA->insert_data(1,i, pA->get_data(1,i) * complex_t(SQR_HALF, SQR_HALF));
	}
}

void Qreg::T_D(int index)
{
	Tensor* pA = q_reg[index-1];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		pA->insert_data(0,i,pA->get_data(0,i));
		pA->insert_data(1,i, pA->get_data(1,i) * complex_t(SQR_HALF, -SQR_HALF));
	}
}

void Qreg::U1(int index, double lambda)
{
	Tensor* pA = q_reg[index-1];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		pA->insert_data(0,i,pA->get_data(0,i));
		pA->insert_data(1,i, pA->get_data(1,i) * exp(complex_t(0,lambda)));
	}
}

void Qreg::U2(int index, double lambda, double phi)
{
	Tensor* pA = q_reg[index-1];
	complex_t temp[2];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		temp[0] = (pA->get_data(0,i) - pA->get_data(1,i) * exp(complex_t(0,lambda))) *SQR_HALF;
		temp[1] = (pA->get_data(0,i) * exp(complex_t(0,phi)) + pA->get_data(1,i) * exp(complex_t(0,lambda+phi))) *SQR_HALF;
		pA->insert_data(0,i, temp[0]);
		pA->insert_data(1,i, temp[1]);
	}
}

void Qreg::U3(int index, double lambda, double phi, double theta)
{
	Tensor* pA = q_reg[index-1];
	complex_t temp[2];
	for (int i = 0; i < GetDim(index-1); i++)
	{
		double cos_ = std::cos(theta/2), sin_ = std::sin(theta/2);
		complex_t exp_lambda = exp(complex_t(0,lambda));
		complex_t exp_phi    = exp(complex_t(0,phi));
		temp[0] = (pA->get_data(0,i) * cos_ - pA->get_data(1,i) * exp_lambda * sin_);
		temp[1] = (pA->get_data(0,i) * exp_phi * sin_ + pA->get_data(1,i) * exp_phi * exp_lambda * cos_);
		pA->insert_data(0,i, temp[0]);
		pA->insert_data(1,i, temp[1]);
	}
}

void Qreg::CNOT(int index_A, int index_B)
{
  //cout << "in CNOT, index_A = " << index_A << " index_B = " << index_B << endl;
  //for MPS
	if(MPS)
	{
		if(index_A + 1 < index_B)
		{
			this->SWAP(index_A,index_B-1);
			this->CNOT(index_B-1,index_B);
			this->SWAP(index_A,index_B-1);
			return;
		}
		else if(index_A  > index_B + 1)
		{
			this->SWAP(index_A-1,index_B);
			this->CNOT(index_A,index_A-1);
			this->SWAP(index_A-1,index_B);
			return;
		}
	}
	//cout << "after if MPS" <<endl;

	bool swapped = false;
	if(index_A >  index_B)
	{
		swap(index_A, index_B);
		swapped = true;
	}
	complex_t* new_data[4];
	int newdata_dim = this->Compose(new_data, index_A, index_B);

	// actual 2-qubit gate calc, may be replaced in the future
	for(int k = 0; k < newdata_dim ; ++k)
	{
		if(!swapped)
			swap(new_data[2][k],new_data[3][k]);
		else
			swap(new_data[1][k],new_data[3][k]);
	}
	this->DeCompose(new_data, index_A, index_B);
}

void Qreg::SWAP(int index_A, int index_B)
{
	if(index_A >  index_B)
	{
		swap(index_A, index_B);
	}
	//for MPS
	if(MPS)
	{
		if(index_A + 1 < index_B)
		{
			int i;
			for(i = index_A; i < index_B; i++)
			{
				this->SWAP(i,i+1);
			}
			for(i = index_B-1; i > index_A; i--)
			{
				this->SWAP(i,i-1);
			}
			return;
		}
	}

	complex_t* new_data[4];
	int newdata_dim = this->Compose(new_data, index_A, index_B);

	// actual 2-qubit gate calc, may be replaced in the future
	for(int k = 0; k < newdata_dim ; ++k)
	{
		swap(new_data[1][k],new_data[2][k]);
	}
	this->DeCompose(new_data, index_A, index_B);
}


void Qreg::Density_Matrix(int index, double** rho)
{
	Tensor *pA, *p_lambdaA, *p_lambdaB;
	if(index == 1)
	{
		pA = q_reg[index - 1];
		p_lambdaB = lambda_reg[index - 1];
		int dim_left = 1, dim_right = p_lambdaB->get_dim();

		for(int a_left = 0; a_left < dim_left ; a_left++)
		{
			for(int a_right = 0; a_right < dim_right ; a_right++)
			{
				for(int i = 0; i < 2 ; i++)
				{
					for(int i_tag = 0; i_tag < 2 ; i_tag++)
					{
						rho[i][i_tag] += 1 *
									 real(pA->get_data(i,a_left*dim_right+a_right) *conj(pA->get_data(i_tag,a_left*dim_right+a_right))) *
									 norm(p_lambdaB->get_data(0,a_right));
					}
				}
			}
		}
	}
	else if(index == this->size_)
	{
		pA = q_reg[index - 1];
		p_lambdaA = lambda_reg[index - 2];
		int dim_left = p_lambdaA->get_dim(), dim_right = 1;

		for(int a_left = 0; a_left < dim_left ; a_left++)
		{
			for(int a_right = 0; a_right < dim_right ; a_right++)
			{
				for(int i = 0; i < 2 ; i++)
				{
					for(int i_tag = 0; i_tag < 2 ; i_tag++)
					{
						rho[i][i_tag] += norm(p_lambdaA->get_data(0,a_left)) *
									 real(pA->get_data(i,a_left*dim_right+a_right) *conj(pA->get_data(i_tag,a_left*dim_right+a_right))) *
									 1;
					}
				}
			}
		}
	}
	else
	{
		pA = q_reg[index - 1];
		p_lambdaA = lambda_reg[index - 2];
		p_lambdaB = lambda_reg[index - 1];
		int dim_left = p_lambdaA->get_dim(), dim_right = p_lambdaB->get_dim();

		for(int a_left = 0; a_left < dim_left ; a_left++)
		{
			for(int a_right = 0; a_right < dim_right ; a_right++)
			{
				for(int i = 0; i < 2 ; i++)
				{
					for(int i_tag = 0; i_tag < 2 ; i_tag++)
					{
						rho[i][i_tag] += norm(p_lambdaA->get_data(0,a_left)) *
									 real(pA->get_data(i,a_left*dim_right+a_right) *conj(pA->get_data(i_tag,a_left*dim_right+a_right))) *
									 norm(p_lambdaB->get_data(0,a_right));
					}
				}
			}
		}
	}

}

double Qreg::Expectation_value_X(int index)
{
	double **rho;
	//memory allocate
	rho = new double*[2];
	for(int i = 0; i < 2; ++i)
	{
		rho[i] = new double[2];
		for(int j = 0; j < 2; ++j)
		{
			rho[i][j] = 0.0;
		}
	}

	Density_Matrix(index, rho);
//	printMatrix(rho, 2, 2);
	double res[2][2] = {{0}};
	// Trace(rho*X)
	res[0][0] = rho[0][0] * 0 +  rho[0][1] * 1;
	res[1][1] = rho[1][0] * 1 +  rho[1][1] * 0;
	return res[0][0]+res[1][1];
}

//probably have bugs
double Qreg::Expectation_value_Y(int index)
{
	double **rho;
	//memory allocate
	rho = new double*[2];
	for(int i = 0; i < 2; ++i)
	{
		rho[i] = new double[2];
		for(int j = 0; j < 2; ++j)
		{
			rho[i][j] = 0.0;
		}
	}

	Density_Matrix(index, rho);
//	printMatrix(rho, 2, 2);
	complex_t res[2][2] = {{0}};
	// Trace(rho*X)
	res[0][0] = (rho[0][0] * 0) +  (rho[0][1] * complex_t(0,1));
	res[1][1] = (rho[1][0] * complex_t(0,-1)) +  (rho[1][1] * 0);
	return imag(res[0][0]+res[1][1]);
}

double Qreg::Expectation_value_Z(int index)
{
	double **rho;
	//memory allocate
	rho = new double*[2];
	for(int i = 0; i < 2; ++i)
	{
		rho[i] = new double[2];
		for(int j = 0; j < 2; ++j)
		{
			rho[i][j] = 0.0;
		}
	}

	Density_Matrix(index, rho);
//	printMatrix(rho, 2, 2);
	double res[2][2] = {{0}};
	// Trace(rho*X)
	res[0][0] = rho[0][0] * 1 +  rho[0][1] * 0;
	res[1][1] = rho[1][0] * 0 +  rho[1][1] * (-1);
	return res[0][0]+res[1][1];
}

void Qreg::printTN()
{
	for(int i=0; i<size_; i++)
	{
		cout << "Gamma  [" << i+1 << "] dim =  " << q_reg[i]->get_dim() << endl;
		q_reg[i]->print();
		if(i != size_- 1)
		{
			cout << "Lambda [" << i+1 << "] dim =  " << lambda_reg[i]->get_dim() << endl;
			lambda_reg[i]->print(0);
		}
	}
	cout << endl;
}


void print_state_vector(complex_t **state_vector, int size,json_t& json_result)
{
	for(int k = 0; k < pow(2,size); k++)
	{
		json_result["results"][k] = {state_vector[k][0].real(), state_vector[k][0].imag()};
	}
}

int Max(int *entangled_dim_between_qubits, int size_)
{
	int max = entangled_dim_between_qubits[0];
	for(int i = 1; i < size_; i++)
	{
		if(max < entangled_dim_between_qubits[i])
		{
			max = entangled_dim_between_qubits[i];
		}
	}
	return max;
}

void Qreg::state_vec_TRY(json_t& json_result)
{
	complex_t** temp;
	complex_t** state_vector;
	unsigned long int exp_m = pow(2,size_);
	//memory allocate
	temp         = new complex_t*[exp_m];
	state_vector = new complex_t*[exp_m];

	int max_dim = Max(entangled_dim_between_qubits, size_);

	for(unsigned long int i = 0; i < exp_m; ++i)
	{
		temp[i] = new complex_t[max_dim];
		state_vector[i] = new complex_t[max_dim];
	}

	//initialize temp with first qubit
	Tensor* pTensor = q_reg[0];
	int temp_dim = pTensor->get_dim();

//	for(unsigned long int i = 0; i < 4; ++i)
//	{
//		temp[i] = new complex_t[temp_dim];
//		state_vector[i] = new complex_t[temp_dim];
//	}

	for(int a1 = 0; a1 < temp_dim; a1++)
	{
		temp[0][a1] = pTensor->get_data(0,a1);
		temp[1][a1] = pTensor->get_data(1,a1);
	}

	//Compose all qubits
	for(int index_B = 2; index_B <= size_ ; index_B++)
	{

		temp_dim = Compose_TRY(state_vector, temp, index_B);
//		swap(temp,state_vector);
		for(int a = 0; a < pow(2,index_B); a++)
		{

			for(int b = 0; b < temp_dim; b++)
			{
				temp[a][b] = state_vector[a][b];
			}
		}
	}
	print_state_vector(state_vector, size_, json_result);
	for(unsigned long int i = 0; i < exp_m; ++i)
	{
		delete [] state_vector[i];
		delete [] temp[i];
	}
	delete [] temp;
	delete [] state_vector;
}

int Qreg::Compose_TRY(complex_t** new_data, complex_t** temp, int index_B)
{
	Tensor* p_lambda = lambda_reg[index_B-2];
	Tensor* pB = q_reg[index_B-1];
	int B_dim = pB->get_dim(), A_dim = 1;
	int entangled_dim = entangled_dim_between_qubits[index_B-2];
	B_dim /= entangled_dim;
	int newdata_dim = B_dim;
	for(int i = 0; i < pow(2,index_B); ++i)
	{
		for(int a1 = 0; a1 < A_dim; ++a1)
		{
			for(int a3 = 0; a3 < B_dim; ++a3)
			{
				new_data[i][a1*B_dim+a3] = 0;
				for(int a2 = 0; a2 < entangled_dim; ++a2)
				{
					new_data[i][a1*B_dim+a3] += temp[i/2][a1*entangled_dim+a2] * pB->get_data(i%2,a2*B_dim+a3) * p_lambda->get_data(0,a2);
				}
			}
		}
	}
	return newdata_dim;
}
int Qreg::Compose(complex_t** new_data, int index_A, int index_B)
{
	Tensor* p_lambda_left;
	Tensor* p_lambda_right;
	if (index_A != 1)
	{
		p_lambda_left = lambda_reg[index_A-2];
	}
	else
	{
		complex_t alpha = 1.0f;
		complex_t beta = 0.0f;
		p_lambda_left = new Tensor(alpha,beta);
	}
	if (index_B != this->size_)
	{
		p_lambda_right = lambda_reg[index_B-1];
	}
	else
	{
		complex_t alpha = 1.0f;
		complex_t beta = 0.0f;
		p_lambda_right = new Tensor(alpha,beta);
	}


	Tensor* pA = q_reg[index_A-1];
	Tensor* p_lambda = lambda_reg[index_A-1];
	Tensor* pB = q_reg[index_B-1];
	int A_dim = pA->get_dim(), B_dim = pB->get_dim();
	int newdata_dim;

	int entangled_dim = entangled_dim_between_qubits[index_A-1];
	pA->dim_div(entangled_dim);
	pB->dim_div(entangled_dim);
	A_dim = pA->get_dim();
	B_dim = pB->get_dim();
	newdata_dim = A_dim*B_dim;
	for(int i = 0; i < 4; ++i)
	{
		new_data[i] = new complex_t[newdata_dim];

		for(int a1 = 0; a1 < A_dim; ++a1)
		{
			for(int a3 = 0; a3 < B_dim; ++a3)
			{
				new_data[i][a1*B_dim+a3] = 0;
				for(int a2 = 0; a2 < entangled_dim; ++a2)
				{
					new_data[i][a1*B_dim+a3] += p_lambda_left->get_data(0,a1) * pA->get_data(i/2,a1*entangled_dim+a2) * p_lambda_right->get_data(0,a3) * pB->get_data(i%2,a2*B_dim+a3) * p_lambda->get_data(0,a2);
				}
			}
		}
	}

	return newdata_dim;
}

void Transpose(complex_t** new_data, complex_t** C, int A_dim, int B_dim)
{
	for(int j = 0; j < A_dim; ++j)
	{
		for(int k = 0; k < B_dim; ++k)
		{
			C[j][k]				    = new_data[0][j*B_dim+k];
			C[j][B_dim + k] 		= new_data[1][j*B_dim+k];
			C[A_dim + j][k] 		= new_data[2][j*B_dim+k];
			C[A_dim + j][B_dim + k] = new_data[3][j*B_dim+k];
		}
	}
}

void Qreg::UnTranspose_U(complex_t** U, double* S, Tensor* pA, int SV_num, int index_A)
{
	int A_dim = pA->get_dim();

	Tensor* p_lambda_left;
	if (index_A != 1)
	{
		p_lambda_left = lambda_reg[index_A-2];
	}
	else
	{
		complex_t alpha = 1.0f;
		complex_t beta = 0.0f;
		p_lambda_left = new Tensor(alpha,beta);
	}

	for(int i = 0; i <= 1; i++)
	{
	  for(int a1 = 0; a1 < A_dim; a1++)
	    {
		  for(int a2 = 0; a2 < SV_num; a2++)
		    {
		      if (a1*SV_num+a2 >= NN)
		    	  cout << "a1 = " << a1 << ", SV_num = " << SV_num << ", a2 = " << a2 << endl;
		      assert(a1*SV_num+a2 < NN);
//		      pA->insert_data(i, a1*SV_num + a2 , U[i*A_dim + a1][a2] * S[a2]);
		      pA->insert_data(i, a1*SV_num + a2 , U[i*A_dim + a1][a2]/p_lambda_left->get_data(0,a1));
		    }
	    }
	}
}

void Qreg::UnTranspose_V(complex_t** V, Tensor* pB, int SV_num, int index_B)
{
	int B_dim = pB->get_dim();

	Tensor* p_lambda_right;
	if (index_B != this->size_)
	{
		p_lambda_right = lambda_reg[index_B-1];
	}
	else
	{
		complex_t alpha = 1.0f;
		complex_t beta = 0.0f;
		p_lambda_right = new Tensor(alpha,beta);
	}

	for(int i = 0; i <= 1; i++)
	{
		for(int a3 = 0; a3 < B_dim; a3++)
		{
			for(int a2 = 0; a2 < SV_num; a2++)
			{
			  assert(a2*B_dim+a3 < NN);
				pB->insert_data(i, a2*B_dim + a3, (std::conj(V[i*B_dim + a3][a2]))/p_lambda_right->get_data(0,a3));
			}
		}
	}
}

void Copy_S_To_lambda_reg(double *S, Tensor* p_lambda, int SV_num)
{
	for(int i = 0; i < SV_num; i++)
	{
		p_lambda->insert_data(0,i,complex_t(S[i]));
	}
}

// Input: vector S of length m that contains the real singular values from the SVD decomposition
// Output: number of elements in S that are greater than 0 (actually greater than 1e-16)
int num_of_SV(double* S, int m)
{
  //cout << "S[] = ";
	int sum = 0;
	for(int i = 0; i < m; ++i)
	{
	  //  cout << S[i] << " ";
	  if(std::norm(S[i]) > 1e-16)
			sum++;
	}
	//	cout <<endl;
	//	cout << "SV_Num = " << sum <<endl;
	if (sum == 0)
	  cout << "SV_Num == 0"<< endl;
	return sum;
}

void Qreg::DeCompose(complex_t** new_data, int index_A, int index_B)
{
	Tensor* pA = q_reg[index_A-1];
	Tensor* p_lambda = lambda_reg[index_A-1];
	Tensor* pB = q_reg[index_B-1];
	int A_dim = pA->get_dim(), B_dim = pB->get_dim();

	complex_t **a, **U, **V;
	double *S;
	int m = 2*A_dim, n = 2*B_dim;

	//memory allocate
	a = new complex_t*[max(m,n)];
	U = new complex_t*[max(m,n)];
	S = new double[max(m,n)];
	V = new complex_t*[max(m,n)];
	for(int i = 0; i < max(m,n); ++i)
	{
		a[i] = new complex_t[max(m,n)];
		U[i] = new complex_t[max(m,n)];
		V[i] = new complex_t[max(m,n)];
		S[i] = 0; // just to be sure
		for(int j = 0; j < max(m,n); ++j)
		{
			U[i][j] = complex_t(0.0,0.0);
			V[i][j] = complex_t(0.0,0.0);
			a[i][j] = complex_t(0.0,0.0);
		}
	}

	//prepare for SVD
	Transpose(new_data, a, A_dim, B_dim);


	if(SHOW_SVD)
	{
	  cout.precision(16);
	  cout << "printing a before SVD" <<endl;
	  cout << "C:" << endl;
	  printMatrix(a, m, n);
	  cout << "m = " << m << ", n = " << n << endl;
	}

	csvd (a, m, n, S, U, V);

	// check SVD result
	if(SHOW_SVD)
	{
	  cout << "U:" << endl;
	  printMatrix(U, m, m);

	  cout << "S:" << endl;
	  for(int i = 0; i < m; ++i) {
	    if(S[i] * S[i] > 0.000000001)
	      cout << fixed << S[i] << ' ';
	    else
	      cout << 0 << ' ';
	    cout << endl;
	  }

	  cout << "V:" << endl;
	  printMatrix(V, n, n);
	}

	//back from SVD
	int SV_num = num_of_SV(S,m);
	UnTranspose_U(U, S, pA, SV_num, index_A);
	Copy_S_To_lambda_reg(S, p_lambda, SV_num);
	UnTranspose_V(V, pB, SV_num, index_B);

	Update_entangled_qubits(index_A,SV_num);
	pA->dim_mult(SV_num);
	p_lambda->insert_dim(SV_num);
	pB->dim_mult(SV_num);


	// delete memory
	for(int i = 0; i < 4; ++i)
	{
		delete [] new_data[i];
	}
	for(int i = 0; i < max(m,n); ++i)
	{
		delete [] U[i];
	}
	delete [] U;
	delete [] S;
	for(int i = 0; i < max(m,n); ++i)
	{
		delete [] V[i];
	}
	delete [] V;
	for(int i = 0; i < max(m,n); ++i)
	{
		delete [] a[i];
	}
	delete [] a;

}

void Qreg::Update_entangled_qubits(int index_A, int SV_num)
{
  if (SV_num == 0)
    cout << "SV_num == 0 " << endl;
  entangled_dim_between_qubits[index_A-1] = SV_num;
}


