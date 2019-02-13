//============================================================================
// Name        : TN.cpp
// Author      : Elad Goldman
// Version     :
// Copyright   :
// Description :
//============================================================================

#include <iostream>
#include <complex>
#include <time.h>
#include <math.h>

#include "Qreg.hpp"
//#include "matrix.hpp"
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
//#include "json_nlohmann.hpp"
using json_t = nlohmann::json;
using namespace std;

int main()
{
	const clock_t begin_time = clock(); //tic

	ifstream infile("/gpfs/haifa/projects/q/qq/team/eladgold/tests/data.txt");
	string n_qubits, gate, qubit, param;
	json_t json_result, full_result;
	json_result["results"] = {0};
	getline(infile, n_qubits);
	int num_qubits = stoi(n_qubits);
	Qreg qreg(num_qubits);
	while (getline(infile, gate))
	{
		getline(infile, qubit);
		// if gate == "state vector" we compute and break
		if(gate == "state vector")
		{
			if(DEBUG) cout << gate << endl;
			qreg.state_vec_TRY(json_result);
			full_result[0] = json_result;
			break;
		}

		int first_qubit = num_qubits - stoi(qubit), second_qubit;
		double lambda, phi, theta;
		if(gate == "x")
		{
			qreg.X(first_qubit);
		}
		else if(gate == "y")
		{
			qreg.Y(first_qubit);
		}
		else if(gate == "z")
		{
			qreg.Z(first_qubit);
		}
		else if(gate == "h")
		{
			qreg.H(first_qubit);
		}
		else if(gate == "id")
		{
			qreg.I(first_qubit);
		}
		else if(gate == "t")
		{
			qreg.T(first_qubit);
		}
		else if(gate == "s")
		{
			qreg.S(first_qubit);
		}
		else if(gate == "sdg")
		{
			qreg.S_D(first_qubit);
		}
		else if(gate == "tdg")
		{
			qreg.T_D(first_qubit);
		}
		else if(gate == "u1")
		{
			getline(infile, param);
			lambda = stod(param);
			qreg.U1(first_qubit, lambda);
		}
		else if(gate == "u2")
		{
			getline(infile, param);
			lambda = stod(param);
			getline(infile, param);
			phi = stod(param);
			qreg.U2(first_qubit, lambda, phi);
		}
		else if(gate == "u3")
		{
			getline(infile, param);
			lambda = stod(param);
			getline(infile, param);
			phi = stod(param);
			getline(infile, param);
			theta = stod(param);
			qreg.U3(first_qubit, lambda, phi, theta);
		}
		else if(gate == "cx")
		{
			getline(infile, qubit);
			second_qubit = num_qubits - stoi(qubit);
			if(DEBUG) cout << gate << first_qubit << second_qubit << endl;
			qreg.CNOT(first_qubit, second_qubit);
		}
		else if(gate == "swap")
		{
			getline(infile, qubit);
			second_qubit = num_qubits - stoi(qubit);
			if(DEBUG) cout << gate << first_qubit << second_qubit << endl;
			qreg.SWAP(first_qubit, second_qubit);
		}
		else if(gate == "expectation X")
		{
			if(DEBUG) cout << gate << first_qubit << endl;
			full_result[0] = qreg.Expectation_value_X(first_qubit);
			break;
		}

		if(DEBUG)
		{
			qreg.printTN();
			cout << json_result << endl;
		}
	}


	full_result[1]["time_taken"][0] = float( clock () - begin_time ) /  CLOCKS_PER_SEC;

	cout << full_result << endl;
//	if(!DEBUG) cout << full_result << endl;


	return 1;
}
