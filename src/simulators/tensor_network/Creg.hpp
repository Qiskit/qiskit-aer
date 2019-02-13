/*
 * Creg.hpp
 *
 *  Created on: Nov 12, 2018
 *      Author: eladgold
 */

#ifndef CREG_HPP_
#define CREG_HPP_

class Creg
{
public:
	Creg():c_reg{0} {}
	void insert(int bit, double value) {c_reg[bit] = value;}
	double read(int bit) { return  c_reg[bit];}
protected:
	double* c_reg;
};



#endif /* CREG_HPP_ */
