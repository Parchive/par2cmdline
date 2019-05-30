//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2019 Michael D. Nahas
//
//  par2cmdline is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  par2cmdline is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


#include <iostream>
#include <stdlib.h>

#include "libpar2internal.h"
#include "reedsolomon.h"


/* trace from test11.log

Creator.SetInput 100
Creator.SetOutput 0 4
Creator.Compute 
creator.process 10004 0 0
creator.process 10004 0 1
creator.process 10004 0 2
creator.process 10004 0 3
creator.process 10004 0 4
creator.process 10004 1 0
creator.process 10004 1 1 
creator.process 10004 1 2
creator.process 10004 1 3
creator.process 10004 1 4
creator.process 10004 2 0
creator.process 10004 2 1
creator.process 10004 2 2
...
creator.process 10004 97 4
creator.process 10004 98 0
creator.process 10004 98 1
creator.process 10004 98 2
creator.process 10004 98 3
creator.process 10004 98 4
creator.process 10004 99 0
creator.process 10004 99 1
creator.process 10004 99 2
creator.process 10004 99 3
creator.process 10004 99 4
...
...
...
Repairer.SetInput 100 true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true false false false true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true true false true true true true true true true true true true false
Repair.SetOutput true 0
Repair.SetOutput true 1
Repair.SetOutput true 2
Repair.SetOutput true 3
Repair.SetOutput true 4
repair.compute
repair.process 10004 0 1
repair.process 10004 0 2
repair.process 10004 0 4
repair.process 10004 0 0
repair.process  10004 0 3

repair.process
repair.process 10004
repair.process 10004
repair.process  1  1
10004
repair.process 2
 1 3
10004
4
1 0
repair.process 10004 2 2
repair.process 10004 2 3
repair.process 10004 2
repair.process 10004 2 1
repair.process
*/


/* Reed-Solomon operations require a field.
   This is a dummy one, using a prime number.
   Due to how Log() is implemented, the prime must be greater than 3.
 */



/**  This is a simpler field than Galois. 
 *   I'm not sure why ReedSolomon was 
 *   specialized for Galois8 and Galois16.
 *   It should be made to work with this type.
 * 
#include <cmath>

template <const unsigned int prime, typename valuetype, int bits>
class PrimeField
{
public:
  typedef valuetype ValueType;

  // Basic constructors
  PrimeField(void) { value = 0; }
  PrimeField(ValueType v) { value = v;}

  // Copy and assignment
  PrimeField(const PrimeField &right) {value = right.value;}
  PrimeField& operator = (const PrimeField &right) { value = right.value; return *this;}

  // Addition
  PrimeField operator + (const PrimeField &right) const { return (valuetype) ((value + (unsigned long) right.value) % prime); }
  PrimeField& operator += (const PrimeField &right) { value = (valuetype) ((value + (unsigned long) right.value) % prime); return *this;}

  // Subtraction
  PrimeField operator - (const PrimeField &right) const { return (right.value <= value) ? value - right.value : prime + value - right.value;}
  PrimeField& operator -= (const PrimeField &right) { value = (right.value <= value) ? value - right.value : prime + value - right.value; return *this;}

  // Multiplication
  PrimeField operator * (const PrimeField &right) const { return (valuetype) ((value * (unsigned long) right.value) % prime); }
  PrimeField& operator *= (const PrimeField &right) { value = (valuetype) ((value * (unsigned long) right.value) % prime); return *this;}

  // Division
  PrimeField operator / (const PrimeField &right) const { return (valuetype) ((value * (unsigned long) reciprocal(right.value)) % prime); }
  PrimeField& operator /= (const PrimeField &right) { value = (valuetype) ((value * (unsigned long) reciprocal(right.value)) % prime); return *this;}

  // Power
  PrimeField pow(unsigned int right) const { return (valuetype) (((long) std::pow((double) value, (double) right)) % prime); }
  PrimeField operator ^ (unsigned int right) const { return (valuetype) (((long) std::pow((double) value, (double) right)) % prime); }
  PrimeField& operator ^= (unsigned int right) { value = (valuetype) (((long) std::pow((double) value, (double) right)) % prime); return *this; }

  // Cast to value and value access
  operator ValueType(void) const {return value;}
  ValueType Value(void) const {return value;}

  // Direct log and antilog
  // base 2
  ValueType Log(void) const {
    ValueType v(1);
    for (int i = 0; i < prime; i++) {
      if (v == value)
	return i;
      v = (valuetype) ((2l*v)%prime);
    }
    cerr << "Did not find multiplicative inverse for " << value << endl;
  }
  ValueType ALog(void) const { return (valuetype) (((long) std::pow((double) 2, (double) value)) % prime); }

  enum 
  {
    Bits  = bits,
    //    Count = GaloisTable<bits,generator,valuetype>::Count,
    //    Limit = GaloisTable<bits,generator,valuetype>::Limit,
  };

protected:
  ValueType reciprocal(ValueType v) {
    if (v == 0)
      return v;
    for (ValueType i = 1; i < prime; i++) {
      if ((i*v) % prime == 1)
	return i;
    }
    cerr << "Did not find multiplicative inverse for " << v << endl;
  }
  
  ValueType value;

};

*/


#define BUF_SIZE 1024

template<typename gtype, typename utype>
int generate_data(unsigned int seed, u8 data[][BUF_SIZE], int in_count, int recovery_count, int low_exponent) {
  int high_exponent = low_exponent + recovery_count - 1;
  
  // random input data
  srand(seed);
  
  for (int i = 0; i < in_count; i++) {
    for (int k = 0; k < BUF_SIZE; k++) {
      data[i][k] = (u8)(rand() % 256);
    }
  }
  // zero recovery
  for (int j = 0; j < recovery_count; j++) {
    for (int k = 0; k < BUF_SIZE; k++) {
      data[in_count + j][k] = (u8)0;
    }
  }


  ReedSolomon<gtype> rs_creator;

  //cout << "creator.setinput" << in_count << endl;
  if (!rs_creator.SetInput(in_count, cout, cerr)) {
    cerr << "rs_creator.SetInput returned false";
    return 1;
  }
  //cout << "creator.setoutput" << low_exponent << " " << high_exponent << endl;
  if (!rs_creator.SetOutput(false, low_exponent, high_exponent)) {
    cerr << "rs_creator.SetOutput returned false";
    return 1;
  }
  //cout << "creator.compute" << endl;
  if (!rs_creator.Compute(nlSilent, cout, cerr)) {
    cerr << "rs_creator.Compute returned false";
    return 1;
  }
  
  for (int i = 0; i < in_count; i++) {
    for (int j = 0; j < recovery_count; j++) {
	//cout << "creator.process " << BUF_SIZE << " " << i << " " << j << endl;
	rs_creator.Process(BUF_SIZE, i, &(data[i][0]), j, &(data[in_count + j][0]));
    }
  }

  return 0;
}


template<typename gtype, typename utype>
int init_repair_rs(ReedSolomon<gtype> &rs_repair, vector<bool> &in_present, vector<bool> &recovery_present, int low_exponent) {

  //cout << "Repairer.SetInput " << in_present.size();
  //for (unsigned int z = 0; z < in_present.size(); z++)
  //  cout << (in_present[z] ? " true": " false");
  //cout << endl;
  if (!rs_repair.SetInput(in_present, cout, cerr)) {
    cerr << "rs_repair.SetInput returned false";
    return 1;
  }
  
  for (unsigned int j = 0; j < recovery_present.size(); j++) {
    //cout << "Repair.SetOutput true " << j << endl;
    if (recovery_present[j]) {
      if (!rs_repair.SetOutput(true, low_exponent + j)) {
	cerr << "rs_repair.SetOutput returned false for " << j;
	return 1;
      }
    }
  }
  
  //cout << "Repair.compute" << endl;
  if (!rs_repair.Compute(nlSilent, cout, cerr)) {
    cerr << "rs_repair.Compute returned false";
    return 1;
  }

  return 0;
}


// Strange that "missing place" is 0,1,2,3...
// no matter what exponent or which in_present are false.
template<typename gtype, typename utype>
int recover_data(ReedSolomon<gtype> &rs_repair, int missing_place, u8 *buffer, u8 data[][BUF_SIZE], vector<bool> &in_present, vector<bool> &recovery_present) {
  memset(buffer, 0, BUF_SIZE);
  
  int index = 0;
  for (unsigned int i = 0; i < in_present.size(); i++) {
    if (in_present[i]) {
      //cout << "repair.process " << BUF_SIZE << " " << index << " " << 0 << endl;
      rs_repair.Process(BUF_SIZE, index, &(data[i][0]), missing_place, buffer);
      index++;
    }
  }
  for (unsigned int j = 0; j < recovery_present.size(); j++) {
    if (recovery_present[j]) {
      //cout << "repair.process " << BUF_SIZE << " " << index << " " << 0 << endl;
      rs_repair.Process(BUF_SIZE, index, &(data[in_present.size() + j][0]), missing_place, buffer);
      index++;
    }
  }

  return 0;
}


int compare_buffer(u8 *buffer, u8 *expected, const char *error_prefix) {
  for (int k = 0; k < BUF_SIZE; k++) {
    if (buffer[k] != expected[k]) {
      cerr << error_prefix << " mismatch at place " << k << endl;
      cerr << "  buffer had " << ((int) buffer[k]) << endl;
      cerr << "  expected " << ((int) expected[k]) << endl;
      return 1;
    }
  }

  return 0;
}



// 4 inputs, recover all possible cases of 2 missing inputs
template<typename gtype, typename utype>
int test1() {
  const int NUM_IN  = 4;
  const int NUM_REC = 2;  // recovery
  const int LOW_EXPONENT = 0; 

  u8 data[NUM_IN + NUM_REC][BUF_SIZE];

  if (generate_data<gtype, utype>(873945932, data, NUM_IN, NUM_REC, LOW_EXPONENT))
    return 1;

  // loop over missing input blocks
  for (int missing1 = 0; missing1 < NUM_IN; missing1++) {
    for (int missing2 = missing1+1; missing2 < NUM_IN; missing2++) {

      vector<bool> in_present;
      for (int i = 0; i < NUM_IN; i++) {
	in_present.push_back(i != missing1 && i != missing2);
      }
      vector<bool> recovery_present;
      for (int i = 0; i < NUM_REC; i++) {
	recovery_present.push_back(true);
      }
      
      ReedSolomon<gtype> rs_repair;
      if (init_repair_rs<gtype,utype>(rs_repair, in_present, recovery_present, LOW_EXPONENT))
	return 1;

      u8 result[BUF_SIZE];
      if (recover_data<gtype,utype>(rs_repair, 0, result, data, in_present, recovery_present))
	return 1;
      if (compare_buffer(result, data[missing1], "test1 - missing1"))
	return 1;

      if (recover_data<gtype,utype>(rs_repair, 1, result, data, in_present, recovery_present))
	return 1;
      if (compare_buffer(result, data[missing2], "test1 - missing2"))
	return 1;
    }
  }

  return 0;
}



// recover when all inputs are missing
template<typename gtype, typename utype>
int test2() {
  const int NUM_IN  = 5;
  const int NUM_REC = 5;  // recovery
  const int LOW_EXPONENT = 0; 

  u8 data[NUM_IN + NUM_REC][BUF_SIZE];

  if (generate_data<gtype, utype>(873945932, data, NUM_IN, NUM_REC, LOW_EXPONENT))
    return 1;

  vector<bool> in_present;
  for (int i = 0; i < NUM_IN; i++) {
    in_present.push_back(false);
  }
  vector<bool> recovery_present;
  for (int i = 0; i < NUM_REC; i++) {
    recovery_present.push_back(true);
  }
      
  ReedSolomon<gtype> rs_repair;
  if (init_repair_rs<gtype,utype>(rs_repair, in_present, recovery_present, LOW_EXPONENT))
    return 1;

  for (int i = 0; i < NUM_IN; i++) {
    u8 result[BUF_SIZE];
    if (recover_data<gtype,utype>(rs_repair, i, result, data, in_present, recovery_present))
      return 1;
    if (compare_buffer(result, data[i], "test2 - missing"))
      return 1;
  }

  return 0;
}


// too many recovery blocks
/* THIS OPERATION WAS ALLOWED, WITHOUT ANY WARNING
** I NEED TO CHANGE IT TO NOT BE ALLOWED.
template<typename gtype, typename utype>
int test3() {
  const int NUM_IN  = 4;
  const int NUM_REC = 2;  // recovery
  const int LOW_EXPONENT = 0; 

  u8 data[NUM_IN + NUM_REC][BUF_SIZE];

  if (generate_data<gtype, utype>(873945932, data, NUM_IN, NUM_REC, LOW_EXPONENT))
    return 1;

  // loop over missing input blocks
  for (int missing1 = 0; missing1 < NUM_IN; missing1++) {
    cerr << "Processing " << missing1 << endl;
    
    vector<bool> in_present;
    for (int i = 0; i < NUM_IN; i++) {
      in_present.push_back(i != missing1);
    }
    vector<bool> recovery_present;
    for (int i = 0; i < NUM_REC; i++) {
      recovery_present.push_back(true);
    }
    
    ReedSolomon<gtype> rs_repair;
    if (init_repair_rs<gtype,utype>(rs_repair, in_present, recovery_present, LOW_EXPONENT))
      return 1;

    u8 result[BUF_SIZE];
    if (recover_data<gtype,utype>(rs_repair, 0, result, data, in_present, recovery_present))
      return 1;
    if (compare_buffer(result, data[missing1], "test3 - missing1"))
      return 1;
  }

  return 0;
}
*/



// Check that the correct constants are being used for Par2
template<typename gtype, typename utype>
int test4(int NUM_IN, int *expected_bases) {
  //const int NUM_IN  = 10;
  const int NUM_REC = 1;  // recovery
  const int LOW_EXPONENT = 1;

  u8 data[NUM_IN + NUM_REC][BUF_SIZE];

  int high_exponent = LOW_EXPONENT + NUM_REC - 1;

  for (int i = 0; i < NUM_IN; i++) {
    // fill with zeros,
    for (int k = 0; k < BUF_SIZE; k++) {
      data[i][k] = (u8)0;
    }
    // EXCEPT put a 1 in a different place for each file
    ((gtype *)(&(data[i][0])))[i] = (utype) 1;
  }
  // zero recovery
  for (int j = 0; j < NUM_REC; j++) {
    for (int k = 0; k < BUF_SIZE; k++) {
      data[NUM_IN + j][k] = (u8)0;
    }
  }


  ReedSolomon<gtype> rs_creator;

  //cout << "creator.setinput" << NUM_IN << endl;
  if (!rs_creator.SetInput(NUM_IN, cout, cerr)) {
    cerr << "rs_creator.SetInput returned false";
    return 1;
  }
  //cout << "creator.setoutput" << LOW_EXPONENT << " " << high_exponent << endl;
  if (!rs_creator.SetOutput(false, LOW_EXPONENT, high_exponent)) {
    cerr << "rs_creator.SetOutput returned false";
    return 1;
  }
  //cout << "creator.compute" << endl;
  if (!rs_creator.Compute(nlSilent, cout, cerr)) {
    cerr << "rs_creator.Compute returned false";
    return 1;
  }

  for (int i = 0; i < NUM_IN; i++) {
    for (int j = 0; j < NUM_REC; j++) {
	//cout << "creator.process " << BUF_SIZE << " " << i << " " << j << endl;
	rs_creator.Process(BUF_SIZE, i, &(data[i][0]), j, &(data[NUM_IN + j][0]));
    }
  }


  // The recovery file has exponent 1 and should
  // contain each base to the power 1.
  for (int i = 0; i < NUM_IN; i++) {
    int base = (utype) ((gtype *) &(data[NUM_IN+0][0]))[i];
    if (base != expected_bases[i]) {
      cerr << "base at location " << i << " did not match expected." << endl;
      cerr << "   base     = " << base << endl;
      cerr << "   expected = " << expected_bases[i] << endl;
      return 1;
    }
  }

  return 0;
}



int main() {
  if (test1<Galois8,u8>()) {
    cerr << "FAILED: test1(8)" << endl;
    return 1;
  }
  if (test1<Galois16,u16>()) {
    cerr << "FAILED: test1(16)" << endl;
    return 1;
  }

  if (test2<Galois8,u8>()) {
    cerr << "FAILED: test2(8)" << endl;
    return 1;
  }
  if (test2<Galois16,u16>()) {
    cerr << "FAILED: test2(16)" << endl;
    return 1;
  }

  // test3 used more parity blocks than missing source blocks.
  // The code should either work or not allow it.
  // Probably not allow it.
  //if (test3<Galois8,u8>()) return 1;  cout << "finished test 3(8)" << endl;
  //if (test3<Galois16,u16>()) return 1;  cout << "finished test 3(16)" << endl;

  // the values for Par1
  int expected_bases8[10] = {1,2,3,4,5,6,7,8,9,10};
  if (test4<Galois8,u8>(10, expected_bases8)) {
    cerr << "FAILED: test4(8)" << endl;
    return 1;
  }

  // from the Par2 standard
  int expected_bases16[10] = {2, 4, 16, 128, 256, 2048, 8192, 16384, 4107, 32856};
  if (test4<Galois16,u16>(10, expected_bases16)) {
    cerr << "FAILED: test4(16)" << endl;
    return 1;
  }
  
  return 0;
}
  


