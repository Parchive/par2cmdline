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
#include "letype.h"


// test one value
int test1() {
  u16 expected = 127*127;
  leu16 x;
  x = expected;
  u16 output = x;
  
  if (output != expected) {
    cerr << "output = " << output << endl;
    cerr << "expected = " << expected << endl;
    return 1;
  }
  
  return 0;
}


// test all 16 bit values
int test2() {
  for (int i = 0; i < 256*256; i++) {
    u16 expected = i;
    leu16 x;
    x = expected;
    u16 output = x;
    
    if (output != expected) {
      cerr << "output = " << output << endl;
      cerr << "expected = " << expected << endl;
      return 1;
    }
  }
  
  return 0;
}


// test one 32-bit value
int test3() {
  u32 expected = 127*127*127*127; 
  leu32 x;
  x = expected;
  u32 output = x;
  
  if (output != expected) {
    cerr << "output = " << output << endl;
    cerr << "expected = " << expected << endl;
    return 1;
  }
  
  return 0;
}


// test random 32-bit values
int test4() {
  srand(113450911);

  for (int i = 0; i < 256*256; i++) {
    unsigned long z = 0;
    z += (rand() % 256)*256*256*256;
    z += (rand() % 256)*256*256;
    z += (rand() % 256)*256;
    z += (rand() % 256);

    u32 expected = z; 
    leu32 x;
    x = expected;
    u32 output = x;
    
    if (output != expected) {
      cerr << "output = " << output << endl;
      cerr << "expected = " << expected << endl;
      return 1;
    }
  }
  
  return 0;
}


// test one 64-bit value
int test5() {
  u64 expected = 127ul*127ul*127ul*127ul*127ul*127ul*127ul*127ul; 
  leu64 x;
  x = expected;
  u64 output = x;
  
  if (output != expected) {
    cerr << "output = " << output << endl;
    cerr << "expected = " << expected << endl;
    return 1;
  }
  
  return 0;
}


// test random 64-bit values
int test6() {
  srand(84395311);

  for (int i = 0; i < 256*256; i++) {
    unsigned long z = 0;
    unsigned long factor = 1;
    for (int j = 0; j < 8; j++) {
      z += (rand() % 256)*factor;
      factor *= 256;
    }
    
    u64 expected = z; 
    leu64 x;
    x = expected;
    u64 output = x;
    
    if (output != expected) {
      cerr << "output = " << output << endl;
      cerr << "expected = " << expected << endl;
      return 1;
    }
  }
  
  return 0;
}


int main() {
  if (test1()) {
    cerr << "FAILED: test1" << endl;
    return 1;
  }
  if (test2()) {
    cerr << "FAILED: test2" << endl;
    return 1;
  }
  if (test3()) {
    cerr << "FAILED: test3" << endl;
    return 1;
  }
  if (test4()) {
    cerr << "FAILED: test4" << endl;
    return 1;
  }
  if (test5()) {
    cerr << "FAILED: test5" << endl;
    return 1;
  }
  if (test6()) {
    cerr << "FAILED: test6" << endl;
    return 1;
  }

  cout << "SUCCESS: letype_test complete." << endl;
  
  return 0;
}
  
