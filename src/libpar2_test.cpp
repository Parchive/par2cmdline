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
#include <fstream>
#include <stdlib.h>

using namespace std;

#include "libpar2.h"


// ComputeRecoveryFileCount
// check when it returns false.
int test1() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scUnknown,
						       1,
						       1,
						       1);
  if (success) {
    cerr << "ComputeRecoveryFileCount worked for unknown scheme" << endl;
    return 1;
  }

  recoveryfilecount = 10;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scVariable,
						       4,
						       4,
						       4);
  if (success) {
    cerr << "ComputeRecoveryFileCount worked with more files than blocks!" << endl;
    return 1;
  }
  
  return 0;
}


// ComputeRecoveryFileCount
// scVariable
int test2() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scVariable,
						       0,
						       4,
						       4);
  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 0) {
    cerr << "ComputeRecoveryFileCount for 0 blocks should return 0" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scVariable,
						       8,
						       4,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    cerr << "ComputeRecoveryFileCount for 8 blocks should return 4" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scVariable,
						       15,
						       4,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    cerr << "ComputeRecoveryFileCount for 15 blocks should return 4" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scVariable,
						       64,
						       4,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    cerr << "ComputeRecoveryFileCount for 64 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scVariable,
						       127,
						       4,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    
  
  return 0;
}

// ComputeRecoveryFileCount
// scUniform
// Doesn't matter the value - long as it's zero at zero and positive after.
int test3() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scUniform,
						       0,
						       4,
						       4);
  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 0) {
    cerr << "ComputeRecoveryFileCount for 0 blocks should return 0" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scUniform,
						       1,
						       4,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount == 0) {
    cerr << "ComputeRecoveryFileCount for 1 block should a positive value" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scUniform,
						       8,
						       4,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount == 0) {
    cerr << "ComputeRecoveryFileCount for 8 blocks should a positive value" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  return 0;
}


// ComputeRecoveryFileCount
// scLimited
// Same as variable with big files
// But differs for smaller ones.
int test4() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       0,
						       4096,
						       4);
  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 0) {
    cerr << "ComputeRecoveryFileCount for 0 blocks should return 0" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       8,
						       4096,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    cerr << "ComputeRecoveryFileCount for 8 blocks should return 4" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       15,
						       4096,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    cerr << "ComputeRecoveryFileCount for 15 blocks should return 4" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       64,
						       4096,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    cerr << "ComputeRecoveryFileCount for 64 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       127,
						       4096,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    


  // smaller largest files
  // 1 2 4 8 10 10 10...
  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       8,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       15,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       16,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 5) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       25,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 5) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    
  
  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       26,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 6) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       35,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 6) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
  }    


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(cout, cerr,
					  &recoveryfilecount,
						       scLimited,
						       35 + 100,
						       40,
						       4);

  if (!success) {
    cerr << "ComputeRecoveryFileCount failed test2.1" << endl;
    return 1;
  }
  if (recoveryfilecount != 6 + 10) {
    cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << endl;
    cerr << "   it returned " << recoveryfilecount << endl;
    return 1;
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

  cout << "SUCCESS: libpar2_test complete." << endl;
  
  return 0;
}
