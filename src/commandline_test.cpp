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

#include "par2cmdline.h"
#include "commandline.h"

// ComputeRecoveryFileCount
// check when it returns false.
int test1() {
  u32 recoveryfilecount = 0;
  bool success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scUnknown,
						       1,
						       1,
						       1);
  if (success) {
    cerr << "ComputeRecoveryFileCount worked for unknown scheme" << endl;
    return 1;
  }

  recoveryfilecount = 10;
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scVariable,
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
  bool success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scVariable,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scVariable,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scVariable,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scVariable,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scVariable,
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
  bool success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scUniform,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scUniform,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scUniform,
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
  bool success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						       CommandLine::scLimited,
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


// ComputeRecoveryBlockCount
// recoveryblockset = true
int test5() {
  u32 recoveryblockcount = 42;
  bool success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							100,
							4,
							0,
							CommandLine::scVariable,
							0,
							true,
							0,
							0,
							40);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test5.1" << endl;
    return 1;
  }
  if (recoveryblockcount != 42) {
    cerr << "ComputeRecoveryBlockCount should not overwrite recoveryblockcount" << endl;
    cerr << "   it returned " << recoveryblockcount << endl;
    return 1;
  }    


  recoveryblockcount = 66000;
  success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							100,
							4,
							0,
							CommandLine::scVariable,
							0,
							true,
							0,
							0,
							40);
  if (success) {
    cerr << "ComputeRecoveryBlockCount should fail for too many blocks" << endl;
    return 1;
  }


  
  recoveryblockcount = 6000;
  success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							100,
							4,
							60000,
							CommandLine::scVariable,
							0,
							true,
							0,
							0,
							40);
  if (success) {
    cerr << "ComputeRecoveryBlockCount should fail for too high a coefficient" << endl;
    return 1;
  }

  return 0;
}

// ComputeRecoveryBlockCount
// redundancy > 0
int test6() {
  u32 recoveryblockcount = 0;
  bool success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							100,
							4,
							0,
							CommandLine::scVariable,
							0,
							false,
							1,
							0,
							40);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test5.1" << endl;
    return 1;
  }
  if (recoveryblockcount != 1) {
    cerr << "ComputeRecoveryBlockCount 1% of 100 is 1" << endl;
    cerr << "   it returned " << recoveryblockcount << endl;
    return 1;
  }    

  recoveryblockcount = 0;
  success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							1000,
							4,
							0,
							CommandLine::scVariable,
							0,
							false,
							5,
							0,
							40);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test5.1" << endl;
    return 1;
  }
  if (recoveryblockcount != 50) {
    cerr << "ComputeRecoveryBlockCount 5% of 1000 is 50" << endl;
    cerr << "   it returned " << recoveryblockcount << endl;
    return 1;
  }    


  recoveryblockcount = 0;
  success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							10,
							4,
							0,
							CommandLine::scVariable,
							0,
							false,
							1,
							0,
							40);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test5.1" << endl;
    return 1;
  }
  if (recoveryblockcount != 1) {
    cerr << "ComputeRecoveryBlockCount 1% of 10 is still positive" << endl;
    cerr << "   it returned " << recoveryblockcount << endl;
    return 1;
  }    

  return 0;
}

// ComputeRecoveryBlockCount
// redundnacysize > 0
int test7_helper(int sourcefilecount, // not used by ComputeRecoveryBlockCount!
		 int sourceblockcount,
		 int blocksize,
		 int redundancysize,
		 int recoveryfilecount)
{  
  // overhead from packet headers, formatting, etc.
  // fixed: main packet header, main packet contents
  const int overhead_fixed = 76;
  // per source file: main packet contents, file desc, file slices header
  const int overhead_persourcefile = 236; // 216 + filename length
  // per source block: file slices contents
  const int overhead_persourceblock = 20;
  // per recovery bloc: recovery block header
  const int overhead_perrecoveryblock = 68; 

  u32 recoveryblockcount;
  bool success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							sourceblockcount,
							blocksize,
							0,
							CommandLine::scUniform,
							recoveryfilecount,
							false,
							0,
							redundancysize,
							blocksize);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test5.1" << endl;
    return 1;
  }

  int usage_per_recoveryfile = overhead_fixed
    + sourcefilecount * overhead_persourcefile
    + sourceblockcount * overhead_persourceblock;
  int usage = recoveryfilecount * usage_per_recoveryfile
    + recoveryblockcount * (overhead_perrecoveryblock + blocksize);
  if (usage <= redundancysize - (overhead_perrecoveryblock + blocksize) ||
      usage > redundancysize) {
    cerr << "ComputeRecoveryBlockCount " << redundancysize << " data limit" << endl;
    cerr << "   but usage was " << usage << " with " << recoveryblockcount << " blocks" << endl;
    cerr << "        sourcefilecount=" << sourcefilecount << endl;
    cerr << "        sourceblockcount=" << sourceblockcount << endl;
    cerr << "        blocksize=" << blocksize << endl;
    cerr << "        recoveryfilecount=" << recoveryfilecount << endl;
    return 2;
  }    
  
  return 0;
}

// ComputeRecoveryBlockCount
// redundnacysize > 0
// scUniform with number of recovery files already determined.
int test7() {
  // CD is 10 files, 600 source blocks, 1MB block size
  // Redundancys is 40MB in 5 files.
  
  //sourcefilecount, sourceblockcount, blocksize, redundancysize, recoveryfilecount
  if (test7_helper(10, 600, 1024*1024, 40*1024*1024, 5))
    return 1;

  if (test7_helper( 1, 600, 1024*1024, 40*1024*1024, 5))
    return 1;

  if (test7_helper(10,  60, 1024*1024, 40*1024*1024, 5))
    return 1;

  if (test7_helper(10, 600,   16*1024, 40*1024*1024, 5))
    return 1;

  if (test7_helper(10, 600, 1024*1024, 10*1024*1024, 5))
    return 1;

  if (test7_helper(10, 600, 1024*1024, 40*1024*1024, 2))
    return 1;


  // DVD is 1 files, 5000 source blocks, 1MB block size
  // Redundancys is 50MB in 5 files.
  if (test7_helper(1, 5000, 1024*1024, 50*1024*1024, 5) == 1)
    return 1;
  

  // if redundancy size is too small, still have 1 block
  u32 recoveryblockcount;
  bool success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							1000, 
							1024,
							0,
							CommandLine::scUniform,
							1,
							false,
							0,
							4,  // = redundancysize
							1024);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test5.1" << endl;
    return 1;
  }
  if (recoveryblockcount != 1) {
    cerr << "ComputeRecoveryBlockCount with small redundancy amount should still return 1" << endl;
    cerr << "   it returned " << recoveryblockcount << endl;
    return 1;
  }    
  
  return 0;
}



// ComputeRecoveryBlockCount
// redundnacysize > 0
int test8_helper(int sourcefilecount, // not used by ComputeRecoveryBlockCount!
		 int sourceblockcount,
		 int blocksize,
		 int redundancysize)
{  
  // overhead from packet headers, formatting, etc.
  // fixed: main packet header, main packet contents
  const int overhead_fixed = 76;
  // per source file: main packet contents, file desc, file slices header
  const int overhead_persourcefile = 236; // 216 + filename length
  // per source block: file slices contents
  const int overhead_persourceblock = 20;
  // per recovery bloc: recovery block header
  const int overhead_perrecoveryblock = 68; 

  u32 recoveryblockcount;
  bool success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							sourceblockcount,
							blocksize,
							0,
							CommandLine::scVariable,
							0, // =recoveryfilecount
							false,
							0,
							redundancysize,
							blocksize);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test8.1" << endl;
    return 1;
  }

  u32 recoveryfilecount = 0;
  success = CommandLine::ComputeRecoveryFileCount(&recoveryfilecount,
						  CommandLine::scVariable,
						  recoveryblockcount,
						  blocksize,
						  blocksize);
  if (!success) {
    cerr << "ComputeRecoveryBlockCount failed test8.2" << endl;
    return 1;
  }
  
  int usage_per_recoveryfile = overhead_fixed
    + sourcefilecount * overhead_persourcefile
    + sourceblockcount * overhead_persourceblock;
  int usage = recoveryfilecount * usage_per_recoveryfile
    + recoveryblockcount * (overhead_perrecoveryblock + blocksize);
  if (usage <= redundancysize - (overhead_perrecoveryblock + blocksize) ||
      usage > redundancysize) {
    cerr << "ComputeRecoveryBlockCount " << redundancysize << " data limit" << endl;
    cerr << "   but usage was " << usage << " with " << recoveryblockcount << " blocks" << endl;
    cerr << "        sourcefilecount=" << sourcefilecount << endl;
    cerr << "        sourceblockcount=" << sourceblockcount << endl;
    cerr << "        blocksize=" << blocksize << endl;
    cerr << "        recoveryfilecount=" << recoveryfilecount << endl;
    return 2;
  }    
  
  return 0;
}


// ComputeRecoveryBlockCount
// redundnacysize > 0
// scVariable with number of recovery files undetermined
int test8() {
  //sourcefilecount, sourceblockcount, blocksize, redundancysize
  if (test8_helper(10, 600, 1024*1024, 40*1024*1024))
    return 1;

  if (test8_helper( 1, 600, 1024*1024, 40*1024*1024))
    return 1;

  if (test8_helper(10,  60, 1024*1024, 40*1024*1024))
    return 1;

  //
  // WARNING: THIS TEST USUALLY FAILS BY A SMALL AMOUNT
  //    The ==1 at the end will make the test fail
  //     if the function doesn't return any value.
  // 
  if (test8_helper(10, 600,   16*1024, 40*1024*1024) == 1)
    return 1;

  if (test8_helper(10, 600, 1024*1024, 10*1024*1024))
    return 1;


  if (test8_helper(1, 5000, 1024*1024, 50*1024*1024))
    return 1;
  
  return 0;
}


int main() {
  if (test1())
    return 1;
  cout << "finished test 1" << endl;
  if (test2())
    return 1;
  cout << "finished test 2" << endl;
  if (test3())
    return 1;
  cout << "finished test 3" << endl;
  if (test4())
    return 1;
  cout << "finished test 4" << endl;
  if (test5())
    return 1;
  cout << "finished test 5" << endl;
  if (test6())
    return 1;
  cout << "finished test 6" << endl;
  if (test7())
    return 1;
  cout << "finished test 7" << endl;
  if (test8())
    return 1;
  cout << "finished test 8" << endl;
  
  return 0;
}
