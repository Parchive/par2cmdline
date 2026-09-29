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
#include <sstream>
#include <stdlib.h>



#include <par2/libpar2.h>

using namespace par2;


// ComputeRecoveryFileCount
// check when it returns false.
int test1() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scUnknown,
						       1,
						       1,
						       1);
  if (success) {
    std::cerr << "ComputeRecoveryFileCount worked for unknown scheme" << std::endl;
    return 1;
  }

  recoveryfilecount = 10;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scVariable,
						       4,
						       4,
						       4);
  if (success) {
    std::cerr << "ComputeRecoveryFileCount worked with more files than blocks!" << std::endl;
    return 1;
  }

  return 0;
}


// ComputeRecoveryFileCount
// scVariable
int test2() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scVariable,
						       0,
						       4,
						       4);
  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 0) {
    std::cerr << "ComputeRecoveryFileCount for 0 blocks should return 0" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scVariable,
						       8,
						       4,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    std::cerr << "ComputeRecoveryFileCount for 8 blocks should return 4" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scVariable,
						       15,
						       4,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    std::cerr << "ComputeRecoveryFileCount for 15 blocks should return 4" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scVariable,
						       64,
						       4,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    std::cerr << "ComputeRecoveryFileCount for 64 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scVariable,
						       127,
						       4,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  return 0;
}

// ComputeRecoveryFileCount
// scUniform
// Doesn't matter the value - long as it's zero at zero and positive after.
int test3() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scUniform,
						       0,
						       4,
						       4);
  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 0) {
    std::cerr << "ComputeRecoveryFileCount for 0 blocks should return 0" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scUniform,
						       1,
						       4,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount == 0) {
    std::cerr << "ComputeRecoveryFileCount for 1 block should a positive value" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scUniform,
						       8,
						       4,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount == 0) {
    std::cerr << "ComputeRecoveryFileCount for 8 blocks should a positive value" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
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
  bool success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       0,
						       4096,
						       4);
  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 0) {
    std::cerr << "ComputeRecoveryFileCount for 0 blocks should return 0" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       8,
						       4096,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    std::cerr << "ComputeRecoveryFileCount for 8 blocks should return 4" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       15,
						       4096,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    std::cerr << "ComputeRecoveryFileCount for 15 blocks should return 4" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       64,
						       4096,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    std::cerr << "ComputeRecoveryFileCount for 64 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       127,
						       4096,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 7) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }


  // smaller largest files
  // 1 2 4 8 10 10 10...
  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       8,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       15,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 4) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       16,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 5) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       25,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 5) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       26,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 6) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }

  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       35,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 6) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }


  recoveryfilecount = 0;
  success = ComputeRecoveryFileCount(std::cout, std::cerr,
					  &recoveryfilecount,
						       scLimited,
						       35 + 100,
						       40,
						       4);

  if (!success) {
    std::cerr << "ComputeRecoveryFileCount failed test2.1" << std::endl;
    return 1;
  }
  if (recoveryfilecount != 6 + 10) {
    std::cerr << "ComputeRecoveryFileCount for 127 blocks should return 7" << std::endl;
    std::cerr << "   it returned " << recoveryfilecount << std::endl;
    return 1;
  }


  return 0;
}


// A memory limit below 1MB still creates and repairs
int test5() {
  const char *const datafile = "libpar2_test5.data";
  const char *const leftovers[] = {datafile, "libpar2_test5.data.1", "libpar2_test5.par2",
				   "libpar2_test5.vol0+1.par2", "libpar2_test5.vol1+2.par2",
				   "libpar2_test5.vol3+1.par2"};

  // A create never overwrites, so whatever an earlier run left goes first
  for (const char *leftover : leftovers)
    remove(leftover);

  {
    std::ofstream data(datafile, std::ofstream::out | std::ofstream::binary);
    for (int i = 0; i < 20000; ++i)
      data.put((char)(i * 31 + i / 256));
  }

  std::ostringstream quiet;
  const std::vector<std::string> files(1, datafile);
  const std::vector<std::string> none;

  Result result = par2create(quiet, quiet, nlSilent, 1, "", 0, 0,
			     "libpar2_test5", files, 4096, 0, scVariable, 0, 4);
  if (result != eSuccess) {
    std::cerr << "par2create with a one byte memory limit returned " << result << std::endl;
  } else {
    {
      std::fstream data(datafile, std::ios::binary | std::ios::in | std::ios::out);
      data.seekp(5000);
      for (int i = 0; i < 3000; ++i)
        data.put((char)0x5a);
    }

    result = par2repair(quiet, quiet, nlSilent, 1, "", 0, 0,
			"libpar2_test5.par2", none, true, true, false, false, 0);
    if (result != eSuccess)
      std::cerr << "par2repair with a one byte memory limit returned " << result << std::endl;
  }

  for (const char *leftover : leftovers)
    remove(leftover);

  return result == eSuccess ? 0 : 1;
}


int main() {
  if (test1()) {
    std::cerr << "FAILED: test1" << std::endl;
    return 1;
  }
  if (test2()) {
    std::cerr << "FAILED: test2" << std::endl;
    return 1;
  }
  if (test3()) {
    std::cerr << "FAILED: test3" << std::endl;
    return 1;
  }
  if (test4()) {
    std::cerr << "FAILED: test4" << std::endl;
    return 1;
  }
  if (test5()) {
    std::cerr << "FAILED: test5" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: libpar2_test complete." << std::endl;

  return 0;
}
