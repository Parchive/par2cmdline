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
#include <iterator>
#include <sstream>
#include <stdlib.h>



#include "libpar2internal.h"

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

  Par2Creator creator("");
  creator.SetSourceFiles(std::vector<std::string>(1, datafile));
  creator.SetBlockSize(4096);
  creator.SetRecoveryBlockCount(4);
  creator.SetMemoryLimit(1);

  Result result = creator.Create("libpar2_test5");
  if (result != eSuccess) {
    std::cerr << "creating with a one byte memory limit returned " << result << std::endl;
  } else {
    {
      std::fstream data(datafile, std::ios::binary | std::ios::in | std::ios::out);
      data.seekp(5000);
      for (int i = 0; i < 3000; ++i)
	data.put((char)0x5a);
    }

    Par2Verifier verifier("");
    verifier.SetMemoryLimit(1);

    result = verifier.AddPar2File("libpar2_test5.par2");
    if (result == eSuccess)
      result = verifier.Verify();
    if (result == eRepairPossible)
      result = verifier.Repair();
    if (result != eSuccess)
      std::cerr << "repairing with a one byte memory limit returned " << result << std::endl;
  }

  for (const char *leftover : leftovers)
    remove(leftover);

  return result == eSuccess ? 0 : 1;
}


// A set which names one file on disk twice says so, rather than racing two
// threads to claim it
int test6() {
  const char *const datafile = "libpar2_test6.data";
  const char *const leftovers[] = {datafile, "libpar2_test6.par2",
				   "libpar2_test6.vol0+1.par2", "libpar2_test6.vol1+2.par2",
				   "libpar2_test6.vol3+1.par2"};

  // A create never overwrites, so whatever an earlier run left goes first
  for (const char *leftover : leftovers)
    remove(leftover);

  {
    std::ofstream data(datafile, std::ofstream::out | std::ofstream::binary);
    for (int i = 0; i < 20000; ++i)
      data.put((char)(i * 17 + i / 256));
  }

  // Counts the errors, which arrive from the threads doing the work
  class Errors : public Par2Observer
  {
  public:
    Errors(void) : count(0) {}
    void OnError(const Par2Error &) override { ++count; }
    std::atomic<int> count;
  };

  // A Par2Creator drops a name given twice, and the engine behind it does not
  const std::vector<std::string> files(2, datafile);

  int failed = 1;
  std::ostringstream quiet;
  Par2SetCreator creator(quiet, quiet, nlSilent);
  Result result = creator.Process(MemoryLimit(0), "", 0, 0,
				  "libpar2_test6", files, 4096, 0, scVariable, 0, 4);
  if (result != eSuccess) {
    std::cerr << "creating a set naming one file twice returned " << result << std::endl;
  } else {
    Errors errors;
    Par2Verifier verifier("");
    verifier.SetObserver(&errors);

    Par2SetInfo info;
    std::vector<Par2FileInfo> infos;
    Par2Error error;
    if (eSuccess != verifier.AddPar2File("libpar2_test6.par2")
	|| !verifier.GetSetInfo(&info) || info.recoverablefilecount != 2) {
      std::cerr << "the set does not describe the file twice" << std::endl;
    } else if ((result = verifier.Verify()) != eFileIOError) {
      std::cerr << "a set which names one file twice returned " << result << std::endl;
    } else if (!verifier.GetLastError(&error) || error.code != ecDuplicateSourceFile) {
      std::cerr << "the reason is not ecDuplicateSourceFile" << std::endl;
    } else if (!verifier.GetFileInfo(&infos) || infos.empty()
	       || error.filename != infos[0].localfilename) {
      std::cerr << "the error does not name the file both entries point at" << std::endl;
    } else if (errors.count != 1) {
      std::cerr << "the duplicate was reported " << errors.count << " times" << std::endl;
    } else {
      failed = 0;
    }
  }

  for (const char *leftover : leftovers)
    remove(leftover);

  return failed;
}


// A recovery block of the wrong size is left out of the set, and the observer
// is told so
int test7() {
  const char *const datafile = "libpar2_test7.data";
  const char *const shortvolume = "libpar2_test7.vol7+1.par2";
  const char *const leftovers[] = {datafile, "libpar2_test7.par2",
				   "libpar2_test7.vol0+1.par2", shortvolume};

  // A create never overwrites, so whatever an earlier run left goes first
  for (const char *leftover : leftovers)
    remove(leftover);

  {
    std::ofstream data(datafile, std::ofstream::out | std::ofstream::binary);
    for (int i = 0; i < 20000; ++i)
      data.put((char)(i * 13 + i / 256));
  }

  // Keeps the warnings, which arrive from the threads doing the work
  class Warnings : public Par2Observer
  {
  public:
    void OnWarning(const Par2Warning &warning) override
    {
      std::lock_guard<std::mutex> lock(mutex);
      warnings.push_back(warning);
    }
    std::mutex mutex;
    std::vector<Par2Warning> warnings;
  };

  int failed = 1;

  Par2Creator creator("");
  creator.SetSourceFiles(std::vector<std::string>(1, datafile));
  creator.SetBlockSize(4096);
  creator.SetRecoveryBlockCount(1);
  creator.SetRecoveryFileScheme(scUniform, 1);

  Result result = creator.Create("libpar2_test7");
  if (result != eSuccess) {
    std::cerr << "creating the set returned " << result << std::endl;
  } else {
    std::ifstream volume("libpar2_test7.vol0+1.par2", std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(volume)), std::istreambuf_iterator<char>());

    const std::string type((const char*)&recoveryblockpacket_type, sizeof(recoveryblockpacket_type));
    const size_t at = bytes.find(type);

    if (at == std::string::npos || at < offsetof(PACKET_HEADER, type)) {
      std::cerr << "the volume holds no recovery block" << std::endl;
    } else {
      // The same block four bytes short, under an exponent of its own
      const size_t start = at - offsetof(PACKET_HEADER, type);
      const u64 length = ((const PACKET_HEADER*)&bytes[start])->length;

      std::string packet = bytes.substr(start, (size_t)length - 4);
      RECOVERYBLOCKPACKET *block = (RECOVERYBLOCKPACKET*)&packet[0];
      block->header.length = length - 4;
      block->exponent = 7;

      MD5Context context;
      context.Update(&packet[offsetof(PACKET_HEADER, setid)], packet.size() - offsetof(PACKET_HEADER, setid));
      context.Final(block->header.hash);

      std::ofstream(shortvolume, std::ofstream::out | std::ofstream::binary) << packet;

      Warnings observer;
      Par2Verifier verifier("");
      verifier.SetObserver(&observer);

      Par2SetInfo info;
      if (eSuccess != verifier.AddPar2File("libpar2_test7.par2") || !verifier.GetSetInfo(&info)) {
	std::cerr << "the set could not be read" << std::endl;
      } else if (info.recoveryblockcount != 1) {
	std::cerr << "the set has " << info.recoveryblockcount << " recovery blocks" << std::endl;
      } else if (observer.warnings.size() != 1 || observer.warnings[0].code != wcPacketDiscarded
		 || observer.warnings[0].message.find("exponent 7") == std::string::npos) {
	std::cerr << "the short block was not reported as discarded" << std::endl;
      } else {
	failed = 0;
      }
    }
  }

  for (const char *leftover : leftovers)
    remove(leftover);

  return failed;
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
  if (test6()) {
    std::cerr << "FAILED: test6" << std::endl;
    return 1;
  }
  if (test7()) {
    std::cerr << "FAILED: test7" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: libpar2_test complete." << std::endl;

  return 0;
}
