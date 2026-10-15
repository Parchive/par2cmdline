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
#include <stdlib.h>



#include "libpar2internal.h"

using namespace par2;


// ComputeRecoveryFileCount
// check when it returns false.
int test1() {
  u32 recoveryfilecount = 0;
  bool success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  bool success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  bool success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  bool success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  success = ComputeRecoveryFileCount(0,
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
  Par2SetCreator creator;
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


// A PAR2 file read alone leaves the volumes beside it unread until they are
// asked for
int test8() {
  const char *const datafile = "libpar2_test8.data";
  const char *const leftovers[] = {datafile, "libpar2_test8.par2",
				   "libpar2_test8.vol0+1.par2", "libpar2_test8.vol1+2.par2"};

  // A create never overwrites, so whatever an earlier run left goes first
  for (const char *leftover : leftovers)
    remove(leftover);

  {
    std::ofstream data(datafile, std::ofstream::out | std::ofstream::binary);
    for (int i = 0; i < 20000; ++i)
      data.put((char)(i * 11 + i / 256));
  }

  // Counts the PAR2 files read
  class Loads : public Par2Observer
  {
  public:
    Loads(void) : count(0) {}
    void OnFile(Phase phase, const std::string &) override { if (phase == phLoading) ++count; }
    std::atomic<int> count;
  };

  int failed = 1;

  Par2Creator creator("");
  creator.SetSourceFiles(std::vector<std::string>(1, datafile));
  creator.SetBlockSize(4096);
  creator.SetRecoveryBlockCount(3);
  creator.SetRecoveryFileScheme(scVariable, 2);

  Result result = creator.Create("libpar2_test8");
  if (result != eSuccess) {
    std::cerr << "creating the set returned " << result << std::endl;
  } else {
    Loads loads;
    Par2Verifier verifier("");
    verifier.SetObserver(&loads);

    Par2SetInfo info;
    if (eSuccess != verifier.AddPar2File("libpar2_test8.vol1+2.par2", false)
	|| !verifier.GetSetInfo(&info)) {
      std::cerr << "the volume read alone does not describe the set" << std::endl;
    } else if (loads.count != 1 || info.recoveryblockcount != 2) {
      std::cerr << "reading one volume read " << loads.count << " files and "
		<< info.recoveryblockcount << " recovery blocks" << std::endl;
    } else if (eSuccess != verifier.AddPar2File("libpar2_test8.vol1+2.par2", false) || loads.count != 1) {
      std::cerr << "reading the volume alone again read " << loads.count << " files" << std::endl;
    } else if (eSuccess != verifier.AddPar2File("libpar2_test8.vol1+2.par2")
	       || !verifier.GetSetInfo(&info)) {
      std::cerr << "the volumes beside it could not be read" << std::endl;
    } else if (loads.count != 3 || info.recoveryblockcount != 3) {
      std::cerr << "asking for the volumes read " << loads.count << " files and "
		<< info.recoveryblockcount << " recovery blocks" << std::endl;
    } else {
      failed = 0;
    }
  }

  for (const char *leftover : leftovers)
    remove(leftover);

  return failed;
}


// The file id of each file description packet of a PAR2 file
static std::vector<MD5Hash> DescribedFileIds(const char *par2file) {
  std::ifstream file(par2file, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::vector<MD5Hash> fileids;
  for (size_t offset = 0; offset + sizeof(PACKET_HEADER) <= bytes.size(); ) {
    const PACKET_HEADER *header = (const PACKET_HEADER*)&bytes[offset];
    if (header->type == filedescriptionpacket_type)
      fileids.push_back(((const FILEDESCRIPTIONPACKET*)header)->fileid);
    offset += (size_t)header->length;
  }
  return fileids;
}

// The unicode filename packets of a PAR2 file, as the file id and the name
// each gives
static std::vector<std::pair<MD5Hash, std::string> > UnicodeNames(const char *par2file) {
  std::ifstream file(par2file, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::vector<std::pair<MD5Hash, std::string> > names;
  for (size_t offset = 0; offset + sizeof(PACKET_HEADER) <= bytes.size(); ) {
    const PACKET_HEADER *header = (const PACKET_HEADER*)&bytes[offset];
    if (header->type == unicodefilenamepacket_type) {
      const UNICODEFILENAMEPACKET *packet = (const UNICODEFILENAMEPACKET*)header;
      size_t count = ((size_t)header->length - sizeof(UNICODEFILENAMEPACKET)) / 2;
      while (count > 0 && packet->name[2*count-2] == 0 && packet->name[2*count-1] == 0)
        --count;
      std::string name;
      UnicodeFilenamePacket::Utf16ToUtf8(packet->name, count, name);
      names.push_back(std::make_pair(packet->fileid, name));
    }
    offset += (size_t)header->length;
  }
  return names;
}

// A create gives a name which is not plain ASCII in a unicode filename packet
// too, and only such a name
int test9() {
  const std::string unicodedata = "libpar2_test9_caf\xc3\xa9_\xf0\x9f\x98\x80.data";
  const char *const asciidata = "libpar2_test9.data";
  const char *const leftovers[] = {"libpar2_test9u.par2", "libpar2_test9u.vol0+1.par2",
                                   "libpar2_test9a.par2", "libpar2_test9a.vol0+1.par2"};

  // DiskFile takes the names in UTF-8 on every system
  auto removeleftovers = [&]() {
    for (const char *leftover : leftovers)
      remove(leftover);
    for (const std::string &name : {unicodedata, std::string(asciidata)}) {
      DiskFile file;
      if (file.Open(name)) {
        file.Close();
        file.Delete();
      }
    }
  };

  removeleftovers();

  for (const std::string &name : {unicodedata, std::string(asciidata)}) {
    std::string data;
    for (int i = 0; i < 20000; ++i)
      data += (char)(i * 7 + i / 256);
    DiskFile file;
    if (!file.Create(name, data.size()) || !file.Write(0, data.data(), data.size())) {
      std::cerr << "could not write " << name << std::endl;
      return 1;
    }
  }

  int failed = 1;

  Par2Creator unicodecreator("");
  unicodecreator.SetSourceFiles(std::vector<std::string>(1, unicodedata));
  unicodecreator.SetBlockSize(4096);
  unicodecreator.SetRecoveryBlockCount(1);

  Par2Creator asciicreator("");
  asciicreator.SetSourceFiles(std::vector<std::string>(1, asciidata));
  asciicreator.SetBlockSize(4096);
  asciicreator.SetRecoveryBlockCount(1);

  if (unicodecreator.Create("libpar2_test9u") != eSuccess
      || asciicreator.Create("libpar2_test9a") != eSuccess) {
    std::cerr << "creating the sets failed" << std::endl;
  } else {
    const std::vector<std::pair<MD5Hash, std::string> > unicode = UnicodeNames("libpar2_test9u.par2");

    Par2Verifier verifier("");
    std::vector<Par2FileInfo> files;

    if (unicode.size() != 1 || unicode[0].second != unicodedata) {
      std::cerr << "the set does not give the name in one unicode filename packet" << std::endl;
    } else if (DescribedFileIds("libpar2_test9u.par2") != std::vector<MD5Hash>(1, unicode[0].first)) {
      std::cerr << "the unicode filename packet is not for the file described" << std::endl;
    } else if (!UnicodeNames("libpar2_test9a.par2").empty()) {
      std::cerr << "a set of a plain ASCII name has a unicode filename packet" << std::endl;
    } else if (verifier.AddPar2File("libpar2_test9u.par2") != eSuccess
               || !verifier.GetFileInfo(&files) || files.size() != 1
               || files[0].filename != unicodedata) {
      std::cerr << "the set does not describe the file by its name" << std::endl;
    } else if (verifier.Verify() != eSuccess) {
      std::cerr << "the file is not found under its name" << std::endl;
    } else {
      failed = 0;
    }
  }

  removeleftovers();

  return failed;
}


static void RemoveTestDirectory(const std::string &path) {
#ifdef _WIN32
  std::wstring wpath;
  if (utf8::Utf8ToWide(path, wpath))
    ::RemoveDirectoryW(wpath.c_str());
#else
  rmdir(path.c_str());
#endif
}

// A complete copy outside the base path is copied into place, and left where
// it is
int test10() {
  const std::string inside = "runlibpar2_test10_a";
  const std::string outside = "runlibpar2_test10_b";
  const std::string datafile = inside + PATHSEP "file.data";
  const std::string copy = outside + PATHSEP "file.data";
  const std::string leftovers[] = {datafile, copy, inside + PATHSEP "file.par2",
				   inside + PATHSEP "file.vol0+1.par2"};

  // A create never overwrites, so whatever an earlier run left goes first
  for (const std::string &leftover : leftovers)
    remove(leftover.c_str());

  std::string content;
  for (int i = 0; i < 20000; ++i)
    content.push_back((char)(i * 13 + i / 256));

  DiskFile directories(0);
  directories.CreateParentDirectory(datafile);
  directories.CreateParentDirectory(copy);
  {
    std::ofstream data(datafile.c_str(), std::ofstream::out | std::ofstream::binary);
    data << content;
  }

  int failed = 1;

  Par2Creator creator("");
  creator.SetSourceFiles(std::vector<std::string>(1, datafile));
  creator.SetBlockSize(4096);
  creator.SetRecoveryBlockCount(1);

  Result result = creator.Create(inside + PATHSEP "file");
  if (result != eSuccess) {
    std::cerr << "creating the set returned " << result << std::endl;
  } else if (0 != rename(datafile.c_str(), copy.c_str())) {
    std::cerr << "the file could not be moved out of the base path" << std::endl;
  } else {
    Par2Verifier verifier("");

    if (eSuccess != (result = verifier.AddPar2File(inside + PATHSEP "file.par2"))) {
      std::cerr << "reading the set returned " << result << std::endl;
    } else if (eRepairPossible != (result = verifier.Verify(std::vector<std::string>(1, copy)))) {
      std::cerr << "verifying with the copy outside the base path returned " << result << std::endl;
    } else if (eSuccess != (result = verifier.Repair())) {
      std::cerr << "repairing from the copy outside the base path returned " << result << std::endl;
    } else {
      std::ifstream repaired(datafile.c_str(), std::ifstream::in | std::ifstream::binary);
      std::ifstream left(copy.c_str(), std::ifstream::in | std::ifstream::binary);
      const std::string repairedcontent((std::istreambuf_iterator<char>(repaired)), std::istreambuf_iterator<char>());
      const std::string leftcontent((std::istreambuf_iterator<char>(left)), std::istreambuf_iterator<char>());

      if (repairedcontent != content) {
	std::cerr << "the file in the base path was not rebuilt" << std::endl;
      } else if (leftcontent != content) {
	std::cerr << "the copy outside the base path was moved" << std::endl;
      } else {
	failed = 0;
      }
    }
  }

  for (const std::string &leftover : leftovers)
    remove(leftover.c_str());
  RemoveTestDirectory(inside);
  RemoveTestDirectory(outside);

  return failed;
}

// A hasher which counts the blocks it is asked to check
class CountingHasher : public ReferenceHasher
{
public:
  explicit CountingHasher(std::atomic<u32> *_checked) : checked(_checked) {}

  void CheckBlocks(const void *data, u32 count, size_t filelength,
                   const void *expected, char *matched) override
  {
    *checked += count;
    ReferenceHasher::CheckBlocks(data, count, filelength, expected, matched);
  }

private:
  std::atomic<u32> *checked;
};

// A file repaired in place is verified by reading back only the blocks the
// repair wrote
int test11() {
  const char *const datafile = "libpar2_test11.data";
  const char *const leftovers[] = {datafile, "libpar2_test11.data.1", "libpar2_test11.par2",
                                   "libpar2_test11.vol0+1.par2", "libpar2_test11.vol1+2.par2",
                                   "libpar2_test11.vol3+1.par2"};

  for (const char *leftover : leftovers)
    remove(leftover);

  std::vector<char> original;
  for (int i = 0; i < 20 * 4096; ++i)
    original.push_back((char)(i * 31 + i / 256 + i / 4099));

  {
    std::ofstream data(datafile, std::ofstream::out | std::ofstream::binary);
    data.write(original.data(), original.size());
  }

  int failed = 0;

  Par2Creator creator("");
  creator.SetSourceFiles(std::vector<std::string>(1, datafile));
  creator.SetBlockSize(4096);
  creator.SetRecoveryBlockCount(4);

  class Written : public Par2Observer
  {
  public:
    Written(void) : warnings(0) {}
    void OnFileDone(Phase phase, const Par2FileResult &result) override
    {
      if (phase != phWriting)
        return;
      std::lock_guard<std::mutex> lock(mutex);
      files.push_back(result);
    }
    void OnWarning(const Par2Warning &) override { ++warnings; }
    std::mutex mutex;
    std::vector<Par2FileResult> files;
    std::atomic<int> warnings;
  };

  Result result = creator.Create("libpar2_test11");
  if (result != eSuccess) {
    std::cerr << "creating returned " << result << std::endl;
    failed = 1;
  } else {
    {
      std::fstream data(datafile, std::ios::binary | std::ios::in | std::ios::out);
      data.seekp(5 * 4096 + 100);
      for (int i = 0; i < 4; ++i)
        data.put((char)0x5a);
    }

    std::atomic<u32> checked(0);

    Backends backends;
    backends.hasher = [&checked](const HasherConfig &) {
      return std::unique_ptr<Hasher>(new CountingHasher(&checked));
    };

    Written written;
    Par2Verifier verifier("", backends);
    verifier.SetInPlaceRepair(true);
    verifier.SetObserver(&written);

    result = verifier.AddPar2File("libpar2_test11.par2");
    if (result == eSuccess)
      result = verifier.Verify();

    if (result != eRepairPossible) {
      std::cerr << "verifying the damaged file returned " << result << std::endl;
      failed = 1;
    } else {
      checked = 0;

      result = verifier.Repair();
      if (result != eSuccess) {
        std::cerr << "repairing in place returned " << result << std::endl;
        failed = 1;
      } else if (checked != 1) {
        std::cerr << "verifying the repair checked " << checked << " blocks rather than 1" << std::endl;
        failed = 1;
      }

      if (written.files.size() != 1 || !written.files[0].inplace || written.files[0].blocksfound != 1) {
        std::cerr << "the file written was not reported as repaired in place, with 1 block rewritten" << std::endl;
        failed = 1;
      }

      if (written.warnings != 0) {
        std::cerr << "repairing in place gave " << written.warnings << " warnings" << std::endl;
        failed = 1;
      }

      std::ifstream backup("libpar2_test11.data.1");
      if (backup.good()) {
        std::cerr << "repairing in place left a backup" << std::endl;
        failed = 1;
      }

      std::ifstream data(datafile, std::ifstream::in | std::ifstream::binary);
      std::vector<char> repaired((std::istreambuf_iterator<char>(data)), std::istreambuf_iterator<char>());
      if (repaired != original) {
        std::cerr << "the file repaired in place does not match the original" << std::endl;
        failed = 1;
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
  if (test8()) {
    std::cerr << "FAILED: test8" << std::endl;
    return 1;
  }
  if (test9()) {
    std::cerr << "FAILED: test9" << std::endl;
    return 1;
  }
  if (test10()) {
    std::cerr << "FAILED: test10" << std::endl;
    return 1;
  }
  if (test11()) {
    std::cerr << "FAILED: test11" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: libpar2_test complete." << std::endl;

  return 0;
}
