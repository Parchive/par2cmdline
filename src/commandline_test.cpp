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

#include "commandline.h"
using namespace std;

// ComputeRecoveryBlockCount
// recoveryblockset = true
int test5() {
  u32 recoveryblockcount = 42;
  bool success = CommandLine::ComputeRecoveryBlockCount(&recoveryblockcount,
							100,
							4,
							0,
							scVariable,
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
							scVariable,
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
							scVariable,
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
							scVariable,
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
							scVariable,
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
							scVariable,
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
							scUniform,
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
							scUniform,
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
							scVariable,
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
  success = ComputeRecoveryFileCount(cout, cerr,
				     &recoveryfilecount,
				     scVariable,
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


// calls Parse and expects it to fail.
// "arg" is command; spaces are used to separate argv words...
int test9_helper(const char *arg)
{
  // copy args into argc/argv format
  // buffer holds copy of arg, with ' ' replaced by '\0'
  const int len = strlen(arg);
  int argc = 0;
  char *buffer = new char[len+1];
  const char **argv = new const char*[len];
  argv[argc]=&(buffer[0]);
  argc++;
  for (int i = 0; i < len; i++) {
    buffer[i] = arg[i];
    if (buffer[i] == ' ') {
      buffer[i] = '\0';
      argv[argc] = &(buffer[i+1]);
      argc++;
    }
  }
  buffer[len] = '\0';
  
  CommandLine commandline;
  if (commandline.Parse(argc, argv)) {
    cout << "CommandLine should not have parsed: \"" << arg << "\"" << endl;
    return 1;
  }

  delete [] buffer;
  delete [] argv;
  
  return 0;
}


int test9() {
  // create input files, in case they are read.
  ofstream par2file;
  par2file.open("foo.par2");
  par2file << "dummy par2 file.  Just has to exist.";
  par2file.close();
  ofstream par2file_bar;
  par2file_bar.open("bar.par2");
  par2file_bar << "dummy par2 file.  Just has to exist.";
  par2file_bar.close();
  
  ofstream input1;
  input1.open("input1.txt");
  input1 << "commandline_test test9 input1.txt\n";
  input1.close();
  ofstream input2;
  input2.open("input2.txt");
  input2 << "commandline_test test9 input2.txt\n";
  input2.close();
  
  
  // mistaken call for help, version, etc.
  if (test9_helper("par2 -?"))
    return 1;
  if (test9_helper("par2 -v"))
    return 1;

  // missing operation
  if (test9_helper("par2"))
    return 1;
  if (test9_helper("par2 repairfoo.par"))
    return 1;
  if (test9_helper("par2 -p repairfoo.par"))
    return 1;
  if (test9_helper("par2 -b100 createfoo.par"))
    return 1;
  
  // missing parfile
  if (test9_helper("par2 repair"))
    return 1;
  if (test9_helper("par2repair"))
    return 1;
  if (test9_helper("par2 create"))
    return 1;
  if (test9_helper("par2create"))
    return 1;
  if (test9_helper("par2verify"))
    return 1;
  if (test9_helper("par2 verify"))
    return 1;
  if (test9_helper("par2 create "))
    return 1;

  // missing input files for create
  if (test9_helper("par2 create foo.par2"))
    return 1;
  
  // wrong options for the chosen operation
  if (test9_helper("par2 create -p foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -N foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -S100 foo.par2 input1.txt input2.txt"))
    return 1;
  // DO WE WANT TO ERROR OUT ON THIS CASE?
  //  if (test9_helper("par2 repair -abar.par2 foo.par2"))
  //    return 1;
  if (test9_helper("par2 repair -b100 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -s100 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -r5 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -rg5 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -c100 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -f10 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -u foo.par2"))
    return 1;
  if (test9_helper("par2 repair -l foo.par2"))
    return 1;
  if (test9_helper("par2 repair -n10 foo.par2"))
    return 1;
  if (test9_helper("par2 repair -R foo.par2"))
    return 1;

  // Should "-v" cause an error (for being in the wrong place)
  // or be treated as a filename?
  //if (test9_helper("par2 create foo.par2 input1.txt input2.txt -v"))
  //  return 1;

  // bad combinations of options
  if (test9_helper("par2 create -v -q foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -q -v foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -v -v -q foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -q -q -v foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -b100 -s100 foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -rg5 -c100 foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -u -l foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -n10 -l foo.par2 input1.txt input2.txt"))
    return 1;

  // badly formatted options
  if (test9_helper("par2 create -zzz foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create --v foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -s3 foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -m50m foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -rt5 foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -bad foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -sad foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -rad foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -cad foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -fad foo.par2 input1.txt input2.txt"))
    return 1;
  if (test9_helper("par2 create -nad foo.par2 input1.txt input2.txt"))
    return 1;


  // delete files that were created at start of test.
  remove("foo.par2");
  remove("bar.par2");
  remove("input1.txt");
  remove("input2.txt");
  return 0;
}


// calls Parse and expects it to call create
// "arg" is command; spaces are used to separate argv words...
int test10_helper(const char *arg,
		  const NoiseLevel noiselevel,
		  const size_t memorylimit,
		  const string &basepath,
#ifdef _OPENMP
		  const u32 nthreads,
		  const u32 filethreads,
#endif
		  const string &parfilename,
		  const vector<string> &extrafiles,
		  const u64 blocksize,
		  const u32 firstblock,
		  const Scheme recoveryfilescheme,
		  const u32 recoveryfilecount,
		  const u32 recoveryblockcount
		  )
{
  // copy args into argc/argv format
  // buffer holds copy of arg, with ' ' replaced by '\0'
  const int len = strlen(arg);
  int argc = 0;
  char *buffer = new char[len+1];
  const char **argv = new const char*[len];
  argv[argc]=&(buffer[0]);
  argc++;
  for (int i = 0; i < len; i++) {
    buffer[i] = arg[i];
    if (buffer[i] == ' ') {
      buffer[i] = '\0';
      argv[argc] = &(buffer[i+1]);
      argc++;
    }
  }
  buffer[len] = '\0';
  
  CommandLine commandline;
  if (!commandline.Parse(argc, argv)) {
    cout << "CommandLine said it was unable to parse \"" << arg << "\"" << endl;
    return 1;
  }

  if (commandline.GetVersion() != CommandLine::verPar2) {
    cout << "test10 fail version  arg=" << arg << endl;
    cout << commandline.GetVersion() << " != " << CommandLine::verPar2 << endl;
    return 1;
  }
  if (commandline.GetOperation() != CommandLine::opCreate) {
    cout << "test10 fail operation  arg=" << arg << endl;
    cout << commandline.GetOperation() << " != " << CommandLine::opCreate << endl;
    return 1;
  }
  
  if (commandline.GetNoiseLevel() != noiselevel) {
    cout << "test10 fail noiselevel  arg=" << arg << endl;
    cout << commandline.GetNoiseLevel() << " != " << noiselevel << endl;
    return 1;
  }
  if (commandline.GetMemoryLimit() != memorylimit) {
    cout << "test10 fail memorylimit  arg=" << arg << endl;
    cout << commandline.GetMemoryLimit() << " != " << memorylimit << endl;
    return 1;
  }
  if (commandline.GetBasePath() != basepath) {
    cout << "test10 fail basepath  arg=" << arg << endl;
    cout << commandline.GetBasePath() << " != " << basepath << endl;
    return 1;
  }
#ifdef _OPENMP
  if (commandline.GetNumThreads() != nthreads) {
    cout << "test10 fail nthreads  arg=" << arg << endl;
    cout << commandline.GetNumThreads() << " != " << nthreads << endl;
    return 1;
  }
  if (commandline.GetFileThreads() != filethreads) {
    cout << "test10 fail filethreads  arg=" << arg << endl;
    cout << commandline.GetFileThreads() << " != " << filethreads << endl;
    return 1;
  }
#endif
  if (commandline.GetParFilename() != parfilename) {
    cout << "test10 fail parfilename  arg=" << arg << endl;
    cout << commandline.GetParFilename() << " != " << parfilename << endl;
    return 1;
  }
  const vector<string> &extrafiles_returned = commandline.GetExtraFiles();
  if (extrafiles_returned.size() != extrafiles.size()) {
    cout << "test10 fail extrafiles.size()  arg=" << arg << endl;
    cout << extrafiles_returned.size() << " != " << extrafiles.size() << endl;
    return 1;
  }
  for (unsigned int i = 0; i < extrafiles.size(); i++) {
    if (extrafiles_returned[i] != extrafiles[i]) {
      cout << "test10 fail extrafiles[" << i << "]  arg=" << arg << endl;
      cout <<  extrafiles_returned[i] << " != " << extrafiles[i] << endl;
      return 1;
    }
  }
  if (commandline.GetBlockSize() != blocksize) {
    cout << "test10 fail blocksize  arg=" << arg << endl;
    cout << commandline.GetBlockSize() << " != " << blocksize << endl;
    return 1;
  }
  if (commandline.GetFirstRecoveryBlock() != firstblock) {
    cout << "test10 fail firstblock  arg=" << arg << endl;
    cout << commandline.GetFirstRecoveryBlock() << " != " << firstblock << endl;
    return 1;
  }
  if (commandline.GetRecoveryFileScheme() != recoveryfilescheme) {
    cout << "test10 fail recoveryfilescheme  arg=" << arg << endl;
    cout << commandline.GetRecoveryFileScheme() << " != " << recoveryfilescheme << endl;
    return 1;
  }
  if (commandline.GetRecoveryFileCount() != recoveryfilecount) {
    cout << "test10 fail recoveryfilecount  arg=" << arg << endl;
    cout << commandline.GetRecoveryFileCount() << " != " << recoveryfilecount << endl;
    return 1;
  }
  if (commandline.GetRecoveryBlockCount() != recoveryblockcount) {
    cout << "test10 fail recoveryblockcount  arg=" << arg << endl;
    cout << commandline.GetRecoveryBlockCount() << " != " << recoveryblockcount << endl;
    return 1;
  }


  delete [] buffer;
  delete [] argv;
  
  return 0;
}


// Test calls to "par2 create"
int test10() {
  // create input files, in case they are read.
  ofstream input1;
  input1.open("input1.txt");
  const char *input1_contents = "commandline_test test10 input1.txt\n";
  input1 << input1_contents;
  input1.close();
  ofstream input2;
  input2.open("input2.txt");
  const char *input2_contents = "commandline_test test10 input2.txt\n";
  input2 << input2_contents;
  input2.close();


  // Call once.  The results are used to initialize some defaults.
  int argc_for_defaults = 5;
  const char *argv_for_defaults[5] = {"par2", "create", "foo.par2", "input1.txt", "input2.txt"};
  CommandLine commandline_for_defaults;
  if (!commandline_for_defaults.Parse(argc_for_defaults, argv_for_defaults)) {
    cout << "CommandLine unable to fetch default values." << endl;
    return 1;
  }

  // Define the default values
  // (Using variable names makes it more obvious when
  // a non-default value is present in test.)
  const NoiseLevel default_noiselevel = nlNormal;
  const size_t default_memorylimit = commandline_for_defaults.GetMemoryLimit();
  const string &default_basepath = commandline_for_defaults.GetBasePath();
#ifdef _OPENMP
  const u32 default_nthreads = 0;
  const u32 default_filethreads = _FILE_THREADS;
#endif
  string default_parfilename = default_basepath + "foo";  // ".par2" is stripped.
  vector<string> default_extrafiles;
  default_extrafiles.push_back(default_basepath + "input1.txt");
  default_extrafiles.push_back(default_basepath + "input2.txt");
  const u64 default_blocksize = 4;  // tries to make 2000 blocks ... this is closest blocksize
  const u32 default_firstblock = 0;
  const Scheme default_recoveryfilescheme = scVariable;
  const u32 default_recoveryfilecount = 0;
  const u32 default_recoveryblockcount = 1; // tries to do 5% recovery data ... this is minimum value
  
  if (test10_helper("par2 create foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
  

  // TODO: -B option

  // -v
  if (test10_helper("par2 create -v foo.par2 input1.txt input2.txt",
		    nlNoisy,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }

  // -v -v
  if (test10_helper("par2 create -v -v foo.par2 input1.txt input2.txt",
		    nlDebug,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }

  // -q
  if (test10_helper("par2 create -q foo.par2 input1.txt input2.txt",
		    nlQuiet,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }

  // -q -q
  if (test10_helper("par2 create -q -q foo.par2 input1.txt input2.txt",
		    nlSilent,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }

  // -m option
  if (test10_helper("par2 create -m16 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    16*1024*1024,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }

#ifdef _OPENMP
  // -t option
  if (test10_helper("par2 create -t42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
		    42,
		    default_filethreads,
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }

  // -T option
  if (test10_helper("par2 create -T42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
		    default_nthreads,
		    42,
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
#endif
  
  // -- option
  if (test10_helper("par2 create -- foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
  if (test10_helper("par2 create -- -foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_basepath + "-foo",
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
  {// scope to hide variables
    ofstream dashinput2;
    dashinput2.open("-input2.txt");
    dashinput2 << "commandline_test test10 -input2.txt\n";
    dashinput2.close();

    vector<string> extrafiles;
    extrafiles.push_back(default_basepath + "input1.txt");
    extrafiles.push_back(default_basepath + "-input2.txt");
    if (test10_helper("par2 create foo.par2 input1.txt -- -input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
      return 1;
    }

    // delete file that was created.
    remove("-input2.txt");
  }
  if (test10_helper("par2 create -afoo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
  int longestfilelen = max(strlen(input1_contents), strlen(input2_contents));
  int longestfilelen_rounded_up = longestfilelen;
  if (longestfilelen_rounded_up % 4 != 0) {
    longestfilelen_rounded_up += 4 - (longestfilelen_rounded_up % 4);
  }
  if (test10_helper("par2 create -b2 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    longestfilelen_rounded_up,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
  if (test10_helper("par2 create -s8 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    8,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) {
    return 1;
  }
  int default_sourceblockcount = 0;
  default_sourceblockcount += strlen(input1_contents)/default_blocksize;
  if (strlen(input1_contents) % default_blocksize != 0)
    default_sourceblockcount++;
  default_sourceblockcount += strlen(input2_contents)/default_blocksize;
  if (strlen(input2_contents) % default_blocksize != 0)
    default_sourceblockcount++;
  if (test10_helper("par2 create -r100 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_sourceblockcount)) {   // recoverycount=sourcecount
    return 1;
  }
  const int overhead_fixed = 76;
  // per source file: main packet contents, file desc, file slices header
  const int overhead_persourcefile = 216 + 12; // 216 + filename length
  // per source block: file slices contents
  const int overhead_persourceblock = 20;
  // per recovery bloc: recovery block header
  const int overhead_perrecoveryblock = 68;
  int default_sourcefilecount = 2;
  int optionr_recoveryfilecount = 4;  // computed commandline, but not returned.
  int optionr_recoveryblockcount = 8;
  int usage_per_recoveryfile = overhead_fixed
    + default_sourcefilecount * overhead_persourcefile
    + default_sourceblockcount * overhead_persourceblock;
  int usage = optionr_recoveryfilecount * usage_per_recoveryfile
    + optionr_recoveryblockcount * (overhead_perrecoveryblock + default_blocksize);
  if (usage > 1024) {
    cout << "Test10 -rk1 should fail because usage limit was 1024 bytes.  Actual usage was " << usage << endl;
    //return 1;
  }

  if (test10_helper("par2 create -rk1 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    optionr_recoveryblockcount)) {
    return 1;
  }
  if (test10_helper("par2 create -c42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    42)) { 
    return 1;
  }
  if (test10_helper("par2 create -f42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    42,
		    default_recoveryfilescheme,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) { 
    return 1;
  }
  if (test10_helper("par2 create -u foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    scUniform,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) { 
    return 1;
  }
  if (test10_helper("par2 create -l foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    scLimited,
		    default_recoveryfilecount,
		    default_recoveryblockcount)) { 
    return 1;
  }
  if (test10_helper("par2 create -n42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    default_blocksize,
		    default_firstblock,
		    default_recoveryfilescheme,
		    42,
		    default_recoveryblockcount)) { 
    return 1;
  }


  // TODO: Test "-R" option (Recurse into subdirectories)
  
  
  // delete files that were created at start of test.
  remove("input1.txt");
  remove("input2.txt");
  return 0;
}



// calls Parse and expects it to call repair/verify
// "arg" is command; spaces are used to separate argv words...
int test11_helper(const char *arg,
		  const NoiseLevel noiselevel,
		  const size_t memorylimit,
		  const string &basepath,
#ifdef _OPENMP
		  const u32 nthreads,
		  const u32 filethreads,
#endif
		  const string &parfilename,
		  const vector<string> &extrafiles,
		  const CommandLine::Version version,
		  const CommandLine::Operation operation,
		  const bool purgefiles,
		  const bool skipdata,
		  const u64 skipleaway
		  )
{
  // copy args into argc/argv format
  // buffer holds copy of arg, with ' ' replaced by '\0'
  const int len = strlen(arg);
  int argc = 0;
  char *buffer = new char[len+1];
  const char **argv = new const char*[len];
  argv[argc]=&(buffer[0]);
  argc++;
  for (int i = 0; i < len; i++) {
    buffer[i] = arg[i];
    if (buffer[i] == ' ') {
      buffer[i] = '\0';
      argv[argc] = &(buffer[i+1]);
      argc++;
    }
  }
  buffer[len] = '\0';
  
  CommandLine commandline;
  if (!commandline.Parse(argc, argv)) {
    cout << "CommandLine said it was unable to parse \"" << arg << "\"" << endl;
    return 1;
  }

  
  if (commandline.GetVersion() != version) {
    cout << "test11 fail version  arg=" << arg << endl;
    cout << commandline.GetVersion() << " != " << version << endl;
    return 1;
  }
  if (commandline.GetOperation() != operation) {
    cout << "test11 fail operation  arg=" << arg << endl;
    cout << commandline.GetOperation() << " != " << operation << endl;
    return 1;
  }

  if (commandline.GetNoiseLevel() != noiselevel) {
    cout << "test11 fail noiselevel  arg=" << arg << endl;
    cout << commandline.GetNoiseLevel() << " != " << noiselevel << endl;
    return 1;
  }
  if (commandline.GetMemoryLimit() != memorylimit) {
    cout << "test11 fail memorylimit  arg=" << arg << endl;
    cout << commandline.GetMemoryLimit() << " != " << memorylimit << endl;
    return 1;
  }
  if (commandline.GetBasePath() != basepath) {
    cout << "test11 fail basepath  arg=" << arg << endl;
    cout << commandline.GetBasePath() << " != " << basepath << endl;
    return 1;
  }
#ifdef _OPENMP
  if (commandline.GetNumThreads() != nthreads) {
    cout << "test11 fail nthreads  arg=" << arg << endl;
    cout << commandline.GetNumThreads() << " != " << nthreads << endl;
    return 1;
  }
  if (commandline.GetFileThreads() != filethreads) {
    cout << "test11 fail filethreads  arg=" << arg << endl;
    cout << commandline.GetFileThreads() << " != " << filethreads << endl;
    return 1;
  }
#endif
  if (commandline.GetParFilename() != parfilename) {
    cout << "test11 fail parfilename  arg=" << arg << endl;
    cout << commandline.GetParFilename() << " != " << parfilename << endl;
    return 1;
  }
  const vector<string> &extrafiles_returned = commandline.GetExtraFiles();
  if (extrafiles_returned.size() != extrafiles.size()) {
    cout << "test11 fail extrafiles.size()  arg=" << arg << endl;
    cout << extrafiles_returned.size() << " != " << extrafiles.size() << endl;
    return 1;
  }
  for (unsigned int i = 0; i < extrafiles.size(); i++) {
    if (extrafiles_returned[i] != extrafiles[i]) {
      cout << "test11 fail extrafiles[" << i << "]  arg=" << arg << endl;
      cout <<  extrafiles_returned[i] << " != " << extrafiles[i] << endl;
      return 1;
    }
  }
  
  if (commandline.GetPurgeFiles() != purgefiles) {
    cout << "test11 fail purgefiles  arg=" << arg << endl;
    cout << commandline.GetPurgeFiles() << " != " << purgefiles << endl;
    return 1;
  }

  if (version == CommandLine::verPar2) {
    if (commandline.GetSkipData() != skipdata) {
      cout << "test11 fail skipdata  arg=" << arg << endl;
      cout << commandline.GetSkipData() << " != " << skipdata << endl;
      return 1;
    }
    if (commandline.GetSkipLeaway() != skipleaway) {
      cout << "test11 fail skipleaway  arg=" << arg << endl;
      cout << commandline.GetSkipLeaway() << " != " << skipleaway << endl;
      return 1;
    }
  }

  delete [] buffer;
  delete [] argv;
  
  return 0;
}



// test calls to repair/verify
int test11() {
  // create input files, in case they are read.
  ofstream par2file;
  par2file.open("foo.par2");
  par2file << "dummy par2 file.  Just has to exist.";
  par2file.close();
  
  ofstream input1;
  input1.open("input1.txt");
  const char *input1_contents = "commandline_test test11 input1.txt\n";
  input1 << input1_contents;
  input1.close();
  ofstream input2;
  input2.open("input2.txt");
  const char *input2_contents = "commandline_test test11 input2.txt\n";
  input2 << input2_contents;
  input2.close();

  // Call once.  The results are used to initialize some defaults.
  int argc_for_defaults = 5;
  const char *argv_for_defaults[5] = {"par2", "repair", "foo.par2", "input1.txt", "input2.txt"};
  CommandLine commandline_for_defaults;
  if (!commandline_for_defaults.Parse(argc_for_defaults, argv_for_defaults)) {
    cout << "CommandLine unable to fetch default values." << endl;
    return 1;
  }
  

  // Define the default values
  // (Using variable names makes it more obvious when
  // a non-default value is present in test.)
  const NoiseLevel default_noiselevel = nlNormal;
  const size_t default_memorylimit = commandline_for_defaults.GetMemoryLimit();
  const string &default_basepath = commandline_for_defaults.GetBasePath();
#ifdef _OPENMP
  const u32 default_nthreads = 0;
  const u32 default_filethreads = _FILE_THREADS;
#endif
  string default_parfilename = "foo.par2"; // relative path, par2 is NOT stripped.
  vector<string> default_extrafiles;
  default_extrafiles.push_back(default_basepath + "input1.txt");
  default_extrafiles.push_back(default_basepath + "input2.txt");
  bool default_purgefiles = false;
  bool default_skipdata = false;
  u64 default_skipleaway = 0;
  
  if (test11_helper("par2 repair foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }

  // TODO: -B option

  // -v
  if (test11_helper("par2 repair -v foo.par2 input1.txt input2.txt",
		    nlNoisy,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  if (test11_helper("par2 verify -v foo.par2 input1.txt input2.txt",
		    nlNoisy,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opVerify,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // -v -v
  if (test11_helper("par2 repair -v -v foo.par2 input1.txt input2.txt",
		    nlDebug,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // -q
  if (test11_helper("par2 repair -q foo.par2 input1.txt input2.txt",
		    nlQuiet,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // -q -q
  if (test11_helper("par2 repair -q -q foo.par2 input1.txt input2.txt",
		    nlSilent,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // -m
  if (test11_helper("par2 repair -m16 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    16*1024*1024,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
#ifdef _OPENMP
  // -t
  if (test11_helper("par2 repair -t42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
		    42,
		    default_filethreads,
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // -T
  if (test11_helper("par2 repair -T42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
		    default_nthreads,
		    42,
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
#endif
  // --
  if (test11_helper("par2 repair -- foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  {// scope to hide variables
    ofstream dashpar2file;
    dashpar2file.open("-foo.par2");
    dashpar2file << "anything\n";
    dashpar2file.close();
    
    if (test11_helper("par2 repair -- -foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    "-foo.par2",
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
      return 1;
    }

    remove("-foo.par2");
  }
  { // scope to hide variables
    ofstream dashinput2;
    dashinput2.open("-input2.txt");
    dashinput2 << "commandline_test test11 -input2.txt\n";
    dashinput2.close();
      
    vector<string> extrafiles;
    extrafiles.push_back(default_basepath + "input1.txt");
    extrafiles.push_back(default_basepath + "-input2.txt");
      
    if (test11_helper("par2 repair foo.par2 input1.txt -- -input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
      return 1;
    }

    // delete file that was created.
    remove("-input2.txt");
  }
  // -p
  if (test11_helper("par2 repair -p foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    true,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // Do we want -p with verify??
  if (test11_helper("par2 verify -p foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opVerify,
		    true,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  // -N
  if (test11_helper("par2 repair -N foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    true,
		    64) // this value should be a named constant somewhere.
      ) {
    return 1;
  }
  // -S
  if (test11_helper("par2 repair -N -S42 foo.par2 input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    default_parfilename,
		    default_extrafiles,
		    CommandLine::verPar2,
		    CommandLine::opRepair,
		    default_purgefiles,
		    true,
		    42)) {
    return 1;
  }
  
  // remove par2 file
  remove("foo.par2");

  // Create par1 file.  Do a few tests.
  //
  ofstream par1file;
  par1file.open("bar.par");
  par1file << "dummy par1 file.  Just has to exist.";
  par1file.close();

  if (test11_helper("par2 repair bar.par input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    "bar.par",
		    default_extrafiles,
		    CommandLine::verPar1,
		    CommandLine::opRepair,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }
  if (test11_helper("par2 verify bar.par input1.txt input2.txt",
		    default_noiselevel,
		    default_memorylimit,
		    default_basepath,
#ifdef _OPENMP
		    default_nthreads,
		    default_filethreads,
#endif
		    "bar.par",
		    default_extrafiles,
		    CommandLine::verPar1,
		    CommandLine::opVerify,
		    default_purgefiles,
		    default_skipdata,
		    default_skipleaway)) {
    return 1;
  }

  
  remove("bar.par");
  remove("input1.txt");
  remove("input2.txt");
  return 0;
}

// test calls to print help, version
int test12() {
  int argc_for_help = 2;
  const char *argv_for_help[5] = {"par2", "--help"};
  CommandLine commandline_for_help;
  if (!commandline_for_help.Parse(argc_for_help, argv_for_help)) {
    cout << "CommandLine failed for help" << endl;
    return 1;
  }

  int argc_for_h = 2;
  const char *argv_for_h[5] = {"par2", "-h"};
  CommandLine commandline_for_h;
  if (!commandline_for_h.Parse(argc_for_h, argv_for_h)) {
    cout << "CommandLine failed for h." << endl;
    return 1;
  }

  int argc_for_help2 = 2;
  const char *argv_for_help2[5] = {"par2create", "--help"};
  CommandLine commandline_for_help2;
  if (!commandline_for_help2.Parse(argc_for_help2, argv_for_help2)) {
    cout << "CommandLine failed for help (par2create)." << endl;
    return 1;
  }

  int argc_for_version = 2;
  const char *argv_for_version[5] = {"par2", "--version"};
  CommandLine commandline_for_version;
  if (!commandline_for_version.Parse(argc_for_version, argv_for_version)) {
    cout << "CommandLine failed for version." << endl;
    return 1;
  }

  int argc_for_V = 2;
  const char *argv_for_V[5] = {"par2", "-V"};
  CommandLine commandline_for_V;
  if (!commandline_for_V.Parse(argc_for_V, argv_for_V)) {
    cout << "CommandLine failed for v." << endl;
    return 1;
  }

  int argc_for_VV = 2;
  const char *argv_for_VV[5] = {"par2", "-VV"};
  CommandLine commandline_for_VV;
  if (!commandline_for_VV.Parse(argc_for_VV, argv_for_VV)) {
    cout << "CommandLine failed for v." << endl;
    return 1;
  }

  int argc_for_version2 = 2;
  const char *argv_for_version2[5] = {"par2create", "--version"};
  CommandLine commandline_for_version2;
  if (!commandline_for_version2.Parse(argc_for_version2, argv_for_version2)) {
    cout << "CommandLine failed for version (par2create)." << endl;
    return 1;
  }

  
  return 0;
}


int main() {
  cout << "Tests 1 through 4 were moved to libpar2_test." << endl;


  if (test5()) {
    cerr << "FAILED: test5" << endl;
    return 1;
  }
  if (test6()) {
    cerr << "FAILED: test6" << endl;
    return 1;
  }
  if (test7()) {
    cerr << "FAILED: test7" << endl;
    return 1;
  }
  if (test8()) {
    cerr << "FAILED: test8" << endl;
    return 1;
  }

  if (test9()) {
    cerr << "FAILED: test9" << endl;
    return 1;
  }
  if (test10()) {
    cerr << "FAILED: test10" << endl;
    return 1;
  }
  if (test11()) {
    cerr << "FAILED: test11" << endl;
    return 1;
  }
  if (test12()) {
    cerr << "FAILED: test12" << endl;
    return 1;
  }
  
  cout << "SUCCESS: commandline_test complete." << endl;
  
  return 0;
}
