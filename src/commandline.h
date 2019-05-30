//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#ifndef __COMMANDLINE_H__
#define __COMMANDLINE_H__

#include <string>
using std::string;

// This is needed by diskfile.h
#ifdef _WIN32
#include <windows.h>

// Heap checking
#ifdef _MSC_VER
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, THIS_FILE, __LINE__)
#endif

#define stricmp  _stricmp

#else

#include <string.h>
#define stricmp strcasecmp

#endif

#define _FILE_THREADS 2

#include "libpar2.h"
#include "diskfile.h"

// The CommandLine object is responsible for understanding the format
// of the command line parameters are parsing the command line to
// extract details as to what the user wants to do.

class CommandLine
{
public:
  CommandLine(void);

  // Parse the supplied command line arguments.
  bool Parse(int argc, const char * const *argv);


  static void showversion(void);
  static void banner(void);
  // Display details of the correct format for command line parameters.
  static void usage(void);

  // What operation will we be carrying out
  typedef enum
  {
    opNone = 0,
    opCreate,        // Create new PAR2 recovery volumes
    opVerify,        // Verify but don't repair damaged data files
    opRepair         // Verify and if possible repair damaged data files
  } Operation;

  typedef enum
  {
    verUnknown = 0,
    verPar1,         // Processing PAR 1.0 files
    verPar2          // Processing PAR 2.0 files
  } Version;

public:
  // Accessor functions for the command line parameters
  CommandLine::Operation GetOperation(void) const          {return operation;}
  CommandLine::Version   GetVersion(void) const            {return version;}
  u64                    GetBlockSize(void) const          {return blocksize;}
  u32                    GetFirstRecoveryBlock(void) const {return firstblock;}
  u32                    GetRecoveryFileCount(void) const  {return recoveryfilecount;}
  u32                    GetRecoveryBlockCount(void) const {return recoveryblockcount;}
  Scheme    GetRecoveryFileScheme(void) const {return recoveryfilescheme;}
  size_t                 GetMemoryLimit(void) const        {return memorylimit;}
  NoiseLevel GetNoiseLevel(void) const        {return noiselevel;}

  string                              GetParFilename(void) const {return parfilename;}
  string                              GetBasePath(void) const    {return basepath;}
  const vector<string>& GetExtraFiles(void) const  {return extrafiles;}
  bool                                GetPurgeFiles(void) const  {return purgefiles;}
  bool                                GetRecursive(void) const   {return recursive;}
  bool                                GetSkipData(void) const    {return skipdata;}
  u64                                 GetSkipLeaway(void) const  {return skipleaway;}
#ifdef _OPENMP
  u32                          GetNumThreads(void) {return nthreads;}
  u32                          GetFileThreads(void) {return filethreads;}
#endif


  static bool ComputeRecoveryBlockCount(u32 *recoveryblockcount,
					u32 sourceblockcount,
					u64 blocksize,
					u32 firstblock,
					Scheme recoveryfilescheme,
					u32 recoveryfilecount,
					bool recoveryblockcountset,
					u32 redundancy,
					u64 redundancysize,
					u64 largestfilesize);

protected:
  // Read the text of arguments into the class's variables
  bool ReadArgs(int argc, const char * const *argv);

  // Returns the memory on the system in BYTES
  // (or 0 if it cannot be determined)
  u64 GetTotalPhysicalMemory();

  // Check values that were set during ReadArgs.
  // If values went unset, set them with default values
  bool CheckValuesAndSetDefaults();

  // Use values like block count to compute the block size
  bool ComputeBlockSize();

  // Use values like % recovery data to compute the number of recover blocks
  bool ComputeRecoveryBlockCount();
  
  bool                         SetParFilename(string filename);

  FileSizeCache filesize_cache;// Caches the size of each file,
                               // to prevent multiple calls to OS.
  
  // options for all operations
  Version version;             // What version files will be processed.
  NoiseLevel noiselevel;       // How much display output should there be.
  size_t memorylimit;          // How much memory is permitted to be used
                               // for the output buffer when creating
                               // or repairing.
  string basepath;             // the path par2 is run from
#ifdef _OPENMP
  u32 nthreads;         // Default number of threads
  u32 filethreads;      // Number of threads for file processing
#endif
  // NOTE: using the "-t" option to set the number of threads does not
  // end up here, but results in a direct call to "omp_set_num_threads"

  string parfilename;          // The name of the PAR2 file to create, or
                               // the name of the first PAR2 file to read
                               // when verifying or repairing.

  list<string> rawfilenames;   // The filenames on command-line
                               // (after expanding wildcards like '*')

  vector<string> extrafiles;   // The filenames that will be used by Par.
                               // These have been verified to exist,
                               // have a path-name relative to the
                               // basepath, etc.. When creating, these will be
                               // the source files, and when verifying or
                               // repairing, this will be additional PAR2
                               // files or data files to be examined.

  // which operation
  Operation operation;         // The operation to be carried out.

  // options for verify/repair operation
  bool purgefiles;             // purge backup and par files on success
                               // recovery
  bool skipdata;               // Whether we should assume that all good
                               // data blocks are within +/- bytes of
                               // where we expect to find them and should
                               // skip data that is too far away.
  u64 skipleaway;              // The maximum leaway +/- that we will
                               // allow when searching for blocks.

  
  // options for creating par files  
  u32 blockcount;              // How many blocks the source files should
                               // be virtually split into.
  u64 blocksize;               // What virtual block size to use.

  u32 firstblock;              // What the exponent value for the first
                               // recovery block will be.

  Scheme recoveryfilescheme;   // How the size of the recovery files should
                               // be calculated.

  u32 recoveryfilecount;       // How many recovery files should be created.

  u32 recoveryblockcount;      // How many recovery blocks should be created.
  bool recoveryblockcountset;  // Set if the recoveryblockcount as been specified

  u32 redundancy;              // What percentage of recovery data should
                               // be created.
  u64 redundancysize;          // target filesize of recovery files

  bool redundancyset;          // Set if the redundancy has been specified

  bool recursive;              // recurse into subdirectories

};

#endif // __COMMANDLINE_H__
