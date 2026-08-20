//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#ifndef __LIBPAR2_H__
#define __LIBPAR2_H__

#include <array>
#include <ostream>
#include <string>
#include <vector>

#include <par2/types.h>
#include <par2/processor.h>
#include <par2/hasher.h>
#include <par2/backends.h>

namespace par2
{


typedef enum
{
  scUnknown = 0,
  scVariable,      // Each PAR2 file will have 2x as many blocks as previous
  scLimited,       // Limit PAR2 file size
  scUniform        // All PAR2 files the same size
} Scheme;


// How much logging/status information to write
// to output or error stream
typedef enum
{
  nlUnknown = 0,
  nlSilent,       // Absolutely no output (other than errors)
  nlQuiet,        // Bare minimum of output
  nlNormal,       // Normal level of output
  nlNoisy,        // Lots of output
  nlDebug         // Extra debugging information
} NoiseLevel;


// Return type of par2cmdline
typedef enum Result
{
  eSuccess                     = 0,

  eRepairPossible              = 1,  // Data files are damaged and there is
                                     // enough recovery data available to
                                     // repair them.

  eRepairNotPossible           = 2,  // Data files are damaged and there is
                                     // insufficient recovery data available
                                     // to be able to repair them.

  eInvalidCommandLineArguments = 3,  // There was something wrong with the
                                     // command line arguments

  eInsufficientCriticalData    = 4,  // The PAR2 files did not contain sufficient
                                     // information about the data files to be able
                                     // to verify them.

  eRepairFailed                = 5,  // Repair completed but the data files
                                     // still appear to be damaged.


  eFileIOError                 = 6,  // An error occurred when accessing files
  eLogicError                  = 7,  // In internal error occurred
  eMemoryError                 = 8,  // Out of memory

} Result;


// What a PAR2 set describes, known once its packets have been loaded
struct Par2SetInfo
{
  std::array<u8, 16> setid;     // The recovery set id, an MD5 in the order its
                                // bytes are stored
  u64 blocksize;                // Size of each block
  u32 datablocks;               // Number of blocks in the recovery set
  u32 recoverablefilecount;     // Files that can be repaired
  u32 otherfilecount;           // Files described but not recoverable
  u64 datasize;                 // Total size of the recoverable files
};


// One of the files a PAR2 set describes
struct Par2FileInfo
{
  std::string filename;         // The name the set records for the file
  u64 filesize;                 // Size of the file
  u32 blockcount;               // Blocks the file is divided into, 0 if it
                                // cannot be recovered
};


// Receives progress and per-file results from a par2 operation.
//
// Every method has an empty default, so an implementation only overrides what
// it needs. The methods are called from whichever thread is doing the work,
// which may be one of several worker threads, so they must be thread safe.
// They are not affected by the NoiseLevel, which only controls what is written
// to the output stream.
class Par2Observer
{
public:
  virtual ~Par2Observer() {}

  // The recovery set has been identified
  virtual void OnSetInfo(const Par2SetInfo &info) {}

  // Work has started on a file: one the set describes, or a PAR2 file being
  // read. Each is followed by an OnFileDone.
  //
  // filename is the name the set records, which is the same on every system.
  // A file the set does not name - a PAR2 file, or an extra file offered to a
  // verify - is named as it is on this one, and keeps that name for both
  // reports.
  virtual void OnFile(const std::string &filename) {}

  // Progress through the current operation, in thousandths, running upwards
  // once per Verify and once per phase of a Repair - the rebuild, and then the
  // pass reading back what it wrote. AddPar2File reports no progress.
  virtual void OnProgress(u32 permille) {}

  // This file has been checked. blocksfound of blocksneeded were usable, both
  // zero for a PAR2 file, which has no blocks of its own to account for.
  virtual void OnFileDone(const std::string &filename,
                          u32 blocksfound,
                          u32 blocksneeded) {}

  // Repair is about to start
  virtual void OnRepairStart(void) {}
};


Result par2create(std::ostream &sout,
			  std::ostream &serr,
			  const NoiseLevel noiselevel,
			  const size_t memorylimit,
			  const std::string &basepath,
			  const u32 nthreads,
			  const u32 filethreads,
			  const std::string &parfilename,
			  const std::vector<std::string> &extrafiles,
			  const u64 blocksize,
			  const u32 firstblock,
			  const Scheme recoveryfilescheme,
			  const u32 recoveryfilecount,
			  const u32 recoveryblockcount,
			  const Backends &backends = Backends()
			  );


Result par2repair(std::ostream &sout,
		  std::ostream &serr,
		  const NoiseLevel noiselevel,
		  const size_t memorylimit,
		  const std::string &basepath,
		  const u32 nthreads,
		  const u32 filethreads,
		  const std::string &parfilename,
		  const std::vector<std::string> &extrafiles,
		  const bool dorepair,   // derived from operation
		  const bool purgefiles,
		  const bool renameonly,
		  const bool skipdata,
		  const u64 skipleaway,
		  const bool fullhash = false,
		  const Backends &backends = Backends()
		  );


Result par1repair(std::ostream &sout,
		  std::ostream &serr,
		  const NoiseLevel noiselevel,
		  const size_t memorylimit,
		  // basepath is not used by Par1
		  const u32 nthreads,
		  // filethreads is not used by Par1
		  const std::string &parfilename,
		  const std::vector<std::string> &extrafiles,
		  const bool dorepair,   // derived from operation
		  const bool purgefiles
		  // skipdata is not used by Par1
		  // skipleaway is not used by Par1
		  );


bool ComputeRecoveryFileCount(std::ostream &sout,
			      std::ostream &serr,
			      u32 *recoveryfilecount,
			      Scheme recoveryfilescheme,
			      u32 recoveryblockcount,
			      u64 largestfilesize,
			      u64 blocksize);

} // namespace par2

#endif // __LIBPAR2_H__
