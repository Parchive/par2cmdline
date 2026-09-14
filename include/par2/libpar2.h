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
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
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

  eCancelled                   = 9,  // The operation was cancelled by the
                                     // caller before it completed

} Result;


// Why an operation failed, refining the Result it returned.
//
// Every Result which reports a failure carries one, and eSuccess and
// eCancelled never do. The Results which report damage - eRepairPossible,
// eRepairNotPossible and eRepairFailed - are described by GetVerifyResult, and
// carry one only when something went wrong with a particular file on the way
// there, such as a data file which could not be read.
typedef enum ErrorCode
{
  ecNone = 0,                 // Nothing failed

  // The application asked for something in an order that cannot be honoured
  ecNotVerified,              // Repair or Reassess before anything was verified

  // The PAR2 files
  ecPar2FileMissing,          // The named PAR2 file is not there, and the files
                              // named after it carried nothing new either
  ecMainPacketMissing,        // Nothing read so far says what the set contains

  // What the set describes
  ecFileDescriptionMissing,   // The set names a recoverable file it carries no
                              // description of
  ecDuplicateSourceFile,      // Two of the set's files are one file on disk
  ecTooManySourceBlocks,      // The set needs more blocks than can be held

  // Reading and writing
  ecFileOpenFailed,
  ecFileCreateFailed,
  ecFileRenameFailed,
  ecFileReadFailed,
  ecFileWriteFailed,

  ecOutOfMemory,              // A buffer could not be allocated
  ecProcessorFailed,          // The compute implementation could not do the work

  ecInternalError,            // An invariant the library relies on did not hold

} ErrorCode;


// Why an operation failed.
struct Par2Error
{
  ErrorCode code;         // ecNone when nothing failed
  std::string message;    // One line, without a trailing newline. May be empty.
  std::string filename;   // The file it concerns, empty when it concerns none
};


// What a PAR2 set describes, known once its packets have been loaded
struct Par2SetInfo
{
  std::array<u8, 16> setid;     // The recovery set id, an MD5 in the order its
                                // bytes are stored
  u64 blocksize;                // Size of each block
  u32 datablocks;               // Number of blocks in the recovery set
  u32 recoveryblocks;           // Recovery blocks read so far, which each
                                // AddPar2File may add to
  u32 recoverablefilecount;     // Files that can be repaired
  u32 otherfilecount;           // Files described but not recoverable
  u64 datasize;                 // Total size of the recoverable files
};


// One of the files a PAR2 set describes
struct Par2FileInfo
{
  std::string filename;         // The name the set records for the file
  std::string localfilename;    // Where that file belongs on this system
  u64 filesize;                 // Size of the file
  u32 blockcount;               // Blocks the file is divided into, 0 if it
                                // cannot be recovered
  std::array<u8, 16> hashfull;  // MD5 of the whole file, in the order its bytes
                                // are stored
  std::array<u8, 16> hash16k;   // MD5 of its first 16k, which is what
                                // identifies a file whose name is unknown
};

// Both names are safe to use as they stand: an absolute path or one climbing
// out with ".." is defused before either is reported, so neither needs
// sanitising again.
//
// Either may still contain a directory separator, since a set may describe
// files in subdirectories. localfilename is not necessarily the basepath and
// one path component, and the last component of filename is not necessarily
// unique within the set.
//
// localfilename is the basepath followed by filename, absolute, and available
// as soon as the packets describing the file have been read.


// What a verify found, and what it would take to repair it
struct Par2VerifyResult
{
  u32 completefilecount;        // Files that are intact
  u32 renamedfilecount;         // Files that are intact under another name
  u32 damagedfilecount;         // Files that exist but are damaged
  u32 missingfilecount;         // Files that are not there at all
  u32 availableblockcount;      // Data blocks found
  u32 missingblockcount;        // Data blocks that would have to be rebuilt
  u32 recoveryblockcount;       // Recovery blocks available to rebuild them
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

  // The recovery set has been identified. A Create only knows this once it has
  // read every source file, so it arrives near the end rather than the start.
  virtual void OnSetInfo(const Par2SetInfo &info) {}

  // Work has started on a file: one the set describes, or a PAR2 file being
  // read. Each is followed by an OnFileDone.
  //
  // filename is the name the set records, which is the same on every system.
  // A file the set does not name - a PAR2 file, or an extra file offered to a
  // verify - is named as it is on this one, and keeps that name for both
  // reports.
  //
  // Several files are read at once, so a Create's pairs interleave, and the
  // order they arrive in is not the order the set ends up recording them in.
  virtual void OnFile(const std::string &filename) {}

  // Progress through the current operation, in thousandths, running upwards
  // once per Verify and once per phase of a Repair - the rebuild, and then the
  // pass reading back what it wrote. AddPar2File reports no progress.
  //
  // A Create runs it twice, once while the source files are hashed and once
  // while the recovery data is computed, and only once when it was asked for
  // no recovery blocks at all.
  virtual void OnProgress(u32 permille) {}

  // This file has been checked. blocksfound of blocksneeded were usable, both
  // zero for a PAR2 file, which has no blocks of its own to account for.
  virtual void OnFileDone(const std::string &filename,
                          u32 blocksfound,
                          u32 blocksneeded) {}

  // Repair is about to start
  virtual void OnRepairStart(void) {}

  // Something went wrong. Called once per error as it is found, from the
  // thread that found it, so a caller wanting every error rather than only the
  // first collects them here. The operation may carry on and may still
  // succeed: an error reading one extra file does not fail a verify.
  virtual void OnError(const Par2Error &error) {}
};


// Verifies and repairs one PAR2 set.
//
// PAR2 files are added one at a time, so a caller which is still collecting
// them can add each as it arrives and ask what the set now describes. Several
// of these may be used at once, as long as each is only used from one thread at
// a time.
//
// Verify and Repair are also available as the par2repair function below, which
// does the whole job in one call.
class Par2Verifier
{
public:
  // basepath is the directory the set's files live in, and the one a repair
  // writes to. It is the -B option of the tool. A separator is appended if it
  // is missing.
  //
  // Left empty it is taken from the first PAR2 file added. Pass "." for the
  // working directory.
  //
  // backends holds the implementations the application supplies, each of which
  // falls back to the one built in when it is left empty.
  Par2Verifier(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
               const std::string &basepath = std::string(),
               const Backends &backends = Backends());
  ~Par2Verifier();

  Par2Verifier(const Par2Verifier &) = delete;
  Par2Verifier &operator=(const Par2Verifier &) = delete;

  // Notify this observer of progress and per-file results. Pass 0 to stop.
  // The observer must outlive this object.
  void SetObserver(Par2Observer *observer);

  // Read the packets of a PAR2 file and of the other PAR2 files named after
  // it. May be called more than once; a file which has already been read is
  // skipped.
  //
  // The name may be that of a file or of a whole set: the volume files beside
  // it are read too, and they carry the critical packets, so naming a set
  // whose index file is absent still describes it. eFileIOError therefore
  // means the named file does not exist *and* nothing new was read.
  //
  // Adding a file after Verify has run is allowed: the next Verify starts a
  // fresh pass over the data, so it reflects both the added file and whatever
  // is on disk at that point.
  Result AddPar2File(const std::string &parfilename);

  // What the packets added so far describe. False until a PAR2 file with the
  // critical packets has been added.
  bool GetSetInfo(Par2SetInfo *info) const;
  bool GetFileInfo(std::vector<Par2FileInfo> *files) const;

  // The CRC32 the set records for each block of the named file, one entry per
  // block starting at block 0. The name is the filename field of Par2FileInfo.
  //
  // False when the set does not describe that file, or describes it without a
  // verification packet, which is a file it cannot recover.
  bool GetBlockChecksums(const std::string &filename,
                         std::vector<u32> *crcs) const;

  // Which blocks of the named file the last verify found in that file at their
  // own offsets, one entry per block starting at block 0. The name is the
  // filename field of Par2FileInfo, and the entries line up one for one with the
  // checksums GetBlockChecksums returns and with what SetKnownBlocks takes, so
  // what this reports may be handed straight back to another verifier.
  //
  // This is what the file itself holds, not what a repair has available. A
  // block which is not here may still have been found somewhere else - in
  // another file of the set, or at a shifted offset in this one - and
  // GetVerifyResult counts those as available. Summing these will therefore
  // not always reach availableblockcount.
  //
  // A block the caller vouched for through SetKnownBlocks reads back as found,
  // since the verify made no distinction.
  //
  // Reads the same as GetVerifyResult does around a repair: after one which
  // read back what it wrote it describes the repaired files, and after one
  // which did not it still describes the state before the repair.
  //
  // False until something has been verified, when the set does not describe
  // that file, or when it describes it without a verification packet, which
  // is a file it cannot recover.
  bool GetFoundBlocks(const std::string &filename,
                      std::vector<bool> *blocks) const;

  // Accept the caller's word that these blocks of the named file are intact,
  // so that they are not read and hashed again. The name is the one the set
  // records, which is the filename field of Par2FileInfo and is the same on
  // every system, and blocks must have one entry per block of that file, set
  // where the block is present at its expected offset.
  //
  // An entry set for every block means the file is intact and it is never
  // read. None set means it holds nothing usable, and it is not read either.
  // An empty vector forgets what was said about the file.
  //
  // False when the set is already known and does not describe a file of that
  // name, or describes it with a different number of blocks, in which case
  // nothing is remembered. Before a PAR2 file has been added there is nothing
  // to check the name against and it is taken as given.
  //
  // The blocks are trusted without being verified. Supplying a block which is
  // not intact will silently produce incorrect output. Vouching for only some
  // of a file's blocks leaves it reported as needing repair.
  bool SetKnownBlocks(const std::string &filename, const std::vector<bool> &blocks);

  // Memory in bytes that Repair may use for its buffers, the -m option, which
  // the command line takes in megabytes. Zero selects the default.
  void SetMemoryLimit(const size_t memorylimit);

  // Look for blocks which are not where the set says they should be, the -N
  // option, with leaway the distance either side to search, the -S option.
  // Zero leaway selects the default. Applies to Verify and to VerifyFile.
  void SetDataSkipping(const bool enabled, const u64 leaway = 0);

  // Threads for the main processing and for hashing files in parallel, the -t
  // and -T options. Either left zero stays at the default. They are read by
  // the next Verify, VerifyFile or Repair.
  void SetThreadCounts(const u32 nthreads, const u32 filethreads);

  // Check the files described by the set against the data on disk. Returns
  // eSuccess if they are all intact, eRepairPossible if they are not but
  // enough recovery data is available, or eRepairNotPossible if it is not.
  //
  // May be called more than once; each call is a fresh pass. Repair works on
  // the results of the Verify that preceded it, so call them in that order.
  Result Verify(const std::vector<std::string> &extrafiles);

  // Scan one file that has become available, matching it against the set the
  // way a Verify would, without reading anything else. Use it to feed files in
  // as they arrive rather than waiting for all of them.
  //
  // Safe to call again for the same file: whatever an earlier scan found for it
  // is discarded first, so a file which was incomplete when it was first
  // scanned can be scanned again once it is finished. Blocks another file
  // supplied are left alone.
  //
  // The name may be one the set describes or one it does not; an unrecognised
  // file is matched by content, as an extra file is.
  //
  // Returns what Verify would return for the set as it stands, so a file which
  // has not been scanned yet still counts as missing. A later Verify replaces
  // everything the individual scans found.
  //
  // May be called before any PAR2 file has been added: the result is then
  // eInsufficientCriticalData, and the file is scanned once one arrives.
  Result VerifyFile(const std::string &filename);

  // The numbers behind the last Verify or Reassess. A repair is possible when
  // recoveryblockcount is at least missingblockcount, and needs
  // missingblockcount - recoveryblockcount more blocks when it is not.
  // False until something has been verified.
  bool GetVerifyResult(Par2VerifyResult *result) const;

  // The damaged files a repair renamed out of the way, which is what par2's
  // own purge option deletes. An application tidying up after a repair can
  // delete or keep them as it prefers.
  //
  // Only files par2 renamed itself are listed. A file the application supplied
  // as an extra file is never included, even if its blocks were used, because
  // the application knows what it supplied and may still want it.
  bool GetBackupFiles(std::vector<std::string> *files) const;

  // The files a verify found under a name other than the one the set records,
  // as the name each was found under paired with the name it belongs under.
  // Extra files the application supplied are included.
  //
  // Reads the same before and after Repair, and is emptied by the next Verify.
  // Both names are absolute.
  bool GetRenamedFiles(std::vector<std::pair<std::string, std::string> > *files) const;

  // Work out again whether what the last Verify found can be repaired with
  // the recovery blocks available now, without reading the data files again.
  // Use it after adding more PAR2 files to a set already verified:
  //
  //   Verify(...)        -> eRepairNotPossible, too few recovery blocks
  //   AddPar2File(...)   -> another volume file arrives
  //   Reassess()         -> eRepairPossible
  //   Repair(...)
  //
  // Returns the same values as Verify, or eLogicError with ecNotVerified if
  // nothing has been verified yet. Adding a file which changes the shape of
  // the set discards the earlier results, and Verify has to be called again.
  Result Reassess(void);

  // Rebuild whatever Verify found to be missing or damaged.
  //
  // Returns eLogicError with ecNotVerified if nothing has been verified yet,
  // and eRepairNotPossible if the last Verify or Reassess found too little
  // recovery data.
  //
  // verifyafter reads back and hashes everything that was rebuilt, and is
  // what turns a repair that did not work into eRepairFailed. With it off the
  // result is eSuccess unless something went wrong along the way, and
  // GetVerifyResult still describes the state before the repair.
  Result Repair(const bool verifyafter = true);

  // Why the last call failed, refining the Result it returned. See ErrorCode
  // for which Results carry one.
  //
  // Describes only the call that returned last, and the first thing that went
  // wrong during it. An observer's OnError sees every one of them as it
  // happens, which is what a parallel scan needs.
  bool GetLastError(Par2Error *error) const;

  // Ask the work in progress to stop, from any thread. Verify or Repair then
  // returns eCancelled, having removed any partly written files. The request
  // stays in force until ClearCancel.
  void Cancel(void);
  void ClearCancel(void);

private:
  class Impl;

  void Restart(void);
  void TakeLastError(void);
  void RecordLastError(const ErrorCode code, const std::string &message);

  std::ostream &sout;
  std::ostream &serr;
  NoiseLevel noiselevel;
  Backends backends;
  Par2Observer *observer;
  size_t memorylimit;
  u32 nthreads;
  u32 filethreads;
  bool skipdata;
  u64 skipleaway;
  std::vector<std::string> par2files;
  std::vector<std::string> scannedfiles;
  std::map<std::string, std::vector<bool> > knownblocks;
  bool verified;
  std::string basepath;
  Par2Error lasterror;
  std::unique_ptr<Impl> impl;
};


// Creates a PAR2 set from an application, with progress, cancellation and the
// implementations the application supplies. The one-shot par2create below does
// the same job in a single call.
//
// A create writes over any PAR2 file of the name it is given as soon as it
// starts, before it reads any data. If a set of that name already existed,
// cancelling - or any failure - destroys it.
class Par2Creator
{
public:
  // basepath is what the names recorded in the set are relative to. Left empty
  // it is taken from the directory of the name passed to Create.
  //
  // backends holds the implementations the application supplies, each of which
  // falls back to the one built in when it is left empty.
  Par2Creator(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
              const std::string &basepath = std::string(),
              const Backends &backends = Backends());
  ~Par2Creator();

  Par2Creator(const Par2Creator &) = delete;
  Par2Creator &operator=(const Par2Creator &) = delete;

  // Notify this observer of progress and per-file results. Pass 0 to stop.
  // The observer must outlive this object.
  void SetObserver(Par2Observer *observer);

  // The files the set will describe and be able to recover. Adding one reads
  // nothing: anything wrong with it is reported by Create.
  void AddSourceFile(const std::string &filename);
  void AddSourceFiles(const std::vector<std::string> &filenames);

  // The size of each block, which must be a multiple of 4. Required.
  void SetBlockSize(const u64 blocksize);

  // How many recovery blocks to compute. Zero creates a set which describes
  // the files without being able to repair any of them.
  void SetRecoveryBlockCount(const u32 recoveryblockcount);

  // How those blocks are spread over the recovery files. recoveryfilecount is
  // only read by scUniform, and zero lets the library choose.
  void SetRecoveryFileScheme(const Scheme scheme, const u32 recoveryfilecount = 0);

  // The exponent the first recovery block is computed with, for creating a set
  // which extends another.
  void SetFirstRecoveryBlock(const u32 firstblock);

  // How much memory the work may use. Zero leaves it at the default.
  void SetMemoryLimit(const size_t memorylimit);

  // The -t and -T equivalents. Either left zero stays at the default.
  void SetThreadCounts(const u32 nthreads, const u32 filethreads);

  // Create the set, writing parfilename and the volume files beside it. A
  // trailing ".par2" is optional: the volume files are named after the set
  // rather than after its index file, so "set" and "set.par2" mean the same.
  //
  // May be called more than once, with the settings changed in between, and
  // each call writes a whole set of its own.
  Result Create(const std::string &parfilename);

  // Ask the work in progress to stop, from any thread. Create then returns
  // eCancelled, leaving no PAR2 file behind: every recovery file it had
  // created, the index file included, is deleted. The source files are only
  // read, never written. The request stays in force until ClearCancel.
  void Cancel(void);
  void ClearCancel(void);

  // Why the last call failed, refining the Result it returned. See ErrorCode
  // for which Results carry one.
  bool GetLastError(Par2Error *error) const;

private:
  class Impl;

  void Restart(void);
  void TakeLastError(void);

  std::ostream &sout;
  std::ostream &serr;
  NoiseLevel noiselevel;
  Backends backends;
  Par2Observer *observer;
  std::vector<std::string> sourcefiles;
  u64 blocksize;
  u32 recoveryblockcount;
  Scheme recoveryfilescheme;
  u32 recoveryfilecount;
  u32 firstrecoveryblock;
  size_t memorylimit;
  u32 nthreads;
  u32 filethreads;
  bool cancelled;
  std::string basepath;
  Par2Error lasterror;
  std::unique_ptr<Impl> impl;
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
