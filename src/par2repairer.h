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

#ifndef __PAR2REPAIRER_H__
#define __PAR2REPAIRER_H__

#include <atomic>

namespace par2
{

class Par2Repairer
{
public:
  Par2Repairer(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel,
               const Backends &backends = Backends());
  ~Par2Repairer(void);

  Result Process(const size_t memorylimit,
		 const std::string &basepath,
		 const u32 nthreads,
		 const u32 filethreads,
		 std::string parfilename,
		 const std::vector<std::string> &extrafiles,
		 const bool dorepair,   // derived from operation
		 const bool purgefiles,
		 const bool renameonly,
		 const bool skipdata,
		 const u64 skipleaway,
		 const bool fullhash
		 );

  // Ask the operation in progress to stop as soon as it can, from any thread.
  // Process then returns eCancelled, having removed any partly written files.
  // The flag stays set, so it must be cleared before reusing this object.
  void Cancel(void) {cancelled.store(true, std::memory_order_relaxed);}
  void ClearCancel(void) {cancelled.store(false, std::memory_order_relaxed);}
  bool IsCancelled(void) const {return cancelled.load(std::memory_order_relaxed);}

  // Set an observer to be notified of progress and per-file results.
  // Pass 0 to stop reporting. The observer must outlive this object.
  void SetObserver(Par2Observer *_observer) {observer = _observer;}

  // List the files the loaded packets describe. Available once packets have
  // been loaded and prepared.
  bool GetFileInfo(std::vector<Par2FileInfo> *files) const;

  // The CRC32 the set records for each block of the named file, one entry per
  // block starting at block 0. False when the set does not describe that file,
  // or describes it without a verification packet.
  bool GetBlockChecksums(const std::string &filename,
                         std::vector<u32> *crcs) const;

  // The numbers behind the last verification
  bool GetVerifyResult(Par2VerifyResult *result) const;

  // The source file of that name, or 0 when the set does not describe one
  Par2RepairerSourceFile *FindSourceFile(const std::string &filename) const;

  // The files a repair renamed out of the way
  bool GetBackupFiles(std::vector<std::string> *files) const;
  bool GetRenamedFiles(std::vector<std::pair<std::string, std::string> > *files) const;

  // Accept the caller's word that these blocks of the named file are intact,
  // so that they are not read and hashed again. The name is the one reported
  // by GetFileInfo and blocks must have one entry per block of that file, set
  // where the block is present at its expected offset.
  //
  // An entry set for every block means the file is intact and it is never
  // read. None set means it holds nothing usable, and it is not read either.
  // An empty vector forgets what was said about the file.
  //
  // The blocks are trusted without being verified. Supplying a block which is
  // not intact will silently produce incorrect output.
  //
  // False when the set is known and does not describe a file of that name, or
  // describes it with a different number of blocks.
  bool SetKnownBlocks(const std::string &filename, const std::vector<bool> &blocks);

protected:
  // Steps in verifying and repairing files:

  // Use the blocks the caller has vouched for instead of scanning the file
  bool TakeKnownBlocks(DiskFile               *diskfile,
                       Par2RepairerSourceFile *sourcefile,
                       std::vector<char>      &matched,
                       u32                    &matchcount);

  // Load packets from a PAR2 file, the files named after it, and the extra files
  bool LoadPackets(const std::string &parfilename,
                   const std::vector<std::string> &extrafiles,
                   bool reread = false);
  // Work out what the packets loaded so far describe
  Result PreparePackets(void);

  // Apply the -t and -T thread counts, zero leaving either alone
  void ApplyThreadCounts(const u32 _nthreads, const u32 _filethreads);

  // Apply the -m memory limit, which bounds the buffers a scan reads into
  void ApplyMemoryLimit(const size_t _memorylimit) {scanmemorylimit = _memorylimit;}

  // Verify the source files and work out whether a repair is needed
  // Scan one file, replacing whatever an earlier scan of it found
  Result ScanFile(const std::string &filename, const std::string &basepath);

  Result VerifyFiles(const std::string &basepath,
                     std::vector<std::string> &extrafiles,
                     const bool renameonly);
  // Rebuild whatever is missing or damaged
  Result RepairFiles(const size_t memorylimit, const std::string &basepath,
                     bool verifyafter = true);

  // Load packets from the specified file
  bool LoadPacketsFromFile(std::string filename, bool reread = false);
  // Finish loading a recovery packet
  bool LoadRecoveryPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header);
  // Finish loading a file description packet
  bool LoadDescriptionPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header);
  // Finish loading a file verification packet
  bool LoadVerificationPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header);
  // Finish loading the main packet
  bool LoadMainPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header);
  // Finish loading the creator packet
  bool LoadCreatorPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header);

  // Load packets from other PAR2 files with names based on the original PAR2 file
  bool LoadPacketsFromOtherFiles(std::string filename);

  // Test whether a filename has a .par2 / .PAR2 / .Par2 extension
  static bool IsPar2Filename(const std::string &filename);

  // Load packets from any other PAR2 files whose names are given on the command line
  bool LoadPacketsFromExtraFiles(const std::vector<std::string> &extrafiles);

  // Check that the packets are consistent and discard any that are not
  bool CheckPacketConsistency(void);

  // Use the information in the main packet to get the source files
  // into the correct order and determine their filenames
  bool CreateSourceFileList(void);

  // Determine the total number of DataBlocks for the recoverable source files
  // The allocate the DataBlocks and assign them to each source file
  bool AllocateSourceBlocks(void);

  // Create a verification hash table for all files for which we have not
  // found a complete version of the file and for which we have
  // a verification packet
  bool PrepareVerificationHashTable(void);

  // Compute the table for the sliding CRC computation
  bool ComputeWindowTable(void);

  // Attempt to verify all of the source files
  bool VerifySourceFiles(const std::string& basepath, std::vector<std::string>& extrafiles);

  // Scan any extra files specified on the command line
  bool VerifyExtraFiles(const std::vector<std::string> &extrafiles, const std::string &basepath, const bool renameonly);

  // Set up the tables a scan needs, once
  bool PrepareForScanning(void);

  // Forget what a scan of this file found: the blocks it supplied and its
  // place as a target or complete file
  void DiscardScannedFile(DiskFile *diskfile);

  // Attempt to match the data in the DiskFile with the source file, reporting
  // the file to the observer for as long as the match takes
  bool VerifyDataFile(DiskFile *diskfile, Par2RepairerSourceFile *sourcefile, const std::string &basepath, ProgressMeter<u64> &progress, const bool renameonly = false);

  // The match itself. sourcefile is changed when the data belongs to another
  // file of the set, and blocksfound is how many of its blocks were found.
  bool MatchDataFile(DiskFile *diskfile, Par2RepairerSourceFile *&sourcefile, const std::string &basepath, ProgressMeter<u64> &progress, const bool renameonly, u32 &blocksfound);

  // Check the blocks of a source file at the offsets where they are expected
  // to be found. One thread reads the file in order while the others check the
  // blocks it has read. This is much faster than the sliding window scan below,
  // and what it does not find narrows that scan down to the parts of the file
  // which are not where they should be.
  bool ScanDataFileAligned(DiskFile               *diskfile,    // [in]     The file being scanned
                           ProgressMeter<u64>     &progress,    // [in]
                           Par2RepairerSourceFile *sourcefile,  // [in]     The file it should match
                           std::vector<char>      &matched,     // [out]    One entry per block
                           u32                    &matchcount,  // [out]
                           MD5Hash                &hashfull,    // [out]    Only set when the whole hash is wanted
                           MD5Hash                &hash16k);    // [out]

  // Perform a sliding window scan of the DiskFile looking for blocks of data that
  // might belong to any of the source files (for which a verification packet was
  // available). If a block of data might be from more than one source file, prefer
  // the one specified by the "sourcefile" parameter. If the first data block
  // found is for a different source file then "sourcefile" is changed accordingly.
  bool ScanDataFile(DiskFile                *diskfile,   // [in]     The file being scanned
                    std::string             basepath,    // [in]
                    ProgressMeter<u64>      &progress,   // [in]
                    const bool              renameonly,  // [in]     Only look for perfect matches
                    Par2RepairerSourceFile* &sourcefile, // [in/out] The source file matched
                    MatchType               &matchtype,  // [out]    The type of match
                    MD5Hash                 &hashfull,   // [out]    The full hash of the file
                    MD5Hash                 &hash16k,    // [out]    The hash of the first 16k
                    u32                     &count);     // [out]    The number of blocks found

  // Find out how much data we have found
  void UpdateVerificationResults(void);

  // Check the verification results and report the results
  bool CheckVerificationResults(void);

  // Rename any damaged or missnamed target files.
  bool RenameTargetFiles(void);

  // Work out which files are being repaired, create them, and allocate
  // target DataBlocks to them, and remember them for later verification.
  bool CreateTargetFiles(void);

  // Work out which data blocks are available, which need to be copied
  // directly to the output, and which need to be recreated, and compute
  // the appropriate Reed Solomon matrix.
  bool ComputeRSmatrix(void);

  // Allocate memory buffers for reading and writing data to disk.
  bool AllocateBuffers(size_t memorylimit);

  // Read source data, process it through the RS matrix and write it to disk.
  bool ProcessData(u64 blockoffset, size_t blocklength, ProgressMeter<u64> &progress);

  // Verify that all of the reconstructed target files are now correct
  bool VerifyTargetFiles(const std::string &basepath);

  // Delete all of the partly reconstructed files
  bool DeleteIncompleteTargetFiles(void);

  // list the files needing verification
  bool RemoveBackupFiles(void);
  bool RemoveParFiles(void);

  // Make the buffers the files being scanned read into, or give them up when
  // no file will have its blocks checked where they are expected to be
  void ResetScanBuffers(const size_t filecount);

  // The number of files to read at once, which is what limits how many are
  // open at a time rather than how much of the work they get
  u32                                 FileThreads(size_t filecount) const
    {return (u32)std::max<size_t>(1, std::min<size_t>(filethreads, filecount));}

protected:
  std::ostream &sout; // stream for output (for commandline, this is cout)
  std::ostream &serr; // stream for errors (for commandline, this is cerr)

  const NoiseLevel noiselevel;              // OnScreen display
  const Backends backends;                  // The implementations the application supplied

  Par2Observer *observer;                   // Notified of progress, or 0

  std::atomic<bool> cancelled;              // Set by Cancel from any thread

  u32                       packetsloaded;           // Useable packets read so far

  // Blocks the caller has vouched for, keyed by the name the set records
  std::map<std::string, std::vector<bool> > knownblocks;

  // The source files by the name each has on this system
  std::map<std::string, Par2RepairerSourceFile*> sourcefilesbyname;

  std::string               searchpath;              // Where to find files on disk

  std::string               basepath;

  u32 totalthreads;            // Number of threads the whole repair may use
  u32 filethreads;             // Number of files to read at once
  size_t scanmemorylimit;      // Memory the buffers a scan reads into may use

  // The threads which check the blocks of every file being read, the buffers
  // those files read into, and how many files are being read at the moment
  std::unique_ptr<TaskPool> blockpool;
  BufferPool                scanbuffers;
  std::atomic<u32>          activereaders;

  bool                      skipdata;                // Should we skip data whilst scanning
  u64                       skipleaway;              // The leaway +/- we should allow whilst scanning
  bool                      fullhash;                // Should the whole of each file be hashed too

  bool                      firstpacket;             // Whether or not a valid packet has been found.
  MD5Hash                   setid;                   // The SetId extracted from the first packet.
  u64                       totaldatasize;           // Total size of the recoverable files

  std::map<u32, RecoveryPacket*> recoverypacketmap;       // One recovery packet for each exponent value.
  MainPacket               *mainpacket;              // One copy of the main packet.
  CreatorPacket            *creatorpacket;           // One copy of the creator packet.

  DiskFileMap               diskFileMap;
  std::mutex                diskFileMapMutex;        // Guards diskFileMap while files are verified in parallel.
  std::mutex                extraFilesMutex;         // Guards the caller's list of extra files.

  std::map<MD5Hash,Par2RepairerSourceFile*> sourcefilemap;// Map from FileId to SourceFile
  std::vector<Par2RepairerSourceFile*>      sourcefiles;  // The source files
  std::vector<Par2RepairerSourceFile*>      verifylist;   // Those source files that are being repaired
  std::vector<DiskFile*>                    backuplist;   // Those source files backups
  // What each renamed file was found as, keyed by the name the set records
  std::map<std::string, std::string>        renamedlist;
  std::list<std::string>                    par2list;     // list of par2 files

  u64                       blocksize;               // The block size.
  u64                       chunksize;               // How much of a block can be processed.
  u32                       sourceblockcount;        // The total number of blocks
  u32                       availableblockcount;     // How many undamaged blocks have been found
  u32                       missingblockcount;       // How many blocks are missing

  bool                      blocksallocated;         // Whether or not the DataBlocks have been allocated
  std::vector<DataBlock>    sourceblocks;            // The DataBlocks that will be read from disk
  std::vector<DataBlock>    targetblocks;            // The DataBlocks that will be written to disk

  u32                       windowtable[256];        // Table for sliding CRCs

  bool                            scanningprepared;        // Whether the tables a scan needs have been built
  bool                            blockverifiable;         // Whether and files can be verified at the block level
  VerificationHashTable           verificationhashtable;   // Hash table for block verification
  std::list<Par2RepairerSourceFile*>   unverifiablesourcefiles; // Files that are not block verifiable

  u32                       completefilecount;       // How many files are fully verified
  u32                       renamedfilecount;        // How many files are verified but have the wrong name
  u32                       damagedfilecount;        // How many files exist but are damaged
  u32                       missingfilecount;        // How many files are completely missing

  std::vector<DataBlock*>   inputblocks;             // Which DataBlocks will be read from disk
  std::vector<DataBlock*>   copyblocks;              // Which DataBlocks will copied back to disk
  std::vector<DataBlock*>   outputblocks;            // Which DataBlocks have to calculated using RS

  ReedSolomon<Galois16>     rs;                      // The Reed Solomon matrix.

  void                     *transferbuffer;          // Input blocks in flight (chunksize * NUM_TRANSFER_BUFFERS)
  void                     *outputbuffer;            // Buffer for writing DataBlocks (chunksize)
  std::unique_ptr<Processor> processor;              // Multiplies the input blocks by the RS matrix
  bool                      ownfactors;              // Whether the processor solved the erasure itself
};

} // namespace par2

#endif // __PAR2REPAIRER_H__
