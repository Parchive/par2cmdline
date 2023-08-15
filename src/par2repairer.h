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


class Par2Repairer
{
public:
  Par2Repairer(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel);
  ~Par2Repairer(void);

  Result Process(const size_t memorylimit,
		 const string &basepath,
#ifdef _OPENMP
		 const u32 nthreads,
		 const u32 filethreads,
#endif
		 string parfilename,
		 const vector<string> &extrafiles,
		 const bool dorepair,   // derived from operation
		 const bool purgefiles,
		 const bool renameonly,
		 const bool skipdata,
		 const u64 skipleaway
		 );

protected:
  // Steps in verifying and repairing files:

  // Load packets from the specified file
  bool LoadPacketsFromFile(string filename);
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
  bool LoadPacketsFromOtherFiles(string filename);

  // Load packets from any other PAR2 files whose names are given on the command line
  bool LoadPacketsFromExtraFiles(const vector<string> &extrafiles);

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
  bool VerifySourceFiles(const std::string& basepath, std::vector<string>& extrafiles);

  // Scan any extra files specified on the command line
  bool VerifyExtraFiles(const vector<string> &extrafiles, const string &basepath, const bool renameonly);

  // Attempt to match the data in the DiskFile with the source file
  bool VerifyDataFile(DiskFile *diskfile, Par2RepairerSourceFile *sourcefile, const string &basepath, const bool renameonly = false);

  // Perform a sliding window scan of the DiskFile looking for blocks of data that
  // might belong to any of the source files (for which a verification packet was
  // available). If a block of data might be from more than one source file, prefer
  // the one specified by the "sourcefile" parameter. If the first data block
  // found is for a different source file then "sourcefile" is changed accordingly.
  bool ScanDataFile(DiskFile                *diskfile,   // [in]     The file being scanned
                    string                  basepath,    // [in]
		    const bool              renameonly,  // [in]					 //
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
  bool ProcessData(u64 blockoffset, size_t blocklength);

  // Verify that all of the reconstructed target files are now correct
  bool VerifyTargetFiles(const string &basepath);

  // Delete all of the partly reconstructed files
  bool DeleteIncompleteTargetFiles(void);

  // list the files needing verification
  bool RemoveBackupFiles(void);
  bool RemoveParFiles(void);

#ifdef _OPENMP
  static u32                          GetFileThreads(void) {return filethreads;}
#endif

protected:
  std::ostream &sout; // stream for output (for commandline, this is cout)
  std::ostream &serr; // stream for errors (for commandline, this is cerr)

  const NoiseLevel noiselevel;              // OnScreen display

  string                    searchpath;              // Where to find files on disk

  std::string               basepath;

#ifdef _OPENMP
  static u32 filethreads;      // Number of threads for file processing
#endif

  bool                      skipdata;                // Should we skip data whilst scanning
  u64                       skipleaway;              // The leaway +/- we should allow whilst scanning

  bool                      firstpacket;             // Whether or not a valid packet has been found.
  MD5Hash                   setid;                   // The SetId extracted from the first packet.

  map<u32, RecoveryPacket*> recoverypacketmap;       // One recovery packet for each exponent value.
  MainPacket               *mainpacket;              // One copy of the main packet.
  CreatorPacket            *creatorpacket;           // One copy of the creator packet.

  DiskFileMap               diskFileMap;

  map<MD5Hash,Par2RepairerSourceFile*> sourcefilemap;// Map from FileId to SourceFile
  vector<Par2RepairerSourceFile*>      sourcefiles;  // The source files
  vector<Par2RepairerSourceFile*>      verifylist;   // Those source files that are being repaired
  vector<DiskFile*>                    backuplist;   // Those source files backups
  list<string>                         par2list;     // list of par2 files

  u64                       blocksize;               // The block size.
  u64                       chunksize;               // How much of a block can be processed.
  u32                       sourceblockcount;        // The total number of blocks
  u32                       availableblockcount;     // How many undamaged blocks have been found
  u32                       missingblockcount;       // How many blocks are missing

  bool                      blocksallocated;         // Whether or not the DataBlocks have been allocated
  vector<DataBlock>         sourceblocks;            // The DataBlocks that will be read from disk
  vector<DataBlock>         targetblocks;            // The DataBlocks that will be written to disk

  u32                       windowtable[256];        // Table for sliding CRCs

  bool                            blockverifiable;         // Whether and files can be verified at the block level
  VerificationHashTable           verificationhashtable;   // Hash table for block verification
  list<Par2RepairerSourceFile*>   unverifiablesourcefiles; // Files that are not block verifiable

  u32                       completefilecount;       // How many files are fully verified
  u32                       renamedfilecount;        // How many files are verified but have the wrong name
  u32                       damagedfilecount;        // How many files exist but are damaged
  u32                       missingfilecount;        // How many files are completely missing

  vector<DataBlock*>        inputblocks;             // Which DataBlocks will be read from disk
  vector<DataBlock*>        copyblocks;              // Which DataBlocks will copied back to disk
  vector<DataBlock*>        outputblocks;            // Which DataBlocks have to calculated using RS

  ReedSolomon<Galois16>     rs;                      // The Reed Solomon matrix.

  void                     *inputbuffer;             // Buffer for reading DataBlocks (chunksize)
  void                     *outputbuffer;            // Buffer for writing DataBlocks (chunksize * missingblockcount)

  u64                       progress;                // How much data has been processed.
  u64                       totaldata;               // Total amount of data to be processed.
#ifdef _OPENMP
  u64                       mttotalsize;             // Total size of files for mt-progress line
  u64                       mttotalextrasize;        // Total size of extra files for mt-progress line
  u64                       mttotalprogress;         // MT total progress
  bool                      mtprocessingextrafiles;  // Are we currently processing extra files
#endif
};

#endif // __PAR2REPAIRER_H__
