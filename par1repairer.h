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

#ifndef __PAR1REPAIRER_H__
#define __PAR1REPAIRER_H__

class Par1Repairer
{
public:
  Par1Repairer(void);
  ~Par1Repairer(void);

  Result Process(const CommandLine &commandline, bool dorepair);

protected:
  // Load the main PAR file
  bool LoadRecoveryFile(string filename);

  // Load other PAR files related to the main PAR file
  bool LoadOtherRecoveryFiles(string filename);

  // Load any extra PAR files specified on the command line
  bool LoadExtraRecoveryFiles(const list<CommandLine::ExtraFile> &extrafiles);

  // Check for the existence of and verify each of the source files
  bool VerifySourceFiles(void);

  // Check any other files specified on the command line to see if they are
  // actually copies of the source files that have the wrong filename
  bool VerifyExtraFiles(const list<CommandLine::ExtraFile> &extrafiles);

  // Attempt to match the data in the DiskFile with the source file
  bool VerifyDataFile(DiskFile *diskfile, Par1RepairerSourceFile *sourcefile);

  // Determine how many files are missing, damaged etc.
  void UpdateVerificationResults(void);

  // Check the verification results and report the details
  bool CheckVerificationResults(void);

  // Rename any damaged or missnamed target files.
  bool RenameTargetFiles(void);

  // Work out which files are being repaired, create them, and allocate
  // target DataBlocks to them, and remember them for later verification.
  bool CreateTargetFiles(void);

  // Work out which data blocks are available, which need to be recreated, 
  // and compute the appropriate Reed Solomon matrix.
  bool ComputeRSmatrix(void);

  // Allocate memory buffers for reading and writing data to disk.
  bool AllocateBuffers(size_t memorylimit);

  // Read source data, process it through the RS matrix and write it to disk.
  bool ProcessData(u64 blockoffset, size_t blocklength);

  // Verify that all of the reconstructed target files are now correct
  bool VerifyTargetFiles(void);

  // Delete all of the partly reconstructed files
  bool DeleteIncompleteTargetFiles(void);

protected:
  string                    searchpath;              // Where to find files on disk
  DiskFileMap               diskfilemap;             // Map from filename to DiskFile

  map<u32, DataBlock*>      recoveryblocks;          // The recovery data (mapped by exponent)

  unsigned char            *filelist;
  u32                       filelistsize;

  u64                       blocksize;               // The size of recovery and data blocks
  u64                       chunksize;               // How much of a block can be processed.

  vector<Par1RepairerSourceFile*> sourcefiles;
  vector<Par1RepairerSourceFile*> extrafiles;

  u32                       completefilecount;
  u32                       renamedfilecount;
  u32                       damagedfilecount;
  u32                       missingfilecount;

  list<Par1RepairerSourceFile*> verifylist;

  vector<DataBlock*>        inputblocks;             // Which DataBlocks will be read from disk
  vector<DataBlock*>        outputblocks;            // Which DataBlocks have to calculated using RS

  ReedSolomon<Galois8>      rs;                      // The Reed Solomon matrix.

  u64                       progress;                // How much data has been processed.
  u64                       totaldata;               // Total amount of data to be processed.

  size_t                    inputbuffersize;
  u8                       *inputbuffer;             // Buffer for reading DataBlocks (chunksize)
  size_t                    outputbufferalignment;
  size_t                    outputbuffersize;
  u8                       *outputbuffer;            // Buffer for writing DataBlocks (chunksize * missingblockcount)
  bool                      ignore16kfilehash;       // The 16k file hash values may be invalid
};

#endif // __PAR1REPAIRER_H__
