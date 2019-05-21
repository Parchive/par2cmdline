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

#ifndef __PAR2CREATOR_H__
#define __PAR2CREATOR_H__

class MainPacket;
class CreatorPacket;
class CriticalPacket;


class Par2Creator
{
public:
  Par2Creator(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel);
  ~Par2Creator(void);

  // Create recovery files from the source files specified on the command line
  Result Process(const size_t memorylimit,
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
		 );

protected:
  // Steps in the creation process:

  // Compute block size from block count or vice versa depending on which was
  // specified on the command line
  bool ComputeBlockCount(const vector<string> &extrafiles);

  // Determine how much recovery data can be computed on one pass
  bool CalculateProcessBlockSize(size_t memorylimit);

  // Open all of the source files, compute the Hashes and CRC values, and store
  // the results in the file verification and file description packets.
  bool OpenSourceFiles(const vector<string> &extrafiles, string basepath);

  // Create the main packet and determine the set_id_hash to use with all packets
  bool CreateMainPacket(void);

  // Create the creator packet.
  bool CreateCreatorPacket(void);

  // Initialise all of the source blocks ready to start reading data from the source files.
  bool CreateSourceBlocks(void);

  // Create all of the output files and allocate all packets to appropriate file offets.
  bool InitialiseOutputFiles(const string &par2filename);

  // Allocate memory buffers for reading and writing data to disk.
  bool AllocateBuffers(void);

  // Compute the Reed Solomon matrix
  bool ComputeRSMatrix(void);

  // Read source data, process it through the RS matrix and write it to disk.
  bool ProcessData(u64 blockoffset, size_t blocklength);

  // Finish computation of the recovery packets and write the headers to disk.
  bool WriteRecoveryPacketHeaders(void);

  // Finish computing the full file hash values of the source files
  bool FinishFileHashComputation(void);

  // Fill in all remaining details in the critical packets.
  bool FinishCriticalPackets(void);

  // Write all other critical packets to disk.
  bool WriteCriticalPackets(void);

  // Close all files.
  bool CloseFiles(void);

#ifdef _OPENMP
  static u32                          GetFileThreads(void) {return filethreads;}
#endif
  
protected:
  std::ostream &sout; // stream for output (for commandline, this is cout)
  std::ostream &serr; // stream for errors (for commandline, this is cerr)
  
  const NoiseLevel noiselevel; // How noisy we should be

#ifdef _OPENMP
  static u32 filethreads;      // Number of threads for file processing
#endif

  u64 blocksize;      // The size of each block.
  size_t chunksize;   // How much of each block will be processed at a 
                      // time (due to memory constraints).

  void *inputbuffer;  // chunksize
  void *outputbuffer; // chunksize * recoveryblockcount
  
  u32 sourcefilecount;   // Number of source files for which recovery data will be computed.
  u32 sourceblockcount;  // Total number of data blocks that the source files will be
                         // virtually sliced into.

  u64 largestfilesize;   // The size of the largest source file

  Scheme recoveryfilescheme;  // What scheme will be used to select the
                                           // sizes for the recovery files.
  
  u32 recoveryfilecount;  // The number of recovery files that will be created
  u32 recoveryblockcount; // The number of recovery blocks that will be placed
                          // in the recovery files.

  u32 firstrecoveryblock; // The lowest exponent value to use for the recovery blocks.

  MainPacket    *mainpacket;    // The main packet
  CreatorPacket *creatorpacket; // The creator packet

  vector<Par2CreatorSourceFile*> sourcefiles;  // Array containing details of the source files
                                               // as well as the file verification and file
                                               // description packets for them.

  vector<DataBlock>          sourceblocks;     // Array with one entry for every source block.

  vector<DiskFile>           recoveryfiles;    // Array with one entry for every recovery file.
  vector<RecoveryPacket>     recoverypackets;  // Array with one entry for every recovery packet.

  list<CriticalPacket*>      criticalpackets;  // A list of all of the critical packets.
  list<CriticalPacketEntry>  criticalpacketentries; // A list of which critical packet will
                                                    // be written to which recovery file.

  ReedSolomon<Galois16> rs;   // The Reed Solomon matrix.

  u64 progress;     // How much data has been processed.
  u64 totaldata;    // Total amount of data to be processed.

  bool deferhashcomputation; // If we have enough memory to compute all recovery data
                             // in one pass, then we can defer the computation of
                             // the full file hash and block crc and hashes until
                             // the recovery data is computed.
#ifdef _OPENMP
  u64 mttotalsize;           // Total size of files for mt-progress line
#endif  
};

#endif // __PAR2CREATOR_H__
