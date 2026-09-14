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

#include "libpar2internal.h"

namespace par2
{

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif


Par2CreatorEngine::Par2CreatorEngine(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel, const Backends &backends)
: sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, backends(backends)
, observer(0)
, cancelled(false)
, totalthreads(default_threads())
, filethreads(_FILE_THREADS)
, blocksize(0)
, chunksize(0)
, transferbuffer(0)
, outputbuffer(0)

, sourcefilecount(0)
, sourceblockcount(0)

, largestfilesize(0)
, totaldatasize(0)
, recoveryfilescheme(scUnknown)
, recoveryfilecount(0)
, recoveryblockcount(0)
, firstrecoveryblock(0)

, mainpacket(0)
, creatorpacket(0)

, sourcefiles()
, sourceblocks()
, recoveryfiles()
, recoverypackets()
, criticalpackets()
, criticalpacketentries()
, rs()

, deferhashcomputation(false)
{
}

Par2CreatorEngine::~Par2CreatorEngine(void)
{
  delete mainpacket;
  delete creatorpacket;

  delete [] (u8*)transferbuffer;
  delete [] (u8*)outputbuffer;

  std::vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
  while (sourcefile != sourcefiles.end())
  {
    delete *sourcefile;
    ++sourcefile;
  }
}

Result Par2CreatorEngine::Process(
			    const size_t _memorylimit,
			    const std::string &_basepath,
			    const u32 nthreads,
			    const u32 _filethreads,
			    const std::string &_parfilename,
			    const std::vector<std::string> &_extrafiles,
			    const u64 _blocksize,
			    const u32 _firstblock,
			    const Scheme _recoveryfilescheme,
			    const u32 _recoveryfilecount,
			    const u32 _recoveryblockcount)
{
  ClearLastError();

  // Get information from commandline
  memorylimit = _memorylimit;
  basepath = _basepath;
  parfilename = _parfilename;
  extrafiles = _extrafiles;
  blocksize = _blocksize;
  sourcefilecount = (u32)extrafiles.size();
  recoveryblockcount = _recoveryblockcount;
  recoveryfilecount = _recoveryfilecount;
  firstrecoveryblock = _firstblock;
  recoveryfilescheme = _recoveryfilescheme;

  ApplyThreadCounts(nthreads, _filethreads);

  Result result = PrepareCreation();
  if (result != eSuccess)
    return result;

  result = HashSourceFiles();
  if (result != eSuccess)
    return result;

  if (IsCancelled())
    return eCancelled;

  result = CreateOutputFiles();
  if (result != eSuccess)
    return result;

  result = ComputeRecoveryData();
  if (result != eSuccess)
    return result;

  result = WriteCriticalData();
  if (result != eSuccess)
    return result;

  if (noiselevel > nlSilent)
    sout << "Done" << std::endl;

  return eSuccess;
}

// Apply the thread counts, leaving either at its default when it is zero
void Par2CreatorEngine::ApplyThreadCounts(const u32 nthreads, const u32 _filethreads)
{
  totalthreads = resolve_threads(nthreads);

  // No more files are read at once than there are threads to hash them with,
  // and never none whatever the caller asked for
  if (_filethreads != 0)
    filethreads = std::max(1u, std::min(_filethreads, totalthreads));
}

// Work out the shape of the set, and check that it can be written
Result Par2CreatorEngine::PrepareCreation(void)
{
  if (!CheckBasepath(parfilename))
  {
    errorlog.RecordIfNone(ecFileCreateFailed, "Could not write beside the set", parfilename);
    return eFileIOError;
  }

  // Compute block size from block count or vice versa depending on which was
  // specified on the command line
  if (!ComputeBlockCount())
  {
    if (IsCancelled())
      return eCancelled;

    errorlog.RecordIfNone(ecInvalidSetting, "The block size cannot be used");
    return eInvalidCommandLineArguments;
  }

  // Determine how many recovery files to create.
  if (!ComputeRecoveryFileCount(sout,
				serr,
				&recoveryfilecount,
				recoveryfilescheme,
				recoveryblockcount,
				largestfilesize,
				blocksize)) {
    errorlog.RecordIfNone(ecInvalidSetting, "The recovery file scheme cannot be used");
    return eInvalidCommandLineArguments;
  }

  // Determine how much recovery data can be computed on one pass
  if (!CalculateProcessBlockSize(memorylimit))
  {
    errorlog.RecordIfNone(ecInternalError, "Could not work out how much to process at a time");
    return eLogicError;
  }

  if (recoveryblockcount > 0 && noiselevel >= nlDebug)
    sout << "[DEBUG] Process chunk size: " << chunksize << std::endl;

  if (noiselevel > nlQuiet)
  {
    // Display information.
    sout << "Block size: " << blocksize << "\n"
      "Source file count: " << sourcefilecount << "\n"
      "Source block count: " << sourceblockcount << "\n"
      "Recovery block count: " << recoveryblockcount << "\n"
      "Recovery file count: " << recoveryfilecount << "\n"
      << std::endl;
  }

  return eSuccess;
}

// Read every source file and record what it contains
Result Par2CreatorEngine::HashSourceFiles(void)
{
  // Open all of the source files, compute the Hashes and CRC values, and store
  // the results in the file verification and file description packets.
  if (!OpenSourceFiles())
  {
    if (IsCancelled())
      return eCancelled;

    errorlog.RecordIfNone(ecFileReadFailed, "Could not read the source files");
    return eFileIOError;
  }

  // Create the main packet and determine the setid to use with all packets
  if (!CreateMainPacket())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not build the packet describing the set");
    return eLogicError;
  }

  if (observer)
  {
    Par2SetInfo info;
    memcpy(info.setid.data(), mainpacket->SetId().hash, sizeof(mainpacket->SetId().hash));
    info.blocksize = blocksize;
    info.datablocks = sourceblockcount;
    info.recoveryblocks = recoveryblockcount;
    info.recoverablefilecount = sourcefilecount;
    info.otherfilecount = 0;
    info.datasize = totaldatasize;

    observer->OnSetInfo(info);
  }

  // Create the creator packet.
  if (!CreateCreatorPacket())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not build the creator packet");
    return eLogicError;
  }

  // Initialise all of the source blocks ready to start reading data from the source files.
  if (!CreateSourceBlocks())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not lay out the source blocks");
    return eLogicError;
  }

  return eSuccess;
}

// Create the recovery files. Nothing has been written before this, and after
// it every file of the set exists at its full size.
Result Par2CreatorEngine::CreateOutputFiles(void)
{
  // Create all of the output files and allocate all packets to appropriate file offsets.
  if (!InitialiseOutputFiles())
  {
    DeleteIncompleteRecoveryFiles();

    if (IsCancelled())
      return eCancelled;

    errorlog.RecordIfNone(ecFileCreateFailed, "Could not create the recovery files");
    return eFileIOError;
  }

  return eSuccess;
}

// Compute the recovery blocks and write them
Result Par2CreatorEngine::ComputeRecoveryData(void)
{
  if (recoveryblockcount == 0)
    return eSuccess;

  // Allocate memory buffers for reading and writing data to disk.
  if (!AllocateBuffers(memorylimit))
  {
    DeleteIncompleteRecoveryFiles();
    return eMemoryError;
  }

  // Compute the Reed Solomon matrix
  if (!ComputeRSMatrix())
  {
    DeleteIncompleteRecoveryFiles();
    errorlog.RecordIfNone(ecProcessorFailed, "Could not compute the Reed Solomon matrix");
    return eLogicError;
  }

  // Set the total amount of data to be processed.
  ProgressMeter<u64> progress(sout, "Processing: ", blocksize * sourceblockcount, noiselevel, observer);

  // Start at an offset of 0 within a block.
  u64 blockoffset = 0;
  while (blockoffset < blocksize) // Continue until the end of the block.
  {
    // Work out how much data to process this time.
    size_t blocklength = (size_t)std::min((u64)chunksize, blocksize-blockoffset);

    // Read source data, process it through the RS matrix and write it to disk.
    if (!ProcessData(blockoffset, blocklength, progress))
    {
      DeleteIncompleteRecoveryFiles();

      if (IsCancelled())
        return eCancelled;

      errorlog.RecordIfNone(ecProcessorFailed, "Could not compute the recovery blocks");
      return eFileIOError;
    }

    blockoffset += blocklength;
  }

  if (noiselevel > nlQuiet)
    sout << "Writing recovery packets" << std::endl;

  // Finish computation of the recovery packets and write the headers to disk.
  if (!WriteRecoveryPacketHeaders())
  {
    DeleteIncompleteRecoveryFiles();
    errorlog.RecordIfNone(ecFileWriteFailed, "Could not write the recovery packet headers");
    return eFileIOError;
  }

  // Finish computing the full file hash values of the source files
  if (!FinishFileHashComputation())
  {
    DeleteIncompleteRecoveryFiles();
    errorlog.RecordIfNone(ecInternalError, "Could not finish hashing the source files");
    return eLogicError;
  }

  return eSuccess;
}

// Write what describes the set, and close everything
Result Par2CreatorEngine::WriteCriticalData(void)
{
  // Fill in all remaining details in the critical packets.
  if (!FinishCriticalPackets())
  {
    DeleteIncompleteRecoveryFiles();
    errorlog.RecordIfNone(ecInternalError, "Could not finish the packets describing the set");
    return eLogicError;
  }

  if (noiselevel > nlQuiet)
    sout << "Writing verification packets" << std::endl;

  // Write all other critical packets to disk.
  if (!WriteCriticalPackets())
  {
    DeleteIncompleteRecoveryFiles();

    if (IsCancelled())
      return eCancelled;

    errorlog.RecordIfNone(ecFileWriteFailed, "Could not write the packets describing the set");
    return eFileIOError;
  }

  // Close all files.
  if (!CloseFiles())
  {
    errorlog.RecordIfNone(ecFileWriteFailed, "Could not close the recovery files");
    return eFileIOError;
  }

  return eSuccess;
}

// Check basepath permission
bool Par2CreatorEngine::CheckBasepath(const std::string &parfilename)
{
  std::string checkfilename = parfilename + ".check.par2";
  std::unique_ptr<DiskFile> diskfile(new DiskFile(sout, serr, &errorlog));
  size_t dummysize = 4096;

  if (!diskfile->Create(checkfilename, dummysize))
    return false;

  diskfile->Close();

  if (!diskfile->Delete())
    return false;

  return true;
}

// Compute block size from block count or vice versa depending on which was
// specified on the command line
bool Par2CreatorEngine::ComputeBlockCount(void)
{
  FileSizeCache filesize_cache;

  largestfilesize = 0;
  totaldatasize = 0;
  for (std::vector<std::string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
  {
    if (IsCancelled())
      return false;

    u64 filesize = filesize_cache.get(*i);
    if (largestfilesize < filesize)
    {
      largestfilesize = filesize;
    }
    totaldatasize += filesize;
  }


  if (blocksize == 0)
  {
    serr << "ERROR: Block size was zero!" << std::endl;
    errorlog.Record(ecInvalidSetting, "The block size was zero");
    return false;
  }

  if (blocksize % 4 != 0)
  {
    serr << "ERROR: Block size was not a multiple of 4 bytes!" << std::endl;
    errorlog.Record(ecInvalidSetting, "The block size was not a multiple of 4 bytes");
    return false;
  }


  u64 count = 0;

  for (std::vector<std::string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
  {
    count += (filesize_cache.get(*i) + blocksize-1) / blocksize;
  }

  if (count > 32768)
  {
    serr << "Block size is too small. It would require " << count << "blocks." << std::endl;
    errorlog.Record(ecTooManySourceBlocks, "The block size would need more blocks than can be held");
    return false;
  }

  sourceblockcount = (u32)count;

  return true;
}



// Determine how much recovery data can be computed on one pass
bool Par2CreatorEngine::CalculateProcessBlockSize(size_t memorylimit)
{
  // Are we computing any recovery blocks
  if (recoveryblockcount == 0)
  {
    chunksize = 0;

    deferhashcomputation = false;
  }
  else
  {
    // Would single pass processing use too much memory
    if (blocksize * recoveryblockcount > memorylimit)
    {
      // Pick a size that is small enough
      chunksize = ~3 & (memorylimit / recoveryblockcount);

      deferhashcomputation = false;
    }
    else
    {
      chunksize = (size_t)blocksize;

      deferhashcomputation = true;
    }

    if (MAX_CHUNK_SIZE != 0 && chunksize > MAX_CHUNK_SIZE)
    {
      chunksize = MAX_CHUNK_SIZE;
      deferhashcomputation = false;
    }
  }

  return true;
}


// Open all of the source files, compute the Hashes and CRC values, and store
// the results in the file verification and file description packets.
bool Par2CreatorEngine::OpenSourceFiles(void)
{
  std::atomic<bool> openfailed(false);

  //Total size of files for mt-progress line
  u64 mttotalsize = 0;
  for (size_t i=0; i<extrafiles.size(); ++i)
    mttotalsize += DiskFile::GetFileSize(extrafiles[i]);

  ProgressMeter<u64> progress(sout, "", mttotalsize, noiselevel, observer);

  foreach_parallel(extrafiles, GetFileThreads(), [&](const std::string &extrafile)
  {
    if (openfailed || IsCancelled())
      return;

    Par2CreatorSourceFile *sourcefile = new Par2CreatorSourceFile;

    std::string name;
    DiskFile::SplitRelativeFilename(extrafile, basepath, name);

    if (noiselevel > nlSilent)
    {
      LockedStream(sout) << "Opening: " << name << std::endl;
    }

    if (observer)
      observer->OnFile(name);

    // Open the source file and compute its Hashes and CRCs.
    if (!sourcefile->Open(noiselevel, sout, serr, extrafile, blocksize, deferhashcomputation, basepath, progress, backends, &cancelled))
    {
      delete sourcefile;
      openfailed = true;
      return;
    }

    // Every block of a file just read is there by definition
    if (observer)
      observer->OnFileDone(name, sourcefile->BlockCount(), sourcefile->BlockCount());

    // Record the file verification and file description packets
    // in the critical packet list.
    {
    std::lock_guard<std::mutex> lock(sourcefilesMutex);
    sourcefile->RecordCriticalPackets(criticalpackets);

    // Add the source file to the sourcefiles array.
    sourcefiles.push_back(sourcefile);
    }
    // Close the source file until its needed
    sourcefile->Close();

  });

  if (openfailed || IsCancelled())
    return false;

  return true;
}

// Create the main packet and determine the setid to use with all packets
bool Par2CreatorEngine::CreateMainPacket(void)
{
  // Construct the main packet from the list of source files and the block size.
  mainpacket = new MainPacket;

  // Add the main packet to the list of critical packets.
  criticalpackets.push_back(mainpacket);

  // Create the packet (sourcefiles will get sorted into FileId order).
  return mainpacket->Create(sourcefiles, blocksize);
}

// Create the creator packet.
bool Par2CreatorEngine::CreateCreatorPacket(void)
{
  // Construct the creator packet
  creatorpacket = new CreatorPacket;

  // Create the packet
  return creatorpacket->Create(mainpacket->SetId());
}

// Initialise all of the source blocks ready to start reading data from the source files.
bool Par2CreatorEngine::CreateSourceBlocks(void)
{
  // Allocate the array of source blocks
  sourceblocks.resize(sourceblockcount);

  std::vector<DataBlock>::iterator sourceblock = sourceblocks.begin();

  for (std::vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
       sourcefile!= sourcefiles.end();
       sourcefile++)
  {
    // Allocate the appropriate number of source blocks to each source file.
    // sourceblock will be advanced.

    (*sourcefile)->InitialiseSourceBlocks(sourceblock, blocksize);
  }

  return true;
}

class FileAllocation
{
public:
  FileAllocation(void)
  : filename("")
  {
    exponent = 0;
    count = 0;
  }

  std::string filename;
  u32 exponent;
  u32 count;
};

// Create all of the output files and allocate all packets to appropriate file offsets.
bool Par2CreatorEngine::InitialiseOutputFiles(void)
{
  // Allocate the recovery packets
  recoverypackets.resize(recoveryblockcount);

  // Choose filenames and decide which recovery blocks to place in each file
  std::vector<FileAllocation> fileallocations;
  fileallocations.resize(recoveryfilecount+1); // One extra file with no recovery blocks
  {
    // Decide how many recovery blocks to place in each file
    u32 exponent = firstrecoveryblock;
    if (recoveryfilecount > 0)
    {
      switch (recoveryfilescheme)
      {
      case scUnknown:
        {
          assert(false);
          return false;
        }
        break;
      case scUniform:
        {
          // Files will have roughly the same number of recovery blocks each.

          u32 base      = recoveryblockcount / recoveryfilecount;
          u32 remainder = recoveryblockcount % recoveryfilecount;

          for (u32 filenumber=0; filenumber<recoveryfilecount; filenumber++)
          {
            fileallocations[filenumber].exponent = exponent;
            fileallocations[filenumber].count = (filenumber<remainder) ? base+1 : base;
            exponent += fileallocations[filenumber].count;
          }
        }
        break;

      case scVariable:
        {
          // Files will have recovery blocks allocated in an exponential fashion.

          // Work out how many blocks to place in the smallest file
          u32 lowblockcount = 1;
          u32 maxrecoveryblocks = (1 << recoveryfilecount) - 1;
          while (maxrecoveryblocks < recoveryblockcount)
          {
            lowblockcount <<= 1;
            maxrecoveryblocks <<= 1;
          }

          // Allocate the blocks.
          u32 blocks = recoveryblockcount;
          for (u32 filenumber=0; filenumber<recoveryfilecount; filenumber++)
          {
            u32 number = std::min(lowblockcount, blocks);
            fileallocations[filenumber].exponent = exponent;
            fileallocations[filenumber].count = number;
            exponent += number;
            blocks -= number;
            lowblockcount <<= 1;
          }
        }
        break;

      case scLimited:
        {
          // Files will be allocated in an exponential fashion but the
          // Maximum file size will be limited.

          u32 largest = (u32)((largestfilesize + blocksize-1) / blocksize);
          u32 filenumber = recoveryfilecount;
          u32 blocks = recoveryblockcount;

          exponent = firstrecoveryblock + recoveryblockcount;

          // Allocate uniformly at the top
          while (blocks >= 2*largest && filenumber > 0)
          {
            filenumber--;
            exponent -= largest;
            blocks -= largest;

            fileallocations[filenumber].exponent = exponent;
            fileallocations[filenumber].count = largest;
          }
          assert(blocks > 0 && filenumber > 0);

          exponent = firstrecoveryblock;
          u32 count = 1;
          u32 files = filenumber;

          // Allocate exponentially at the bottom
          for (filenumber=0; filenumber<files; filenumber++)
          {
            u32 number = std::min(count, blocks);
            fileallocations[filenumber].exponent = exponent;
            fileallocations[filenumber].count = number;

            exponent += number;
            blocks -= number;
            count <<= 1;
          }
        }
        break;
      }
    }

     // There will be an extra file with no recovery blocks.
    fileallocations[recoveryfilecount].exponent = exponent;
    fileallocations[recoveryfilecount].count = 0;

    // Determine digit widths for recovery filenames
    u32 digitsLow = 1, digitsCount = 1;
    {
      u32 limitLow = 0;
      u32 limitCount = 0;
      for (u32 filenumber=0; filenumber<=recoveryfilecount; filenumber++)
      {
        if (limitLow < fileallocations[filenumber].exponent)
        {
          limitLow = fileallocations[filenumber].exponent;
        }
        if (limitCount < fileallocations[filenumber].count)
        {
          limitCount = fileallocations[filenumber].count;
        }
      }

      for (u32 t=limitLow; t>=10; t/=10)
      {
        digitsLow++;
      }

      for (u32 t=limitCount; t>=10; t/=10)
      {
        digitsCount++;
      }
    }

    // Set the filenames
    for (u32 filenumber=0; filenumber<recoveryfilecount; filenumber++)
    {
      std::ostringstream filename;
      filename << parfilename
               << ".vol" << std::setw(digitsLow) << std::setfill('0') << fileallocations[filenumber].exponent
               << "+" << std::setw(digitsCount) << std::setfill('0') << fileallocations[filenumber].count
               << ".par2";

      if (filename.str().length() > _MAX_PATH)
      {
        serr << filename.str() << " pathlength is more than " << _MAX_PATH << "." << std::endl;
        return false;
      }
      fileallocations[filenumber].filename = filename.str();
    }

    std::string mainpar = parfilename + ".par2";
    if (mainpar.length() > _MAX_PATH)
    {
      serr << mainpar << " pathlength is more than " << _MAX_PATH << "." << std::endl;
      return false;
    }
    fileallocations[recoveryfilecount].filename = mainpar;
  }

  // Allocate the recovery files
  {
    recoveryfiles.resize(recoveryfilecount+1, DiskFile(sout, serr, &errorlog)); // pass default constructor.

    // Sort critical packets, so we get consistency.
    criticalpackets.sort(CriticalPacket::CompareLess);

    // Allocate packets to the output files
    {
      const MD5Hash &setid = mainpacket->SetId();
      std::vector<RecoveryPacket>::iterator recoverypacket = recoverypackets.begin();

      std::vector<DiskFile>::iterator recoveryfile = recoveryfiles.begin();
      std::vector<FileAllocation>::iterator fileallocation = fileallocations.begin();

      // For each recovery file:
      while (recoveryfile != recoveryfiles.end())
      {
        // How many recovery blocks in this file
        u32 count = fileallocation->count;

        // start at the beginning of the recovery file
        u64 offset = 0;

        if (count == 0)
        {
          // Write one set of critical packets
          std::list<CriticalPacket*>::const_iterator nextCriticalPacket = criticalpackets.begin();

          while (nextCriticalPacket != criticalpackets.end())
          {
            criticalpacketentries.push_back(CriticalPacketEntry(&*recoveryfile,
                                                                offset,
                                                                *nextCriticalPacket));
            offset += (*nextCriticalPacket)->PacketLength();

            ++nextCriticalPacket;
          }
        }
        else
        {
          // How many copies of each critical packet
          u32 copies = 0;
          for (u32 t=count; t>0; t>>=1)
          {
            copies++;
          }

          // Get ready to iterate through the critical packets
          u64 packetCount = 0;
          std::list<CriticalPacket*>::const_iterator nextCriticalPacket = criticalpackets.end();

          // What is the first exponent
          u32 exponent = fileallocation->exponent;

          // Start allocating the recovery packets
          u32 limit = exponent + count;
          while (exponent < limit)
          {
            // Add the next recovery packet
            recoverypacket->Create(&*recoveryfile, offset, blocksize, exponent, setid);

            offset += recoverypacket->PacketLength();
            ++recoverypacket;
            ++exponent;

            // Add some critical packets
            packetCount += copies * criticalpackets.size();
            while (packetCount >= count)
            {
              if (nextCriticalPacket == criticalpackets.end()) nextCriticalPacket = criticalpackets.begin();
              criticalpacketentries.push_back(CriticalPacketEntry(&*recoveryfile,
                                                                  offset,
                                                                  *nextCriticalPacket));
              offset += (*nextCriticalPacket)->PacketLength();
              ++nextCriticalPacket;

              packetCount -= count;
            }
          }
        }

        // Add one copy of the creator packet
        criticalpacketentries.push_back(CriticalPacketEntry(&*recoveryfile,
                                                            offset,
                                                            creatorpacket));
        offset += creatorpacket->PacketLength();

        // Create the file on disk and make it the required size
        if (IsCancelled() || !recoveryfile->Create(fileallocation->filename, offset))
          return false;

        ++recoveryfile;
        ++fileallocation;
      }
    }
  }

  return true;
}

// Delete every recovery file created so far, so that a create which stops
// part way leaves nothing of the set behind.
//
// The vector itself is left in place: recoverypackets and criticalpacketentries
// hold pointers into it.
void Par2CreatorEngine::DeleteIncompleteRecoveryFiles(void)
{
  for (std::vector<DiskFile>::iterator recoveryfile = recoveryfiles.begin();
       recoveryfile != recoveryfiles.end();
       ++recoveryfile)
  {
    // The allocation loop may not have reached this one
    if (!recoveryfile->Exists())
      continue;

    if (recoveryfile->IsOpen())
      recoveryfile->Close();

    recoveryfile->Delete();
  }
}

// Allocate memory buffers for reading and writing data to disk.
bool Par2CreatorEngine::AllocateBuffers(size_t memorylimit)
{
  transferbuffer = new u8[chunksize * NUM_TRANSFER_BUFFERS];
  outputbuffer = new u8[chunksize];

  if (transferbuffer == NULL || outputbuffer == NULL)
  {
    serr << "Could not allocate buffer memory." << std::endl;
    errorlog.Record(ecOutOfMemory, "Could not allocate the transfer buffers");
    return false;
  }

  ProcessorConfig config;
  config.numthreads = totalthreads;
  config.memorylimit = memorylimit;

  processor = backends.processor
    ? backends.processor(config)
    : std::unique_ptr<Processor>(new ReferenceProcessor(rs, totalthreads));

  if (!processor)
  {
    serr << "Could not allocate buffer memory." << std::endl;
    errorlog.Record(ecProcessorFailed, "The processor the application supplied built nothing");
    return false;
  }

  if (!processor->Init(chunksize, recoveryblockcount))
  {
    serr << "Could not allocate buffer memory." << std::endl;
    errorlog.Record(ecOutOfMemory, "The processor could not allocate its buffers");
    return false;
  }

  return true;
}

// Compute the Reed Solomon matrix
bool Par2CreatorEngine::ComputeRSMatrix(void)
{
  // Set the number of input blocks
  if (!rs.SetInput(sourceblockcount, sout, serr))
    return false;

  // Set the number of output blocks to be created
  if (!rs.SetOutput(false,
                    (u16)firstrecoveryblock,
                    (u16)firstrecoveryblock + (u16)(recoveryblockcount-1)))
    return false;

  // Compute the RS matrix
  if (!rs.Compute(noiselevel, sout, serr, observer))
    return false;

  return true;
}

// Read source data, process it through the RS matrix and write it to disk.
bool Par2CreatorEngine::ProcessData(u64 blockoffset, size_t blocklength, ProgressMeter<u64> &progress)
{
  processor->SetChunkLength(blocklength);
  processor->ResetOutput();

  // Offer the exponents, so that an implementation able to work out its own
  // coefficients is not made to read a column out of the matrix.
  std::vector<u16> exponents(recoveryblockcount);
  for (u32 recoveryblock=0; recoveryblock<recoveryblockcount; recoveryblock++)
    exponents[recoveryblock] = (u16)(firstrecoveryblock + recoveryblock);

  const bool ownfactors = processor->OfferRecoveryExponents((u32)sourceblocks.size(), exponents.data(), recoveryblockcount);

  // The matrix column for one input block, unused when the processor has its own
  std::vector<u16> factors(ownfactors ? 0 : recoveryblockcount);

  // Every buffer starts free
  std::future<void> bufferfree[NUM_TRANSFER_BUFFERS];
  for (u32 buffer=0; buffer<NUM_TRANSFER_BUFFERS; buffer++)
  {
    std::promise<void> free;
    free.set_value();
    bufferfree[buffer] = free.get_future();
  }
  u32 bufferindex = 0;

  // If we have deferred computation of the file hash and block crc and hashes
  // sourcefile and sourceindex will be used to update them during
  // the main recovery block computation
  std::vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
  u32 sourceindex = 0;

  std::vector<DataBlock>::iterator sourceblock;
  u32 inputblock;

  DiskFile *lastopenfile = NULL;

  // For each input block
  for ((sourceblock=sourceblocks.begin()),(inputblock=0);
       sourceblock != sourceblocks.end();
       ++sourceblock, ++inputblock)
  {
    if (IsCancelled())
    {
      if (lastopenfile != NULL)
        lastopenfile->Close();

      return false;
    }

    // Are we reading from a new file?
    if (lastopenfile != (*sourceblock).GetDiskFile())
    {
      // Close the last file
      if (lastopenfile != NULL)
      {
        lastopenfile->Close();
      }

      // Open the new file
      lastopenfile = (*sourceblock).GetDiskFile();
      if (!lastopenfile->Open())
      {
        return false;
      }
    }

    // Wait for the next input buffer to come free
    void *inputbuffer = &((u8*)transferbuffer)[chunksize * bufferindex];
    bufferfree[bufferindex].get();

    // Read data from the current input block
    if (!sourceblock->ReadData(blockoffset, blocklength, inputbuffer))
      return false;

    if (deferhashcomputation)
    {
      assert(blockoffset == 0 && blocklength == blocksize);
      assert(sourcefile != sourcefiles.end());

      (*sourcefile)->UpdateHashes(sourceindex, inputbuffer, blocklength);
    }

    // Look up the matrix column and process the data against every output block
    if (!ownfactors)
    {
      for (u32 outputblock=0; outputblock<recoveryblockcount; outputblock++)
        factors[outputblock] = rs.GetFactor(inputblock, outputblock);
    }

    processor->WaitForAdd();
    bufferfree[bufferindex] = processor->AddInput(inputbuffer, blocklength, inputblock,
                                                  ownfactors ? NULL : factors.data());
    bufferindex = (bufferindex + 1) % NUM_TRANSFER_BUFFERS;

    progress.Add(blocklength);

    // Work out which source file the next block belongs to
    if (++sourceindex >= (*sourcefile)->BlockCount())
    {
      sourceindex = 0;
      ++sourcefile;
    }
  }

  processor->EndInput();

  // Close the last file
  if (lastopenfile != NULL)
  {
    lastopenfile->Close();
  }

  if (noiselevel > nlQuiet)
    sout << "Writing recovery packets\r";

  // For each output block
  for (u32 outputblock=0; outputblock<recoveryblockcount;outputblock++)
  {
    if (IsCancelled())
      return false;

    // Take the accumulated output block from the processor
    const void *outbuf = processor->PeekOutput(outputblock);
    if (outbuf == NULL)
    {
      if (!processor->GetOutput(outputblock, outputbuffer))
      {
        serr << "Could not read the recovery data back from the processor." << std::endl;
        return false;
      }
      outbuf = outputbuffer;
    }

    // Write the data to the recovery packet
    if (!recoverypackets[outputblock].WriteData(blockoffset, blocklength, outbuf))
      return false;
  }

  if (noiselevel > nlQuiet)
    sout << "Wrote " << recoveryblockcount * blocklength << " bytes to disk" << std::endl;

  return true;
}

// Finish computation of the recovery packets and write the headers to disk.
bool Par2CreatorEngine::WriteRecoveryPacketHeaders(void)
{
  // For each recovery packet
  for (std::vector<RecoveryPacket>::iterator recoverypacket = recoverypackets.begin();
       recoverypacket != recoverypackets.end();
       ++recoverypacket)
  {
    // Finish the packet header and write it to disk
    if (!recoverypacket->WriteHeader())
      return false;
  }

  return true;
}

bool Par2CreatorEngine::FinishFileHashComputation(void)
{
  // If we deferred the computation of the full file hash, then we finish it now
  if (deferhashcomputation)
  {
    // For each source file
    std::vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();

    while (sourcefile != sourcefiles.end())
    {
      (*sourcefile)->FinishHashes();

      ++sourcefile;
    }
  }

  return true;
}

// Fill in all remaining details in the critical packets.
bool Par2CreatorEngine::FinishCriticalPackets(void)
{
  // Get the setid from the main packet
  const MD5Hash &setid = mainpacket->SetId();

  for (std::list<CriticalPacket*>::iterator criticalpacket=criticalpackets.begin();
       criticalpacket!=criticalpackets.end();
       criticalpacket++)
  {
    // Store the setid in each of the critical packets
    // and compute the packet_hash of each one.

    (*criticalpacket)->FinishPacket(setid);
  }

  return true;
}

// Write all other critical packets to disk.
bool Par2CreatorEngine::WriteCriticalPackets(void)
{
  std::list<CriticalPacketEntry>::const_iterator packetentry = criticalpacketentries.begin();

  // For each critical packet
  while (packetentry != criticalpacketentries.end())
  {
    // Write it to disk
    if (IsCancelled() || !packetentry->WritePacket())
      return false;

    ++packetentry;
  }

  return true;
}

// Close all files.
bool Par2CreatorEngine::CloseFiles(void)
{
//  // Close each source file.
//  for (std::vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
//       sourcefile != sourcefiles.end();
//       ++sourcefile)
//  {
//    (*sourcefile)->Close();
//  }

  // Close each recovery file.
  for (std::vector<DiskFile>::iterator recoveryfile = recoveryfiles.begin();
       recoveryfile != recoveryfiles.end();
       ++recoveryfile)
  {
    recoveryfile->Close();
  }

  return true;
}

} // namespace par2
