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

#include "par2cmdline.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

Par2Creator::Par2Creator(void)
: noiselevel(CommandLine::nlUnknown)
, blocksize(0)
, chunksize(0)
, inputbuffer(0)
, outputbuffer(0)

, sourcefilecount(0)
, sourceblockcount(0)

, largestfilesize(0)
, recoveryfilescheme(CommandLine::scUnknown)
, recoveryfilecount(0)
, recoveryblockcount(0)
, firstrecoveryblock(0)

, mainpacket(0)
, creatorpacket(0)

, deferhashcomputation(false)
{
}

Par2Creator::~Par2Creator(void)
{
  delete mainpacket;
  delete creatorpacket;

  delete [] (u8*)inputbuffer;
  delete [] (u8*)outputbuffer;

  vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
  while (sourcefile != sourcefiles.end())
  {
    delete *sourcefile;
    ++sourcefile;
  }
}

Result Par2Creator::Process(const CommandLine &commandline)
{
  // Get information from commandline
  noiselevel = commandline.GetNoiseLevel();
  blocksize = commandline.GetBlockSize();
  sourceblockcount = commandline.GetBlockCount();
  const list<CommandLine::ExtraFile> extrafiles = commandline.GetExtraFiles();
  sourcefilecount = (u32)extrafiles.size();
  u32 redundancy = commandline.GetRedundancy();
  u64 redundancysize = commandline.GetRedundancySize();
  recoveryblockcount = commandline.GetRecoveryBlockCount();
  recoveryfilecount = commandline.GetRecoveryFileCount();
  firstrecoveryblock = commandline.GetFirstRecoveryBlock();
  recoveryfilescheme = commandline.GetRecoveryFileScheme();
  string par2filename = commandline.GetParFilename();
  string basepath = commandline.GetBasePath();
  size_t memorylimit = commandline.GetMemoryLimit();
  largestfilesize = commandline.GetLargestSourceSize();

  // Compute block size from block count or vice versa depending on which was
  // specified on the command line
  if (!ComputeBlockSizeAndBlockCount(extrafiles))
    return eInvalidCommandLineArguments;

  // Determine how many recovery blocks to create based on the source block
  // count and the requested level of redundancy.
  if ((redundancy > 0 || redundancysize >0) && !ComputeRecoveryBlockCount(redundancy, redundancysize))
    return eInvalidCommandLineArguments;

  // Determine how much recovery data can be computed on one pass
  if (!CalculateProcessBlockSize(memorylimit))
    return eLogicError;

  // Determine how many recovery files to create.
  if (!ComputeRecoveryFileCount())
    return eInvalidCommandLineArguments;

  if (noiselevel > CommandLine::nlQuiet)
  {
    // Display information.
    cout << "Block size: " << blocksize << endl;
    cout << "Source file count: " << sourcefilecount << endl;
    cout << "Source block count: " << sourceblockcount << endl;
    if (redundancy>0 || recoveryblockcount==0)
      cout << "Redundancy: " << redundancy << '%' << endl;
    cout << "Recovery block count: " << recoveryblockcount << endl;
    cout << "Recovery file count: " << recoveryfilecount << endl;
    cout << endl;
  }

  // Open all of the source files, compute the Hashes and CRC values, and store
  // the results in the file verification and file description packets.
  if (!OpenSourceFiles(extrafiles, basepath))
    return eFileIOError;

  // Create the main packet and determine the setid to use with all packets
  if (!CreateMainPacket())
    return eLogicError;

  // Create the creator packet.
  if (!CreateCreatorPacket())
    return eLogicError;

  // Initialise all of the source blocks ready to start reading data from the source files.
  if (!CreateSourceBlocks())
    return eLogicError;

  // Create all of the output files and allocate all packets to appropriate file offets.
  if (!InitialiseOutputFiles(par2filename))
    return eFileIOError;

  if (recoveryblockcount > 0)
  {
    // Allocate memory buffers for reading and writing data to disk.
    if (!AllocateBuffers())
      return eMemoryError;

    // Compute the Reed Solomon matrix
    if (!ComputeRSMatrix())
      return eLogicError;

    // Set the total amount of data to be processed.
    progress = 0;
    totaldata = blocksize * sourceblockcount * recoveryblockcount;

    // Start at an offset of 0 within a block.
    u64 blockoffset = 0;
    while (blockoffset < blocksize) // Continue until the end of the block.
    {
      // Work out how much data to process this time.
      size_t blocklength = (size_t)min((u64)chunksize, blocksize-blockoffset);

      // Read source data, process it through the RS matrix and write it to disk.
      if (!ProcessData(blockoffset, blocklength))
        return eFileIOError;

      blockoffset += blocklength;
    }

    if (noiselevel > CommandLine::nlQuiet)
      cout << "Writing recovery packets" << endl;

    // Finish computation of the recovery packets and write the headers to disk.
    if (!WriteRecoveryPacketHeaders())
      return eFileIOError;

    // Finish computing the full file hash values of the source files
    if (!FinishFileHashComputation())
      return eLogicError;
  }

  // Fill in all remaining details in the critical packets.
  if (!FinishCriticalPackets())
    return eLogicError;

  if (noiselevel > CommandLine::nlQuiet)
    cout << "Writing verification packets" << endl;

  // Write all other critical packets to disk.
  if (!WriteCriticalPackets())
    return eFileIOError;

  // Close all files.
  if (!CloseFiles())
    return eFileIOError;

  if (noiselevel > CommandLine::nlSilent)
    cout << "Done" << endl;

  return eSuccess;
}

// Compute block size from block count or vice versa depending on which was
// specified on the command line
bool Par2Creator::ComputeBlockSizeAndBlockCount(const list<CommandLine::ExtraFile> &extrafiles)
{
  // Determine blocksize from sourceblockcount or vice-versa
  if (blocksize > 0)
  {
    u64 count = 0;

    for (ExtraFileIterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
    {
      count += (i->FileSize() + blocksize-1) / blocksize;
    }

    if (count > 32768)
    {
      cerr << "Block size is too small. It would require " << count << "blocks." << endl;
      return false;
    }

    sourceblockcount = (u32)count;
  }
  else if (sourceblockcount > 0)
  {
    if (sourceblockcount < extrafiles.size())
    {
      // The block count cannot be less than the number of files.

      cerr << "Block count (" << sourceblockcount <<
              ") cannot be smaller than the number of files(" << extrafiles.size() << "). " << endl;
      return false;
    }
    else if (sourceblockcount == extrafiles.size())
    {
      // If the block count is the same as the number of files, then the block
      // size is the size of the largest file (rounded up to a multiple of 4).

      u64 largestsourcesize = 0;
      for (ExtraFileIterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
      {
        if (largestsourcesize < i->FileSize())
        {
          largestsourcesize = i->FileSize();
        }
      }

      blocksize = (largestsourcesize + 3) & ~3;
    }
    else
    {
      u64 totalsize = 0;
      for (ExtraFileIterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
      {
        totalsize += (i->FileSize() + 3) / 4;
      }

      if (sourceblockcount > totalsize)
      {
        sourceblockcount = (u32)totalsize;
        blocksize = 4;
      }
      else
      {
        // Absolute lower bound and upper bound on the source block size that will
        // result in the requested source block count.
        u64 lowerBound = totalsize / sourceblockcount;
        u64 upperBound = (totalsize + sourceblockcount - extrafiles.size() - 1) / (sourceblockcount - extrafiles.size());

        u64 count = 0;
        u64 size;

        do
        {
          size = (lowerBound + upperBound)/2;

          count = 0;
          for (ExtraFileIterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
          {
            count += ((i->FileSize()+3)/4 + size-1) / size;
          }
          if (count > sourceblockcount)
          {
            lowerBound = size+1;
            if (lowerBound >= upperBound)
            {
              size = lowerBound;
              count = 0;
              for (ExtraFileIterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
              {
                count += ((i->FileSize()+3)/4 + size-1) / size;
              }
            }
          }
          else
          {
            upperBound = size;
          }
        }
        while (lowerBound < upperBound);

        if (count > 32768)
        {
          cerr << "Error calculating block size. cannot be higher than 32768." << endl;
          return false;
        }
        else if (count == 0)
        {
          cerr << "Error calculating block size. cannot be 0." << endl;
          return false;
        }

        sourceblockcount = (u32)count;
        blocksize = size*4;
      }
    }
  }

  return true;
}


// Determine how many recovery blocks to create based on the source block
// count and the requested level of redundancy.
bool Par2Creator::ComputeRecoveryBlockCount(u32 redundancy, u64 redundancysize)
{
  if (redundancy > 0)
  {
    // Determine recoveryblockcount
    recoveryblockcount = (sourceblockcount * redundancy + 50) / 100;
  }
  else if (redundancysize > 0)
  {
    bool closest = false;

    u64 overhead;
    overhead = sourceblockcount * 21;

    u32 estimatedFileCount;
    u64 totalOverhead;

    do
    {
      if (recoveryfilecount == 0)
      {
        estimatedFileCount = 15;
      }
      else
      {
        closest = true;
        estimatedFileCount = recoveryfilecount + 1;
      }
      totalOverhead = estimatedFileCount * overhead;
      if (totalOverhead > redundancysize)
      {
        recoveryblockcount = 1;
      }
      else
      {
        recoveryblockcount = (redundancysize - totalOverhead) / (blocksize + 70);
      }
      ComputeRecoveryFileCount();
    }
    while(closest == false);

    recoveryfilecount = 0;
  }
  else
  {
    cerr << "Redundancy and Redundancysize not set." << endl;
    return false;
  }

  // Force valid values if necessary
  if (recoveryblockcount == 0 && redundancy > 0)
    recoveryblockcount = 1;

  if (recoveryblockcount > 65536)
  {
    cerr << "Too many recovery blocks requested." << endl;
    return false;
  }

  // Check that the last recovery block number would not be too large
  if (firstrecoveryblock + recoveryblockcount >= 65536)
  {
    cerr << "First recovery block number is too high." << endl;
    return false;
  }

      cout << endl;
  return true;
}

// Determine how much recovery data can be computed on one pass
bool Par2Creator::CalculateProcessBlockSize(size_t memorylimit)
{
  // Are we computing any recovery blocks
  if (recoveryblockcount == 0)
  {
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
  }

  return true;
}

// Determine how many recovery files to create.
bool Par2Creator::ComputeRecoveryFileCount(void)
{
  // Are we computing any recovery blocks
  if (recoveryblockcount == 0)
  {
    recoveryfilecount = 0;
    return true;
  }

  switch (recoveryfilescheme)
  {
  case CommandLine::scUnknown:
    {
      assert(false);
      return false;
    }
    break;
  case CommandLine::scVariable:
  case CommandLine::scUniform:
    {
      if (recoveryfilecount == 0)
      {
        // If none specified then then filecount is roughly log2(blockcount)
        // This prevents you getting excessively large numbers of files
        // when the block count is high and also allows the files to have
        // sizes which vary exponentially.

        for (u32 blocks=recoveryblockcount; blocks>0; blocks>>=1)
        {
          recoveryfilecount++;
        }
      }

      if (recoveryfilecount > recoveryblockcount)
      {
        // You cannot have move recovery files that there are recovery blocks
        // to put in them.
        cerr << "Too many recovery files specified." << endl;
        return false;
      }
    }
    break;

  case CommandLine::scLimited:
    {
      // No recovery file will contain more recovery blocks than would
      // be required to reconstruct the largest source file if it
      // were missing. Other recovery files will have recovery blocks
      // distributed in an exponential scheme.

      u32 largest = (u32)((largestfilesize + blocksize-1) / blocksize);
      u32 whole = recoveryblockcount / largest;
      whole = (whole >= 1) ? whole-1 : 0;

      u32 extra = recoveryblockcount - whole * largest;
      recoveryfilecount = whole;
      for (u32 blocks=extra; blocks>0; blocks>>=1)
      {
        recoveryfilecount++;
      }
    }
    break;
  }

  return true;
}

// Open all of the source files, compute the Hashes and CRC values, and store
// the results in the file verification and file description packets.
bool Par2Creator::OpenSourceFiles(const list<CommandLine::ExtraFile> &extrafiles, string basepath)
{
  ExtraFileIterator extrafile = extrafiles.begin();
  while (extrafile != extrafiles.end())
  {
    Par2CreatorSourceFile *sourcefile = new Par2CreatorSourceFile;

    string name;
    DiskFile::SplitRelativeFilename(extrafile->FileName(), basepath, name);

    if (noiselevel > CommandLine::nlSilent)
      cout << "Opening: " << name << endl;

    // Open the source file and compute its Hashes and CRCs.
    if (!sourcefile->Open(noiselevel, *extrafile, blocksize, deferhashcomputation, basepath))
    {
      delete sourcefile;
      return false;
    }

    // Record the file verification and file description packets
    // in the critical packet list.
    sourcefile->RecordCriticalPackets(criticalpackets);

    // Add the source file to the sourcefiles array.
    sourcefiles.push_back(sourcefile);

    // Close the source file until its needed
    sourcefile->Close();

    ++extrafile;
  }

  return true;
}

// Create the main packet and determine the setid to use with all packets
bool Par2Creator::CreateMainPacket(void)
{
  // Construct the main packet from the list of source files and the block size.
  mainpacket = new MainPacket;

  // Add the main packet to the list of critical packets.
  criticalpackets.push_back(mainpacket);

  // Create the packet (sourcefiles will get sorted into FileId order).
  return mainpacket->Create(sourcefiles, blocksize);
}

// Create the creator packet.
bool Par2Creator::CreateCreatorPacket(void)
{
  // Construct the creator packet
  creatorpacket = new CreatorPacket;

  // Create the packet
  return creatorpacket->Create(mainpacket->SetId());
}

// Initialise all of the source blocks ready to start reading data from the source files.
bool Par2Creator::CreateSourceBlocks(void)
{
  // Allocate the array of source blocks
  sourceblocks.resize(sourceblockcount);

  vector<DataBlock>::iterator sourceblock = sourceblocks.begin();

  for (vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
       sourcefile!= sourcefiles.end();
       sourcefile++)
  {
    // Allocate the appopriate number of source blocks to each source file.
    // sourceblock will be advanced.

    (*sourcefile)->InitialiseSourceBlocks(sourceblock, blocksize);
  }

  return true;
}

class FileAllocation
{
public:
  FileAllocation(void)
  {
    filename = "";
    exponent = 0;
    count = 0;
  }

  string filename;
  u32 exponent;
  u32 count;
};

// Create all of the output files and allocate all packets to appropriate file offets.
bool Par2Creator::InitialiseOutputFiles(string par2filename)
{
  // Allocate the recovery packets
  recoverypackets.resize(recoveryblockcount);

  // Choose filenames and decide which recovery blocks to place in each file
  vector<FileAllocation> fileallocations;
  fileallocations.resize(recoveryfilecount+1); // One extra file with no recovery blocks
  {
    // Decide how many recovery blocks to place in each file
    u32 exponent = firstrecoveryblock;
    if (recoveryfilecount > 0)
    {
      switch (recoveryfilescheme)
      {
      case CommandLine::scUnknown:
        {
          assert(false);
          return false;
        }
        break;
      case CommandLine::scUniform:
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

      case CommandLine::scVariable:
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
            u32 number = min(lowblockcount, blocks);
            fileallocations[filenumber].exponent = exponent;
            fileallocations[filenumber].count = number;
            exponent += number;
            blocks -= number;
            lowblockcount <<= 1;
          }
        }
        break;

      case CommandLine::scLimited:
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
            u32 number = min(count, blocks);
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

    // Determine the format to use for filenames of recovery files
    char filenameformat[_MAX_PATH];
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

      u32 digitsLow = 1;
      for (u32 t=limitLow; t>=10; t/=10)
      {
        digitsLow++;
      }

      u32 digitsCount = 1;
      for (u32 t=limitCount; t>=10; t/=10)
      {
        digitsCount++;
      }

      sprintf(filenameformat, "%%s.vol%%0%dd+%%0%dd.par2", digitsLow, digitsCount);
    }

    // Set the filenames
    for (u32 filenumber=0; filenumber<recoveryfilecount; filenumber++)
    {
      char filename[_MAX_PATH];
      snprintf(filename, sizeof(filename), filenameformat, par2filename.c_str(), fileallocations[filenumber].exponent, fileallocations[filenumber].count);
      fileallocations[filenumber].filename = filename;
    }
    fileallocations[recoveryfilecount].filename = par2filename + ".par2";
  }

  // Allocate the recovery files
  {
    recoveryfiles.resize(recoveryfilecount+1);

    // Allocate packets to the output files
    {
      const MD5Hash &setid = mainpacket->SetId();
      vector<RecoveryPacket>::iterator recoverypacket = recoverypackets.begin();

      vector<DiskFile>::iterator recoveryfile = recoveryfiles.begin();
      vector<FileAllocation>::iterator fileallocation = fileallocations.begin();

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
          list<CriticalPacket*>::const_iterator nextCriticalPacket = criticalpackets.begin();

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
          u32 packetCount = 0;
          list<CriticalPacket*>::const_iterator nextCriticalPacket = criticalpackets.end();

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
        if (!recoveryfile->Create(fileallocation->filename, offset))
          return false;

        ++recoveryfile;
        ++fileallocation;
      }
    }
  }

  return true;
}

// Allocate memory buffers for reading and writing data to disk.
bool Par2Creator::AllocateBuffers(void)
{
  inputbuffer = new u8[chunksize];
  outputbuffer = new u8[chunksize * recoveryblockcount];

  if (inputbuffer == NULL || outputbuffer == NULL)
  {
    cerr << "Could not allocate buffer memory." << endl;
    return false;
  }

  return true;
}

// Compute the Reed Solomon matrix
bool Par2Creator::ComputeRSMatrix(void)
{
  // Set the number of input blocks
  if (!rs.SetInput(sourceblockcount))
    return false;

  // Set the number of output blocks to be created
  if (!rs.SetOutput(false,
                    (u16)firstrecoveryblock,
                    (u16)firstrecoveryblock + (u16)(recoveryblockcount-1)))
    return false;

  // Compute the RS matrix
  if (!rs.Compute(noiselevel))
    return false;

  return true;
}

// Read source data, process it through the RS matrix and write it to disk.
bool Par2Creator::ProcessData(u64 blockoffset, size_t blocklength)
{
  // Clear the output buffer
  memset(outputbuffer, 0, chunksize * recoveryblockcount);

  // If we have defered computation of the file hash and block crc and hashes
  // sourcefile and sourceindex will be used to update them during
  // the main recovery block computation
  vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
  u32 sourceindex = 0;

  vector<DataBlock>::iterator sourceblock;
  u32 inputblock;

  DiskFile *lastopenfile = NULL;

  // For each input block
  for ((sourceblock=sourceblocks.begin()),(inputblock=0);
       sourceblock != sourceblocks.end();
       ++sourceblock, ++inputblock)
  {
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

    // Read data from the current input block
    if (!sourceblock->ReadData(blockoffset, blocklength, inputbuffer))
      return false;

    if (deferhashcomputation)
    {
      assert(blockoffset == 0 && blocklength == blocksize);
      assert(sourcefile != sourcefiles.end());

      (*sourcefile)->UpdateHashes(sourceindex, inputbuffer, blocklength);
    }

    // For each output block
    for (u32 outputblock=0; outputblock<recoveryblockcount; outputblock++)
    {
      // Select the appropriate part of the output buffer
      void *outbuf = &((u8*)outputbuffer)[chunksize * outputblock];

      // Process the data through the RS matrix
      rs.Process(blocklength, inputblock, inputbuffer, outputblock, outbuf);

      if (noiselevel > CommandLine::nlQuiet)
      {
        // Update a progress indicator
        u32 oldfraction = (u32)(1000 * progress / totaldata);
        progress += blocklength;
        u32 newfraction = (u32)(1000 * progress / totaldata);

        if (oldfraction != newfraction)
        {
          cout << "Processing: " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;
        }
      }
    }

    // Work out which source file the next block belongs to
    if (++sourceindex >= (*sourcefile)->BlockCount())
    {
      sourceindex = 0;
      ++sourcefile;
    }
  }

  // Close the last file
  if (lastopenfile != NULL)
  {
    lastopenfile->Close();
  }

  if (noiselevel > CommandLine::nlQuiet)
    cout << "Writing recovery packets\r";

  // For each output block
  for (u32 outputblock=0; outputblock<recoveryblockcount;outputblock++)
  {
    // Select the appropriate part of the output buffer
    char *outbuf = &((char*)outputbuffer)[chunksize * outputblock];

    // Write the data to the recovery packet
    if (!recoverypackets[outputblock].WriteData(blockoffset, blocklength, outbuf))
      return false;
  }

  if (noiselevel > CommandLine::nlQuiet)
    cout << "Wrote " << recoveryblockcount * blocklength << " bytes to disk" << endl;

  return true;
}

// Finish computation of the recovery packets and write the headers to disk.
bool Par2Creator::WriteRecoveryPacketHeaders(void)
{
  // For each recovery packet
  for (vector<RecoveryPacket>::iterator recoverypacket = recoverypackets.begin();
       recoverypacket != recoverypackets.end();
       ++recoverypacket)
  {
    // Finish the packet header and write it to disk
    if (!recoverypacket->WriteHeader())
      return false;
  }

  return true;
}

bool Par2Creator::FinishFileHashComputation(void)
{
  // If we defered the computation of the full file hash, then we finish it now
  if (deferhashcomputation)
  {
    // For each source file
    vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();

    while (sourcefile != sourcefiles.end())
    {
      (*sourcefile)->FinishHashes();

      ++sourcefile;
    }
  }

  return true;
}

// Fill in all remaining details in the critical packets.
bool Par2Creator::FinishCriticalPackets(void)
{
  // Get the setid from the main packet
  const MD5Hash &setid = mainpacket->SetId();

  for (list<CriticalPacket*>::iterator criticalpacket=criticalpackets.begin();
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
bool Par2Creator::WriteCriticalPackets(void)
{
  list<CriticalPacketEntry>::const_iterator packetentry = criticalpacketentries.begin();

  // For each critical packet
  while (packetentry != criticalpacketentries.end())
  {
    // Write it to disk
    if (!packetentry->WritePacket())
      return false;

    ++packetentry;
  }

  return true;
}

// Close all files.
bool Par2Creator::CloseFiles(void)
{
//  // Close each source file.
//  for (vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
//       sourcefile != sourcefiles.end();
//       ++sourcefile)
//  {
//    (*sourcefile)->Close();
//  }

  // Close each recovery file.
  for (vector<DiskFile>::iterator recoveryfile = recoveryfiles.begin();
       recoveryfile != recoveryfiles.end();
       ++recoveryfile)
  {
    recoveryfile->Close();
  }

  return true;
}
