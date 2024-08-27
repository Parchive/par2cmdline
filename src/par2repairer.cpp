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

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif


// static variable
#ifdef _OPENMP
u32 Par2Repairer::filethreads = _FILE_THREADS;
#endif


Par2Repairer::Par2Repairer(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel)
: sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, searchpath()
, basepath()
, setid()
, recoverypacketmap()
, diskFileMap()
, sourcefilemap()
, sourcefiles()
, verifylist()
, backuplist()
, par2list()
, sourceblocks()
, targetblocks()
, blockverifiable(false)
, verificationhashtable()
, unverifiablesourcefiles()
, inputblocks()
, copyblocks()
, outputblocks()
, rs()
{
  skipdata = false;
  skipleaway = 0;

  firstpacket = true;
  mainpacket = 0;
  creatorpacket = 0;

  blocksize = 0;
  chunksize = 0;

  sourceblockcount = 0;
  availableblockcount = 0;
  missingblockcount = 0;

  memset(windowtable, 0, sizeof(windowtable));

  blocksallocated = false;

  completefilecount = 0;
  renamedfilecount = 0;
  damagedfilecount = 0;
  missingfilecount = 0;

  inputbuffer = 0;
  outputbuffer = 0;

  progress = 0;
  totaldata = 0;

#ifdef _OPENMP
  mttotalsize = 0;
  mttotalextrasize = 0;
  mttotalprogress = 0;
  mtprocessingextrafiles = false;
#endif
}

Par2Repairer::~Par2Repairer(void)
{
  delete [] (u8*)inputbuffer;
  delete [] (u8*)outputbuffer;

  map<u32,RecoveryPacket*>::iterator rp = recoverypacketmap.begin();
  while (rp != recoverypacketmap.end())
  {
    delete (*rp).second;

    ++rp;
  }

  map<MD5Hash,Par2RepairerSourceFile*>::iterator sf = sourcefilemap.begin();
  while (sf != sourcefilemap.end())
  {
    Par2RepairerSourceFile *sourcefile = (*sf).second;
    delete sourcefile;

    ++sf;
  }

  delete mainpacket;
  delete creatorpacket;
}

Result Par2Repairer::Process(
			     const size_t memorylimit,
			     const string &_basepath,
#ifdef _OPENMP
			     const u32 nthreads,
			     const u32 _filethreads,
#endif
			     string parfilename,
			     const vector<string> &_extrafiles,
			     const bool dorepair,   // derived from operation
			     const bool purgefiles,
			     const bool _skipdata,
			     const u64 _skipleaway
			     )
{
#ifdef _OPENMP
  filethreads = _filethreads;
#endif

  // Should we skip data whilst scanning files
  skipdata = _skipdata;

  // How much leaway should we allow when scanning files
  skipleaway = _skipleaway;

  // Get filenames from the command line
  basepath = _basepath;
  std::vector<string> extrafiles = _extrafiles;

#ifdef _OPENMP
  // Set the number of threads
  if (nthreads != 0)
    omp_set_num_threads(nthreads);
#endif

  // Determine the searchpath from the location of the main PAR2 file
  string name;
  DiskFile::SplitFilename(parfilename, searchpath, name);

  par2list.push_back(parfilename);

  // Load packets from the main PAR2 file
  if (!LoadPacketsFromFile(searchpath + name))
    return eLogicError;

  // Load packets from other PAR2 files with names based on the original PAR2 file
  if (!LoadPacketsFromOtherFiles(parfilename))
    return eLogicError;

  // Load packets from any other PAR2 files whose names are given on the command line
  if (!LoadPacketsFromExtraFiles(extrafiles))
    return eLogicError;

  if (noiselevel > nlQuiet)
    sout << endl;

  // Check that the packets are consistent and discard any that are not
  if (!CheckPacketConsistency())
    return eInsufficientCriticalData;

  // Use the information in the main packet to get the source files
  // into the correct order and determine their filenames
  if (!CreateSourceFileList())
    return eLogicError;

  // Determine the total number of DataBlocks for the recoverable source files
  // The allocate the DataBlocks and assign them to each source file
  if (!AllocateSourceBlocks())
    return eLogicError;

  // Create a verification hash table for all files for which we have not
  // found a complete version of the file and for which we have
  // a verification packet
  if (!PrepareVerificationHashTable())
    return eLogicError;

  // Compute the table for the sliding CRC computation
  if (!ComputeWindowTable())
    return eLogicError;

  // Attempt to verify all of the source files
  if (!VerifySourceFiles(basepath, extrafiles))
    return eFileIOError;

  if (completefilecount < mainpacket->RecoverableFileCount())
  {
    // Scan any extra files specified on the command line
    if (!VerifyExtraFiles(extrafiles, basepath))
      return eLogicError;
  }

  // Find out how much data we have found
  UpdateVerificationResults();

  if (noiselevel > nlSilent)
    sout << endl;

  // Check the verification results and report the results
  if (!CheckVerificationResults())
    return eRepairNotPossible;

  // Are any of the files incomplete
  if (completefilecount < mainpacket->RecoverableFileCount())
  {
    // Do we want to carry out a repair
    if (dorepair)
    {
      if (noiselevel > nlSilent)
        sout << endl;

      // Rename any damaged or missnamed target files.
      if (!RenameTargetFiles())
        return eFileIOError;

      // Are we still missing any files
      if (completefilecount < mainpacket->RecoverableFileCount())
      {
        // Work out which files are being repaired, create them, and allocate
        // target DataBlocks to them, and remember them for later verification.
        if (!CreateTargetFiles())
          return eFileIOError;

        // Work out which data blocks are available, which need to be copied
        // directly to the output, and which need to be recreated, and compute
        // the appropriate Reed Solomon matrix.
        if (!ComputeRSmatrix())
        {
          // Delete all of the partly reconstructed files
          DeleteIncompleteTargetFiles();
          return eFileIOError;
        }

        if (noiselevel > nlSilent)
          sout << endl;

        // Allocate memory buffers for reading and writing data to disk.
        if (!AllocateBuffers(memorylimit))
        {
          // Delete all of the partly reconstructed files
          DeleteIncompleteTargetFiles();
          return eMemoryError;
        }

        // Set the total amount of data to be processed.
        progress = 0;
        totaldata = blocksize * sourceblockcount * (missingblockcount > 0 ? missingblockcount : 1);

        // Start at an offset of 0 within a block.
        u64 blockoffset = 0;
        while (blockoffset < blocksize) // Continue until the end of the block.
        {
          // Work out how much data to process this time.
          size_t blocklength = (size_t)min((u64)chunksize, blocksize-blockoffset);

          // Read source data, process it through the RS matrix and write it to disk.
          if (!ProcessData(blockoffset, blocklength))
          {
            // Delete all of the partly reconstructed files
            DeleteIncompleteTargetFiles();
            return eFileIOError;
          }

          // Advance to the need offset within each block
          blockoffset += blocklength;
        }

        if (noiselevel > nlSilent)
          sout << endl << "Verifying repaired files:" << endl << endl;

        // Verify that all of the reconstructed target files are now correct
        if (!VerifyTargetFiles(basepath))
        {
          // Delete all of the partly reconstructed files
          DeleteIncompleteTargetFiles();
          return eFileIOError;
        }
      }

      // Are all of the target files now complete?
      if (completefilecount<mainpacket->RecoverableFileCount())
      {
        serr << "Repair Failed." << endl;
        return eRepairFailed;
      }
      else
      {
        if (noiselevel > nlSilent)
          sout << endl << "Repair complete." << endl;
      }
    }
    else
    {
      return eRepairPossible;
    }
  }

  if (purgefiles == true)
  {
    RemoveBackupFiles();
    RemoveParFiles();
  }

  return eSuccess;
}

// Load the packets from the specified file
bool Par2Repairer::LoadPacketsFromFile(string filename)
{
  // Skip the file if it has already been processed
  if (diskFileMap.Find(filename) != 0)
  {
    return true;
  }

  DiskFile *diskfile = new DiskFile(sout, serr);

  // Open the file
  if (!diskfile->Open(filename))
  {
    // If we could not open the file, ignore the error and
    // proceed to the next file
    delete diskfile;
    return true;
  }

  if (noiselevel > nlSilent)
  {
    string path;
    string name;
    DiskFile::SplitFilename(filename, path, name);
    sout << "Loading \"" << utf8::console(name) << "\"." << endl;
  }

  // How many useable packets have we found
  u32 packets = 0;

  // How many recovery packets were there
  u32 recoverypackets = 0;

  // How big is the file
  u64 filesize = diskfile->FileSize();
  if (filesize > 0)
  {
    // Allocate a buffer to read data into
    // The buffer should be large enough to hold a whole
    // critical packet (i.e. file verification, file description, main,
    // and creator), but not necessarily a whole recovery packet.
    size_t buffersize = (size_t)min((u64)1048576, filesize);
    u8 *buffer = new u8[buffersize];

    // Progress indicator
    u64 progress = 0;

    // Start at the beginning of the file
    u64 offset = 0;

    // Continue as long as there is at least enough for the packet header
    while (offset + sizeof(PACKET_HEADER) <= filesize)
    {
      if (noiselevel > nlQuiet)
      {
        // Update a progress indicator
        u32 oldfraction = (u32)(1000 * progress / filesize);
        u32 newfraction = (u32)(1000 * offset / filesize);
        if (oldfraction != newfraction)
        {
          sout << "Loading: " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;
          progress = offset;
        }
      }

      // Attempt to read the next packet header
      PACKET_HEADER header;
      if (!diskfile->Read(offset, &header, sizeof(header)))
        break;

      // Does this look like it might be a packet
      if (packet_magic != header.magic)
      {
        offset++;

        // Is there still enough for at least a whole packet header
        while (offset + sizeof(PACKET_HEADER) <= filesize)
        {
          // How much can we read into the buffer
          size_t want = (size_t)min((u64)buffersize, filesize-offset);

          // Fill the buffer
          if (!diskfile->Read(offset, buffer, want))
          {
            offset = filesize;
            break;
          }

          // Scan the buffer for the magic value
          u8 *current = buffer;
          u8 *limit = &buffer[want-sizeof(PACKET_HEADER)];
          while (current <= limit && packet_magic != ((PACKET_HEADER*)current)->magic)
          {
            current++;
          }

          // What file offset did we reach
          offset += current-buffer;

          // Did we find the magic
          if (current <= limit)
          {
            memcpy(&header, current, sizeof(header));
            break;
          }
        }

        // Did we reach the end of the file
        if (offset + sizeof(PACKET_HEADER) > filesize)
        {
          break;
        }
      }

      // We have found the magic

      // Check the packet length
      if (sizeof(PACKET_HEADER) > header.length || // packet length is too small
          0 != (header.length & 3) ||              // packet length is not a multiple of 4
          filesize < offset + header.length)       // packet would extend beyond the end of the file
      {
        offset++;
        continue;
      }

      // Compute the MD5 Hash of the packet
      MD5Context context;
      context.Update(&header.setid, sizeof(header)-offsetof(PACKET_HEADER, setid));

      // How much more do I need to read to get the whole packet
      u64 current = offset+sizeof(PACKET_HEADER);
      u64 limit = offset+header.length;
      while (current < limit)
      {
        size_t want = (size_t)min((u64)buffersize, limit-current);

        if (!diskfile->Read(current, buffer, want))
          break;

        context.Update(buffer, want);

        current += want;
      }

      // Did the whole packet get processed
      if (current<limit)
      {
        offset++;
        continue;
      }

      // Check the calculated packet hash against the value in the header
      MD5Hash hash;
      context.Final(hash);
      if (hash != header.hash)
      {
        offset++;
        continue;
      }

      // If this is the first packet that we have found then record the setid
      if (firstpacket)
      {
        setid = header.setid;
        firstpacket = false;
      }

      // Is the packet from the correct set
      if (setid == header.setid)
      {
        // Is it a packet type that we are interested in
        if (recoveryblockpacket_type == header.type)
        {
          if (LoadRecoveryPacket(diskfile, offset, header))
          {
            recoverypackets++;
            packets++;
          }
        }
        else if (fileverificationpacket_type == header.type)
        {
          if (LoadVerificationPacket(diskfile, offset, header))
          {
            packets++;
          }
        }
        else if (filedescriptionpacket_type == header.type)
        {
          if (LoadDescriptionPacket(diskfile, offset, header))
          {
            packets++;
          }
        }
        else if (mainpacket_type == header.type)
        {
          if (LoadMainPacket(diskfile, offset, header))
          {
            packets++;
          }
        }
        else if (creatorpacket_type == header.type)
        {
          if (LoadCreatorPacket(diskfile, offset, header))
          {
            packets++;
          }
        }
      }

      // Advance to the next packet
      offset += header.length;
    }

    delete [] buffer;
  }

  // We have finished with the file for now
  diskfile->Close();

  // Did we actually find any interesting packets
  if (packets > 0)
  {
    if (noiselevel > nlQuiet)
    {
      sout << "Loaded " << packets << " new packets";
      if (recoverypackets > 0) sout << " including " << recoverypackets << " recovery blocks";
      sout << endl;
    }

    // Remember that the file was processed
    bool success = diskFileMap.Insert(diskfile);
    assert(success);
  }
  else
  {
    if (noiselevel > nlQuiet)
      sout << "No new packets found" << endl;
    delete diskfile;
  }

  return true;
}

// Finish loading a recovery packet
bool Par2Repairer::LoadRecoveryPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  RecoveryPacket *packet = new RecoveryPacket;

  // Load the packet from disk
  if (!packet->Load(diskfile, offset, header))
  {
    delete packet;
    return false;
  }

  // What is the exponent value of this recovery packet
  u32 exponent = packet->Exponent();

  // Try to insert the new packet into the recovery packet map
  pair<map<u32,RecoveryPacket*>::const_iterator, bool> location = recoverypacketmap.insert(pair<u32,RecoveryPacket*>(exponent, packet));

  // Did the insert fail
  if (!location.second)
  {
    // The packet must be a duplicate of one we already have
    delete packet;
    return false;
  }

  return true;
}

// Finish loading a file description packet
bool Par2Repairer::LoadDescriptionPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  DescriptionPacket *packet = new DescriptionPacket;

  // Load the packet from disk
  if (!packet->Load(diskfile, offset, header))
  {
    delete packet;
    return false;
  }

  // What is the fileid
  const MD5Hash &fileid = packet->FileId();

  // Look up the fileid in the source file map for an existing source file entry
  map<MD5Hash, Par2RepairerSourceFile*>::iterator sfmi = sourcefilemap.find(fileid);
  Par2RepairerSourceFile *sourcefile = (sfmi == sourcefilemap.end()) ? 0 :sfmi->second;

  // Was there an existing source file
  if (sourcefile)
  {
    // Does the source file already have a description packet
    if (sourcefile->GetDescriptionPacket())
    {
      // Yes. We don't need another copy
      delete packet;
      return false;
    }
    else
    {
      // No. Store the packet in the source file
      sourcefile->SetDescriptionPacket(packet);
      return true;
    }
  }
  else
  {
    // Create a new source file for the packet
    sourcefile = new Par2RepairerSourceFile(packet, NULL);

    // Record the source file in the source file map
    sourcefilemap.insert(pair<MD5Hash, Par2RepairerSourceFile*>(fileid, sourcefile));

    return true;
  }
}

// Finish loading a file verification packet
bool Par2Repairer::LoadVerificationPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  VerificationPacket *packet = new VerificationPacket;

  // Load the packet from disk
  if (!packet->Load(diskfile, offset, header))
  {
    delete packet;
    return false;
  }

  // What is the fileid
  const MD5Hash &fileid = packet->FileId();

  // Look up the fileid in the source file map for an existing source file entry
  map<MD5Hash, Par2RepairerSourceFile*>::iterator sfmi = sourcefilemap.find(fileid);
  Par2RepairerSourceFile *sourcefile = (sfmi == sourcefilemap.end()) ? 0 :sfmi->second;

  // Was there an existing source file
  if (sourcefile)
  {
    // Does the source file already have a verification packet
    if (sourcefile->GetVerificationPacket())
    {
      // Yes. We don't need another copy.
      delete packet;
      return false;
    }
    else
    {
      // No. Store the packet in the source file
      sourcefile->SetVerificationPacket(packet);

      return true;
    }
  }
  else
  {
    // Create a new source file for the packet
    sourcefile = new Par2RepairerSourceFile(NULL, packet);

    // Record the source file in the source file map
    sourcefilemap.insert(pair<MD5Hash, Par2RepairerSourceFile*>(fileid, sourcefile));

    return true;
  }
}

// Finish loading the main packet
bool Par2Repairer::LoadMainPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  // Do we already have a main packet
  if (0 != mainpacket)
    return false;

  MainPacket *packet = new MainPacket;

  // Load the packet from disk;
  if (!packet->Load(diskfile, offset, header))
  {
    delete packet;
    return false;
  }

  mainpacket = packet;

  return true;
}

// Finish loading the creator packet
bool Par2Repairer::LoadCreatorPacket(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  // Do we already have a creator packet
  if (0 != creatorpacket)
    return false;

  CreatorPacket *packet = new CreatorPacket;

  // Load the packet from disk;
  if (!packet->Load(diskfile, offset, header))
  {
    delete packet;
    return false;
  }

  creatorpacket = packet;

  return true;
}

// Load packets from other PAR2 files with names based on the original PAR2 file
bool Par2Repairer::LoadPacketsFromOtherFiles(string filename)
{
  // Split the original PAR2 filename into path and name parts
  string path;
  string name;
  DiskFile::SplitFilename(filename, path, name);

  string::size_type where;

  // Trim ".par2" off of the end original name

  // Look for the last "." in the filename
  while (string::npos != (where = name.find_last_of('.')))
  {
    // Trim what follows the last .
    string tail = name.substr(where+1);
    name = name.substr(0,where);

    // Was what followed the last "." "par2"
    if (0 == stricmp(tail.c_str(), "par2"))
      break;
  }

  // If what is left ends in ".volNNN-NNN" or ".volNNN+NNN" strip that as well

  // Is there another "."
  if (string::npos != (where = name.find_last_of('.')))
  {
    // What follows the "."
    string tail = name.substr(where+1);

    // Scan what follows the last "." to see of it matches vol123-456 or vol123+456
    int n = 0;
    string::const_iterator p;
    for (p=tail.begin(); p!=tail.end(); ++p)
    {
      char ch = *p;

      if (0 == n)
      {
        if (tolower(ch) == 'v') { n++; } else { break; }
      }
      else if (1 == n)
      {
        if (tolower(ch) == 'o') { n++; } else { break; }
      }
      else if (2 == n)
      {
        if (tolower(ch) == 'l') { n++; } else { break; }
      }
      else if (3 == n)
      {
        if (isdigit(ch)) {} else if (ch == '-' || ch == '+') { n++; } else { break; }
      }
      else if (4 == n)
      {
        if (isdigit(ch)) {} else { break; }
      }
    }

    // If we matched then retain only what precedes the "."
    if (p == tail.end())
    {
      name = name.substr(0,where);
    }
  }

  // Find files called "*.par2" or "name.*.par2"

  {
    string wildcard = name.empty() ? "*.par2" : name + ".*.par2";
    std::unique_ptr< list<string> > files(
					DiskFile::FindFiles(path, wildcard, false)
					);
    par2list.merge(*files);

    string wildcardu = name.empty() ? "*.PAR2" : name + ".*.PAR2";
    std::unique_ptr< list<string> > filesu(
					 DiskFile::FindFiles(path, wildcardu, false)
					 );
    par2list.merge(*filesu);

    // Load packets from each file that was found
    for (list<string>::const_iterator s=par2list.begin(); s!=par2list.end(); ++s)
    {
      LoadPacketsFromFile(*s);
    }

    // delete files;  Taken care of by unique_ptr<>
    // delete filesu;
  }

  return true;
}

// Load packets from any other PAR2 files whose names are given on the command line
bool Par2Repairer::LoadPacketsFromExtraFiles(const vector<string> &extrafiles)
{
  for (vector<string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
  {
    string filename = *i;

    // If the filename contains ".par2" anywhere
    if (string::npos != filename.find(".par2") ||
        string::npos != filename.find(".PAR2"))
    {
      LoadPacketsFromFile(filename);
    }
  }

  return true;
}

// Check that the packets are consistent and discard any that are not
bool Par2Repairer::CheckPacketConsistency(void)
{
  // Do we have a main packet
  if (0 == mainpacket)
  {
    // If we don't have a main packet, then there is nothing more that we can do.
    // We cannot verify or repair any files.

    serr << "Main packet not found." << endl;
    return false;
  }

  // Remember the block size from the main packet
  blocksize = mainpacket->BlockSize();

  // Check that the recovery blocks have the correct amount of data
  // and discard any that don't
  {
    map<u32,RecoveryPacket*>::iterator rp = recoverypacketmap.begin();
    while (rp != recoverypacketmap.end())
    {
      if (rp->second->BlockSize() == blocksize)
      {
        ++rp;
      }
      else
      {
        serr << "Incorrect sized recovery block for exponent " << rp->second->Exponent() << " discarded" << endl;

        delete rp->second;
        map<u32,RecoveryPacket*>::iterator x = rp++;
        recoverypacketmap.erase(x);
      }
    }
  }

  // Check for source files that have no description packet or where the
  // verification packet has the wrong number of entries and discard them.
  {
    map<MD5Hash, Par2RepairerSourceFile*>::iterator sf = sourcefilemap.begin();
    while (sf != sourcefilemap.end())
    {
      // Do we have a description packet
      DescriptionPacket *descriptionpacket = sf->second->GetDescriptionPacket();
      if (descriptionpacket == 0)
      {
        // No description packet

        // Discard the source file
        delete sf->second;
        map<MD5Hash, Par2RepairerSourceFile*>::iterator x = sf++;
        sourcefilemap.erase(x);

        continue;
      }

      // Compute and store the block count from the filesize and blocksize
      sf->second->SetBlockCount(blocksize);

      // Do we have a verification packet
      VerificationPacket *verificationpacket = sf->second->GetVerificationPacket();
      if (verificationpacket == 0)
      {
        // No verification packet

        // That is ok, but we won't be able to use block verification.

        // Proceed to the next file.
        ++sf;

        continue;
      }

      // Work out the block count for the file from the file size
      // and compare that with the verification packet
      u64 filesize = descriptionpacket->FileSize();
      u32 blockcount = verificationpacket->BlockCount();

      if ((filesize + blocksize-1) / blocksize != (u64)blockcount)
      {
        // The block counts are different!

        serr << "Incorrectly sized verification packet for \"" << descriptionpacket->FileName() << "\" discarded" << endl;

        // Discard the source file

        delete sf->second;
        map<MD5Hash, Par2RepairerSourceFile*>::iterator x = sf++;
        sourcefilemap.erase(x);

        continue;
      }

      // Everything is ok.

      // Proceed to the next file
      ++sf;
    }
  }

  if (noiselevel > nlQuiet)
  {
    sout << "There are "
      << mainpacket->RecoverableFileCount()
      << " recoverable files and "
      << mainpacket->TotalFileCount() - mainpacket->RecoverableFileCount()
      << " other files."
      << endl;

    sout << "The block size used was "
      << blocksize
      << " bytes."
      << endl;
  }

  return true;
}

// Use the information in the main packet to get the source files
// into the correct order and determine their filenames
bool Par2Repairer::CreateSourceFileList(void)
{
  // For each FileId entry in the main packet
  for (u32 filenumber=0; filenumber<mainpacket->TotalFileCount(); filenumber++)
  {
    const MD5Hash &fileid = mainpacket->FileId(filenumber);

    // Look up the fileid in the source file map
    map<MD5Hash, Par2RepairerSourceFile*>::iterator sfmi = sourcefilemap.find(fileid);
    Par2RepairerSourceFile *sourcefile = (sfmi == sourcefilemap.end()) ? 0 :sfmi->second;

    if (sourcefile)
    {
      sourcefile->ComputeTargetFileName(sout, serr, noiselevel, basepath);

#ifdef _OPENMP
      // Need actual filesize on disk for mt-progress line
      sourcefile->SetDiskFileSize();
#endif
    }

    sourcefiles.push_back(sourcefile);
  }

  return true;
}

// Determine the total number of DataBlocks for the recoverable source files
// The allocate the DataBlocks and assign them to each source file
bool Par2Repairer::AllocateSourceBlocks(void)
{
  sourceblockcount = 0;

  u32 filenumber = 0;
  vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // For each recoverable source file
  while (filenumber < mainpacket->RecoverableFileCount() && sf != sourcefiles.end())
  {
    // Do we have a source file
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile)
    {
      sourceblockcount += sourcefile->BlockCount();
    }
    else
    {
      // No details for this source file so we don't know what the
      // total number of source blocks is
      //      sourceblockcount = 0;
      //      break;
    }

    ++sf;
    ++filenumber;
  }

  // Did we determine the total number of source blocks
  if (sourceblockcount > 0)
  {
    // Yes.

    // Allocate all of the Source and Target DataBlocks (which will be used
    // to read and write data to disk).

    sourceblocks.resize(sourceblockcount);
    targetblocks.resize(sourceblockcount);

    // Which DataBlocks will be allocated first
    vector<DataBlock>::iterator sourceblock = sourceblocks.begin();
    vector<DataBlock>::iterator targetblock = targetblocks.begin();

    u64 totalsize = 0;
    u32 blocknumber = 0;

    filenumber = 0;
    sf = sourcefiles.begin();

    while (filenumber < mainpacket->RecoverableFileCount() && sf != sourcefiles.end())
    {
      Par2RepairerSourceFile *sourcefile = *sf;

      if (sourcefile)
      {
        totalsize += sourcefile->GetDescriptionPacket()->FileSize();
        u32 blockcount = sourcefile->BlockCount();

        // Allocate the source and target DataBlocks to the sourcefile
        sourcefile->SetBlocks(blocknumber, blockcount, sourceblock, targetblock, blocksize);

        blocknumber++;

        sourceblock += blockcount;
        targetblock += blockcount;
      }

      ++sf;
      ++filenumber;
    }

    blocksallocated = true;

    if (noiselevel > nlQuiet)
    {
      sout << "There are a total of "
        << sourceblockcount
        << " data blocks."
        << endl;

      sout << "The total size of the data files is "
        << totalsize
        << " bytes."
        << endl;
    }
  }

  return true;
}

// Create a verification hash table for all files for which we have not
// found a complete version of the file and for which we have
// a verification packet
bool Par2Repairer::PrepareVerificationHashTable(void)
{
  if (noiselevel >= nlDebug)
    sout << "[DEBUG] Prepare verification hashtable" << endl;

  // Choose a size for the hash table
  verificationhashtable.SetLimit(sourceblockcount);

  // Will any files be block verifiable
  blockverifiable = false;

  // For each source file
  vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();
  while (sf != sourcefiles.end())
  {
    // Get the source file
    Par2RepairerSourceFile *sourcefile = *sf;

    if (sourcefile)
    {
      // Do we have a verification packet
      if (0 != sourcefile->GetVerificationPacket())
      {
        // Yes. Load the verification entries into the hash table
        verificationhashtable.Load(sourcefile, blocksize);

        blockverifiable = true;
      }
      else
      {
        // No. We can only check the whole file
        unverifiablesourcefiles.push_back(sourcefile);
      }
    }

    ++sf;
  }

  return true;
}

// Compute the table for the sliding CRC computation
bool Par2Repairer::ComputeWindowTable(void)
{
  if (noiselevel >= nlDebug)
    sout << "[DEBUG] compute window table" << endl;

  if (blockverifiable)
  {
    GenerateWindowTable(blocksize, windowtable);
  }

  return true;
}

static bool SortSourceFilesByFileName(Par2RepairerSourceFile *low,
                                      Par2RepairerSourceFile *high)
{
  return low->TargetFileName() < high->TargetFileName();
}

// Attempt to verify all of the source files
bool Par2Repairer::VerifySourceFiles(const std::string& basepath, std::vector<string>& extrafiles)
{
  if (noiselevel > nlQuiet)
    sout << endl << "Verifying source files:" << endl << endl;

  bool finalresult = true;

  // Created a sorted list of the source files and verify them in that
  // order rather than the order they are in the main packet.
  vector<Par2RepairerSourceFile*> sortedfiles;

  u32 filenumber = 0;
  vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

#ifdef _OPENMP
  mttotalsize = 0;
  mttotalprogress = 0;
#endif

  while (sf != sourcefiles.end())
  {
    // Do we have a source file
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile)
    {
      sortedfiles.push_back(sourcefile);
#ifdef _OPENMP
      // Total filesizes for mt-progress line
      mttotalsize += sourcefile->DiskFileSize();
#endif
     }
    else
    {
      // Was this one of the recoverable files
      if (filenumber < mainpacket->RecoverableFileCount())
      {
        serr << "No details available for recoverable file number " << filenumber+1 << "." << endl << "Recovery will not be possible." << endl;

        // Set error but let verification of other files continue
        finalresult = false;
      }
      else
      {
        serr << "No details available for non-recoverable file number " << filenumber - mainpacket->RecoverableFileCount() + 1 << endl;
      }
    }

    ++sf;
  }

  sort(sortedfiles.begin(), sortedfiles.end(), SortSourceFilesByFileName);

  // Start verifying the files
  #pragma omp parallel for schedule(dynamic) num_threads(Par2Repairer::GetFileThreads())
  for (int i=0; i< static_cast<int>(sortedfiles.size()); ++i)
  {
    // Do we have a source file
    Par2RepairerSourceFile *sourcefile = sortedfiles[i];

    // What filename does the file use
    const std::string& file = sourcefile->TargetFileName();
    const std::string& name = DiskFile::SplitRelativeFilename(file, basepath);
    const std::string& target_pathname = DiskFile::GetCanonicalPathname(file);

    if (noiselevel >= nlDebug)
    {
      #pragma omp critical
      {
      sout << "[DEBUG] VerifySourceFiles ----" << endl;
      sout << "[DEBUG] file: " << file << endl;
      sout << "[DEBUG] name: " << name << endl;
      sout << "[DEBUG] targ: " << target_pathname << endl;
      }
    }

    // if the target file is in the list of extra files, we remove it
    // from the extra files.
    #pragma omp critical
    {
      vector<string>::iterator it = extrafiles.begin();
      for (; it != extrafiles.end(); ++it)
      {
	const string& e = *it;
	const std::string& extra_pathname = e;
	if (!extra_pathname.compare(target_pathname))
	{
	  extrafiles.erase(it);
	  break;
	}
      }
    }

    // Check to see if we have already used this file
    bool b;
    #pragma omp critical
    b = diskFileMap.Find(file) != 0;
    if (b)
    {
      // The file has already been used!
      #pragma omp critical
      serr << "Source file " << name << " is a duplicate." << endl;

      finalresult = false;
    }
    else
    {
      DiskFile *diskfile = new DiskFile(sout, serr);

      // Does the target file exist
      if (diskfile->Open(file))
      {
        // Yes. Record that fact.
        sourcefile->SetTargetExists(true);

        // Remember that the DiskFile is the target file
        sourcefile->SetTargetFile(diskfile);

        // Remember that we have processed this file
        bool success;
        #pragma omp critical
        success = diskFileMap.Insert(diskfile);
        assert(success);
        // Do the actual verification
        if (!VerifyDataFile(diskfile, sourcefile, basepath))
          finalresult = false;

        // We have finished with the file for now
        diskfile->Close();
      }
      else
      {
        // The file does not exist.
        delete diskfile;

        if (noiselevel > nlSilent)
        {
          #pragma omp critical
          sout << "Target: \"" << utf8::console(name) << "\" - missing." << endl;
        }
      }
    }
  }

  // Find out how much data we have found
  UpdateVerificationResults();

  return finalresult;
}

// Scan any extra files specified on the command line
bool Par2Repairer::VerifyExtraFiles(const vector<string> &extrafiles, const string &basepath)
{
  if (noiselevel > nlQuiet)
    sout << endl << "Scanning extra files:" << endl << endl;

  if (completefilecount < mainpacket->RecoverableFileCount())
  {
#ifdef _OPENMP
    // Total size of extra files for mt-progress line
    mtprocessingextrafiles = true;
    mttotalprogress = 0;
    mttotalextrasize = 0;

    for (size_t i=0; i<extrafiles.size(); ++i)
      mttotalextrasize += DiskFile::GetFileSize(extrafiles[i]);
#endif

    #pragma omp parallel for schedule(dynamic) num_threads(Par2Repairer::GetFileThreads())
    for (int i=0; i< static_cast<int>(extrafiles.size()); ++i)
    {
      string filename = extrafiles[i];

      // If the filename does not include ".par2" we are interested in it.
      if (string::npos == filename.find(".par2") &&
          string::npos == filename.find(".PAR2"))
      {
        filename = DiskFile::GetCanonicalPathname(filename);

        // Has this file already been dealt with
        bool b;
        #pragma omp critical
        b = diskFileMap.Find(filename) == 0;
        if (b)
        {
          DiskFile *diskfile = new DiskFile(sout, serr);

          // Does the file exist
          if (!diskfile->Open(filename))
          {
            delete diskfile;
            continue;
          }

          // Remember that we have processed this file
          bool success;
          #pragma omp critical
          success = diskFileMap.Insert(diskfile);
          assert(success);

          // Do the actual verification
          VerifyDataFile(diskfile, 0, basepath);
          // Ignore errors

          // We have finished with the file for now
          diskfile->Close();
        }
      }
    }
  }
  // Find out how much data we have found
  UpdateVerificationResults();

#if _OPENMP
    mtprocessingextrafiles = false;
#endif

  return true;
}

// Attempt to match the data in the DiskFile with the source file
bool Par2Repairer::VerifyDataFile(DiskFile *diskfile, Par2RepairerSourceFile *sourcefile, const string &basepath)
{
  MatchType matchtype; // What type of match was made
  MD5Hash hashfull;    // The MD5 Hash of the whole file
  MD5Hash hash16k;     // The MD5 Hash of the files 16k of the file

  // Are there any files that can be verified at the block level
  if (blockverifiable)
  {
    u32 count;

    // Scan the file at the block level.

    if (!ScanDataFile(diskfile,   // [in]      The file to scan
                      basepath,
                      sourcefile, // [in/out]  Modified in the match is for another source file
                      matchtype,  // [out]
                      hashfull,   // [out]
                      hash16k,    // [out]
                      count))     // [out]
      return false;

    switch (matchtype)
    {
      case eNoMatch:
        // No data was found at all.

        // Continue to next test.
        break;
      case ePartialMatch:
        {
          // We found some data.

          // Return them.
          return true;
        }
        break;
      case eFullMatch:
        {
          // We found a perfect match.

          sourcefile->SetCompleteFile(diskfile);

          // Return the match
          return true;
        }
        break;
    }
  }

  // We did not find a match for any blocks of data within the file, but if
  // there are any files for which we did not have a verification packet
  // we can try a simple match of the hash for the whole file.

  // Are there any files that cannot be verified at the block level
  if (!unverifiablesourcefiles.empty())
  {
    // Would we have already computed the file hashes
    if (!blockverifiable)
    {
      u64 filesize = diskfile->FileSize();

      size_t buffersize = 1024*1024;
      if (buffersize > min(blocksize, filesize))
        buffersize = (size_t)min(blocksize, filesize);

      char *buffer = new char[buffersize];

      u64 offset = 0;

      MD5Context context;

      while (offset < filesize)
      {
        size_t want = (size_t)min((u64)buffersize, filesize-offset);

        if (!diskfile->Read(offset, buffer, want))
        {
          delete [] buffer;
          return false;
        }

        // Will the newly read data reach the 16k boundary
        if (offset < 16384 && offset + want >= 16384)
        {
          context.Update(buffer, (size_t)(16384-offset));

          // Compute the 16k hash
          MD5Context temp = context;
          temp.Final(hash16k);

          // Is there more data
          if (offset + want > 16384)
          {
            context.Update(&buffer[16384-offset], (size_t)(offset+want)-16384);
          }
        }
        else
        {
          context.Update(buffer, want);
        }

        offset += want;
      }

      // Compute the file hash
      MD5Hash hashfull;
      context.Final(hashfull);

      // If we did not have 16k of data, then the 16k hash
      // is the same as the full hash
      if (filesize < 16384)
      {
        hash16k = hashfull;
      }
    }

    list<Par2RepairerSourceFile*>::iterator sf = unverifiablesourcefiles.begin();

    // Compare the hash values of each source file for a match
    while (sf != unverifiablesourcefiles.end())
    {
      sourcefile = *sf;

      // Does the file match
      if (sourcefile->GetCompleteFile() == 0 &&
          diskfile->FileSize() == sourcefile->GetDescriptionPacket()->FileSize() &&
          hash16k == sourcefile->GetDescriptionPacket()->Hash16k() &&
          hashfull == sourcefile->GetDescriptionPacket()->HashFull())
      {
        if (noiselevel > nlSilent)
        {
          #pragma omp critical
          sout << diskfile->FileName() << " is a perfect match for " << sourcefile->GetDescriptionPacket()->FileName() << endl;
        }
        // Record that we have a perfect match for this source file
        sourcefile->SetCompleteFile(diskfile);

        if (blocksallocated)
        {
          // Allocate all of the DataBlocks for the source file to the DiskFile

          u64 offset = 0;
          u64 filesize = sourcefile->GetDescriptionPacket()->FileSize();

          vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();

          while (offset < filesize)
          {
            DataBlock &datablock = *sb;

            datablock.SetLocation(diskfile, offset);
            datablock.SetLength(min(blocksize, filesize-offset));

            offset += blocksize;
            ++sb;
          }
        }

        // Return the match
        return true;
      }

      ++sf;
    }
  }

  return true;
}

// Perform a sliding window scan of the DiskFile looking for blocks of data that
// might belong to any of the source files (for which a verification packet was
// available). If a block of data might be from more than one source file, prefer
// the one specified by the "sourcefile" parameter. If the first data block
// found is for a different source file then "sourcefile" is changed accordingly.
bool Par2Repairer::ScanDataFile(DiskFile                *diskfile,    // [in]
                                string                  basepath,     // [in]
                                Par2RepairerSourceFile* &sourcefile,  // [in/out]
                                MatchType               &matchtype,   // [out]
                                MD5Hash                 &hashfull,    // [out]
                                MD5Hash                 &hash16k,     // [out]
                                u32                     &count)       // [out]
{
  // Remember which file we wanted to match
  Par2RepairerSourceFile *originalsourcefile = sourcefile;

  matchtype = eNoMatch;

  string name;
  DiskFile::SplitRelativeFilename(diskfile->FileName(), basepath, name);

  // Is the file empty
  if (diskfile->FileSize() == 0)
  {
    // If the file is empty, then just return
    if (noiselevel > nlSilent)
    {
      if (originalsourcefile != 0)
      {
        #pragma omp critical
        sout << "Target: \"" << utf8::console(name) << "\" - empty." << endl;
      }
      else
      {
        #pragma omp critical
        sout << "File: \"" << utf8::console(name) << "\" - empty." << endl;
      }
    }
    return true;
  }

  string shortname;
  if (name.size() > 56)
  {
    shortname = name.substr(0, 28) + "..." + name.substr(name.size()-28);
  }
  else
  {
    shortname = name;
  }

  // Create the checksummer for the file and start reading from it
  FileCheckSummer filechecksummer(diskfile, blocksize, windowtable);
  if (!filechecksummer.Start())
    return false;

  // Assume we will make a perfect match for the file
  matchtype = eFullMatch;

  // How many matches have we had
  count = 0;

  // How many blocks have already been found
  u32 duplicatecount = 0;

  // Have we found data blocks in this file that belong to more than one target file
  bool multipletargets = false;

  // Which block do we expect to find first
  const VerificationHashEntry *nextentry = 0;

  // How far will we scan the file (1 byte at a time)
  // before skipping ahead looking for the next block
  u64 scandistance = min(skipleaway<<1, blocksize);

  // Distance to skip forward if we don't find a block
  u64 scanskip = skipdata ? blocksize - scandistance : 0;

  // Assume with are half way through scanning
  u64 scanoffset = scandistance >> 1;

  // Total number of bytes that were skipped whilst scanning
  u64 skippeddata = 0;

  // Offset of last data that was found
  u64 lastmatchoffset = 0;

  bool progressline = false;

  u64 oldoffset = 0;
  u64 printprogress = 0;

#ifdef _OPENMP
  if (noiselevel > nlQuiet)
  {
    #pragma omp critical
    sout << "Opening: \"" << shortname << "\"" << endl;
  }
#endif

  // Whilst we have not reached the end of the file
  while (filechecksummer.Offset() < diskfile->FileSize())
  {
// OPENMP progress line printing
#ifdef _OPENMP
    if (noiselevel > nlQuiet)
    {
      // Are we processing extrafiles? Use correct total size
      u64 ts = mtprocessingextrafiles ? mttotalextrasize : mttotalsize;

      // Update progress indicator
      printprogress += filechecksummer.Offset() - oldoffset;
      if (printprogress == blocksize || filechecksummer.ShortBlock())
      {
        u32 oldfraction;
        u32 newfraction;
        #pragma omp critical
        {
        oldfraction = (u32)(1000 * mttotalprogress / ts);
        mttotalprogress += printprogress;
        newfraction = (u32)(1000 * mttotalprogress / ts);
        }

        printprogress = 0;

        if (oldfraction != newfraction)
        {
          #pragma omp critical
          sout << "Scanning: " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;

          progressline = true;
        }
      }
      oldoffset = filechecksummer.Offset();

    }
// NON-OPENMP progress line printing
#else
    if (noiselevel > nlQuiet)
    {
      // Update progress indicator
      printprogress += filechecksummer.Offset() - oldoffset;
      if (printprogress == blocksize || filechecksummer.ShortBlock())
      {
        u32 oldfraction = (u32)(1000 * (filechecksummer.Offset() - printprogress) / diskfile->FileSize());
        u32 newfraction = (u32)(1000 * filechecksummer.Offset() / diskfile->FileSize());
        printprogress = 0;

        if (oldfraction != newfraction)
        {
          sout << "Scanning: \"" << shortname << "\": " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;

          progressline = true;
        }
      }
      oldoffset = filechecksummer.Offset();
    }
#endif

    // If we fail to find a match, it might be because it was a duplicate of a block
    // that we have already found.
    bool duplicate;

    // Look for a match
    const VerificationHashEntry *currententry = verificationhashtable.FindMatch(nextentry, sourcefile, filechecksummer, duplicate);

    // Did we find a match
    if (currententry != 0)
    {
      if (lastmatchoffset < filechecksummer.Offset() && noiselevel > nlNormal)
      {
        if (progressline)
        {
          #pragma omp critical
          sout << endl;
          progressline = false;
        }
        #pragma omp critical
        sout << "No data found between offset " << lastmatchoffset
          << " and " << filechecksummer.Offset() << endl;
      }

      // Is this the first match
      if (count == 0)
      {
        // Which source file was it
        sourcefile = currententry->SourceFile();

        // If the first match found was not actually the first block
        // for the source file, or it was not at the start of the
        // data file: then this is a partial match.
        if (!currententry->FirstBlock() || filechecksummer.Offset() != 0)
        {
          matchtype = ePartialMatch;
        }
      }
      else
      {
        // If the match found is not the one which was expected
        // then this is a partial match

        if (currententry != nextentry)
        {
          matchtype = ePartialMatch;
        }

        // Is the match from a different source file
        if (sourcefile != currententry->SourceFile())
        {
          multipletargets = true;
        }
      }

      if (blocksallocated)
      {
        // Record the match
        currententry->SetBlock(diskfile, filechecksummer.Offset());
      }

      // Update the number of matches found
      count++;

      // What entry do we expect next
      nextentry = currententry->Next();

      // Advance to the next block
      if (!filechecksummer.Jump(currententry->GetDataBlock()->GetLength()))
        return false;

      // If the next match fails, assume we hare half way through scanning for the next block
      scanoffset = scandistance >> 1;

      // Update offset of last match
      lastmatchoffset = filechecksummer.Offset();
    }
    else
    {
      // This cannot be a perfect match
      matchtype = ePartialMatch;

      // Was this a duplicate match
      if (duplicate && false) // ignore duplicates
      {
        duplicatecount++;

        // What entry would we expect next
        nextentry = 0;

        // Advance one whole block
        if (!filechecksummer.Jump(blocksize))
          return false;
      }
      else
      {
        // What entry do we expect next
        nextentry = 0;

        if (!filechecksummer.Step())
          return false;

        u64 skipfrom = filechecksummer.Offset();

        // Have we scanned too far without finding a block?
        if (scanskip > 0
            && ++scanoffset >= scandistance
            && skipfrom < diskfile->FileSize())
        {
          // Skip forwards to where we think we might find more data
          if (!filechecksummer.Jump(scanskip))
            return false;

          // Update the count of skipped data
          skippeddata += filechecksummer.Offset() - skipfrom;

          // Reset scan offset to 0
          scanoffset = 0;
        }
      }
    }
  }

#ifdef _OPENMP
  if (noiselevel > nlQuiet)
  {
    if (filechecksummer.Offset() == diskfile->FileSize()) {
      #pragma omp atomic
      mttotalprogress += filechecksummer.Offset() - oldoffset;
    }
  }
#endif

  if (lastmatchoffset < filechecksummer.Offset() && noiselevel > nlNormal)
  {
    if (progressline)
    {
      #pragma omp critical
      sout << endl;
      progressline = false;
    }

    #pragma omp critical
    sout << "No data found between offset " << lastmatchoffset
      << " and " << filechecksummer.Offset() << endl;
  }

  // Get the Full and 16k hash values of the file
  filechecksummer.GetFileHashes(hashfull, hash16k);

  if (noiselevel >= nlDebug)
  {
    #pragma omp critical
    {
    // Clear out old scanning line
    sout << std::setw(shortname.size()+19) << std::setfill(' ') << "";

    if (duplicatecount > 0)
      sout << "\r[DEBUG] duplicates: " << duplicatecount << endl;
    sout << "\r[DEBUG] matchcount: " << count << endl;
    sout << "[DEBUG] ----------------------" << endl;
    }
  }

  // Did we make any matches at all
  if (count > 0)
  {
    // If this still might be a perfect match, check the
    // hashes, file size, and number of blocks to confirm.
    if (matchtype            != eFullMatch ||
        count                != sourcefile->GetVerificationPacket()->BlockCount() ||
        diskfile->FileSize() != sourcefile->GetDescriptionPacket()->FileSize() ||
        hashfull             != sourcefile->GetDescriptionPacket()->HashFull() ||
        hash16k              != sourcefile->GetDescriptionPacket()->Hash16k())
    {
      matchtype = ePartialMatch;

      if (noiselevel > nlSilent)
      {
        // Did we find data from multiple target files
        if (multipletargets)
        {
          // Were we scanning the target file or an extra file
          if (originalsourcefile != 0)
          {
            #pragma omp critical
            sout << "Target: \""
              << utf8::console(name)
              << "\" - damaged, found "
              << count
              << " data blocks from several target files."
              << endl;
          }
          else
          {
            #pragma omp critical
            sout << "File: \""
              << utf8::console(name)
              << "\" - found "
              << count
              << " data blocks from several target files."
              << endl;
          }
        }
        else
        {
          // Did we find data blocks that belong to the target file
          if (originalsourcefile == sourcefile)
          {
            #pragma omp critical
            sout << "Target: \""
              << utf8::console(name)
              << "\" - damaged. Found "
              << count
              << " of "
              << sourcefile->GetVerificationPacket()->BlockCount()
              << " data blocks."
              << endl;
          }
          // Were we scanning the target file or an extra file
          else if (originalsourcefile != 0)
          {
            string targetname;
            DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

            #pragma omp critical
            sout << "Target: \""
              << utf8::console(name)
              << "\" - damaged. Found "
              << count
              << " of "
              << sourcefile->GetVerificationPacket()->BlockCount()
              << " data blocks from \""
              << utf8::console(targetname)
              << "\"."
              << endl;
          }
          else
          {
            string targetname;
            DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

            #pragma omp critical
            sout << "File: \""
              << utf8::console(name)
              << "\" - found "
              << count
              << " of "
              << sourcefile->GetVerificationPacket()->BlockCount()
              << " data blocks from \""
              << utf8::console(targetname)
              << "\"."
              << endl;
          }
        }

        if (skippeddata > 0)
        {
          #pragma omp critical
          sout << skippeddata << " bytes of data were skipped whilst scanning." << endl
            << "If there are not enough blocks found to repair: try again "
            << "with the -N option." << endl;
        }
      }
    }
    else
    {
      if (noiselevel > nlSilent)
      {
        // Did we match the target file
        if (originalsourcefile == sourcefile)
        {
          #pragma omp critical
          sout << "Target: \"" << utf8::console(name) << "\" - found." << endl;
        }
        // Were we scanning the target file or an extra file
        else if (originalsourcefile != 0)
        {
          string targetname;
          DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

          #pragma omp critical
          sout << "Target: \""
            << utf8::console(name)
            << "\" - is a match for \""
            << utf8::console(targetname)
            << "\"."
            << endl;
        }
        else
        {
          string targetname;
          DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

          #pragma omp critical
          sout << "File: \""
            << utf8::console(name)
            << "\" - is a match for \""
            << utf8::console(targetname)
            << "\"."
            << endl;
        }
      }
    }
  }
  else
  {
    matchtype = eNoMatch;

    if (noiselevel > nlSilent)
    {
      // We found not data, but did the file actually contain blocks we
      // had already found in other files.
      if (duplicatecount > 0)
      {
        #pragma omp critical
        sout << "File: \""
          << utf8::console(name)
          << "\" - found "
          << duplicatecount
          << " duplicate data blocks."
          << endl;
      }
      else
      {
        #pragma omp critical
        sout << "File: \""
          << utf8::console(name)
          << "\" - no data found."
          << endl;
      }

      if (skippeddata > 0)
      {
        #pragma omp critical
        sout << skippeddata << " bytes of data were skipped whilst scanning." << endl
          << "If there are not enough blocks found to repair: try again "
          << "with the -N option." << endl;
      }
    }
  }

  return true;
}

// Find out how much data we have found
void Par2Repairer::UpdateVerificationResults(void)
{
  availableblockcount = 0;
  missingblockcount = 0;

  completefilecount = 0;
  renamedfilecount = 0;
  damagedfilecount = 0;
  missingfilecount = 0;

  u32 filenumber = 0;
  vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Check the recoverable files
  while (sf != sourcefiles.end() && filenumber < mainpacket->TotalFileCount())
  {
    Par2RepairerSourceFile *sourcefile = *sf;

    if (sourcefile)
    {
      // Was a perfect match for the file found
      if (sourcefile->GetCompleteFile() != 0)
      {
        // Is it the target file or a different one
        if (sourcefile->GetCompleteFile() == sourcefile->GetTargetFile())
        {
          completefilecount++;
        }
        else
        {
          renamedfilecount++;
        }

        availableblockcount += sourcefile->BlockCount();
      }
      else
      {
        // Count the number of blocks that have been found
        vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();
        for (u32 blocknumber=0; blocknumber<sourcefile->BlockCount(); ++blocknumber, ++sb)
        {
          DataBlock &datablock = *sb;

          if (datablock.IsSet())
            availableblockcount++;
        }

        // Does the target file exist
        if (sourcefile->GetTargetExists())
        {
          damagedfilecount++;
        }
        else
        {
          missingfilecount++;
        }
      }
    }
    else
    {
      missingfilecount++;
    }

    ++filenumber;
    ++sf;
  }

  missingblockcount = sourceblockcount - availableblockcount;
}

// Check the verification results and report the results
bool Par2Repairer::CheckVerificationResults(void)
{
  // Is repair needed
  if (completefilecount < mainpacket->RecoverableFileCount() ||
      renamedfilecount > 0 ||
      damagedfilecount > 0 ||
      missingfilecount > 0)
  {
    if (noiselevel > nlSilent)
      sout << "Repair is required." << endl;
    if (noiselevel > nlQuiet)
    {
      if (renamedfilecount > 0) sout << renamedfilecount << " file(s) have the wrong name." << endl;
      if (missingfilecount > 0) sout << missingfilecount << " file(s) are missing." << endl;
      if (damagedfilecount > 0) sout << damagedfilecount << " file(s) exist but are damaged." << endl;
      if (completefilecount > 0) sout << completefilecount << " file(s) are ok." << endl;

      sout << "You have " << availableblockcount
        << " out of " << sourceblockcount
        << " data blocks available." << endl;
      if (recoverypacketmap.size() > 0)
        sout << "You have " << (u32)recoverypacketmap.size()
          << " recovery blocks available." << endl;
    }

    // Is repair possible
    if (recoverypacketmap.size() >= missingblockcount)
    {
      if (noiselevel > nlSilent)
        sout << "Repair is possible." << endl;

      if (noiselevel > nlQuiet)
      {
        if (recoverypacketmap.size() > missingblockcount)
          sout << "You have an excess of "
            << (u32)recoverypacketmap.size() - missingblockcount
            << " recovery blocks." << endl;

        if (missingblockcount > 0)
          sout << missingblockcount
            << " recovery blocks will be used to repair." << endl;
        else if (recoverypacketmap.size())
          sout << "None of the recovery blocks will be used for the repair." << endl;
      }

      return true;
    }
    else
    {
      if (noiselevel > nlSilent)
      {
        sout << "Repair is not possible." << endl;
        sout << "You need " << missingblockcount - recoverypacketmap.size()
          << " more recovery blocks to be able to repair." << endl;
      }

      return false;
    }
  }
  else
  {
    if (noiselevel > nlSilent)
      sout << "All files are correct, repair is not required." << endl;

    return true;
  }

  return true;
}

// Rename any damaged or missnamed target files.
bool Par2Repairer::RenameTargetFiles(void)
{
  u32 filenumber = 0;
  vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Rename any damaged target files
  while (sf != sourcefiles.end() && filenumber < mainpacket->TotalFileCount())
  {
    Par2RepairerSourceFile *sourcefile = *sf;

    // If the target file exists but is not a complete version of the file
    if (sourcefile->GetTargetExists() &&
        sourcefile->GetTargetFile() != sourcefile->GetCompleteFile())
    {
      DiskFile *targetfile = sourcefile->GetTargetFile();

      // Rename it
      diskFileMap.Remove(targetfile);

      if (!targetfile->Rename())
        return false;

      backuplist.push_back(targetfile);

      bool success = diskFileMap.Insert(targetfile);
      assert(success);

      // We no longer have a target file
      sourcefile->SetTargetExists(false);
      sourcefile->SetTargetFile(0);
    }

    ++sf;
    ++filenumber;
  }

  filenumber = 0;
  sf = sourcefiles.begin();

  // Rename any missnamed but complete versions of the files
  while (sf != sourcefiles.end() && filenumber < mainpacket->TotalFileCount())
  {
    Par2RepairerSourceFile *sourcefile = *sf;

    // If there is no targetfile and there is a complete version
    if (sourcefile->GetTargetFile() == 0 &&
        sourcefile->GetCompleteFile() != 0)
    {
      DiskFile *targetfile = sourcefile->GetCompleteFile();

      // Rename it
      diskFileMap.Remove(targetfile);

      if (!targetfile->Rename(sourcefile->TargetFileName()))
        return false;

      bool success = diskFileMap.Insert(targetfile);
      assert(success);

      // This file is now the target file
      sourcefile->SetTargetExists(true);
      sourcefile->SetTargetFile(targetfile);

      // We have one more complete file
      completefilecount++;
    }

    ++sf;
    ++filenumber;
  }

  return true;
}

// Work out which files are being repaired, create them, and allocate
// target DataBlocks to them, and remember them for later verification.
bool Par2Repairer::CreateTargetFiles(void)
{
  u32 filenumber = 0;
  vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Create any missing target files
  while (sf != sourcefiles.end() && filenumber < mainpacket->TotalFileCount())
  {
    Par2RepairerSourceFile *sourcefile = *sf;

    // If the file does not exist
    if (!sourcefile->GetTargetExists())
    {
      DiskFile *targetfile = new DiskFile(sout, serr);
      string filename = sourcefile->TargetFileName();
      u64 filesize = sourcefile->GetDescriptionPacket()->FileSize();

      // Create the target file
      if (!targetfile->Create(filename, filesize))
      {
        delete targetfile;
        return false;
      }

      // This file is now the target file
      sourcefile->SetTargetExists(true);
      sourcefile->SetTargetFile(targetfile);

      // Remember this file
      bool success = diskFileMap.Insert(targetfile);
      assert(success);

      u64 offset = 0;
      vector<DataBlock>::iterator tb = sourcefile->TargetBlocks();

      // Allocate all of the target data blocks
      while (offset < filesize)
      {
        DataBlock &datablock = *tb;

        datablock.SetLocation(targetfile, offset);
        datablock.SetLength(min(blocksize, filesize-offset));

        offset += blocksize;
        ++tb;
      }

      // Add the file to the list of those that will need to be verified
      // once the repair has completed.
      verifylist.push_back(sourcefile);
    }

    ++sf;
    ++filenumber;
  }

  return true;
}

// Work out which data blocks are available, which need to be copied
// directly to the output, and which need to be recreated, and compute
// the appropriate Reed Solomon matrix.
bool Par2Repairer::ComputeRSmatrix(void)
{
  inputblocks.resize(sourceblockcount);   // The DataBlocks that will read from disk
  copyblocks.resize(availableblockcount); // Those DataBlocks which need to be copied
  outputblocks.resize(missingblockcount); // Those DataBlocks that will re recalculated

  vector<DataBlock*>::iterator inputblock  = inputblocks.begin();
  vector<DataBlock*>::iterator copyblock   = copyblocks.begin();
  vector<DataBlock*>::iterator outputblock = outputblocks.begin();

  // Build an array listing which source data blocks are present and which are missing
  vector<bool> present;
  present.resize(sourceblockcount);

  vector<DataBlock>::iterator sourceblock  = sourceblocks.begin();
  vector<DataBlock>::iterator targetblock  = targetblocks.begin();
  vector<bool>::iterator              pres = present.begin();

  // Iterate through all source blocks for all files
  while (sourceblock != sourceblocks.end())
  {
    // Was this block found
    if (sourceblock->IsSet())
    {
      //// Open the file the block was found in.
      //if (!sourceblock->Open())
      //  return false;

      // Record that the block was found
      *pres = true;

      // Add the block to the list of those which will be read
      // as input (and which might also need to be copied).
      *inputblock = &*sourceblock;
      *copyblock = &*targetblock;

      ++inputblock;
      ++copyblock;
    }
    else
    {
      // Record that the block was missing
      *pres = false;

      // Add the block to the list of those to be written
      *outputblock = &*targetblock;
      ++outputblock;
    }

    ++sourceblock;
    ++targetblock;
    ++pres;
  }

  // Set the number of source blocks and which of them are present
  if (!rs.SetInput(present, sout, serr))
    return false;

  // Start iterating through the available recovery packets
  map<u32,RecoveryPacket*>::iterator rp = recoverypacketmap.begin();

  // Continue to fill the remaining list of data blocks to be read
  while (inputblock != inputblocks.end())
  {
    // Get the next available recovery packet
    u32 exponent = rp->first;
    RecoveryPacket* recoverypacket = rp->second;

    // Get the DataBlock from the recovery packet
    DataBlock *recoveryblock = recoverypacket->GetDataBlock();

    //// Make sure the file is open
    //if (!recoveryblock->Open())
    //  return false;

    // Add the recovery block to the list of blocks that will be read
    *inputblock = recoveryblock;

    // Record that the corresponding exponent value is the next one
    // to use in the RS matrix
    if (!rs.SetOutput(true, (u16)exponent))
      return false;

    ++inputblock;
    ++rp;
  }

  // If we need to, compute and solve the RS matrix
  if (missingblockcount == 0)
    return true;

  bool success = rs.Compute(noiselevel, sout, serr);

  return success;
}

// Allocate memory buffers for reading and writing data to disk.
bool Par2Repairer::AllocateBuffers(size_t memorylimit)
{
  // Would single pass processing use too much memory
  if (blocksize * missingblockcount > memorylimit)
  {
    // Pick a size that is small enough
    chunksize = ~3 & (memorylimit / missingblockcount);
  }
  else
  {
    chunksize = (size_t)blocksize;
  }

  // Allocate the two buffers
  inputbuffer = new u8[(size_t)chunksize];
  outputbuffer = new u8[(size_t)chunksize * missingblockcount];

  if (inputbuffer == NULL || outputbuffer == NULL)
  {
    serr << "Could not allocate buffer memory." << endl;
    return false;
  }

  return true;
}

// Read source data, process it through the RS matrix and write it to disk.
bool Par2Repairer::ProcessData(u64 blockoffset, size_t blocklength)
{
  u64 totalwritten = 0;

  // Clear the output buffer
  memset(outputbuffer, 0, (size_t)chunksize * missingblockcount);

  vector<DataBlock*>::iterator inputblock = inputblocks.begin();
  vector<DataBlock*>::iterator copyblock  = copyblocks.begin();
  u32                          inputindex = 0;

  DiskFile *lastopenfile = NULL;

  // Are there any blocks which need to be reconstructed
  if (missingblockcount > 0)
  {
    // For each input block
    while (inputblock != inputblocks.end())
    {
      // Are we reading from a new file?
      if (lastopenfile != (*inputblock)->GetDiskFile())
      {
        // Close the last file
        if (lastopenfile != NULL)
        {
          lastopenfile->Close();
        }

        // Open the new file
        lastopenfile = (*inputblock)->GetDiskFile();
        if (!lastopenfile->Open())
        {
          return false;
        }
      }

      // Read data from the current input block
      if (!(*inputblock)->ReadData(blockoffset, blocklength, inputbuffer))
        return false;

      // Have we reached the last source data block
      if (copyblock != copyblocks.end())
      {
        // Does this block need to be copied to the target file
        if ((*copyblock)->IsSet())
        {
          size_t wrote;

          // Write the block back to disk in the new target file
          if (!(*copyblock)->WriteData(blockoffset, blocklength, inputbuffer, wrote))
            return false;

          totalwritten += wrote;
        }
        ++copyblock;
      }

      // For each output block
      #pragma omp parallel for
      for (i64 outputindex=0; outputindex<missingblockcount; outputindex++)
      {
        u32 internalOutputindex = (u32) outputindex;
        // Select the appropriate part of the output buffer
        void *outbuf = &((u8*)outputbuffer)[chunksize * internalOutputindex];

        // Process the data
        rs.Process(blocklength, inputindex, inputbuffer, internalOutputindex, outbuf);

        if (noiselevel > nlQuiet)
        {
          // Update a progress indicator
          u32 oldfraction = (u32)(1000 * progress / totaldata);
          #pragma omp atomic
          progress += blocklength;
          u32 newfraction = (u32)(1000 * progress / totaldata);

          if (oldfraction != newfraction)
          {
            #pragma omp critical
            sout << "Repairing: " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;
          }
        }
      }

      ++inputblock;
      ++inputindex;
    }
  }
  else
  {
    // Reconstruction is not required, we are just copying blocks between files

    // For each block that might need to be copied
    while (copyblock != copyblocks.end())
    {
      // Does this block need to be copied
      if ((*copyblock)->IsSet())
      {
        // Are we reading from a new file?
        if (lastopenfile != (*inputblock)->GetDiskFile())
        {
          // Close the last file
          if (lastopenfile != NULL)
          {
            lastopenfile->Close();
          }

          // Open the new file
          lastopenfile = (*inputblock)->GetDiskFile();
          if (!lastopenfile->Open())
          {
            return false;
          }
        }

        // Read data from the current input block
        if (!(*inputblock)->ReadData(blockoffset, blocklength, inputbuffer))
          return false;

        size_t wrote;
        if (!(*copyblock)->WriteData(blockoffset, blocklength, inputbuffer, wrote))
          return false;
        totalwritten += wrote;
      }

      if (noiselevel > nlQuiet)
      {
        // Update a progress indicator
        u32 oldfraction = (u32)(1000 * progress / totaldata);
        progress += blocklength;
        u32 newfraction = (u32)(1000 * progress / totaldata);

        if (oldfraction != newfraction)
        {
          sout << "Processing: " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;
        }
      }

      ++copyblock;
      ++inputblock;
    }
  }

  // Close the last file
  if (lastopenfile != NULL)
  {
    lastopenfile->Close();
  }

  if (noiselevel > nlQuiet)
    sout << "Writing recovered data\r";

  // For each output block that has been recomputed
  vector<DataBlock*>::iterator outputblock = outputblocks.begin();
  for (u32 outputindex=0; outputindex<missingblockcount;outputindex++)
  {
    // Select the appropriate part of the output buffer
    char *outbuf = &((char*)outputbuffer)[chunksize * outputindex];

    // Write the data to the target file
    size_t wrote;
    if (!(*outputblock)->WriteData(blockoffset, blocklength, outbuf, wrote))
      return false;
    totalwritten += wrote;

    ++outputblock;
  }

  if (noiselevel > nlQuiet)
    sout << "Wrote " << totalwritten << " bytes to disk" << endl;

  return true;
}

// Verify that all of the reconstructed target files are now correct
bool Par2Repairer::VerifyTargetFiles(const string &basepath)
{
  bool finalresult = true;

  // Verify the target files in alphabetical order
  sort(verifylist.begin(), verifylist.end(), SortSourceFilesByFileName);

#ifdef _OPENMP
  mttotalsize = 0;
  mttotalprogress = 0;

  for (size_t i=0; i<verifylist.size(); ++i)
  {
    if (verifylist[i])
      mttotalsize += verifylist[i]->GetDescriptionPacket()->FileSize();
  }
#endif

  // Iterate through each file in the verification list
  #pragma omp parallel for schedule(dynamic) num_threads(Par2Repairer::GetFileThreads())
  for (int i=0; i< static_cast<int>(verifylist.size()); ++i)
  {
    Par2RepairerSourceFile *sourcefile = verifylist[i];
    DiskFile *targetfile = sourcefile->GetTargetFile();

    // Close the file
    if (targetfile->IsOpen())
      targetfile->Close();

    // Mark all data blocks for the file as unknown
    vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();
    for (u32 blocknumber=0; blocknumber<sourcefile->BlockCount(); blocknumber++)
    {
      sb->ClearLocation();
      ++sb;
    }

    // Say we don't have a complete version of the file
    sourcefile->SetCompleteFile(0);

    // Re-open the target file
    if (!targetfile->Open())
    {
      finalresult = false;
      continue;
    }

    // Verify the file again
    if (!VerifyDataFile(targetfile, sourcefile, basepath))
      finalresult = false;

    // Close the file again
    targetfile->Close();
  }

  // Find out how much data we have found
  UpdateVerificationResults();

  return finalresult;
}

// Delete all of the partly reconstructed files
bool Par2Repairer::DeleteIncompleteTargetFiles(void)
{
  vector<Par2RepairerSourceFile*>::iterator sf = verifylist.begin();

  // Iterate through each file in the verification list
  while (sf != verifylist.end())
  {
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile->GetTargetExists())
    {
      DiskFile *targetfile = sourcefile->GetTargetFile();

      // Close and delete the file
      if (targetfile->IsOpen())
        targetfile->Close();
      targetfile->Delete();

      // Forget the file
      diskFileMap.Remove(targetfile);
      delete targetfile;

      // There is no target file
      sourcefile->SetTargetExists(false);
      sourcefile->SetTargetFile(0);
    }

    ++sf;
  }

  return true;
}

bool Par2Repairer::RemoveBackupFiles(void)
{
  vector<DiskFile*>::iterator bf = backuplist.begin();

  if (noiselevel > nlSilent
      && bf != backuplist.end())
  {
    sout << endl << "Purge backup files." << endl;
  }

  // Iterate through each file in the backuplist
  while (bf != backuplist.end())
  {
    if (noiselevel > nlSilent)
    {
      string name;
      string path;
      DiskFile::SplitFilename((*bf)->FileName(), path, name);
      sout << "Remove \"" << utf8::console(name) << "\"." << endl;
    }

    if ((*bf)->IsOpen())
      (*bf)->Close();
    (*bf)->Delete();

    ++bf;
  }

  return true;
}

bool Par2Repairer::RemoveParFiles(void)
{
  if (noiselevel > nlSilent
      && !par2list.empty())
  {
    sout << endl << "Purge par files." << endl;
  }

  for (list<string>::const_iterator s=par2list.begin(); s!=par2list.end(); ++s)
  {
    DiskFile *diskfile = new DiskFile(sout, serr);

    if (diskfile->Open(*s))
    {
      if (noiselevel > nlSilent)
      {
        string name;
        string path;
        DiskFile::SplitFilename((*s), path, name);
        sout << "Remove \"" << utf8::console(name) << "\"." << endl;
      }

      if (diskfile->IsOpen())
        diskfile->Close();
      diskfile->Delete();
    }

    delete diskfile;
  }

  return true;
}
