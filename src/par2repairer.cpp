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

#include <functional>

namespace par2
{

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

// Test whether filename has a .par2 / .PAR2 / .Par2 extension.
bool Par2Repairer::IsPar2Filename(const std::string &filename)
{
  if (filename.size() < 5)
    return false;

  // Check that filename ends with ".par2" (case-insensitive).
  const char *ext = filename.c_str() + filename.size() - 5;
  if (ext[0] != '.')
    return false;

  return (tolower(static_cast<unsigned char>(ext[1])) == 'p'
    && tolower(static_cast<unsigned char>(ext[2])) == 'a'
    && tolower(static_cast<unsigned char>(ext[3])) == 'r'
    && ext[4] == '2');
}

Par2Repairer::Par2Repairer(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel, const Backends &backends)
: sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, backends(backends)
, observer(0)
, cancelled(false)
, packetsloaded(0)
, searchpath()
, basepath()
, totalthreads(default_threads())
, filethreads(_FILE_THREADS)
, scanmemorylimit(DEFAULT_MEMORY_LIMIT)
, blockpool()
, activereaders(0)
, setid()
, totaldatasize(0)
, recoverypacketmap()
, diskFileMap()
, sourcefilemap()
, sourcefiles()
, verifylist()
, backuplist()
, par2list()
, sourceblocks()
, targetblocks()
, scanningprepared(false)
, blockverifiable(false)
, verificationhashtable()
, unverifiablesourcefiles()
, inputblocks()
, copyblocks()
, outputblocks()
, rs()
{
  fullhash = false;
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

  ownfactors = false;

  memset(windowtable, 0, sizeof(windowtable));

  blocksallocated = false;

  completefilecount = 0;
  renamedfilecount = 0;
  damagedfilecount = 0;
  missingfilecount = 0;

  transferbuffer = 0;
  outputbuffer = 0;
}

Par2Repairer::~Par2Repairer(void)
{
  delete [] (u8*)transferbuffer;
  delete [] (u8*)outputbuffer;

  std::map<u32,RecoveryPacket*>::iterator rp = recoverypacketmap.begin();
  while (rp != recoverypacketmap.end())
  {
    delete (*rp).second;

    ++rp;
  }

  std::map<MD5Hash,Par2RepairerSourceFile*>::iterator sf = sourcefilemap.begin();
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
			     const std::string &_basepath,
			     const u32 nthreads,
			     const u32 _filethreads,
			     std::string parfilename,
			     const std::vector<std::string> &_extrafiles,
			     const bool dorepair,   // derived from operation
			     const bool purgefiles,
			     const bool renameonly,
			     const bool _skipdata,
			     const u64 _skipleaway,
			     const bool _fullhash
			     )
{
  ClearLastError();

  // Should we skip data whilst scanning files
  skipdata = _skipdata;

  // How much leaway should we allow when scanning files
  skipleaway = _skipleaway;

  // Should the whole of each file be hashed as well as its blocks
  fullhash = _fullhash;

  // Get filenames from the command line
  basepath = _basepath;
  std::vector<std::string> extrafiles = _extrafiles;

  ApplyThreadCounts(nthreads, _filethreads);
  ApplyMemoryLimit(memorylimit);

  if (!LoadPackets(parfilename, extrafiles))
  {
    if (IsCancelled())
      return eCancelled;

    errorlog.RecordIfNone(ecInternalError, "Could not load the PAR2 packets", parfilename);
    return eLogicError;
  }

  if (noiselevel > nlQuiet)
    sout << '\n';

  if (IsCancelled())
    return eCancelled;

  Result preparedresult = PreparePackets();
  if (preparedresult != eSuccess)
    return preparedresult;

  Result verifyresult = VerifyFiles(basepath, extrafiles, renameonly);

  // Are any of the files incomplete
  if (verifyresult == eRepairPossible)
  {
    // Do we want to carry out a repair
    if (!dorepair)
      return eRepairPossible;

    Result repairresult = RepairFiles(memorylimit, basepath);
    if (repairresult != eSuccess)
      return repairresult;
  }
  else if (verifyresult != eSuccess)
  {
    return verifyresult;
  }

  if (purgefiles == true)
  {
    RemoveBackupFiles();
    RemoveParFiles();
  }

  return eSuccess;
}

// Apply the thread counts, leaving either at its default when it is zero
void Par2Repairer::ApplyThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  totalthreads = resolve_threads(_nthreads);

  // No more files are read at once than there are threads to hash them with,
  // and never none whatever the caller asked for
  if (_filethreads != 0)
    filethreads = std::max(1u, std::min(_filethreads, totalthreads));
}

// Verify the source files and work out whether a repair is needed or possible
// The hash table and window table are built from the packets loaded so far and
// are not rebuilt per scan: PrepareVerificationHashTable appends to
// unverifiablesourcefiles and loads the hash table, so calling it twice would
// duplicate both.
bool Par2Repairer::PrepareForScanning(void)
{
  if (scanningprepared)
    return true;

  if (!PrepareVerificationHashTable())
    return false;

  if (!ComputeWindowTable())
    return false;

  scanningprepared = true;

  return true;
}

void Par2Repairer::DiscardScannedFile(DiskFile *diskfile)
{
  for (std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin(); sf != sourcefiles.end(); ++sf)
  {
    Par2RepairerSourceFile *sourcefile = *sf;

    if (0 == sourcefile)
      continue;

    if (sourcefile->GetTargetFile() == diskfile)
    {
      sourcefile->SetTargetFile(0);
      sourcefile->SetTargetExists(false);
    }

    if (sourcefile->GetCompleteFile() == diskfile)
      sourcefile->SetCompleteFile(0);

    // Only the blocks this file supplied: another file may hold the rest
    if (sourcefile->GetDescriptionPacket() != 0)
    {
      std::vector<DataBlock>::iterator block = sourcefile->SourceBlocks();
      for (u32 i = 0; i < sourcefile->BlockCount(); ++i, ++block)
      {
        if (block->IsSet() && block->GetDiskFile() == diskfile)
          block->ClearLocation();
      }
    }
  }

  diskFileMap.Remove(diskfile);
  delete diskfile;
}

Result Par2Repairer::ScanFile(const std::string &filename, const std::string &basepath)
{
  ClearLastError();

  if (0 == mainpacket)
  {
    errorlog.RecordIfNone(ecMainPacketMissing, "The PAR2 files do not describe a set");
    return eInsufficientCriticalData;
  }

  if (!PrepareForScanning())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not prepare to scan files");
    return eLogicError;
  }

  const std::string pathname = DiskFile::GetCanonicalPathname(filename);

  DiskFile *previous = diskFileMap.Find(pathname);
  if (previous != 0)
    DiscardScannedFile(previous);

  // Which source file the set expects at this path, if any
  Par2RepairerSourceFile *sourcefile = 0;
  for (std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin(); sf != sourcefiles.end(); ++sf)
  {
    if (*sf != 0 && (*sf)->GetDescriptionPacket() != 0 &&
        DiskFile::GetCanonicalPathname((*sf)->TargetFileName()) == pathname)
    {
      sourcefile = *sf;
      break;
    }
  }

  DiskFile *diskfile = new DiskFile(sout, serr);
  if (!diskfile->Open(pathname))
  {
    delete diskfile;
    UpdateVerificationResults();

    return CheckVerificationResults() ? eRepairPossible : eRepairNotPossible;
  }

  if (!diskFileMap.Insert(diskfile))
  {
    diskfile->Close();
    delete diskfile;

    errorlog.RecordIfNone(ecInternalError, "Could not track the file being scanned", pathname);
    return eLogicError;
  }

  if (0 != sourcefile)
  {
    sourcefile->SetTargetExists(true);
    sourcefile->SetTargetFile(diskfile);
  }

  ProgressMeter<u64> progress(sout, "Scanning: ", diskfile->FileSize(), noiselevel, observer);

  ResetScanBuffers(1);

  VerifyDataFile(diskfile, sourcefile, basepath, progress);

  diskfile->Close();

  // Nothing is scanned again until the next file arrives, so the buffers are
  // given up rather than held against the memory a repair needs
  scanbuffers.Reset(0, 0);

  if (IsCancelled())
    return eCancelled;

  UpdateVerificationResults();

  if (!CheckVerificationResults())
    return eRepairNotPossible;

  if (completefilecount < mainpacket->RecoverableFileCount())
    return eRepairPossible;

  return eSuccess;
}

Result Par2Repairer::VerifyFiles(const std::string &basepath,
                                 std::vector<std::string> &extrafiles,
                                 const bool renameonly)
{
  ClearLastError();

  renamedlist.clear();

  if (!PrepareForScanning())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not prepare to scan files");
    return eLogicError;
  }

  ResetScanBuffers(std::max(sourcefiles.size(), extrafiles.size()));

  // Attempt to verify all of the source files
  if (!VerifySourceFiles(basepath, extrafiles))
  {
    if (IsCancelled())
      return eCancelled;

    errorlog.RecordIfNone(ecFileReadFailed, "Could not verify the source files");
    return eFileIOError;
  }

  if (IsCancelled())
    return eCancelled;

  if (completefilecount < mainpacket->RecoverableFileCount())
  {
    // Scan any extra files specified on the command line
    if (!VerifyExtraFiles(extrafiles, basepath, renameonly))
    {
      if (IsCancelled())
        return eCancelled;

      errorlog.RecordIfNone(ecInternalError, "Could not scan the extra files");
      return eLogicError;
    }
  }

  // Find out how much data we have found
  UpdateVerificationResults();

  // Nothing is scanned again until the repaired files are verified, so the
  // buffers are given up rather than held against the memory a repair needs
  scanbuffers.Reset(0, 0);

  if (noiselevel > nlSilent)
    sout << '\n';

  // Check the verification results and report the results
  if (!CheckVerificationResults())
    return eRepairNotPossible;
  // Are any of the files incomplete
  if (completefilecount < mainpacket->RecoverableFileCount())
    return eRepairPossible;

  return eSuccess;
}

// Rebuild whatever is missing or damaged
Result Par2Repairer::RepairFiles(const size_t memorylimit, const std::string &basepath,
                                 bool verifyafter)
{
  ClearLastError();

  ApplyMemoryLimit(memorylimit);

  if (noiselevel > nlSilent)
    sout << '\n';

  // Rename any damaged or missnamed target files.
  if (!RenameTargetFiles())
  {
    errorlog.RecordIfNone(ecFileRenameFailed, "Could not rename the damaged or misnamed files");
    return eFileIOError;
  }

  // Are we still missing any files
  if (completefilecount < mainpacket->RecoverableFileCount())
  {
    // Work out which files are being repaired, create them, and allocate
    // target DataBlocks to them, and remember them for later verification.
    if (!CreateTargetFiles())
    {
      errorlog.RecordIfNone(ecFileCreateFailed, "Could not create the files to repair into");
      return eFileIOError;
    }

    // Allocate memory buffers for reading and writing data to disk, and
    // build the processor, which is offered the erasures below.
    if (!AllocateBuffers(memorylimit))
    {
      // Delete all of the partly reconstructed files
      DeleteIncompleteTargetFiles();
      errorlog.RecordIfNone(ecOutOfMemory, "Could not allocate buffer memory");
      return eMemoryError;
    }

    // Work out which data blocks are available, which need to be copied
    // directly to the output, and which need to be recreated, and compute
    // the appropriate Reed Solomon matrix.
    if (!ComputeRSmatrix())
    {
      // Delete all of the partly reconstructed files
      DeleteIncompleteTargetFiles();
      errorlog.RecordIfNone(ecProcessorFailed, "Could not compute the Reed Solomon matrix");
      return eFileIOError;
    }

    if (noiselevel > nlSilent)
      sout << '\n';

    if (observer)
      observer->OnRepairStart();

    // Set the total amount of data to be processed.
    ProgressMeter<u64> progress(sout, missingblockcount > 0 ? "Repairing: " : "Processing: ", blocksize * sourceblockcount, noiselevel, observer);

    // Start at an offset of 0 within a block.
    u64 blockoffset = 0;
    while (blockoffset < blocksize) // Continue until the end of the block.
    {
      // Work out how much data to process this time.
      size_t blocklength = (size_t)std::min((u64)chunksize, blocksize-blockoffset);

      // Read source data, process it through the RS matrix and write it to disk.
      if (!ProcessData(blockoffset, blocklength, progress))
      {
        // Delete all of the partly reconstructed files
        DeleteIncompleteTargetFiles();

        if (IsCancelled())
          return eCancelled;

        errorlog.RecordIfNone(ecProcessorFailed, "Could not rebuild the missing blocks");
        return eFileIOError;
      }

      if (IsCancelled())
      {
        // Delete all of the partly reconstructed files
        DeleteIncompleteTargetFiles();
        return eCancelled;
      }

      // Advance to the need offset within each block
      blockoffset += blocklength;
    }

    // The repaired files are scanned into buffers of their own, so the ones
    // the repair read and wrote through are given up first
    delete [] (u8*)transferbuffer;
    transferbuffer = 0;
    delete [] (u8*)outputbuffer;
    outputbuffer = 0;

    if (verifyafter)
    {
      if (noiselevel > nlSilent)
        sout << "\nVerifying repaired files:\n" << std::endl;

      // Verify that all of the reconstructed target files are now correct
      ResetScanBuffers(verifylist.size());

      if (!VerifyTargetFiles(basepath))
      {
        // Delete all of the partly reconstructed files
        DeleteIncompleteTargetFiles();

        if (IsCancelled())
          return eCancelled;

        errorlog.RecordIfNone(ecFileReadFailed, "Could not verify the repaired files");
        return eFileIOError;
      }
    }
    else
    {
      // Close what the skipped pass would have closed
      for (size_t i = 0; i < verifylist.size(); ++i)
      {
        Par2RepairerSourceFile *sourcefile = verifylist[i];
        if (0 == sourcefile)
          continue;

        DiskFile *targetfile = sourcefile->GetTargetFile();
        if (0 != targetfile && targetfile->IsOpen())
          targetfile->Close();
      }
    }

    if (IsCancelled())
    {
      // Delete all of the partly reconstructed files
      DeleteIncompleteTargetFiles();
      return eCancelled;
    }
  }

  // Are all of the target files now complete?
  if (verifyafter && completefilecount<mainpacket->RecoverableFileCount())
  {
    serr << "Repair Failed." << std::endl;
    return eRepairFailed;
  }

  if (noiselevel > nlSilent)
    sout << "\nRepair complete." << std::endl;

  return eSuccess;
}


// The source file the set records under that name
Par2RepairerSourceFile *Par2Repairer::FindSourceFile(const std::string &filename) const
{
  std::map<std::string, Par2RepairerSourceFile*>::const_iterator sf =
    sourcefilesbyname.find(filename);

  return sf == sourcefilesbyname.end() ? 0 : sf->second;
}

// List the files the loaded packets describe
bool Par2Repairer::GetBlockChecksums(const std::string &filename,
                                    std::vector<u32> *crcs) const
{
  if (0 == crcs)
    return false;

  crcs->clear();

  const Par2RepairerSourceFile *sourcefile = FindSourceFile(filename);
  if (0 == sourcefile)
    return false;

  const VerificationPacket *verificationpacket = sourcefile->GetVerificationPacket();
  if (0 == verificationpacket)
    return false;

  const u32 blockcount = verificationpacket->BlockCount();
  crcs->reserve(blockcount);

  for (u32 blocknumber=0; blocknumber<blockcount; ++blocknumber)
    crcs->push_back(verificationpacket->VerificationEntry(blocknumber)->crc);

  return true;
}

// Which blocks of a file the last verification found
bool Par2Repairer::GetFoundBlocks(const std::string &filename,
                                  std::vector<bool> *blocks) const
{
  if (0 == blocks)
    return false;

  blocks->clear();

  const Par2RepairerSourceFile *sourcefile = FindSourceFile(filename);
  if (0 == sourcefile)
    return false;

  const VerificationPacket *verificationpacket = sourcefile->GetVerificationPacket();
  if (0 == verificationpacket)
    return false;

  const u32 blockcount = verificationpacket->BlockCount();
  blocks->reserve(blockcount);

  const DiskFile *targetfile = sourcefile->GetTargetFile();

  std::vector<DataBlock>::iterator sourceblock = sourcefile->SourceBlocks();
  for (u32 blocknumber=0; blocknumber<blockcount; ++blocknumber, ++sourceblock)
    blocks->push_back(sourceblock->IsSet()
                      && sourceblock->GetDiskFile() == targetfile
                      && sourceblock->GetOffset() == blocknumber * blocksize);

  return true;
}

bool Par2Repairer::GetFileInfo(std::vector<Par2FileInfo> *files) const
{
  if (0 == files)
    return false;

  files->clear();

  if (0 == mainpacket)
    return false;

  for (std::vector<Par2RepairerSourceFile*>::const_iterator sf = sourcefiles.begin();
       sf != sourcefiles.end();
       ++sf)
  {
    const Par2RepairerSourceFile *sourcefile = *sf;
    if (0 == sourcefile || 0 == sourcefile->GetDescriptionPacket())
      continue;

    const VerificationPacket *verificationpacket = sourcefile->GetVerificationPacket();

    const DescriptionPacket *descriptionpacket = sourcefile->GetDescriptionPacket();

    Par2FileInfo info;
    info.filename = descriptionpacket->FileName();
    info.localfilename = sourcefile->TargetFileName();
    info.filesize = descriptionpacket->FileSize();
    info.blockcount = verificationpacket ? verificationpacket->BlockCount() : 0;
    memcpy(info.hashfull.data(), descriptionpacket->HashFull().hash, 16);
    memcpy(info.hash16k.data(), descriptionpacket->Hash16k().hash, 16);

    files->push_back(info);
  }

  return true;
}

// The files this repair renamed out of the way, which is what par2's own
// purge deletes. Only files par2 renamed itself are listed, never a file the
// caller supplied.
// What each renamed file was found as, against what the set calls it
bool Par2Repairer::GetRenamedFiles(std::vector<std::pair<std::string, std::string> > *files) const
{
  if (0 == files)
    return false;

  files->clear();

  for (std::map<std::string, std::string>::const_iterator rf = renamedlist.begin();
       rf != renamedlist.end();
       ++rf)
  {
    files->push_back(std::make_pair(rf->second, rf->first));
  }

  return true;
}

bool Par2Repairer::GetBackupFiles(std::vector<std::string> *files) const
{
  if (0 == files)
    return false;

  files->clear();

  for (std::vector<DiskFile*>::const_iterator bf = backuplist.begin();
       bf != backuplist.end();
       ++bf)
  {
    files->push_back((*bf)->FileName());
  }

  return true;
}

// The numbers behind the last verification
bool Par2Repairer::GetVerifyResult(Par2VerifyResult *result) const
{
  if (0 == result || 0 == mainpacket)
    return false;

  result->completefilecount = completefilecount;
  result->renamedfilecount = renamedfilecount;
  result->damagedfilecount = damagedfilecount;
  result->missingfilecount = missingfilecount;
  result->availableblockcount = availableblockcount;
  result->missingblockcount = missingblockcount;
  result->recoveryblockcount = (u32)recoverypacketmap.size();

  return true;
}

// Accept the caller's word that these blocks are intact
bool Par2Repairer::SetKnownBlocks(const std::string &filename,
                                  const std::vector<bool> &blocks)
{
  if (blocks.empty())
  {
    knownblocks.erase(filename);
    return true;
  }

  // Checked against the set where it is already known, so that a name or a
  // length which would never be used is refused rather than quietly ignored.
  // Nothing is known before the packets are read, and the check is made again
  // when the blocks come to be used.
  const Par2RepairerSourceFile *sourcefile = FindSourceFile(filename);
  if (0 != sourcefile)
  {
    const VerificationPacket *verificationpacket = sourcefile->GetVerificationPacket();
    if (0 == verificationpacket || blocks.size() != verificationpacket->BlockCount())
      return false;
  }
  else if (!sourcefilesbyname.empty())
  {
    return false;
  }

  knownblocks[filename] = blocks;

  return true;
}

// Use the blocks the caller has vouched for instead of scanning the file. The
// same conditions as the aligned scan apply: without a verification packet, or
// if the file is not exactly the right length, nothing can be said about where
// the blocks are.
bool Par2Repairer::TakeKnownBlocks(DiskFile               *diskfile,
                                  Par2RepairerSourceFile *sourcefile,
                                  std::vector<char>      &matched,
                                  u32                    &matchcount)
{
  matchcount = 0;

  if (knownblocks.empty() || 0 == sourcefile)
    return false;

  const DescriptionPacket *descriptionpacket = sourcefile->GetDescriptionPacket();
  const VerificationPacket *verificationpacket = sourcefile->GetVerificationPacket();
  if (0 == descriptionpacket || 0 == verificationpacket)
    return false;

  // A source file which has already been matched must not claim its blocks again
  if (0 != sourcefile->GetCompleteFile())
    return false;

  if (diskfile->FileSize() != descriptionpacket->FileSize())
    return false;

  std::map<std::string, std::vector<bool> >::const_iterator kb =
    knownblocks.find(descriptionpacket->FileName());
  if (kb == knownblocks.end())
    return false;

  const u32 blockcount = verificationpacket->BlockCount();
  if (0 == blockcount || kb->second.size() != blockcount)
    return false;

  matched.assign(blockcount, 0);

  for (u32 blocknumber=0; blocknumber<blockcount; ++blocknumber)
  {
    if (kb->second[blocknumber])
    {
      matched[blocknumber] = 1;
      ++matchcount;
    }
  }

  return true;
}

// How many blocks the verification packet says a source file should have,
// or zero when there is no verification packet for it.
static u32 BlocksNeeded(const Par2RepairerSourceFile *sourcefile)
{
  if (sourcefile == 0 || sourcefile->GetVerificationPacket() == 0)
    return 0;

  return sourcefile->GetVerificationPacket()->BlockCount();
}

// The name the set records for a file, which is the same on every system. A
// file the set does not name has only the name it has on this one.
static std::string ReportedName(const Par2RepairerSourceFile *sourcefile, const std::string &localname)
{
  if (sourcefile == 0 || sourcefile->GetDescriptionPacket() == 0)
    return localname;

  return sourcefile->GetDescriptionPacket()->FileName();
}

// Load packets from the specified PAR2 file, from the other PAR2 files whose
// names are based on it, and from any additional files supplied by the caller.
// Files that have already been loaded are skipped.
bool Par2Repairer::LoadPackets(const std::string &parfilename,
                              const std::vector<std::string> &extrafiles,
                              bool reread)
{
  // Determine the searchpath from the location of the main PAR2 file
  std::string name;
  DiskFile::SplitFilename(parfilename, searchpath, name);

  par2list.push_back(parfilename);

  // Load packets from the main PAR2 file, which is the only one reread applies to
  if (!LoadPacketsFromFile(searchpath + name, reread))
    return false;

  // Load packets from other PAR2 files with names based on the original PAR2 file
  if (!LoadPacketsFromOtherFiles(parfilename))
    return false;

  // Load packets from any other PAR2 files whose names are given on the command line
  if (!LoadPacketsFromExtraFiles(extrafiles))
    return false;

  return true;
}

// Work out what the packets loaded so far describe. Rebuilt from scratch each
// time so that it can be called again after more packets have been loaded.
Result Par2Repairer::PreparePackets(void)
{
  ClearLastError();

  sourcefiles.clear();

  // Check that the packets are consistent and discard any that are not
  if (!CheckPacketConsistency())
  {
    errorlog.RecordIfNone(ecMainPacketMissing, "The PAR2 files do not describe a set");
    return eInsufficientCriticalData;
  }

  // Use the information in the main packet to get the source files
  // into the correct order and determine their filenames
  if (!CreateSourceFileList())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not build the list of source files");
    return eLogicError;
  }

  // Determine the total number of DataBlocks for the recoverable source files
  // The allocate the DataBlocks and assign them to each source file
  if (!AllocateSourceBlocks())
  {
    errorlog.RecordIfNone(ecInternalError, "Could not allocate the source blocks");
    return eLogicError;
  }

  // The name each source file has on this system, for looking one up by it
  sourcefilesbyname.clear();
  for (std::vector<Par2RepairerSourceFile*>::const_iterator sf = sourcefiles.begin();
       sf != sourcefiles.end();
       ++sf)
  {
    Par2RepairerSourceFile *sourcefile = *sf;
    if (0 == sourcefile || 0 == sourcefile->GetDescriptionPacket())
      continue;

    sourcefilesbyname.insert(std::make_pair(sourcefile->GetDescriptionPacket()->FileName(), sourcefile));
  }

  if (observer)
  {
    Par2SetInfo info;
    memcpy(info.setid.data(), setid.hash, sizeof(setid.hash));
    info.blocksize = blocksize;
    info.datablocks = sourceblockcount;
    info.recoveryblocks = (u32)recoverypacketmap.size();
    info.recoverablefilecount = mainpacket->RecoverableFileCount();
    info.otherfilecount = mainpacket->TotalFileCount() - mainpacket->RecoverableFileCount();
    info.datasize = totaldatasize;

    observer->OnSetInfo(info);
  }

  return eSuccess;
}

// Load the packets from the specified file. reread asks for a file that has
// already been processed to be read again.
bool Par2Repairer::LoadPacketsFromFile(std::string filename, bool reread)
{
  DiskFile *known = diskFileMap.Find(filename);

  // Skip the file if it has already been processed
  if (known != 0 && !reread)
  {
    return true;
  }

  // Reuse the DiskFile of a known file: packets already loaded from it hold
  // that pointer. Reopening it refreshes the recorded size. The map owns it.
  const bool owned = (0 == known);
  DiskFile *diskfile = known;

  if (0 != known)
  {
    known->Close();

    if (!known->Open(filename))
      return true;
  }
  else
  {
    diskfile = new DiskFile(sout, serr);

    // Open the file
    if (!diskfile->Open(filename))
    {
      // If we could not open the file, ignore the error and
      // proceed to the next file
      delete diskfile;
      return true;
    }
  }

  std::string name;
  {
    std::string path;
    DiskFile::SplitFilename(filename, path, name);

    if (noiselevel > nlSilent)
      sout << "Loading \"" << name << "\"." << std::endl;

    if (observer)
      observer->OnFile(name);
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
    size_t buffersize = (size_t)std::min((u64)1048576, filesize);
    u8 *buffer = new u8[buffersize];

    // Progress indicator
    ProgressMeter<u64> progress(sout, "Loading: ", filesize, noiselevel);

    // Start at the beginning of the file
    u64 offset = 0;

    // Continue as long as there is at least enough for the packet header
    while (offset + sizeof(PACKET_HEADER) <= filesize)
    {
      if (IsCancelled())
        break;

      progress.Update(offset);

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
          size_t want = (size_t)std::min((u64)buffersize, filesize-offset);

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
        size_t want = (size_t)std::min((u64)buffersize, limit-current);

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
    progress.Update(offset);

    delete [] buffer;
  }

  // We have finished with the file for now
  diskfile->Close();

  // Did we actually find any interesting packets
  packetsloaded += packets;

  if (packets > 0)
  {
    if (noiselevel > nlQuiet)
    {
      sout << "Loaded " << packets << " new packets";
      if (recoverypackets > 0) sout << " including " << recoverypackets << " recovery blocks";
      sout << std::endl;
    }

    // Remember that the file was processed
    if (owned)
    {
      bool success = diskFileMap.Insert(diskfile);
      assert(success);
    }
  }
  else
  {
    if (noiselevel > nlQuiet)
      sout << "No new packets found" << std::endl;

    if (owned)
      delete diskfile;
  }

  if (observer)
    observer->OnFileDone(name, 0, 0);

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
  std::pair<std::map<u32,RecoveryPacket*>::const_iterator, bool> location = recoverypacketmap.insert(std::pair<u32,RecoveryPacket*>(exponent, packet));

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
  std::map<MD5Hash, Par2RepairerSourceFile*>::iterator sfmi = sourcefilemap.find(fileid);
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
    sourcefilemap.insert(std::pair<MD5Hash, Par2RepairerSourceFile*>(fileid, sourcefile));

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
  std::map<MD5Hash, Par2RepairerSourceFile*>::iterator sfmi = sourcefilemap.find(fileid);
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
    sourcefilemap.insert(std::pair<MD5Hash, Par2RepairerSourceFile*>(fileid, sourcefile));

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
bool Par2Repairer::LoadPacketsFromOtherFiles(std::string filename)
{
  // Split the original PAR2 filename into path and name parts
  std::string path;
  std::string name;
  DiskFile::SplitFilename(filename, path, name);

  std::string::size_type where;

  // Trim ".par2" off of the end original name

  // Look for the last "." in the filename
  while (std::string::npos != (where = name.find_last_of('.')))
  {
    // Trim what follows the last .
    std::string tail = name.substr(where+1);
    name = name.substr(0,where);

    // Was what followed the last "." "par2"
    if (0 == stricmp(tail.c_str(), "par2"))
      break;
  }

  // If what is left ends in ".volNNN-NNN" or ".volNNN+NNN" strip that as well

  // Is there another "."
  if (std::string::npos != (where = name.find_last_of('.')))
  {
    // What follows the "."
    std::string tail = name.substr(where+1);

    // Scan what follows the last "." to see of it matches vol123-456 or vol123+456
    int n = 0;
    std::string::const_iterator p;
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
    std::string wildcard = name.empty() ? "*.par2" : name + ".*.par2";
    std::unique_ptr< std::list<std::string> > files(
					DiskFile::FindFiles(path, wildcard, false)
					);
    par2list.splice(par2list.end(), *files);

    std::string wildcardu = name.empty() ? "*.PAR2" : name + ".*.PAR2";
    std::unique_ptr< std::list<std::string> > filesu(
					 DiskFile::FindFiles(path, wildcardu, false)
					 );
    par2list.splice(par2list.end(), *filesu);

    // Load packets from each file that was found
    for (std::list<std::string>::const_iterator s=par2list.begin(); s!=par2list.end(); ++s)
    {
      LoadPacketsFromFile(*s);
    }

    // delete files;  Taken care of by unique_ptr<>
    // delete filesu;
  }

  return true;
}

// Load packets from any other PAR2 files whose names are given on the command line
bool Par2Repairer::LoadPacketsFromExtraFiles(const std::vector<std::string> &extrafiles)
{
  for (std::vector<std::string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
  {
    if (IsCancelled())
      break;

    std::string filename = *i;

    // If the filename has a .par2 / .PAR2 / .Par2 extension
    if (IsPar2Filename(filename))
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

    serr << "Main packet not found." << std::endl;
    return false;
  }

  // Remember the block size from the main packet
  blocksize = mainpacket->BlockSize();

  // Check that the recovery blocks have the correct amount of data
  // and discard any that don't
  {
    std::map<u32,RecoveryPacket*>::iterator rp = recoverypacketmap.begin();
    while (rp != recoverypacketmap.end())
    {
      if (rp->second->BlockSize() == blocksize)
      {
        ++rp;
      }
      else
      {
        serr << "Incorrect sized recovery block for exponent " << rp->second->Exponent() << " discarded" << std::endl;

        delete rp->second;
        std::map<u32,RecoveryPacket*>::iterator x = rp++;
        recoverypacketmap.erase(x);
      }
    }
  }

  // Check for source files that have no description packet or where the
  // verification packet has the wrong number of entries and discard them.
  {
    std::map<MD5Hash, Par2RepairerSourceFile*>::iterator sf = sourcefilemap.begin();
    while (sf != sourcefilemap.end())
    {
      // Do we have a description packet
      DescriptionPacket *descriptionpacket = sf->second->GetDescriptionPacket();
      if (descriptionpacket == 0)
      {
        // No description packet

        // Discard the source file
        delete sf->second;
        std::map<MD5Hash, Par2RepairerSourceFile*>::iterator x = sf++;
        sourcefilemap.erase(x);

        continue;
      }

      // Compute and store the block count from the filesize and blocksize
      if (!sf->second->SetBlockCount(blocksize))
      {
        serr << "Too many blocks in source file \"" << descriptionpacket->FileName() << "\" discarded" << std::endl;

        delete sf->second;
        std::map<MD5Hash, Par2RepairerSourceFile*>::iterator x = sf++;
        sourcefilemap.erase(x);

        continue;
      }

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

      // Compare the calculated block count with the verification packet.
      if (sf->second->BlockCount() != verificationpacket->BlockCount())
      {
        // The block counts are different!

        serr << "Incorrectly sized verification packet for \"" << descriptionpacket->FileName() << "\" discarded" << std::endl;

        // Discard the source file

        delete sf->second;
        std::map<MD5Hash, Par2RepairerSourceFile*>::iterator x = sf++;
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
      << " other files.\n"
         "The block size used was "
      << blocksize
      << " bytes."
      << std::endl;
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
    std::map<MD5Hash, Par2RepairerSourceFile*>::iterator sfmi = sourcefilemap.find(fileid);
    Par2RepairerSourceFile *sourcefile = (sfmi == sourcefilemap.end()) ? 0 :sfmi->second;

    if (sourcefile)
    {
      sourcefile->ComputeTargetFileName(sout, serr, noiselevel, basepath);

      // Need actual filesize on disk for mt-progress line
      sourcefile->SetDiskFileSize();
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
  std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // For each recoverable source file
  while (filenumber < mainpacket->RecoverableFileCount() && sf != sourcefiles.end())
  {
    // Do we have a source file
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile)
    {
      u32 blockcount = sourcefile->BlockCount();
      if (blockcount > ((u32)~0) - sourceblockcount)
      {
        serr << "Too many source blocks in recovery set." << std::endl;
        return false;
      }

      sourceblockcount += blockcount;
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
    std::vector<DataBlock>::iterator sourceblock = sourceblocks.begin();
    std::vector<DataBlock>::iterator targetblock = targetblocks.begin();

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
    totaldatasize = totalsize;

    if (noiselevel > nlQuiet)
    {
      sout << "There are a total of "
        << sourceblockcount
        << " data blocks.\n"
           "The total size of the data files is "
        << totalsize
        << " bytes."
        << std::endl;
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
    sout << "[DEBUG] Prepare verification hashtable" << std::endl;

  // Choose a size for the hash table
  verificationhashtable.SetLimit(sourceblockcount);

  // Will any files be block verifiable
  blockverifiable = false;

  // For each source file
  std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();
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
    sout << "[DEBUG] compute window table" << std::endl;

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
bool Par2Repairer::VerifySourceFiles(const std::string& basepath, std::vector<std::string>& extrafiles)
{
  if (noiselevel > nlQuiet)
    sout << "\nVerifying source files:\n" << std::endl;

  std::atomic<bool> finalresult(true);

  // Created a sorted list of the source files and verify them in that
  // order rather than the order they are in the main packet.
  std::vector<Par2RepairerSourceFile*> sortedfiles;

  u32 filenumber = 0;
  std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  u64 mttotalsize = 0;

  while (sf != sourcefiles.end())
  {
    // Do we have a source file
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile)
    {
      sortedfiles.push_back(sourcefile);
      // Total filesizes for mt-progress line
      mttotalsize += sourcefile->DiskFileSize();
     }
    else
    {
      // Was this one of the recoverable files
      if (filenumber < mainpacket->RecoverableFileCount())
      {
        serr << "No details available for recoverable file number " << filenumber+1 << ".\nRecovery will not be possible." << std::endl;

        // Set error but let verification of other files continue
        finalresult = false;
      }
      else
      {
        serr << "No details available for non-recoverable file number " << filenumber - mainpacket->RecoverableFileCount() + 1 << std::endl;
      }
    }

    ++filenumber;
    ++sf;
  }

  std::sort(sortedfiles.begin(), sortedfiles.end(), SortSourceFilesByFileName);
  ProgressMeter<u64> progress(sout, "Scanning: ", mttotalsize, noiselevel, observer);

  // Start verifying the files
  foreach_parallel(sortedfiles, FileThreads(sortedfiles.size()), [&](Par2RepairerSourceFile *sourcefile)
  {
    if (IsCancelled())
      return;

    // What filename does the file use
    const std::string& file = sourcefile->TargetFileName();
    const std::string& name = DiskFile::SplitRelativeFilename(file, basepath);
    const std::string& target_pathname = DiskFile::GetCanonicalPathname(file);

    if (noiselevel >= nlDebug)
    {
      LockedStream(sout) << "[DEBUG] VerifySourceFiles ----\n"
        "[DEBUG] file: " << file << "\n"
        "[DEBUG] name: " << name << "\n"
        "[DEBUG] targ: " << target_pathname << std::endl;
    }

    // if the target file is in the list of extra files, we remove it
    // from the extra files.
    {
      std::lock_guard<std::mutex> lock(extraFilesMutex);
      std::vector<std::string>::iterator it = extrafiles.begin();
      for (; it != extrafiles.end(); ++it)
      {
	const std::string& e = *it;
	const std::string& extra_pathname = e;
	if (!extra_pathname.compare(target_pathname))
	{
	  extrafiles.erase(it);
	  break;
	}
      }
    }

    DiskFile *diskfile = new DiskFile(sout, serr);

    // Does the target file exist
    if (!diskfile->Open(file))
    {
      // The file does not exist.
      delete diskfile;

      if (noiselevel > nlSilent)
      {
        LockedStream(sout) << "Target: \"" << name << "\" - missing." << std::endl;
      }

      if (observer)
      {
        const std::string reported = ReportedName(sourcefile, name);

        observer->OnFile(reported);
        observer->OnFileDone(reported, 0, BlocksNeeded(sourcefile));
      }

      return;
    }

    // Remember that we have processed this file. Two source files of the set
    // may name the same one, and the insert is what settles which thread gets
    // it.
    bool claimed;
    {
      std::lock_guard<std::mutex> lock(diskFileMapMutex);
      claimed = diskFileMap.Insert(diskfile);
    }

    if (!claimed)
    {
      // The file has already been used!
      diskfile->Close();
      delete diskfile;

      LockedStream(serr) << "Source file " << name << " is a duplicate." << std::endl;

      if (observer)
      {
        const std::string reported = ReportedName(sourcefile, name);

        observer->OnFile(reported);
        observer->OnFileDone(reported, 0, BlocksNeeded(sourcefile));
      }

      finalresult = false;

      return;
    }

    // Yes. Record that fact.
    sourcefile->SetTargetExists(true);

    // Remember that the DiskFile is the target file
    sourcefile->SetTargetFile(diskfile);

    // Do the actual verification
    if (!VerifyDataFile(diskfile, sourcefile, basepath, progress))
      finalresult = false;

    // We have finished with the file for now
    diskfile->Close();
  });

  // Find out how much data we have found
  UpdateVerificationResults();

  return finalresult;
}

// Scan any extra files specified on the command line
bool Par2Repairer::VerifyExtraFiles(const std::vector<std::string> &extrafiles, const std::string &basepath, const bool renameonly)
{
  if (noiselevel > nlQuiet)
    sout << "\nScanning extra files:\n" << std::endl;

  if (completefilecount < mainpacket->RecoverableFileCount())
  {
    // Total size of extra files for mt-progress line
    u64 mttotalextrasize = 0;
    for (size_t i=0; i<extrafiles.size(); ++i)
      mttotalextrasize += DiskFile::GetFileSize(extrafiles[i]);

    ProgressMeter<u64> progress(sout, "Scanning: ", mttotalextrasize, noiselevel, observer);

    foreach_parallel(extrafiles, FileThreads(extrafiles.size()), [&](const std::string &extrafile)
    {
      if (IsCancelled())
        return;

      std::string filename = extrafile;

      // If the filename does not have a .par2 / .PAR2 / .Par2 extension we are interested in it.
      if (!IsPar2Filename(filename))
      {
        filename = DiskFile::GetCanonicalPathname(filename);

        // Has this file already been dealt with
        bool b;
        {
          std::lock_guard<std::mutex> lock(diskFileMapMutex);
          b = diskFileMap.Find(filename) == 0;
        }
        if (b)
        {
          DiskFile *diskfile = new DiskFile(sout, serr);

          // Does the file exist
          if (!diskfile->Open(filename))
          {
            delete diskfile;
            return;
          }

          // Remember that we have processed this file. Another thread may be
          // scanning the same one, and the insert is what settles which gets
          // it.
          bool claimed;
          {
            std::lock_guard<std::mutex> lock(diskFileMapMutex);
            claimed = diskFileMap.Insert(diskfile);
          }

          if (!claimed)
          {
            diskfile->Close();
            delete diskfile;
            return;
          }

          // Do the actual verification
          VerifyDataFile(diskfile, 0, basepath, progress, renameonly);
          // Ignore errors

          // We have finished with the file for now
          diskfile->Close();
        }
      }
    });
  }
  // Find out how much data we have found
  UpdateVerificationResults();

  return true;
}

// Attempt to match the data in the DiskFile with the source file
bool Par2Repairer::VerifyDataFile(DiskFile *diskfile, Par2RepairerSourceFile *sourcefile, const std::string &basepath, ProgressMeter<u64> &progress, const bool renameonly)
{
  std::string localname;
  DiskFile::SplitRelativeFilename(diskfile->FileName(), basepath, localname);

  // An extra file the set does not name keeps the name it has on disk for
  // both reports, even where the scan goes on to find it is a file of the set.
  const std::string name = ReportedName(sourcefile, localname);

  if (observer)
    observer->OnFile(name);

  u32 blocksfound = 0;
  const bool matched = MatchDataFile(diskfile, sourcefile, basepath, progress, renameonly, blocksfound);

  if (observer)
    observer->OnFileDone(name, blocksfound, BlocksNeeded(sourcefile));

  return matched;
}

bool Par2Repairer::MatchDataFile(DiskFile *diskfile, Par2RepairerSourceFile *&sourcefile, const std::string &basepath, ProgressMeter<u64> &progress, const bool renameonly, u32 &blocksfound)
{
  MatchType matchtype; // What type of match was made
  MD5Hash hashfull;    // The MD5 Hash of the whole file
  MD5Hash hash16k;     // The MD5 Hash of the files 16k of the file

  // Are there any files that can be verified at the block level
  if (blockverifiable)
  {
    // Scan the file at the block level.

    if (!ScanDataFile(diskfile,   // [in]      The file to scan
                      basepath,
                      progress,
                      renameonly, // [in]      Only look for perfect matches
                      sourcefile, // [in/out]  Modified in the match is for another source file
                      matchtype,  // [out]
                      hashfull,   // [out]
                      hash16k,    // [out]
                      blocksfound)) // [out]
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
      if (buffersize > std::min(blocksize, filesize))
        buffersize = (size_t)std::min(blocksize, filesize);

      char *buffer = new char[buffersize];

      u64 offset = 0;

      MD5Context context;

      while (offset < filesize)
      {
        size_t want = (size_t)std::min((u64)buffersize, filesize-offset);

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
      context.Final(hashfull);

      // If we did not have 16k of data, then the 16k hash
      // is the same as the full hash
      if (filesize < 16384)
      {
        hash16k = hashfull;
      }
    }

    std::list<Par2RepairerSourceFile*>::iterator sf = unverifiablesourcefiles.begin();

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
          LockedStream(sout) << diskfile->FileName() << " is a perfect match for " << sourcefile->GetDescriptionPacket()->FileName() << std::endl;
        }
        // Record that we have a perfect match for this source file
        sourcefile->SetCompleteFile(diskfile);

        if (blocksallocated)
        {
          // Allocate all of the DataBlocks for the source file to the DiskFile

          u64 offset = 0;
          u64 filesize = sourcefile->GetDescriptionPacket()->FileSize();

          std::vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();

          while (offset < filesize)
          {
            DataBlock &datablock = *sb;

            datablock.SetLocation(diskfile, offset);
            datablock.SetLength(std::min(blocksize, filesize-offset));

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

// Check every block of a source file at the offset where it is expected to be.
bool Par2Repairer::ScanDataFileAligned(DiskFile               *diskfile,   // [in]
                                       ProgressMeter<u64>     &progress,   // [in]
                                       Par2RepairerSourceFile *sourcefile, // [in]
                                       std::vector<char>      &matched,    // [out]
                                       u32                    &matchcount, // [out]
                                       MD5Hash                &hashfull,   // [out]
                                       MD5Hash                &hash16k)    // [out]
{
  matchcount = 0;

  // We must know which source file the data is supposed to belong to
  if (0 == sourcefile)
    return false;

  VerificationPacket *verificationpacket = sourcefile->GetVerificationPacket();
  if (0 == verificationpacket)
    return false;

  // A source file which has already been matched must not claim its blocks again
  if (0 != sourcefile->GetCompleteFile())
    return false;

  // Unless the file is exactly the right length it cannot be a perfect match
  const u64 filesize = sourcefile->GetDescriptionPacket()->FileSize();
  if (diskfile->FileSize() != filesize)
    return false;

  const u32 blockcount = verificationpacket->BlockCount();
  if (0 == blockcount)
    return false;

  // A single thread gains nothing from checking the blocks up front, and a
  // damaged file would then be read a second time by the scan below
  if (!blockpool || blockpool->ThreadCount() < 2
      || scanbuffers.Count() < 2 || scanbuffers.Size() < blocksize)
    return false;

  matched.assign(blockcount, 0);

  // The files being scanned at once share the pool's threads and the buffers
  // they read into, so a file left scanning on its own reads as far ahead as
  // the whole pool can keep up with
  struct ActiveReader
  {
    explicit ActiveReader(std::atomic<u32> &count) : count(count) {++count;}
    ~ActiveReader(void) {--count;}
    std::atomic<u32> &count;
  } active(activereaders);

  FileHasher filehasher(fullhash);

  const size_t slots = scanbuffers.Count();
  const u32    batchblocks = (u32)(scanbuffers.Size() / blocksize);

  std::unique_ptr<TaskPool::Batch[]> batches(new TaskPool::Batch[slots]);

  // The blocks of a batch are next to each other, so they are read in one go.
  // Only the last block of a file can be short, and its entry covers it padded
  // out to the full block size with zeroes
  auto readbatch = [&](char *into, const u32 first, const u32 blocks)
  {
    const u64 offset = static_cast<u64>(first) * blocksize;
    const size_t span = static_cast<size_t>(blocks) * blocksize;
    const size_t length = (size_t)std::min(static_cast<u64>(span), filesize - offset);

    if (!diskfile->Read(offset, into, length))
      return false;

    filehasher.Update(offset, into, length);

    if (length < span)
      memset(&into[length], 0, span - length);

    return true;
  };

  // A verification entry is the 20 bytes a block is expected to hash to, so the
  // packet is handed to the hasher as it stands
  static_assert(sizeof(FILEVERIFICATIONENTRY) == 20, "a verification entry is a block hash");

  // A hasher belongs to one thread at a time, and the blocks of a batch go to
  // whichever of the pool's threads and the threads waiting on it take them, so
  // a block is checked with one taken for it and given back afterwards. There is
  // one for every thread which could be checking this file at once
  std::vector<std::unique_ptr<Hasher> > hashers;
  std::vector<Hasher*>                  idle;
  std::mutex                            idlemutex;

  for (u32 i = 0; i < blockpool->ThreadCount() + filethreads; ++i)
  {
    HasherConfig config;

    std::unique_ptr<Hasher> hasher = backends.hasher
      ? backends.hasher(config)
      : std::unique_ptr<Hasher>(new ReferenceHasher());

    if (!hasher || !hasher->Init(filesize, (size_t)blocksize, false))
      return false;

    idle.push_back(hasher.get());
    hashers.push_back(std::move(hasher));
  }

  struct Borrowed
  {
    Borrowed(std::vector<Hasher*> &idle, std::mutex &mutex)
    : idle(idle)
    , mutex(mutex)
    {
      std::lock_guard<std::mutex> lock(mutex);

      hasher = idle.back();
      idle.pop_back();
    }

    ~Borrowed(void)
    {
      std::lock_guard<std::mutex> lock(mutex);

      idle.push_back(hasher);
    }

    std::vector<Hasher*> &idle;
    std::mutex           &mutex;
    Hasher               *hasher;
  };

  // One block goes in a submission, because that is how the pool hands them
  // out. Filling the lanes of a hasher which takes several at once means having
  // it hand out a range of them instead
  auto checkblock = [&](const char *from, const u32 first, const u32 block)
  {
    const u64 length = std::min(blocksize, filesize - static_cast<u64>(block) * blocksize);
    const char *data = &from[static_cast<size_t>(block - first) * blocksize];

    char result = 0;

    {
      Borrowed borrowed(idle, idlemutex);

      borrowed.hasher->CheckBlocks(data, 1, 0, verificationpacket->VerificationEntry(block), &result);
    }

    if (!result)
      return;

    matched[block] = 1;

    progress.Add(length);
  };

  // What each batch which has been given to the pool is checking. The pool
  // takes its blocks one at a time, alongside the blocks of every other file
  // being scanned.
  struct Slot
  {
    decltype(&checkblock) check;
    BufferPool           *buffers;
    size_t                held;   // the buffer it read into
    u32                   first;

    void operator()(const size_t block) const
      {(*check)(buffers->At(held), first, (u32)block);}
  };

  std::vector<Slot> slot(slots);
  for (size_t s = 0; s < slots; ++s)
  {
    slot[s].check = &checkblock;
    slot[s].buffers = &scanbuffers;
  }

  bool readfailed = false;
  u32  nextblock = 0;

  size_t next = 0;         // the slot the next batch is given
  size_t oldest = 0;       // the slot of the batch which has been out longest
  size_t outstanding = 0;

  // Waits for the batch which has been with the pool longest, helping to check
  // it, and gives back the buffer it was reading from
  const std::function<void(void)> retire = [&]()
  {
    blockpool->Wait(batches[oldest]);
    scanbuffers.Give(slot[oldest].held);

    oldest = (oldest + 1) % slots;
    --outstanding;
  };

  // The pool has to be finished with every batch before the buffers and the
  // state the batches point at go away, whichever way the scan is left. This
  // is declared last so that it runs before any of them.
  struct Drain
  {
    const std::function<void(void)> *retire;
    const size_t                    *outstanding;

    ~Drain(void)
    {
      while (*outstanding > 0)
      {
        try
        {
          (*retire)();
        }
        catch (...)
        {
          // A batch which failed still has to be waited for, and the buffer
          // it was reading into still has to be given back
        }
      }
    }
  } drain{&retire, &outstanding};

  while (nextblock < blockcount)
  {
    if (IsCancelled())
      return false;

    // A file keeps to its share of the buffers while others are being read,
    // and takes back the oldest of its own rather than waiting on them
    const size_t share = std::max<size_t>(2, slots / std::max(1u, activereaders.load()));

    size_t buffer = 0;

    for (;;)
    {
      // Over its share, this file takes back one of its own before it may
      // read any further ahead
      if (outstanding >= share)
      {
        retire();
        continue;
      }

      if (scanbuffers.TryTake(buffer))
        break;

      // Every buffer is with another file. Waiting for this file's own oldest
      // batch keeps it from waiting on the files it is sharing them with,
      // which it has to do only when it has no batch of its own to take back
      if (0 == outstanding)
      {
        buffer = scanbuffers.Take();
        break;
      }

      retire();
    }

    const u32 blocks = std::min(batchblocks, blockcount - nextblock);

    slot[next].held = buffer;
    slot[next].first = nextblock;

    // The buffer is the pool's to give back only once the batch reading into
    // it has been submitted
    try
    {
      if (!readbatch(scanbuffers.At(buffer), nextblock, blocks))
      {
        scanbuffers.Give(buffer);
        readfailed = true;
        break;
      }

      blockpool->Submit(batches[next], nextblock, nextblock + blocks, slot[next]);
    }
    catch (...)
    {
      scanbuffers.Give(buffer);
      throw;
    }

    next = (next + 1) % slots;
    ++outstanding;
    nextblock += blocks;
  }

  while (outstanding > 0)
    retire();

  if (readfailed)
    return false;

  filehasher.GetHashes(filesize, hashfull, hash16k);

  for (u32 b=0; b<blockcount; ++b)
    if (matched[b])
      matchcount++;

  return true;
}

// Perform a sliding window scan of the DiskFile looking for blocks of data that
// might belong to any of the source files (for which a verification packet was
// available). If a block of data might be from more than one source file, prefer
// the one specified by the "sourcefile" parameter. If the first data block
// found is for a different source file then "sourcefile" is changed accordingly.
bool Par2Repairer::ScanDataFile(DiskFile                *diskfile,    // [in]
                                std::string             basepath,     // [in]
                                ProgressMeter<u64>      &progress,    // [in]
                                const bool              renameonly,   // [in]
                                Par2RepairerSourceFile* &sourcefile,  // [in/out]
                                MatchType               &matchtype,   // [out]
                                MD5Hash                 &hashfull,    // [out] only set if there are unverifiable source files
                                MD5Hash                 &hash16k,     // [out] only set if there are unverifiable source files
                                u32                     &count)       // [out]
{
  // Remember which file we wanted to match
  Par2RepairerSourceFile *originalsourcefile = sourcefile;

  std::string name;
  DiskFile::SplitRelativeFilename(diskfile->FileName(), basepath, name);

  // Is the file empty
  if (diskfile->FileSize() == 0)
  {
    matchtype = eNoMatch;
    count = 0;
    // The hash of an empty file is needed to match against source files
    // which have no verification packet.
    if (!unverifiablesourcefiles.empty())
    {
      MD5Context context;
      context.Final(hash16k);
      hashfull = hash16k;
    }

    // If the file is empty, then just return
    if (noiselevel > nlSilent)
    {
      if (originalsourcefile != 0)
      {
        LockedStream(sout) << "Target: \"" << name << "\" - empty." << std::endl;
      }
      else
      {
        LockedStream(sout) << "File: \"" << name << "\" - empty." << std::endl;
      }
    }

    return true;
  }

  std::string shortname;
  if (name.size() > 56)
  {
    shortname = name.substr(0, 28) + "..." + name.substr(name.size()-28);
  }
  else
  {
    shortname = name;
  }

  if (noiselevel > nlQuiet)
  {
    LockedStream(sout) << "Opening: \"" << shortname << "\"" << std::endl;
  }

  // Assume we will make a perfect match for the file
  matchtype = eFullMatch;

  // How many matches have we had
  count = 0;

  // How many blocks have already been found
  u32 duplicatecount = 0;

  // Have we found data blocks in this file that belong to more than one target file
  bool multipletargets = false;

  // Total number of bytes that were skipped whilst scanning
  u64 skippeddata = 0;

  const u64 filesize = diskfile->FileSize();

  std::vector<char> alignedmatch;
  u32 alignedcount = 0;
  const bool vouched = TakeKnownBlocks(diskfile, sourcefile, alignedmatch, alignedcount);
  const bool aligned = vouched
                       || ScanDataFileAligned(diskfile, progress, sourcefile,
                                              alignedmatch, alignedcount,
                                              hashfull, hash16k);

  // Being told that none of the blocks are usable is an answer in itself, so
  // the file is not searched after all
  if (vouched && alignedcount == 0)
  {
    matchtype = eNoMatch;
    count = 0;

    return true;
  }

  // The parts of the file which still have to be searched a byte at a time
  std::vector<std::pair<u64, u64> > searchranges;

  if (aligned && alignedcount > 0)
  {
    const u32 blockcount = (u32)alignedmatch.size();

    // Record the blocks which were found where they were expected. Their
    // lengths were set when the source file was given its data blocks.
    std::vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();

    for (u32 blocknumber=0; blocknumber<blockcount; ++blocknumber)
    {
      if (alignedmatch[blocknumber])
      {
        if (blocksallocated)
          (*sb).SetLocation(diskfile, (u64)blocknumber * blocksize);

        count++;
      }

      ++sb;
    }

    if (alignedcount < blockcount)
    {
      matchtype = ePartialMatch;

      // In rename-only mode, skip files that are not perfect matches
      if (renameonly)
        return true;

      // Search each run of blocks which was not where it was expected. The
      // blocks on either side of a run have been claimed already, so nothing
      // outside these ranges is left to find.
      for (u32 blocknumber=0; blocknumber<blockcount; )
      {
        if (alignedmatch[blocknumber])
        {
          ++blocknumber;
          continue;
        }

        const u32 gapfirst = blocknumber;
        while (blocknumber < blockcount && !alignedmatch[blocknumber])
          ++blocknumber;

        searchranges.push_back(std::make_pair((u64)gapfirst * blocksize,
                                              std::min(filesize, (u64)blocknumber * blocksize)));
      }
    }
  }
  else
  {
    // Nothing is known about where the data is, so search all of it
    searchranges.push_back(std::make_pair((u64)0, filesize));
  }

  // Whichever scan read the whole of the file from its start produced the 16k
  // hash, and the whole file hash if that was asked for. Vouched blocks are
  // taken without reading anything
  bool filehashes = aligned && !vouched;

  if (!searchranges.empty())
  {

  const bool wholefilesearched = 1 == searchranges.size()
                                 && 0 == searchranges[0].first
                                 && filesize == searchranges[0].second;

  // The MD5 hash of the whole file is only needed to match against source
  // files which have no verification packet, and when it was asked for.
  const bool computefilehashes = ((fullhash && !aligned) || !unverifiablesourcefiles.empty())
                                 && wholefilesearched;

  filehashes = filehashes || wholefilesearched;

  // Create the checksummer for the file
  FileCheckSummer filechecksummer(diskfile, blocksize, windowtable, computefilehashes);

  // How far will we scan the file (1 byte at a time)
  // before skipping ahead looking for the next block
  u64 scandistance = std::min(skipleaway<<1, blocksize);

  // Distance to skip forward if we don't find a block
  u64 scanskip = skipdata ? blocksize - scandistance : 0;

  for (size_t range=0; range<searchranges.size(); ++range)
  {
  const u64 rangestart = searchranges[range].first;
  const u64 rangeend = searchranges[range].second;

  if (!filechecksummer.Start(rangestart))
    return false;

  // Which block do we expect to find first. Nothing is suggested at the start
  // of a range, just as nothing is suggested after a block is not found.
  const VerificationHashEntry *nextentry = 0;

  // Assume with are half way through scanning
  u64 scanoffset = scandistance >> 1;

  // Offset of last data that was found
  u64 lastmatchoffset = rangestart;

  u64 oldoffset = rangestart;
  u64 printprogress = 0;

  // Whilst we have not reached the end of the range
  while (filechecksummer.Offset() < rangeend)
  {
    if (IsCancelled())
      break;

    // Update progress indicator
    printprogress += filechecksummer.Offset() - oldoffset;
    if (printprogress >= blocksize || filechecksummer.ShortBlock())
    {
      progress.Add(printprogress);
      printprogress = 0;
    }
    oldoffset = filechecksummer.Offset();

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
        progress.PrintLine((std::ostringstream()
          << "No data found between offset " << lastmatchoffset
          << " and " << filechecksummer.Offset()).str());
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

          // In rename-only mode, skip files that are not perfect matches
          if (renameonly)
          {
            return true;
          }
        }
      }
      else
      {
        // If the match found is not the one which was expected
        // then this is a partial match

        if (currententry != nextentry)
        {
          matchtype = ePartialMatch;

          // In rename-only mode, skip files that are not perfect matches
          if (renameonly)
          {
            return true;
          }
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

      // In rename-only mode, skip files that are not perfect matches
      if (renameonly)
      {
        return true;
      }

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
            && skipfrom < rangeend)
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

  if (filechecksummer.Offset() >= rangeend)
    progress.Add(filechecksummer.Offset() - oldoffset);

  if (lastmatchoffset < filechecksummer.Offset() && noiselevel > nlNormal)
  {
    progress.PrintLine((std::ostringstream()
      << "No data found between offset " << lastmatchoffset
      << " and " << filechecksummer.Offset()).str());
  }

  }

  // Get the Full and 16k hash values of the file
  if (wholefilesearched)
    filechecksummer.GetFileHashes(hashfull, hash16k);

  }

  if (noiselevel >= nlDebug)
  {
    std::ostringstream ss;
    if (duplicatecount > 0)
      ss << "[DEBUG] duplicates: " << duplicatecount << '\n';
    ss << "[DEBUG] matchcount: " << count << "\n"
      "[DEBUG] ----------------------";
    progress.PrintLine(ss.str());
  }

  // Did we make any matches at all
  if (count > 0)
  {
    // If this still might be a perfect match, check the file size and number
    // of blocks to confirm. A full match verifies the file against the
    // verification packet, and the description packet's hashes are a
    // separate claim, so the 16k hash is checked as well, and the hash of
    // the whole file when that was asked for.
    if (matchtype            != eFullMatch ||
        count                != sourcefile->GetVerificationPacket()->BlockCount() ||
        diskfile->FileSize() != sourcefile->GetDescriptionPacket()->FileSize() ||
        (filehashes &&
         (hash16k != sourcefile->GetDescriptionPacket()->Hash16k() ||
          (fullhash &&
           hashfull != sourcefile->GetDescriptionPacket()->HashFull()))))
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
            LockedStream(sout) << "Target: \""
              << name
              << "\" - damaged, found "
              << count
              << " data blocks from several target files."
              << std::endl;
          }
          else
          {
            LockedStream(sout) << "File: \""
              << name
              << "\" - found "
              << count
              << " data blocks from several target files."
              << std::endl;
          }
        }
        else
        {
          // Did we find data blocks that belong to the target file
          if (originalsourcefile == sourcefile)
          {
            LockedStream(sout) << "Target: \""
              << name
              << "\" - damaged. Found "
              << count
              << " of "
              << sourcefile->GetVerificationPacket()->BlockCount()
              << " data blocks."
              << std::endl;
          }
          // Were we scanning the target file or an extra file
          else if (originalsourcefile != 0)
          {
            std::string targetname;
            DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

            LockedStream(sout) << "Target: \""
              << name
              << "\" - damaged. Found "
              << count
              << " of "
              << sourcefile->GetVerificationPacket()->BlockCount()
              << " data blocks from \""
              << targetname
              << "\"."
              << std::endl;
          }
          else
          {
            std::string targetname;
            DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

            LockedStream(sout) << "File: \""
              << name
              << "\" - found "
              << count
              << " of "
              << sourcefile->GetVerificationPacket()->BlockCount()
              << " data blocks from \""
              << targetname
              << "\"."
              << std::endl;
          }
        }

        if (skippeddata > 0)
        {
          LockedStream(sout) << skippeddata << " bytes of data were skipped whilst scanning.\n"
            "If there are not enough blocks found to repair: try again "
            "with the -N option." << std::endl;
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
          LockedStream(sout) << "Target: \"" << name << "\" - found." << std::endl;
        }
        // Were we scanning the target file or an extra file
        else if (originalsourcefile != 0)
        {
          std::string targetname;
          DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

          LockedStream(sout) << "Target: \""
            << name
            << "\" - is a match for \""
            << targetname
            << "\"."
            << std::endl;
        }
        else
        {
          std::string targetname;
          DiskFile::SplitRelativeFilename(sourcefile->TargetFileName(), basepath, targetname);

          LockedStream(sout) << "File: \""
            << name
            << "\" - is a match for \""
            << targetname
            << "\"."
            << std::endl;
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
        LockedStream(sout) << "File: \""
          << name
          << "\" - found "
          << duplicatecount
          << " duplicate data blocks."
          << std::endl;
      }
      else
      {
        LockedStream(sout) << "File: \""
          << name
          << "\" - no data found."
          << std::endl;
      }

      if (skippeddata > 0)
      {
        LockedStream(sout) << skippeddata << " bytes of data were skipped whilst scanning.\n"
          "If there are not enough blocks found to repair: try again "
          "with the -N option." << std::endl;
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
  std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

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

          renamedlist[sourcefile->TargetFileName()] =
            sourcefile->GetCompleteFile()->FileName();
        }

        availableblockcount += sourcefile->BlockCount();
      }
      else
      {
        // Count the number of blocks that have been found
        std::vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();
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
      sout << "Repair is required." << std::endl;
    if (noiselevel > nlQuiet)
    {
      if (renamedfilecount > 0) sout << renamedfilecount << " file(s) have the wrong name.\n";
      if (missingfilecount > 0) sout << missingfilecount << " file(s) are missing.\n";
      if (damagedfilecount > 0) sout << damagedfilecount << " file(s) exist but are damaged.\n";
      if (completefilecount > 0) sout << completefilecount << " file(s) are ok.\n";

      sout << "You have " << availableblockcount
        << " out of " << sourceblockcount
        << " data blocks available." << std::endl;
      if (recoverypacketmap.size() > 0)
        sout << "You have " << (u32)recoverypacketmap.size()
          << " recovery blocks available." << std::endl;
    }

    // Is repair possible
    if (recoverypacketmap.size() >= missingblockcount)
    {
      if (noiselevel > nlSilent)
        sout << "Repair is possible." << std::endl;

      if (noiselevel > nlQuiet)
      {
        if (recoverypacketmap.size() > missingblockcount)
          sout << "You have an excess of "
            << (u32)recoverypacketmap.size() - missingblockcount
            << " recovery blocks." << std::endl;

        if (missingblockcount > 0)
          sout << missingblockcount
            << " recovery blocks will be used to repair." << std::endl;
        else if (recoverypacketmap.size())
          sout << "None of the recovery blocks will be used for the repair." << std::endl;
      }

      return true;
    }
    else
    {
      if (noiselevel > nlSilent)
      {
        sout << "Repair is not possible.\n"
          "You need " << missingblockcount - recoverypacketmap.size()
          << " more recovery blocks to be able to repair." << std::endl;
      }

      return false;
    }
  }
  else
  {
    if (noiselevel > nlSilent)
      sout << "All files are correct, repair is not required." << std::endl;

    return true;
  }

  return true;
}

// Rename any damaged or missnamed target files.
bool Par2Repairer::RenameTargetFiles(void)
{
  u32 filenumber = 0;
  std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Rename any damaged target files
  while (sf != sourcefiles.end() && filenumber < mainpacket->TotalFileCount())
  {
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile)
    {
      // If the target file exists but is not a complete version of the file
      if (sourcefile->GetTargetExists() &&
          sourcefile->GetTargetFile() != sourcefile->GetCompleteFile())
      {
        DiskFile *targetfile = sourcefile->GetTargetFile();

        // Rename it
        diskFileMap.Remove(targetfile);

        const bool renamed = targetfile->Rename();

        bool success = diskFileMap.Insert(targetfile);
        assert(success);

        if (!renamed)
          return false;

        backuplist.push_back(targetfile);

        // We no longer have a target file
        sourcefile->SetTargetExists(false);
        sourcefile->SetTargetFile(0);
      }
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
    if (sourcefile)
    {
      // If there is no targetfile and there is a complete version
      if (sourcefile->GetTargetFile() == 0 &&
          sourcefile->GetCompleteFile() != 0)
      {
        DiskFile *targetfile = sourcefile->GetCompleteFile();

        // Rename it
        diskFileMap.Remove(targetfile);

        const bool renamed = targetfile->Rename(sourcefile->TargetFileName());

        bool success = diskFileMap.Insert(targetfile);
        assert(success);

        if (!renamed)
          return false;

        // This file is now the target file
        sourcefile->SetTargetExists(true);
        sourcefile->SetTargetFile(targetfile);

        // We have one more complete file
        completefilecount++;
      }
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
  std::vector<Par2RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Create any missing target files
  while (sf != sourcefiles.end() && filenumber < mainpacket->TotalFileCount())
  {
    Par2RepairerSourceFile *sourcefile = *sf;
    if (sourcefile)
    {
      // If the file does not exist
      if (!sourcefile->GetTargetExists())
      {
        DiskFile *targetfile = new DiskFile(sout, serr);
        std::string filename = sourcefile->TargetFileName();
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
        std::vector<DataBlock>::iterator tb = sourcefile->TargetBlocks();

        // Allocate all of the target data blocks
        while (offset < filesize)
        {
          DataBlock &datablock = *tb;

          datablock.SetLocation(targetfile, offset);
          datablock.SetLength(std::min(blocksize, filesize-offset));

          offset += blocksize;
          ++tb;
        }

        // Add the file to the list of those that will need to be verified
        // once the repair has completed.
        verifylist.push_back(sourcefile);
      }
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

  std::vector<DataBlock*>::iterator inputblock  = inputblocks.begin();
  std::vector<DataBlock*>::iterator copyblock   = copyblocks.begin();
  std::vector<DataBlock*>::iterator outputblock = outputblocks.begin();

  // Build an array listing which source data blocks are present and which are missing
  std::vector<bool> present;
  present.resize(sourceblockcount);

  std::vector<DataBlock>::iterator sourceblock  = sourceblocks.begin();
  std::vector<DataBlock>::iterator targetblock  = targetblocks.begin();
  std::vector<bool>::iterator              pres = present.begin();

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
  std::map<u32,RecoveryPacket*>::iterator rp = recoverypacketmap.begin();

  // The exponents of those recovery blocks, kept for the processor
  std::vector<u16> exponents;

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

    exponents.push_back((u16)exponent);

    ++inputblock;
    ++rp;
  }

  // If we need to, compute and solve the RS matrix
  if (missingblockcount == 0)
    return true;

  // Offer the erasure pattern, so that an implementation able to solve it for
  // itself is not made to wait for the matrix to be inverted only to read
  // columns out of it
  ownfactors = processor->OfferErasures(present, exponents.data(), (u32)exponents.size());

  if (ownfactors)
    return true;

  bool success = rs.Compute(noiselevel, sout, serr);

  return success;
}

// The files being read take the buffers they read into from these, which
// between them hold two batches for each of the filecount files which may be
// read at once. A batch is a whole number of blocks, at least one, and no more
// than MAX_CHUNK_SIZE unless a single block is already larger than that.
void Par2Repairer::ResetScanBuffers(const size_t filecount)
{
  // The blocks of a file are only checked where they are expected to be when
  // there are verification packets to check them against and more than one
  // thread to do it with, so otherwise nothing would ever be read into them
  if (!blockverifiable || totalthreads < 2)
  {
    scanbuffers.Reset(0, 0);
    return;
  }

  // Every file being read shares these threads to check its blocks with
  if (!blockpool)
    blockpool.reset(new TaskPool(totalthreads));

  if (blockpool->ThreadCount() < 2)
  {
    scanbuffers.Reset(0, 0);
    return;
  }

  const u32 readers = FileThreads(filecount);

  const size_t batchsize = (size_t)std::max(1u, totalthreads / readers) * (size_t)blocksize;
  const size_t maxbatchsize = MAX_CHUNK_SIZE != 0
    ? std::max((size_t)blocksize, (size_t)MAX_CHUNK_SIZE)
    : batchsize;

  // Never more than the caller allowed for the work, down to a block a buffer,
  // which is the least a batch can be
  const size_t affordable = std::max((size_t)blocksize, scanmemorylimit / (2 * readers));

  scanbuffers.Reset(2 * readers, std::min(std::min(batchsize, maxbatchsize), affordable));
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

  if (MAX_CHUNK_SIZE != 0 && chunksize > MAX_CHUNK_SIZE)
    chunksize = MAX_CHUNK_SIZE;

  // Allocate the two buffers
  transferbuffer = new u8[(size_t)chunksize * NUM_TRANSFER_BUFFERS];
  outputbuffer = new u8[(size_t)chunksize];

  ProcessorConfig config;
  config.numthreads = totalthreads;
  config.memorylimit = memorylimit;

  processor = backends.processor
    ? backends.processor(config)
    : std::unique_ptr<Processor>(new ReferenceProcessor(rs, totalthreads));

  if (!processor || !processor->Init((size_t)chunksize, missingblockcount))
  {
    serr << "Could not allocate buffer memory." << std::endl;
    return false;
  }

  if (noiselevel >= nlDebug)
    sout << "[DEBUG] Process chunk size: " << chunksize << std::endl;

  if (transferbuffer == NULL || outputbuffer == NULL)
  {
    serr << "Could not allocate buffer memory." << std::endl;
    return false;
  }

  return true;
}

// Read source data, process it through the RS matrix and write it to disk.
bool Par2Repairer::ProcessData(u64 blockoffset, size_t blocklength, ProgressMeter<u64> &progress)
{
  u64 totalwritten = 0;

  std::vector<DataBlock*>::iterator inputblock = inputblocks.begin();
  std::vector<DataBlock*>::iterator copyblock  = copyblocks.begin();
  u32                          inputindex = 0;

  DiskFile *lastopenfile = NULL;

  // Are there any blocks which need to be reconstructed
  if (missingblockcount > 0)
  {
    processor->SetChunkLength(blocklength);
    processor->ResetOutput();

    // The matrix column for one input block, unused when the processor has its own
    std::vector<u16> factors(ownfactors ? 0 : missingblockcount);

    // Every buffer starts free
    std::future<void> bufferfree[NUM_TRANSFER_BUFFERS];
    for (u32 buffer=0; buffer<NUM_TRANSFER_BUFFERS; buffer++)
    {
      std::promise<void> free;
      free.set_value();
      bufferfree[buffer] = free.get_future();
    }
    u32 bufferindex = 0;

    // For each input block
    while (inputblock != inputblocks.end())
    {
      if (IsCancelled())
        break;

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

      // Wait for the next input buffer to come free
      void *inputbuffer = &((u8*)transferbuffer)[(size_t)chunksize * bufferindex];
      bufferfree[bufferindex].get();

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

      // Look up the matrix column and process the data against every output block
      if (!ownfactors)
      {
        for (u32 outputindex=0; outputindex<missingblockcount; outputindex++)
          factors[outputindex] = rs.GetFactor(inputindex, outputindex);
      }

      processor->WaitForAdd();
      bufferfree[bufferindex] = processor->AddInput(inputbuffer, blocklength, inputindex,
                                                    ownfactors ? NULL : factors.data());
      bufferindex = (bufferindex + 1) % NUM_TRANSFER_BUFFERS;

      progress.Add(blocklength);

      ++inputblock;
      ++inputindex;
    }

    processor->EndInput();
  }
  else
  {
    // Reconstruction is not required, we are just copying blocks between files

    // For each block that might need to be copied
    while (copyblock != copyblocks.end())
    {
      if (IsCancelled())
        break;

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
        if (!(*inputblock)->ReadData(blockoffset, blocklength, transferbuffer))
          return false;

        size_t wrote;
        if (!(*copyblock)->WriteData(blockoffset, blocklength, transferbuffer, wrote))
          return false;
        totalwritten += wrote;
      }

      progress.Add(blocklength);

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
  std::vector<DataBlock*>::iterator outputblock = outputblocks.begin();
  for (u32 outputindex=0; outputindex<missingblockcount;outputindex++)
  {
    // Take the accumulated output block from the processor
    const void *outbuf = processor->PeekOutput(outputindex);
    if (outbuf == NULL)
    {
      if (!processor->GetOutput(outputindex, outputbuffer))
      {
        serr << "Could not read the repaired data back from the processor." << std::endl;
        return false;
      }
      outbuf = outputbuffer;
    }

    // Write the data to the target file
    size_t wrote;
    if (!(*outputblock)->WriteData(blockoffset, blocklength, outbuf, wrote))
      return false;
    totalwritten += wrote;

    ++outputblock;
  }

  if (noiselevel > nlQuiet)
    sout << "Wrote " << totalwritten << " bytes to disk" << std::endl;

  return true;
}

// Verify that all of the reconstructed target files are now correct
bool Par2Repairer::VerifyTargetFiles(const std::string &basepath)
{
  std::atomic<bool> finalresult(true);

  // Verify the target files in alphabetical order
  std::sort(verifylist.begin(), verifylist.end(), SortSourceFilesByFileName);

  u64 mttotalsize = 0;

  for (size_t i=0; i<verifylist.size(); ++i)
  {
    if (verifylist[i])
      mttotalsize += verifylist[i]->GetDescriptionPacket()->FileSize();
  }
  ProgressMeter<u64> progress(sout, "Scanning: ", mttotalsize, noiselevel, observer);

  // Iterate through each file in the verification list
  foreach_parallel(verifylist, FileThreads(verifylist.size()), [&](Par2RepairerSourceFile *sourcefile)
  {
    if (IsCancelled())
      return;

    DiskFile *targetfile = sourcefile->GetTargetFile();

    // Close the file
    if (targetfile->IsOpen())
      targetfile->Close();

    // Mark all data blocks for the file as unknown
    std::vector<DataBlock>::iterator sb = sourcefile->SourceBlocks();
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
      return;
    }

    // Verify the file again
    if (!VerifyDataFile(targetfile, sourcefile, basepath, progress))
      finalresult = false;

    // Close the file again
    targetfile->Close();
  });

  // Find out how much data we have found
  UpdateVerificationResults();

  return finalresult;
}

// Delete all of the partly reconstructed files
bool Par2Repairer::DeleteIncompleteTargetFiles(void)
{
  std::vector<Par2RepairerSourceFile*>::iterator sf = verifylist.begin();

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
  std::vector<DiskFile*>::iterator bf = backuplist.begin();

  if (noiselevel > nlSilent
      && bf != backuplist.end())
  {
    sout << "\nPurge backup files." << std::endl;
  }

  // Iterate through each file in the backuplist
  while (bf != backuplist.end())
  {
    if (noiselevel > nlSilent)
    {
      std::string name;
      std::string path;
      DiskFile::SplitFilename((*bf)->FileName(), path, name);
      sout << "Remove \"" << name << "\"." << std::endl;
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
    sout << "\nPurge par files." << std::endl;
  }

  for (std::list<std::string>::const_iterator s=par2list.begin(); s!=par2list.end(); ++s)
  {
    DiskFile *diskfile = new DiskFile(sout, serr);

    if (diskfile->Open(*s))
    {
      if (noiselevel > nlSilent)
      {
        std::string name;
        std::string path;
        DiskFile::SplitFilename((*s), path, name);
        sout << "Remove \"" << name << "\"." << std::endl;
      }

      if (diskfile->IsOpen())
        diskfile->Close();
      diskfile->Delete();
    }

    delete diskfile;
  }

  return true;
}

} // namespace par2
