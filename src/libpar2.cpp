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

#include "libpar2internal.h"

namespace par2
{

// Par2Repairer carries out the work; deriving from it reaches the individual
// steps without making them part of its public interface.
class Par2Verifier::Impl : public Par2Repairer
{
public:
  Impl(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
       const std::string &_basepath)
    : Par2Repairer(sout, serr, noiselevel)
  {
    basepath = _basepath;
  }

  // setchanged reports whether the set the packets describe is now a
  // different shape, which is what makes an earlier scan of the data useless
  Result Add(const std::string &parfilename, bool *setchanged)
  {
    const std::vector<std::string> none;

    const u32 blocksbefore = sourceblockcount;
    const size_t filesbefore = sourcefiles.size();

    // LoadPackets moves on quietly when nothing can be read, but a caller
    // naming one file at a time needs to know whether the name was any use.
    // The name may be that of a file or of a whole set, and its packets may
    // already have been read, so it is only useless when there is no such
    // file and nothing new arrived.
    const u32 before = packetsloaded;

    // Read it again even if it has been seen before
    if (!LoadPackets(parfilename, none, true))
      return IsCancelled() ? eCancelled : eLogicError;

    if (packetsloaded == before && !DiskFile::FileExists(parfilename))
      return eFileIOError;

    const Result result = PreparePackets();

    if (setchanged)
      *setchanged = (sourceblockcount != blocksbefore) || (sourcefiles.size() != filesbefore);

    return result;
  }

  void SetBasePath(const std::string &_basepath)
  {
    basepath = _basepath;
  }

  // Whether the recovery data covers what the last scan found missing
  bool CanRepair(void) const
  {
    return recoverypacketmap.size() >= missingblockcount;
  }

  // Work out again whether the data found by an earlier scan can be repaired
  // with the recovery blocks available now
  Result Reassess(void)
  {
    if (0 == mainpacket)
      return eInsufficientCriticalData;

    UpdateVerificationResults();

    if (!CheckVerificationResults())
      return eRepairNotPossible;

    if (completefilecount < mainpacket->RecoverableFileCount())
      return eRepairPossible;

    return eSuccess;
  }

  Result Check(const std::vector<std::string> &_extrafiles,
               const size_t memorylimit,
               const bool _skipdata,
               const u64 _skipleaway,
               const u32 _nthreads,
               const u32 _filethreads)
  {
    if (0 == mainpacket)
      return eInsufficientCriticalData;

    ApplyThreadCounts(_nthreads, _filethreads);
    ApplyMemoryLimit(memorylimit);

    skipdata = _skipdata;
    skipleaway = _skipleaway;

    std::vector<std::string> extrafiles = _extrafiles;

    return VerifyFiles(basepath, extrafiles, false);
  }

  Result Rebuild(const size_t memorylimit,
                 const u32 _nthreads,
                 const u32 _filethreads)
  {
    if (0 == mainpacket)
      return eInsufficientCriticalData;

    ApplyThreadCounts(_nthreads, _filethreads);

    return RepairFiles(memorylimit, basepath);
  }

  bool SetInfo(Par2SetInfo *info) const
  {
    if (0 == info || 0 == mainpacket)
      return false;

    memcpy(info->setid.data(), setid.hash, sizeof(setid.hash));
    info->blocksize = blocksize;
    info->datablocks = sourceblockcount;
    info->recoverablefilecount = mainpacket->RecoverableFileCount();
    info->otherfilecount = mainpacket->TotalFileCount() - mainpacket->RecoverableFileCount();
    info->datasize = totaldatasize;
    if (creatorpacket)
      info->creator = creatorpacket->Client();

    return true;
  }
};

// Append a path separator unless there is one already. Empty is left alone.
static std::string WithSeparator(const std::string &path)
{
  if (path.empty())
    return path;

  const std::string last = path.substr(path.length() - 1);
  if (PATHSEP == last || ALTPATHSEP == last)
    return path;

  return path + PATHSEP;
}

// Absolute and ending in a separator, which is the form every path this API
// reports. The separator goes on first, so "." names the directory itself.
// Empty is left alone.
static std::string NormaliseBasePath(const std::string &path)
{
  if (path.empty())
    return path;

  return WithSeparator(DiskFile::GetCanonicalPathname(WithSeparator(path)));
}

// The directory a PAR2 file is in, which is where the tool looks with no -B
static std::string BasePathFor(const std::string &parfilename)
{
  std::string path;
  std::string name;
  DiskFile::SplitFilename(parfilename, path, name);

  std::string basepath = DiskFile::GetCanonicalPathname(path);

  if (basepath.empty())
    basepath = DiskFile::GetCanonicalPathname("./");

  return basepath;
}

// Verifying leaves a Par2Repairer with the results of that one pass, so a
// second pass has to start from a new one. The PAR2 files that were added are
// read again, which is cheap next to scanning the data files.
void Par2Verifier::Restart(void)
{
  const bool wascancelled = impl->IsCancelled();

  impl.reset(new Impl(sout, serr, noiselevel, basepath));

  // The replay repeats work the observer has already been told about, so it is
  // told none of it. The observer is attached once the handle is back where it
  // was, and the cancel with it, so that neither affects the replay itself.

  for (std::map<std::string, std::vector<bool> >::const_iterator kb = knownblocks.begin();
       kb != knownblocks.end();
       ++kb)
  {
    impl->SetKnownBlocks(kb->first, kb->second);
  }

  for (const auto &par2file : par2files)
  {
    impl->Add(par2file, 0);
  }

  impl->SetObserver(observer);

  if (wascancelled)
    impl->Cancel();

  verified = false;
  scanned = false;
}

Par2Verifier::Par2Verifier(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
                           const std::string &_basepath)
: sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, observer(0)
, memorylimit(DEFAULT_MEMORY_LIMIT)
, nthreads(0)
, filethreads(0)
, par2files()
, knownblocks()
, verified(false)
, scanned(false)
, basepath(NormaliseBasePath(_basepath))
, impl(new Impl(sout, serr, noiselevel, basepath))
{
}

Par2Verifier::~Par2Verifier() = default;

void Par2Verifier::SetObserver(Par2Observer *_observer)
{
  observer = _observer;
  impl->SetObserver(_observer);
}

void Par2Verifier::SetMemoryLimit(const size_t _memorylimit)
{
  memorylimit = (_memorylimit != 0) ? _memorylimit : DEFAULT_MEMORY_LIMIT;
}

void Par2Verifier::SetThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  nthreads = _nthreads;
  filethreads = _filethreads;
}

Result Par2Verifier::AddPar2File(const std::string &parfilename)
{
  // Naming the same file again is not an error, there is simply nothing to do
  if (std::find(par2files.begin(), par2files.end(), parfilename) != par2files.end())
    return eSuccess;

  // Take it from the first PAR2 file named, before any packets are read
  if (basepath.empty())
  {
    basepath = NormaliseBasePath(BasePathFor(parfilename));
    impl->SetBasePath(basepath);
  }

  bool setchanged = false;
  const Result result = impl->Add(parfilename, &setchanged);

  // Remembered even without the critical packets, so that a later restart
  // replays it alongside the file that completes the set
  if (result != eFileIOError)
    par2files.push_back(parfilename);

  // Extra recovery data leaves what the scan found still true, so it is kept
  // and Reassess can use it. A set of a different shape does not.
  if (scanned && setchanged)
    Restart();

  return result;
}

Result Par2Verifier::Reassess(void)
{
  if (!verified)
    return eLogicError;

  return impl->Reassess();
}

bool Par2Verifier::GetSetInfo(Par2SetInfo *info) const
{
  return impl->SetInfo(info);
}

bool Par2Verifier::GetFileInfo(std::vector<Par2FileInfo> *files) const
{
  return impl->GetFileInfo(files);
}

bool Par2Verifier::GetBackupFiles(std::vector<std::string> *files) const
{
  return impl->GetBackupFiles(files);
}

bool Par2Verifier::GetRenamedFiles(std::vector<std::pair<std::string, std::string> > *files) const
{
  return impl->GetRenamedFiles(files);
}

bool Par2Verifier::GetVerifyResult(Par2VerifyResult *result) const
{
  if (!verified)
    return false;

  return impl->GetVerifyResult(result);
}

bool Par2Verifier::SetKnownBlocks(const std::string &filename,
                                 const std::vector<bool> &blocks)
{
  if (!impl->SetKnownBlocks(filename, blocks))
    return false;

  if (blocks.empty())
    knownblocks.erase(filename);
  else
    knownblocks[filename] = blocks;

  return true;
}

Result Par2Verifier::Verify(const std::vector<std::string> &extrafiles,
                           const bool skipdata,
                           const u64 skipleaway)
{
  if (scanned)
    Restart();

  const Result result = impl->Check(extrafiles, memorylimit, skipdata, skipleaway,
                                    nthreads, filethreads);

  if (result != eInsufficientCriticalData)
    scanned = true;

  if (result != eInsufficientCriticalData && result != eCancelled)
    verified = true;

  return result;
}

Result Par2Verifier::Repair(void)
{
  if (!verified)
    return eLogicError;

  if (!impl->CanRepair())
    return eRepairNotPossible;

  return impl->Rebuild(memorylimit, nthreads, filethreads);
}

void Par2Verifier::Cancel(void)
{
  impl->Cancel();
}

void Par2Verifier::ClearCancel(void)
{
  impl->ClearCancel();
}


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
		  const Backends &backends
		  )
{
  Par2Creator creator(sout, serr, noiselevel, backends);
  Result result = creator.Process(
				  memorylimit,
				  basepath,
				  nthreads,
				  filethreads,
				  parfilename,
				  extrafiles,
				  blocksize,
				  firstblock,
				  recoveryfilescheme,
				  recoveryfilecount,
				  recoveryblockcount
				  );
  return result;
}


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
		  const bool fullhash,
		  const Backends &backends
		  )
{
  Par2Repairer repairer(sout, serr, noiselevel, backends);
  Result result = repairer.Process(
				   memorylimit,
				   basepath,
				   nthreads,
				   filethreads,
				   parfilename,
				   extrafiles,
				   dorepair,
				   purgefiles,
				   renameonly,
				   skipdata,
				   skipleaway,
				   fullhash);

  return result;
}


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
		  )
{
  Par1Repairer repairer(sout, serr, noiselevel);
  Result result = repairer.Process(memorylimit,
				   nthreads,
				   parfilename,
				   extrafiles,
				   dorepair,
				   purgefiles);
  return result;
}


// Determine how many recovery files to create.
bool ComputeRecoveryFileCount(std::ostream &sout,
			      std::ostream &serr,
			      u32 *recoveryfilecount,
			      Scheme recoveryfilescheme,
			      u32 recoveryblockcount,
			      u64 largestfilesize,
			      u64 blocksize)
{
  // Are we computing any recovery blocks
  if (recoveryblockcount == 0)
  {
    *recoveryfilecount = 0;
    return true;
  }

  switch (recoveryfilescheme)
  {
  case scUnknown:
    {
      //assert(false);
      serr << "Scheme unspecified (create, verify, or repair)." << std::endl;
      return false;
    }
    break;
  case scVariable:
  case scUniform:
    {
      if (*recoveryfilecount == 0)
      {
        // If none specified then then filecount is roughly log2(blockcount)
        // This prevents you getting excessively large numbers of files
        // when the block count is high and also allows the files to have
        // sizes which vary exponentially.

        for (u32 blocks=recoveryblockcount; blocks>0; blocks>>=1)
        {
          (*recoveryfilecount)++;
        }
      }

      if (*recoveryfilecount > recoveryblockcount)
      {
        // You cannot have more recovery files than there are recovery blocks
        // to put in them.
        serr << "Too many recovery files specified." << std::endl;
        return false;
      }
    }
    break;

  case scLimited:
    {
      // No recovery file will contain more recovery blocks than would
      // be required to reconstruct the largest source file if it
      // were missing. Other recovery files will have recovery blocks
      // distributed in an exponential scheme.

      u32 largest = (u32)((largestfilesize + blocksize-1) / blocksize);
      u32 whole = recoveryblockcount / largest;
      whole = (whole >= 1) ? whole-1 : 0;

      u32 extra = recoveryblockcount - whole * largest;
      *recoveryfilecount = whole;
      for (u32 blocks=extra; blocks>0; blocks>>=1)
      {
        (*recoveryfilecount)++;
      }
    }
    break;
  }

  return true;
}

} // namespace par2
