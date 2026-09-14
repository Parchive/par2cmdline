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

// The command line falls back to this when -m is not given
static const size_t DEFAULT_MEMORY_LIMIT = 256 * 1048576;

// Par2Repairer carries out the work; deriving from it reaches the individual
// steps without making them part of its public interface.
class Par2Verifier::Impl : public Par2Repairer
{
public:
  Impl(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
       const std::string &_basepath, const Backends &backends)
    : Par2Repairer(sout, serr, noiselevel, backends)
  {
    basepath = _basepath;
  }

  // setchanged reports whether the set the packets describe is now a
  // different shape, which is what makes an earlier scan of the data useless
  Result Add(const std::string &parfilename, bool *setchanged)
  {
    ClearLastError();

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
    {
      if (IsCancelled())
        return eCancelled;

      errorlog.Record(ecInternalError, "Could not load the PAR2 packets", parfilename);
      return eLogicError;
    }

    if (packetsloaded == before && !DiskFile::FileExists(parfilename))
    {
      errorlog.Record(ecPar2FileMissing, "There is no such PAR2 file", parfilename);
      return eFileIOError;
    }

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
    ClearLastError();

    if (0 == mainpacket)
    {
      errorlog.Record(ecMainPacketMissing, "The PAR2 files do not describe a set");
      return eInsufficientCriticalData;
    }

    UpdateVerificationResults();

    if (!CheckVerificationResults())
      return eRepairNotPossible;

    if (completefilecount < mainpacket->RecoverableFileCount())
      return eRepairPossible;

    return eSuccess;
  }

  Result Scan(const std::string &filename, const u32 _nthreads, const u32 _filethreads)
  {
    ApplyThreadCounts(_nthreads, _filethreads);

    return ScanFile(filename, basepath);
  }

  void SetDataSkipping(const bool _skipdata, const u64 _skipleaway)
  {
    skipdata = _skipdata;
    skipleaway = _skipleaway;
  }

  Result Check(const std::vector<std::string> &_extrafiles,
               const u32 _nthreads,
               const u32 _filethreads)
  {
    ClearLastError();

    if (0 == mainpacket)
    {
      errorlog.Record(ecMainPacketMissing, "The PAR2 files do not describe a set");
      return eInsufficientCriticalData;
    }

    ApplyThreadCounts(_nthreads, _filethreads);

    std::vector<std::string> extrafiles = _extrafiles;

    return VerifyFiles(basepath, extrafiles, false);
  }

  Result Rebuild(const size_t memorylimit,
                 const u32 _nthreads,
                 const u32 _filethreads,
                 const bool verifyafter)
  {
    ClearLastError();

    if (0 == mainpacket)
    {
      errorlog.Record(ecMainPacketMissing, "The PAR2 files do not describe a set");
      return eInsufficientCriticalData;
    }

    ApplyThreadCounts(_nthreads, _filethreads);

    return RepairFiles(memorylimit, basepath, verifyafter);
  }

  bool SetInfo(Par2SetInfo *info) const
  {
    if (0 == info || 0 == mainpacket)
      return false;

    memcpy(info->setid.data(), setid.hash, sizeof(setid.hash));
    info->blocksize = blocksize;
    info->datablocks = sourceblockcount;
    info->recoveryblocks = (u32)recoverypacketmap.size();
    info->recoverablefilecount = mainpacket->RecoverableFileCount();
    info->otherfilecount = mainpacket->TotalFileCount() - mainpacket->RecoverableFileCount();
    info->datasize = totaldatasize;

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
  impl.reset(new Impl(sout, serr, noiselevel, basepath, backends));
  impl->SetObserver(observer);
  impl->SetDataSkipping(skipdata, skipleaway);

  for (std::map<std::string, std::vector<bool> >::const_iterator kb = knownblocks.begin();
       kb != knownblocks.end();
       ++kb)
  {
    impl->SetKnownBlocks(kb->first, kb->second);
  }

  for (std::vector<std::string>::const_iterator f = par2files.begin();
       f != par2files.end();
       ++f)
  {
    impl->Add(*f, 0);
  }

  verified = false;

  const std::vector<std::string> scanned = scannedfiles;
  scannedfiles.clear();

  for (std::vector<std::string>::const_iterator f = scanned.begin();
       f != scanned.end();
       ++f)
  {
    VerifyFile(*f);
  }
}

Par2Verifier::Par2Verifier(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
                           const std::string &_basepath, const Backends &_backends)
: sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, backends(_backends)
, observer(0)
, memorylimit(DEFAULT_MEMORY_LIMIT)
, nthreads(0)
, filethreads(0)
, skipdata(false)
, skipleaway(0)
, par2files()
, scannedfiles()
, knownblocks()
, verified(false)
, basepath(NormaliseBasePath(_basepath))
, lasterror()
, impl(new Impl(sout, serr, noiselevel, basepath, backends))
{
}

Par2Verifier::~Par2Verifier()
{
}

// The engine records the error, but Restart throws the engine away, so the
// handle keeps its own copy of what the call it is returning from recorded.
void Par2Verifier::TakeLastError(void)
{
  lasterror = Par2Error();
  impl->GetLastError(&lasterror);
}

// What the handle itself has to report, rather than the work it delegates
void Par2Verifier::RecordLastError(const ErrorCode code, const std::string &message)
{
  lasterror = Par2Error();
  lasterror.code = code;
  lasterror.message = message;

  if (observer)
    observer->OnError(lasterror);
}

bool Par2Verifier::GetLastError(Par2Error *error) const
{
  if (0 == error || ecNone == lasterror.code)
    return false;

  *error = lasterror;
  return true;
}

void Par2Verifier::SetObserver(Par2Observer *_observer)
{
  observer = _observer;
  impl->SetObserver(_observer);
}

void Par2Verifier::SetMemoryLimit(const size_t _memorylimit)
{
  memorylimit = (_memorylimit != 0) ? _memorylimit : DEFAULT_MEMORY_LIMIT;
}

void Par2Verifier::SetDataSkipping(const bool enabled, const u64 leaway)
{
  skipdata = enabled;
  skipleaway = leaway;

  impl->SetDataSkipping(skipdata, skipleaway);
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
  {
    lasterror = Par2Error();
    return eSuccess;
  }

  // Take it from the first PAR2 file named, before any packets are read
  if (basepath.empty())
  {
    basepath = NormaliseBasePath(BasePathFor(parfilename));
    impl->SetBasePath(basepath);
  }

  bool setchanged = false;
  const Result result = impl->Add(parfilename, &setchanged);

  // Restart replays the scans through VerifyFile, which would otherwise leave
  // the handle holding what the replay found rather than what Add recorded
  TakeLastError();
  const Par2Error added = lasterror;

  // Remembered even without the critical packets, so that a later restart
  // replays it alongside the file that completes the set
  if (result != eFileIOError)
    par2files.push_back(parfilename);

  // Extra recovery data leaves what the scan found still true, so it is kept
  // and Reassess can use it. A set of a different shape does not. Files scanned
  // before the set was known are replayed by the same restart.
  if (setchanged && (verified || !scannedfiles.empty()))
    Restart();

  lasterror = added;

  return result;
}

Result Par2Verifier::Reassess(void)
{
  if (!verified)
  {
    RecordLastError(ecNotVerified, "Nothing has been verified yet");
    return eLogicError;
  }

  const Result result = impl->Reassess();
  TakeLastError();

  return result;
}

bool Par2Verifier::GetSetInfo(Par2SetInfo *info) const
{
  return impl->SetInfo(info);
}

bool Par2Verifier::GetFileInfo(std::vector<Par2FileInfo> *files) const
{
  return impl->GetFileInfo(files);
}

bool Par2Verifier::GetBlockChecksums(const std::string &filename,
                                    std::vector<u32> *crcs) const
{
  return impl->GetBlockChecksums(filename, crcs);
}

// Guarded by verified, unlike GetBlockChecksums: before anything has been
// scanned every block would read as not found, which is not the same as
// nothing having been looked at.
bool Par2Verifier::GetFoundBlocks(const std::string &filename,
                                  std::vector<bool> *blocks) const
{
  if (!verified)
    return false;

  return impl->GetFoundBlocks(filename, blocks);
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

void Par2Verifier::SetKnownBlocks(const std::string &filename,
                                 const std::vector<bool> &blocks)
{
  if (blocks.empty())
    knownblocks.erase(filename);
  else
    knownblocks[filename] = blocks;

  impl->SetKnownBlocks(filename, blocks);
}

Result Par2Verifier::Verify(const std::vector<std::string> &extrafiles)
{
  // A full pass covers everything the individual scans did, so they are dropped
  // rather than replayed into it
  scannedfiles.clear();

  if (verified)
    Restart();

  const Result result = impl->Check(extrafiles, nthreads, filethreads);
  TakeLastError();
  verified = true;

  return result;
}

Result Par2Verifier::VerifyFile(const std::string &filename)
{
  const Result result = impl->Scan(filename, nthreads, filethreads);
  TakeLastError();

  if (result == eCancelled)
    return result;

  // Remembered even when the set is not known yet, so that adding the PAR2 file
  // which describes it replays the scan rather than losing it
  scannedfiles.push_back(filename);

  if (result != eInsufficientCriticalData)
    verified = true;

  return result;
}

Result Par2Verifier::Repair(const bool verifyafter)
{
  if (!verified)
  {
    RecordLastError(ecNotVerified, "Nothing has been verified yet");
    return eLogicError;
  }

  if (!impl->CanRepair())
  {
    lasterror = Par2Error();
    return eRepairNotPossible;
  }

  const Result result = impl->Rebuild(memorylimit, nthreads, filethreads, verifyafter);
  TakeLastError();

  return result;
}

void Par2Verifier::Cancel(void)
{
  impl->Cancel();
}

void Par2Verifier::ClearCancel(void)
{
  impl->ClearCancel();
}



// Par2CreatorEngine carries out the work; deriving from it reaches the
// individual steps without making them part of its public interface.
class Par2Creator::Impl : public Par2CreatorEngine
{
public:
  Impl(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel, const Backends &backends)
    : Par2CreatorEngine(sout, serr, noiselevel, backends)
  {
  }
};

// A create leaves the engine holding the packets of the set it wrote, so a
// second one has to start from a new engine. Nothing is replayed into it:
// every setting lives on the handle and is passed into the run.
void Par2Creator::Restart(void)
{
  impl.reset(new Impl(sout, serr, noiselevel, backends));
  impl->SetObserver(observer);

  if (cancelled)
    impl->Cancel();
}

// The engine records the error, but Restart throws the engine away, so the
// handle keeps its own copy of what the call it is returning from recorded.
void Par2Creator::TakeLastError(void)
{
  lasterror = Par2Error();
  impl->GetLastError(&lasterror);
}

Par2Creator::Par2Creator(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
                         const std::string &_basepath, const Backends &_backends)
: sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, backends(_backends)
, observer(0)
, sourcefiles()
, blocksize(0)
, recoveryblockcount(0)
, recoveryfilescheme(scVariable)
, recoveryfilecount(0)
, firstrecoveryblock(0)
, memorylimit(DEFAULT_MEMORY_LIMIT)
, nthreads(0)
, filethreads(0)
, cancelled(false)
, basepath(NormaliseBasePath(_basepath))
, lasterror()
, impl(new Impl(sout, serr, noiselevel, backends))
{
}

Par2Creator::~Par2Creator()
{
}

void Par2Creator::SetObserver(Par2Observer *_observer)
{
  observer = _observer;
  impl->SetObserver(_observer);
}

void Par2Creator::AddSourceFile(const std::string &filename)
{
  sourcefiles.push_back(filename);
}

void Par2Creator::AddSourceFiles(const std::vector<std::string> &filenames)
{
  sourcefiles.insert(sourcefiles.end(), filenames.begin(), filenames.end());
}

void Par2Creator::SetBlockSize(const u64 _blocksize)
{
  blocksize = _blocksize;
}

void Par2Creator::SetRecoveryBlockCount(const u32 _recoveryblockcount)
{
  recoveryblockcount = _recoveryblockcount;
}

void Par2Creator::SetRecoveryFileScheme(const Scheme scheme, const u32 _recoveryfilecount)
{
  recoveryfilescheme = scheme;
  recoveryfilecount = _recoveryfilecount;
}

void Par2Creator::SetFirstRecoveryBlock(const u32 firstblock)
{
  firstrecoveryblock = firstblock;
}

void Par2Creator::SetMemoryLimit(const size_t _memorylimit)
{
  memorylimit = (_memorylimit != 0) ? _memorylimit : DEFAULT_MEMORY_LIMIT;
}

void Par2Creator::SetThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  nthreads = _nthreads;
  filethreads = _filethreads;
}

Result Par2Creator::Create(const std::string &parfilename)
{
  // Taken from the name of the set, before any file is read
  if (basepath.empty())
    basepath = NormaliseBasePath(BasePathFor(parfilename));

  // The volume files are named after the set rather than after its index file,
  // so the name is taken either way round
  std::string setname = parfilename;
  if (setname.length() > 5 &&
      0 == stricmp(setname.substr(setname.length()-5, 5).c_str(), ".par2"))
  {
    setname = setname.substr(0, setname.length()-5);
  }

  // Resolved against the working directory, so that the names the set records
  // come out relative to the basepath whatever the caller wrote them as
  std::vector<std::string> files;
  files.reserve(sourcefiles.size());
  for (std::vector<std::string>::const_iterator f = sourcefiles.begin();
       f != sourcefiles.end();
       ++f)
  {
    files.push_back(DiskFile::GetCanonicalPathname(*f));
  }

  Restart();

  const Result result = impl->Process(memorylimit,
                                      basepath,
                                      nthreads,
                                      filethreads,
                                      setname,
                                      files,
                                      blocksize,
                                      firstrecoveryblock,
                                      recoveryfilescheme,
                                      recoveryfilecount,
                                      recoveryblockcount);
  TakeLastError();

  return result;
}

void Par2Creator::Cancel(void)
{
  cancelled = true;
  impl->Cancel();
}

void Par2Creator::ClearCancel(void)
{
  cancelled = false;
  impl->ClearCancel();
}

bool Par2Creator::GetLastError(Par2Error *error) const
{
  if (0 == error || ecNone == lasterror.code)
    return false;

  *error = lasterror;
  return true;
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
  Par2CreatorEngine creator(sout, serr, noiselevel, backends);
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
