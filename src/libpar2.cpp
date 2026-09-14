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

// This webpage has code to get physical memory size on many OSes
// http://nadeausoftware.com/articles/2012/09/c_c_tip_how_get_physical_memory_size_system

#ifdef _WIN32
u64 GetTotalPhysicalMemory(void)
{
  u64 TotalPhysicalMemory = 0;

  HMODULE hLib = ::LoadLibraryA("kernel32.dll");
  if (NULL != hLib)
  {
    BOOL (WINAPI *pfn)(LPMEMORYSTATUSEX) = (BOOL (WINAPI*)(LPMEMORYSTATUSEX))::GetProcAddress(hLib, "GlobalMemoryStatusEx");

    if (NULL != pfn)
    {
      MEMORYSTATUSEX mse;
      mse.dwLength = sizeof(mse);
      if (pfn(&mse))
      {
	TotalPhysicalMemory = mse.ullTotalPhys;
      }
    }

    ::FreeLibrary(hLib);
  }

  if (TotalPhysicalMemory == 0)
  {
    MEMORYSTATUS ms;
    ::ZeroMemory(&ms, sizeof(ms));
    ::GlobalMemoryStatus(&ms);

    TotalPhysicalMemory = ms.dwTotalPhys;
  }

  return TotalPhysicalMemory;
}
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
// POSIX compliant OSes, including OSX/MacOS and Cygwin.  Also works for Linux.
u64 GetTotalPhysicalMemory(void)
{
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return 0;

  return (u64)pages * (u64)page_size;
}
#else
// default version == unable to request memory size
u64 GetTotalPhysicalMemory(void)
{
  return 0;
}
#endif

size_t DefaultMemoryLimit(void)
{
  // 1/8th of total physical memory, floored to 256MiB on a machine with more
  // than that, and 256MiB when it cannot be found
  const u64 total = GetTotalPhysicalMemory() / 1048576;
  u64 limit = total / 8;
  if (limit < 256 && (total == 0 || total > 256))
    limit = 256;

  // limit to 1GB on 32-bit platforms to avoid exhausing the addressable memory space
  if (sizeof(uintptr_t) < 8 && limit > 1024)
    limit = 1024;

  return (size_t)limit * 1048576;
}

// What the work may use: the caller's limit, or the default when it set none,
// and never less than the 1MB the command line allows
static size_t MemoryLimit(const size_t requested)
{
  return std::max<size_t>((requested != 0) ? requested : DefaultMemoryLimit(), 1048576);
}

// Par2Repairer carries out the work; deriving from it reaches the individual
// steps without making them part of its public interface.
class Par2Verifier::Impl : public Par2Repairer
{
public:
  Impl(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
       const std::string &_basepath, const Backends &backends)
    : Par2Repairer(sout, serr, noiselevel, backends)
    , prepared(eInsufficientCriticalData)
  {
    basepath = _basepath;
  }

  // How many of the source files have a description packet
  size_t DescribedFileCount(void) const
  {
    return std::count_if(sourcefiles.begin(), sourcefiles.end(),
                         [](const Par2RepairerSourceFile *sf) { return sf != 0; });
  }

  // How many of the source files have a verification packet
  size_t VerifiableFileCount(void) const
  {
    return std::count_if(sourcefiles.begin(), sourcefiles.end(),
                         [](const Par2RepairerSourceFile *sf) { return sf != 0 && sf->GetVerificationPacket() != 0; });
  }

  // setchanged reports whether the set the packets describe is now a
  // different shape, which is what makes an earlier scan of the data useless
  Result Add(const std::string &parfilename, bool *setchanged)
  {
    ClearLastError();

    const std::vector<std::string> none;

    const u32 blocksbefore = sourceblockcount;
    const size_t filesbefore = DescribedFileCount();
    const size_t verifiablebefore = VerifiableFileCount();

    // LoadPackets moves on quietly when nothing can be read, but a caller
    // naming one file at a time needs to know whether the name was any use.
    // The name may be that of a file or of a whole set, and its packets may
    // already have been read, so it is only useless when there is no such
    // file and nothing new arrived.
    const u32 before = packetsloaded;

    // Read it again even if it has been seen before
    if (!LoadPackets(parfilename, none, true) || IsCancelled())
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

    // Nor is one which is there but could not be opened
    Par2Error failure;
    if (packetsloaded == before && errorlog.First(&failure) && ecFileOpenFailed == failure.code
        && DiskFile::GetCanonicalPathname(failure.filename) == DiskFile::GetCanonicalPathname(parfilename))
      return eFileIOError;

    prepared = PreparePackets();

    if (setchanged)
      *setchanged = (sourceblockcount != blocksbefore) || (DescribedFileCount() != filesbefore)
                    || (VerifiableFileCount() != verifiablebefore);

    return prepared;
  }

  void SetNoiseLevel(const NoiseLevel _noiselevel)
  {
    noiselevel = _noiselevel;
  }

  // Forget every scan, keeping what the PAR2 files hold
  void DiscardScans(void)
  {
    for (auto *sourcefile : sourcefiles)
    {
      if (0 == sourcefile)
        continue;

      sourcefile->SetTargetFile(0);
      sourcefile->SetTargetExists(false);
      sourcefile->SetCompleteFile(0);

      if (sourcefile->GetDescriptionPacket() != 0)
      {
        auto block = sourcefile->SourceBlocks();
        for (u32 i = 0; i < sourcefile->BlockCount(); ++i, ++block)
        {
          if (block->IsSet())
            block->ClearLocation();
        }
      }
    }

    renamedlist.clear();

    for (auto *diskfile : diskFileMap.Files())
    {
      if (0 == packetfiles.count(diskfile))
      {
        diskFileMap.Remove(diskfile);
        delete diskfile;
      }
    }
  }

  void SetBasePath(const std::string &_basepath)
  {
    basepath = _basepath;
  }

  void SetFullHash(const bool _fullhash)
  {
    fullhash = _fullhash;
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

    if (prepared != eSuccess)
    {
      errorlog.Record(ecTooManySourceBlocks, "The set needs more blocks than can be held");
      return prepared;
    }

    return ScanOutcome();
  }

  Result Scan(const std::string &filename, const size_t memorylimit,
              const u32 _nthreads, const u32 _filethreads)
  {
    if (0 != mainpacket && prepared != eSuccess)
    {
      errorlog.Record(ecTooManySourceBlocks, "The set needs more blocks than can be held");
      return prepared;
    }

    ApplyThreadCounts(_nthreads, _filethreads);
    ApplyMemoryLimit(memorylimit);

    return ScanFile(filename, basepath);
  }

  void SetDataSkipping(const bool _skipdata, const u64 _skipleaway)
  {
    skipdata = _skipdata;
    skipleaway = _skipleaway;
  }

  Result Check(const std::vector<std::string> &_extrafiles,
               const size_t memorylimit,
               const u32 _nthreads,
               const u32 _filethreads)
  {
    ClearLastError();

    if (0 == mainpacket)
    {
      errorlog.Record(ecMainPacketMissing, "The PAR2 files do not describe a set");
      return eInsufficientCriticalData;
    }

    if (prepared != eSuccess)
    {
      errorlog.Record(ecTooManySourceBlocks, "The set needs more blocks than can be held");
      return prepared;
    }

    ApplyThreadCounts(_nthreads, _filethreads);
    ApplyMemoryLimit(memorylimit);

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

    if (prepared != eSuccess)
    {
      errorlog.Record(ecTooManySourceBlocks, "The set needs more blocks than can be held");
      return prepared;
    }

    ApplyThreadCounts(_nthreads, _filethreads);

    return RepairFiles(memorylimit, basepath, verifyafter);
  }

private:
  Result prepared;                          // What the last PreparePackets returned
};

// Append a path separator unless there is one already. Empty is left alone.
std::string WithSeparator(const std::string &path)
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
std::string BasePathFor(const std::string &parfilename)
{
  std::string path;
  std::string name;
  DiskFile::SplitFilename(parfilename, path, name);

  std::string basepath = DiskFile::GetCanonicalPathname(path);

  if (basepath.empty())
    basepath = DiskFile::GetCanonicalPathname("./");

  return basepath;
}

// A repair, or a set which changes shape after files were scanned, leaves a
// Par2Repairer with results which no longer hold, so the work starts again from
// a new one. The PAR2 files that were added are read again.
void Par2Verifier::Restart(void)
{
  // A cancel which arrives while the engine is replaced and the PAR2 files
  // are read again is kept for the new one rather than given to either
  {
    std::lock_guard<std::mutex> lock(cancelmutex);
    restarting = true;
    impl = std::make_unique<Impl>(sout, serr, nlSilent, basepath, backends);
  }

  // The replay repeats work the observer has already been told about, so it is
  // told none of it, and nothing is written to the streams again. The observer
  // is attached once the handle is back where it was.
  impl->SetDataSkipping(skipdata, skipleaway);
  impl->SetFullHash(fullhash);

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

  verified = false;
  scanned = false;
  repaired = false;

  // A cancel reaches the rescan, which stops where it got to
  {
    std::lock_guard<std::mutex> lock(cancelmutex);
    restarting = false;

    if (cancelled)
      impl->Cancel();
  }

  const std::set<std::string> rescan = scannedfiles;
  scannedfiles.clear();

  bool cutshort = false;

  for (const auto &f : rescan)
  {
    if (impl->IsCancelled() || eCancelled == VerifyFile(f))
    {
      cutshort = true;
      break;
    }
  }

  // What the rescan did not reach is kept for the next one
  if (cutshort)
  {
    scannedfiles = rescan;
    verified = false;
  }

  impl->SetNoiseLevel(noiselevel);
  impl->SetObserver(observer);
}

// Discards everything written to it
class NullStream : public std::ostream
{
public:
  NullStream(void)
  : std::ostream(&buffer)
  , buffer()
  {
  }

private:
  class Buffer : public std::streambuf
  {
  protected:
    int_type overflow(int_type c) override
    {
      return c;
    }

    std::streamsize xsputn(const char *, std::streamsize n) override
    {
      return n;
    }
  };

  Buffer buffer;
};

Par2Verifier::Par2Verifier(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
                           const std::string &_basepath, Backends _backends)
: nullstream()
, sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, backends(std::move(_backends))
, observer(0)
, memorylimit(MemoryLimit(0))
, nthreads(0)
, filethreads(0)
, skipdata(false)
, skipleaway(DEFAULT_SKIP_LEAWAY)
, fullhash(false)
, par2files()
, scannedfiles()
, knownblocks()
, verified(false)
, scanned(false)
, repaired(false)
, readback(false)
, cancelled(false)
, restarting(false)
, basepath(NormaliseBasePath(_basepath))
, lasterror()
, impl(new Impl(sout, serr, noiselevel, basepath, backends))
{
}

Par2Verifier::Par2Verifier(const std::string &_basepath, Backends _backends)
: Par2Verifier(std::unique_ptr<NullStream>(new NullStream), _basepath, std::move(_backends))
{
}

Par2Verifier::Par2Verifier(std::unique_ptr<NullStream> _nullstream, const std::string &_basepath, Backends _backends)
: Par2Verifier(*_nullstream, *_nullstream, nlSilent, _basepath, std::move(_backends))
{
  nullstream = std::move(_nullstream);
}

Par2Verifier::~Par2Verifier() = default;

// The engine records the error, but Restart throws the engine away, so the
// handle keeps its own copy of what the call it is returning from recorded.
void Par2Verifier::TakeLastError(void)
{
  lasterror = Par2Error();
  impl->GetLastError(&lasterror);
}

// Make an error the one the call reports, and tell the observer
static void ReportError(Par2Error &lasterror, Par2Observer *observer, const ErrorCode code,
                        const std::string &message, const std::string &filename = std::string())
{
  lasterror = Par2Error();
  lasterror.code = code;
  lasterror.message = message;
  lasterror.filename = filename;

  if (observer)
    observer->OnError(lasterror);
}

// What the handle itself has to report, rather than the work it delegates
void Par2Verifier::RecordLastError(const ErrorCode code, const std::string &message)
{
  ReportError(lasterror, observer, code, message);
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
  memorylimit = MemoryLimit(_memorylimit);
}

void Par2Verifier::SetDataSkipping(const bool enabled, const u64 leaway)
{
  skipdata = enabled;
  skipleaway = (leaway != 0) ? leaway : DEFAULT_SKIP_LEAWAY;

  impl->SetDataSkipping(skipdata, skipleaway);
}

void Par2Verifier::SetFullHash(const bool enabled)
{
  fullhash = enabled;

  impl->SetFullHash(fullhash);
}

void Par2Verifier::SetThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  nthreads = _nthreads;
  filethreads = _filethreads;
}

Result Par2Verifier::AddPar2File(const std::string &_parfilename)
{
  const std::string parfilename = DiskFile::GetCanonicalPathname(_parfilename);

  // Naming the same file again is not an error, there is simply nothing to do
  if (std::find(par2files.begin(), par2files.end(), parfilename) != par2files.end())
  {
    lasterror = Par2Error();
    return eSuccess;
  }

  // Take it from the first PAR2 file named, before any packets are read
  const bool derived = basepath.empty();
  if (derived)
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
  // replays it alongside the file that completes the set. A cancelled read is
  // not.
  if (result != eFileIOError && result != eCancelled)
  {
    par2files.push_back(parfilename);
  }
  else if (derived && result == eFileIOError)
  {
    // Taken again from the next one
    basepath.clear();
    impl->SetBasePath(basepath);
  }

  // Extra recovery data leaves what the scan found still true, so it is kept
  // and Reassess can use it. A set of a different shape does not, even when the
  // scan was cancelled. Files scanned before the set was known are replayed by
  // the same restart.
  if (setchanged && (scanned || !scannedfiles.empty()))
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

  if (repaired)
  {
    RecordLastError(ecNotVerified, "Nothing has been verified since the last repair");
    return eLogicError;
  }

  const Result result = impl->Reassess();
  TakeLastError();

  return result;
}

bool Par2Verifier::GetSetInfo(Par2SetInfo *info) const
{
  return impl->GetSetInfo(info);
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
  if (!verified || (repaired && !readback))
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

Result Par2Verifier::Verify(const std::vector<std::string> &extrafiles)
{
  // A full pass covers everything the individual scans did, so they are dropped
  // rather than replayed into it. Whatever was scanned before, by a pass that
  // was cancelled too, is started afresh.
  scannedfiles.clear();

  if (repaired)
    Restart();
  else
    impl->DiscardScans();

  verified = false;

  const Result result = impl->Check(extrafiles, memorylimit, nthreads, filethreads);
  TakeLastError();

  if (result != eInsufficientCriticalData)
    scanned = true;

  if (result == eSuccess || result == eRepairPossible || result == eRepairNotPossible)
    verified = true;

  return result;
}

Result Par2Verifier::VerifyFile(const std::string &filename)
{
  // After a repair a new engine starts from nothing, and each file is scanned
  // again as it is fed in
  if (repaired)
  {
    scannedfiles.clear();
    Restart();
  }

  const Result result = impl->Scan(filename, memorylimit, nthreads, filethreads);
  TakeLastError();

  if (result != eInsufficientCriticalData)
    scanned = true;

  if (result == eCancelled)
    return result;

  // Remembered even when the set is not known yet, so that adding the PAR2 file
  // which describes it replays the scan rather than losing it. Scanning one
  // again replaces what the last scan of it found, so it is remembered once.
  scannedfiles.insert(filename);

  if (result == eSuccess || result == eRepairPossible || result == eRepairNotPossible)
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

  if (repaired)
  {
    RecordLastError(ecNotVerified, "Nothing has been verified since the last repair");
    return eLogicError;
  }

  if (!impl->CanRepair())
  {
    lasterror = Par2Error();
    return eRepairNotPossible;
  }

  repaired = true;

  const Result result = impl->Rebuild(memorylimit, nthreads, filethreads, verifyafter);
  readback = verifyafter && result == eSuccess;
  TakeLastError();

  return result;
}

void Par2Verifier::Cancel(void)
{
  std::lock_guard<std::mutex> lock(cancelmutex);
  cancelled = true;

  if (!restarting)
    impl->Cancel();
}

void Par2Verifier::ClearCancel(void)
{
  std::lock_guard<std::mutex> lock(cancelmutex);
  cancelled = false;

  if (!restarting)
    impl->ClearCancel();
}



// Par2SetCreator carries out the work
class Par2Creator::Impl : public Par2SetCreator
{
public:
  Impl(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel, const Backends &backends)
    : Par2SetCreator(sout, serr, noiselevel, backends)
  {
  }
};

// A create leaves the engine holding the packets of the set it wrote, so a
// second one has to start from a new engine. Nothing is replayed into it:
// every setting lives on the handle and is passed into the run.
void Par2Creator::Restart(void)
{
  std::lock_guard<std::mutex> lock(cancelmutex);

  impl = std::make_unique<Impl>(sout, serr, noiselevel, backends);
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
                         const std::string &_basepath, Backends _backends)
: nullstream()
, sout(sout)
, serr(serr)
, noiselevel(noiselevel)
, backends(std::move(_backends))
, observer(0)
, sourcefiles()
, blocksize(0)
, recoveryblockcount(0)
, recoveryfilescheme(scVariable)
, recoveryfilecount(0)
, firstrecoveryblock(0)
, memorylimit(MemoryLimit(0))
, nthreads(0)
, filethreads(0)
, cancelled(false)
, basepath(NormaliseBasePath(_basepath))
, lasterror()
, impl(new Impl(sout, serr, noiselevel, backends))
{
}

Par2Creator::Par2Creator(const std::string &_basepath, Backends _backends)
: Par2Creator(std::unique_ptr<NullStream>(new NullStream), _basepath, std::move(_backends))
{
}

Par2Creator::Par2Creator(std::unique_ptr<NullStream> _nullstream, const std::string &_basepath, Backends _backends)
: Par2Creator(*_nullstream, *_nullstream, nlSilent, _basepath, std::move(_backends))
{
  nullstream = std::move(_nullstream);
}

Par2Creator::~Par2Creator() = default;

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
  memorylimit = MemoryLimit(_memorylimit);
}

void Par2Creator::SetThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  nthreads = _nthreads;
  filethreads = _filethreads;
}

Result Par2Creator::Create(const std::string &parfilename)
{
  // Taken from the name of each set when none was given, before any file is
  // read
  const std::string setbasepath = basepath.empty()
    ? NormaliseBasePath(BasePathFor(parfilename))
    : basepath;

  // The volume files are named after the set rather than after its index file,
  // so the name is taken either way round
  std::string setname = parfilename;
  if (setname.length() > 5 &&
      0 == stricmp(setname.substr(setname.length()-5, 5).c_str(), ".par2"))
  {
    setname = setname.substr(0, setname.length()-5);
  }

  // Resolved against the working directory, so that the names the set records
  // come out relative to the basepath whatever the caller wrote them as. Empty
  // files and a file named twice are left out.
  std::vector<std::string> files;
  files.reserve(sourcefiles.size());
  for (const auto &sourcefile : sourcefiles)
  {
    const std::string file = DiskFile::GetCanonicalPathname(sourcefile);

    if (DiskFile::FileExists(file) && 0 == DiskFile::GetFileSize(file))
      continue;

    if (std::find(files.begin(), files.end(), file) == files.end())
      files.push_back(file);
  }

  if (files.empty())
  {
    ReportError(lasterror, observer, ecInvalidSetting, "There are no files with any data to create a set for");

    return eInvalidCommandLineArguments;
  }

  // A set records each name relative to the basepath, so a file outside it
  // cannot be described
  for (const auto &file : files)
  {
    if (file.compare(0, setbasepath.length(), setbasepath) != 0)
    {
      ReportError(lasterror, observer, ecInvalidSetting, "The file is not inside the basepath", file);

      return eInvalidCommandLineArguments;
    }
  }

  Restart();

  const Result result = impl->Process(memorylimit,
                                      setbasepath,
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
  std::lock_guard<std::mutex> lock(cancelmutex);
  cancelled = true;
  impl->Cancel();
}

void Par2Creator::ClearCancel(void)
{
  std::lock_guard<std::mutex> lock(cancelmutex);
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
  Par2SetCreator creator(sout, serr, noiselevel, backends);
  Result result = creator.Process(
				  MemoryLimit(memorylimit),
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
				   MemoryLimit(memorylimit),
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
  Result result = repairer.Process(MemoryLimit(memorylimit),
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

      if (0 == blocksize || 0 == largestfilesize)
      {
        serr << "The source files are empty." << std::endl;
        return false;
      }

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
