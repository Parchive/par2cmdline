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

size_t MemoryLimit(const size_t requested)
{
  return std::max<size_t>((requested != 0) ? requested : DefaultMemoryLimit(), 1048576);
}

// Par2Repairer carries out the work; deriving from it reaches the individual
// steps without making them part of its public interface.
class Par2Verifier::Impl : public Par2Repairer
{
public:
  Impl(const std::string &_basepath, const Backends &backends)
    : Par2Repairer(backends)
    , prepared(eInsufficientCriticalData)
    , preparefailure()
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

    // Read it again even if it has been seen before. What a cancel left read is
    // prepared all the same, so that no packet stays unchecked.
    bool opened = true;
    const bool loaded = LoadPackets(parfilename, none, true, &opened);
    const bool stopped = IsCancelled();

    if (!loaded && !stopped)
    {
      errorlog.Record(ecInternalError, "Could not load the PAR2 packets", parfilename);
      return eLogicError;
    }

    if (!stopped && packetsloaded == before && !DiskFile::FileExists(parfilename))
    {
      errorlog.Record(ecPar2FileMissing, "There is no such PAR2 file", parfilename);
      return eFileIOError;
    }

    // Nor is one which is there but could not be opened
    if (!stopped && packetsloaded == before && !opened)
      return eFileIOError;

    prepared = PreparePackets();
    preparefailure = Par2Error();
    errorlog.First(&preparefailure);

    if (setchanged)
      *setchanged = (sourceblockcount != blocksbefore) || (DescribedFileCount() != filesbefore)
                    || (VerifiableFileCount() != verifiablebefore);

    return stopped ? eCancelled : prepared;
  }

  // What the packets read so far amount to, recorded as Add records it
  Result Prepared(void)
  {
    ClearLastError();

    if (prepared != eSuccess)
      errorlog.Record(preparefailure.code, preparefailure.message, preparefailure.filename);

    return prepared;
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

  Result Scan(const std::string &filename, const size_t memorylimit,
              const u32 _nthreads, const u32 _filethreads)
  {
    ClearLastError();

    if (0 != mainpacket && prepared != eSuccess)
    {
      errorlog.Record(preparefailure.code, preparefailure.message, preparefailure.filename);
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
               const u32 _filethreads,
               const bool renameonly)
  {
    ClearLastError();

    if (0 == mainpacket)
    {
      errorlog.Record(ecMainPacketMissing, "The PAR2 files do not describe a set");
      return eInsufficientCriticalData;
    }

    if (prepared != eSuccess)
    {
      errorlog.Record(preparefailure.code, preparefailure.message, preparefailure.filename);
      return prepared;
    }

    ApplyThreadCounts(_nthreads, _filethreads);
    ApplyMemoryLimit(memorylimit);

    std::vector<std::string> extrafiles = _extrafiles;

    return VerifyFiles(basepath, extrafiles, renameonly);
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
      errorlog.Record(preparefailure.code, preparefailure.message, preparefailure.filename);
      return prepared;
    }

    ApplyThreadCounts(_nthreads, _filethreads);

    return RepairFiles(memorylimit, basepath, verifyafter);
  }

private:
  Result prepared;                          // What the last PreparePackets returned
  Par2Error preparefailure;                 // and why, when it failed
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

// The name of a set, without the ".par2" its index file ends in
std::string SetNameFor(const std::string &parfilename)
{
  if (parfilename.length() > 5 &&
      0 == stricmp(parfilename.substr(parfilename.length()-5, 5).c_str(), ".par2"))
    return parfilename.substr(0, parfilename.length()-5);

  return parfilename;
}

// The separator goes on first, so "." names the directory itself
std::string NormaliseBasePath(const std::string &path)
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

// What a Par2Verifier keeps of its own, apart from the engine
struct Par2Verifier::State
{
  State(const std::string &_basepath, Backends _backends)
  : backends(std::move(_backends))
  , observer(0)
  , memorylimit(MemoryLimit(0))
  , nthreads(0)
  , filethreads(0)
  , skipdata(false)
  , skipleaway(DEFAULT_SKIP_LEAWAY)
  , fullhash(false)
  , renameonly(false)
  , verbosity(vbNone)
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
  {
  }

  Backends backends;
  Par2Observer *observer;
  size_t memorylimit;
  u32 nthreads;
  u32 filethreads;
  bool skipdata;
  u64 skipleaway;
  bool fullhash;
  bool renameonly;
  Verbosity verbosity;
  std::vector<std::string> par2files;
  std::set<std::string> scannedfiles;
  std::map<std::string, std::vector<bool> > knownblocks;
  bool verified;
  bool scanned;
  bool repaired;
  bool readback;
  std::mutex workmutex;
  std::mutex cancelmutex;
  bool cancelled;
  bool restarting;
  std::string basepath;
  Par2Error lasterror;
};

// A repair, or a set which changes shape after files were scanned, leaves a
// Par2Repairer with results which no longer hold, so the work starts again from
// a new one. The PAR2 files that were added are read again.
void Par2Verifier::Restart(void)
{
  // A cancel which arrives while the engine is replaced and the PAR2 files
  // are read again is kept for the new one rather than given to either
  {
    std::lock_guard<std::mutex> lock(state->cancelmutex);
    state->restarting = true;
    impl = std::make_unique<Impl>(state->basepath, state->backends);
  }

  // The replay repeats work the observer has already been told about, so it is
  // told none of it. The observer is attached once the handle is back where it
  // was.
  impl->SetDataSkipping(state->skipdata, state->skipleaway);
  impl->SetFullHash(state->fullhash);
  impl->SetVerbosity(state->verbosity);

  for (std::map<std::string, std::vector<bool> >::const_iterator kb = state->knownblocks.begin();
       kb != state->knownblocks.end();
       ++kb)
  {
    impl->SetKnownBlocks(kb->first, kb->second);
  }

  for (const auto &par2file : state->par2files)
  {
    impl->Add(par2file, 0);
  }

  state->verified = false;
  state->scanned = false;
  state->repaired = false;

  // A cancel reaches the rescan, which stops where it got to
  {
    std::lock_guard<std::mutex> lock(state->cancelmutex);
    state->restarting = false;

    if (state->cancelled)
      impl->Cancel();
  }

  const std::set<std::string> rescan = state->scannedfiles;
  state->scannedfiles.clear();

  bool cutshort = false;

  for (const auto &f : rescan)
  {
    if (impl->IsCancelled() || eCancelled == DoVerifyFile(f))
    {
      cutshort = true;
      break;
    }
  }

  // What the rescan did not reach is kept for the next one
  if (cutshort)
  {
    state->scannedfiles = rescan;
    state->verified = false;
  }

  impl->SetObserver(state->observer);
}

Par2Verifier::Par2Verifier(const std::string &_basepath, Backends _backends)
: state(new State(_basepath, std::move(_backends)))
, impl(new Impl(state->basepath, state->backends))
{
}

Par2Verifier::~Par2Verifier() = default;

// The engine records the error, but Restart throws the engine away, so the
// handle keeps its own copy of what the call it is returning from recorded. A
// call which succeeded or was cancelled carries none.
void Par2Verifier::TakeLastError(const Result result)
{
  state->lasterror = Par2Error();
  if (result != eSuccess && result != eCancelled)
    impl->GetLastError(&state->lasterror);
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

// Report the exception being handled as the failure of the call it stopped
static Result Thrown(Par2Error &lasterror, Par2Observer *observer)
{
  try
  {
    throw;
  }
  catch (const std::bad_alloc &)
  {
    ReportError(lasterror, observer, ecOutOfMemory, "Memory ran out");
    return eMemoryError;
  }
  catch (...)
  {
    ReportError(lasterror, observer, ecInternalError, "The work stopped on an exception");
    return eLogicError;
  }
}

// What the handle itself has to report, rather than the work it delegates
void Par2Verifier::RecordLastError(const ErrorCode code, const std::string &message)
{
  ReportError(state->lasterror, state->observer, code, message);
}

bool Par2Verifier::GetLastError(Par2Error *error) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  if (0 == error || ecNone == state->lasterror.code)
    return false;

  *error = state->lasterror;
  return true;
}

void Par2Verifier::SetObserver(Par2Observer *_observer)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->observer = _observer;
  impl->SetObserver(_observer);
}

void Par2Verifier::SetMemoryLimit(const size_t _memorylimit)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->memorylimit = MemoryLimit(_memorylimit);
}

void Par2Verifier::SetDataSkipping(const bool enabled, const u64 leaway)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->skipdata = enabled;
  state->skipleaway = (leaway != 0) ? leaway : DEFAULT_SKIP_LEAWAY;

  impl->SetDataSkipping(state->skipdata, state->skipleaway);
}

void Par2Verifier::SetFullHash(const bool enabled)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->fullhash = enabled;

  impl->SetFullHash(state->fullhash);
}

void Par2Verifier::SetRenameOnly(const bool enabled)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->renameonly = enabled;
}

void Par2Verifier::SetVerbosity(const Verbosity verbosity)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->verbosity = verbosity;
  impl->SetVerbosity(verbosity);
}

void Par2Verifier::SetThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->nthreads = _nthreads;
  state->filethreads = _filethreads;
}

Result Par2Verifier::AddPar2File(const std::string &parfilename)
{
  std::lock_guard<std::mutex> lock(state->workmutex);
  return DoAddPar2File(parfilename);
}

Result Par2Verifier::DoAddPar2File(const std::string &_parfilename)
try
{
  const std::string parfilename = DiskFile::GetCanonicalPathname(_parfilename);

  // Naming the same file again reads nothing more, and says what the packets
  // read so far amount to
  if (std::find(state->par2files.begin(), state->par2files.end(), parfilename) != state->par2files.end())
  {
    const Result result = impl->Prepared();
    TakeLastError(result);
    return result;
  }

  // Take it from the first PAR2 file named, before any packets are read
  const bool derived = state->basepath.empty();
  if (derived)
  {
    state->basepath = NormaliseBasePath(BasePathFor(parfilename));
    impl->SetBasePath(state->basepath);
  }

  bool setchanged = false;
  const Result result = impl->Add(parfilename, &setchanged);

  // Restart replays the scans through VerifyFile, which would otherwise leave
  // the handle holding what the replay found rather than what Add recorded
  TakeLastError(result);
  const Par2Error added = state->lasterror;

  // Remembered even without the critical packets, so that a later restart
  // replays it alongside the file that completes the set. A cancelled load is
  // not remembered, so that naming it again after ClearCancel reads the rest.
  if (result != eFileIOError && result != eCancelled)
  {
    state->par2files.push_back(parfilename);
  }
  else if (derived && result == eFileIOError)
  {
    // Taken again from the next one
    state->basepath.clear();
    impl->SetBasePath(state->basepath);
  }

  // Extra recovery data leaves what the scan found still true, so it is kept
  // and a repair can use it. A set of a different shape does not, even when the
  // scan was cancelled. Files scanned before the set was known are replayed by
  // the same restart.
  if (setchanged && (state->scanned || !state->scannedfiles.empty()))
    Restart();

  state->lasterror = added;

  return result;
}
catch (...)
{
  return Thrown(state->lasterror, state->observer);
}

bool Par2Verifier::GetSetInfo(Par2SetInfo *info) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  return impl->GetSetInfo(info);
}

bool Par2Verifier::GetFileInfo(std::vector<Par2FileInfo> *files) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  return impl->GetFileInfo(files);
}

bool Par2Verifier::GetBlockChecksums(const std::string &filename,
                                    std::vector<u32> *crcs) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  return impl->GetBlockChecksums(filename, crcs);
}

// Guarded by verified, unlike GetBlockChecksums: before anything has been
// scanned every block would read as not found, which is not the same as
// nothing having been looked at. After a repair which rebuilt files without
// reading them back, what was found describes the files it replaced.
bool Par2Verifier::GetFoundBlocks(const std::string &filename,
                                  std::vector<bool> *blocks) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  if (!state->verified || (state->repaired && !state->readback))
    return false;

  return impl->GetFoundBlocks(filename, blocks);
}

std::vector<std::string> Par2Verifier::GetBackupFiles(void) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  std::vector<std::string> files;
  impl->GetBackupFiles(&files);
  return files;
}

std::vector<std::pair<std::string, std::string> > Par2Verifier::GetRenamedFiles(void) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  std::vector<std::pair<std::string, std::string> > files;
  impl->GetRenamedFiles(&files);
  return files;
}

bool Par2Verifier::GetVerifyResult(Par2VerifyResult *result) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  if (!state->verified)
    return false;

  return impl->GetVerifyResult(result);
}

bool Par2Verifier::SetKnownBlocks(const std::string &filename,
                                 const std::vector<bool> &blocks)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  if (!impl->SetKnownBlocks(filename, blocks))
    return false;

  if (blocks.empty())
    state->knownblocks.erase(filename);
  else
    state->knownblocks[filename] = blocks;

  return true;
}

std::map<std::string, std::vector<bool> > Par2Verifier::GetKnownBlocks(void) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  return state->knownblocks;
}

Result Par2Verifier::Verify(const std::vector<std::string> &extrafiles)
{
  std::lock_guard<std::mutex> lock(state->workmutex);
  return DoVerify(extrafiles);
}

Result Par2Verifier::DoVerify(const std::vector<std::string> &extrafiles)
try
{
  // A full pass covers everything the individual scans did, so they are dropped
  // rather than replayed into it. Whatever was scanned before, by a pass that
  // was cancelled too, is started afresh.
  state->scannedfiles.clear();

  if (state->repaired)
    Restart();
  else
    impl->DiscardScans();

  state->verified = false;

  const Result result = impl->Check(extrafiles, state->memorylimit, state->nthreads, state->filethreads,
                                    state->renameonly);
  TakeLastError(result);

  if (result != eInsufficientCriticalData)
    state->scanned = true;

  if (result == eSuccess || result == eRepairPossible || result == eRepairNotPossible)
    state->verified = true;

  return result;
}
catch (...)
{
  return Thrown(state->lasterror, state->observer);
}

Result Par2Verifier::VerifyFile(const std::string &filename)
{
  std::lock_guard<std::mutex> lock(state->workmutex);
  return DoVerifyFile(filename);
}

Result Par2Verifier::DoVerifyFile(const std::string &filename)
try
{
  // After a repair a new engine starts from nothing, and each file is scanned
  // again as it is fed in
  if (state->repaired)
  {
    state->scannedfiles.clear();
    Restart();
  }

  const Result result = impl->Scan(filename, state->memorylimit, state->nthreads, state->filethreads);
  TakeLastError(result);

  if (result != eInsufficientCriticalData)
    state->scanned = true;

  if (result == eCancelled)
    return result;

  // Remembered even when the set is not known yet, so that adding the PAR2 file
  // which describes it replays the scan rather than losing it. Scanning one
  // again replaces what the last scan of it found, so it is remembered once.
  state->scannedfiles.insert(DiskFile::GetCanonicalPathname(filename));

  if (result == eSuccess || result == eRepairPossible || result == eRepairNotPossible)
    state->verified = true;

  return result;
}
catch (...)
{
  return Thrown(state->lasterror, state->observer);
}

Result Par2Verifier::Repair(const bool verifyafter)
{
  std::lock_guard<std::mutex> lock(state->workmutex);
  return DoRepair(verifyafter);
}

Result Par2Verifier::DoRepair(const bool verifyafter)
try
{
  if (!state->verified)
  {
    RecordLastError(ecNotVerified, "Nothing has been verified yet");
    return eLogicError;
  }

  if (state->repaired)
  {
    RecordLastError(ecNotVerified, "Nothing has been verified since the last repair");
    return eLogicError;
  }

  if (!impl->CanRepair())
  {
    state->lasterror = Par2Error();
    return eRepairNotPossible;
  }

  Par2VerifyResult before;
  impl->GetVerifyResult(&before);

  // A repair forgets every block vouched for
  for (const auto &kb : state->knownblocks)
    impl->SetKnownBlocks(kb.first, std::vector<bool>());
  state->knownblocks.clear();

  state->repaired = true;

  const Result result = impl->Rebuild(state->memorylimit, state->nthreads, state->filethreads, verifyafter);

  // A repair which only renames files writes nothing to read back
  state->readback = result == eSuccess
                    && (verifyafter || before.damagedfilecount + before.missingfilecount == 0);
  TakeLastError(result);

  return result;
}
catch (...)
{
  return Thrown(state->lasterror, state->observer);
}

void Par2Verifier::Cancel(void)
{
  std::lock_guard<std::mutex> lock(state->cancelmutex);
  state->cancelled = true;

  if (!state->restarting)
    impl->Cancel();
}

void Par2Verifier::ClearCancel(void)
{
  std::lock_guard<std::mutex> lock(state->cancelmutex);
  state->cancelled = false;

  if (!state->restarting)
    impl->ClearCancel();
}



// Par2SetCreator carries out the work
class Par2Creator::Impl : public Par2SetCreator
{
public:
  explicit Impl(const Backends &backends)
    : Par2SetCreator(backends)
  {
  }
};

// What a Par2Creator keeps of its own, apart from the engine
struct Par2Creator::State
{
  State(const std::string &_basepath, Backends _backends)
  : backends(std::move(_backends))
  , observer(0)
  , sourcefiles()
  , blocksize(0)
  , sourceblockcount(0)
  , recoveryblockcount(0)
  , redundancy(0)
  , recoveryfilescheme(scVariable)
  , recoveryfilecount(0)
  , firstrecoveryblock(0)
  , memorylimit(MemoryLimit(0))
  , nthreads(0)
  , filethreads(0)
  , verbosity(vbNone)
  , cancelled(false)
  , basepath(NormaliseBasePath(_basepath))
  , lasterror()
  {
  }

  Backends backends;
  Par2Observer *observer;
  std::vector<std::string> sourcefiles;
  u64 blocksize;
  u32 sourceblockcount;
  u32 recoveryblockcount;
  u32 redundancy;
  Scheme recoveryfilescheme;
  u32 recoveryfilecount;
  u32 firstrecoveryblock;
  size_t memorylimit;
  u32 nthreads;
  u32 filethreads;
  Verbosity verbosity;
  std::mutex workmutex;
  std::mutex cancelmutex;
  bool cancelled;
  std::string basepath;
  Par2Error lasterror;
};

// A create leaves the engine holding the packets of the set it wrote, so a
// second one has to start from a new engine. Nothing is replayed into it:
// every setting lives on the handle and is passed into the run.
void Par2Creator::Restart(void)
{
  std::lock_guard<std::mutex> lock(state->cancelmutex);

  impl = std::make_unique<Impl>(state->backends);
  impl->SetObserver(state->observer);
  impl->SetVerbosity(state->verbosity);

  if (state->cancelled)
    impl->Cancel();
}

// The engine records the error, but Restart throws the engine away, so the
// handle keeps its own copy of what the call it is returning from recorded. A
// call which succeeded or was cancelled carries none.
void Par2Creator::TakeLastError(const Result result)
{
  state->lasterror = Par2Error();
  if (result != eSuccess && result != eCancelled)
    impl->GetLastError(&state->lasterror);
}

Par2Creator::Par2Creator(const std::string &_basepath, Backends _backends)
: state(new State(_basepath, std::move(_backends)))
, impl(new Impl(state->backends))
{
}

Par2Creator::~Par2Creator() = default;

void Par2Creator::SetObserver(Par2Observer *_observer)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->observer = _observer;
  impl->SetObserver(_observer);
}

void Par2Creator::SetSourceFiles(const std::vector<std::string> &filenames)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->sourcefiles = filenames;
}

void Par2Creator::SetBlockSize(const u64 _blocksize)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->blocksize = _blocksize;
  state->sourceblockcount = 0;
}

void Par2Creator::SetSourceBlockCount(const u32 blockcount)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->sourceblockcount = blockcount;
  state->blocksize = 0;
}

void Par2Creator::SetRecoveryBlockCount(const u32 _recoveryblockcount)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->recoveryblockcount = _recoveryblockcount;
  state->redundancy = 0;
}

void Par2Creator::SetRedundancy(const u32 percent)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->redundancy = percent;
  state->recoveryblockcount = 0;
}

void Par2Creator::SetRecoveryFileScheme(const Scheme scheme, const u32 _recoveryfilecount)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->recoveryfilescheme = scheme;
  state->recoveryfilecount = _recoveryfilecount;
}

void Par2Creator::SetFirstRecoveryBlock(const u32 firstblock)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->firstrecoveryblock = firstblock;
}

void Par2Creator::SetMemoryLimit(const size_t _memorylimit)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->memorylimit = MemoryLimit(_memorylimit);
}

void Par2Creator::SetThreadCounts(const u32 _nthreads, const u32 _filethreads)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->nthreads = _nthreads;
  state->filethreads = _filethreads;
}

void Par2Creator::SetVerbosity(const Verbosity verbosity)
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  state->verbosity = verbosity;
  impl->SetVerbosity(verbosity);
}

Result Par2Creator::Create(const std::string &parfilename)
{
  std::lock_guard<std::mutex> lock(state->workmutex);
  return DoCreate(parfilename);
}

Result Par2Creator::DoCreate(const std::string &parfilename)
try
{
  // Taken from the name of each set when none was given, before any file is
  // read
  const std::string setbasepath = state->basepath.empty()
    ? NormaliseBasePath(BasePathFor(parfilename))
    : state->basepath;

  // The volume files are named after the set rather than after its index file,
  // so the name is taken either way round
  const std::string setname = SetNameFor(DiskFile::GetCanonicalPathname(parfilename));

  // Resolved against the working directory, so that the names the set records
  // come out relative to the basepath whatever the caller wrote them as. Empty
  // files and a file named twice are left out.
  std::vector<std::string> files;
  files.reserve(state->sourcefiles.size());
  std::set<std::string> seen;
  for (const auto &sourcefile : state->sourcefiles)
  {
    const std::string file = DiskFile::GetCanonicalPathname(sourcefile);

    if (DiskFile::FileExists(file) && 0 == DiskFile::GetFileSize(file))
      continue;

    if (seen.insert(file).second)
      files.push_back(file);
  }

  if (files.empty())
  {
    ReportError(state->lasterror, state->observer, ecInvalidSetting, "There are no files with any data to create a set for");

    return eInvalidCommandLineArguments;
  }

  // A block count and a redundancy come to a block size and a recovery block
  // count for these files
  u64 setblocksize = state->blocksize;
  u32 setrecoveryblockcount = state->recoveryblockcount;
  if (0 != state->sourceblockcount || 0 != state->redundancy)
  {
    std::vector<u64> filesizes;
    for (const auto &file : files)
      filesizes.push_back(DiskFile::GetFileSize(file));

    std::string error;
    if (0 != state->sourceblockcount
        && !ComputeBlockSizeFromCount(&error, &setblocksize, state->sourceblockcount, filesizes))
    {
      ReportError(state->lasterror, state->observer, ecInvalidSetting, error);

      return eInvalidCommandLineArguments;
    }

    if (0 != state->redundancy && 0 != setblocksize)
    {
      u32 blockcount = 0;
      for (const u64 filesize : filesizes)
        blockcount += (u32)((filesize + setblocksize - 1) / setblocksize);

      setrecoveryblockcount = ComputeRecoveryBlockCountFromRedundancy(blockcount, state->redundancy);
    }
  }

  Restart();

  const Result result = impl->Process(state->memorylimit,
                                      setbasepath,
                                      state->nthreads,
                                      state->filethreads,
                                      setname,
                                      files,
                                      setblocksize,
                                      state->firstrecoveryblock,
                                      state->recoveryfilescheme,
                                      state->recoveryfilecount,
                                      setrecoveryblockcount);
  TakeLastError(result);

  return result;
}
catch (...)
{
  return Thrown(state->lasterror, state->observer);
}

void Par2Creator::Cancel(void)
{
  std::lock_guard<std::mutex> lock(state->cancelmutex);
  state->cancelled = true;
  impl->Cancel();
}

void Par2Creator::ClearCancel(void)
{
  std::lock_guard<std::mutex> lock(state->cancelmutex);
  state->cancelled = false;
  impl->ClearCancel();
}

bool Par2Creator::GetLastError(Par2Error *error) const
{
  std::lock_guard<std::mutex> lock(state->workmutex);

  if (0 == error || ecNone == state->lasterror.code)
    return false;

  *error = state->lasterror;
  return true;
}


// Determine how many recovery files to create.
bool ComputeRecoveryFileCount(std::string *error,
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
      if (error)
        *error = "No recovery file scheme was given";
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
        if (error)
          *error = "There are more recovery files than recovery blocks to put in them";
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
        if (error)
          *error = "The source files are empty";
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

// Work out the block size which divides files of these sizes into blockcount
// blocks, or as near to that as a multiple of 4 allows.
bool ComputeBlockSizeFromCount(std::string *error,
			       u64 *blocksize,
			       u32 blockcount,
			       const std::vector<u64> &filesizes)
{
  if (blockcount < filesizes.size())
  {
    // The block count cannot be less than the number of files.

    if (error)
      *error = "The block count (" + std::to_string(blockcount)
               + ") cannot be smaller than the number of files ("
               + std::to_string(filesizes.size()) + ")";
    return false;
  }
  else if (blockcount == filesizes.size())
  {
    // If the block count is the same as the number of files, then the block
    // size is the size of the largest file (rounded up to a multiple of 4).

    u64 largestfilesize = 0;
    for (std::vector<u64>::const_iterator i=filesizes.begin(); i!=filesizes.end(); i++)
    {
	u64 filesize = *i;
	if (filesize > largestfilesize)
	{
	  largestfilesize = filesize;
	}
    }
    *blocksize = (largestfilesize + 3) & ~3;
  }
  else
  {
    u64 totalsize = 0;
    for (std::vector<u64>::const_iterator i=filesizes.begin(); i!=filesizes.end(); i++)
    {
      totalsize += (*i + 3) / 4;
    }

    if (blockcount > totalsize)
    {
      *blocksize = 4;
    }
    else
    {
      // Absolute lower bound and upper bound on the source block size that will
      // result in the requested source block count.
      u64 lowerBound = totalsize / blockcount;
      u64 upperBound = (totalsize + blockcount - filesizes.size() - 1) / (blockcount - filesizes.size());

      u64 count = 0;
      u64 size;

      do
      {
        size = (lowerBound + upperBound)/2;

        count = 0;
        for (std::vector<u64>::const_iterator i=filesizes.begin(); i!=filesizes.end(); i++)
        {
          count += ((*i+3)/4 + size-1) / size;
        }
        if (count > blockcount)
        {
          lowerBound = size+1;
          if (lowerBound >= upperBound)
          {
            size = lowerBound;
            count = 0;
            for (std::vector<u64>::const_iterator i=filesizes.begin(); i!=filesizes.end(); i++)
            {
              count += ((*i+3)/4 + size-1) / size;
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
        if (error)
          *error = "The block size for this block count would need more than 32768 blocks";
        return false;
      }
      else if (count == 0)
      {
        if (error)
          *error = "The block size for this block count would give no blocks";
        return false;
      }

      *blocksize = size*4;
    }
  }

  return true;
}

// How many recovery blocks redundancy percent of sourceblockcount comes to,
// and at least one.
u32 ComputeRecoveryBlockCountFromRedundancy(u32 sourceblockcount, u32 redundancy)
{
  return std::max<u32>((sourceblockcount * redundancy + 50) / 100, 1);
}

} // namespace par2
