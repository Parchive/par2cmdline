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
      return IsCancelled() ? eCancelled : eLogicError;

    if (packetsloaded == before && !DiskFile::FileExists(parfilename))
      return eFileIOError;

    prepared = PreparePackets();

    if (setchanged)
      *setchanged = (sourceblockcount != blocksbefore) || (DescribedFileCount() != filesbefore)
                    || (VerifiableFileCount() != verifiablebefore);

    return prepared;
  }

  // What the packets read so far amount to
  Result Prepared(void) const
  {
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

  Result Scan(const std::string &filename, const size_t memorylimit,
              const u32 _nthreads, const u32 _filethreads)
  {
    if (prepared != eSuccess)
      return prepared;

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
    if (prepared != eSuccess)
      return prepared;

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
    if (prepared != eSuccess)
      return prepared;

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

Par2Verifier::Par2Verifier(std::ostream &sout, std::ostream &serr, NoiseLevel noiselevel,
                           const std::string &_basepath, Backends _backends)
: sout(sout)
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
, cancelled(false)
, restarting(false)
, basepath(NormaliseBasePath(_basepath))
, impl(new Impl(sout, serr, noiselevel, basepath, backends))
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

  // Naming the same file again reads nothing more, and says what the packets
  // read so far amount to
  if (std::find(par2files.begin(), par2files.end(), parfilename) != par2files.end())
    return impl->Prepared();

  // Take it from the first PAR2 file named, before any packets are read
  const bool derived = basepath.empty();
  if (derived)
  {
    basepath = NormaliseBasePath(BasePathFor(parfilename));
    impl->SetBasePath(basepath);
  }

  bool setchanged = false;
  const Result result = impl->Add(parfilename, &setchanged);

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
  // and a repair can use it. A set of a different shape does not, even when the
  // scan was cancelled. Files scanned before the set was known are replayed by
  // the same restart.
  if (setchanged && (scanned || !scannedfiles.empty()))
    Restart();

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

std::vector<std::string> Par2Verifier::GetBackupFiles(void) const
{
  std::vector<std::string> files;
  impl->GetBackupFiles(&files);
  return files;
}

std::vector<std::pair<std::string, std::string> > Par2Verifier::GetRenamedFiles(void) const
{
  std::vector<std::pair<std::string, std::string> > files;
  impl->GetRenamedFiles(&files);
  return files;
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

std::map<std::string, std::vector<bool> > Par2Verifier::GetKnownBlocks(void) const
{
  return knownblocks;
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

  if (result != eInsufficientCriticalData)
    scanned = true;

  if (result == eCancelled)
    return result;

  // Remembered even when the set is not known yet, so that adding the PAR2 file
  // which describes it replays the scan rather than losing it. Scanning one
  // again replaces what the last scan of it found, so it is remembered once.
  scannedfiles.insert(DiskFile::GetCanonicalPathname(filename));

  if (result == eSuccess || result == eRepairPossible || result == eRepairNotPossible)
    verified = true;

  return result;
}

Result Par2Verifier::Repair(const bool verifyafter)
{
  if (!verified)
    return eLogicError;

  if (repaired)
    return eLogicError;

  if (!impl->CanRepair())
    return eRepairNotPossible;

  // A repair forgets every block vouched for
  for (const auto &kb : knownblocks)
    impl->SetKnownBlocks(kb.first, std::vector<bool>());
  knownblocks.clear();

  repaired = true;

  return impl->Rebuild(memorylimit, nthreads, filethreads, verifyafter);
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
