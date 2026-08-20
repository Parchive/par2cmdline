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
