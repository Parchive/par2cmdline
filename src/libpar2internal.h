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

#ifndef __PARCMDLINE_H__
#define __PARCMDLINE_H__

#ifdef _WIN32
// Windows includes
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// System includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <fcntl.h>
#include <assert.h>

#define snprintf _snprintf_s
#define unlink   _unlink

#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN    4321
#define __PDP_ENDIAN    3412

#define __BYTE_ORDER __LITTLE_ENDIAN

#ifndef _SIZE_T_DEFINED
#  ifdef _WIN64
typedef unsigned __int64 size_t;
#  else
typedef unsigned int     size_t;
#  endif
#  define _SIZE_T_DEFINED
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#else // _WIN32
#ifdef HAVE_CONFIG_H

#include <config.h>

#ifdef HAVE_STDLIB_H
#  include <stdlib.h>
#endif

#ifdef HAVE_STDIO_H
#  include <stdio.h>
#endif

#include <fcntl.h>

#if HAVE_DIRENT_H
#  include <dirent.h>
#  define NAMELEN(dirent) strlen((dirent)->d_name)
#else
#  define dirent direct
#  define NAMELEN(dirent) (dirent)->d_namelen
#  if HAVE_SYS_NDIR_H
#    include <sys/ndir.h>
#  endif
#  if HAVE_SYS_DIR_H
#    include <sys/dir.h>
#  endif
#  if HAVE_NDIR_H
#    include <ndir.h>
#  endif
#endif

#if STDC_HEADERS
#  include <string.h>
#else
#  if !HAVE_MEMCPY
#    define memcpy(d, s, n) bcopy((s), (d), (n))
#  endif
#endif

#if HAVE_MEMORY_H
#  include <memory.h>
#endif



#if HAVE_SYS_STAT_H
#  include <sys/stat.h>
#endif

#if HAVE_LIMITS_H
#  include <limits.h>
#endif

#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

#if HAVE_UNISTD_H
#  include <unistd.h>
#endif

#include <errno.h>

#ifdef _WIN32
// _WIN32: Redefine _MAX_PATH to support Windows long paths (\\?\ prefix)
// Windows normally defines _MAX_PATH as 260, but with long path support
// enabled and the \\?\ prefix, paths can be up to 32767 characters
#   define _MAX_PATH 32767
#else
#   define _MAX_PATH 4095
#endif


#if HAVE_ENDIAN_H
#  include <endian.h>
#  ifndef __LITTLE_ENDIAN
#    ifdef _LITTLE_ENDIAN
#      define __LITTLE_ENDIAN _LITTLE_ENDIAN
#      define __BIG_ENDIAN _BIG_ENDIAN
#      define __PDP_ENDIAN _PDP_ENDIAN
#      define __BYTE_ORDER _BYTE_ORDER
#    else
#      error <endian.h> does not define __LITTLE_ENDIAN etc.
#    endif
#  endif
#else
#  define __LITTLE_ENDIAN 1234
#  define __BIG_ENDIAN    4321
#  define __PDP_ENDIAN    3412
#  if WORDS_BIGENDIAN
#    define __BYTE_ORDER __BIG_ENDIAN
#  else
#    define __BYTE_ORDER __LITTLE_ENDIAN
#  endif
#endif

#else // HAVE_CONFIG_H

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <assert.h>

#include <errno.h>

#define _MAX_PATH 4095

#endif
#endif

// Input blocks held in flight, so that a backend still working on one block
// does not stop the next being read.
#define NUM_TRANSFER_BUFFERS 2

#define MAX_CHUNK_SIZE 32*1048576 // too large chunks are likely detrimental to performance; set to 0 to disable
#define SCAN_BATCH_PER_THREAD 2 // blocks in a batch for each thread checking it, so one which finishes early has more to take

// How far either side of where a block should be that data skipping searches
// when the caller sets no distance of its own
#define DEFAULT_SKIP_LEAWAY 64

#define LONGMULTIPLY

// STL includes
#include <list>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <limits>

#include <ctype.h>
#include <iomanip>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <cassert>

// Holds a lock for the duration of one output statement, so that lines written
// from several threads do not interleave.
class LockedStream
{
public:
  explicit LockedStream(std::ostream &stream)
    : stream(stream)
    , lock(Mutex())
  {
  }

  template<typename T>
  LockedStream& operator<<(const T &value)
  {
    stream << value;
    return *this;
  }

  LockedStream& operator<<(std::ostream& (*manipulator)(std::ostream&))
  {
    stream << manipulator;
    return *this;
  }

private:
  static std::mutex& Mutex(void)
  {
    static std::mutex mutex;
    return mutex;
  }

  std::ostream &stream;
  std::lock_guard<std::mutex> lock;
};

#ifdef offsetof
#undef offsetof
#endif
#define offsetof(TYPE, MEMBER) ((size_t) ((char*)(&((TYPE *)1)->MEMBER) - (char*)1))

// par2cmdline includes
#include <par2/libpar2.h>

// Case-insensitive string comparison
#ifdef _WIN32
#  define stricmp  _stricmp
#else
#  include <string.h>
#  define stricmp strcasecmp
#endif

// Path separators
#ifdef _WIN32
#  define PATHSEP "\\"
#  define ALTPATHSEP "/"
#else
#  define PATHSEP "/"
#  define ALTPATHSEP "\\"
#endif

// Default number of file threads
#define _FILE_THREADS 2

namespace par2
{

// The physical memory of the machine in bytes, or 0 if it cannot be found
u64 GetTotalPhysicalMemory(void);

// What the work may use when the caller sets no limit of its own: an eighth of
// the physical memory, and no less than 256MB on a machine with more, or 256MB
// when the memory cannot be found
size_t DefaultMemoryLimit(void);

// What the work may use: the caller's limit, or the default when it set none,
// and never less than the 1MB the command line allows
size_t MemoryLimit(const size_t requested);

// The directory a PAR2 file is in, which is where the tool looks with no -B
std::string BasePathFor(const std::string &parfilename);

// A path with a separator appended unless it has one already. Empty is left
// alone.
std::string WithSeparator(const std::string &path);

// The name of a set, without the ".par2" its index file ends in
std::string SetNameFor(const std::string &parfilename);

// How much logging/status information to write
// to output or error stream
typedef enum
{
  nlUnknown = 0,
  nlSilent,       // Absolutely no output (other than errors)
  nlQuiet,        // Bare minimum of output
  nlNormal,       // Normal level of output
  nlNoisy,        // Lots of output
  nlDebug         // Extra debugging information
} NoiseLevel;

// The tool's create, verify and repair, each in one call, writing what the tool
// reports to sout and serr
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
                  const Backends &backends = Backends()
                  );

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
                  const bool fullhash = false,
                  const Backends &backends = Backends()
                  );

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
                  );

} // namespace par2


#include "letype.h"
#include "errorlog.h"
#include "foreach_parallel.h"
#include "bufferpool.h"
#include "taskpool.h"
#include "progressmeter.h"

#include "galois.h"
#include "crc.h"
#include "md5.h"
#include "par2fileformat.h"
#include "reedsolomon.h"
#include "reference_processor.h"

#include "diskfile.h"
#include "datablock.h"

#include "criticalpacket.h"
#include "par2creatorsourcefile.h"

#include "mainpacket.h"
#include "creatorpacket.h"
#include "descriptionpacket.h"
#include "verificationpacket.h"
#include "recoverypacket.h"

#include "par2repairersourcefile.h"

#include "filechecksummer.h"
#include "reference_hasher.h"
#include "verificationhashtable.h"

#include "par2creator.h"
#include "par2repairer.h"

#include "par1fileformat.h"
#include "par1repairersourcefile.h"
#include "par1repairer.h"

#ifdef _WIN32
#include "utf8.h"
#endif

// Heap checking
#ifdef _MSC_VER
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, THIS_FILE, __LINE__)
#endif

#endif // __PARCMDLINE_H__
