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
#define sprintf  sprintf_s
#define stricmp  _stricmp
#define unlink   _unlink
#define stat _stat

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
#  if !HAVE_STRCHR
#    define strchr index
#    define strrchr rindex
#  endif
char *strchr(), *strrchr();
#  if !HAVE_MEMCPY
#    define memcpy(d, s, n) bcopy((s), (d), (n))
#    define memove(d, s, n) bcopy((s), (d), (n))
#  endif
#endif

#if HAVE_MEMORY_H
#  include <memory.h>
#endif

#if !HAVE_STRICMP
#  if HAVE_STRCASECMP
#    define stricmp strcasecmp
#  endif
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

#define _MAX_PATH 4095

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
#define stricmp strcasecmp
#define _stat stat

#endif
#endif

#ifdef _WIN32
#define PATHSEP "\\"
#define ALTPATHSEP "/"
#else
#define PATHSEP "/"
#define ALTPATHSEP "\\"
#endif

#define _FILE_THREADS 2

#define LONGMULTIPLY

// STL includes
#include <list>
#include <map>
#include <algorithm>

#include <ctype.h>
#include <iomanip>

#include <cassert>

using namespace std;

#ifdef offsetof
#undef offsetof
#endif
#define offsetof(TYPE, MEMBER) ((size_t) ((char*)(&((TYPE *)1)->MEMBER) - (char*)1))

// par2cmdline includes
#include "libpar2.h"

#include "letype.h"

#include "galois.h"
#include "crc.h"
#include "md5.h"
#include "par2fileformat.h"
#include "reedsolomon.h"

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
#include "verificationhashtable.h"

#include "par2creator.h"
#include "par2repairer.h"

#include "par1fileformat.h"
#include "par1repairersourcefile.h"
#include "par1repairer.h"

// Heap checking
#ifdef _MSC_VER
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, THIS_FILE, __LINE__)
#endif

// OpenMP
#ifdef _OPENMP
# include <omp.h>
#endif


#endif // __PARCMDLINE_H__

