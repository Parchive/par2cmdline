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

#ifndef __PARCMDLINE_H__
#define __PARCMDLINE_H__

// Windows includes
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// STL includes
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>

#include <ctype.h>
#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <iomanip>


// System includes
#ifdef WIN32

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <fcntl.h>
#include <assert.h>

#define snprintf _snprintf
#define stat _stat

#else

#ifdef linux

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

#define _MAX_PATH 255
#define stricmp strcasecmp
#define _stat stat

#else

#include <io.h>
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

#define _MAX_PATH 255

#endif
#endif

#ifdef WIN32

#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN    4321
#define __PDP_ENDIAN    3412

#define __BYTE_ORDER __LITTLE_ENDIAN

#else
#ifdef linux

#include <endian.h>

#else

#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN    4321
#define __PDP_ENDIAN    3412

#define __BYTE_ORDER __LITTLE_ENDIAN

#endif
#endif


using namespace std;

// numeric and other simple types

// 8-bit, 16-bit, and 32-bit unsigned values
// Used in the definition of certain fields
// in the PAR2 file format.
typedef   unsigned char        u8;
typedef   unsigned short       u16;
typedef   unsigned long        u32;

// 64-bit unsigned value.
// Used for all file size and offset values.
#ifdef _MSC_VER
typedef   unsigned __int64     u64;
#else
typedef   unsigned long long   u64;
#endif

// An architecture dependent type used to
// represent the size of an in memory object.
// It is the return type of the "sizeof()" operator
// and a parameter of the "new" operator and 
// "malloc()" function.
#ifdef _MSC_VER
#ifndef _SIZE_T_DEFINED
#ifdef  _WIN64
typedef unsigned __int64    size_t;
#else
typedef unsigned int        size_t;
#endif
#define _SIZE_T_DEFINED
#endif
#else
typedef unsigned int        size_t;
#endif

//#ifndef offsetof
//#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)
//#endif

#ifdef offsetof
#undef offsetof
#endif
#define offsetof(TYPE, MEMBER) ((size_t) ((char*)(&((TYPE *)1)->MEMBER) - (char*)1))

#ifdef WIN32
#define PATHSEP "\\"
#define ALTPATHSEP "/"
#else
#define PATHSEP "/"
#define ALTPATHSEP "\\"
#endif

// Return type of par2cmdline
typedef enum Result
{
  eSuccess                     = 0,

  eRepairPossible              = 1,  // Data files are damaged and there is
                                     // enough recovery data available to
                                     // repair them.

  eRepairNotPossible           = 2,  // Data files are damaged and there is
                                     // insufficient recovery data available
                                     // to be able to repair them.

  eInvalidCommandLineArguments = 3,  // There was something wrong with the
                                     // command line arguments

  eInsufficientCriticalData    = 4,  // The PAR2 files did not contain sufficient
                                     // information about the data files to be able
                                     // to verify them.

  eRepairFailed                = 5,  // Repair completed but the data files
                                     // still appear to be damaged.


  eFileIOError                 = 6,  // An error occured when accessing files
  eLogicError                  = 7,  // In internal error occurred
  eMemoryError                 = 8,  // Out of memory

} Result;

#define LONGMULTIPLY

// par2cmdline includes
#include "letype.h"

#include "galois.h"
#include "reedsolomon.h"
#include "crc.h"
#include "md5.h"
#include "par2fileformat.h"
#include "commandline.h"

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

// Heap checking 
#ifdef _MSC_VER
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, THIS_FILE, __LINE__)
#endif

extern string version;

#endif // __PARCMDLINE_H__
