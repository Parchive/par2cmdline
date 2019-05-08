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

#ifndef __LIBPAR2_H__
#define __LIBPAR2_H__

#ifdef _MSC_VER

// _MSC_VER is for Microsoft Visual C++ Compiler.
//
// The ifdef is not for _WIN32, because MinGW (Mingw32)
// is a port of gcc's compiler that targets Windows
// and we don't want this code run for MinGW.

typedef unsigned char    u8;
typedef signed char      i8;
typedef unsigned short   u16;
typedef signed short     i16;
typedef unsigned int     u32;
typedef signed int       i32;
typedef unsigned __int64 u64;
typedef signed __int64   i64;

#else // _MSC_VER
#ifdef HAVE_CONFIG_H

#include <config.h>

#if HAVE_INTTYPES_H
#  include <inttypes.h>
#endif

#if HAVE_STDINT_H
#  include <stdint.h>
typedef uint8_t            u8;
typedef int8_t             i8;
typedef uint16_t           u16;
typedef int16_t            i16;
typedef uint32_t           u32;
typedef int32_t            i32;
typedef uint64_t           u64;
typedef int64_t            i64;
#else
typedef unsigned char      u8;
typedef signed char        i8;
typedef unsigned short     u16;
typedef signed short       i16;
typedef unsigned int       u32;
typedef signed int         i32;
typedef unsigned long long u64;
typedef signed long long   i64;
#endif

#else // HAVE_CONFIG_H

typedef   unsigned char        u8;
typedef   unsigned short       u16;
typedef   unsigned int         u32;
typedef   unsigned long long   u64;

#endif
#endif


#include <ostream>
#include <vector>
#include <string>


typedef enum
{
  scUnknown = 0,
  scVariable,      // Each PAR2 file will have 2x as many blocks as previous
  scLimited,       // Limit PAR2 file size
  scUniform        // All PAR2 files the same size
} Scheme;


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


  eFileIOError                 = 6,  // An error occurred when accessing files
  eLogicError                  = 7,  // In internal error occurred
  eMemoryError                 = 8,  // Out of memory

} Result;


Result par2create(std::ostream &sout,
			  std::ostream &serr,
			  const NoiseLevel noiselevel,
			  const size_t memorylimit,
			  const std::string &basepath,
#ifdef _OPENMP
			  const u32 nthreads,
			  const u32 filethreads,
#endif
			  const std::string &parfilename,
			  const std::vector<std::string> &extrafiles,
			  const u64 blocksize,
			  const u32 firstblock,
			  const Scheme recoveryfilescheme,
			  const u32 recoveryfilecount,
			  const u32 recoveryblockcount
			  );


Result par2repair(std::ostream &sout,
		  std::ostream &serr,
		  const NoiseLevel noiselevel,
		  const size_t memorylimit,
		  const std::string &basepath,
#ifdef _OPENMP
		  const u32 nthreads,
		  const u32 filethreads,
#endif
		  const std::string &parfilename,
		  const std::vector<std::string> &extrafiles,
		  const bool dorepair,   // derived from operation
		  const bool purgefiles,
		  const bool skipdata,
		  const u64 skipleaway
		  );


Result par1repair(std::ostream &sout,
		  std::ostream &serr,
		  const NoiseLevel noiselevel,
		  const size_t memorylimit,
		  // basepath is not used by Par1
#ifdef _OPENMP
		  const u32 nthreads,
		  // filethreads is not used by Par1
#endif
		  const std::string &parfilename,
		  const std::vector<std::string> &extrafiles,
		  const bool dorepair,   // derived from operation
		  const bool purgefiles
		  // skipdata is not used by Par1
		  // skipleaway is not used by Par1
		  );


bool ComputeRecoveryFileCount(std::ostream &sout,
			      std::ostream &serr,
			      u32 *recoveryfilecount,
			      Scheme recoveryfilescheme,
			      u32 recoveryblockcount,
			      u64 largestfilesize,
			      u64 blocksize);

#endif // __LIBPAR2_H__
