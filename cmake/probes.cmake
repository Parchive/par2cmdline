##  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
##  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
##
##  Copyright (c) 2026 Michael Nightingale
##
##  par2cmdline is free software; you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation; either version 2 of the License, or
##  (at your option) any later version.
##
##  par2cmdline is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program; if not, write to the Free Software
##  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)
include(CheckSymbolExists)
include(TestBigEndian)

check_include_file_cxx(dirent.h HAVE_DIRENT_H)
check_include_file_cxx(endian.h HAVE_ENDIAN_H)
check_include_file_cxx(limits.h HAVE_LIMITS_H)
check_include_file_cxx(memory.h HAVE_MEMORY_H)
check_include_file_cxx(ndir.h HAVE_NDIR_H)
check_include_file_cxx(stdio.h HAVE_STDIO_H)
check_include_file_cxx(stdlib.h HAVE_STDLIB_H)
check_include_file_cxx(string.h HAVE_STRING_H)
check_include_file_cxx(sys/dir.h HAVE_SYS_DIR_H)
check_include_file_cxx(sys/ndir.h HAVE_SYS_NDIR_H)
check_include_file_cxx(sys/stat.h HAVE_SYS_STAT_H)
check_include_file_cxx(sys/types.h HAVE_SYS_TYPES_H)
check_include_file_cxx(unistd.h HAVE_UNISTD_H)

if(HAVE_STDIO_H AND HAVE_STDLIB_H AND HAVE_STRING_H AND HAVE_LIMITS_H)
  set(STDC_HEADERS 1)
endif()

check_symbol_exists(memcpy string.h HAVE_MEMCPY)
check_symbol_exists(fseeko stdio.h HAVE_FSEEKO)

test_big_endian(WORDS_BIGENDIAN)

# DiskFile::Read is given the position to read from, using ReadFile with an
# OVERLAPPED offset on Windows and pread everywhere else.
if(NOT WIN32)
  check_symbol_exists(pread unistd.h HAVE_PREAD)
  if(NOT HAVE_PREAD)
    message(FATAL_ERROR
      "par2cmdline needs pread, or the Windows API, to read a file at an explicit offset")
  endif()
endif()

set(PAR2_ATOMIC_PROGRAM "
#include <atomic>
#include <cstdint>
int main() { std::atomic<uint64_t> value(0); return (int)value.fetch_add(1); }
")

check_cxx_source_compiles("${PAR2_ATOMIC_PROGRAM}" HAVE_INLINE_64BIT_ATOMICS)
if(NOT HAVE_INLINE_64BIT_ATOMICS)
  set(CMAKE_REQUIRED_LIBRARIES atomic)
  check_cxx_source_compiles("${PAR2_ATOMIC_PROGRAM}" HAVE_LIBATOMIC_64BIT_ATOMICS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(HAVE_LIBATOMIC_64BIT_ATOMICS)
    set(PAR2_ATOMIC_LIBRARY atomic)
  else()
    message(FATAL_ERROR "par2cmdline needs 64-bit atomics")
  endif()
endif()
