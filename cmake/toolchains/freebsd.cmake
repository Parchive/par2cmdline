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

# Builds for FreeBSD with clang and lld against a FreeBSD sysroot, which
# PAR2_FREEBSD_SYSROOT names, for the target PAR2_FREEBSD_TARGET:
#
#   cmake -B build-freebsd -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/freebsd.cmake \
#     -DPAR2_FREEBSD_SYSROOT=/path/to/sysroot \
#     -DPAR2_FREEBSD_TARGET=x86_64-unknown-freebsd15
#
# The sysroot needs usr/include, usr/lib and lib from a FreeBSD base.txz.

if(NOT PAR2_FREEBSD_SYSROOT OR NOT PAR2_FREEBSD_TARGET)
  message(FATAL_ERROR
    "Set PAR2_FREEBSD_SYSROOT and PAR2_FREEBSD_TARGET to cross-compile for FreeBSD")
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES PAR2_FREEBSD_SYSROOT PAR2_FREEBSD_TARGET)

set(CMAKE_SYSTEM_NAME FreeBSD)
string(REGEX MATCH "^[^-]+" CMAKE_SYSTEM_PROCESSOR "${PAR2_FREEBSD_TARGET}")
set(CMAKE_SYSROOT "${PAR2_FREEBSD_SYSROOT}")

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER_TARGET "${PAR2_FREEBSD_TARGET}")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")

if(CMAKE_VERSION VERSION_LESS 3.29)
  set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
else()
  set(CMAKE_LINKER_TYPE LLD)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
