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

# What the GNU cross toolchains here share: the compiler is named after
# PAR2_TOOLCHAIN_PREFIX, and libraries and headers are looked for only in its
# sysroot, where Debian and Ubuntu install it.

set(CMAKE_CXX_COMPILER ${PAR2_TOOLCHAIN_PREFIX}-g++)

if(IS_DIRECTORY /usr/${PAR2_TOOLCHAIN_PREFIX})
  set(CMAKE_FIND_ROOT_PATH /usr/${PAR2_TOOLCHAIN_PREFIX})
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# The tests run under qemu-user, for the architecture PAR2_QEMU_ARCH names,
# when it is installed
if(PAR2_QEMU_ARCH)
  find_program(PAR2_QEMU NAMES qemu-${PAR2_QEMU_ARCH} qemu-${PAR2_QEMU_ARCH}-static)
  if(PAR2_QEMU)
    set(CMAKE_CROSSCOMPILING_EMULATOR ${PAR2_QEMU} -L /usr/${PAR2_TOOLCHAIN_PREFIX})
  endif()
endif()
