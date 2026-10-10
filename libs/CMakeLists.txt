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

# cxxopts parses the tool's command line: the copy here, or the system's with
# PAR2_SYSTEM_CXXOPTS
add_library(par2_cxxopts INTERFACE)

if(PAR2_SYSTEM_CXXOPTS)
  find_package(cxxopts 3 CONFIG REQUIRED)
  set_target_properties(cxxopts::cxxopts PROPERTIES IMPORTED_GLOBAL TRUE)
  target_link_libraries(par2_cxxopts INTERFACE cxxopts::cxxopts)
else()
  target_include_directories(par2_cxxopts SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/cxxopts")
endif()
