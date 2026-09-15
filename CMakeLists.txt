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

cmake_minimum_required(VERSION 3.16)

project(par2cmdline
  VERSION 1.4.0
  DESCRIPTION "A PAR 2.0 compatible file verification and repair tool"
  HOMEPAGE_URL "https://github.com/Parchive/par2cmdline"
  LANGUAGES CXX
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(PAR2_IS_TOP_LEVEL TRUE)
else()
  set(PAR2_IS_TOP_LEVEL FALSE)
endif()

option(PAR2_BUILD_TOOL "Build the par2 command line tool" ${PAR2_IS_TOP_LEVEL})
option(PAR2_BUILD_TESTS "Build the test suite" ${PAR2_IS_TOP_LEVEL})
option(PAR2_INSTALL "Generate the install rules" ${PAR2_IS_TOP_LEVEL})
option(PAR2_KEEP_ASSERTS "Keep assert checks in optimised builds" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

# Take NDEBUG out of the optimised builds' flags
if(PAR2_KEEP_ASSERTS)
  foreach(config RELEASE RELWITHDEBINFO MINSIZEREL)
    string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
  endforeach()
endif()

if(PAR2_IS_TOP_LEVEL AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(probes)

configure_file(cmake/config.h.in "${PROJECT_BINARY_DIR}/config.h")

find_package(Threads REQUIRED)

set(PAR2_OPTIMISED "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>")

function(par2_add_size_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      "$<${PAR2_OPTIMISED}:/Gy>"
      "$<${PAR2_OPTIMISED}:/Gw>"
    )
  else()
    target_compile_options(${target} PRIVATE
      "$<${PAR2_OPTIMISED}:-ffunction-sections>"
      "$<${PAR2_OPTIMISED}:-fdata-sections>"
    )
  endif()
endfunction()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
foreach(config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER "${config}" config)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}")
endforeach()

set(PAR2_PUBLIC_HEADERS
  include/par2/backends.h
  include/par2/hasher.h
  include/par2/libpar2.h
  include/par2/processor.h
  include/par2/types.h
)

add_library(libpar2 STATIC
  src/crc.cpp src/crc.h
  src/creatorpacket.cpp src/creatorpacket.h
  src/criticalpacket.cpp src/criticalpacket.h
  src/datablock.cpp src/datablock.h
  src/descriptionpacket.cpp src/descriptionpacket.h
  src/diskfile.cpp src/diskfile.h
  src/errorlog.h
  src/filechecksummer.cpp src/filechecksummer.h
  src/galois.cpp src/galois.h
  src/letype.h
  src/mainpacket.cpp src/mainpacket.h
  src/md5.cpp src/md5.h
  src/par1fileformat.cpp src/par1fileformat.h
  src/par1repairer.cpp src/par1repairer.h
  src/par1repairersourcefile.cpp src/par1repairersourcefile.h
  src/par2creator.cpp src/par2creator.h
  src/par2creatorsourcefile.cpp src/par2creatorsourcefile.h
  src/par2fileformat.cpp src/par2fileformat.h
  src/par2repairer.cpp src/par2repairer.h
  src/par2repairersourcefile.cpp src/par2repairersourcefile.h
  src/foreach_parallel.h
  src/bufferpool.h
  src/taskpool.h
  src/reference_hasher.h
  src/reference_processor.h
  src/progressmeter.h
  src/recoverypacket.cpp src/recoverypacket.h
  src/reedsolomon.cpp src/reedsolomon.h
  src/verificationhashtable.cpp src/verificationhashtable.h
  src/verificationpacket.cpp src/verificationpacket.h
  src/libpar2.cpp src/libpar2internal.h
  ${PAR2_PUBLIC_HEADERS}
)

add_library(par2::par2 ALIAS libpar2)

if(WIN32)
  target_sources(libpar2 PRIVATE src/utf8.cpp src/utf8.h)
endif()

set_target_properties(libpar2 PROPERTIES
  OUTPUT_NAME par2
  EXPORT_NAME par2
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(libpar2
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
)

target_compile_definitions(libpar2 PRIVATE HAVE_CONFIG_H)

if(NOT WIN32)
  target_compile_definitions(libpar2 PRIVATE
    _FILE_OFFSET_BITS=64
    _LARGEFILE_SOURCE
    _LARGE_FILES
  )
endif()

if(MSVC)
  target_compile_options(libpar2 PRIVATE /utf-8)
else()
  target_compile_options(libpar2 PRIVATE -Wall)
endif()

par2_add_size_options(libpar2)

target_link_libraries(libpar2 PUBLIC Threads::Threads ${PAR2_ATOMIC_LIBRARY})

if(PAR2_BUILD_TOOL)
  add_executable(par2
    src/par2cmdline.cpp
    src/commandline.cpp src/commandline.h
  )

  if(WIN32)
    target_sources(par2 PRIVATE src/wargs.cpp src/wargs.h)
  endif()

  set_target_properties(par2 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_include_directories(par2 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
  )

  target_compile_definitions(par2 PRIVATE HAVE_CONFIG_H)

  if(NOT WIN32)
    target_compile_definitions(par2 PRIVATE
      _FILE_OFFSET_BITS=64
      _LARGEFILE_SOURCE
      _LARGE_FILES
    )
  endif()

  if(MSVC)
    target_compile_options(par2 PRIVATE /utf-8)
  else()
    target_compile_options(par2 PRIVATE -Wall)
  endif()

  if(MINGW)
    target_link_options(par2 PRIVATE -municode)
  endif()

  par2_add_size_options(par2)

  if(MSVC)
    target_link_options(par2 PRIVATE
      "$<${PAR2_OPTIMISED}:/OPT:REF>"
      "$<${PAR2_OPTIMISED}:/OPT:ICF>"
    )
  elseif(APPLE)
    target_link_options(par2 PRIVATE "$<${PAR2_OPTIMISED}:-Wl,-dead_strip>")
  else()
    target_link_options(par2 PRIVATE "$<${PAR2_OPTIMISED}:-Wl,--gc-sections>")
  endif()

  target_link_libraries(par2 PRIVATE libpar2)
endif()

if(PAR2_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PAR2_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS libpar2
    EXPORT par2-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )

  install(DIRECTORY include/par2 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  install(EXPORT par2-targets
    FILE par2-targets.cmake
    NAMESPACE par2::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/par2"
  )

  configure_package_config_file(
    cmake/par2-config.cmake.in
    "${PROJECT_BINARY_DIR}/par2-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/par2"
  )

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/par2-config-version.cmake"
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${PROJECT_BINARY_DIR}/par2-config.cmake"
    "${PROJECT_BINARY_DIR}/par2-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/par2"
  )

  if(PAR2_BUILD_TOOL)
    install(TARGETS par2 RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(FILES man/par2.1 DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")

    if(NOT WIN32)
      configure_file(
        cmake/install-aliases.cmake.in
        "${PROJECT_BINARY_DIR}/install-aliases.cmake"
        @ONLY
      )
      install(SCRIPT "${PROJECT_BINARY_DIR}/install-aliases.cmake")
    endif()
  endif()
endif()
