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

# 77 is the exit status automake reports as SKIP, and the tests still use it.
set(PAR2_SKIP_RETURN_CODE 77)

function(par2_add_unit_test name)
  cmake_parse_arguments(arg "LINK_LIBPAR2" "" "SOURCES" ${ARGN})

  add_executable(${name} ${PROJECT_SOURCE_DIR}/src/${name}.cpp ${arg_SOURCES})

  set_target_properties(${name} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests"
  )
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${config}" config)
    set_property(TARGET ${name} PROPERTY
      RUNTIME_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}/tests")
  endforeach()

  target_include_directories(${name} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}"
  )

  target_compile_definitions(${name} PRIVATE HAVE_CONFIG_H)

  if(NOT WIN32)
    target_compile_definitions(${name} PRIVATE
      _FILE_OFFSET_BITS=64
      _LARGEFILE_SOURCE
      _LARGE_FILES
    )
  endif()

  if(MSVC)
    target_compile_options(${name} PRIVATE /utf-8)
  else()
    target_compile_options(${name} PRIVATE -Wall)
  endif()

  if(arg_LINK_LIBPAR2)
    target_link_libraries(${name} PRIVATE libpar2)
  else()
    target_link_libraries(${name} PRIVATE Threads::Threads ${PAR2_ATOMIC_LIBRARY})
  endif()

  if(CMAKE_HOST_WIN32)
    add_test(NAME ${name} COMMAND ${name})
  else()
    add_test(NAME ${name}
      COMMAND /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/unit_test_runner" "$<TARGET_FILE:${name}>")
  endif()

  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE ${PAR2_SKIP_RETURN_CODE})
endfunction()

par2_add_unit_test(letype_test)
par2_add_unit_test(crc_test SOURCES ${PROJECT_SOURCE_DIR}/src/crc.cpp)
par2_add_unit_test(md5_test SOURCES ${PROJECT_SOURCE_DIR}/src/md5.cpp)
par2_add_unit_test(galois_test SOURCES ${PROJECT_SOURCE_DIR}/src/galois.cpp)
par2_add_unit_test(reedsolomon_test SOURCES ${PROJECT_SOURCE_DIR}/src/reedsolomon.cpp)
par2_add_unit_test(utf8_test SOURCES
  ${PROJECT_SOURCE_DIR}/src/utf8.cpp
  ${PROJECT_SOURCE_DIR}/src/wargs.cpp)

par2_add_unit_test(diskfile_test LINK_LIBPAR2
  SOURCES ${PROJECT_SOURCE_DIR}/src/diskfile.cpp)
par2_add_unit_test(filechecksummer_test LINK_LIBPAR2
  SOURCES ${PROJECT_SOURCE_DIR}/src/filechecksummer.cpp)
par2_add_unit_test(descriptionpacket_test LINK_LIBPAR2
  SOURCES ${PROJECT_SOURCE_DIR}/src/descriptionpacket.cpp)
par2_add_unit_test(criticalpacket_test LINK_LIBPAR2
  SOURCES ${PROJECT_SOURCE_DIR}/src/criticalpacket.cpp)
par2_add_unit_test(commandline_test LINK_LIBPAR2
  SOURCES ${PROJECT_SOURCE_DIR}/src/commandline.cpp)
par2_add_unit_test(libpar2_test LINK_LIBPAR2)
par2_add_unit_test(foreach_parallel_test LINK_LIBPAR2)
par2_add_unit_test(taskpool_test LINK_LIBPAR2)
par2_add_unit_test(bufferpool_test LINK_LIBPAR2)

if(NOT PAR2_BUILD_TOOL)
  return()
endif()

# The integration tests run the built binary from the build root, where they
# also create and remove their own working directories.
# The scripts read srcdir relative to the directory they run in, which they
# resolve through getcwd, so both ends have to be real paths.
get_filename_component(PAR2_REAL_BINARY_DIR "${PROJECT_BINARY_DIR}" REALPATH)
get_filename_component(PAR2_REAL_SOURCE_DIR "${PROJECT_SOURCE_DIR}" REALPATH)
file(RELATIVE_PATH PAR2_SRCDIR "${PAR2_REAL_BINARY_DIR}" "${PAR2_REAL_SOURCE_DIR}")
if(PAR2_SRCDIR STREQUAL "")
  set(PAR2_SRCDIR ".")
endif()

if(CMAKE_HOST_WIN32)
  file(GLOB PAR2_INTEGRATION_TESTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test[0-9]*.ps1")
else()
  file(GLOB PAR2_INTEGRATION_TESTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test[0-9]*")
  list(FILTER PAR2_INTEGRATION_TESTS EXCLUDE REGEX "\\.[^/]*$")
endif()

list(SORT PAR2_INTEGRATION_TESTS)

foreach(script ${PAR2_INTEGRATION_TESTS})
  get_filename_component(name "${script}" NAME_WE)

  if(CMAKE_HOST_WIN32)
    add_test(NAME ${name}
      COMMAND pwsh -NoProfile -File "${script}"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT "PARBINARY=${PROJECT_BINARY_DIR}/par2${CMAKE_EXECUTABLE_SUFFIX}")
  else()
    add_test(NAME ${name}
      COMMAND /bin/sh "${script}"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT "srcdir=${PAR2_SRCDIR}")
  endif()

  set_tests_properties(${name} PROPERTIES
    SKIP_RETURN_CODE ${PAR2_SKIP_RETURN_CODE}
    RUN_SERIAL TRUE)
endforeach()

if(NOT CMAKE_HOST_WIN32)
  if(PAR2_ATOMIC_LIBRARY)
    set(PAR2_CONSUMER_LIBS "-l${PAR2_ATOMIC_LIBRARY}")
  else()
    set(PAR2_CONSUMER_LIBS "")
  endif()

  string(TOUPPER "${CMAKE_BUILD_TYPE}" PAR2_BUILD_TYPE)
  set(PAR2_CONSUMER_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${PAR2_BUILD_TYPE}}")

  # A generator which hands the SDK to the compiler itself leaves it out of the
  # flags, and the consumer is compiled by hand.
  if(APPLE)
    set(PAR2_SDK "${CMAKE_OSX_SYSROOT}")
    if(NOT IS_DIRECTORY "${PAR2_SDK}")
      if(NOT PAR2_SDK)
        set(PAR2_SDK macosx)
      endif()
      execute_process(COMMAND xcrun --sdk "${PAR2_SDK}" --show-sdk-path
        OUTPUT_VARIABLE PAR2_SDK
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    endif()
    if(IS_DIRECTORY "${PAR2_SDK}")
      string(APPEND PAR2_CONSUMER_CXXFLAGS " -isysroot ${PAR2_SDK}")
    endif()
  endif()
  if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET)
    string(APPEND PAR2_CONSUMER_CXXFLAGS
      " -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()

  string(STRIP "${PAR2_CONSUMER_CXXFLAGS}" PAR2_CONSUMER_CXXFLAGS)
  if(PAR2_CONSUMER_CXXFLAGS STREQUAL "")
    set(PAR2_CONSUMER_CXXFLAGS "-O0")
  endif()

  add_test(NAME consumer_build
    COMMAND /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/consumer_build"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

  set_tests_properties(consumer_build PROPERTIES
    SKIP_RETURN_CODE ${PAR2_SKIP_RETURN_CODE}
    ENVIRONMENT
      "srcdir=${PAR2_SRCDIR};CXX=${CMAKE_CXX_COMPILER};CXXFLAGS=${PAR2_CONSUMER_CXXFLAGS};PTHREAD_FLAGS=${CMAKE_THREAD_LIBS_INIT};LIBS=${PAR2_CONSUMER_LIBS};EXEEXT=${CMAKE_EXECUTABLE_SUFFIX}")
else()
  if(CMAKE_GENERATOR_PLATFORM)
    set(PAR2_CONSUMER_PLATFORM ${CMAKE_GENERATOR_PLATFORM})
  else()
    set(PAR2_CONSUMER_PLATFORM x64)
  endif()

  add_test(NAME consumer_build
    COMMAND pwsh -NoProfile -File "${CMAKE_CURRENT_SOURCE_DIR}/consumer_build.ps1"
      -Platform ${PAR2_CONSUMER_PLATFORM}
      -LibPath "$<TARGET_FILE:libpar2>"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

  set_tests_properties(consumer_build PROPERTIES
    SKIP_RETURN_CODE ${PAR2_SKIP_RETURN_CODE}
    ENVIRONMENT "srcdir=${PAR2_SRCDIR}")
endif()
