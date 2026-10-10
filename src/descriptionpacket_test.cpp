//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "libpar2internal.h"

using namespace par2;

// The file separator
std::string fs(PATHSEP);


int test1() {
  if (DescriptionPacket::UrlEncodeChar('\t') != "%09") {
    std::cout << "UrlEncodeChar tab" << std::endl;
    return 1;
  }
  if (DescriptionPacket::UrlEncodeChar(':') != "%3A") {
    std::cout << "UrlEncodeChar tab" << std::endl;
    return 1;
  }
  // not illegal, but tests range of function.
  if (DescriptionPacket::UrlEncodeChar('\xFF') != "%FF") {
    std::cout << "UrlEncodeChar tab" << std::endl;
    return 1;
  }

  return 0;
}

// What a translation warns of, as the observer is told
class Heard : public Par2Observer
{
public:
  Heard(void)
  {
    errorlog.SetObserver(this);
  }

  void OnWarning(const Par2Warning &warning) override
  {
    warnings.push_back(warning);
  }

  ErrorLog errorlog;
  std::vector<Par2Warning> warnings;
};

// A name, what it translates to, and how many warnings that gives
struct Translation
{
  std::string from;
  std::string to;
  size_t warnings;
};

// Whether translating from gave other than expected: a different name, or a
// different number of warnings, or any not of code or not naming the file the
// warning is about
static bool Unexpected(const char *translator, const Translation &expected,
                       const std::string &to, const Heard &heard,
                       const WarningCode code, const std::string &warned)
{
  bool unexpected = false;

  if (to != expected.to) {
    std::cout << translator << " returned \"" << to << "\" for \"" << expected.from
              << "\", not \"" << expected.to << "\"" << std::endl;
    unexpected = true;
  }
  if (heard.warnings.size() != expected.warnings) {
    std::cout << translator << " warned " << heard.warnings.size() << " times for \""
              << expected.from << "\", not " << expected.warnings << std::endl;
    unexpected = true;
  }
  for (const Par2Warning &warning : heard.warnings) {
    if (warning.code != code || warning.filename != warned) {
      std::cout << translator << " warned of \"" << warning.filename << "\" with code "
                << warning.code << " for \"" << expected.from << "\"" << std::endl;
      unexpected = true;
    }
  }

  return unexpected;
}

// test TranslateFilenameFromLocalToPar2
int test2() {
  // The input to this function is the filename from a Par2 file.
  // The output is a "safe" filename
  const std::vector<Translation> translations = {
    {"input1.txt", "input1.txt", 0},
    {"dir" + fs + "input1.txt", "dir/input1.txt", 0},
    // leading dash is ugly, but allowed
    {"-input1.txt", "-input1.txt", 0},
    // tabs are a control character
    {"\tinput1.txt", "\tinput1.txt", 1},
    // colon causes problem on Windows and OSX/MacOS
    {":input1.txt", ":input1.txt", 1},
    // Astrix causes problems everywhere
    {"*input1.txt", "*input1.txt", 1},
    {"?input1.txt", "?input1.txt", 1},
#ifdef _WIN32
    // UNIX backslash on Windows systems
    {"/input1.txt", "/input1.txt", 1},
#else
    // Windows backslash on UNIX systems
    {"\\input1.txt", "\\input1.txt", 1},
#endif
    // absolute path on Windows, for the colon and for where it is
    {"C:" + fs + "input1.txt", "C:/input1.txt", 2},
    // absolute path on UNIX
    {fs + "input1.txt", "/input1.txt", 1},
    // referencing parent directory
    {".." + fs + "input1.txt", "../input1.txt", 1},
    {"tricky" + fs + ".." + fs + ".." + fs + "input1.txt", "tricky/../../input1.txt", 1},
  };

  for (const Translation &translation : translations) {
    Heard heard;
    const std::string to = DescriptionPacket::TranslateFilenameFromLocalToPar2(translation.from, &heard.errorlog);
    if (Unexpected("TranslateFilenameFromLocalToPar2", translation, to, heard,
                   wcFilenameUnsafe, translation.to))
      return 1;
  }

  return 0;
}

// tests TranslateFilenameFromPar2ToLocal
int test3() {
  const std::vector<Translation> translations = {
    {"input1.txt", "input1.txt", 0},
    {"dir/input1.txt", "dir" + fs + "input1.txt", 0},
    // no one likes control characters, like tab.
    {"\t", DescriptionPacket::UrlEncodeChar('\t'), 1},
#ifdef _WIN32
    // Windows does not allow certain characters in filenames
    {"\"*:<>?|%abcd",
     DescriptionPacket::UrlEncodeChar('\"')
     + DescriptionPacket::UrlEncodeChar('*')
     + DescriptionPacket::UrlEncodeChar(':')
     + DescriptionPacket::UrlEncodeChar('<')
     + DescriptionPacket::UrlEncodeChar('>')
     + DescriptionPacket::UrlEncodeChar('?')
     + DescriptionPacket::UrlEncodeChar('|')
     + "%abcd",
     7},
    // Do not allow absolute paths on Windows
    {"C:/system_file", "C" + DescriptionPacket::UrlEncodeChar(':') + "\\system_file", 1},
#else
    // other UNIXes - no need to test.
    {"\"*:<>?|%abcd", "\"*:<>?|%abcd", 0},
    // UNIXes and OSX/MacOS check for absolute paths
    {"/system_file", DescriptionPacket::UrlEncodeChar('/') + "system_file", 1},
    // and take a Windows slash for a mistake
    {"dir\\input1.txt", "dir/input1.txt", 1},
#endif
    // prevent access through parents
    {"../system_file",
     DescriptionPacket::UrlEncodeChar('.') + DescriptionPacket::UrlEncodeChar('.')
     + fs + "system_file",
     1},
    {"tricky/../../system_file",
     "tricky" + fs
     + DescriptionPacket::UrlEncodeChar('.') + DescriptionPacket::UrlEncodeChar('.') + fs
     + DescriptionPacket::UrlEncodeChar('.') + DescriptionPacket::UrlEncodeChar('.') + fs
     + "system_file",
     2},
  };

  for (const Translation &translation : translations) {
    Heard heard;
    const std::string to = DescriptionPacket::TranslateFilenameFromPar2ToLocal(translation.from, &heard.errorlog);
    if (Unexpected("TranslateFilenameFromPar2ToLocal", translation, to, heard,
                   wcFilenameChanged, translation.from))
      return 1;
  }

  return 0;
}

// With no errorlog to tell, the name is still made safe
int test4() {
  if (DescriptionPacket::TranslateFilenameFromPar2ToLocal("\t") != DescriptionPacket::UrlEncodeChar('\t')) {
    std::cout << "TranslateFilenameFromPar2ToLocal with nothing to tell" << std::endl;
    return 1;
  }

  return 0;
}

// A warning which a few lines explain further carries them, and one which
// needs none carries none
int test5() {
  Heard parent;
  DescriptionPacket::TranslateFilenameFromLocalToPar2(".." + fs + "input1.txt", &parent.errorlog);
  if (parent.warnings.size() != 1 || parent.warnings[0].detail.empty()
      || parent.warnings[0].detail.back() != '\n') {
    std::cout << "a parent directory is explained further" << std::endl;
    return 1;
  }

  Heard unsafe;
  DescriptionPacket::TranslateFilenameFromLocalToPar2("*input1.txt", &unsafe.errorlog);
  if (unsafe.warnings.size() != 1 || !unsafe.warnings[0].detail.empty()) {
    std::cout << "an unsafe character needs no further explaining" << std::endl;
    return 1;
  }

  return 0;
}

int main() {
  if (test1()) {
    std::cerr << "FAILED: test1" << std::endl;
    return 1;
  }
  if (test2()) {
    std::cerr << "FAILED: test2" << std::endl;
    return 1;
  }
  if (test3()) {
    std::cerr << "FAILED: test3" << std::endl;
    return 1;
  }
  if (test4()) {
    std::cerr << "FAILED: test4" << std::endl;
    return 1;
  }
  if (test5()) {
    std::cerr << "FAILED: test5" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: descriptionpacket_test complete." << std::endl;

  return 0;
}
