//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
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
#include <string>
#include <vector>

#include "libpar2internal.h"

using namespace par2;

// The bytes of UTF-16 code units, each little endian
static std::vector<u8> Units(const std::vector<u32> &units) {
  std::vector<u8> bytes;
  for (u32 unit : units) {
    bytes.push_back((u8)(unit & 0xff));
    bytes.push_back((u8)(unit >> 8));
  }
  return bytes;
}

static bool Converts(const std::vector<u32> &units, const std::string &expected) {
  const std::vector<u8> bytes = Units(units);
  std::string utf8 = "left over";
  return UnicodeFilenamePacket::Utf16ToUtf8(bytes.data(), units.size(), utf8) && utf8 == expected;
}

static bool Rejects(const std::vector<u32> &units) {
  const std::vector<u8> bytes = Units(units);
  std::string utf8;
  return !UnicodeFilenamePacket::Utf16ToUtf8(bytes.data(), units.size(), utf8);
}

// Characters of one, two, three and four bytes in UTF-8
int test1() {
  if (!Converts({'a', '.', 'b'}, "a.b")) {
    std::cout << "ASCII was not converted" << std::endl;
    return 1;
  }
  if (!Converts({0x00E9}, "\xc3\xa9")) {
    std::cout << "U+00E9 was not converted" << std::endl;
    return 1;
  }
  if (!Converts({0x65E5, 0x672C}, "\xe6\x97\xa5\xe6\x9c\xac")) {
    std::cout << "U+65E5 U+672C were not converted" << std::endl;
    return 1;
  }
  if (!Converts({0xD83D, 0xDE00}, "\xf0\x9f\x98\x80")) {
    std::cout << "the surrogate pair for U+1F600 was not converted" << std::endl;
    return 1;
  }
  if (!Converts({0xDBFF, 0xDFFF}, "\xf4\x8f\xbf\xbf")) {
    std::cout << "the surrogate pair for U+10FFFF was not converted" << std::endl;
    return 1;
  }
  if (!Converts({0xFFFD}, "\xef\xbf\xbd")) {
    std::cout << "U+FFFD was not converted" << std::endl;
    return 1;
  }

  return 0;
}

// Units which are not UTF-16
int test2() {
  if (!Rejects({'a', 0xD83D})) {
    std::cout << "a high surrogate at the end was accepted" << std::endl;
    return 1;
  }
  if (!Rejects({0xD83D, 'a'})) {
    std::cout << "a high surrogate without its low one was accepted" << std::endl;
    return 1;
  }
  if (!Rejects({0xDE00, 'a'})) {
    std::cout << "a low surrogate on its own was accepted" << std::endl;
    return 1;
  }
  if (!Rejects({0xD83D, 0xD83D, 0xDE00})) {
    std::cout << "two high surrogates were accepted" << std::endl;
    return 1;
  }
  if (!Rejects({'a', 0, 'b'})) {
    std::cout << "a zero within the name was accepted" << std::endl;
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

  std::cout << "SUCCESS: unicodefilenamepacket_test complete." << std::endl;

  return 0;
}
