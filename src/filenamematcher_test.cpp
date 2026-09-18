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

#include "libpar2internal.h"

using namespace par2;

// One name as three machines write it: a Mac writes the accent as a character
// of its own, Windows and Linux write the letter which carries it, and a
// machine using a code page writes one byte for that letter.
static const std::string decomposed = "fre\xcc\x80nch_german_demo\xcc\x88.bin";
static const std::string composed = "fr\xc3\xa8nch_german_dem\xc3\xb6.bin";
static const std::string windows1252 = "fr\xe8nch_german_dem\xf6.bin";

// A name which is already UTF-8 is left alone
int test1() {
  if (FilenameToUtf8("plain.bin") != "plain.bin") {
    std::cout << "FilenameToUtf8 changed an ASCII name" << std::endl;
    return 1;
  }
  if (FilenameToUtf8(composed) != composed) {
    std::cout << "FilenameToUtf8 changed a UTF-8 name" << std::endl;
    return 1;
  }
  if (FilenameToUtf8(decomposed) != decomposed) {
    std::cout << "FilenameToUtf8 changed a decomposed UTF-8 name" << std::endl;
    return 1;
  }

  return 0;
}

// A name which is not UTF-8 is read as Windows-1252
int test2() {
  if (FilenameToUtf8(windows1252) != composed) {
    std::cout << "FilenameToUtf8 did not read a Windows-1252 name" << std::endl;
    return 1;
  }
  // The characters Windows-1252 puts where Latin-1 has controls
  if (FilenameToUtf8("\x80.bin") != "\xe2\x82\xac.bin") {
    std::cout << "FilenameToUtf8 did not read the euro sign" << std::endl;
    return 1;
  }

  return 0;
}

// The three spellings are one name
int test3() {
  if (!SameFilename(composed, decomposed)) {
    std::cout << "SameFilename did not match the two UTF-8 spellings" << std::endl;
    return 1;
  }
  if (!SameFilename(windows1252, composed)) {
    std::cout << "SameFilename did not match the Windows-1252 spelling" << std::endl;
    return 1;
  }
  if (!SameFilename(windows1252, decomposed)) {
    std::cout << "SameFilename did not match a code page against a decomposed name" << std::endl;
    return 1;
  }
  if (!SameFilename(composed, composed)) {
    std::cout << "SameFilename did not match a name against itself" << std::endl;
    return 1;
  }

  return 0;
}

// Names which are not the same name do not match
int test4() {
  if (SameFilename("one.bin", "two.bin")) {
    std::cout << "SameFilename matched two names" << std::endl;
    return 1;
  }
  // The accent is part of the name
  if (SameFilename(composed, "french_german_demo.bin")) {
    std::cout << "SameFilename ignored the accents" << std::endl;
    return 1;
  }
  // A file system which ignores case has found the file by name already
  if (SameFilename("One.bin", "one.bin")) {
    std::cout << "SameFilename ignored the case" << std::endl;
    return 1;
  }

  return 0;
}

// Marks are compared in the order Unicode puts them in, not the order they
// were written in
int test5() {
  const std::string aboveandbelow = "q\xcc\x87\xcc\xa3.bin";
  const std::string belowandabove = "q\xcc\xa3\xcc\x87.bin";

  if (!SameFilename(aboveandbelow, belowandabove)) {
    std::cout << "SameFilename did not put the marks in order" << std::endl;
    return 1;
  }

  return 0;
}

// A Hangul syllable is written out by arithmetic
int test6() {
  const std::string syllable = "\xea\xb0\x81.bin";
  const std::string jamo = "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8.bin";

  if (!SameFilename(syllable, jamo)) {
    std::cout << "SameFilename did not write out a Hangul syllable" << std::endl;
    return 1;
  }

  return 0;
}

// Writes a file and gives back the name it was written under, or an empty
// name when this file system will not hold that name
static std::string Write(const std::string &name) {
  DiskFile file;
  if (!file.Create(name, 4) || !file.Write(0, "data", 4))
    return std::string();

  file.Close();

  return name;
}

static void Remove(const std::string &name) {
  DiskFile file;
  if (file.Open(name)) {
    file.Close();
    file.Delete();
  }
}

// The file on disk is found from the name the set records
int test7() {
  const std::string ondisk = Write("runfilenamematcher_" + decomposed);
  if (ondisk.empty()) {
    std::cout << "SKIPPING: this file system will not hold the name" << std::endl;
    return 0;
  }

  FilenameMatcher matcher;
  const std::string found = matcher.Resolve("runfilenamematcher_" + composed);

  Remove(ondisk);

  if (found.empty()) {
    std::cout << "Resolve did not find the file under the other spelling" << std::endl;
    return 1;
  }

  std::string path;
  std::string name;
  DiskFile::SplitFilename(found, path, name);

  if (name != ondisk) {
    std::cout << "Resolve found \"" << name << "\" rather than the file written" << std::endl;
    return 1;
  }

  return 0;
}

// Nothing is found when the file is there under the name recorded, and when
// there is no file of that name at all
int test8() {
  const std::string ondisk = Write("runfilenamematcher_" + composed);
  if (ondisk.empty()) {
    std::cout << "SKIPPING: this file system will not hold the name" << std::endl;
    return 0;
  }

  FilenameMatcher asnamed;
  const std::string found = asnamed.Resolve(ondisk);

  Remove(ondisk);

  if (!found.empty()) {
    std::cout << "Resolve found \"" << found << "\" for a file which is there as named" << std::endl;
    return 1;
  }

  FilenameMatcher missing;
  if (!missing.Resolve("runfilenamematcher_nothing_of_that_name.bin").empty()) {
    std::cout << "Resolve found a file which is not there" << std::endl;
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
  if (test6()) {
    std::cerr << "FAILED: test6" << std::endl;
    return 1;
  }
  if (test7()) {
    std::cerr << "FAILED: test7" << std::endl;
    return 1;
  }
  if (test8()) {
    std::cerr << "FAILED: test8" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: filenamematcher_test complete." << std::endl;

  return 0;
}
