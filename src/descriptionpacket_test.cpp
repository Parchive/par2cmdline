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

#include "libpar2internal.h"

// The file separator
#ifdef _WIN32
string fs("\\");
#else
string fs("/");
#endif  


int test1() {
  if (DescriptionPacket::UrlEncodeChar('\t') != "%09") {
    cout << "UrlEncodeChar tab" << endl;
    return 1;
  }
  if (DescriptionPacket::UrlEncodeChar(':') != "%3A") {
    cout << "UrlEncodeChar tab" << endl;
    return 1;
  }
  // not illegal, but tests range of function.
  if (DescriptionPacket::UrlEncodeChar('\xFF') != "%FF") {
    cout << "UrlEncodeChar tab" << endl;
    return 1;
  }

  return 0;
}

// test TranslateFilenameFromLocalToPar2
int test2() {
  // The input to this function is the filename from a Par2 file.
  // The output is a "safe" filename
  string par2filename;
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "input1.txt");
  if (par2filename != "input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 nothing" << endl;
    return 1;
  }
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "dir" + fs + "input1.txt");
  if (par2filename != "dir/input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 " << fs << endl;
    return 1;
  }
  // leading dash is ugly, but allowed
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "-input1.txt");
  if (par2filename != "-input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 nothing" << endl;
    return 1;
  }
  
  cout << "---------------------------------------------------------" << endl;
  cout << "The following calls to Translate should produce warnings:" << endl;
  cout << "---------------------------------------------------------" << endl;
  // tabs are a control character
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "\tinput1.txt");
  if (par2filename != "\tinput1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 tab" << endl;
    return 1;
  }
  // colon causes problem on Windows and OSX/MacOS
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, ":input1.txt");
  if (par2filename != ":input1.txt") {
  cout << "TranslateFilenameFromLocalToPar2 :" << endl;
    return 1;
  }
  // Astrix causes problems everywhere
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "*input1.txt");
  if (par2filename != "*input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 2" << endl;
    return 1;
  }
  // Astrix causes problems everywhere
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "?input1.txt");
  if (par2filename != "?input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 3" << endl;
    return 1;
  }
#ifdef _WIN32
  // UNIX backslash on Windows systems
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "/input1.txt");
  if (par2filename != "/input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 4" << endl;
    return 1;
  }
#else  
  // Windows backslash on UNIX systems
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "\\input1.txt");
  if (par2filename != "\\input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 5" << endl;
    return 1;
  }
#endif

  // absolute path on Windows
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "C:" + fs + "input1.txt");
  if (par2filename != "C:/input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 2" << endl;
    return 1;
  }
  // absolute path on UNIX
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, fs + "input1.txt");
if (par2filename != "/input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 2" << endl;
    return 1;
  }
  // referencing parent directory
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, ".." + fs + "input1.txt");
  if (par2filename != "../input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 2" << endl;
    return 1;
  }
  par2filename = DescriptionPacket::TranslateFilenameFromLocalToPar2(cout, cerr, nlNormal, "tricky" + fs + ".." + fs + ".." + fs + "input1.txt");
  if (par2filename != "tricky/../../input1.txt") {
    cout << "TranslateFilenameFromLocalToPar2 2" << endl;
    return 1;
  }

  cout << "--------------------------------------" << endl;
  cout << "End of code meant to produce warnings." << endl;
  cout << "--------------------------------------" << endl;

  return 0;
}

// tests TranslateFilenameFromPar2ToLocal
int test3() {
  string local_filename;
  string expected;

  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "input1.txt");
  if (local_filename != "input1.txt") {
    cout << "TranslateFilenameFromPar2ToLocal normal" << endl;
    return 1;
  }
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "dir/input1.txt");
  if (local_filename != "dir" + fs + "input1.txt") {
    cout << "TranslateFilenameFromPar2ToLocal directory" << endl;
    return 1;
  }

  // no one likes control characters, like tab.
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "\t");
  expected = DescriptionPacket::UrlEncodeChar('\t');
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal tab" << endl;
    return 1;
  }


#ifdef _WIN32
  // Windows does not allow certain characters in filenames
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "\"*:<>?|%abcd");
  expected = DescriptionPacket::UrlEncodeChar('\"')
    + DescriptionPacket::UrlEncodeChar('*')
    + DescriptionPacket::UrlEncodeChar(':')
    + DescriptionPacket::UrlEncodeChar('<')
    + DescriptionPacket::UrlEncodeChar('>')
    + DescriptionPacket::UrlEncodeChar('?')
    + DescriptionPacket::UrlEncodeChar('|')
    + "%abcd";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal windows" << endl;
    return 1;
  }
#elif __APPLE__ 
  // (Assuming OSX/MacOS and not IOS!)
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "\"*:<>?|%abcd");
  expected = "\"*" 
    + DescriptionPacket::UrlEncodeChar(':')
    + "<>?|%abcd";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal OSX/MacOS" << endl;
    return 1;
  }
#else
  // other UNIXes - no need to test.
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "\"*:<>?|%abcd");
  expected = "\"*:<>?|%abcd";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal UNIX" << endl;
    return 1;
  }
#endif


#ifdef _WIN32
  // Do not allow absolute paths on Windows
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "C:/system_file");
  expected = "C"
    + DescriptionPacket::UrlEncodeChar(':')
    + "\\system_file";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal windows absolute" << endl;
    return 1;
  }
#else
  // UNIXes and OSX/MacOS check for absolute paths
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "/system_file");
  expected = DescriptionPacket::UrlEncodeChar('/')
    + "system_file";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal UNIX absolute" << endl;
    return 1;
  }
#endif

  // prevent access through parents
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "../system_file");
  expected = DescriptionPacket::UrlEncodeChar('.')
    + DescriptionPacket::UrlEncodeChar('.')
    + fs
    + "system_file";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal parent" << endl;
    cout << "    returned = " << local_filename << endl;
    cout << "    expected = " << expected << endl;
    return 1;
  }
  local_filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(cout, cerr, nlNormal, "tricky/../../system_file");
  expected = "tricky" 
    + fs 
    + DescriptionPacket::UrlEncodeChar('.')
    + DescriptionPacket::UrlEncodeChar('.')
    + fs
    + DescriptionPacket::UrlEncodeChar('.')
    + DescriptionPacket::UrlEncodeChar('.')
    + fs
    + "system_file";
  if (local_filename != expected) {
    cout << "TranslateFilenameFromPar2ToLocal parent" << endl;
    cout << "    returned = " << local_filename << endl;
    cout << "    expected = " << expected << endl;
    return 1;
  }

  return 0;
}
  

int main() {
  if (test1()) {
    cerr << "FAILED: test1" << endl;
    return 1;
  }
  if (test2()) {
    cerr << "FAILED: test2" << endl;
    return 1;
  }
  if (test3()) {
    cerr << "FAILED: test3" << endl;
    return 1;
  }

  cout << "SUCCESS: descriptionpacket_test complete." << endl;
  
  return 0;
}

