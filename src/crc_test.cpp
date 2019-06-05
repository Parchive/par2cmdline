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


// This is just a simple set of tests on md5 hash.
// The initial version just showed it was self-consistent, not accurate.

// I compile with:
//    g++ -DHAVE_CONFIG_H -I.. crc_test.cpp crc.cpp 


#include "libpar2internal.h"

#include <iostream>
#include <stdlib.h>

#include "crc.h"


// Example usage:
//   u32 checksum = ~0 ^ CRCUpdateBlock(~0, (size_t)blocksize, buffer);


// compares UpdateBlock(crc, length) to UpdateBlock(crc,buffer,buffersize) 
int test1() {
  unsigned char buffer[] = {0,0,0,0,0,0,0,0};
  
  u32 checksum1 = ~0 ^ CRCUpdateBlock(~0, sizeof(buffer), buffer);
  u32 checksum2 = ~0 ^ CRCUpdateBlock(~0, sizeof(buffer));

  if (checksum1 != checksum2) {
    cerr << "checksum1 = " << checksum1 << endl;
    cerr << "checksum2 = " << checksum2 << endl;
    return 1;
  }
  
  return 0;
}


// CRC32 of "123456789" yields 0xCBF43926
// according to http://www.ross.net/crc/download/crc_v3.txt
int test2() {
  unsigned char buffer[] = "123456789";
  size_t buffer_length = 9;
  u32 expected_checksum = 0xCBF43926u;
  
  u32 checksum1 = ~0 ^ CRCUpdateBlock(~0, buffer_length, buffer);

  if (checksum1 != expected_checksum) {
    cerr << "checksum was not precalculated value: " << hex << checksum1 << dec << endl;
    cerr << "   expected " << hex << expected_checksum << dec << endl; 
    return 1;
  }
  
  return 0;
}


// generate random data.
// put it into checksum using different length blocks
// make sure output is the same.
int test3() {
  srand(345087209);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }
  
  u32 checksum1 = ~0;
  unsigned int offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (unsigned int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    checksum1 = CRCUpdateBlock(checksum1, length, buffer + offset);
    offset += length;
  }
  checksum1 = ~0 ^ checksum1;


  u32 checksum2 = ~0;
  offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (unsigned int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    checksum2 = CRCUpdateBlock(checksum2, length, buffer + offset);
    offset += length;
  }
  checksum2 = ~0 ^ checksum2;

  if (checksum1 != checksum2) {
    cerr << "random checksum1 = " << checksum1 << endl;
    cerr << "random checksum2 = " << checksum2 << endl;
    return 1;
  }
  
  return 0;
}


// generate random data.
// compare char-at-a-time vs block
// make sure output is the same.
int test4() {
  srand(113450911);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }
  
  u32 checksum1 = ~0;
  unsigned int offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    checksum1 = CRCUpdateBlock(checksum1, length, buffer + offset);
    offset += length;
  }
  checksum1 = ~0 ^ checksum1;


  u32 checksum2 = ~0;
  for (offset = 0; offset < sizeof(buffer); offset++) {
    checksum2 = CRCUpdateChar(checksum2, *(buffer + offset));
  }
  checksum2 = ~0 ^ checksum2;

  if (checksum1 != checksum2) {
    cerr << "random checksum1 = " << checksum1 << endl;
    cerr << "random checksum2 = " << checksum2 << endl;
    return 1;
  }
  
  return 0;
}



// check windowing on random buffer
int test5() {
  srand(113450911);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }

  u64 window = 1024;
  
  u32 windowtable[256];
  GenerateWindowTable(window, windowtable);
  u32 windowmask = ComputeWindowMask(window);

  int result = 0;
  
  u32 crc = ~0 ^ CRCUpdateBlock(~0, window, buffer);
  for (int offset = 0; offset + window < sizeof(buffer) - 1; offset++) {
    // compare against reference
    u32 othercrc = ~0 ^ CRCUpdateBlock(~0, window, buffer + offset);
    if (crc != othercrc) {
      cerr << "error in window at offset " << offset << endl;
      cerr << "  checksum1 = " << crc << endl;
      cerr << "  checksum2 = " << othercrc << endl;
      result = 1;
    }

    // slide window
    crc = windowmask ^ CRCSlideChar(windowmask ^ crc, buffer[offset + window], buffer[offset], windowtable);
  }
    
  return result;
}


// Checksum of checksum table
// stolen from:
// http://www.efg2.com/Lab/Mathematics/CRC.htm
int test6() {
  u32 checksum1 = ~0 ^ CRCUpdateBlock(~0, sizeof(ccitttable), &ccitttable);
  u32 expected = 0x6FCF9E13;
  if (checksum1 != expected) {
      cerr << "error when computing checksum of checksum table " << endl;
      cerr << "  checksum1 = " << checksum1 << endl;
      cerr << "  expected = " << expected << endl;
      return 0;
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
  if (test4()) {
    cerr << "FAILED: test4" << endl;
    return 1;
  }
  if (test5()) {
    cerr << "FAILED: test5" << endl;
    return 1;
  }
  if (test6()) {
    cerr << "FAILED: test6" << endl;
    return 1;
  }

  cout << "SUCCESS: crc_test complete." << endl;
  
  return 0;
}
  
