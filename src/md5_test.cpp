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

#include "libpar2internal.h"

#include <iostream>
#include <stdlib.h>

#include "md5.h"


// compares Update(length) to Update(buffer,buffersize) 
int test1() {
  unsigned char buffer[] = {0,0,0,0,0,0,0,0};
  
  MD5Context context1;
  context1.Update(buffer, sizeof(buffer));
  MD5Hash hash1;
  context1.Final(hash1);

  MD5Context context2;
  context2.Update(sizeof(buffer));
  MD5Hash hash2;
  context2.Final(hash2);

  if (hash1 != hash2) {
    cerr << "hash1 = " << hash1 << endl;
    cerr << "hash2 = " << hash2 << endl;
    return 1;
  }
  
  return 0;
}


// "MD5 of a null string is d41d8cd98f00b204e9800998ecf8427e"
// "MD5 of a null string is d4 1d 8c d9  8f 00 b2 04  e9 80 09 98  ec f8 42 7e"
// according to https://news.ycombinator.com/item?id=5653698
int test2() {

  MD5Context context1;
  MD5Hash hash1;
  context1.Final(hash1);

  MD5Hash hash2;
  hash2.hash[0] = 0xd4;
  hash2.hash[1] = 0x1d;
  hash2.hash[2] = 0x8c;
  hash2.hash[3] = 0xd9;
  hash2.hash[4] = 0x8f;
  hash2.hash[5] = 0x00;
  hash2.hash[6] = 0xb2;
  hash2.hash[7] = 0x04;
  hash2.hash[8] = 0xe9;
  hash2.hash[9] = 0x80;
  hash2.hash[10] = 0x09;
  hash2.hash[11] = 0x98;
  hash2.hash[12] = 0xec;
  hash2.hash[13] = 0xf8;
  hash2.hash[14] = 0x42;
  hash2.hash[15] = 0x7e;

  for (int i = 0; i < 16; i++) {
    if (hash1.hash[i] != hash2.hash[i]) {
      cerr << "hash1 and hash2 differ in location " << i << endl;
      cerr << "  hash1 = " << ((int) hash1.hash[i]) << endl;
      cerr << "  hash2 = " << ((int) hash2.hash[i]) << endl;
      return 1;
    }
  }
  
  return 0;
}


// test comparison operators
int test3() {
  MD5Context context1;
  MD5Hash hash1;
  context1.Final(hash1);

  MD5Hash hash2;
  hash2 = hash1;

  if (!(hash1 == hash2)) {
    cerr << "equal fail" << endl;
    return 1;
  }
  if (hash1 != hash2) {
    cerr << "not equal fail" << endl;
    return 1;
  }
  if (hash1 < hash2) {
    cerr << "less than fail" << endl;
    return 1;
  }
  if (hash1 > hash2) {
    cerr << "greater than fail" << endl;
    return 1;
  }
  if (!(hash1 <= hash2)) {
    cerr << "less than or equal fail" << endl;
    return 1;
  }
  if (!(hash1 >= hash2)) {
    cerr << "greater than or equal fail" << endl;
    return 1;
  }

  // make hash1 less than hash2 in first place
  hash1.hash[0] = 0x0;
  hash2.hash[0] = 0x1;
  if (hash1 == hash2) {
    cerr << "equal fail 2" << endl;
    return 1;
  }
  if (!(hash1 != hash2)) {
    cerr << "not equal fail 2" << endl;
    return 1;
  }
  if (!(hash1 < hash2)) {
    cerr << "less than fail 2" << endl;
    return 1;
  }
  if (hash1 > hash2) {
    cerr << "greater than fail 2" << endl;
    return 1;
  }
  if (!(hash1 <= hash2)) {
    cerr << "less than or equal fail 2" << endl;
    return 1;
  }
  if (hash1 >= hash2) {
    cerr << "greater than or equal fail 2" << endl;
    return 1;
  }

  // make them equal again
  hash1.hash[0] = 0x0;
  hash2.hash[0] = 0x0;
  // now make hash1 less than in 15th place
  hash1.hash[15] = 0x0;
  hash2.hash[15] = 0x1;

  if (hash1 == hash2) {
    cerr << "equal fail 3" << endl;
    return 1;
  }
  if (!(hash1 != hash2)) {
    cerr << "not equal fail 3" << endl;
    return 1;
  }
  if (!(hash1 < hash2)) {
    cerr << "less than fail 3" << endl;
    return 1;
  }
  if (hash1 > hash2) {
    cerr << "greater than fail 3" << endl;
    return 1;
  }
  if (!(hash1 <= hash2)) {
    cerr << "less than or equal fail 3" << endl;
    return 1;
  }
  if (hash1 >= hash2) {
    cerr << "greater than or equal fail 3" << endl;
    return 1;
  }

  return 0;
}


// generate random data.
// put it into two different contexts in different lengths
// make sure output is the same.
int test4() {
  srand(345087209);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }
  
  MD5Context context1;
  unsigned int offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    context1.Update(buffer + offset, length);
    offset += length;
  }
  MD5Hash hash1;
  context1.Final(hash1);

  MD5Context context2;
  offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    context2.Update(buffer + offset, length);
    offset += length;
  }
  MD5Hash hash2;
  context2.Final(hash2);

  if (hash1 != hash2) {
    cerr << "random hash1 = " << hash1 << endl;
    cerr << "random hash2 = " << hash2 << endl;
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
  if (test4()) {
    cerr << "FAILED: test4" << endl;
    return 1;
  }
  
  cout << "SUCCESS: md5_test complete." << endl;

  return 0;
}
  
