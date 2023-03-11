//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#include "libpar2internal.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

// The one and only CCITT CRC32 lookup table
//
// NOTE: the constant is the reversed polynomial for CRC-32
// as listed on Wikipedia's page:
// https://en.wikipedia.org/wiki/Cyclic_redundancy_check
crc32table ccitttable(0xEDB88320L);


// GF32 multiplication
#define NEGATE32(n) (u32)(-((i32)(n)))
static u32 GF32Multiply(u32 a, u32 b, u32 polynomial)
{
  u32 product = 0;
  for (u32 i=0; i<31; i++)
  {
    product ^= NEGATE32(b >> 31) & a;
    a = ((a >> 1) ^ (polynomial & NEGATE32(a & 1)));
    b <<= 1;
  }
  product ^= NEGATE32(b >> 31) & a;
  return product;
}

// Compute 2^(8n) in CRC's GF
static u32 CRCExp8(u64 n)
{
  u32 result = 0x80000000;
  u32 power = 0;
  n %= 0xffffffff;
  while (n)
  {
    if (n & 1)
      result = GF32Multiply(result, ccitttable.power[power], ccitttable.polynom);
    n >>= 1;
    power = (power + 1) & 31;
  }
  return result;
}


// Construct the CRC32 lookup table from the specified polynomial
//
// This seems to follow:
// http://www.efg2.com/Lab/Library/UseNet/1999/0117.txt
crc32table::crc32table(u32 polynomial)
{
  polynom = polynomial;
  for (u32 i = 0; i <= 255 ; i++)
  {
    u32 crc = i;

    for (u32 j = 0; j < 8; j++)
    {
      crc = (crc >> 1) ^ ((crc & 1) ? polynomial : 0);
    }

    table[i] = crc;
  }
  
  // Also construct the table used for computing power
  // Note that the table is rotated by 3 entries, since we operate on bytes, i.e. 1<<3 bits
  u32 k = 0x80000000 >> 1;
  for (u32 i = 0; i < 32; i++)
  {
    power[(i - 3) & 31] = k;
    k = GF32Multiply(k, k, polynomial);
  }
}

// Construct a CRC32 lookup table for windowing
void GenerateWindowTable(u64 window, u32 (&target)[256])
{
  // Window coefficient
  u32 coeff = CRCExp8(window);
  // Extend initial CRC to window length
  u32 mask = GF32Multiply(~0, coeff, ccitttable.polynom);
  // Xor initial CRC with that extended by one byte
  mask = GF32Multiply(mask, 0x80800000, ccitttable.polynom);
  // Since we have a table, may as well invert all bits to save doing it later
  mask ^= ~0;
  
  // Generate table
  for (i16 i=0; i<=255; i++)
  {
    target[i] = GF32Multiply(ccitttable.table[i], coeff, ccitttable.polynom) ^ mask;
  }
}

u32 CRCUpdateBlock(u32 crc, u64 length)
{
  return GF32Multiply(crc, CRCExp8(length), ccitttable.polynom);
}
