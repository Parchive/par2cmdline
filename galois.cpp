//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#include "par2cmdline.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

// Construct the log and antilog tables from the generator

GaloisTable::GaloisTable(void)
{
  u32 b = 1;

  for (u32 l=0; l<limit; l++)
  {
    log[b]     = (ValueType)l;
    antilog[l] = (ValueType)b;

    b <<= 1;
    if (b & count) b ^= generator;
  }

  log[0] = (ValueType)count;
  antilog[count] = 0;
}

// The one and only galois log/antilog table object
GaloisTable Galois::table;

#ifdef LONGMULTIPLY
GaloisLongMultiplyTable::GaloisLongMultiplyTable(void)
{
  Galois *table = tables;

  for (unsigned int i=0; i<bytes; i++)
  {
    for (unsigned int j=i; j<bytes; j++)
    {
      for (unsigned int ii=0; ii<256; ii++)
      {
        for (unsigned int jj=0; jj<256; jj++)
        {
          *table++ = Galois(ii << (8*i)) * Galois(jj << (8*j));
        }
      }
    }
  }
}

GaloisLongMultiplyTable::~GaloisLongMultiplyTable(void)
{
}
#endif
