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
#ifdef __CUDACC__
#ifndef __GALOIS_CU_H__
#define __GALOIS_CU_H__

#include "libpar2.h"
#include "galois.h"

template <const unsigned int bits, const unsigned int generator, typename valuetype> class GaloisTable;
template <const unsigned int bits, const unsigned int generator, typename valuetype> class Galois;

template <class g> class GaloisLongMultiplyTable;

// This source file defines the Galois object for carrying out
// arithmetic in GF(2^16) using the generator 0x1100B.

// Also defined are the GaloisTable object (which contains log and
// anti log tables for use in multiplication and division), and
// the GaloisLongMultiplyTable object (which contains tables for
// carrying out multiplation of 16-bit galois numbers 8 bits at a time).

template <const unsigned int bits, const unsigned int generator, typename valuetype>
class GaloisTable
{
public:
  typedef valuetype ValueType;

  GaloisTable(void);

  enum
  {
    Bits = bits,
    Count = 1<<Bits,
    Limit = Count-1,
    Generator = generator,
  };

  ValueType log[Count];
  ValueType antilog[Count];
};

template <const unsigned int bits, const unsigned int generator, typename valuetype>
class Galois_Cu
{
public:
  typedef valuetype ValueType;

  // Basic constructors
  __device__ Galois_Cu(void) {};
  __device__ Galois_Cu(ValueType v);

  // Copy and assignment
  __device__ inline Galois_Cu(const Galois_Cu &right) {value = right.value;}
  __device__ inline Galois_Cu& operator = (const Galois_Cu &right) { value = right.value; return *this;}

  // Addition
  __device__ inline Galois_Cu operator + (const Galois_Cu &right) const { return (value ^ right.value); }
  __device__ inline Galois_Cu& operator += (const Galois_Cu &right) { value ^= right.value; return *this;}

  // Subtraction
  __device__ inline Galois_Cu operator - (const Galois_Cu &right) const { return (value ^ right.value); }
  __device__ inline Galois_Cu& operator -= (const Galois_Cu &right) { value ^= right.value; return *this;}

  // Multiplication
  __device__ Galois_Cu operator * (const Galois_Cu &right) const;
  __device__ Galois_Cu& operator *= (const Galois_Cu &right);

  // Division
  __device__ Galois_Cu operator / (const Galois_Cu &right) const;
  __device__ Galois_Cu& operator /= (const Galois_Cu &right);

  // Power
  __device__ Galois_Cu pow(unsigned int right) const;
  __device__ Galois_Cu operator ^ (unsigned int right) const;
  __device__ Galois_Cu& operator ^= (unsigned int right);

  // Cast to value and value access
  __device__ operator ValueType(void) const {return value;}
  __device__ ValueType Value(void) const {return value;}

  // Direct log and antilog
  __device__ ValueType Log(void) const;
  __device__ ValueType ALog(void) const;

  enum
  {
    Bits  = GaloisTable<bits,generator,valuetype>::Bits,
    Count = GaloisTable<bits,generator,valuetype>::Count,
    Limit = GaloisTable<bits,generator,valuetype>::Limit,
  };

protected:
  ValueType value;

  static GaloisTable<bits,generator,valuetype> table;
};

#ifdef LONGMULTIPLY
template <class g>
class GaloisLongMultiplyTable
{
public:
  GaloisLongMultiplyTable(void);

  typedef g G;

  enum
  {
    Bytes = ((G::Bits + 7) >> 3),
    Count = ((Bytes * (Bytes+1)) / 2),
  };

  G tables[Count * 256 * 256];
};
#endif

/*
// Construct the log and antilog tables from the generator

template <const unsigned int bits, const unsigned int generator, typename valuetype>
inline GaloisTable<bits,generator,valuetype>::GaloisTable(void)
{
  u32 b = 1;

  for (u32 l=0; l<Limit; l++)
  {
    log[b]     = (ValueType)l;
    antilog[l] = (ValueType)b;

    b <<= 1;
    if (b & Count) b ^= Generator;
  }

  log[0] = (ValueType)Limit;
  antilog[Limit] = 0;
}
*/

// The one and only galois log/antilog table object

template <const unsigned int bits, const unsigned int generator, typename valuetype>
GaloisTable<bits,generator,valuetype> Galois<bits,generator,valuetype>::table;

#ifdef LONGMULTIPLY
template <class g>
inline GaloisLongMultiplyTable<g>::GaloisLongMultiplyTable(void)
{
  G *table = tables;

  for (unsigned int i=0; i<Bytes; i++)
  {
    for (unsigned int j=i; j<Bytes; j++)
    {
      for (unsigned int ii=0; ii<256; ii++)
      {
        for (unsigned int jj=0; jj<256; jj++)
        {
          *table++ = G(ii << (8*i)) * G(jj << (8*j));
        }
      }
    }
  }
}
#endif

typedef Galois<8,0x11D,u8> Galois8;
typedef Galois<16,0x1100B,u16> Galois16;

#endif // __GALOIS_CU_H__
#endif // CUDACC