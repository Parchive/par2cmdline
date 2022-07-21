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

#pragma once

#include "galois.h"
#include "helper_cuda.cuh"

// This source file defines the CUDA version of Galois object for carrying out
// arithmetic in GF(2^16) using the generator 0x1100B on CUDA device.

// Also defined are the GaloisTable object (which contains log and
// anti log tables for use in multiplication and division), and
// the GaloisLongMultiplyTable object (which contains tables for
// carrying out multiplation of 16-bit galois numbers 8 bits at a time).

// CUDA Device global galois log/antilog table object
template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ GaloisTable<bits,generator,valuetype> *d_table;

template <const unsigned int bits, const unsigned int generator, typename valuetype>
class GaloisCu
{
public:
  typedef valuetype ValueType;

  // Basic constructors
  __device__ GaloisCu(void) {};
  __device__ GaloisCu(ValueType v);

  // Copy and assignment
  __device__ GaloisCu(const GaloisCu &right) {value = right.value;}
  __device__ GaloisCu& operator = (const GaloisCu &right) { value = right.value; return *this;}

  // Addition
  __device__ GaloisCu operator + (const GaloisCu &right) const { return (value ^ right.value); }
  __device__ GaloisCu& operator += (const GaloisCu &right) { value ^= right.value; return *this;}

  // Subtraction
  __device__ GaloisCu operator - (const GaloisCu &right) const { return (value ^ right.value); }
  __device__ GaloisCu& operator -= (const GaloisCu &right) { value ^= right.value; return *this;}

  // Multiplication
  __device__ GaloisCu operator * (const GaloisCu &right) const;
  __device__ GaloisCu& operator *= (const GaloisCu &right);

  // Division
  __device__ GaloisCu operator / (const GaloisCu &right) const;
  __device__ GaloisCu& operator /= (const GaloisCu &right);

  // Power
  __device__ GaloisCu pow(unsigned int right) const;
  __device__ GaloisCu operator ^ (unsigned int right) const;
  __device__ GaloisCu& operator ^= (unsigned int right);

  // Cast to value and value access
  __device__ operator ValueType(void) const {return value;}
  __device__ ValueType Value(void) const {return value;}

  // Direct log and antilog
  __device__ ValueType Log(void) const;
  __device__ ValueType ALog(void) const;

  // Upload Galois Table to CUDA device
  static void uploadTable(void)
  {
    if (d_table<bits, generator, valuetype>) return;

    GaloisTable<bits,generator,valuetype> table, *d;

    cudaErrchk( cudaMalloc((void**) &d, sizeof(GaloisTable<bits, generator, valuetype>)) );
    cudaErrchk( cudaMemcpy(d, &table, sizeof(GaloisTable<bits, generator, valuetype>), cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpyToSymbol(d_table<bits, generator, valuetype>, &d, sizeof(d), 0, cudaMemcpyHostToDevice) );

#ifdef _DEBUG
    printf("Copied Galois table to device.\n");
#endif // _DEBUG
  }

  // Free Galois Table from CUDA device memory.
  // To be called at the end of the program.
  static void freeTable(void)
  {
    if (!d_table<bits, generator, valuetype>) return;
    cudaFree(d_table<bits, generator, valuetype>);
    d_table<bits, generator, valuetype> = nullptr;
  }

  enum
  {
    Bits  = GaloisTable<bits,generator,valuetype>::Bits,
    Count = GaloisTable<bits,generator,valuetype>::Count,
    Limit = GaloisTable<bits,generator,valuetype>::Limit,
  };

protected:
  ValueType value;
};

// #ifdef LONGMULTIPLY
// template <class g>
// class GaloisLongMultiplyTable
// {
// public:
//   GaloisLongMultiplyTable(void);

//   typedef g G;

//   enum
//   {
//     Bytes = ((G::Bits + 7) >> 3),
//     Count = ((Bytes * (Bytes+1)) / 2),
//   };

//   G tables[Count * 256 * 256];
// };
// #endif

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype>::GaloisCu(typename GaloisCu<bits,generator,valuetype>::ValueType v)
{
  value = v;
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype> GaloisCu<bits,generator,valuetype>::operator * (const GaloisCu<bits,generator,valuetype> &right) const
{
  unsigned int sum = d_table<bits,generator,valuetype>->log[value] + d_table<bits,generator,valuetype>->log[right.value];
  if (sum >= Limit)
  {
    return d_table<bits,generator,valuetype>->antilog[sum - Limit];
  }
  else
  {
    return d_table<bits,generator,valuetype>->antilog[sum];
  }
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype>& GaloisCu<bits,generator,valuetype>::operator *= (const GaloisCu<bits,generator,valuetype> &right)
{
  if (value == 0 || right.value == 0)
  {
    value = 0;
  }
  else
  {
    unsigned int sum = table.log[value] + table.log[right.value];
    if (sum >= Limit)
    {
      value = table.antilog[sum-Limit];
    }
    else
    {
      value = table.antilog[sum];
    }
  }

  return *this;
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype> GaloisCu<bits,generator,valuetype>::operator / (const GaloisCu<bits,generator,valuetype> &right) const
{
  if (value == 0) return 0;

  assert(right.value != 0);
  if (right.value == 0) {return 0;} // Division by 0!

  int sum = table.log[value] - table.log[right.value];
  if (sum < 0)
  {
    return table.antilog[sum+Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype>& GaloisCu<bits,generator,valuetype>::operator /= (const GaloisCu<bits,generator,valuetype> &right)
{
  if (value == 0) return *this;

  assert(right.value != 0);
  if (right.value == 0) {return *this;} // Division by 0!

  int sum = table.log[value] - table.log[right.value];
  if (sum < 0)
  {
    value = table.antilog[sum+Limit];
  }
  else
  {
    value = table.antilog[sum];
  }

  return *this;
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype> GaloisCu<bits,generator,valuetype>::pow(unsigned int right) const
{
  if (right == 0) return 1;
  if (value == 0) return 0;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> Bits) + (sum & Limit);
  if (sum >= Limit)
  {
    return table.antilog[sum-Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype> GaloisCu<bits,generator,valuetype>::operator ^ (unsigned int right) const
{
  if (right == 0) return 1;
  if (value == 0) return 0;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> Bits) + (sum & Limit);
  if (sum >= Limit)
  {
    return table.antilog[sum-Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline GaloisCu<bits,generator,valuetype>& GaloisCu<bits,generator,valuetype>::operator ^= (unsigned int right)
{
  if (right == 0) {value = 1; return *this;}
  if (value == 0) return *this;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> Bits) + (sum & Limit);
  if (sum >= Limit)
  {
    value = table.antilog[sum-Limit];
  }
  else
  {
    value = table.antilog[sum];
  }

  return *this;
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline valuetype GaloisCu<bits,generator,valuetype>::Log(void) const
{
  return table.log[value];
}

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline valuetype GaloisCu<bits,generator,valuetype>::ALog(void) const
{
  return table.antilog[value];
}

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

typedef GaloisCu<8,0x11D,u8> GaloisCu8;
typedef GaloisCu<16,0x1100B,u16> GaloisCu16;
