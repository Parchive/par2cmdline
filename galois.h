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

#ifndef __GALOIS_H__
#define __GALOIS_H__

// This source file defines the Galois object for carrying out
// arithmetic in GF(2^16) using the generator 0x1100B.

// Also defined are the GaloisTable object (which contains log and
// anti log tables for use in multiplication and division), and
// the GaloisLongMultiplyTable object (which contains tables for
// carrying out multiplation of 16-bit galois numbers 8 bits at a time).

class GaloisTable
{
public:
  typedef u16 ValueType;

  GaloisTable(void);

  enum
  {
    bits = 16,
    count = 1<<bits,
    limit = count-1,
    generator = 0x1100B,
  };

  ValueType log[count];
  ValueType antilog[count];
};

class Galois
{
public:
  typedef u16 ValueType;

  // Basic constructors
  Galois(void) {};
  Galois(ValueType v);

  // Copy and assignment
  Galois(const Galois &right) {value = right.value;}
  Galois& operator = (const Galois &right) { value = right.value; return *this;}

  // Addition
  Galois operator + (const Galois &right) const { return (value ^ right.value); }
  Galois& operator += (const Galois &right) { value ^= right.value; return *this;}

  // Subtraction
  Galois operator - (const Galois &right) const { return (value ^ right.value); }
  Galois& operator -= (const Galois &right) { value ^= right.value; return *this;}

  // Multiplication
  Galois operator * (const Galois &right) const;
  Galois& operator *= (const Galois &right);

  // Division
  Galois operator / (const Galois &right) const;
  Galois& operator /= (const Galois &right);

  // Power
  Galois pow(unsigned int right) const;
  Galois operator ^ (unsigned int right) const;
  Galois& operator ^= (unsigned int right);

  // Cast to value and value access
  operator ValueType(void) const {return value;}
  ValueType Value(void) const {return value;}

  // Direct log and antilog
  ValueType Log(void) const;
  ValueType ALog(void) const;

  enum 
  {
    bits  = GaloisTable::bits,
    count = GaloisTable::count,
    limit = GaloisTable::limit,
  };

protected:
  ValueType value;

  static GaloisTable table;
};

#ifdef LONGMULTIPLY
class GaloisLongMultiplyTable
{
public:
  GaloisLongMultiplyTable(void);
  ~GaloisLongMultiplyTable(void);

  enum
  {
    bytes = ((Galois::bits + 7) >> 3),
    count = ((bytes * (bytes+1)) / 2),
  };

  Galois tables[count * 256 * 256];
};
#endif

inline Galois::Galois(Galois::ValueType v)
{
  value = v;
}

inline Galois Galois::operator * (const Galois &right) const
{ 
  if (value == 0 || right.value == 0) return 0;
  unsigned int sum = table.log[value] + table.log[right.value];
  if (sum >= limit) 
  {
    return table.antilog[sum-limit];
  }
  else
  {
    return table.antilog[sum];
  }
}

inline Galois& Galois::operator *= (const Galois &right)
{ 
  if (value == 0 || right.value == 0) 
  {
    value = 0;
  }
  else
  {
    unsigned int sum = table.log[value] + table.log[right.value];
    if (sum >= limit) 
    {
      value = table.antilog[sum-limit];
    }
    else
    {
      value = table.antilog[sum];
    }
  }

  return *this;
}

inline Galois Galois::operator / (const Galois &right) const
{ 
  if (value == 0) return 0;

  assert(right.value != 0);
  if (right.value == 0) {return 0;} // Division by 0!

  int sum = table.log[value] - table.log[right.value];
  if (sum < 0) 
  {
    return table.antilog[sum+limit];
  }
  else
  {
    return table.antilog[sum];
  }
}

inline Galois& Galois::operator /= (const Galois &right)
{ 
  if (value == 0) return *this;

  assert(right.value != 0);
  if (right.value == 0) {return *this;} // Division by 0!

  int sum = table.log[value] - table.log[right.value];
  if (sum < 0) 
  {
    value = table.antilog[sum+limit];
  }
  else
  {
    value = table.antilog[sum];
  }

  return *this;
}

inline Galois Galois::pow(unsigned int right) const
{
  if (value == 0) return 0;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> bits) + (sum & limit);
  if (sum >= limit) 
  {
    return table.antilog[sum-limit];
  }
  else
  {
    return table.antilog[sum];
  }  
}

inline Galois Galois::operator ^ (unsigned int right) const
{
  if (value == 0) return 0;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> bits) + (sum & limit);
  if (sum >= limit) 
  {
    return table.antilog[sum-limit];
  }
  else
  {
    return table.antilog[sum];
  }  
}

inline Galois& Galois::operator ^= (unsigned int right)
{
  if (value == 0) return *this;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> bits) + (sum & limit);
  if (sum >= limit) 
  {
    value = table.antilog[sum-limit];
  }
  else
  {
    value = table.antilog[sum];
  }

  return *this;
}

inline Galois::ValueType Galois::Log(void) const
{
  return table.log[value];
}

inline Galois::ValueType Galois::ALog(void) const
{
  return table.antilog[value];
}

#endif // __GALOIS_H__
