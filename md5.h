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

#ifndef __MD5_H__
#define __MD5_H__

// This file defines the MD5Hash and MD5Context objects which are used
// to compute and manipulate the MD5 Hash values for a block of data.

//  Usage:
//
//  MD5Context context;
//  context.Update(buffer, length);
//
//  MD5Hash hash;
//  context.Final(hash);



// MD5 Hash value

class MD5Hash
{
public:
  // Constructor does not initialise the value
  MD5Hash(void) {};

  // Comparison operators
  bool operator==(const MD5Hash &other) const;
  bool operator!=(const MD5Hash &other) const;

  bool operator<(const MD5Hash &other) const;
  bool operator>=(const MD5Hash &other) const;
  bool operator>(const MD5Hash &other) const;
  bool operator<=(const MD5Hash &other) const;

  // Convert value to hex
  friend ostream& operator<<(ostream &s, const MD5Hash &hash);
  string print(void) const;

  // Copy and assignment
  MD5Hash(const MD5Hash &other);
  MD5Hash& operator=(const MD5Hash &other);

public:
  u8 hash[16]; // 16 byte MD5 Hash value
};

// Intermediate computation state

class MD5State
{
public:
  MD5State(void);
  void Reset(void);

public:
  void UpdateState(const u32 (&block)[16]);

protected:
  u32 state[4]; // 16 byte MD5 computation state
};

// MD5 computation context with 64 byte buffer

class MD5Context : public MD5State
{
public:
  MD5Context(void);
  ~MD5Context(void) {};
  void Reset(void);

  // Process data from a buffer
  void Update(const void *buffer, size_t length);

  // Process 0 bytes
  void Update(size_t length);

  // Compute the final hash value
  void Final(MD5Hash &output);

  // Get the Hash value and the total number of bytes processed.
  MD5Hash Hash(void) const;
  u64 Bytes(void) const {return bytes;}

  friend ostream& operator<<(ostream &s, const MD5Context &context);
  string print(void) const;

protected:
  enum {buffersize = 64};
  unsigned char block[buffersize];
  size_t used;

  u64 bytes;
};

// Compare hash values

inline bool MD5Hash::operator==(const MD5Hash &other) const
{
  for (unsigned int i=0; i<4; i++)
  {
    if (((u32*)&hash)[i] != ((u32*)&other.hash)[i])
    {
      return false;
    }
  }
  return true;
}
inline bool MD5Hash::operator!=(const MD5Hash &other) const
{
  return !operator==(other);
}

inline bool MD5Hash::operator<(const MD5Hash &other) const
{
  if (*(u32*)(&hash[12]) < *(u32*)(&other.hash[12]))
    return true;
  else if (*(u32*)(&hash[12]) > *(u32*)(&other.hash[12]))
    return false;

  else if (*(u32*)(&hash[8]) < *(u32*)(&other.hash[8]))
    return true;
  else if (*(u32*)(&hash[8]) > *(u32*)(&other.hash[8]))
    return false;

  else if (*(u32*)(&hash[4]) < *(u32*)(&other.hash[4]))
    return true;
  else if (*(u32*)(&hash[4]) > *(u32*)(&other.hash[4]))
    return false;

  else if (*(u32*)(&hash[0]) < *(u32*)(&other.hash[0]))
    return true;
  else 
    return false;
}
inline bool MD5Hash::operator>=(const MD5Hash &other) const 
{
  return !operator<(other);
}
inline bool MD5Hash::operator>(const MD5Hash &other) const 
{
  return other.operator<(*this);
}
inline bool MD5Hash::operator<=(const MD5Hash &other) const 
{
  return !other.operator<(*this);
}

inline MD5Hash::MD5Hash(const MD5Hash &other)
{
  memcpy(&hash, &other.hash, sizeof(hash));
}

inline MD5Hash& MD5Hash::operator=(const MD5Hash &other)
{
  memcpy(&hash, &other.hash, sizeof(hash));

  return *this;
}

#endif // __MD5_H__
