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

#ifndef __PAR1FILEFORMAT_H__
#define __PAR1FILEFORMAT_H__

#ifdef WIN32
#pragma pack(push, 1)
#define PACKED
#else
#define PACKED __attribute__ ((packed))
#endif

#ifdef _MSC_VER
#pragma warning(disable:4200)
#endif

struct PAR1MAGIC {u8 magic[8];}PACKED;

struct PAR1FILEHEADER
{
  PAR1MAGIC   magic;
  leu32       fileversion;
  leu32       programversion;
  MD5Hash     controlhash;
  MD5Hash     sethash;
  leu64       volumenumber;
  leu64       numberoffiles;
  leu64       filelistoffset;
  leu64       filelistsize;
  leu64       dataoffset;
  leu64       datasize;
}PACKED;

struct PAR1FILEENTRY
{
  leu64       entrysize;
  leu64       status;
  leu64       filesize;
  MD5Hash     hashfull;
  MD5Hash     hash16k;
  leu16       name[];
}PACKED;

enum FILEENTRYSTATUS
{
  INPARITYVOLUME = 1,
  CHECKED = 2,
};

#ifdef _MSC_VER
#pragma warning(default:4200)
#endif

#ifdef WIN32
#pragma pack(pop)
#endif
#undef PACKED

// Operators for comparing the MAGIC values

inline bool operator == (const PAR1MAGIC &left, const PAR1MAGIC &right)
{
  return (0==memcmp(&left, &right, sizeof(left)));
}

inline bool operator != (const PAR1MAGIC &left, const PAR1MAGIC &right)
{
  return !operator==(left, right);
}

extern PAR1MAGIC par1_magic;

#endif //__PAR1FILEFORMAT_H__
