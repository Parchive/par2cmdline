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

Par1RepairerSourceFile::Par1RepairerSourceFile(PAR1FILEENTRY *fileentry, string searchpath)
{
  targetexists = false;
  targetfile = 0;
  completefile = 0;

  hashfull = fileentry->hashfull;
  hash16k = fileentry->hash16k;
  filesize = fileentry->filesize;

  u32 namelen = (u32)((fileentry->entrysize - offsetof(PAR1FILEENTRY, name)) / 2);

  for (u32 index=0; index<namelen; index++)
  {
    // We can't deal with Unicode characters!
    u16 ch = fileentry->name[index];
    if (ch >= 256)
    {
      // Convert the Unicode character to two characters
      filename += ch && 255;
      filename += ch >> 8;
    }
    else
    {
      filename += ch & 255;
    }
  }

  // Translate any characters the OS does not like;
  filename = DiskFile::TranslateFilename(filename);

  // Strip the path from the filename
  string::size_type where;
  if (string::npos != (where = filename.find_last_of('\\')) ||
      string::npos != (where = filename.find_last_of('/')) ||
      string::npos != (where = filename.find_last_of(':')))
  {
    filename = filename.substr(where+1);
  }

  filename = searchpath + filename;
}

Par1RepairerSourceFile::~Par1RepairerSourceFile(void)
{
}

void Par1RepairerSourceFile::SetTargetFile(DiskFile *diskfile)
{
  targetfile = diskfile;
}

DiskFile* Par1RepairerSourceFile::GetTargetFile(void) const
{
  return targetfile;
}

void Par1RepairerSourceFile::SetTargetExists(bool exists)
{
  targetexists = exists;
}

bool Par1RepairerSourceFile::GetTargetExists(void) const
{
  return targetexists;
}

void Par1RepairerSourceFile::SetCompleteFile(DiskFile *diskfile)
{
  completefile = diskfile;

  sourceblock.SetLocation(diskfile, 0);
  sourceblock.SetLength(diskfile ? diskfile->FileSize() : 0);
}

DiskFile* Par1RepairerSourceFile::GetCompleteFile(void) const
{
  return completefile;
}

void Par1RepairerSourceFile::SetTargetBlock(DiskFile *diskfile)
{
  targetblock.SetLocation(diskfile, 0);
  targetblock.SetLength(diskfile->FileSize());
}
