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

Par2RepairerSourceFile::Par2RepairerSourceFile(DescriptionPacket *_descriptionpacket,
                                               VerificationPacket *_verificationpacket)
: sourceblocks()
, targetblocks()
, targetfilename()
{
  descriptionpacket = _descriptionpacket;
  verificationpacket = _verificationpacket;

  blockcount = 0;
  firstblocknumber = 0;

//  verificationhashtable = 0;

  targetexists = false;
  targetfile = 0;
  completefile = 0;

#ifdef _OPENMP
  diskfilesize = 0;
#endif
}

Par2RepairerSourceFile::~Par2RepairerSourceFile(void)
{
  delete descriptionpacket;
  delete verificationpacket;

//  delete verificationhashtable;
}


void Par2RepairerSourceFile::SetDescriptionPacket(DescriptionPacket *_descriptionpacket)
{
  descriptionpacket = _descriptionpacket;
}

void Par2RepairerSourceFile::SetVerificationPacket(VerificationPacket *_verificationpacket)
{
  verificationpacket = _verificationpacket;
}

void Par2RepairerSourceFile::ComputeTargetFileName(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel, const string &path)
{
  // Get a version of the filename compatible with the OS
  string filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(sout, serr, noiselevel, descriptionpacket->FileName());

  targetfilename = path + filename;
}

string Par2RepairerSourceFile::TargetFileName(void) const
{
  return targetfilename;
}

void Par2RepairerSourceFile::SetTargetFile(DiskFile *diskfile)
{
  targetfile = diskfile;
}

DiskFile* Par2RepairerSourceFile::GetTargetFile(void) const
{
  return targetfile;
}

void Par2RepairerSourceFile::SetTargetExists(bool exists)
{
  targetexists = exists;
}

bool Par2RepairerSourceFile::GetTargetExists(void) const
{
  return targetexists;
}

void Par2RepairerSourceFile::SetCompleteFile(DiskFile *diskfile)
{
  completefile = diskfile;
}

DiskFile* Par2RepairerSourceFile::GetCompleteFile(void) const
{
  return completefile;
}

// Remember which source and target blocks will be used by this file
// and set their lengths appropriately
void Par2RepairerSourceFile::SetBlocks(u32 _blocknumber,
                                       u32 _blockcount,
                                       vector<DataBlock>::iterator _sourceblocks,
                                       vector<DataBlock>::iterator _targetblocks,
                                       u64 blocksize)
{
  firstblocknumber = _blocknumber;
  blockcount = _blockcount;
  sourceblocks = _sourceblocks;
  targetblocks = _targetblocks;

  if (blockcount > 0)
  {
    u64 filesize = descriptionpacket->FileSize();

    vector<DataBlock>::iterator sb = sourceblocks;
    for (u32 blocknumber=0; blocknumber<blockcount; ++blocknumber, ++sb)
    {
      DataBlock &datablock = *sb;

      u64 blocklength = min(blocksize, filesize-(u64)blocknumber*blocksize);

      datablock.SetFilesize(filesize);
      datablock.SetLength(blocklength);
    }
  }
}

// Determine the block count from the file size and block size.
void Par2RepairerSourceFile::SetBlockCount(u64 blocksize)
{
  if (descriptionpacket)
  {
    blockcount = (u32)((descriptionpacket->FileSize() + blocksize-1) / blocksize);
  }
  else
  {
    blockcount = 0;
  }
}

#ifdef _OPENMP
void Par2RepairerSourceFile::SetDiskFileSize()
{
  diskfilesize = DiskFile::GetFileSize(targetfilename);
}
#endif
