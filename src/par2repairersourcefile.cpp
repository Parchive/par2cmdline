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

namespace par2
{

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
  unicodefilenamepacket = 0;

  blockcount = 0;
  firstblocknumber = 0;

//  verificationhashtable = 0;

  targetexists = false;
  targetfile = 0;
  completefile = 0;

  diskfilesize = 0;
}

Par2RepairerSourceFile::~Par2RepairerSourceFile(void)
{
  delete descriptionpacket;
  delete verificationpacket;
  delete unicodefilenamepacket;

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

void Par2RepairerSourceFile::SetUnicodeFilenamePacket(UnicodeFilenamePacket *_unicodefilenamepacket)
{
  unicodefilenamepacket = _unicodefilenamepacket;
}

std::string Par2RepairerSourceFile::FileName(void) const
{
  if (unicodefilenamepacket && !unicodefilenamepacket->FileName().empty())
    return unicodefilenamepacket->FileName();

  return descriptionpacket ? descriptionpacket->FileName() : std::string();
}

void Par2RepairerSourceFile::ComputeTargetFileName(const std::string &path, FilenameMatcher &matcher, const ErrorLog *errorlog)
{
  // Get a version of the filename compatible with the OS, saying what was
  // changed only the first time it is worked out
  const bool first = targetfilename.empty();
  std::string filename = DescriptionPacket::TranslateFilenameFromPar2ToLocal(FileName(), first ? errorlog : 0);

#ifndef _WIN32
  // A name the set records in a code page is written in UTF-8
  const std::string utf8 = FilenameToUtf8(filename);
  if (utf8 != filename && first && errorlog)
    errorlog->Warn(wcFilenameChanged,
                   "The set records \"" + utf8 + "\" in a code page rather than UTF-8.",
                   FileName());
  filename = utf8;
#endif

  targetfilename = path + filename;

  // The name in the description packet, when the unicode name is used instead
  otherfilenames.clear();
  if (descriptionpacket && FileName() != descriptionpacket->FileName())
  {
    const std::string described = path + DescriptionPacket::TranslateFilenameFromPar2ToLocal(descriptionpacket->FileName());
    if (described != targetfilename)
      otherfilenames.push_back(described);
  }

  // Or another spelling of either name, under which the file is on disk
  std::vector<std::string> names(1, targetfilename);
  names.insert(names.end(), otherfilenames.begin(), otherfilenames.end());

  for (const auto &name : names)
  {
    const std::string spelling = matcher.Resolve(name);
    if (!spelling.empty() && spelling != targetfilename &&
        std::find(otherfilenames.begin(), otherfilenames.end(), spelling) == otherfilenames.end())
      otherfilenames.push_back(spelling);
  }
}

std::string Par2RepairerSourceFile::TargetFileName(void) const
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
                                       std::vector<DataBlock>::iterator _sourceblocks,
                                       std::vector<DataBlock>::iterator _targetblocks,
                                       u64 blocksize)
{
  firstblocknumber = _blocknumber;
  blockcount = _blockcount;
  sourceblocks = _sourceblocks;
  targetblocks = _targetblocks;

  if (blockcount > 0)
  {
    u64 filesize = descriptionpacket->FileSize();

    std::vector<DataBlock>::iterator sb = sourceblocks;
    for (u32 blocknumber=0; blocknumber<blockcount; ++blocknumber, ++sb)
    {
      DataBlock &datablock = *sb;

      u64 blocklength = std::min(blocksize, filesize-(u64)blocknumber*blocksize);

      datablock.SetFilesize(filesize);
      datablock.SetLength(blocklength);
    }
  }
}

// Determine the block count from the file size and block size.
bool Par2RepairerSourceFile::SetBlockCount(u64 blocksize)
{
  if (descriptionpacket)
  {
    u64 filesize = descriptionpacket->FileSize();
    u64 count = filesize / blocksize + (filesize % blocksize != 0);

    if (count > (u64)~(u32)0)
    {
      blockcount = 0;
      return false;
    }

    blockcount = (u32)count;
  }
  else
  {
    blockcount = 0;
  }

  return true;
}

void Par2RepairerSourceFile::SetDiskFileSize()
{
  diskfilesize = DiskFile::GetFileSize(targetfilename);
}

} // namespace par2
