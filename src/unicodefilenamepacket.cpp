//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
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

bool UnicodeFilenamePacket::Load(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  // Is the packet big enough to hold a name
  if (header.length <= sizeof(UNICODEFILENAMEPACKET))
  {
    return false;
  }

  // Is the packet too large (what is the longest permissible filename)
  if (header.length - sizeof(UNICODEFILENAMEPACKET) > 200000)
  {
    return false;
  }

  // Allocate the packet
  UNICODEFILENAMEPACKET *packet = (UNICODEFILENAMEPACKET *)AllocatePacket((size_t)header.length);

  packet->header = header;

  // Read the rest of the packet from disk
  if (!diskfile->Read(offset + sizeof(PACKET_HEADER),
                      &packet->fileid,
                      (size_t)packet->header.length - sizeof(PACKET_HEADER)))
    return false;

  // The name less the zero units it is padded with
  size_t count = ((size_t)packet->header.length - sizeof(UNICODEFILENAMEPACKET)) / 2;
  while (count > 0 && packet->name[2*count-2] == 0 && packet->name[2*count-1] == 0)
    --count;

  if (count == 0 || !Utf16ToUtf8(packet->name, count, filename))
    filename.clear();

  return true;
}

bool UnicodeFilenamePacket::Utf16ToUtf8(const u8 *units, size_t count, std::string &utf8)
{
  utf8.clear();

  for (size_t i = 0; i < count; i++)
  {
    u32 character = units[2*i] | ((u32)units[2*i+1] << 8);

    if (character == 0)
      return false;

    // A high surrogate and the low one which follows it are one character
    if (character >= 0xD800 && character <= 0xDBFF)
    {
      if (i + 1 == count)
        return false;

      const u32 low = units[2*i+2] | ((u32)units[2*i+3] << 8);
      if (low < 0xDC00 || low > 0xDFFF)
        return false;

      character = 0x10000 + ((character - 0xD800) << 10) + (low - 0xDC00);
      i++;
    }
    else if (character >= 0xDC00 && character <= 0xDFFF)
    {
      return false;
    }

    if (character < 0x80)
    {
      utf8 += (char)character;
    }
    else if (character < 0x800)
    {
      utf8 += (char)(0xC0 | (character >> 6));
      utf8 += (char)(0x80 | (character & 0x3F));
    }
    else if (character < 0x10000)
    {
      utf8 += (char)(0xE0 | (character >> 12));
      utf8 += (char)(0x80 | ((character >> 6) & 0x3F));
      utf8 += (char)(0x80 | (character & 0x3F));
    }
    else
    {
      utf8 += (char)(0xF0 | (character >> 18));
      utf8 += (char)(0x80 | ((character >> 12) & 0x3F));
      utf8 += (char)(0x80 | ((character >> 6) & 0x3F));
      utf8 += (char)(0x80 | (character & 0x3F));
    }
  }

  return true;
}

} // namespace par2
