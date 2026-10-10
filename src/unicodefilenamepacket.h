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

#ifndef __UNICODEFILENAMEPACKET_H__
#define __UNICODEFILENAMEPACKET_H__

namespace par2
{

// The unicode filename packet gives the name of a file in UTF-16. The name
// is used in place of the one in the file's description packet.

class UnicodeFilenamePacket : public CriticalPacket
{
public:
  // Construct the packet
  UnicodeFilenamePacket(void) {};
  ~UnicodeFilenamePacket(void) {};

public:
  // Load a unicode filename packet from a specified file
  bool Load(DiskFile *diskfile, u64 offset, PACKET_HEADER &header);

  // The file the name belongs to
  const MD5Hash& FileId(void) const;

  // The name in UTF-8, empty when the packet does not hold UTF-16
  const std::string& FileName(void) const {return filename;}

  // The UTF-8 of count UTF-16 code units, each two bytes little endian. False
  // when they are not UTF-16: a surrogate without its other half, or a zero.
  static bool Utf16ToUtf8(const u8 *units, size_t count, std::string &utf8);

protected:
  std::string filename;
};

inline const MD5Hash& UnicodeFilenamePacket::FileId(void) const
{
  assert(packetdata != 0);

  return ((const UNICODEFILENAMEPACKET*)packetdata)->fileid;
}

} // namespace par2

#endif // __UNICODEFILENAMEPACKET_H__
