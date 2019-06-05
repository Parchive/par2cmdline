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

bool CriticalPacket::WritePacket(DiskFile &diskfile, u64 fileoffset) const
{
  assert(packetdata != 0 && packetlength != 0);

  return diskfile.Write(fileoffset, packetdata, packetlength);
}

void CriticalPacket::FinishPacket(const MD5Hash &setid)
{
  assert(packetdata != 0 && packetlength >= sizeof(PACKET_HEADER));

  PACKET_HEADER *header = (PACKET_HEADER*)packetdata;
  header->setid = setid;

  MD5Context packetcontext;
  packetcontext.Update(&header->setid, packetlength - offsetof(PACKET_HEADER, setid));
  packetcontext.Final(header->hash);
}


// Order is:
// Main Packet
// File Description Packet (sorted by FileID)
// File Verification Packet  (sorted by FileID)
// Creator Packet
// Recovery Packet (sorted by exponent)

bool CriticalPacket::CompareLess(const CriticalPacket* const &left, const CriticalPacket* const &right)
{
  PACKET_HEADER *left_header  = (PACKET_HEADER *)left->packetdata;
  PACKET_HEADER *right_header = (PACKET_HEADER *)right->packetdata;

  int left_value;
  switch (left_header->type.type[8])
  {
  case 'M':
    left_value = 0;
    break;
  case 'F':
    left_value = 1;
    break;
  case 'I':
    left_value = 2;
    break;
  case 'C':
    left_value = 3;
    break;
  case 'R':
    left_value = 4;
    break;
  default:
    left_value = 5;
    break;
  }
  
  int right_value;
  switch (right_header->type.type[8])
  {
  case 'M':
    right_value = 0;
    break;
  case 'F':
    right_value = 1;
    break;
  case 'I':
    right_value = 2;
    break;
  case 'C':
    right_value = 3;
    break;
  case 'R':
    right_value = 4;
    break;
  default:
    right_value = 5;
    break;
  }

  if (left_value < right_value)
    return true;
  if (left_value > right_value)
    return false;

  if (left_value == 1) // file description packets
  {
    return ((FILEDESCRIPTIONPACKET *)left->packetdata)->fileid
      < ((FILEDESCRIPTIONPACKET *)right->packetdata)->fileid;
  }
  else if (left_value == 2) // file verification packets
  {
    return ((FILEVERIFICATIONPACKET *)left->packetdata)->fileid
      < ((FILEVERIFICATIONPACKET *)right->packetdata)->fileid;
  }
  else if (left_value == 4) // recovery packet
  {
    return ((RECOVERYBLOCKPACKET *)left->packetdata)->exponent
      < ((RECOVERYBLOCKPACKET *)right->packetdata)->exponent;
  }
  else
  {
    // they're equal.
    return false;
  }
}

