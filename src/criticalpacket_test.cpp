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

#include <iostream>
#include <fstream>

#include "libpar2internal.h"

// CriticalPacket::CompareLess
int test1() {
  CriticalPacket criticalpackets[11];

  u8 *data;
  data = (u8 *) criticalpackets[0].AllocatePacket(sizeof(MAINPACKET), 0);
  ((MAINPACKET *)data)->header.type = mainpacket_type;
  data = (u8 *) criticalpackets[1].AllocatePacket(sizeof(FILEDESCRIPTIONPACKET), 0);
  ((FILEDESCRIPTIONPACKET *)data)->header.type = filedescriptionpacket_type;
  ((FILEDESCRIPTIONPACKET *)data)->fileid.hash[0] = 0;
  data = (u8 *) criticalpackets[2].AllocatePacket(sizeof(FILEDESCRIPTIONPACKET), 0);
  ((FILEDESCRIPTIONPACKET *)data)->header.type = filedescriptionpacket_type;
  ((FILEDESCRIPTIONPACKET *)data)->fileid.hash[0] = 1;
  data = (u8 *) criticalpackets[3].AllocatePacket(sizeof(FILEDESCRIPTIONPACKET), 0);
  ((FILEDESCRIPTIONPACKET *)data)->header.type = filedescriptionpacket_type;
  ((FILEDESCRIPTIONPACKET *)data)->fileid.hash[1] = 1;
  data = (u8 *) criticalpackets[4].AllocatePacket(sizeof(FILEVERIFICATIONPACKET), 0);
  ((FILEVERIFICATIONPACKET *)data)->header.type = fileverificationpacket_type;
  ((FILEVERIFICATIONPACKET *)data)->fileid.hash[0] = 0;
  data = (u8 *) criticalpackets[5].AllocatePacket(sizeof(FILEVERIFICATIONPACKET), 0);
  ((FILEVERIFICATIONPACKET *)data)->header.type = fileverificationpacket_type;
  ((FILEVERIFICATIONPACKET *)data)->fileid.hash[0] = 1;
  data = (u8 *) criticalpackets[6].AllocatePacket(sizeof(FILEVERIFICATIONPACKET), 0);
  ((FILEVERIFICATIONPACKET *)data)->header.type = fileverificationpacket_type;
  ((FILEVERIFICATIONPACKET *)data)->fileid.hash[1] = 1;
  data = (u8 *) criticalpackets[7].AllocatePacket(sizeof(CREATORPACKET), 0);
  ((CREATORPACKET *)data)->header.type = creatorpacket_type;
  data = (u8 *) criticalpackets[8].AllocatePacket(sizeof(RECOVERYBLOCKPACKET), 0);
  ((RECOVERYBLOCKPACKET *)data)->header.type = recoveryblockpacket_type;
  ((RECOVERYBLOCKPACKET *)data)->exponent = 0;
  data = (u8 *) criticalpackets[9].AllocatePacket(sizeof(RECOVERYBLOCKPACKET), 0);
  ((RECOVERYBLOCKPACKET *)data)->header.type = recoveryblockpacket_type;
  ((RECOVERYBLOCKPACKET *)data)->exponent = 1;
  data = (u8 *) criticalpackets[10].AllocatePacket(sizeof(RECOVERYBLOCKPACKET), 0);
  // type is zeroed by AllocatePacket.
  
  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 11; j++) {
      if (CriticalPacket::CompareLess(criticalpackets + i, criticalpackets + j) != (i < j)) {
	cout << "CompareLess failed for " << i << " " << j << endl;
	return 1;
      }
    }
  }
  
  return 0;
}


int main() {
  if (test1()) {
    cerr << "FAILED: test1" << endl;
    return 1;
  }

  cout << "SUCCESS: criticalpacket_test complete." << endl;
  
  return 0;
}

