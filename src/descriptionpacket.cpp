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

// Construct the packet and store the filename and size.

bool DescriptionPacket::Create(string filename, u64 filesize)
{
  // Allocate some extra bytes for the packet in memory so that strlen() can
  // be used on the filename. The extra bytes do not get written to disk.
  FILEDESCRIPTIONPACKET *packet = (FILEDESCRIPTIONPACKET *)AllocatePacket(sizeof(*packet) + (~3 & (3 + (u32)filename.size())), 4);

  // Store everything that is currently known in the packet.

  packet->header.magic  = packet_magic;
  packet->header.length = packetlength;
  //packet->header.hash;  // Not known yet
  //packet->header.setid; // Not known yet
  packet->header.type   = filedescriptionpacket_type;

  //packet->fileid;       // Not known yet
  //packet->hashfull;     // Not known yet
  //packet->hash16k;      // Not known yet
  packet->length        = filesize;

  memcpy(packet->name, filename.c_str(), filename.size());

  return true;
}


void DescriptionPacket::Hash16k(const MD5Hash &hash)
{
  ((FILEDESCRIPTIONPACKET *)packetdata)->hash16k = hash;
}

void DescriptionPacket::HashFull(const MD5Hash &hash)
{
  ((FILEDESCRIPTIONPACKET *)packetdata)->hashfull = hash;
}

void DescriptionPacket::ComputeFileId(void)
{
  FILEDESCRIPTIONPACKET *packet = ((FILEDESCRIPTIONPACKET *)packetdata);

  // Compute the fileid from the hash, length, and name fields in the packet.

  MD5Context context;
  context.Update(&packet->hash16k, 
                 sizeof(FILEDESCRIPTIONPACKET)-offsetof(FILEDESCRIPTIONPACKET,hash16k)
                 +strlen((const char*)packet->name));
  context.Final(packet->fileid);
}

// Load a description packet from a specified file
bool DescriptionPacket::Load(DiskFile *diskfile, u64 offset, PACKET_HEADER &header)
{
  // Is the packet big enough
  if (header.length <= sizeof(FILEDESCRIPTIONPACKET))
  {
    return false;
  }

  // Is the packet too large (what is the longest permissible filename)
  if (header.length - sizeof(FILEDESCRIPTIONPACKET) > 100000)
  {
    return false;
  }

  // Allocate the packet (with a little extra so we will have NULLs after the filename)
  FILEDESCRIPTIONPACKET *packet = (FILEDESCRIPTIONPACKET *)AllocatePacket((size_t)header.length, 4);

  packet->header = header;

  // Read the rest of the packet from disk
  if (!diskfile->Read(offset + sizeof(PACKET_HEADER), 
                      &packet->fileid, 
                      (size_t)packet->header.length - sizeof(PACKET_HEADER)))
    return false;

  // Are the file and 16k hashes consistent
  if (packet->length <= 16384 && packet->hash16k != packet->hashfull)
  {
    return false;
  }

  return true;
}


// Returns the URL-style encoding of a character:
// "%HH" where H is a hex-digit.
string DescriptionPacket::UrlEncodeChar(char c)
{
  string result("%");

  char high_bits = ((c >> 4) & 0xf);
  if (high_bits < 10)
    result += '0' + high_bits;
  else
    result += 'A' + (high_bits - 10);

  char low_bits = (c & 0xf);
  if (low_bits < 10)
    result += '0' + low_bits;
  else
    result += 'A' + (low_bits - 10);
  
  return result;
}



// Converts the filename from that on disk to the version
// in the Par file.  Par uses HTML-style slashes ('/' or
// UNIX-style slashes) to denote directories.  This
// function also prints a warning if a character may be
// illegal on another machine.
//
// NOTE: I decided to warn users, not change the files.
// If a user is just backing up files on their own system
// and not sending them to users on another operating
// system, we don't want to change the filenames.
string DescriptionPacket::TranslateFilenameFromLocalToPar2(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel, string local_filename)
{
  string par2_encoded_filename;

  string::iterator p = local_filename.begin();
  while (p != local_filename.end())
  {
    unsigned char ch = *p;

    bool ok = true;
    if (ch < 32)
    {
      ok = false;
    }
    else
    {
      switch (ch)
      {
      case '"':
      case '*':
      case ':':
      case '<':
      case '>':
      case '?':
      case '|':
        ok = false;
      }
    }

    if (!ok)
    {
      if (noiselevel >= nlNormal)
      {
	serr << "WARNING: A filename contains the character \'" << ch << "\' which some systems do not allow in filenames." << endl;
      }
    }
      
#ifdef _WIN32
    // replace Windows-slash with HTML-slash
    if (ch == '\\') {
      ch = '/';
    }
#else
    if (ch == '\\') {
      if (noiselevel >= nlNormal)
      {
	serr << "WARNING: Found Windows-style slash '\\' in filename.  Windows systems may have trouble with it." << endl;
      }
    }
#endif

    par2_encoded_filename += ch;

    ++p;
  }

  // Par files should never contain an absolute path.  On Windows,
  // These start "C:\...", etc.  An attacker could put an absolute
  // path into a Par file and overwrite system files.
  if (par2_encoded_filename.at(1) == ':')
  {
    if (noiselevel >= nlNormal)
    {
      serr << "WARNING: The second character in the filename \"" << par2_encoded_filename << "\" is a colon (':')." << endl;
      serr << "       This may be interpreted by Windows systems as an absolute path." << endl;
      serr << "       This file may be ignored by Par clients because absolute paths" << endl;
      serr << "        are a way for an attacker to overwrite system files." << endl;
    }
  }
  if (par2_encoded_filename.at(0) == '/')
  {
    if (noiselevel >= nlNormal)
    {
      serr << "WARNING: The first character in the filename \"" << par2_encoded_filename << "\" is an HTML-slash ('/')." << endl;
      serr << "       This may be interpreted by UNIX systems as an absolute path." << endl;
      serr << "       This file may be ignored by Par clients because absolute paths" << endl;
      serr << "        are a way for an attacker to overwrite system files." << endl;
    }
  }
  if (par2_encoded_filename.find("../") != string::npos)
  {
    if (noiselevel >= nlQuiet)
    {
      serr << "WARNING: The filename \"" << par2_encoded_filename << "\" contains \"..\"." << endl;
      serr << "       This is a parent directory. This file may be ignored" << endl;
      serr << "       by Par clients because parent directories are a way" << endl;
      serr << "       for an attacker to overwrite system files." << endl;
    }
  }
  if (par2_encoded_filename.length() > 255)
  {
    if (noiselevel >= nlNormal)
    {
      serr << "WARNING: A filename is over 255 characters.  That may be too long" << endl;
      serr << "         for Windows systems to handle." << endl;
    }
  }
  
  return par2_encoded_filename;
}

// Take a filename that matches the PAR2 standard ('/' slashes, etc.)
// and convert it to a legal filename on the local system.
// While at it, try to fix things: illegal characters, attempts
// to write an absolute path, directories named "..", etc.
//
// This implementation changes any illegal char into the URL-style
// encoding of %HH where H is a hex-digit.
//
// NOTE: Windows limits path names to 255 characters.  I'm not
// sure that anything can be done here for that.  
string DescriptionPacket::TranslateFilenameFromPar2ToLocal(std::ostream &sout, std::ostream &serr, const NoiseLevel noiselevel, string par2_encoded_filename)
{
  string local_filename;

  string::iterator p = par2_encoded_filename.begin();
  while (p != par2_encoded_filename.end())
  {
    unsigned char ch = *p;

    bool ok = true;
#ifdef _WIN32
    if (ch < 32)
    {
      ok = false;
    }
    else
    {
      switch (ch)
      {
      case '"':
      case '*':
      case ':':
      case '<':
      case '>':
      case '?':
      case '|':
        ok = false;
      }
    }
#elif __APPLE__
    // (Assuming OSX/MacOS and not IOS!)
    // Does not allow ':'
    if (ch < 32 || ch == ':') {
      ok = false;
    }
#else
    // other UNIXes allow anything.
    if (ch < 32)
    {
      ok = false;
    }
#endif

    // replace unix / to windows \ or windows \ to unix /
#ifdef _WIN32
    if (ch == '/')
    {
      ch = '\\';
    }
#else
    if (ch == '\\')
    {
      if (noiselevel >= nlQuiet)
      {
	// This is a legal Par2 character, but assume someone screwed up.
	serr << "INFO: Found Windows-style slash in filename.  Changing to UNIX-style slash." << endl;
	ch = '/';
      }
    }
#endif


    if (ok)
    {
      local_filename += ch;
    }
    else
    {
      if (noiselevel >= nlQuiet)
      {
	serr << "INFO: Found illegal character '" << ch << "' in filename.  Changed it to \"" << UrlEncodeChar(ch) << "\"" << endl;
	// convert problem characters to hex
	local_filename += UrlEncodeChar(ch);
      }
    }

    ++p;
  }


#ifdef _WIN32
  // Par files should never contain an absolute path.  On Windows,
  // These start "C:\...", etc.  An attacker could put an absolute
  // path into a Par file and overwrite system files.  For Windows
  // systems, we've already changed ':' to "%3A", so any absolute
  // path should have become relative.

  // Replace any references to ".." which could also be used by
  // an attacker.
  while (true) {
    size_t index = local_filename.find("..\\");
    if (index == string::npos)
      break;

    if (noiselevel >= nlQuiet)
    {
      serr << "INFO: Found attempt to write parent directory.  Changing \"..\" to \"" << UrlEncodeChar('.') << UrlEncodeChar('.') << "\"" << endl;
    }
    
    local_filename.replace(index, 2, UrlEncodeChar('.')+UrlEncodeChar('.'));
  }
#else
  // On UNIX systems, we don't want to allow filename to start with a slash,
  // because someone could be sneakily trying to overwrite a system file.
  if (local_filename.at(0) == '/')
  {
    if (noiselevel >= nlQuiet)
    {
      serr << "INFO: Found attempt to write absolute path.  Changing '/' at start of filename to \"" << UrlEncodeChar('/') << "\"" << endl;
    }
    
    local_filename.replace(0, 1, UrlEncodeChar('/'));
  }

  // replace any references to ".." which could also be sneaking
  while (true) {
    size_t index = local_filename.find("../");
    if (index == string::npos)
      break;
    if (noiselevel >= nlQuiet)
    {
      serr << "INFO: Found attempt to write parent directory.  Changing \"..\" to \"" << UrlEncodeChar('.') << UrlEncodeChar('.') << "\"" << endl;
    }
    local_filename.replace(index, 2, UrlEncodeChar('.')+UrlEncodeChar('.'));
  }
#endif
 
  return local_filename;
}


