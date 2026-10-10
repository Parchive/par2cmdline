//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
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

#ifndef __FILENAMEMATCHER_H__
#define __FILENAMEMATCHER_H__

namespace par2
{

// A PAR2 set records a name as bytes and says nothing about what they were
// written in, so the name a set records and the name the file has on this
// system are not always the same bytes: the set may have been written on a
// machine using a code page rather than UTF-8, and a file system may write
// an accented letter as its own character or as the letter followed by the
// accent.

// The name as UTF-8. A name which is already UTF-8 is returned unchanged; one
// which is not is read as Windows-1252.
std::string FilenameToUtf8(const std::string &name);

// Whether two names are two spellings of one name. Case is left alone: a file
// system which ignores case has found the file by name already.
bool SameFilename(const std::string &a, const std::string &b);

// Finds the file a recorded name names when the file system spells that name
// differently. The directories it reads are kept, so one matcher belongs to
// one thread and reads a directory as it was when it first looked.
class FilenameMatcher
{
public:
  // The pathname of the file in the same directory as pathname whose name is
  // another spelling of its name. Empty when the directory has the name as it
  // is recorded, when nothing there is another spelling of it, and when more
  // than one thing is. Only the name is matched, not the directories leading
  // to it.
  std::string Resolve(const std::string &pathname);

private:
  // The name a directory holds for each spelling, empty where it holds more
  // than one name of that spelling.
  const std::map<std::string, std::string>& Read(const std::string &path);

  std::map<std::string, std::map<std::string, std::string> > listings;
};

} // namespace par2

#endif // __FILENAMEMATCHER_H__
