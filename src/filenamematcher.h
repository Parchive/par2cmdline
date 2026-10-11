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

// The characters of a UTF-8 string. False when the string is not UTF-8, which
// includes a sequence longer than the character needs, a half of a surrogate
// pair, and a character above the last one Unicode has.
bool DecodeUtf8(const std::string &text, std::vector<u32> &characters);

// Append a character to a UTF-8 string
void AppendUtf8(std::string &text, u32 character);

// The name as UTF-8. A name which is already UTF-8 is returned unchanged; one
// which is not is read as Windows opens it, in its ANSI code page, and
// elsewhere as Windows-1252, where a byte the code page leaves undefined is
// written as %XX.
std::string FilenameToUtf8(const std::string &name);

// The one spelling every spelling of a name has in common, so two names are
// the same name when theirs are equal. Case is left alone: a file system
// which ignores case has found the file by name already.
std::string CanonicalFilename(const std::string &name);

// Finds the file a recorded name names when the file system spells that name,
// or the name of a directory leading to it, differently. The directories it
// reads are kept, so one matcher belongs to one thread and reads a directory
// as it was when it first looked.
class FilenameMatcher
{
public:
  // The pathname under which the file at pathname is on disk when that is
  // another spelling of its name, or of the directories leading to it. Empty
  // when the file opens as given, when nothing is another spelling of it, and
  // when more than one thing is.
  std::string Resolve(const std::string &pathname);

private:
  // The directory a path names, ending in a separator: the path itself when
  // it exists, otherwise the directory under another spelling of its last
  // component in the directory its parent resolves to. Empty when there is
  // none.
  std::string ResolveDirectory(const std::string &path);

  // The other spelling of name among the files, or the directories, in path.
  // Empty when there is none, or more than one, and for a plain ASCII name.
  std::string Find(const std::string &path, const std::string &name, const bool directory);

  // The name a directory holds for each spelling of its files, or of its
  // directories, empty where it holds more than one name of that spelling.
  const std::map<std::string, std::string>& Read(const std::string &path, const bool directories);

  std::map<std::string, std::map<std::string, std::string> > filelistings;
  std::map<std::string, std::map<std::string, std::string> > directorylistings;
};

} // namespace par2

#endif // __FILENAMEMATCHER_H__
