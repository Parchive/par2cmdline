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

#include "libpar2internal.h"
#include "unicodedata.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

namespace par2
{

// The characters Windows-1252 puts where Latin-1 has controls. The rest of
// the code page is the Latin-1 character of the same value.
static const u32 windows1252[] =
{
  0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
  0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
  0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
  0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
};

// Hangul syllables are written out by arithmetic rather than by a table.
static const u32 hangulsyllablefirst = 0xAC00;
static const u32 hangulleadfirst = 0x1100;
static const u32 hangulvowelfirst = 0x1161;
static const u32 hangultrailfirst = 0x11A7;
static const u32 hangulvowelcount = 21;
static const u32 hangultrailcount = 28;
static const u32 hangulsyllablecount = 19 * hangulvowelcount * hangultrailcount;

// Reads the characters of a UTF-8 string. False when the string is not UTF-8,
// which includes a sequence longer than the character needs, a half of a
// surrogate pair, and a character above the last one Unicode has.
static bool DecodeUtf8(const std::string &text, std::vector<u32> &characters)
{
  characters.clear();

  std::string::const_iterator p = text.begin();
  while (p != text.end())
  {
    const unsigned char lead = (unsigned char)*p++;
    u32 character;
    size_t following;
    u32 lowest;

    if (lead < 0x80)
    {
      characters.push_back(lead);
      continue;
    }
    else if (lead >= 0xC2 && lead <= 0xDF)
    {
      character = lead & 0x1F;
      following = 1;
      lowest = 0x80;
    }
    else if (lead >= 0xE0 && lead <= 0xEF)
    {
      character = lead & 0x0F;
      following = 2;
      lowest = 0x800;
    }
    else if (lead >= 0xF0 && lead <= 0xF4)
    {
      character = lead & 0x07;
      following = 3;
      lowest = 0x10000;
    }
    else
    {
      return false;
    }

    while (following-- > 0)
    {
      if (p == text.end())
        return false;

      const unsigned char next = (unsigned char)*p++;
      if (next < 0x80 || next > 0xBF)
        return false;

      character = (character << 6) | (next & 0x3F);
    }

    if (character < lowest || character > 0x10FFFF ||
        (character >= 0xD800 && character <= 0xDFFF))
      return false;

    characters.push_back(character);
  }

  return true;
}

static void AppendUtf8(std::string &text, u32 character)
{
  if (character < 0x80)
  {
    text += (char)character;
  }
  else if (character < 0x800)
  {
    text += (char)(0xC0 | (character >> 6));
    text += (char)(0x80 | (character & 0x3F));
  }
  else if (character < 0x10000)
  {
    text += (char)(0xE0 | (character >> 12));
    text += (char)(0x80 | ((character >> 6) & 0x3F));
    text += (char)(0x80 | (character & 0x3F));
  }
  else
  {
    text += (char)(0xF0 | (character >> 18));
    text += (char)(0x80 | ((character >> 12) & 0x3F));
    text += (char)(0x80 | ((character >> 6) & 0x3F));
    text += (char)(0x80 | (character & 0x3F));
  }
}

std::string FilenameToUtf8(const std::string &name)
{
  std::vector<u32> characters;
  if (DecodeUtf8(name, characters))
    return name;

  std::string utf8;
  for (std::string::const_iterator p = name.begin(); p != name.end(); ++p)
  {
    const unsigned char ch = (unsigned char)*p;
    AppendUtf8(utf8, (ch >= 0x80 && ch < 0xA0) ? windows1252[ch - 0x80] : ch);
  }

  return utf8;
}

// The order the marks of one character are written in. A character which is
// not a mark has no order of its own and separates the runs which are sorted.
static u8 CombiningClassOf(u32 character)
{
  size_t low = 0;
  size_t high = sizeof(combiningclasses) / sizeof(combiningclasses[0]);

  while (low < high)
  {
    const size_t middle = low + (high - low) / 2;

    if (combiningclasses[middle].codepoint < character)
      low = middle + 1;
    else if (combiningclasses[middle].codepoint > character)
      high = middle;
    else
      return combiningclasses[middle].combiningclass;
  }

  return 0;
}

static void AppendWrittenOut(std::vector<u32> &characters, u32 character)
{
  if (character >= hangulsyllablefirst &&
      character < hangulsyllablefirst + hangulsyllablecount)
  {
    const u32 index = character - hangulsyllablefirst;

    characters.push_back(hangulleadfirst + index / (hangulvowelcount * hangultrailcount));
    characters.push_back(hangulvowelfirst + (index % (hangulvowelcount * hangultrailcount)) / hangultrailcount);

    if (index % hangultrailcount != 0)
      characters.push_back(hangultrailfirst + index % hangultrailcount);

    return;
  }

  size_t low = 0;
  size_t high = sizeof(canonicalmappings) / sizeof(canonicalmappings[0]);

  while (low < high)
  {
    const size_t middle = low + (high - low) / 2;

    if (canonicalmappings[middle].codepoint < character)
    {
      low = middle + 1;
    }
    else if (canonicalmappings[middle].codepoint > character)
    {
      high = middle;
    }
    else
    {
      const u32 *mapping = canonicalmappings[middle].mapping;
      for (size_t i = 0; i < sizeof(canonicalmappings[middle].mapping) / sizeof(u32) && mapping[i] != 0; i++)
        characters.push_back(mapping[i]);

      return;
    }
  }

  characters.push_back(character);
}

// The characters of a name written out in full, with the marks of each
// character in the order Unicode puts them in, which is the one spelling
// every spelling of that name has in common.
static std::vector<u32> WriteOut(const std::string &name)
{
  std::vector<u32> characters;
  if (!DecodeUtf8(name, characters))
    return std::vector<u32>();

  std::vector<u32> written;
  written.reserve(characters.size());

  for (std::vector<u32>::const_iterator p = characters.begin(); p != characters.end(); ++p)
    AppendWrittenOut(written, *p);

  for (size_t i = 1; i < written.size(); i++)
  {
    const u8 order = CombiningClassOf(written[i]);
    if (order == 0)
      continue;

    size_t j = i;
    while (j > 0)
    {
      const u8 before = CombiningClassOf(written[j - 1]);
      if (before == 0 || before <= order)
        break;

      std::swap(written[j - 1], written[j]);
      j--;
    }
  }

  return written;
}

// The one spelling every spelling of a name has in common, which two names
// have the same of when they are the same name.
static std::string CanonicalFilename(const std::string &name)
{
  const std::vector<u32> written = WriteOut(FilenameToUtf8(name));

  std::string canonical;
  for (std::vector<u32>::const_iterator p = written.begin(); p != written.end(); ++p)
    AppendUtf8(canonical, *p);

  return canonical;
}

bool SameFilename(const std::string &a, const std::string &b)
{
  return a == b || CanonicalFilename(a) == CanonicalFilename(b);
}

const std::map<std::string, std::string>& FilenameMatcher::Read(const std::string &path)
{
  std::map<std::string, std::map<std::string, std::string> >::iterator read = listings.find(path);
  if (read != listings.end())
    return read->second;

  std::map<std::string, std::string> &spellings = listings[path];

  std::unique_ptr< std::list<std::string> > entries(DiskFile::FindFiles(path, "*", false));
  for (std::list<std::string>::const_iterator entry = entries->begin(); entry != entries->end(); ++entry)
  {
    std::string entrypath;
    std::string entryname;
    DiskFile::SplitFilename(*entry, entrypath, entryname);

    if (!spellings.insert(std::make_pair(CanonicalFilename(entryname), entryname)).second)
      spellings[CanonicalFilename(entryname)] = std::string();
  }

  return spellings;
}

std::string FilenameMatcher::Resolve(const std::string &pathname)
{
  std::string path;
  std::string name;
  DiskFile::SplitFilename(pathname, path, name);

  const std::map<std::string, std::string> &spellings = Read(path);

  std::map<std::string, std::string>::const_iterator spelling = spellings.find(CanonicalFilename(name));
  if (spelling == spellings.end() || spelling->second.empty() || spelling->second == name)
    return std::string();

  return path + spelling->second;
}

} // namespace par2
