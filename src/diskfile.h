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

#ifndef __DISKFILE_H__
#define __DISKFILE_H__


// MAX_LENGTH is the maximum read/write size.  It can be OS-dependant.
// The "& ~7" is to make it 8-byte aligned.
// LengthType is the type required for the read/write by the OS.
#ifdef _WIN32
#define MAX_LENGTH (0xffffffffUL & ~7)
#define LengthType DWORD
#else // !_WIN32
#define MAX_LENGTH (0xffffffffUL & ~7)
#define LengthType size_t
#endif


#include <list>
using std::list;
#include <map>
using std::map;
#include <vector>
using std::vector;
#include <memory>

// A disk file can be any type of file that par2cmdline needs
// to read or write data from or to.

class DiskFile
{
public:
  DiskFile(std::ostream &sout, std::ostream &serr);
  ~DiskFile(void);

  // Ensures the specified path's parent directory exists
  bool CreateParentDirectory(string pathname);

  // Create a file and set its length
  bool Create(string filename, u64 filesize);

  // Write some data to the file
  // maxlength should be the default value, except during testing.
  bool Write(u64 offset, const void *buffer, size_t length,
	     LengthType maxlength = MAX_LENGTH);

  // Open the file
  bool Open(void);
  bool Open(const string &filename);
  bool Open(const string &filename, u64 filesize);

  // Check to see if the file is open
#ifdef _WIN32
  bool IsOpen(void) const {return hFile != INVALID_HANDLE_VALUE;}
#else
  bool IsOpen(void) const {return file != 0;}
#endif

  // Read some data from the file
  // maxlength should be the default value, except during testing.
  bool Read(u64 offset, void *buffer, size_t length,
	    LengthType maxlength = MAX_LENGTH);

  // Close the file
  void Close(void);

  // Get the size of the file
  u64 FileSize(void) const {return filesize;}

  // Get the name of the file
  string FileName(void) const {return filename;}

  // Does the file exist
  bool Exists(void) const {return exists;}

  // Rename the file
  bool Rename(void); // Pick a filename automatically
  bool Rename(string filename);

  // Delete the file
  bool Delete(void);

public:
  static string GetCanonicalPathname(string filename);

  static void SplitFilename(string filename, string &path, string &name);
  static void SplitRelativeFilename(string filename, string basepath, string &name);
  static std::string SplitRelativeFilename(const std::string& filename, const std::string& basepath)
  {
    std::string ret;
    SplitRelativeFilename(filename, basepath, ret);
    return ret;
  }

  static bool FileExists(string filename);
  static u64 GetFileSize(string filename);

  // Search the specified path for files which match the specified wildcard
  // and return their names in a list.
  static std::unique_ptr< list<string> > FindFiles(string path, string wildcard, bool recursive);

protected:
  // NOTE: These are pointers so that the operator= works correctly.
  // The references used elsewhere cannot be reassigned.
  // (Operator= is needed when vectors are resized.)
  std::ostream *sout; // stream for output (for commandline, this is cout)
  std::ostream *serr; // stream for errors (for commandline, this is cerr)
  
  string filename;
  u64    filesize;

  // OS file handle
#ifdef _WIN32
  HANDLE hFile;
#else
  FILE *file;
#endif

  // Current offset within the file
  u64    offset;

  // Does the file exist
  bool   exists;

protected:
#ifdef _WIN32
  static string ErrorMessage(DWORD error);
#endif
};

// This class keeps track of which DiskFile objects exist
// and which file on disk they are associated with.
// It is used to avoid a file being processed twice.
class DiskFileMap
{
public:
  DiskFileMap(void);
  ~DiskFileMap(void);

  bool Insert(DiskFile *diskfile);
  void Remove(DiskFile *diskfile);
  DiskFile* Find(string filename) const;

protected:
  map<string, DiskFile*>    diskfilemap;             // Map from filename to DiskFile
};

class FileSizeCache
{
public:  
  FileSizeCache();
  u64 get(const string &filename);
protected:
  map<string, u64> cache; 
};

#endif // __DISKFILE_H__
