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

#if defined(__FreeBSD_kernel__)
#include <sys/disk.h>
#define BLKGETSIZE64 DIOCGMEDIASIZE
#endif


#ifdef _WIN32
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define OffsetType __int64
#define MaxOffset 0x7fffffffffffffffI64

DiskFile::DiskFile(std::ostream &sout, std::ostream &serr)
: sout(&sout)
, serr(&serr)
{
  filename = "";
  filesize = 0;
  offset = 0;

  hFile = INVALID_HANDLE_VALUE;

  exists = false;
}


DiskFile::~DiskFile(void)
{
  if (hFile != INVALID_HANDLE_VALUE)
    ::CloseHandle(hFile);
}

bool DiskFile::CreateParentDirectory(string _pathname)
{
  // do we have a path separator in the filename ?
  string::size_type where;
  if (string::npos != (where = _pathname.find_last_of('/')) ||
      string::npos != (where = _pathname.find_last_of('\\')))
  {
    string path = filename.substr(0, where);

    struct stat st;
    if (stat(path.c_str(), &st) == 0)
      return true; // let the caller deal with non-directories

    if (!DiskFile::CreateParentDirectory(path))
      return false;

    if (!CreateDirectory(path.c_str(), NULL))
    {
      DWORD error = ::GetLastError();

      *serr << "Could not create the " << path << " directory: " << ErrorMessage(error) << endl;

      return false;
    }
  }
  return true;
}

// Create new file on disk and make sure that there is enough
// space on disk for it.
bool DiskFile::Create(string _filename, u64 _filesize)
{
  assert(hFile == INVALID_HANDLE_VALUE);

  filename = _filename;
  filesize = _filesize;

  if (!DiskFile::CreateParentDirectory(filename))
    return false;

  // Create the file
  hFile = ::CreateFileA(_filename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
  {
    DWORD error = ::GetLastError();

    *serr << "Could not create \"" << _filename << "\": " << ErrorMessage(error) << endl;

    return false;
  }

  if (filesize > 0)
  {
    // Seek to the end of the file
    LONG* ptrfilesize = (LONG*)&filesize;
    LONG lowoffset = ptrfilesize[0];
    LONG highoffset = ptrfilesize[1];

    if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, lowoffset, &highoffset, FILE_BEGIN))
    {
      DWORD error = ::GetLastError();

      *serr << "Could not set size of \"" << _filename << "\": " << ErrorMessage(error) << endl;

      ::CloseHandle(hFile);
      hFile = INVALID_HANDLE_VALUE;
      ::DeleteFile(_filename.c_str());

      return false;
    }

    // Set the end of the file
    if (!::SetEndOfFile(hFile))
    {
      DWORD error = ::GetLastError();

      *serr << "Could not set size of \"" << _filename << "\": " << ErrorMessage(error) << endl;

      ::CloseHandle(hFile);
      hFile = INVALID_HANDLE_VALUE;
      ::DeleteFile(_filename.c_str());

      return false;
    }
  }

  offset = filesize;

  exists = true;
  return true;
}

// Write some data to disk

bool DiskFile::Write(u64 _offset, const void *buffer, size_t length, LengthType maxlength)
{
  assert(hFile != INVALID_HANDLE_VALUE);

  if (offset != _offset)
  {
    LONG* ptroffset = (LONG*)&_offset;
    LONG lowoffset = ptroffset[0];
    LONG highoffset = ptroffset[1];

    // Seek to the required offset
    if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, lowoffset, &highoffset, FILE_BEGIN))
    {
      DWORD error = ::GetLastError();

      *serr << "Could not write " << (u64)length << " bytes to \"" << filename << "\" at offset " << _offset << ": " << ErrorMessage(error) << endl;

      return false;
    }
    offset = _offset;
  }


  while (length > 0) {

    DWORD write;
    if (length > maxlength)
      write = maxlength;
    else
      write = (LengthType) length;
    DWORD wrote = 0;

    // Write the data
    if (!::WriteFile(hFile, buffer, write, &wrote, NULL))
    {
      DWORD error = ::GetLastError();
      
      *serr << "Could not write " << write << " bytes to \"" << filename << "\" at offset " << _offset << ": " << ErrorMessage(error) << endl;
      
      return false;
    }

    if (wrote != write)
    {
      *serr << "INFO: Incomplete write to \"" << filename << "\" at offset " << _offset << ".  Expected to write " << write << " bytes and wrote " << wrote << " bytes." << endl;
    }

    offset += wrote;
    length -= wrote;
    buffer = ((char *) buffer) + wrote; 
    
    if (filesize < offset)
    {
      filesize = offset;
    }
  }

  return true;
}

// Open the file

bool DiskFile::Open(const string &_filename, u64 _filesize)
{
  assert(hFile == INVALID_HANDLE_VALUE);

  filename = _filename;
  filesize = _filesize;

  hFile = ::CreateFileA(_filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
  {
    DWORD error = ::GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      break;
    default:
      *serr << "Could not open \"" << _filename << "\": " << ErrorMessage(error) << endl;
    }

    return false;
  }

  offset = 0;
  exists = true;

  return true;
}

// Read some data from disk

bool DiskFile::Read(u64 _offset, void *buffer, size_t length, LengthType maxlength)
{
  assert(hFile != INVALID_HANDLE_VALUE);

  if (offset != _offset)
  {
    LONG* ptroffset = (LONG*)&_offset;
    LONG lowoffset = ptroffset[0];
    LONG highoffset = ptroffset[1];

    // Seek to the required offset
    if (INVALID_SET_FILE_POINTER == SetFilePointer(hFile, lowoffset, &highoffset, FILE_BEGIN))
    {
      DWORD error = ::GetLastError();

      *serr << "Could not read " << (u64)length << " bytes from \"" << filename << "\" at offset " << _offset << ": " << ErrorMessage(error) << endl;

      return false;
    }
    offset = _offset;
  }

  while (length > 0) {
  
    DWORD want;
    if (length > maxlength)
      want = maxlength;
    else
      want = (LengthType)length;
    DWORD got = 0;

    // Read the data
    if (!::ReadFile(hFile, buffer, want, &got, NULL))
    {
      DWORD error = ::GetLastError();

      *serr << "Could not read " << (u64)length << " bytes from \"" << filename << "\" at offset " << _offset << ": " << ErrorMessage(error) << endl;

      return false;
    }

    if (want != got)
    {
      *serr << "Incomplete read from \"" << filename << "\" at offset " << offset << ".  Tried to read " << want << " bytes and received " << got << " bytes." << endl;
    }
    
    offset += got;
    length -= got;
    buffer = ((char *) buffer) + got; 

    // write updates filesize.  Do we want to do that here?
  }

  return true;
}

void DiskFile::Close(void)
{
  if (hFile != INVALID_HANDLE_VALUE)
  {
    ::CloseHandle(hFile);
    hFile = INVALID_HANDLE_VALUE;
  }
}

string DiskFile::GetCanonicalPathname(string filename)
{
  char fullname[MAX_PATH];
  char *filepart;

  // Resolve a relative path to a full path
  unsigned int length = ::GetFullPathName(filename.c_str(), sizeof(fullname), fullname, &filepart);
  if (length <= 0 || sizeof(fullname) < length)
    return filename;

  // Make sure the drive letter is upper case.
  fullname[0] = toupper(fullname[0]);

  // Translate all /'s to \'s
  char *current = strchr(fullname, '/');
  while (current)
  {
    *current++ = '\\';
    current  = strchr(current, '/');
  }

  // Copy the root directory to the output string
  string longname(fullname, 3);

  // Start processing at the first path component
  current = &fullname[3];
  char *limit = &fullname[length];

  // Process until we reach the end of the full name
  while (current < limit)
  {
    char *tail;

    // Find the next \, or the end of the string
    (tail = strchr(current, '\\')) || (tail = limit);
    *tail = 0;

    // Create a wildcard to search for the path
    string wild = longname + current;
    WIN32_FIND_DATA finddata;
    HANDLE hFind = ::FindFirstFile(wild.c_str(), &finddata);
    if (hFind == INVALID_HANDLE_VALUE)
    {
      // If the component was not found then just copy the rest of the path to the
      // output buffer verbatim.
      longname += current;
      break;
    }
    ::FindClose(hFind);

    // Copy the component found to the output
    longname += finddata.cFileName;

    current = tail + 1;

    // If we have not reached the end of the name, add a "\"
    if (current < limit)
      longname += '\\';
  }

  return longname;
}

std::unique_ptr< list<string> > DiskFile::FindFiles(string path, string wildcard, bool recursive)
{
  // check path, if not ending with path separator, add one
  char pathend = *path.rbegin();
  if (pathend != '\\')
  {
    path += '\\';
  }
  list<string> *matches = new list<string>;

  wildcard = path + wildcard;
  WIN32_FIND_DATA fd;
  HANDLE h = ::FindFirstFile(wildcard.c_str(), &fd);
  if (h != INVALID_HANDLE_VALUE)
  {
    do
    {
      if (0 == (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
      {
        matches->push_back(path + fd.cFileName);
      }
      else if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
      {
        if (fd.cFileName[0] == '.') {
          continue;
        }

        string nwwildcard="*";
	std::unique_ptr< list<string> > dirmatches(
						 DiskFile::FindFiles(fd.cFileName, nwwildcard, true)
						 );

        matches->merge(*dirmatches);
      }
    } while (::FindNextFile(h, &fd));
    ::FindClose(h);
  }

  return std::unique_ptr< list<string> >(matches);
}

u64 DiskFile::GetFileSize(string filename)
{
  struct _stati64 st;
  if ((0 == _stati64(filename.c_str(), &st)) && (0 != (st.st_mode & S_IFREG)))
  {
    return st.st_size;
  }
  else
  {
    return 0;
  }
}

bool DiskFile::FileExists(string filename)
{
  struct _stati64 st;
  return ((0 == _stati64(filename.c_str(), &st)) && (0 != (st.st_mode & S_IFREG)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#else // !_WIN32
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef HAVE_FSEEKO
# define OffsetType off_t
# define MaxOffset ((off_t)0x7fffffffffffffffULL)
# define fseek fseeko
#else
# if _FILE_OFFSET_BITS == 64
#  define OffsetType unsigned long long
#  define MaxOffset 0x7fffffffffffffffULL
# else
#  define OffsetType long
#  define MaxOffset 0x7fffffffUL
# endif
#endif


DiskFile::DiskFile(std::ostream &sout, std::ostream &serr)
: sout(&sout)
, serr(&serr)
{
  //filename;
  filesize = 0;
  offset = 0;

  file = 0;

  exists = false;
}


DiskFile::~DiskFile(void)
{
  if (file != 0)
    fclose(file);
}

bool DiskFile::CreateParentDirectory(string _pathname)
{
  // do we have a path separator in the filename ?
  string::size_type where;
  if (string::npos != (where = _pathname.find_last_of('/')) ||
      string::npos != (where = _pathname.find_last_of('\\')))
  {
    string path = filename.substr(0, where);

    struct stat st;
    if (stat(path.c_str(), &st) == 0)
      return true; // let the caller deal with non-directories

    if (!DiskFile::CreateParentDirectory(path))
      return false;

    if (mkdir(path.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH))
    {
      *serr << "Could not create the " << path << " directory: " << strerror(errno) << endl;
      return false;
    }
  }
  return true;
}

// Create new file on disk and make sure that there is enough
// space on disk for it.
bool DiskFile::Create(string _filename, u64 _filesize)
{
  assert(file == 0);

  filename = _filename;
  filesize = _filesize;

  if (!DiskFile::CreateParentDirectory(filename))
    return false;

  // This is after CreateParentDirectory because
  // the Windows code would error out after too.
  if (FileExists(filename))
  {
    *serr << "Could not create \"" << _filename << "\": File already exists." << endl;
    return false;
  }

  file = fopen(_filename.c_str(), "wb");
  if (file == 0)
  {
    *serr << "Could not create " << _filename << ": " << strerror(errno) << endl;

    return false;
  }

  if (_filesize > (u64)MaxOffset)
  {
    *serr << "Requested file size for " << _filename << " is too large." << endl;
    return false;
  }

  if (_filesize > 0)
  {
    if (fseek(file, (OffsetType)_filesize-1, SEEK_SET))
    {
      *serr << "Could not set end of file of " << _filename << ": " << strerror(errno) << endl;

      fclose(file);
      file = 0;
      ::remove(filename.c_str());
      return false;
    }

    if (1 != fwrite(&_filesize, 1, 1, file))
    {
      *serr << "Could not set end of file of " << _filename << ": " << strerror(errno) << endl;

      fclose(file);
      file = 0;
      ::remove(filename.c_str());
      return false;
    }
  }

  offset = filesize;

  exists = true;
  return true;
}

// Write some data to disk

bool DiskFile::Write(u64 _offset, const void *buffer, size_t length, LengthType maxlength)
{
  assert(file != 0);

  if (offset != _offset)
  {
    if (_offset > (u64)MaxOffset)
    {
        *serr << "Could not write " << (u64)length << " bytes to " << filename << " at offset " << _offset << endl;
      return false;
    }


    if (fseek(file, (OffsetType)_offset, SEEK_SET))
    {
      *serr << "Could not write " << (u64)length << " bytes to " << filename << " at offset " << _offset << ": " << strerror(errno) << endl;
      return false;
    }
    offset = _offset;
  }

  while (length > 0) {

    LengthType write;
    if (length > maxlength)
      write = maxlength;
    else
      write = length;
    
    LengthType wrote = fwrite(buffer, 1, write, file);
    if (wrote != write)
    {
      *serr << "Could not write " << (u64)length << " bytes to " << filename << " at offset " << _offset << ": " << strerror(errno) << endl;
      return false;
    }

    offset += wrote;
    length -= wrote;
    buffer = ((char *) buffer) + wrote; 

    if (filesize < offset)
    {
      filesize = offset;
    }
  }

  return true;
}

// Open the file

bool DiskFile::Open(const string &_filename, u64 _filesize)
{
  assert(file == 0);

  filename = _filename;
  filesize = _filesize;

  if (_filesize > (u64)MaxOffset)
  {
    *serr << "File size for " << _filename << " is too large." << endl;
    return false;
  }

  file = fopen(filename.c_str(), "rb");
  if (file == 0)
  {
    return false;
  }

  offset = 0;
  exists = true;

  return true;
}

// Read some data from disk

bool DiskFile::Read(u64 _offset, void *buffer, size_t length, LengthType maxlength)
{
  assert(file != 0);

  if (offset != _offset)
  {
    if (_offset > (u64)MaxOffset)
    {
      *serr << "Could not read " << (u64)length << " bytes from " << filename << " at offset " << _offset << endl;
      return false;
    }


    if (fseek(file, (OffsetType)_offset, SEEK_SET))
    {
      *serr << "Could not read " << (u64)length << " bytes from " << filename << " at offset " << _offset << ": " << strerror(errno) << endl;
      return false;
    }
    offset = _offset;
  }


  while (length > 0) {

    LengthType want;
    if (length > maxlength)
      want = maxlength;
    else
      want = length;

    LengthType got = fread(buffer, 1, want, file);
    if (got != want) 
    {
      // NOTE: This can happen on error or when hitting the end-of-file.
      
      *serr << "Could not read " << (u64)length << " bytes from " << filename << " at offset " << _offset << ": " << strerror(errno) << endl;
      return false;
    }

    offset += got;
    length -= got;
    buffer = ((char *) buffer) + got;

    // Write() updates filesize.  Should we do that here too?
  }
  
  return true;
}

void DiskFile::Close(void)
{
  if (file != 0)
  {
    fclose(file);
    file = 0;
  }
}

// Attempt to get the full pathname of the file
string DiskFile::GetCanonicalPathname(string filename)
{
  // Is the supplied path already an absolute one
  if (filename.size() == 0 || filename[0] == '/')
    return filename;

  // Get the current directory
#ifdef PATH_MAX
  char curdir[PATH_MAX];
  if (0 == getcwd(curdir, sizeof(curdir)))
#else
  // Avoid unconditional use of PATH_MAX (not defined on hurd)
  char *curdir = get_current_dir_name();
  if (curdir == NULL)
#endif
  {
    return filename;
  }


  // Allocate a work buffer and copy the resulting full path into it.
  char *work = new char[strlen(curdir) + filename.size() + 2];
  strcpy(work, curdir);
#ifndef PATH_MAX
  free(curdir);
#endif
  if (work[strlen(work)-1] != '/')
    strcat(work, "/");
  strcat(work, filename.c_str());

  char *in = work;
  char *out = work;

  while (*in)
  {
    if (*in == '/')
    {
      if (in[1] == '.' && in[2] == '/')
      {
        // skip the input past /./
        in += 2;
      }
      else if (in[1] == '.' && in[2] == '.' && in[3] == '/')
      {
        // backtrack the output if /../ was found on the input
        in += 3;
        if (out > work)
        {
          do
          {
            out--;
          } while (out > work && *out != '/');
        }
      }
      else
      {
        *out++ = *in++;
      }
    }
    else
    {
      *out++ = *in++;
    }
  }
  *out = 0;

  string result = work;
  delete [] work;

  return result;
}

std::unique_ptr< list<string> > DiskFile::FindFiles(string path, string wildcard, bool recursive)
{
  // check path, if not ending with path separator, add one
  char pathend = *path.rbegin();
  if (pathend != '/')
  {
    path += '/';
  }
  list<string> *matches = new list<string>;

  string::size_type where;

  if ((where = wildcard.find_first_of('*')) != string::npos ||
      (where = wildcard.find_first_of('?')) != string::npos)
  {
    string front = wildcard.substr(0, where);
    bool multiple = wildcard[where] == '*';
    string back = wildcard.substr(where+1);

    DIR *dirp = opendir(path.c_str());
    if (dirp != 0)
    {
      struct dirent *d;
      while ((d = readdir(dirp)) != 0)
      {
        string name = d->d_name;

        if (name == "." || name == "..")
          continue;

        if (multiple)
        {
          if (name.size() >= wildcard.size() &&
              name.substr(0, where) == front &&
              name.substr(name.size()-back.size()) == back)
          {
            struct stat st;
            string fn = path + name;
            if (stat(fn.c_str(), &st) == 0)
            {
              if (S_ISDIR(st.st_mode) &&
                  recursive == true)
              {

                string nwwildcard="*";
                std::unique_ptr< list<string> > dirmatches(
							 DiskFile::FindFiles(fn, nwwildcard, true)
							 );
                matches->merge(*dirmatches);
              }
              else if (S_ISREG(st.st_mode))
              {
                matches->push_back(path + name);
              }
            }
          }
        }
        else
        {
          if (name.size() == wildcard.size())
          {
            string::const_iterator pw = wildcard.begin();
            string::const_iterator pn = name.begin();
            while (pw != wildcard.end())
            {
              if (*pw != '?' && *pw != *pn)
                break;
              ++pw;
              ++pn;
            }

            if (pw == wildcard.end())
            {
              struct stat st;
              string fn = path + name;
              if (stat(fn.c_str(), &st) == 0)
              {
                if (S_ISDIR(st.st_mode) &&
                    recursive == true)
                {

                  string nwwildcard="*";
		  std::unique_ptr< list<string> > dirmatches(
							   DiskFile::FindFiles(fn, nwwildcard, true)
							   );

                  matches->merge(*dirmatches);
                }
                else if (S_ISREG(st.st_mode))
                {
                  matches->push_back(path + name);
                }
              }
            }
          }
        }

      }
      closedir(dirp);
    }
  }
  else
  {
    struct stat st;
    string fn = path + wildcard;
    if (stat(fn.c_str(), &st) == 0)
    {
      if (S_ISDIR(st.st_mode) &&
          recursive == true)
      {
        string nwwildcard="*";
	std::unique_ptr< list<string> > dirmatches(
						 DiskFile::FindFiles(fn, nwwildcard, true)
						 );

        matches->merge(*dirmatches);
      }
      else if (S_ISREG(st.st_mode))
      {
        matches->push_back(path + wildcard);
      }
    }
  }

  return std::unique_ptr< list<string> >(matches);
}

u64 DiskFile::GetFileSize(string filename)
{
  struct stat st;
  if ((0 == stat(filename.c_str(), &st)) && (0 != (st.st_mode & S_IFREG)))
  {
    return st.st_size;
  }
  else
  {
    return 0;
  }
}

bool DiskFile::FileExists(string filename)
{
  struct stat st;
  return ((0 == stat(filename.c_str(), &st)) && (0 != (st.st_mode & S_IFREG)));
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

bool DiskFile::Open(void)
{
  string _filename = filename;

  return Open(_filename);
}

bool DiskFile::Open(const string &_filename)
{
  return Open(_filename, GetFileSize(_filename));
}

// Delete the file

bool DiskFile::Delete(void)
{
#ifdef _WIN32
  assert(hFile == INVALID_HANDLE_VALUE);
#else
  assert(file == 0);
#endif

  if (filename.size() > 0 && 0 == unlink(filename.c_str()))
  {
    exists = false;
    return true;
  }
  else
  {
    *serr << "Cannot delete " << filename << endl;

    return false;
  }
}

//string DiskFile::GetPathFromFilename(string filename)
//{
//  string::size_type where;
//
//  if (string::npos != (where = filename.find_last_of('/')) ||
//      string::npos != (where = filename.find_last_of('\\')))
//  {
//    return filename.substr(0, where+1);
//  }
//  else
//  {
//    return "." PATHSEP;
//  }
//}

void DiskFile::SplitFilename(string filename, string &path, string &name)
{
  string::size_type where;

  if (string::npos != (where = filename.find_last_of('/')) ||
      string::npos != (where = filename.find_last_of('\\')))
  {
    path = filename.substr(0, where+1);
    name = filename.substr(where+1);
  }
  else
  {
    path = "." PATHSEP;
    name = filename;
  }
}

void DiskFile::SplitRelativeFilename(string filename, string basepath, string &name)
{
  name = filename;
  name.erase(0, basepath.length());
}

bool DiskFile::Rename(void)
{
  char newname[_MAX_PATH+1];
  u32 index = 0;

  struct stat st;

  do
  {
    int length = snprintf(newname, _MAX_PATH, "%s.%u", filename.c_str(), (unsigned int) ++index);
    if (length < 0)
    {
      *serr << filename << " cannot be renamed." << endl;
      return false;
    }
    else if (length > _MAX_PATH)
    {
      *serr << filename << " pathlength is more than " << _MAX_PATH << "." << endl;
      return false;
    }
    newname[length] = 0;
  } while (stat(newname, &st) == 0);

  return Rename(newname);
}

bool DiskFile::Rename(string _filename)
{
#ifdef _WIN32
  assert(hFile == INVALID_HANDLE_VALUE);
#else
  assert(file == 0);
#endif

  if (::rename(filename.c_str(), _filename.c_str()) == 0)
  {
    filename = _filename;

    return true;
  }
  else
  {
    *serr << filename << " cannot be renamed to " << _filename << endl;

    return false;
  }
}

#ifdef _WIN32
string DiskFile::ErrorMessage(DWORD error)
{
  string result;

  LPVOID lpMsgBuf;
  if (::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL,
                       error,
                       MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       (LPSTR)&lpMsgBuf,
                       0,
                       NULL))
  {
    result = (char*)lpMsgBuf;
    LocalFree(lpMsgBuf);
  }
  else
  {
    char message[40];
    snprintf(message, sizeof(message), "Unknown error code (%lu)", error);
    result = message;
  }

  return result;
}
#endif

DiskFileMap::DiskFileMap(void)
{
}

DiskFileMap::~DiskFileMap(void)
{
  map<string, DiskFile*>::iterator fi = diskfilemap.begin();
  while (fi != diskfilemap.end())
  {
    delete (*fi).second;

    ++fi;
  }
}

bool DiskFileMap::Insert(DiskFile *diskfile)
{
  string filename = diskfile->FileName();
  assert(filename.length() != 0);

  pair<map<string,DiskFile*>::const_iterator,bool> location = diskfilemap.insert(pair<string,DiskFile*>(filename, diskfile));

  return location.second;
}

void DiskFileMap::Remove(DiskFile *diskfile)
{
  string filename = diskfile->FileName();
  assert(filename.length() != 0);

  diskfilemap.erase(filename);
}

DiskFile* DiskFileMap::Find(string filename) const
{
  assert(filename.length() != 0);

  map<string, DiskFile*>::const_iterator f = diskfilemap.find(filename);

  return (f != diskfilemap.end()) ?  f->second : 0;
}


FileSizeCache::FileSizeCache()
{
}

u64 FileSizeCache::get(const string &filename) {
  map<string, u64>::const_iterator f = cache.find(filename);
  if (f != cache.end())
    return f->second;

  // go to disk
  u64 filesize = DiskFile::GetFileSize(filename);

  cache.insert(pair<string,u64>(filename, filesize));
  //  pair<map<string,u64>::const_iterator,bool> location = cache.insert(pair<string,u64>(filename, filesize));
  //  if (!location.second) {
  //    throw exception?
  //  }
  return filesize;
}
