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

#if !defined(_WIN32) && !defined(O_NOFOLLOW)
#define O_NOFOLLOW 0
#endif

#if !defined(_WIN32) && !defined(O_CLOEXEC)
#define O_CLOEXEC 0
#endif


#ifdef _WIN32
#include "utf8.h"
#include <cwctype>
#include <iostream>
#include <set>
#endif

namespace par2
{

DiskFile::Failure::~Failure(void)
{
  if (file.errorlog)
    file.errorlog->Record(code, message.str(), name);
}


#ifdef _WIN32
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define OffsetType __int64
#define MaxOffset 0x7fffffffffffffffI64

DiskFile::DiskFile(ErrorLog *errorlog)
: errorlog(errorlog)
{
  filename = "";
  filesize = 0;
  modified = 0;

  hFile = INVALID_HANDLE_VALUE;

  exists = false;
}


DiskFile::~DiskFile(void)
{
  if (hFile != INVALID_HANDLE_VALUE)
    ::CloseHandle(hFile);
}

bool DiskFile::CreateParentDirectory(std::string _pathname)
{
  // do we have a path separator in the filename ?
  std::string::size_type where;
  if (std::string::npos != (where = _pathname.find_last_of(PATHSEP)) ||
      std::string::npos != (where = _pathname.find_last_of(ALTPATHSEP)))
  {
    std::string path = _pathname.substr(0, where);

    // Handle Windows root path (e.g., "C:" or empty path from "\file")
    // If path is empty or is a drive letter (e.g., "C:"), the root already exists
    if (path.empty() ||
        (path.length() == 2 && path[1] == ':'))
    {
      return true;
    }

    std::wstring wpath;
    if (!utf8::Utf8ToWide(path, wpath))
    {
      Failure(*this, ecFileCreateFailed, path) << "Could not convert \"" << path << "\" to a wide string.";
      return false;
    }

    struct _stati64 st;
    if (_wstati64(wpath.c_str(), &st) == 0)
      return true; // let the caller deal with non-directories

    if (!DiskFile::CreateParentDirectory(path))
      return false;

    if (!CreateDirectoryW(wpath.c_str(), NULL))
    {
      DWORD error = ::GetLastError();

      Failure(*this, ecFileCreateFailed, path) << "Could not create the " << path << " directory: " << ErrorMessage(error);

      return false;
    }
  }
  return true;
}

// Create new file on disk and make sure that there is enough
// space on disk for it.
bool DiskFile::Create(std::string _filename, u64 _filesize)
{
  assert(hFile == INVALID_HANDLE_VALUE);

  filename = _filename;
  filesize = _filesize;

  if (!DiskFile::CreateParentDirectory(filename))
    return false;

  // Create the file
  std::wstring wfilename;
  if (!utf8::Utf8ToWide(_filename, wfilename))
  {
    Failure(*this, ecFileCreateFailed, _filename) << "Could not convert \"" << _filename << "\" to a wide string.";
    return false;
  }

  hFile = ::CreateFileW(wfilename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
  {
    DWORD error = ::GetLastError();

    Failure(*this, ecFileCreateFailed, _filename) << "Could not create \"" << _filename << "\": " << ErrorMessage(error);

    return false;
  }

  if (filesize > 0)
  {
    // Seek to the end of the file
    LARGE_INTEGER end;
    end.QuadPart = (LONGLONG)filesize;

    if (!::SetFilePointerEx(hFile, end, NULL, FILE_BEGIN))
    {
      DWORD error = ::GetLastError();

      Failure(*this, ecFileCreateFailed, _filename) << "Could not set size of \"" << _filename << "\": " << ErrorMessage(error);

      ::CloseHandle(hFile);
      hFile = INVALID_HANDLE_VALUE;
      ::DeleteFileW(wfilename.c_str());

      return false;
    }

    // Set the end of the file
    if (!::SetEndOfFile(hFile))
    {
      DWORD error = ::GetLastError();

      Failure(*this, ecFileCreateFailed, _filename) << "Could not set size of \"" << _filename << "\": " << ErrorMessage(error);

      ::CloseHandle(hFile);
      hFile = INVALID_HANDLE_VALUE;
      ::DeleteFileW(wfilename.c_str());

      return false;
    }
  }

  exists = true;
  return true;
}

// Where a read or a write starts, without using the handle's file pointer
static OVERLAPPED Overlapped(const u64 offset)
{
  OVERLAPPED overlapped = {};
  overlapped.Offset = (DWORD)offset;
  overlapped.OffsetHigh = (DWORD)(offset >> 32);
  return overlapped;
}

// Write some data to disk

bool DiskFile::Write(u64 _offset, const void *buffer, size_t length, LengthType maxlength)
{
  assert(hFile != INVALID_HANDLE_VALUE);

  while (length > 0) {

    DWORD write;
    if (length > maxlength)
      write = maxlength;
    else
      write = (LengthType) length;
    DWORD wrote = 0;

    // Write the data
    OVERLAPPED overlapped = Overlapped(_offset);
    if (!::WriteFile(hFile, buffer, write, &wrote, &overlapped))
    {
      DWORD error = ::GetLastError();

      Failure(*this, ecFileWriteFailed, filename) << "Could not write " << write << " bytes to \"" << filename << "\" at offset " << _offset << ": " << ErrorMessage(error);

      return false;
    }

    if (wrote != write)
    {
      std::ostringstream message;
      message << "Incomplete write to \"" << filename << "\" at offset " << _offset << ".  Expected to write " << write << " bytes and wrote " << wrote << " bytes.";

      if (errorlog)
        errorlog->Warn(wcIncompleteWrite, message.str(), filename);
    }

    _offset += wrote;
    length -= wrote;
    buffer = ((char *) buffer) + wrote;

    if (filesize < _offset)
    {
      filesize = _offset;
    }
  }

  return true;
}

// Open the file

bool DiskFile::Open(const std::string &_filename, u64 _filesize)
{
  assert(hFile == INVALID_HANDLE_VALUE);

  filename = _filename;
  filesize = _filesize;

  std::wstring wfilename;
  if (!utf8::Utf8ToWide(_filename, wfilename))
  {
    Failure(*this, ecFileOpenFailed, _filename) << "Could not convert \"" << _filename << "\" to a wide string.";
    return false;
  }

  hFile = ::CreateFileW(wfilename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
  {
    DWORD error = ::GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      break;
    default:
      Failure(*this, ecFileOpenFailed, _filename) << "Could not open \"" << _filename << "\": " << ErrorMessage(error);
    }

    return false;
  }

  FILETIME written;
  modified = ::GetFileTime(hFile, NULL, NULL, &written)
             ? ((u64)written.dwHighDateTime << 32) | written.dwLowDateTime : 0;

  exists = true;

  return true;
}

// Open an existing file for both reading and writing

bool DiskFile::OpenForUpdate(std::string *reason)
{
  assert(hFile == INVALID_HANDLE_VALUE);

  auto refuse = [&](const std::string &why)
  {
    if (hFile != INVALID_HANDLE_VALUE)
      ::CloseHandle(hFile);
    hFile = INVALID_HANDLE_VALUE;
    if (reason)
      *reason = why;
    return false;
  };

  std::wstring wfilename;
  if (filename.empty() || !utf8::Utf8ToWide(filename, wfilename))
    return refuse("its name cannot be used");

  // A reparse point is opened as itself, and refused as Create refuses
  // symbolic links
  hFile = ::CreateFileW(wfilename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (hFile == INVALID_HANDLE_VALUE)
    return refuse(ErrorMessage(::GetLastError()));

  BY_HANDLE_FILE_INFORMATION info;
  if (!::GetFileInformationByHandle(hFile, &info))
    return refuse(ErrorMessage(::GetLastError()));
  if (0 != (info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)))
    return refuse("it is not a regular file");
  if (info.nNumberOfLinks != 1)
    return refuse("it has more than one hard link");
  if ((((u64)info.nFileSizeHigh << 32) | info.nFileSizeLow) != filesize
      || (((u64)info.ftLastWriteTime.dwHighDateTime << 32) | info.ftLastWriteTime.dwLowDateTime) != modified)
    return refuse("it changed since it was verified");

  exists = true;

  return true;
}

// Read some data from disk

bool DiskFile::Read(u64 _offset, void *buffer, size_t length, LengthType maxlength)
{
  assert(hFile != INVALID_HANDLE_VALUE);

  while (length > 0) {

    DWORD want;
    if (length > maxlength)
      want = maxlength;
    else
      want = (LengthType)length;
    DWORD got = 0;

    // Read the data
    OVERLAPPED overlapped = Overlapped(_offset);
    if (!::ReadFile(hFile, buffer, want, &got, &overlapped))
    {
      DWORD error = ::GetLastError();

      Failure(*this, ecFileReadFailed, filename) << "Could not read " << (u64)length << " bytes from \"" << filename << "\" at offset " << _offset << ": " << ErrorMessage(error);

      return false;
    }

    if (got == 0)
    {
      Failure(*this, ecFileReadFailed, filename) << "Could not read " << (u64)length << " bytes from \"" << filename << "\" at offset " << _offset << ": unexpected end of file.";

      return false;
    }

    if (want != got)
    {
      std::ostringstream message;
      message << "Incomplete read from \"" << filename << "\" at offset " << _offset << ".  Tried to read " << want << " bytes and received " << got << " bytes.";

      if (errorlog)
        errorlog->Warn(wcIncompleteRead, message.str(), filename);
    }

    _offset += got;
    length -= got;
    buffer = ((char *) buffer) + got;
  }

  return true;
}

bool DiskFile::Close(void)
{
  bool closed = true;

  if (hFile != INVALID_HANDLE_VALUE)
  {
    closed = (0 != ::CloseHandle(hFile));
    hFile = INVALID_HANDLE_VALUE;
  }

  return closed;
}

std::string DiskFile::GetCanonicalPathname(std::string filename)
{
  std::wstring wfilename;
  if (!utf8::Utf8ToWide(filename, wfilename))
    return filename;

  // First call to get required buffer size
  DWORD length = GetFullPathNameW(wfilename.c_str(), 0, nullptr, nullptr);
  if (length == 0) 
  {
    return filename; 
  }

  // Allocate buffer with required size
  auto wfullname = std::make_unique<wchar_t[]>(length);

  // Second call to get the actual path
  length = GetFullPathNameW(wfilename.c_str(), length, wfullname.get(), nullptr);
  if (length == 0)
  {
    return filename;
  }

  wfullname[0] = towupper(wfullname[0]);
  std::replace(wfullname.get(), wfullname.get() + length, L'/', L'\\');

  std::string fullname;
  if (!utf8::WideToUtf8(wfullname.get(), fullname))
    return filename;

  return fullname;
}

std::unique_ptr< std::list<std::string> > DiskFile::FindFiles(std::string path, std::string wildcard, bool recursive, bool followlinks)
{
  // check path, if not ending with path separator, add one
  char pathend = *path.rbegin();
  if (pathend != PATHSEP[0])
  {
    path += PATHSEP;
  }
  std::list<std::string> *matches = new std::list<std::string>;

  std::wstring wwildcard;
  if (!utf8::Utf8ToWide(path + wildcard, wwildcard))
    return std::unique_ptr< std::list<std::string> >(matches);

  // The directories which match, which are taken whole
  std::set<std::string> taken;

  WIN32_FIND_DATAW fd;
  HANDLE h = ::FindFirstFileW(wwildcard.c_str(), &fd);
  if (h != INVALID_HANDLE_VALUE)
  {
    do
    {
      if (0 == (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
      {
        std::string name;
        if (utf8::WideToUtf8(fd.cFileName, name))
          matches->push_back(path + name);
      }
      else if (recursive == true)
      {
        if (fd.cFileName[0] == '.') {
          continue;
        }

        std::string name;
        if (!utf8::WideToUtf8(fd.cFileName, name))
          continue;

        taken.insert(name);

        std::string nwwildcard="*";
        std::unique_ptr< std::list<std::string> > dirmatches(
          DiskFile::FindFiles(path + name, nwwildcard, true, followlinks)
        );

        // append without requiring ordering
        if (dirmatches)
          matches->splice(matches->end(), *dirmatches);
      }
    } while (::FindNextFileW(h, &fd));
    ::FindClose(h);
  }

  // A directory which does not match is searched for files which do
  std::wstring wall;
  if (recursive && wildcard.find_first_of("*?") != std::string::npos &&
      utf8::Utf8ToWide(path + "*", wall))
  {
    h = ::FindFirstFileW(wall.c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE)
    {
      do
      {
        if (0 == (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == '.')
          continue;

        std::string name;
        if (!utf8::WideToUtf8(fd.cFileName, name) || taken.count(name) != 0)
          continue;

        std::unique_ptr< std::list<std::string> > dirmatches(
          DiskFile::FindFiles(path + name, wildcard, true, followlinks)
        );

        if (dirmatches)
          matches->splice(matches->end(), *dirmatches);
      } while (::FindNextFileW(h, &fd));
      ::FindClose(h);
    }
  }

  return std::unique_ptr< std::list<std::string> >(matches);
}

u64 DiskFile::GetFileSize(std::string filename)
{
  std::wstring wfilename;
  if (!utf8::Utf8ToWide(filename, wfilename))
    return 0;

  struct _stati64 st;
  if ((0 == _wstati64(wfilename.c_str(), &st)) && (0 != (st.st_mode & S_IFREG)))
  {
    return st.st_size;
  }
  else
  {
    return 0;
  }
}

bool DiskFile::FileExists(std::string filename)
{
  std::wstring wfilename;
  if (!utf8::Utf8ToWide(filename, wfilename))
    return false;

  struct _stati64 st;
  return ((0 == _wstati64(wfilename.c_str(), &st)) && (0 != (st.st_mode & _S_IFREG)));
}

bool DiskFile::DirectoryExists(std::string path)
{
  // The system takes a directory without its trailing separator, except a
  // drive's root
  while (path.size() > 1 && (*path.rbegin() == PATHSEP[0] || *path.rbegin() == ALTPATHSEP[0]) &&
         path[path.size() - 2] != ':')
    path.erase(path.size() - 1);

  std::wstring wpath;
  if (!utf8::Utf8ToWide(path, wpath))
    return false;

  const DWORD attributes = ::GetFileAttributesW(wpath.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES && 0 != (attributes & FILE_ATTRIBUTE_DIRECTORY);
}

std::unique_ptr< std::list<std::string> > DiskFile::FindDirectories(std::string path)
{
  if (*path.rbegin() != PATHSEP[0])
    path += PATHSEP;

  std::list<std::string> *matches = new std::list<std::string>;

  std::wstring wwildcard;
  if (!utf8::Utf8ToWide(path + "*", wwildcard))
    return std::unique_ptr< std::list<std::string> >(matches);

  WIN32_FIND_DATAW fd;
  HANDLE h = ::FindFirstFileW(wwildcard.c_str(), &fd);
  if (h != INVALID_HANDLE_VALUE)
  {
    do
    {
      std::string name;
      if (0 != (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
          0 == (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) &&
          utf8::WideToUtf8(fd.cFileName, name) && name != "." && name != "..")
        matches->push_back(path + name);
    } while (::FindNextFileW(h, &fd));
    ::FindClose(h);
  }

  return std::unique_ptr< std::list<std::string> >(matches);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#else // !_WIN32
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define OffsetType off_t
#define MaxOffset ((std::numeric_limits<OffsetType>::max)())

// When a file was last written, in nanoseconds where the system gives them
static u64 ModifiedTime(const struct stat &st)
{
#if defined(HAVE_STRUCT_STAT_ST_MTIM)
  return (u64)st.st_mtim.tv_sec * 1000000000 + (u64)st.st_mtim.tv_nsec;
#elif defined(HAVE_STRUCT_STAT_ST_MTIMESPEC)
  return (u64)st.st_mtimespec.tv_sec * 1000000000 + (u64)st.st_mtimespec.tv_nsec;
#else
  return (u64)st.st_mtime * 1000000000;
#endif
}


DiskFile::DiskFile(ErrorLog *errorlog)
: errorlog(errorlog)
{
  //filename;
  filesize = 0;
  modified = 0;

  fd = -1;

  exists = false;
}


DiskFile::~DiskFile(void)
{
  if (fd >= 0)
    close(fd);
}

bool DiskFile::CreateParentDirectory(std::string _pathname)
{
  // do we have a path separator in the filename ?
  std::string::size_type where;
  if (std::string::npos != (where = _pathname.find_last_of(PATHSEP)) ||
      std::string::npos != (where = _pathname.find_last_of(ALTPATHSEP)))
  {
    std::string path = _pathname.substr(0, where);

    // Handle root path - if path is empty, the root already exists
    if (path.empty())
    {
      return true;
    }

    struct stat st;
    if (stat(path.c_str(), &st) == 0)
      return true; // let the caller deal with non-directories

    if (!DiskFile::CreateParentDirectory(path))
      return false;

    if (mkdir(path.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH))
    {
      Failure(*this, ecFileCreateFailed, path) << "Could not create the " << path << " directory: " << strerror(errno);
      return false;
    }
  }
  return true;
}

// Create new file on disk and make sure that there is enough
// space on disk for it.
bool DiskFile::Create(std::string _filename, u64 _filesize)
{
  assert(fd < 0);

  filename = _filename;
  filesize = _filesize;

  if (!DiskFile::CreateParentDirectory(filename))
    return false;

  if (_filesize > (u64)MaxOffset)
  {
    Failure(*this, ecFileCreateFailed, _filename) << "Requested file size for " << _filename << " is too large.";
    return false;
  }

  fd = open(_filename.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd < 0)
  {
    Failure(*this, ecFileCreateFailed, _filename) << "Could not create " << _filename << ": " << strerror(errno);

    return false;
  }

  if (ftruncate(fd, (OffsetType)_filesize))
  {
    Failure(*this, ecFileCreateFailed, _filename) << "Could not set end of file of " << _filename << ": " << strerror(errno);

    close(fd);
    fd = -1;
    ::remove(filename.c_str());
    return false;
  }

  exists = true;
  return true;
}

// Write some data to disk

bool DiskFile::Write(u64 _offset, const void *buffer, size_t length, LengthType maxlength)
{
  assert(fd >= 0);

  if (_offset > (u64)MaxOffset)
  {
    Failure(*this, ecFileWriteFailed, filename) << "Could not write " << (u64)length << " bytes to " << filename << " at offset " << _offset;
    return false;
  }

  while (length > 0) {

    LengthType write;
    if (length > maxlength)
      write = maxlength;
    else
      write = length;

    ssize_t wrote = pwrite(fd, buffer, write, (OffsetType)_offset);
    if (wrote < 0 && errno == EINTR)
      continue;

    if (wrote < 0)
    {
      Failure(*this, ecFileWriteFailed, filename) << "Could not write " << (u64)length << " bytes to " << filename << " at offset " << _offset << ": " << strerror(errno);
      return false;
    }

    if (wrote == 0)
    {
      Failure(*this, ecFileWriteFailed, filename) << "Could not write " << (u64)length << " bytes to " << filename << " at offset " << _offset << ": nothing was written.";
      return false;
    }

    _offset += wrote;
    length -= wrote;
    buffer = ((char *) buffer) + wrote;

    if (filesize < _offset)
    {
      filesize = _offset;
    }
  }

  return true;
}

// Open the file

bool DiskFile::Open(const std::string &_filename, u64 _filesize)
{
  assert(fd < 0);

  filename = _filename;
  filesize = _filesize;

  if (_filesize > (u64)MaxOffset)
  {
    Failure(*this, ecFileOpenFailed, _filename) << "File size for " << _filename << " is too large.";
    return false;
  }

  fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
  {
    if (errno != ENOENT && errno != ENOTDIR)
      Failure(*this, ecFileOpenFailed, _filename) << "Could not open " << _filename << ": " << strerror(errno);

    return false;
  }

  struct stat st;
  modified = (0 == fstat(fd, &st)) ? ModifiedTime(st) : 0;

  exists = true;

  return true;
}

// Open an existing file for both reading and writing

bool DiskFile::OpenForUpdate(std::string *reason)
{
  assert(fd < 0);

  auto refuse = [&](const std::string &why)
  {
    if (fd >= 0)
      close(fd);
    fd = -1;
    if (reason)
      *reason = why;
    return false;
  };

  if (filename.empty())
    return refuse("its name cannot be used");
  if (filesize > (u64)MaxOffset)
    return refuse("it is too large");

  // No O_CREAT or O_TRUNC, and O_NOFOLLOW as in Create
  fd = open(filename.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return refuse(strerror(errno));

  struct stat st;
  if (fstat(fd, &st) != 0)
    return refuse(strerror(errno));
  if (!S_ISREG(st.st_mode))
    return refuse("it is not a regular file");
  if (st.st_nlink != 1)
    return refuse("it has more than one hard link");
  if ((u64)st.st_size != filesize || ModifiedTime(st) != modified)
    return refuse("it changed since it was verified");

  exists = true;

  return true;
}

// Read some data from disk

bool DiskFile::Read(u64 _offset, void *buffer, size_t length, LengthType maxlength)
{
  assert(fd >= 0);

  if (_offset > (u64)MaxOffset)
  {
    Failure(*this, ecFileReadFailed, filename) << "Could not read " << (u64)length << " bytes from " << filename << " at offset " << _offset;
    return false;
  }

  while (length > 0) {

    LengthType want;
    if (length > maxlength)
      want = maxlength;
    else
      want = length;

    ssize_t got = pread(fd, buffer, want, (OffsetType)_offset);
    if (got < 0 && errno == EINTR)
      continue;

    if (got < 0)
    {
      Failure(*this, ecFileReadFailed, filename) << "Could not read " << (u64)length << " bytes from " << filename << " at offset " << _offset << ": " << strerror(errno);
      return false;
    }

    if (got == 0)
    {
      Failure(*this, ecFileReadFailed, filename) << "Could not read " << (u64)length << " bytes from " << filename << " at offset " << _offset << ": unexpected end of file.";
      return false;
    }

    _offset += got;
    length -= got;
    buffer = ((char *) buffer) + got;
  }

  return true;
}

bool DiskFile::Close(void)
{
  bool closed = true;

  if (fd >= 0)
  {
    closed = (0 == close(fd));
    fd = -1;
  }

  return closed;
}

// Attempt to get the full pathname of the file
std::string DiskFile::GetCanonicalPathname(std::string filename)
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

  std::string result = work;
  delete [] work;

  return result;
}

// Whether name matches a shell-style wildcard, where '*' matches any run of
// characters, none included, and '?' matches exactly one, anywhere in it
static bool MatchesWildcard(const std::string &name, const std::string &wildcard)
{
  std::string::size_type n = 0, w = 0;
  std::string::size_type restart_n = std::string::npos, restart_w = std::string::npos;

  while (n < name.size())
  {
    if (w < wildcard.size() && (wildcard[w] == '?' || wildcard[w] == name[n]))
    {
      // Literal character or '?' matched one character of the name
      ++n;
      ++w;
    }
    else if (w < wildcard.size() && wildcard[w] == '*')
    {
      // A '*' matches nothing at first, and is where matching starts again
      // when what follows it does not match
      restart_w = w++;
      restart_n = n;
    }
    else if (restart_w != std::string::npos)
    {
      // The last '*' has to absorb one more character of the name
      w = restart_w + 1;
      n = ++restart_n;
    }
    else
    {
      return false;
    }
  }

  // Only trailing '*'s can still be unmatched in the pattern
  while (w < wildcard.size() && wildcard[w] == '*')
    ++w;

  return w == wildcard.size();
}

std::unique_ptr< std::list<std::string> > DiskFile::FindFiles(std::string path, std::string wildcard, bool recursive, bool followlinks)
{
  // check path, if not ending with path separator, add one
  char pathend = *path.rbegin();
  if (pathend != '/')
  {
    path += '/';
  }
  std::list<std::string> *matches = new std::list<std::string>;

  if (wildcard.find_first_of("*?") != std::string::npos)
  {
    DIR *dirp = opendir(path.c_str());
    if (dirp != 0)
    {
      struct dirent *d;
      while ((d = readdir(dirp)) != 0)
      {
        std::string name = d->d_name;

        if (name == "." || name == "..")
          continue;

        const bool matched = MatchesWildcard(name, wildcard);
        if (matched || recursive)
        {
          struct stat st;
          std::string fn = path + name;
          if (lstat(fn.c_str(), &st) == 0)
          {
            if (S_ISDIR(st.st_mode) &&
                recursive == true)
            {
              // A directory which matches is taken whole, and one which does
              // not is searched for files which do
              std::string nwwildcard = matched ? "*" : wildcard;
              std::unique_ptr< std::list<std::string> > dirmatches(
                               DiskFile::FindFiles(fn, nwwildcard, true, followlinks)
                               );
              matches->splice(matches->end(), *dirmatches);
            }
            else if (matched && S_ISREG(st.st_mode))
            {
              matches->push_back(path + name);
            }
            else if (matched && followlinks && S_ISLNK(st.st_mode))
            {
              struct stat stt;
              if (stat(fn.c_str(), &stt) == 0 && S_ISREG(stt.st_mode))
              {
                matches->push_back(path + name);
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
    std::string fn = path + wildcard;
    if (lstat(fn.c_str(), &st) == 0)
    {
      if (S_ISDIR(st.st_mode) &&
          recursive == true)
      {
        std::string nwwildcard="*";
	std::unique_ptr< std::list<std::string> > dirmatches(
						 DiskFile::FindFiles(fn, nwwildcard, true, followlinks)
						 );

        matches->splice(matches->end(), *dirmatches);
      }
      else if (S_ISREG(st.st_mode))
      {
        matches->push_back(path + wildcard);
      }
      else if (followlinks && S_ISLNK(st.st_mode))
      {
        struct stat stt;
        if (stat(fn.c_str(), &stt) == 0 && S_ISREG(stt.st_mode))
        {
          matches->push_back(path + wildcard);
        }
      }
    }
  }

  return std::unique_ptr< std::list<std::string> >(matches);
}

u64 DiskFile::GetFileSize(std::string filename)
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

bool DiskFile::FileExists(std::string filename)
{
  struct stat st;
  return ((0 == stat(filename.c_str(), &st)) && (0 != (st.st_mode & S_IFREG)));
}

bool DiskFile::DirectoryExists(std::string path)
{
  struct stat st;
  return ((0 == stat(path.c_str(), &st)) && S_ISDIR(st.st_mode));
}

std::unique_ptr< std::list<std::string> > DiskFile::FindDirectories(std::string path)
{
  if (*path.rbegin() != '/')
    path += '/';

  std::list<std::string> *matches = new std::list<std::string>;

  DIR *dirp = opendir(path.c_str());
  if (dirp != 0)
  {
    struct dirent *d;
    while ((d = readdir(dirp)) != 0)
    {
      const std::string name = d->d_name;
      if (name == "." || name == "..")
        continue;

      struct stat st;
      if (lstat((path + name).c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        matches->push_back(path + name);
    }
    closedir(dirp);
  }

  return std::unique_ptr< std::list<std::string> >(matches);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

bool DiskFile::Open(void)
{
  std::string _filename = filename;

  return Open(_filename);
}

bool DiskFile::Open(const std::string &_filename)
{
  return Open(_filename, GetFileSize(_filename));
}

// Delete the file

bool DiskFile::Delete(void)
{
#ifdef _WIN32
  assert(hFile == INVALID_HANDLE_VALUE);

  std::wstring wfilename;
  if (!filename.empty() && utf8::Utf8ToWide(filename, wfilename) && ::DeleteFileW(wfilename.c_str()))
  {
    exists = false;
    return true;
  }
#else
  assert(fd < 0);

  if (filename.size() > 0 && 0 == unlink(filename.c_str()))
  {
    exists = false;
    return true;
  }
#endif
  else
  {
    return false;
  }
}

//std::string DiskFile::GetPathFromFilename(std::string filename)
//{
//  std::string::size_type where;
//
//  if (std::string::npos != (where = filename.find_last_of('/')) ||
//      std::string::npos != (where = filename.find_last_of('\\')))
//  {
//    return filename.substr(0, where+1);
//  }
//  else
//  {
//    return "." PATHSEP;
//  }
//}

void DiskFile::SplitFilename(std::string filename, std::string &path, std::string &name)
{
  std::string::size_type where;

  if (std::string::npos != (where = filename.find_last_of(PATHSEP)) ||
      std::string::npos != (where = filename.find_last_of(ALTPATHSEP)))
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

void DiskFile::SplitRelativeFilename(std::string filename, std::string basepath, std::string &name)
{
  name = filename;
  name.erase(0, basepath.length());
}

#ifdef _WIN32
bool DiskFile::Rename(void)
{
  u32 index = 0;
  std::string newname;
  std::wstring wnewname;
  struct _stati64 st;

  do
  {
    // Build the new filename with index suffix
    newname = filename + "." + std::to_string(++index);

    // Check path length against maximum
    if (newname.length() > _MAX_PATH)
    {
      Failure(*this, ecFileRenameFailed, filename) << filename << " pathlength is more than " << _MAX_PATH << ".";
      return false;
    }

    if (!utf8::Utf8ToWide(newname, wnewname))
    {
      Failure(*this, ecFileRenameFailed, filename) << "Could not convert \"" << newname << "\" to a wide string.";
      return false;
    }

    // Check if file exists using wide-character stat
  } while (_wstati64(wnewname.c_str(), &st) == 0);

  return Rename(newname);
}
#else
bool DiskFile::Rename(void)
{
  u32 index = 0;
  std::string newname;
  struct stat st;

  do
  {
    // Build the new filename with index suffix
    newname = filename + "." + std::to_string(++index);

    // Check path length against maximum
    if (newname.length() > _MAX_PATH)
    {
      Failure(*this, ecFileRenameFailed, filename) << filename << " pathlength is more than " << _MAX_PATH << ".";
      return false;
    }
  } while (stat(newname.c_str(), &st) == 0);

  return Rename(newname);
}
#endif

#ifdef _WIN32
std::string DiskFile::ErrorMessage(DWORD error)
{
  LPVOID lpMsgBuf;
  if (::FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL,
                       error,
                       MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       (LPWSTR)&lpMsgBuf,
                       0,
                       NULL))
  {
    std::string converted;
    const bool ok = utf8::WideToUtf8((wchar_t*)lpMsgBuf, converted);
    LocalFree(lpMsgBuf);

    if (ok)
    {
      return converted;
    }
  }

  char message[40];
  snprintf(message, sizeof(message), "Unknown error code (%lu)", error);

  return message;
}

bool DiskFile::MoveTo(std::string _filename, std::string &reason)
{
  assert(hFile == INVALID_HANDLE_VALUE);

  std::wstring wfilename, _wfilename;
  if (!utf8::Utf8ToWide(filename, wfilename) || !utf8::Utf8ToWide(_filename, _wfilename))
  {
    reason = filename + " cannot be renamed to " + _filename + ": the name could not be converted to a wide string";
    return false;
  }

  if (!::MoveFileW(wfilename.c_str(), _wfilename.c_str()))
  {
    reason = filename + " cannot be renamed to " + _filename + ": " + ErrorMessage(::GetLastError());
    return false;
  }

  filename.swap(_filename);

  return true;
}
#else
bool DiskFile::MoveTo(std::string _filename, std::string &reason)
{
  assert(fd < 0);

  if (::rename(filename.c_str(), _filename.c_str()) != 0)
  {
    reason = filename + " cannot be renamed to " + _filename + ": " + strerror(errno);
    return false;
  }

  filename.swap(_filename);

  return true;
}
#endif

bool DiskFile::Rename(std::string _filename)
{
  if (!CreateParentDirectory(_filename))
    return false;

  std::string reason;
  if (MoveTo(_filename, reason))
    return true;

  Failure(*this, ecFileRenameFailed, filename) << reason;

  return false;
}

bool DiskFile::Rename(std::string _filename, std::string &reason)
{
  if (!CreateParentDirectory(_filename))
  {
    reason = "Could not create the directory for " + _filename;
    return false;
  }

  return MoveTo(_filename, reason);
}

DiskFileMap::DiskFileMap(void)
{
}

DiskFileMap::~DiskFileMap(void)
{
  std::map<std::string, DiskFile*>::iterator fi = diskfilemap.begin();
  while (fi != diskfilemap.end())
  {
    delete (*fi).second;

    ++fi;
  }
}

bool DiskFileMap::Insert(DiskFile *diskfile)
{
  std::string filename = diskfile->FileName();
  assert(filename.length() != 0);

  std::pair<std::map<std::string,DiskFile*>::const_iterator,bool> location = diskfilemap.insert(std::pair<std::string,DiskFile*>(filename, diskfile));

  return location.second;
}

void DiskFileMap::Remove(DiskFile *diskfile)
{
  std::string filename = diskfile->FileName();
  assert(filename.length() != 0);

  diskfilemap.erase(filename);
}

DiskFile* DiskFileMap::Find(std::string filename) const
{
  assert(filename.length() != 0);

  std::map<std::string, DiskFile*>::const_iterator f = diskfilemap.find(filename);

  return (f != diskfilemap.end()) ?  f->second : 0;
}

std::vector<DiskFile*> DiskFileMap::Files(void) const
{
  std::vector<DiskFile*> files;

  for (const auto &f : diskfilemap)
    files.push_back(f.second);

  return files;
}


FileSizeCache::FileSizeCache()
{
}

u64 FileSizeCache::get(const std::string &filename) {
  std::map<std::string, u64>::const_iterator f = cache.find(filename);
  if (f != cache.end())
    return f->second;

  // go to disk
  u64 filesize = DiskFile::GetFileSize(filename);

  cache.insert(std::pair<std::string,u64>(filename, filesize));
  //  std::pair<std::map<std::string,u64>::const_iterator,bool> location = cache.insert(std::pair<std::string,u64>(filename, filesize));
  //  if (!location.second) {
  //    throw exception?
  //  }
  return filesize;
}

} // namespace par2
