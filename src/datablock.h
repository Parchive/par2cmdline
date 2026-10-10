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

#ifndef __DATABLOCK_H__
#define __DATABLOCK_H__

namespace par2
{

class DiskFile;

// A Data Block is a block of data of a specific length at a specific
// offset in a specific file.

// It may be either a block of data in a source file from which recovery
// data is being computed, a block of recovery data in a recovery file, or
// a block in a target file that is being reconstructed.

class DataBlock
{
public:
  DataBlock(void);
  DataBlock(const DataBlock &other);
  DataBlock &operator=(const DataBlock &other);
  ~DataBlock(void);

public:
  // Set the length of the block
  void SetLength(u64 length);

  // Set the location of the block
  void SetLocation(DiskFile *diskfile, u64 offset);
  void ClearLocation(void);

  void SetFilesize(u64 filesize);

public:
  // Check to see if the location of the block has been set
  bool IsSet(void) const;

  // Which disk file is this data block in
  DiskFile* GetDiskFile(void) const;

  // What offset is the block located at
  u64 GetOffset(void) const;

  // What is the length of this block
  u64 GetLength(void) const;

public:
  // Open the disk file if it is not already open (so that it can be read)
  bool Open(void);

  // Read some of the data from disk into memory.
  bool ReadData(u64 position, size_t size, void *buffer);

  // Write some of the data from memory to disk
  bool WriteData(u64 position, size_t size, const void *buffer, size_t &wrote);

protected:
  // Set by one thread while others read it: offset is stored before
  // diskfile, and diskfile is loaded before offset
  std::atomic<DiskFile*> diskfile;  // Which disk file is the block associated with
  std::atomic<u64>       offset;    // What is the file offset
  u64                    length;    // How large is the block
  u64                    filesize;  // How large was the original file
};


// Construct the data block
inline DataBlock::DataBlock(void)
: diskfile(0)
, offset(0)
, length(0)
, filesize(0)
{
}

inline DataBlock::DataBlock(const DataBlock &other)
: diskfile(other.GetDiskFile())
, offset(other.GetOffset())
, length(other.length)
, filesize(other.filesize)
{
}

inline DataBlock &DataBlock::operator=(const DataBlock &other)
{
  offset.store(other.GetOffset(), std::memory_order_relaxed);
  diskfile.store(other.GetDiskFile(), std::memory_order_release);
  length = other.length;
  filesize = other.filesize;
  return *this;
}

// Destroy the data block
inline DataBlock::~DataBlock(void)
{
}

// Set the length of the block
inline void DataBlock::SetLength(u64 _length)
{
  length = _length;
}

inline void DataBlock::SetFilesize(u64 _filesize)
{
  filesize = _filesize;
}

// Set the location of the block
inline void DataBlock::SetLocation(DiskFile *_diskfile, u64 _offset)
{
  offset.store(_offset, std::memory_order_relaxed);
  diskfile.store(_diskfile, std::memory_order_release);
}

// Clear the location of the block
inline void DataBlock::ClearLocation(void)
{
  diskfile.store(0, std::memory_order_release);
  offset.store(0, std::memory_order_relaxed);
}

// Check to see of the location is known
inline bool DataBlock::IsSet(void) const
{
  DiskFile *file = GetDiskFile();

  if (filesize > 0)
  {
    if (file != 0)
    {
      if ((GetOffset() + length) > file->FileSize()
          && filesize > file->FileSize())
      {
        return false;
      }
      else
      {
        return (file != 0);
      }
    }
  }
  return (file != 0);
}

// Which disk file is this data block in
inline DiskFile* DataBlock::GetDiskFile(void) const
{
  return diskfile.load(std::memory_order_acquire);
}

// What offset is the block located at
inline u64 DataBlock::GetOffset(void) const
{
  return offset.load(std::memory_order_relaxed);
}

// What is the length of this block
inline u64 DataBlock::GetLength(void) const
{
  return length;
}

} // namespace par2

#endif // __DATABLOCK_H__
