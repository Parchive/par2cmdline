//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
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

// The scan window is a whole block wide and slides beyond the end of the
// file, where the data reads as zeros. These tests walk files of every shape
// around the block size with every mixture of Step and Jump, and compare the
// window at each offset against the file data followed by those zeros.

#include <iostream>
#include <vector>

#include "libpar2internal.h"

static const char *testfile = "filechecksummer_test.tmp";

// The data the scan window holds at an offset: the file data, then zeros
static void Window(std::vector<char> &window,
                   const std::vector<char> &data,
                   u64 offset,
                   u64 blocksize)
{
  window.assign((size_t)blocksize, 0);

  for (u64 i = 0; i < blocksize && offset+i < data.size(); i++)
    window[(size_t)i] = data[(size_t)(offset+i)];
}

static bool CheckWindow(FileCheckSummer &checksummer,
                        const std::vector<char> &data,
                        u64 blocksize,
                        std::vector<char> &window)
{
  const u64 offset = checksummer.Offset();
  Window(window, data, offset, blocksize);

  const u32 checksum = ~0 ^ CRCUpdateBlock(~0, (size_t)blocksize, &window[0]);
  if (checksummer.Checksum() != checksum)
  {
    std::cerr << "checksum at offset " << offset << " was " << std::hex
              << checksummer.Checksum() << ", expected " << checksum
              << std::dec << std::endl;
    return false;
  }

  MD5Context context;
  context.Update(&window[0], (size_t)blocksize);
  MD5Hash hash;
  context.Final(hash);
  if (checksummer.Hash() != hash)
  {
    std::cerr << "hash at offset " << offset << " was "
              << checksummer.Hash().print() << ", expected " << hash.print()
              << std::endl;
    return false;
  }

  // The last block of a file is matched by its own length, with the rest of
  // the window taken to be zeros
  const u64 blocklength = checksummer.BlockLength();
  if (checksummer.ShortChecksum(blocklength) != checksum)
  {
    std::cerr << "short checksum at offset " << offset << " was " << std::hex
              << checksummer.ShortChecksum(blocklength) << ", expected "
              << checksum << std::dec << std::endl;
    return false;
  }
  if (checksummer.ShortHash(blocklength) != hash)
  {
    std::cerr << "short hash at offset " << offset << " was "
              << checksummer.ShortHash(blocklength).print() << ", expected "
              << hash.print() << std::endl;
    return false;
  }

  return true;
}

// Walk to the end of the file from startoffset, taking a jump of
// jumpdistance after every steps single byte steps. A jumpdistance of 0
// only steps.
static bool Walk(DiskFile &diskfile,
                 const std::vector<char> &data,
                 const u32 (&windowtable)[256],
                 u64 blocksize,
                 u64 startoffset,
                 u64 steps,
                 u64 jumpdistance,
                 std::vector<char> &window)
{
  FileCheckSummer checksummer(&diskfile, blocksize, windowtable);
  if (!checksummer.Start(startoffset))
  {
    std::cerr << "could not start at offset " << startoffset << std::endl;
    return false;
  }

  u64 stepstaken = 0;
  while (checksummer.Offset() < data.size())
  {
    if (!CheckWindow(checksummer, data, blocksize, window))
    {
      std::cerr << "  filesize " << data.size()
                << ", blocksize " << blocksize
                << ", startoffset " << startoffset
                << ", steps " << steps
                << ", jumpdistance " << jumpdistance << std::endl;
      return false;
    }

    if (jumpdistance > 0 && ++stepstaken > steps)
    {
      stepstaken = 0;

      if (!checksummer.Jump(jumpdistance))
      {
        std::cerr << "could not jump at offset " << checksummer.Offset() << std::endl;
        return false;
      }
    }
    else if (!checksummer.Step())
    {
      std::cerr << "could not step at offset " << checksummer.Offset() << std::endl;
      return false;
    }
  }

  return true;
}

// Every byte is non zero, so that data left behind in the buffer cannot be
// mistaken for the zeros beyond the end of the file
static bool WriteTestFile(std::vector<char> &data, u64 filesize)
{
  data.resize((size_t)filesize);
  for (u64 i = 0; i < filesize; i++)
    data[(size_t)i] = (char)(1 + (i * 37 + (i >> 3)) % 255);

  DiskFile diskfile(std::cout, std::cerr);
  const bool ok = diskfile.Create(testfile, filesize)
                  && (filesize == 0 || diskfile.Write(0, &data[0], (size_t)filesize));
  diskfile.Close();

  if (!ok)
    std::cerr << "could not write " << testfile << std::endl;

  return ok;
}

static bool TestFileSize(u64 blocksize, u64 filesize)
{
  std::vector<char> data;
  if (!WriteTestFile(data, filesize))
    return false;

  u32 windowtable[256];
  GenerateWindowTable(blocksize, windowtable);

  DiskFile diskfile(std::cout, std::cerr);
  if (!diskfile.Open(testfile, filesize))
  {
    std::cerr << "could not open " << testfile << std::endl;
    return false;
  }

  // Jumps of a whole block are what an undamaged file is scanned with, the
  // rest land the buffer at every offset either side of the block boundary
  const u64 jumpdistances[] = {0, 1, 2, 3, blocksize/2, blocksize-1, blocksize};

  std::vector<char> window;
  bool ok = true;

  // Starting at the end of the file leaves nothing to scan
  for (u64 startoffset = 0; ok && startoffset <= filesize; startoffset++)
  {
    for (size_t j = 0; ok && j < sizeof(jumpdistances)/sizeof(jumpdistances[0]); j++)
    {
      for (u64 steps = 0; ok && steps <= 3; steps += 3)
      {
        ok = Walk(diskfile, data, windowtable, blocksize, startoffset,
                  steps, jumpdistances[j], window);
      }
    }
  }

  diskfile.Close();
  remove(testfile);

  return ok;
}

int main()
{
  static constexpr u64 blocksizes[] = {4, 8, 12, 20};

  for (const auto blocksize : blocksizes)
  {
    // Every size from a fraction of a block up to three whole blocks, which
    // covers a short last block at each offset in the two block buffer
    for (u64 filesize = 0; filesize <= 3*blocksize+2; filesize++)
    {
      if (!TestFileSize(blocksize, filesize))
      {
        std::cerr << "FAILED: blocksize " << blocksize
                  << ", filesize " << filesize << std::endl;
        return 1;
      }
    }
  }

  std::cout << "SUCCESS: filechecksummer_test complete." << std::endl;

  return 0;
}
