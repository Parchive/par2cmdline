//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
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

#ifndef __HASHER_H__
#define __HASHER_H__

// Hashes the blocks of one file, and the file itself. Several consecutive
// blocks are submitted at once, so that an implementation may hash them
// alongside each other, and the hashes of the file are advanced by the same
// submissions rather than by a second pass over the data. One instance belongs
// to one thread at a time.
class Hasher
{
public:
  virtual ~Hasher(void) {}

  // Start on a file of filesize bytes whose blocks are blocklength bytes each.
  // The first 16k of the file is always hashed, the whole of it only when
  // wholefile is set.
  virtual bool Init(u64 filesize, size_t blocklength, bool wholefile) = 0;

  // Submit count consecutive blocks, in file order, laid out end to end in
  // data. Where the file runs out part way through the last of them the
  // remainder is already zero, because a block is hashed padded out to
  // blocklength. filelength is how many of the submitted bytes belong to the
  // file and advances its hashes; zero leaves them alone, for a caller which
  // hashes the file elsewhere.
  virtual void SubmitBlocks(const void *data, u32 count, size_t filelength) = 0;

  // Take the hashes of the blocks submitted so far, 20 bytes each in
  // submission order: 16 bytes of MD5 then the CRC32 little endian, which is
  // the layout of a verification packet entry.
  virtual void CollectBlocks(void *out, u32 count) = 0;

  // Submit blocks as SubmitBlocks does, but only far enough to say whether each
  // one hashes to the 20 bytes at expected for it, setting matched[index] to 1
  // or 0. An implementation may stop on a block as soon as it knows, and need
  // not keep what it computed: CollectBlocks does not follow this.
  virtual void CheckBlocks(const void *data, u32 count, size_t filelength,
                           const void *expected, char *matched) = 0;

  // Take the hashes of the file, 16 bytes each. The full hash is only set for
  // a hasher asked for it, or a file no larger than 16k.
  virtual void EndFile(void *hashfull, void *hash16k) = 0;
};

#endif // __HASHER_H__
