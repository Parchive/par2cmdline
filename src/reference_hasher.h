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

#ifndef __REFERENCE_HASHER_H__
#define __REFERENCE_HASHER_H__

// Hashes each block of a batch on its own, and the file alongside them.
class ReferenceHasher : public Hasher
{
public:
  ReferenceHasher(void)
    : filesize(0)
    , blocklength(0)
    , fileoffset(0)
    , filehasher(true)
  {
  }

  bool Init(u64 _filesize, size_t _blocklength, bool wholefile)
  {
    filesize = _filesize;
    blocklength = _blocklength;
    fileoffset = 0;
    filehasher = FileHasher(wholefile);
    results.clear();

    return blocklength > 0;
  }

  void SubmitBlocks(const void *data, u32 count, size_t filelength)
  {
    const u8 *at = (const u8*)data;

    for (u32 block=0; block<count; block++)
    {
      const size_t was = results.size();
      results.resize(was + 20);

      BlockHash(at, &results[was]);
      BlockCRC(at, &results[was + 16]);

      at += blocklength;
    }

    UpdateFile(data, filelength);
  }

  void CheckBlocks(const void *data, u32 count, size_t filelength,
                   const void *expected, char *matched)
  {
    const u8 *at = (const u8*)data;
    const u8 *want = (const u8*)expected;

    for (u32 block=0; block<count; block++)
    {
      u8 crc[4];
      BlockCRC(at, crc);

      // The crc settles most blocks, so the hash is only worth taking when it
      // does not
      if (memcmp(crc, &want[16], 4) == 0)
      {
        u8 hash[16];
        BlockHash(at, hash);

        matched[block] = memcmp(hash, want, 16) == 0 ? 1 : 0;
      }
      else
      {
        matched[block] = 0;
      }

      at += blocklength;
      want += 20;
    }

    UpdateFile(data, filelength);
  }

  void CollectBlocks(void *out, u32 count)
  {
    const size_t taken = (size_t)count * 20;

    memcpy(out, &results[0], taken);
    results.erase(results.begin(), results.begin() + taken);
  }

  void EndFile(void *hashfull, void *hash16k)
  {
    MD5Hash full;
    MD5Hash first16k;

    filehasher.GetHashes(filesize, full, first16k);

    memcpy(hashfull, full.hash, 16);
    memcpy(hash16k, first16k.hash, 16);
  }

private:
  void BlockHash(const u8 *at, u8 *out)
  {
    MD5Context context;
    context.Update(at, blocklength);

    MD5Hash hash;
    context.Final(hash);

    memcpy(out, hash.hash, 16);
  }

  void BlockCRC(const u8 *at, u8 *out)
  {
    const u32 crc = ~0 ^ CRCUpdateBlock(~0, blocklength, at);

    out[0] = (u8)crc;
    out[1] = (u8)(crc >> 8);
    out[2] = (u8)(crc >> 16);
    out[3] = (u8)(crc >> 24);
  }

  void UpdateFile(const void *data, size_t filelength)
  {
    if (filelength > 0)
    {
      filehasher.Update(fileoffset, data, filelength);
      fileoffset += filelength;
    }
  }

  u64               filesize;
  size_t            blocklength;
  u64               fileoffset;
  FileHasher        filehasher;
  std::vector<u8>   results;
};

#endif // __REFERENCE_HASHER_H__
