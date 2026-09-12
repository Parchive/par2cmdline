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

#ifndef __BUFFERPOOL_H__
#define __BUFFERPOOL_H__

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

// A fixed set of equally sized buffers which the threads using them take from
// and give back to. They all come from one allocation, so that a thread which
// holds several of them, or every one of them, costs no more than the pool
// does and none of them is made and thrown away as the demand for them moves.
class BufferPool
{
public:
  BufferPool()
  : buffers(0)
  , buffersize(0)
  {
  }

  BufferPool(const BufferPool &) = delete;
  BufferPool& operator=(const BufferPool &) = delete;

  // Makes count buffers of size bytes, none of them in use. Every buffer taken
  // from an earlier set is given up, so no thread may still hold one.
  void Reset(const size_t count, const size_t size)
  {
    std::vector<char>(count * size).swap(slab);

    buffers = count;
    buffersize = size;

    unused.clear();
    unused.reserve(count);
    for (size_t index = 0; index < count; ++index)
      unused.push_back(index);
  }

  size_t Count() const {return buffers;}
  size_t Size() const {return buffersize;}

  // The buffer a taken index stands for
  char* At(const size_t index)
  {
    assert(index < buffers);
    return slab.data() + index * buffersize;
  }

  // Takes a buffer, and returns whether there was one which was not in use
  bool TryTake(size_t &index)
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (unused.empty())
      return false;

    index = unused.back();
    unused.pop_back();

    return true;
  }

  // Takes a buffer, waiting for one to be given back if every one of them is
  // in use
  size_t Take()
  {
    std::unique_lock<std::mutex> lock(mutex);

    given.wait(lock, [this]{return !unused.empty();});

    const size_t index = unused.back();
    unused.pop_back();

    return index;
  }

  void Give(const size_t index)
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      unused.push_back(index);
    }

    given.notify_one();
  }

private:
  std::vector<char>       slab;        // every buffer, in one allocation
  size_t                  buffers;
  size_t                  buffersize;
  std::vector<size_t>     unused;      // the buffers which are not in use
  std::mutex              mutex;
  std::condition_variable given;       // a buffer has been given back
};

#endif // __BUFFERPOOL_H__
