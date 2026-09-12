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


#include <chrono>
#include <cstring>
#include <iostream>
#include <set>

#include "libpar2internal.h"
#include "bufferpool.h"


// A pool hands out every buffer it was made with, and no more than that
int test1() {
  const size_t count = 8;

  BufferPool pool;
  pool.Reset(count, 1024);

  if (pool.Count() != count || pool.Size() != 1024)
  {
    std::cerr << "test1: the pool is " << pool.Count() << " buffers of "
              << pool.Size() << " bytes" << std::endl;
    return 1;
  }

  std::set<size_t> taken;
  for (size_t i = 0; i < count; i++)
  {
    size_t index;
    if (!pool.TryTake(index))
    {
      std::cerr << "test1: only " << i << " of " << count << " buffers could be taken" << std::endl;
      return 1;
    }

    if (!taken.insert(index).second)
    {
      std::cerr << "test1: buffer " << index << " was handed out twice" << std::endl;
      return 1;
    }
  }

  size_t index;
  if (pool.TryTake(index))
  {
    std::cerr << "test1: a buffer was taken from a pool with none left" << std::endl;
    return 1;
  }

  // Giving one back makes exactly one available again
  pool.Give(*taken.begin());

  if (!pool.TryTake(index))
  {
    std::cerr << "test1: the buffer which was given back could not be taken" << std::endl;
    return 1;
  }

  if (pool.TryTake(index))
  {
    std::cerr << "test1: two buffers came back from one Give" << std::endl;
    return 1;
  }

  return 0;
}


// The buffers do not overlap, and each is the size the pool was made with
int test2() {
  const size_t count = 6;
  const size_t size = 512;

  BufferPool pool;
  pool.Reset(count, size);

  std::vector<size_t> taken;
  std::vector<char*>  at;

  for (size_t i = 0; i < count; i++)
  {
    size_t index;
    if (!pool.TryTake(index))
    {
      std::cerr << "test2: only " << i << " buffers could be taken" << std::endl;
      return 1;
    }

    taken.push_back(index);
    at.push_back(pool.At(index));
    memset(at.back(), (int)i + 1, size);
  }

  // Writing the whole of every buffer did not disturb any other one
  for (size_t i = 0; i < count; i++)
  {
    for (size_t byte = 0; byte < size; byte++)
    {
      if (at[i][byte] != (char)(i + 1))
      {
        std::cerr << "test2: buffer " << i << " byte " << byte << " was overwritten" << std::endl;
        return 1;
      }
    }
  }

  return 0;
}


// Take waits for a buffer when every one of them is in use
int test3() {
  BufferPool pool;
  pool.Reset(1, 64);

  size_t held;
  if (!pool.TryTake(held))
  {
    std::cerr << "test3: the only buffer could not be taken" << std::endl;
    return 1;
  }

  std::atomic<bool> waiting(true);
  std::atomic<size_t> got((size_t)-1);

  std::mutex mutex;
  std::condition_variable reached;
  bool taking = false;

  std::thread taker([&] {
    {
      std::lock_guard<std::mutex> lock(mutex);
      taking = true;
    }
    reached.notify_one();

    const size_t index = pool.Take();
    got.store(index);
    waiting.store(false);
  });

  // Wait for the other thread to be about to take, so that finding it still
  // waiting below says that Take held it up rather than that it had yet to
  // call Take at all
  {
    std::unique_lock<std::mutex> lock(mutex);
    reached.wait(lock, [&]{ return taking; });
  }

  // The buffer is still held, so the other thread cannot have been given one
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  if (!waiting.load())
  {
    std::cerr << "test3: Take returned while every buffer was in use" << std::endl;
    taker.join();
    return 1;
  }

  pool.Give(held);
  taker.join();

  if (got.load() != held)
  {
    std::cerr << "test3: Take gave buffer " << got.load() << ", expected " << held << std::endl;
    return 1;
  }

  return 0;
}


// Buffers taken and given back by many threads at once are never held twice
int test4() {
  const size_t count = 4;
  const size_t threads = 8;
  const size_t rounds = 500;

  BufferPool pool;
  pool.Reset(count, 64);

  std::vector<std::atomic<int> > holders(count);
  for (size_t i = 0; i < count; i++)
    holders[i].store(0);

  std::atomic<int> clashes(0);

  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < threads; worker++)
  {
    workers.emplace_back([&] {
      for (size_t round = 0; round < rounds; round++)
      {
        const size_t index = pool.Take();

        if (holders[index].fetch_add(1) != 0)
          clashes.fetch_add(1);

        // The buffer is this thread's alone to write to
        memset(pool.At(index), (int)(round & 0xff), pool.Size());

        holders[index].fetch_sub(1);
        pool.Give(index);
      }
    });
  }

  for (std::vector<std::thread>::iterator worker = workers.begin(); worker != workers.end(); ++worker)
    worker->join();

  if (clashes.load() != 0)
  {
    std::cerr << "test4: a buffer was held by two threads at once " << clashes.load() << " times" << std::endl;
    return 1;
  }

  // Every buffer came back
  for (size_t i = 0; i < count; i++)
  {
    size_t index;
    if (!pool.TryTake(index))
    {
      std::cerr << "test4: only " << i << " of " << count << " buffers were given back" << std::endl;
      return 1;
    }
  }

  return 0;
}


// A pool can be made again with a different set of buffers
int test5() {
  BufferPool pool;

  pool.Reset(2, 128);

  size_t index;
  if (!pool.TryTake(index))
  {
    std::cerr << "test5: the first set of buffers could not be taken" << std::endl;
    return 1;
  }

  pool.Reset(5, 32);

  if (pool.Count() != 5 || pool.Size() != 32)
  {
    std::cerr << "test5: the pool is " << pool.Count() << " buffers of "
              << pool.Size() << " bytes" << std::endl;
    return 1;
  }

  for (size_t i = 0; i < 5; i++)
  {
    if (!pool.TryTake(index))
    {
      std::cerr << "test5: only " << i << " of the new buffers could be taken" << std::endl;
      return 1;
    }
  }

  if (pool.TryTake(index))
  {
    std::cerr << "test5: a buffer was taken from a pool with none left" << std::endl;
    return 1;
  }

  return 0;
}


int main() {
  if (test1()) {
    std::cerr << "FAILED: test1" << std::endl;
    return 1;
  }
  if (test2()) {
    std::cerr << "FAILED: test2" << std::endl;
    return 1;
  }
  if (test3()) {
    std::cerr << "FAILED: test3" << std::endl;
    return 1;
  }
  if (test4()) {
    std::cerr << "FAILED: test4" << std::endl;
    return 1;
  }
  if (test5()) {
    std::cerr << "FAILED: test5" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: bufferpool_test complete." << std::endl;

  return 0;
}
