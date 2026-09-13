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
#include <iostream>
#include <stdexcept>

#include "libpar2internal.h"
#include "taskpool.h"


// Counts how many times each value in [first, last) was run, and reports any
// value which was not run exactly once
static int check_counts(const char *what,
                        const std::vector<std::atomic<int> > &counts,
                        size_t first,
                        size_t last)
{
  for (size_t i = 0; i < counts.size(); i++)
  {
    const int expected = (i >= first && i < last) ? 1 : 0;
    if (counts[i].load() != expected)
    {
      std::cerr << what << ": value " << i << " ran " << counts[i].load()
                << " times, expected " << expected << std::endl;
      return 1;
    }
  }

  return 0;
}


// Every value of a batch runs exactly once, whatever the thread count
int test1() {
  const size_t size = 1000;

  for (u32 numthreads = 1; numthreads <= 8; numthreads++)
  {
    std::vector<std::atomic<int> > counts(size);
    for (size_t i = 0; i < size; i++)
      counts[i].store(0);

    TaskPool pool(numthreads);

    auto body = [&](size_t value) { counts[value].fetch_add(1); };

    TaskPool::Batch batch;
    pool.Submit(batch, 0, size, body);
    pool.Wait(batch);

    if (check_counts("test1", counts, 0, size))
      return 1;
  }

  return 0;
}


// Only the values of the range asked for are run
int test2() {
  const size_t size = 64;

  std::vector<std::atomic<int> > counts(size);
  for (size_t i = 0; i < size; i++)
    counts[i].store(0);

  TaskPool pool(4);

  auto body = [&](size_t value) { counts[value].fetch_add(1); };

  TaskPool::Batch batch;
  pool.Submit(batch, 16, 48, body);
  pool.Wait(batch);

  return check_counts("test2", counts, 16, 48);
}


// An empty range finishes without running anything
int test3() {
  TaskPool pool(4);

  std::atomic<int> runs(0);
  auto body = [&](size_t) { runs.fetch_add(1); };

  TaskPool::Batch batch;

  pool.Submit(batch, 2, 2, body);
  pool.Wait(batch);

  pool.Submit(batch, 3, 1, body);
  pool.Wait(batch);

  if (runs.load() != 0)
  {
    std::cerr << "test3: an empty range ran " << runs.load() << " values" << std::endl;
    return 1;
  }

  return 0;
}


// A batch can be submitted again once it has been waited for
int test4() {
  const size_t size = 100;
  const int rounds = 20;

  std::vector<std::atomic<int> > counts(size);
  for (size_t i = 0; i < size; i++)
    counts[i].store(0);

  TaskPool pool(4);

  auto body = [&](size_t value) { counts[value].fetch_add(1); };

  TaskPool::Batch batch;

  for (int round = 0; round < rounds; round++)
  {
    pool.Submit(batch, 0, size, body);
    pool.Wait(batch);
  }

  for (size_t i = 0; i < size; i++)
  {
    if (counts[i].load() != rounds)
    {
      std::cerr << "test4: value " << i << " ran " << counts[i].load()
                << " times, expected " << rounds << std::endl;
      return 1;
    }
  }

  return 0;
}


// Work submitted from many threads at once is shared between them all, even
// when there are more threads submitting than the pool has
int test5() {
  const size_t submitters = 8;
  const size_t size = 500;

  for (u32 numthreads = 1; numthreads <= 4; numthreads++)
  {
    std::vector<std::atomic<int> > counts(submitters * size);
    for (size_t i = 0; i < counts.size(); i++)
      counts[i].store(0);

    TaskPool pool(numthreads);

    std::vector<std::thread> threads;
    for (size_t submitter = 0; submitter < submitters; submitter++)
    {
      threads.emplace_back([&, submitter] {
        auto body = [&](size_t value) { counts[value].fetch_add(1); };

        TaskPool::Batch batch;
        pool.Submit(batch, submitter * size, (submitter + 1) * size, body);
        pool.Wait(batch);
      });
    }

    for (std::vector<std::thread>::iterator thread = threads.begin(); thread != threads.end(); ++thread)
      thread->join();

    if (check_counts("test5", counts, 0, counts.size()))
      return 1;
  }

  return 0;
}


// A thread waiting for its own batch runs the work which is still queued, so
// two values which have to run at the same time can, even on a pool of one
int test6() {
  TaskPool pool(1);

  // The work can only be shared with the thread waiting for it if the pool
  // managed to start a thread of its own
  if (pool.ThreadCount() < 1)
    return 0;

  std::mutex mutex;
  std::condition_variable arrived;
  int waiting = 0;
  bool timedout = false;

  auto body = [&](size_t) {
    std::unique_lock<std::mutex> lock(mutex);

    if (++waiting == 2)
    {
      arrived.notify_all();
      return;
    }

    if (!arrived.wait_for(lock, std::chrono::seconds(30), [&]{ return waiting == 2; }))
    {
      // Let the other value out rather than leaving the test hanging
      timedout = true;
      waiting = 2;
      arrived.notify_all();
    }
  };

  TaskPool::Batch batch;
  pool.Submit(batch, 0, 2, body);
  pool.Wait(batch);

  if (timedout)
  {
    std::cerr << "test6: the thread waiting for the batch did not run any of it" << std::endl;
    return 1;
  }

  return 0;
}


// A failure of the body is rethrown by Wait, which still returns once every
// value the batch had out has finished
//
// How many values run before the failure is recorded is not fixed: the threads
// which have already claimed one run it whatever happens, so only the values
// no thread had reached are abandoned. What is checked here is that the batch
// finishes rather than hanging, which is what wrong accounting in Abandon
// would cost, and that it is left fit to be submitted again.
int test7() {
  const size_t size = 200;

  for (u32 numthreads = 1; numthreads <= 4; numthreads++)
  {
    TaskPool pool(numthreads);

    std::atomic<int> runs(0);
    auto body = [&](size_t value) {
      runs.fetch_add(1);
      if (value == 0)
        throw std::runtime_error("failed");
    };

    TaskPool::Batch batch;
    pool.Submit(batch, 0, size, body);

    bool caught = false;
    try
    {
      pool.Wait(batch);
    }
    catch (const std::runtime_error &)
    {
      caught = true;
    }

    if (!caught)
    {
      std::cerr << "test7: the failure was not rethrown" << std::endl;
      return 1;
    }

    // The batch is usable again, and no longer holds the failure
    runs.store(0);
    auto clean = [&](size_t) { runs.fetch_add(1); };
    pool.Submit(batch, 0, 10, clean);
    pool.Wait(batch);

    if (runs.load() != 10)
    {
      std::cerr << "test7: " << runs.load() << " values ran after the failure, expected 10" << std::endl;
      return 1;
    }
  }

  return 0;
}


// Several batches from one thread can be in flight at once, and each is waited
// for on its own
int test8() {
  const size_t size = 200;

  std::vector<std::atomic<int> > counts(2 * size);
  for (size_t i = 0; i < counts.size(); i++)
    counts[i].store(0);

  TaskPool pool(4);

  auto body = [&](size_t value) { counts[value].fetch_add(1); };

  TaskPool::Batch first;
  TaskPool::Batch second;

  pool.Submit(first, 0, size, body);
  pool.Submit(second, size, 2 * size, body);

  pool.Wait(second);
  pool.Wait(first);

  return check_counts("test8", counts, 0, counts.size());
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
  if (test6()) {
    std::cerr << "FAILED: test6" << std::endl;
    return 1;
  }
  if (test7()) {
    std::cerr << "FAILED: test7" << std::endl;
    return 1;
  }
  if (test8()) {
    std::cerr << "FAILED: test8" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: taskpool_test complete." << std::endl;

  return 0;
}
