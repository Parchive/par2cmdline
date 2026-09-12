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


#include <iostream>

#include "libpar2internal.h"
#include "foreach_parallel.h"


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


// Every value of a range runs exactly once, whatever the thread count
int test1() {
  const size_t size = 1000;

  for (u32 numthreads = 1; numthreads <= 8; numthreads++)
  {
    std::vector<std::atomic<int> > counts(size);
    for (size_t i = 0; i < size; i++)
      counts[i].store(0);

    foreach_parallel(0, size, numthreads, [&](size_t value) {
      counts[value].fetch_add(1);
    });

    if (check_counts("test1", counts, 0, size))
    {
      std::cerr << "numthreads = " << numthreads << std::endl;
      return 1;
    }
  }

  return 0;
}


// A range which does not start at zero runs only its own values
int test2() {
  const size_t size = 64;

  std::vector<std::atomic<int> > counts(size);
  for (size_t i = 0; i < size; i++)
    counts[i].store(0);

  foreach_parallel(16, 48, 4, [&](size_t value) {
    counts[value].fetch_add(1);
  });

  return check_counts("test2", counts, 16, 48);
}


// An empty range, a single value, and a backwards range
int test3() {
  const size_t size = 4;

  for (u32 numthreads = 1; numthreads <= 4; numthreads++)
  {
    std::vector<std::atomic<int> > counts(size);
    for (size_t i = 0; i < size; i++)
      counts[i].store(0);

    // Empty
    foreach_parallel(2, 2, numthreads, [&](size_t value) {
      counts[value].fetch_add(1);
    });
    if (check_counts("test3 empty", counts, 0, 0))
      return 1;

    // Backwards, which is also empty
    foreach_parallel(3, 1, numthreads, [&](size_t value) {
      counts[value].fetch_add(1);
    });
    if (check_counts("test3 backwards", counts, 0, 0))
      return 1;

    // A single value
    foreach_parallel(1, 2, numthreads, [&](size_t value) {
      counts[value].fetch_add(1);
    });
    if (check_counts("test3 single", counts, 1, 2))
      return 1;
  }

  return 0;
}


// More threads than there are values to run
int test4() {
  const size_t size = 3;

  std::vector<std::atomic<int> > counts(size);
  for (size_t i = 0; i < size; i++)
    counts[i].store(0);

  foreach_parallel(0, size, 64, [&](size_t value) {
    counts[value].fetch_add(1);
  });

  return check_counts("test4", counts, 0, size);
}


// The vector overload runs the loop over the elements of the collection
int test5() {
  std::vector<std::string> collection;
  collection.push_back("one");
  collection.push_back("two");
  collection.push_back("three");

  std::mutex mutex;
  std::vector<std::string> seen;

  foreach_parallel(collection, 4, [&](const std::string &value) {
    std::lock_guard<std::mutex> lock(mutex);
    seen.push_back(value);
  });

  std::sort(seen.begin(), seen.end());

  if (seen.size() != collection.size())
  {
    std::cerr << "test5: saw " << seen.size() << " values, expected "
              << collection.size() << std::endl;
    return 1;
  }
  if (seen[0] != "one" || seen[1] != "three" || seen[2] != "two")
  {
    std::cerr << "test5: saw " << seen[0] << " " << seen[1] << " " << seen[2] << std::endl;
    return 1;
  }

  return 0;
}


// One runner runs many loops, each value of each of them exactly once
int test6() {
  const size_t size = 50;
  const size_t loops = 100;

  ParallelRunner runner(4);

  for (size_t loop = 0; loop < loops; loop++)
  {
    std::vector<std::atomic<int> > counts(size);
    for (size_t i = 0; i < size; i++)
      counts[i].store(0);

    runner.Run(0, size, [&](size_t value) {
      counts[value].fetch_add(1);
    });

    if (check_counts("test6", counts, 0, size))
    {
      std::cerr << "loop = " << loop << std::endl;
      return 1;
    }
  }

  return 0;
}


// A runner with one thread, and with none, still runs every value
int test7() {
  const size_t size = 16;

  for (u32 numthreads = 0; numthreads <= 1; numthreads++)
  {
    ParallelRunner runner(numthreads);

    std::vector<std::atomic<int> > counts(size);
    for (size_t i = 0; i < size; i++)
      counts[i].store(0);

    runner.Run(0, size, [&](size_t value) {
      counts[value].fetch_add(1);
    });

    if (check_counts("test7", counts, 0, size))
    {
      std::cerr << "numthreads = " << numthreads << std::endl;
      return 1;
    }
  }

  return 0;
}


// An exception thrown by the loop body reaches the caller of Run
int test8() {
  ParallelRunner runner(4);

  try
  {
    runner.Run(0, 64, [&](size_t value) {
      if (value == 0)
        throw std::runtime_error("from the loop body");
    });
  }
  catch (const std::runtime_error &)
  {
    // The runner is still usable afterwards
    std::atomic<int> ran(0);
    runner.Run(0, 8, [&](size_t) { ran.fetch_add(1); });

    if (ran.load() != 8)
    {
      std::cerr << "test8: ran " << ran.load() << " values after the throw, expected 8" << std::endl;
      return 1;
    }

    return 0;
  }

  std::cerr << "test8: the exception did not reach the caller" << std::endl;
  return 1;
}


// resolve_threads takes the default when nothing was asked for, and never
// hands back more than MAX_THREAD_COUNT
int test9() {
  if (resolve_threads(0) != default_threads())
  {
    std::cerr << "test9: resolve_threads(0) = " << resolve_threads(0)
              << ", expected " << default_threads() << std::endl;
    return 1;
  }
  if (default_threads() < 1)
  {
    std::cerr << "test9: default_threads() = " << default_threads() << std::endl;
    return 1;
  }
  if (resolve_threads(3) != 3)
  {
    std::cerr << "test9: resolve_threads(3) = " << resolve_threads(3) << std::endl;
    return 1;
  }
  if (resolve_threads(1000000) != MAX_THREAD_COUNT)
  {
    std::cerr << "test9: resolve_threads(1000000) = " << resolve_threads(1000000)
              << ", expected " << MAX_THREAD_COUNT << std::endl;
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
  if (test9()) {
    std::cerr << "FAILED: test9" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: foreach_parallel_test complete." << std::endl;

  return 0;
}
