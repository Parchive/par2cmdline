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

#ifndef __FOREACH_PARALLEL_H__
#define __FOREACH_PARALLEL_H__

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// The largest number of threads a requested thread count is taken up to. Each
// thread costs a stack and a share of the buffers a loop divides between them.
#define MAX_THREAD_COUNT 256

// The number of threads to use when no particular number was asked for
inline u32 default_threads(void)
{
  return std::max(1u, (u32)std::thread::hardware_concurrency());
}

// The number of threads to use for a requested count, where 0 asks for the
// default
inline u32 resolve_threads(const u32 requested)
{
  return requested != 0 ? std::min<u32>(requested, MAX_THREAD_COUNT) : default_threads();
}

// Runs a loop across a fixed set of threads which stay alive from one call to
// the next, so that a loop entered many times pays for its threads once.
class ParallelRunner
{
public:
  // Runs on numthreads threads, one of which is the thread calling Run
  explicit ParallelRunner(const u32 numthreads)
  : threads()
  , job(0)
  , next(0)
  , end(0)
  , generation(0)
  , outstanding(0)
  , stopping(false)
  , mutex()
  , haswork()
  , alldone()
  , error()
  {
    const size_t helpers = numthreads > 1 ? (size_t)numthreads - 1 : 0;
    threads.reserve(helpers);

    for (size_t i = 0; i < helpers; ++i)
    {
      try
      {
        threads.emplace_back(&ParallelRunner::Serve, this);
      }
      catch (const std::system_error &)
      {
        // The loop runs on however many threads could be started
        break;
      }
    }
  }

  ~ParallelRunner(void)
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      ++generation;
    }
    haswork.notify_all();

    for (std::vector<std::thread>::iterator thread = threads.begin(); thread != threads.end(); ++thread)
      thread->join();
  }

  ParallelRunner(const ParallelRunner &) = delete;
  ParallelRunner& operator=(const ParallelRunner &) = delete;

  // Calls fn for every value in [first, last), giving the next value to
  // whichever thread is free, and returns once every value has been run.
  // An exception thrown by fn abandons the rest of the loop and is rethrown
  // here, whichever thread it came from.
  template<typename Fn>
  void Run(const size_t first, const size_t last, Fn &&fn)
  {
    if (threads.empty() || last <= first + 1)
    {
      for (size_t value = first; value < last; ++value)
        fn(value);
      return;
    }

    TypedJob<typename std::remove_reference<Fn>::type> typed(fn);

    {
      std::lock_guard<std::mutex> lock(mutex);
      job = &typed;
      next.store(first, std::memory_order_relaxed);
      end = last;
      outstanding = threads.size();
      ++generation;
    }
    haswork.notify_all();

    Claim(typed);
    Wait();

    if (error)
    {
      std::exception_ptr failure = error;
      error = std::exception_ptr();
      std::rethrow_exception(failure);
    }
  }

private:
  // The loop body for one Run, held without owning it
  class Job
  {
  public:
    virtual ~Job(void) {}
    virtual void Call(size_t value) = 0;
  };

  template<typename Fn>
  class TypedJob : public Job
  {
  public:
    explicit TypedJob(Fn &fn) : fn(fn) {}
    virtual void Call(const size_t value) { fn(value); }
  private:
    Fn &fn;
  };

  void Serve(void)
  {
    u64 seen = 0;

    for (;;)
    {
      Job *running;

      {
        std::unique_lock<std::mutex> lock(mutex);
        haswork.wait(lock, [&]{ return generation != seen; });
        seen = generation;
        if (stopping)
          return;
        running = job;
      }

      Claim(*running);

      {
        std::lock_guard<std::mutex> lock(mutex);
        --outstanding;
      }
      alldone.notify_one();
    }
  }

  void Wait(void)
  {
    std::unique_lock<std::mutex> lock(mutex);
    alldone.wait(lock, [this]{ return outstanding == 0; });
    job = 0;
  }

  void Claim(Job &running)
  {
    for (;;)
    {
      const size_t value = next.fetch_add(1, std::memory_order_relaxed);
      if (value >= end)
        return;

      try
      {
        running.Call(value);
      }
      catch (...)
      {
        // The rest of the loop is abandoned, and the first failure is handed
        // to the thread which called Run
        {
          std::lock_guard<std::mutex> lock(mutex);
          if (!error)
            error = std::current_exception();
        }
        next.store(end, std::memory_order_relaxed);
        return;
      }
    }
  }

  std::vector<std::thread> threads;
  Job                     *job;         // the body of the loop being run
  std::atomic<size_t>      next;        // the next value a thread may claim
  size_t                   end;         // one past the last value to run
  u64                      generation;  // counts the loops handed to the threads
  size_t                   outstanding; // threads still running the loop
  bool                     stopping;
  std::mutex               mutex;
  std::condition_variable  haswork;     // a loop has been handed out
  std::condition_variable  alldone;     // a thread has finished the loop
  std::exception_ptr       error;       // the first failure of the loop body
};

// Calls fn for every value in [first, last), on numthreads threads, giving the
// next value to whichever thread is free. One thread runs them in order.
// A loop entered more than once should hold its own ParallelRunner instead.
template<typename Fn>
void foreach_parallel(const size_t first, const size_t last, const u32 numthreads, Fn &&fn)
{
  const size_t count = last > first ? last - first : 0;

  ParallelRunner runner((u32)std::min<size_t>(numthreads, count));
  runner.Run(first, last, std::forward<Fn>(fn));
}

template<typename T, typename Fn>
void foreach_parallel(const std::vector<T> &collection, u32 numthreads, Fn &&fn)
{
  foreach_parallel(0, collection.size(), numthreads, [&](size_t index) { fn(collection[index]); });
}

#endif // __FOREACH_PARALLEL_H__
