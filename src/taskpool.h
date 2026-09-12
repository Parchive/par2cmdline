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

#ifndef __TASKPOOL_H__
#define __TASKPOOL_H__

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

// A fixed set of threads which run work submitted to them from anywhere, so
// that work submitted from several places at once is shared out between all of
// them rather than each place being given a share of the threads.
//
// Submitting does not wait for the work to run, and a thread which waits for
// a submission of its own runs queued work while it waits. The pool's own
// threads wait for nothing but the queue, so a thread which is waiting for its
// work can never leave the queue unattended.
class TaskPool
{
public:
  // One submission of work. The thread which submits it owns it, and must keep
  // both it and the body it was given alive until Wait has returned.
  class Batch
  {
    friend class TaskPool;

  public:
    Batch()
    : fn(nullptr)
    , call(nullptr)
    , next(0)
    , end(0)
    , remaining(0) {
    }

    Batch(const Batch &) = delete;
    Batch& operator=(const Batch &) = delete;

  private:
    void  *fn;                              // the body, held without owning it
    void (*call)(void *fn, size_t value);   // calls the body for one value
    size_t next;                            // the next value a thread may claim
    size_t end;                             // one past the last value to run
    size_t remaining;                       // values which have not finished
    std::exception_ptr error;               // the first failure of the body
  };

  explicit TaskPool(const size_t numthreads)
  : stopping(false) {
    const size_t wanted = std::max<size_t>(1, numthreads);
    threads.reserve(wanted);

    for (size_t i = 0; i < wanted; ++i)
    {
      try
      {
        threads.emplace_back(&TaskPool::Serve, this);
      }
      catch (const std::system_error &)
      {
        // The work runs on however many threads could be started, and on the
        // threads waiting for it
        break;
      }
    }
  }

  ~TaskPool()
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    haswork.notify_all();

    for (auto & thread : threads)
      thread.join();
  }

  TaskPool(const TaskPool &) = delete;
  TaskPool& operator=(const TaskPool &) = delete;

  size_t ThreadCount() const {return threads.size();}

  // Queues fn to be called for every value in [first, last) and returns
  // without waiting for any of them to run. The batch must be handed to Wait
  // before either it or fn goes away.
  template<typename Fn>
  void Submit(Batch &batch, const size_t first, const size_t last, Fn &fn)
  {
    {
      std::lock_guard<std::mutex> lock(mutex);

      batch.fn = &fn;
      batch.call = &Invoke<Fn>;
      batch.next = first;
      batch.end = last;
      batch.remaining = last > first ? last - first : 0;
      batch.error = std::exception_ptr();

      if (batch.remaining == 0)
        return;

      queue.push_back(&batch);
    }

    haswork.notify_all();

    // A thread waiting for a batch of its own runs whatever is queued while it
    // waits, so it is woken for this one as well as the pool's own threads
    batchdone.notify_all();
  }

  // Runs queued work until the batch has finished. An exception thrown by the
  // body abandons the rest of that batch and is rethrown here.
  void Wait(Batch &batch)
  {
    std::unique_lock<std::mutex> lock(mutex);

    while (batch.remaining != 0)
    {
      // Nothing is left to help with when every value has been claimed, and
      // whichever threads hold those claims will finish them
      if (!Claim(lock))
        batchdone.wait(lock);
    }

    if (batch.error)
    {
      const std::exception_ptr failure = batch.error;
      batch.error = std::exception_ptr();
      std::rethrow_exception(failure);
    }
  }

private:
  template<typename Fn>
  static void Invoke(void *fn, const size_t value)
  {
    (*static_cast<Fn *>(fn))(value);
  }

  void Serve()
  {
    std::unique_lock<std::mutex> lock(mutex);

    for (;;)
    {
      haswork.wait(lock, [this]{ return stopping || !queue.empty(); });

      if (stopping)
        return;

      Claim(lock);
    }
  }

  // Runs one value of the batch at the head of the queue and returns whether
  // there was one. Called with the lock held, which is dropped while the body
  // runs and held again on return.
  bool Claim(std::unique_lock<std::mutex> &lock)
  {
    if (queue.empty())
      return false;

    Batch *batch = queue.front();
    const size_t value = batch->next++;

    // A batch is queued only while it has values left for a thread to claim
    if (batch->next == batch->end)
      queue.pop_front();

    lock.unlock();

    std::exception_ptr failure;

    try
    {
      batch->call(batch->fn, value);
    }
    catch (...)
    {
      failure = std::current_exception();
    }

    lock.lock();

    if (failure)
    {
      if (!batch->error)
        batch->error = failure;

      Abandon(batch);
    }

    if (--batch->remaining == 0)
      batchdone.notify_all();

    return true;
  }

  // Drops the values of a failed batch which no thread has claimed yet
  void Abandon(Batch *batch)
  {
    if (batch->next == batch->end)
      return;

    const auto queued = std::find(queue.begin(), queue.end(), batch);
    if (queued != queue.end())
      queue.erase(queued);

    batch->remaining -= batch->end - batch->next;
    batch->next = batch->end;
  }

  std::vector<std::thread> threads;
  std::deque<Batch*>       queue;      // batches with values left to claim
  bool                     stopping;
  std::mutex               mutex;
  std::condition_variable  haswork;    // a batch has been queued
  std::condition_variable  batchdone;  // a batch has finished
};

#endif // __TASKPOOL_H__
