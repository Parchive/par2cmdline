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

#ifndef __REFERENCE_PROCESSOR_H__
#define __REFERENCE_PROCESSOR_H__

#include <deque>

// Multiplies each input block by the matrix on numthreads threads, which stay
// alive from one submission to the next, and keeps the accumulated output
// blocks in one buffer. AddInput queues a submission and returns, the block
// being multiplied on the thread this holds while the caller reads the next
// one.
class ReferenceProcessor : public Processor
{
public:
  ReferenceProcessor(ReedSolomon<Galois16> &rs, u32 numthreads)
    : rs(rs)
    , numthreads(numthreads)
    , runner()
    , maxlength(0)
    , outputcount(0)
    , currentlength(0)
    , outputbuffer(0)
    , queue()
    , stopping(false)
    , mutex()
    , haswork()
    , drained()
    , worker()
  {
    worker = std::thread(&ReferenceProcessor::Serve, this);
  }

  ~ReferenceProcessor(void)
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    haswork.notify_one();
    worker.join();

    delete [] outputbuffer;
  }

  bool Init(size_t _maxlength, u32 _outputcount)
  {
    delete [] outputbuffer;

    maxlength = _maxlength;
    outputcount = _outputcount;
    currentlength = _maxlength;
    outputbuffer = new u8[maxlength * outputcount];

    runner.reset(new ParallelRunner(std::min(numthreads, outputcount)));

    return outputbuffer != nullptr;
  }

  void SetChunkLength(size_t length)
  {
    currentlength = length;
  }

  void ResetOutput(void)
  {
    memset(outputbuffer, 0, maxlength * outputcount);
  }

  void WaitForAdd(void)
  {
  }

  std::future<void> AddInput(const void *data, size_t length, u32 inputindex, const u16 *factors)
  {
    (void)inputindex;

    std::future<void> processed;

    {
      std::lock_guard<std::mutex> lock(mutex);

      queue.emplace_back(data, length, factors);
      processed = queue.back().processed.get_future();
    }

    haswork.notify_one();

    return processed;
  }

  void EndInput(void)
  {
    std::unique_lock<std::mutex> lock(mutex);
    drained.wait(lock, [this]{ return queue.empty(); });
  }

  const void *PeekOutput(u32 index)
  {
    return &outputbuffer[maxlength * index];
  }

  bool GetOutput(u32 index, void *out)
  {
    memcpy(out, &outputbuffer[maxlength * index], currentlength);
    return true;
  }

private:
  // One input block waiting to be multiplied, or being multiplied. data and
  // factors belong to the caller, which keeps them until processed is ready.
  struct Submission
  {
    Submission(const void *data, size_t length, const u16 *factors)
      : data(data)
      , length(length)
      , factors(factors)
      , processed()
    {
    }

    const void         *data;
    size_t              length;
    const u16          *factors;
    std::promise<void>  processed;
  };

  void Multiply(const void *data, size_t length, const u16 *factors)
  {
    runner->Run(0, outputcount, [&](size_t outputindex)
    {
      rs.MultiplyAdd(factors[outputindex], length, data, &outputbuffer[maxlength * outputindex]);
    });
  }

  void Serve(void)
  {
    for (;;)
    {
      Submission *job;

      {
        std::unique_lock<std::mutex> lock(mutex);
        haswork.wait(lock, [this]{ return stopping || !queue.empty(); });

        if (stopping)
          return;

        // A submission is left at the front of the queue while it runs, and
        // adding another behind it leaves this reference good
        job = &queue.front();
      }

      std::exception_ptr failure;

      try
      {
        Multiply(job->data, job->length, job->factors);
      }
      catch (...)
      {
        failure = std::current_exception();
      }

      std::promise<void> processed(std::move(job->processed));

      {
        std::lock_guard<std::mutex> lock(mutex);
        queue.pop_front();
      }

      drained.notify_one();

      if (failure)
        processed.set_exception(failure);
      else
        processed.set_value();
    }
  }

  ReedSolomon<Galois16> &rs;
  u32 numthreads;
  std::unique_ptr<ParallelRunner> runner;
  size_t maxlength;
  u32 outputcount;
  size_t currentlength;
  u8 *outputbuffer;

  std::deque<Submission>   queue;    // submissions in flight, oldest first
  bool                     stopping;
  std::mutex               mutex;
  std::condition_variable  haswork;  // a submission has been added
  std::condition_variable  drained;  // the queue has emptied
  std::thread              worker;   // multiplies the submitted blocks
};

#endif // __REFERENCE_PROCESSOR_H__
