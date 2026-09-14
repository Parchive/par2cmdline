//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See https://parchive.sourceforge.net for details of PAR 2.0.
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

#ifndef __ERRORLOG_H__
#define __ERRORLOG_H__

#include <atomic>
#include <mutex>
#include <string>

#include <par2/libpar2.h>

namespace par2
{

// Keeps the first error of an operation and offers every one to an observer.
//
// Errors are recorded from whichever thread found them. The first is kept
// rather than the last, so that a step which reports a failure of its own does
// not replace the more particular one that caused it.
class ErrorLog
{
public:
  ErrorLog(void)
    : mutex()
    , first()
    , observer(0)
  {
    first.code = ecNone;
  }

  // May be set or cleared while work is in progress
  void SetObserver(Par2Observer *_observer)
  {
    observer.store(_observer, std::memory_order_relaxed);
  }

  void Clear(void)
  {
    std::lock_guard<std::mutex> lock(mutex);

    first = Par2Error();
    first.code = ecNone;
  }

  void Record(const ErrorCode code,
              const std::string &message,
              const std::string &filename = std::string())
  {
    Par2Error error;
    error.code = code;
    error.message = message;
    error.filename = filename;

    {
      std::lock_guard<std::mutex> lock(mutex);

      if (ecNone == first.code)
        first = error;
    }

    // Outside the lock, so that an observer may read the log back
    Par2Observer *target = observer.load(std::memory_order_relaxed);
    if (target)
      target->OnError(error);
  }

  // Record this only when nothing has been recorded yet, so that a step which
  // summarises a failure does not repeat what the step below it already said.
  void RecordIfNone(const ErrorCode code,
                    const std::string &message,
                    const std::string &filename = std::string())
  {
    Par2Error error;
    error.code = code;
    error.message = message;
    error.filename = filename;

    {
      std::lock_guard<std::mutex> lock(mutex);

      if (ecNone != first.code)
        return;

      first = error;
    }

    // Outside the lock, so that an observer may read the log back
    Par2Observer *target = observer.load(std::memory_order_relaxed);
    if (target)
      target->OnError(error);
  }

  bool First(Par2Error *error) const
  {
    if (0 == error)
      return false;

    std::lock_guard<std::mutex> lock(mutex);

    if (ecNone == first.code)
      return false;

    *error = first;

    return true;
  }

private:
  mutable std::mutex mutex;
  Par2Error first;
  std::atomic<Par2Observer *> observer;
};

} // namespace par2

#endif // __ERRORLOG_H__
