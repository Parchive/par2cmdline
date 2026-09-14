//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#ifndef __PROGRESSMETER_H__
#define __PROGRESSMETER_H__

#include <chrono>
#include <mutex>

#include <par2/libpar2.h>

namespace par2
{

template<typename TValue>
class ProgressMeter
{
  using steady_clock = std::chrono::steady_clock;
  const std::chrono::milliseconds PRINT_INTERVAL = std::chrono::milliseconds(50);

  std::ostream &sout;        // stream for output (for commandline, this is cout)
  const std::string message; // message to display alongside percentage
  const float scale;         // pre-computed multiplier to convert progress value into a percentage*10
  std::atomic<TValue> current; // last known progress value
  std::atomic<steady_clock::duration::rep> printed; // last time progress was outputted
  const bool print;          // whether the percentage is written to sout
  const Phase phase;         // which step of the work this counts
  Par2Observer *observer;    // notified of progress whatever the noise level
  std::mutex reporting;      // held while a fraction is claimed and reported
  u32 reported;              // highest fraction reported so far

  inline u32 CalcThousandths(TValue val) const
  {
    return (u32)(scale * val + 0.5f);
  }
  inline bool PrintFraction(TValue oldval, TValue newval)
  {
    // if the displayed value won't change, don't print
    u32 newfraction = CalcThousandths(newval);
    if (CalcThousandths(oldval) == newfraction)
      return false;

    // check if enough time has passed
    steady_clock::duration::rep lastprinted = printed.load(std::memory_order_relaxed);
    
    steady_clock::time_point now = steady_clock::now();
    steady_clock::time_point lastpoint = steady_clock::time_point(steady_clock::duration(lastprinted));

    // if enough time has passed, print the current progress, and update the time record
    if (now - lastpoint >= PRINT_INTERVAL || newfraction == 1000)
    {
      // Threads report as they finish, so a larger fraction can arrive first.
      // Nothing is reported which would take the count backwards, and the lock
      // is held across the report so the observer is told in the same order.
      std::lock_guard<std::mutex> lock(reporting);

      if (newfraction <= reported)
        return false;

      reported = newfraction;

      if (print)
        LockedStream(sout) << message << newfraction/10 << '.' << newfraction%10 << "%\r" << std::flush;

      if (observer)
        observer->OnProgress(phase, newfraction);

      printed.store(now.time_since_epoch().count(), std::memory_order_relaxed);
      return true;
    }
    return false;
  }

public:
  ProgressMeter(std::ostream &sout, const std::string &message, TValue total,
                NoiseLevel noiselevel, Phase phase, Par2Observer *observer = 0) :
    sout(sout), message(message), scale(total != 0 ? 1000.0f / total : 0.0f), current(0), printed(0),
    print(noiselevel > nlQuiet), phase(phase), observer(observer), reporting(), reported(0) {}
  ProgressMeter(std::ostream &sout, const char *message, TValue total,
                NoiseLevel noiselevel, Phase phase, Par2Observer *observer = 0) :
    sout(sout), message(message), scale(total != 0 ? 1000.0f / total : 0.0f), current(0), printed(0),
    print(noiselevel > nlQuiet), phase(phase), observer(observer), reporting(), reported(0) {}

  // NOTE: Update() doesn't always update current value, so don't mix it with Add()
  void Update(TValue newval)
  {
    TValue oldval = current.load(std::memory_order_relaxed);
    if (PrintFraction(oldval, newval))
      current.store(newval, std::memory_order_relaxed);
  }
  void Add(TValue amount)
  {
    TValue newval = current.fetch_add(amount, std::memory_order_relaxed) + amount;
    PrintFraction(newval - amount, newval);
  }

  // print a line whilst progress is still running
  void PrintLine(const std::string &line)
  {
    TValue val = current.load(std::memory_order_relaxed);
    u32 fraction = CalcThousandths(val);
    LockedStream(sout) << std::setw(message.size()+7) << std::setfill(' ') << "\r"
      << line << '\n'
      << message << fraction/10 << '.' << fraction%10 << "%\r" << std::flush;
  }
};


} // namespace par2

#endif // __PROGRESSMETER_H__
