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

template<typename TValue>
class ProgressMeter
{
  using steady_clock = std::chrono::steady_clock;
  const std::chrono::milliseconds INTERVAL = std::chrono::milliseconds(50);

  std::ostream &sout;
  const std::string message;
  const float scale;
  TValue current;
  steady_clock::duration::rep printed;

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
    steady_clock::duration::rep lastprinted;
#if defined(_OPENMP) && _OPENMP >= 201107
    #pragma omp atomic read
#endif
    lastprinted = printed;
    
    steady_clock::time_point now = steady_clock::now();
    steady_clock::time_point lastpoint = steady_clock::time_point(steady_clock::duration(lastprinted));
    // if enough time has passed, print the current progress, and update the time record
    if (now - lastpoint >= INTERVAL || newfraction == 1000)
    {
      #pragma omp critical(stdio)
      sout << message << newfraction/10 << '.' << newfraction%10 << "%\r" << std::flush;
#if defined(_OPENMP) && _OPENMP >= 201107
      #pragma omp atomic write
#endif
      printed = now.time_since_epoch().count();
      return true;
    }
    return false;
  }

public:
  ProgressMeter(std::ostream &sout, const std::string &message, TValue total) :
    sout(sout), message(message), scale(1000.0f / total), current(0), printed(0) {}
  ProgressMeter(std::ostream &sout, const char *message, TValue total) :
    sout(sout), message(message), scale(1000.0f / total), current(0), printed(0) {}

  // NOTE: Update() doesn't always update current value, so don't mix it with Add()
  void Update(TValue newval)
  {
    TValue oldval;
#if defined(_OPENMP) && _OPENMP >= 201107
    #pragma omp atomic read
#endif
    oldval = current;
    if (PrintFraction(oldval, newval))
    {
#if defined(_OPENMP) && _OPENMP >= 201107
      #pragma omp atomic write
#endif
      current = newval;
    }
  }
  void Add(TValue amount)
  {
    TValue newval;
#if defined(_OPENMP) && _OPENMP >= 201107
    #pragma omp atomic capture
    newval = current += amount;
#else
    newval = current + amount;
    #pragma omp atomic
    current += amount;
#endif
    PrintFraction(newval - amount, newval);
  }
  inline void AddSilent(TValue amount)
  {
    #pragma omp atomic
    current += amount;
  }

  // print a line whilst progress is still running
  void PrintLine(const std::string &line)
  {
    TValue val;
#if defined(_OPENMP) && _OPENMP >= 201107
    #pragma omp atomic read
#endif
    val = current;
    u32 fraction = CalcThousandths(val);
    #pragma omp critical(stdio)
    sout << std::setw(message.size()+7) << std::setfill(' ') << "\r"
      << line << '\n'
      << message << fraction/10 << '.' << fraction%10 << "%\r" << std::flush;
  }
};


#endif // __PROGRESSMETER_H__
