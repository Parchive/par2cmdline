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

template<typename TValue>
class ProgressMeter
{
  std::ostream &sout;
  std::string message;
  TValue total;
  TValue current;

  inline u32 CalcThousandths(TValue val)
  {
    return (u32)(1000 * val / total);
  }

public:
  ProgressMeter(std::ostream &sout, std::string &message, TValue total) :
    sout(sout), message(message), total(total), current(0) {}
  ProgressMeter(std::ostream &sout, const char *message, TValue total) :
    sout(sout), message(message), total(total), current(0) {}

  // NOTE: Update() doesn't always update current value, so don't mix it with Add()
  void Update(TValue newval)
  {
    TValue oldval;
#if defined(_OPENMP) && _OPENMP >= 201107
    #pragma omp atomic read
#endif
    oldval = current;
    u32 oldfraction = CalcThousandths(oldval);
    u32 newfraction = CalcThousandths(newval);
    if (oldfraction != newfraction)
    {
      #pragma omp critical(stdio)
      sout << message << newfraction/10 << '.' << newfraction%10 << "%\r" << std::flush;
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
    u32 oldfraction = CalcThousandths(newval - amount);
    u32 newfraction = CalcThousandths(newval);
    if (oldfraction != newfraction)
    {
      #pragma omp critical(stdio)
      sout << message << newfraction/10 << '.' << newfraction%10 << "%\r" << std::flush;
    }
  }
  inline void AddSilent(TValue amount)
  {
    #pragma omp atomic
    current += amount;
  }
  void ClearLine()
  {
    #pragma omp critical(stdio)
    sout << std::setw(message.size()+6) << std::setfill(' ') << "\r";
  }
  void Print()
  {
    TValue val;
#if defined(_OPENMP) && _OPENMP >= 201107
    #pragma omp atomic read
#endif
    val = current;
    u32 fraction = CalcThousandths(val);
    #pragma omp critical(stdio)
    sout << message << fraction/10 << '.' << fraction%10 << "%\r" << std::flush;
  }
};


#endif // __PROGRESSMETER_H__
