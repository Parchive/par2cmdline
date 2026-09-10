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

#ifndef __BACKENDS_H__
#define __BACKENDS_H__

#include <functional>
#include <memory>

#include "processor.h"

// The budgets a backend is built with. What it accumulates is given separately,
// by Processor::Init, once the number of recovery blocks is known.
struct ProcessorConfig
{
  u32    numthreads;   // The thread budget, from -t
  size_t memorylimit;  // The memory budget, from -m
};

// The implementations an application supplies. A factory left empty selects the
// one built into par2cmdline.
struct Backends
{
  std::function<std::unique_ptr<Processor>(const ProcessorConfig &)> processor;
};

#endif // __BACKENDS_H__
