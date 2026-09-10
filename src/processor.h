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

#ifndef __PROCESSOR_H__
#define __PROCESSOR_H__

#include <future>

// Multiplies input blocks by the Reed Solomon matrix and accumulates the
// results. One input block is submitted against every output block at once, so
// that an implementation chooses how to tile the work and how to spread it over
// the threads it is given.
class Processor
{
public:
  virtual ~Processor(void) {}

  // Accumulate outputcount output blocks of up to slicesize bytes each.
  virtual bool Init(size_t slicesize, u32 outputcount) = 0;

  // The length of the blocks submitted until the next call.
  virtual void SetSliceSize(size_t length) = 0;

  // Set every accumulated output block back to zero.
  virtual void DiscardOutput(void) = 0;

  // Offer the exponents of the output blocks. An implementation that would
  // rather work them out for itself returns true, and is then given the index
  // of each input block instead of a column of coefficients. Only creation can
  // offer them: after a repair matrix is solved its coefficients are no longer
  // a function of the exponents.
  virtual bool SetRecoveryExponents(const u16 *exponents, u32 count)
  {
    (void)exponents;
    (void)count;
    return false;
  }

  // Wait until AddInput will not block.
  virtual void WaitForAdd(void) = 0;

  // Submit one input block against every output block. factors[index] is the
  // matrix coefficient for output block index, and is null when
  // SetRecoveryExponents returned true. The returned future becomes ready once
  // data may be overwritten.
  virtual std::future<void> AddInput(const void *data, size_t length, u32 inputindex, const u16 *factors) = 0;

  // Wait for every submitted input block to be processed.
  virtual void EndInput(void) = 0;

  // Copy accumulated output block index into out, which holds the length last
  // given to SetSliceSize.
  virtual bool GetOutput(u32 index, void *out) = 0;

  // Point at accumulated output block index where the implementation keeps it
  // somewhere the caller can read, saving the copy GetOutput makes. Null means
  // it does not, and the caller uses GetOutput instead. Any pointer returned
  // stays good until the next submission.
  virtual const void *PeekOutput(u32 index)
  {
    (void)index;
    return nullptr;
  }
};

#endif // __PROCESSOR_H__
