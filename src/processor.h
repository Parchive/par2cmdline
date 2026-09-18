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

#ifndef __PROCESSOR_H__
#define __PROCESSOR_H__

#include <cstddef>
#include <future>
#include <vector>

#include "types.h"

// Multiplies input blocks by the Reed Solomon matrix and accumulates the
// results. One input block is submitted against every output block at once, so
// that an implementation chooses how to tile the work and how to spread it over
// the threads it is given, which reach it in the ProcessorConfig it is built
// from.
//
// Every block, in and out, is a sequence of 16 bit Galois values, each little
// endian, which is the layout they have on disk. The values belong to GF(2^16)
// with the generator 0x1100B.
//
// The work is done in chunks of up to the length Init is given, one pass over
// every input block for each chunk:
//
//   Init
//   OfferErasures                        repairing only
//   for each chunk:
//     SetChunkLength
//     ResetOutput
//     OfferRecoveryExponents             creating only
//     for each input block:
//       WaitForAdd
//       AddInput
//     EndInput
//     for each output block:
//       PeekOutput, and GetOutput where it gives nothing
//
// Only the methods declared pure need implementing. The three carrying a body
// are chances to do less work, and an implementation which overrides none of
// them is correct.
class Processor
{
public:
  virtual ~Processor(void) {}

  // Accumulate outputcount output blocks of up to maxlength bytes each. The
  // accumulated blocks need not start at zero, ResetOutput coming before
  // anything is submitted against them.
  virtual bool Init(size_t maxlength, u32 outputcount) = 0;

  // The length of the blocks submitted until the next call, never more than
  // the maxlength given to Init.
  virtual void SetChunkLength(size_t length) = 0;

  // Set every accumulated output block back to zero.
  virtual void ResetOutput(void) = 0;

  // Offer the number of input blocks which will be submitted, and the
  // exponents of the exponentcount output blocks, which are the outputcount
  // given to Init. An implementation that would rather work the coefficients
  // out for itself returns true, and is given no factors from then on. Only
  // creation offers them: after a repair matrix is solved its coefficients are
  // no longer a function of the exponents.
  virtual bool OfferRecoveryExponents(u32 inputcount, const u16 *exponents, u32 exponentcount)
  {
    (void)inputcount;
    (void)exponents;
    (void)exponentcount;
    return false;
  }

  // Offer the erasure pattern rather than a solved matrix. present[index] says
  // whether input block index was found, and exponents holds those of the
  // exponentcount recovery blocks standing in for the ones that were not, in
  // the order they are submitted after the blocks that were found. An
  // implementation that solves the erasure for itself returns true, and is
  // given no factors from then on; the caller does not invert the matrix at
  // all. Only repair offers this, creation having no erasures to solve.
  virtual bool OfferErasures(const std::vector<bool> &present, const u16 *exponents, u32 exponentcount)
  {
    (void)present;
    (void)exponents;
    (void)exponentcount;
    return false;
  }

  // Wait until AddInput will not block. The caller holds a bounded number of
  // submissions in flight and calls this before each of them.
  virtual void WaitForAdd(void) = 0;

  // Submit one input block against every output block. length is the length
  // last given to SetChunkLength. factors[index] is the matrix coefficient for
  // output block index, and is null where an offer was taken, leaving
  // inputindex to say which input block this is:
  //
  //   creating, it is the source block, counting from zero
  //   repairing, it counts the submissions: first the source blocks which were
  //   found, in the order present marks them, then one for each exponent
  //   OfferErasures was given, in that order
  //
  // Called from one thread, and always after WaitForAdd. The returned future
  // becomes ready once data and factors may both be overwritten, and one which
  // is ready already says the implementation has finished with them. An
  // implementation which returns before it has multiplied the block need copy
  // neither of them.
  virtual std::future<void> AddInput(const void *data, size_t length, u32 inputindex, const u16 *factors) = 0;

  // Wait for every submitted input block to be processed.
  virtual void EndInput(void) = 0;

  // Point at accumulated output block index where the implementation keeps it
  // somewhere the caller can read, saving the copy GetOutput makes. Null means
  // it does not, and the caller uses GetOutput instead; it never means the
  // block could not be produced, which only GetOutput reports. The block holds
  // the length last given to SetChunkLength, and the pointer stays good until
  // the next submission.
  virtual const void *PeekOutput(u32 index)
  {
    (void)index;
    return nullptr;
  }

  // Copy accumulated output block index into out, which holds the length last
  // given to SetChunkLength. False where the block could not be produced, which
  // ends the operation.
  virtual bool GetOutput(u32 index, void *out) = 0;
};

#endif // __PROCESSOR_H__
