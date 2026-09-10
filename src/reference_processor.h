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

#ifndef __REFERENCE_PROCESSOR_H__
#define __REFERENCE_PROCESSOR_H__

// Multiplies each input block by the matrix on numthreads threads and keeps the
// accumulated output blocks in one buffer. Every submission is complete by the
// time AddInput returns.
class ReferenceProcessor : public Processor
{
public:
  ReferenceProcessor(ReedSolomon<Galois16> &rs, u32 numthreads)
    : rs(rs)
    , numthreads(numthreads)
    , slicesize(0)
    , outputcount(0)
    , currentlength(0)
    , outputbuffer(0)
  {
  }

  ~ReferenceProcessor(void)
  {
    delete [] outputbuffer;
  }

  bool Init(size_t _slicesize, u32 _outputcount)
  {
    delete [] outputbuffer;

    slicesize = _slicesize;
    outputcount = _outputcount;
    currentlength = _slicesize;
    outputbuffer = new u8[slicesize * outputcount];

    return outputbuffer != nullptr;
  }

  void SetSliceSize(size_t length)
  {
    currentlength = length;
  }

  void DiscardOutput(void)
  {
    memset(outputbuffer, 0, slicesize * outputcount);
  }

  void WaitForAdd(void)
  {
  }

  std::future<void> AddInput(const void *data, size_t length, u32 inputindex, const u16 *factors)
  {
    (void)inputindex;

    foreach_parallel(0, outputcount, numthreads, [&](size_t outputindex)
    {
      rs.MultiplyAdd(factors[outputindex], length, data, &outputbuffer[slicesize * outputindex]);
    });

    std::promise<void> processed;
    processed.set_value();
    return processed.get_future();
  }

  void EndInput(void)
  {
  }

  bool GetOutput(u32 index, void *out)
  {
    memcpy(out, &outputbuffer[slicesize * index], currentlength);
    return true;
  }

private:
  ReedSolomon<Galois16> &rs;
  u32 numthreads;
  size_t slicesize;
  u32 outputcount;
  size_t currentlength;
  u8 *outputbuffer;
};

#endif // __REFERENCE_PROCESSOR_H__
