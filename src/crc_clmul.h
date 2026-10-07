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

#ifndef __CRC_CLMUL_H__
#define __CRC_CLMUL_H__

// CRC32 using the x86 carry-less multiply, over 128-bit registers (PCLMULQDQ)
// and 256-bit ones (VPCLMULQDQ). Included by crc.cpp.

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) \
  && (defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__))
# define PAR2_CRC_X86 1
# include <immintrin.h>
# ifdef _MSC_VER
#  include <intrin.h>
# else
#  include <cpuid.h>
# endif
# if defined(__GNUC__) || defined(__clang__)
#  define PAR2_CRC_X86_TARGET(isa) __attribute__((target(isa)))
# else
#  define PAR2_CRC_X86_TARGET(isa)
# endif
# if (defined(__clang__) && __clang_major__ >= 8) \
  || (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8) \
  || (defined(_MSC_VER) && !defined(__clang__) && _MSC_VER >= 1920)
#  define PAR2_CRC_X86_VPCLMUL 1
# endif
#endif

#ifdef PAR2_CRC_X86

namespace par2
{

// The folding algorithm is Intel's, from
//
//   "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction"
//   Vinodh Gopal, Erdinc Ozturk, Jim Guilford, Gil Wolrich, Wajdi Feghali,
//   Martin Dixon and Deniz Karakoyunlu, Intel Corporation, December 2009.
//
// Folding a 128-bit chunk multiplies it by x^k mod P(x), carrying it past the
// k bits that follow so it can be xored onto a later chunk. Data is held
// bit-reflected, as a little-endian load leaves it and as par2's CRC32 works;
// the multipliers are reflected residues shifted up one place, a carry-less
// multiply of two reflected values coming out one bit low.
struct crcfoldconstants
{
  // Multiplier pairs for folding a chunk forward over N bits. A chunk's low
  // half is 64 bits further from the end than its high half, so it takes the
  // larger power of x.
  u64 by16[2];    // N = 128,  a single chunk
  u64 by32[2];    // N = 256,  one 256-bit register
  u64 by64[2];    // N = 512,  four chunks
  u64 by128[2];   // N = 1024, four 256-bit registers
  u64 to64[2];    // x^96, folding a 96-bit remainder down to 64 bits
  u64 barrett[2]; // P'(x), and floor(x^64 / P(x)) for the Barrett reduction
};

// For the CCITT polynomial.
static const crcfoldconstants crcfold =
{
  { 0x1751997d0ULL, 0x0ccaa009eULL }, // x^192,  x^128
  { 0x0f1da05aaULL, 0x15a546366ULL }, // x^320,  x^256
  { 0x154442bd4ULL, 0x1c6e41596ULL }, // x^576,  x^512
  { 0x1e88ef372ULL, 0x14a7fe880ULL }, // x^1088, x^1024
  { 0x163cd6124ULL, 0              }, // x^96
  { 0x1db710641ULL, 0x1f7011641ULL }  // P'(x), mu
};

// A constant pair as a vector: the first in the low half, the second in the high.
PAR2_CRC_X86_TARGET("sse2")
static inline __m128i CRCConstants(const u64 (&pair)[2])
{
  return _mm_setr_epi32((int)(u32)pair[0], (int)(u32)(pair[0] >> 32),
                        (int)(u32)pair[1], (int)(u32)(pair[1] >> 32));
}

// Fold one chunk forward and xor it onto another.
PAR2_CRC_X86_TARGET("sse2,pclmul")
static inline __m128i CRCFold(__m128i chunk, __m128i multiplier, __m128i onto)
{
  onto = _mm_xor_si128(onto, _mm_clmulepi64_si128(chunk, multiplier, 0x00));
  return _mm_xor_si128(onto, _mm_clmulepi64_si128(chunk, multiplier, 0x11));
}

// Reduce the final 128-bit chunk to the 32-bit CRC.
PAR2_CRC_X86_TARGET("sse2,pclmul")
static u32 CRCReduce(__m128i chunk)
{
  // The low 32 bits of each half.
  const __m128i low32 = _mm_setr_epi32(~0, 0, ~0, 0);

  // 128 bits to 96: fold the low half onto the high half.
  __m128i multiplier = CRCConstants(crcfold.by16);
  chunk = _mm_xor_si128(_mm_srli_si128(chunk, 8),
                        _mm_clmulepi64_si128(chunk, multiplier, 0x10));

  // 96 bits to 64.
  multiplier = CRCConstants(crcfold.to64);
  __m128i high = _mm_srli_si128(chunk, 4);
  chunk = _mm_and_si128(chunk, low32);
  chunk = _mm_xor_si128(_mm_clmulepi64_si128(chunk, multiplier, 0x00), high);

  // 64 bits to 32, by Barrett reduction: times mu gives the quotient, times
  // the polynomial gives what to cancel.
  multiplier = CRCConstants(crcfold.barrett);
  __m128i cancel = _mm_and_si128(chunk, low32);
  cancel = _mm_clmulepi64_si128(cancel, multiplier, 0x10);
  cancel = _mm_and_si128(cancel, low32);
  cancel = _mm_clmulepi64_si128(cancel, multiplier, 0x00);
  chunk = _mm_xor_si128(chunk, cancel);

  return (u32)_mm_cvtsi128_si32(_mm_srli_si128(chunk, 4));
}

// Fold whatever whole chunks are left onto a single accumulator and finish.
PAR2_CRC_X86_TARGET("sse2,pclmul")
static u32 CRCFoldTail(__m128i chunk, const unsigned char *current, size_t length)
{
  const __m128i multiplier = CRCConstants(crcfold.by16);

  while (length >= 16)
  {
    chunk = CRCFold(chunk, multiplier, _mm_loadu_si128((const __m128i *)current));
    current += 16;
    length -= 16;
  }

  return CRCUpdateBlockScalar(CRCReduce(chunk), length, current);
}

PAR2_CRC_X86_TARGET("sse2,pclmul")
static u32 CRCUpdateBlockPclMul(u32 crc, size_t length, const void *buffer)
{
  const unsigned char *current = (const unsigned char *)buffer;

  if (length < 64)
    return CRCUpdateBlockScalar(crc, length, current);

  // Four accumulators, one per chunk of the first 64 bytes. The incoming CRC
  // is the remainder of everything before the buffer, so it xors onto the front.
  __m128i x0 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)current),
                             _mm_cvtsi32_si128((int)crc));
  __m128i x1 = _mm_loadu_si128((const __m128i *)(current + 16));
  __m128i x2 = _mm_loadu_si128((const __m128i *)(current + 32));
  __m128i x3 = _mm_loadu_si128((const __m128i *)(current + 48));
  current += 64;
  length -= 64;

  const __m128i by64 = CRCConstants(crcfold.by64);
  while (length >= 64)
  {
    x0 = CRCFold(x0, by64, _mm_loadu_si128((const __m128i *)current));
    x1 = CRCFold(x1, by64, _mm_loadu_si128((const __m128i *)(current + 16)));
    x2 = CRCFold(x2, by64, _mm_loadu_si128((const __m128i *)(current + 32)));
    x3 = CRCFold(x3, by64, _mm_loadu_si128((const __m128i *)(current + 48)));
    current += 64;
    length -= 64;
  }

  // Collapse the four, each one chunk ahead of the next, onto the last.
  const __m128i by16 = CRCConstants(crcfold.by16);
  x1 = CRCFold(x0, by16, x1);
  x2 = CRCFold(x1, by16, x2);
  x3 = CRCFold(x2, by16, x3);

  return CRCFoldTail(x3, current, length);
}

#ifdef PAR2_CRC_X86_VPCLMUL

// The same folding with each register holding two chunks. VPCLMULQDQ
// multiplies the halves of a 256-bit register independently, so a broadcast
// multiplier keeps the two lanes as separate folds.
PAR2_CRC_X86_TARGET("avx2,pclmul,vpclmulqdq")
static inline __m256i CRCFold2(__m256i chunks, __m256i multiplier, __m256i onto)
{
  onto = _mm256_xor_si256(onto, _mm256_clmulepi64_epi128(chunks, multiplier, 0x00));
  return _mm256_xor_si256(onto, _mm256_clmulepi64_epi128(chunks, multiplier, 0x11));
}

PAR2_CRC_X86_TARGET("avx2,pclmul,vpclmulqdq")
static u32 CRCUpdateBlockVPclMul(u32 crc, size_t length, const void *buffer)
{
  const unsigned char *current = (const unsigned char *)buffer;

  if (length < 128)
    return CRCUpdateBlockPclMul(crc, length, buffer);

  __m256i y0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)current),
                                _mm256_inserti128_si256(_mm256_setzero_si256(),
                                                        _mm_cvtsi32_si128((int)crc), 0));
  __m256i y1 = _mm256_loadu_si256((const __m256i *)(current + 32));
  __m256i y2 = _mm256_loadu_si256((const __m256i *)(current + 64));
  __m256i y3 = _mm256_loadu_si256((const __m256i *)(current + 96));
  current += 128;
  length -= 128;

  const __m256i by128 = _mm256_broadcastsi128_si256(CRCConstants(crcfold.by128));
  while (length >= 128)
  {
    y0 = CRCFold2(y0, by128, _mm256_loadu_si256((const __m256i *)current));
    y1 = CRCFold2(y1, by128, _mm256_loadu_si256((const __m256i *)(current + 32)));
    y2 = CRCFold2(y2, by128, _mm256_loadu_si256((const __m256i *)(current + 64)));
    y3 = CRCFold2(y3, by128, _mm256_loadu_si256((const __m256i *)(current + 96)));
    current += 128;
    length -= 128;
  }

  // Collapse the four registers, each two chunks ahead of the next.
  const __m256i by32 = _mm256_broadcastsi128_si256(CRCConstants(crcfold.by32));
  y1 = CRCFold2(y0, by32, y1);
  y2 = CRCFold2(y1, by32, y2);
  y3 = CRCFold2(y2, by32, y3);

  // Then the two lanes of what is left, the low one a chunk ahead of the high.
  __m128i chunk = CRCFold(_mm256_castsi256_si128(y3),
                          CRCConstants(crcfold.by16),
                          _mm256_extracti128_si256(y3, 1));

  return CRCFoldTail(chunk, current, length);
}

#endif // PAR2_CRC_X86_VPCLMUL

static void CRCCpuId(unsigned leaf, unsigned subleaf, unsigned (&registers)[4])
{
#ifdef _MSC_VER
  __cpuidex((int *)registers, (int)leaf, (int)subleaf);
#else
  __cpuid_count(leaf, subleaf, registers[0], registers[1], registers[2], registers[3]);
#endif
}

static bool X86HasPclMul()
{
  unsigned registers[4];

  CRCCpuId(0, 0, registers);
  if (registers[0] < 1)
    return false;

  CRCCpuId(1, 0, registers);
  return (registers[2] & (1u << 1)) != 0;
}

#ifdef PAR2_CRC_X86_VPCLMUL
static bool X86HasVPclMul()
{
  unsigned registers[4];

  CRCCpuId(0, 0, registers);
  if (registers[0] < 7)
    return false;

  CRCCpuId(1, 0, registers);
  if ((registers[2] & (1u << 27)) == 0 || (registers[2] & (1u << 28)) == 0)
    return false;

  // The OS must also be saving the upper halves across a context switch.
#ifdef _MSC_VER
  const u32 xcr0 = (u32)_xgetbv(0);
#else
  u32 lower, upper;
  __asm__ __volatile__("xgetbv" : "=a"(lower), "=d"(upper) : "c"(0));
  (void)upper;
  const u32 xcr0 = lower;
#endif
  if ((xcr0 & 0x6) != 0x6)
    return false;

  CRCCpuId(7, 0, registers);
  return (registers[1] & (1u << 5)) != 0 && (registers[2] & (1u << 10)) != 0;
}
#endif // PAR2_CRC_X86_VPCLMUL

} // namespace par2

#endif // PAR2_CRC_X86

#endif // __CRC_CLMUL_H__
