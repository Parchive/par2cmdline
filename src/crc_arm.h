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

#ifndef __CRC_ARM_H__
#define __CRC_ARM_H__

// CRC32 using the ARM CRC32 extension. Included by crc.cpp.

#if defined(__aarch64__) || defined(_M_ARM64)
# define PAR2_CRC_ARM64 1
#endif

// GCC ships an arm_acle.h whose CRC32 intrinsics do not compile, in 7.0 to 8.1
// on 32-bit ARM and in 9.4 on aarch64.
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81497
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100985
#if defined(__GNUC__) && !defined(__clang__)
# if !defined(PAR2_CRC_ARM64) && __GNUC__ >= 7 \
   && (__GNUC__ < 8 \
    || (__GNUC__ == 8 && __GNUC_MINOR__ < 1) \
    || (__GNUC__ == 8 && __GNUC_MINOR__ == 1 && __GNUC_PATCHLEVEL__ < 1))
#  define PAR2_CRC_BROKEN_ACLE 1
# endif
# if defined(PAR2_CRC_ARM64) && __GNUC__ == 9 && __GNUC_MINOR__ == 4
#  define PAR2_CRC_BROKEN_ACLE 1
# endif
#endif

#if (defined(PAR2_CRC_ARM64) || defined(__arm__) || defined(_M_ARM)) \
  && !defined(PAR2_CRC_BROKEN_ACLE) \
  && (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
// MSVC has no CRC32 intrinsics for 32-bit ARM
# if defined(_MSC_VER) && !defined(__clang__) && defined(PAR2_CRC_ARM64)
#  define PAR2_CRC_ARM 1
#  define PAR2_CRC_ARM_TARGET
#  include <intrin.h>
# elif defined(__ARM_FEATURE_CRC32)
#  define PAR2_CRC_ARM 1
#  define PAR2_CRC_ARM_TARGET
#  include <arm_acle.h>
// Without __ARM_FEATURE_CRC32, arm_acle.h only declares the intrinsics from
// GCC 7 and Clang 16
# elif defined(PAR2_CRC_ARM64) \
   && ((defined(__clang__) && __clang_major__ >= 16) \
    || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 7))
#  define PAR2_CRC_ARM 1
#  define PAR2_CRC_ARM_TARGET __attribute__((target("+crc")))
#  include <arm_acle.h>
# endif
#endif

#ifdef PAR2_CRC_ARM

# if defined(__APPLE__)
#  include <sys/sysctl.h>
# elif defined(__linux__) || defined(__FreeBSD__)
#  include <sys/auxv.h>
#  ifndef HWCAP_CRC32
#   define HWCAP_CRC32 (1 << 7)
#  endif
#  ifndef HWCAP2_CRC32
#   define HWCAP2_CRC32 (1 << 4)
#  endif
# endif

namespace par2
{

template<typename T> static inline T CRCRead(const unsigned char *current)
{
  T value;
  memcpy(&value, current, sizeof(value));
  return value;
}

// The instructions take the data a word at a time, low byte first, matching
// the order the reversed CCITT polynomial consumes it in.
PAR2_CRC_ARM_TARGET
static u32 CRCUpdateBlockArm(u32 crc, size_t length, const void *buffer)
{
  const unsigned char *current = (const unsigned char *)buffer;

#ifdef PAR2_CRC_ARM64
  while (length >= sizeof(u64))
  {
    crc = __crc32d(crc, CRCRead<u64>(current));
    current += sizeof(u64);
    length -= sizeof(u64);
  }
  if (length & sizeof(u32))
  {
    crc = __crc32w(crc, CRCRead<u32>(current));
    current += sizeof(u32);
  }
#else
  while (length >= sizeof(u32))
  {
    crc = __crc32w(crc, CRCRead<u32>(current));
    current += sizeof(u32);
    length -= sizeof(u32);
  }
#endif
  if (length & sizeof(u16))
  {
    crc = __crc32h(crc, CRCRead<u16>(current));
    current += sizeof(u16);
  }
  if (length & sizeof(u8))
    crc = __crc32b(crc, *current);

  return crc;
}

static bool ArmHasCRC()
{
#if defined(__ARM_FEATURE_CRC32)
  return true;
#elif defined(__APPLE__)
  int present = 0;
  size_t size = sizeof(present);
  return sysctlbyname("hw.optional.armv8_crc32", &present, &size, NULL, 0) == 0 && present != 0;
#elif defined(__linux__)
# ifdef PAR2_CRC_ARM64
  return (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
# else
  return (getauxval(AT_HWCAP2) & HWCAP2_CRC32) != 0;
# endif
#elif defined(__FreeBSD__) && defined(PAR2_CRC_ARM64)
  unsigned long hwcap = 0;
  return elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap)) == 0 && (hwcap & HWCAP_CRC32) != 0;
#elif defined(_WIN32)
  return IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) != 0;
#else
  return false;
#endif
}

} // namespace par2

#endif // PAR2_CRC_ARM

#endif // __CRC_ARM_H__
