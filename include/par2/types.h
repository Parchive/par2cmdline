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

#ifndef __TYPES_H__
#define __TYPES_H__

#include <cstdint>

namespace par2
{

typedef std::uint8_t  u8;
typedef std::int8_t   i8;
typedef std::uint16_t u16;
typedef std::int16_t  i16;
typedef std::uint32_t u32;
typedef std::int32_t  i32;
typedef std::uint64_t u64;
typedef std::int64_t  i64;

static_assert(sizeof(u8) == 1 && sizeof(i8) == 1
		&& sizeof(u16) == 2 && sizeof(i16) == 2
		&& sizeof(u32) == 4 && sizeof(i32) == 4
		&& sizeof(u64) == 8 && sizeof(i64) == 8,
		"the integer types are the widths their names give");

} // namespace par2

#endif // __TYPES_H__
