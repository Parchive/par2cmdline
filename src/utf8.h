//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
//  Copyright (c) 2019 Michael D. Nahas
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
// 

#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

#include <string>

namespace utf8 {
#if defined(_WIN32) && defined(UNICODE)
  std::string narrow(const std::wstring& wstr, UINT cp);
  std::string narrow(const std::wstring& wstr);
  std::wstring widen(const std::string& str);
  std::string console(const std::string& str);
#else
  std::string narrow(std::string wstr, unsigned int cp);
  std::string narrow(std::string wstr);
  std::string widen(std::string str);
  std::string console(std::string str);
#endif
}
