//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See https://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2024-2025 Denis <denis@nzbget.com>
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

#ifndef __UTF8_H__
#define __UTF8_H__

#ifdef _WIN32

#include <string>

namespace utf8
{
  extern const int MAX_ARGS;
  extern const size_t MAX_DIR_PATH;

  // False if the string is not well formed, leaving out untouched. Otherwise out
  // holds the conversion, and a path longer than MAX_DIR_PATH has gained a
  // \\?\ or \\?\UNC prefix so that the Win32 calls accept it.
  bool Utf8ToWide(const std::string& str, std::wstring& out);
  bool WideToUtf8(const std::wstring& str, std::string& out);

  class WideToUtf8ArgsAdapter final
  {
  public:
    WideToUtf8ArgsAdapter(int argc, wchar_t* argv_[]) noexcept(false);

    const char* const* GetUtf8Args() const noexcept;

    // The number of arguments in GetUtf8Args(), which is less than the argc
    // passed in when an argument could not be used.
    int GetArgc() const noexcept;

    WideToUtf8ArgsAdapter() = delete;
    WideToUtf8ArgsAdapter(const WideToUtf8ArgsAdapter&) = delete;
    WideToUtf8ArgsAdapter(WideToUtf8ArgsAdapter&&) = delete;
    WideToUtf8ArgsAdapter& operator=(const WideToUtf8ArgsAdapter&) = delete;
    WideToUtf8ArgsAdapter& operator=(WideToUtf8ArgsAdapter&&) = delete;

    ~WideToUtf8ArgsAdapter();

  private:
    char** m_argv;
    int m_argc;
  };
}

#endif // _WIN32

#endif // __UTF8_H__
