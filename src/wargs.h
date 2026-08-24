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

#ifndef __WARGS_H__
#define __WARGS_H__

#ifdef _WIN32

namespace par2
{
namespace utf8
{
  extern const int MAX_ARGS;

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
}

#endif // _WIN32

#endif // __WARGS_H__
