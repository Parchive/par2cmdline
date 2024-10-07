//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See https://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2024 Denis <denis@nzbget.com>
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

#include "libpar2internal.h"

#include <codecvt>
#include <iostream>
#include <cstring>
#include <exception>
#include "utf8.h"

namespace utf8
{
  constexpr int MAX_ARGS = 128;
  static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> UTF8_CONVERTER;

  std::wstring Utf8ToWide(const std::string& str)
  {
    if (str.empty()) return std::wstring();

    return UTF8_CONVERTER.from_bytes(str.c_str());
  }

  std::string WideToUtf8(const std::wstring& str)
  {
    if (str.empty()) return std::string();

    return UTF8_CONVERTER.to_bytes(str.c_str());
  }

  WideToUtf8ArgsAdapter::WideToUtf8ArgsAdapter(int argc, wchar_t* wargv[]) noexcept(false)
    : m_argc(argc)
  {
    if (wargv == nullptr)
    {
      throw std::invalid_argument("Invalid argument: wargv cannot be nullptr.");
    }

    if (m_argc > MAX_ARGS)
    {
      std::cerr
        << "Too many arguments (" << argc << "/" << MAX_ARGS << ").\n"
        << "Only " << MAX_ARGS << " will be processed." << std::endl;

      m_argc = MAX_ARGS;
    }

    m_argv = new char* [m_argc + 1];
    for (int i = 0; i < m_argc; ++i)
    {
      if (wargv[i] == nullptr)
      {
        std::cerr
          << "Invalid argument: encountered nullptr in wargv.\n"
          << "Skipping " << i << "argument." << std::endl;
        --m_argc;
        --i;
        continue;
      }

      std::string arg = WideToUtf8(wargv[i]);
      size_t size = arg.size() + 1;
      m_argv[i] = new char[size];
      std::strcpy(m_argv[i], arg.c_str());
    }
    m_argv[m_argc] = nullptr;
  }

  const char* const* WideToUtf8ArgsAdapter::GetUtf8Args() const noexcept
  {
    return m_argv;
  }

  WideToUtf8ArgsAdapter::~WideToUtf8ArgsAdapter()
  {
    if (m_argv)
    {
      for (size_t i = 0; m_argv[i]; ++i)
      {
        delete m_argv[i];
      }
      delete[] m_argv;
    }
  }
}
