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

#include "libpar2internal.h"

#ifdef _WIN32

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "utf8.h"

namespace utf8
{
  const int MAX_ARGS = 128;
  const size_t MAX_DIR_PATH = 248;

  static void ApplyLongPathPrefix(std::wstring& wpath)
  {
    if (wpath.size() <= MAX_DIR_PATH ||
      wpath.find(L"\\\\?\\") != std::wstring::npos)
    {
      return;
    }

    if (wpath.compare(0, 2, L"\\\\") == 0)
    {
      wpath = L"\\\\?\\UNC" + wpath;
    }
    else
    {
      wpath = L"\\\\?\\" + wpath;
    }
  }

  bool Utf8ToWide(const std::string& str, std::wstring& out)
  {
    if (str.empty())
    {
      out.clear();
      return true;
    }

    const int length = (int)str.size();
    const int required = ::MultiByteToWideChar(
      CP_UTF8,
      MB_ERR_INVALID_CHARS,
      str.c_str(),
      length,
      nullptr,
      0
    );
    if (required <= 0)
      return false;

    std::wstring wpath(required, L'\0');
    if (::MultiByteToWideChar(
      CP_UTF8,
      MB_ERR_INVALID_CHARS,
      str.c_str(),
      length,
      &wpath[0],
      required
    ) <= 0)
      return false;

    ApplyLongPathPrefix(wpath);

    out.swap(wpath);
    return true;
  }

  bool WideToUtf8(const std::wstring& str, std::string& out)
  {
    if (str.empty())
    {
      out.clear();
      return true;
    }

    const int length = (int)str.size();
    const int required = ::WideCharToMultiByte(
      CP_UTF8,
      WC_ERR_INVALID_CHARS,
      str.c_str(),
      length,
      nullptr,
      0,
      nullptr,
      nullptr
    );
    if (required <= 0)
      return false;

    std::string utf8(required, '\0');
    if (::WideCharToMultiByte(
      CP_UTF8,
      WC_ERR_INVALID_CHARS,
      str.c_str(),
      length,
      &utf8[0],
      required,
      nullptr,
      nullptr
    ) <= 0)
      return false;

    out.swap(utf8);
    return true;
  }

  WideToUtf8ArgsAdapter::WideToUtf8ArgsAdapter(int argc, wchar_t* wargv[]) noexcept(false)
    : m_argv(nullptr)
    , m_argc(argc)
  {
    if (wargv == nullptr)
    {
      throw std::invalid_argument("Invalid argument: wargv cannot be nullptr.");
    }

    if (m_argc > MAX_ARGS)
    {
      std::cerr
        << "Too many arguments (" << argc << "/" << MAX_ARGS << ").\n"
           "Only " << MAX_ARGS << " will be processed." << std::endl;

      m_argc = MAX_ARGS;
    }

    m_argv = new char* [m_argc + 1];

    int argcount = 0;
    for (int i = 0; i < m_argc; ++i)
    {
      if (wargv[i] == nullptr)
      {
        std::cerr
          << "Invalid argument: encountered nullptr in wargv.\n"
             "Skipping argument " << i << "." << std::endl;
        continue;
      }

      std::string arg;
      if (!WideToUtf8(wargv[i], arg))
      {
        std::cerr
          << "Failed to convert wide to UTF-8 string.\n"
             "Skipping argument " << i << "." << std::endl;
        continue;
      }

      const size_t size = arg.size() + 1;
      m_argv[argcount] = new char[size];
      std::memcpy(m_argv[argcount], arg.c_str(), size);
      ++argcount;
    }

    m_argc = argcount;
    m_argv[m_argc] = nullptr;
  }

  const char* const* WideToUtf8ArgsAdapter::GetUtf8Args() const noexcept
  {
    return m_argv;
  }

  int WideToUtf8ArgsAdapter::GetArgc() const noexcept
  {
    return m_argc;
  }

  WideToUtf8ArgsAdapter::~WideToUtf8ArgsAdapter()
  {
    if (m_argv)
    {
      for (int i = 0; i < m_argc; ++i)
      {
        delete[] m_argv[i];
      }
      delete[] m_argv;
    }
  }
}

#endif // _WIN32
