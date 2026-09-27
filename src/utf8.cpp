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

namespace par2
{
namespace utf8
{
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
      wpath = L"\\\\?\\UNC" + wpath.substr(1);
    }
    else
    {
      wpath = L"\\\\?\\" + wpath;
    }
  }

  static bool Decode(UINT codepage, const std::string& str, std::wstring& out)
  {
    const int length = (int)str.size();
    const int required = ::MultiByteToWideChar(
      codepage,
      MB_ERR_INVALID_CHARS,
      str.c_str(),
      length,
      nullptr,
      0
    );
    if (required <= 0)
      return false;

    std::wstring wide(required, L'\0');
    if (::MultiByteToWideChar(
      codepage,
      MB_ERR_INVALID_CHARS,
      str.c_str(),
      length,
      &wide[0],
      required
    ) <= 0)
      return false;

    out.swap(wide);
    return true;
  }

  bool Utf8ToWide(const std::string& str, std::wstring& out)
  {
    if (str.empty())
    {
      out.clear();
      return true;
    }

    std::wstring wpath;
    if (!Decode(CP_UTF8, str, wpath) && !Decode(CP_ACP, str, wpath))
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
}
}

#endif // _WIN32
