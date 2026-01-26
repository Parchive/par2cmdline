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

#include <iostream>
#include "utf8.h"


using namespace utf8;

int test1()
{
  std::string emptyString = "";
  std::wstring expectedWide = L"";
  std::wstring actualWide = Utf8ToWide(emptyString);

  std::string expectedUtf8 = "";
  std::string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test2()
{
  std::string asciiString = "Hello, World!";
  std::wstring expectedWide = L"Hello, World!";
  std::wstring actualWide = Utf8ToWide(asciiString);

  std::string expectedUtf8 = "Hello, World!";
  std::string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test3()
{
  // "Привет, мир!"
  std::string nonAsciiString = "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82, \xD0\xBC\xD0\xB8\xD1\x80!";
  std::wstring expectedWide = L"\x041F\x0440\x0438\x0432\x0435\x0442, \x043C\x0438\x0440!";
  std::wstring actualWide = Utf8ToWide(nonAsciiString);

  std::string expectedUtf8 = "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82, \xD0\xBC\xD0\xB8\xD1\x80!";
  std::string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test4()
{
  std::string specialCharsString = "This std::string has: !@#$%^&*()_+=-`~[]{}:;'<>,.?/";
  std::wstring expectedWide = L"This std::string has: !@#$%^&*()_+=-`~[]{}:;'<>,.?/";
  std::wstring actualWide = Utf8ToWide(specialCharsString);

  std::string expectedUtf8 = "This std::string has: !@#$%^&*()_+=-`~[]{}:;'<>,.?/";
  std::string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test5()
{
  // "Привет! こんにちは世界! 안녕하세요!"
  std::string multiLangString = "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82! \xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF\xE4\xB8\x96\xE7\x95\x8C! \xEC\x95\x88\xEB\x85\x95\xED\x95\x98\xEC\x84\xB8\xEC\x9A\x94!";
  std::wstring expectedWide = L"\x041F\x0440\x0438\x0432\x0435\x0442! \x3053\x3093\x306B\x3061\x306F\x4E16\x754C! \xC548\xB155\xD558\xC138\xC694!";
  std::wstring actualWide = Utf8ToWide(multiLangString);

  std::string expectedUtf8 = "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82! \xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF\xE4\xB8\x96\xE7\x95\x8C! \xEC\x95\x88\xEB\x85\x95\xED\x95\x98\xEC\x84\xB8\xEC\x9A\x94!";
  std::string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test6()
{
  wchar_t* wargv[1] = { nullptr };
  WideToUtf8ArgsAdapter adapter(0, wargv);
  const char* const* utf8Args = adapter.GetUtf8Args();

  return nullptr == utf8Args;
}

int test7()
{
  // L"Привет", L"мир", L"!"
  wchar_t* wargv[3] = { const_cast<wchar_t*>(L"\x041F\x0440\x0438\x0432\x0435\x0442"), const_cast<wchar_t*>(L"\x043C\x0438\x0440"), const_cast<wchar_t*>(L"!") };
  WideToUtf8ArgsAdapter adapter(3, wargv);
  const char* const* utf8Args = adapter.GetUtf8Args();

  for (int i = 0; i < 3; ++i) {
    if (WideToUtf8(wargv[i]) != std::string(utf8Args[i]))
    {
      return 1;
    }
  }

  return 0;
}

int test8()
{
  wchar_t* wargv[3] = { const_cast<wchar_t*>(L"arg1"), nullptr, const_cast<wchar_t*>(L"arg3") };
  WideToUtf8ArgsAdapter adapter(3, wargv);
  const char* const* utf8Args = adapter.GetUtf8Args();
  int argc = 3;
  for (int i = 0; i < argc; ++i)
  {
    if (wargv[i] == nullptr)
    {
      --i;
      --argc;
      continue;
    }

    if (WideToUtf8(wargv[i]) != std::string(utf8Args[i]))
    {
      return 1;
    }
  }

  return 0;
}

int main()
{
  if (test1())
  {
    std::cerr << "FAILED: test1" << std::endl;
    return 1;
  }

  if (test2())
  {
    std::cerr << "FAILED: test2" << std::endl;
    return 1;
  }

  if (test3())
  {
    std::cerr << "FAILED: test3" << std::endl;
    return 1;
  }

  if (test4())
  {
    std::cerr << "FAILED: test4" << std::endl;
    return 1;
  }

  if (test5())
  {
    std::cerr << "FAILED: test5" << std::endl;
    return 1;
  }

  if (test6())
  {
    std::cerr << "FAILED: test6" << std::endl;
    return 1;
  }

  if (test7())
  {
    std::cerr << "FAILED: test7" << std::endl;
    return 1;
  }

  if (test8())
  {
    std::cerr << "FAILED: test8" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: utf8_test complete." << std::endl;

  return 0;
}
