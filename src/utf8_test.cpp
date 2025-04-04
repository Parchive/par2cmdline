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

using namespace std;
using namespace utf8;

int test1()
{
  string emptyString = "";
  wstring expectedWide = L"";
  wstring actualWide = Utf8ToWide(emptyString);

  string expectedUtf8 = "";
  string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test2()
{
  string asciiString = "Hello, World!";
  wstring expectedWide = L"Hello, World!";
  wstring actualWide = Utf8ToWide(asciiString);

  string expectedUtf8 = "Hello, World!";
  string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test3()
{
  string nonAsciiString = "Привет, мир!";
  wstring expectedWide = L"Привет, мир!";
  wstring actualWide = Utf8ToWide(nonAsciiString);

  string expectedUtf8 = "Привет, мир!";
  string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test4()
{
  string specialCharsString = "This string has: !@#$%^&*()_+=-`~[]{}:;'<>,.?/";
  wstring expectedWide = L"This string has: !@#$%^&*()_+=-`~[]{}:;'<>,.?/";
  wstring actualWide = Utf8ToWide(specialCharsString);

  string expectedUtf8 = "This string has: !@#$%^&*()_+=-`~[]{}:;'<>,.?/";
  string actualUtf8 = WideToUtf8(expectedWide);

  if (actualWide == expectedWide && actualUtf8 == expectedUtf8)
    return 0;

  return 1;
}

int test5()
{
  string multiLangString = "Привет! こんにちは世界! 안녕하세요!";
  wstring expectedWide = L"Привет! こんにちは世界! 안녕하세요!";
  wstring actualWide = Utf8ToWide(multiLangString);

  string expectedUtf8 = "Привет! こんにちは世界! 안녕하세요!";
  string actualUtf8 = WideToUtf8(expectedWide);

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
  wchar_t* wargv[3] = { L"Привет", L"мир", L"!" };
  WideToUtf8ArgsAdapter adapter(3, wargv);
  const char* const* utf8Args = adapter.GetUtf8Args();

  for (int i = 0; i < 3; ++i) {
    if (WideToUtf8(wargv[i]) != string(utf8Args[i]))
    {
      return 1;
    }
  }

  return 0;
}

int test8()
{
  wchar_t* wargv[3] = { L"arg1", nullptr, L"arg3" };
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

    if (WideToUtf8(wargv[i]) != string(utf8Args[i]))
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
    cerr << "FAILED: test1" << endl;
    return 1;
  }

  if (test2())
  {
    cerr << "FAILED: test2" << endl;
    return 1;
  }

  if (test3())
  {
    cerr << "FAILED: test3" << endl;
    return 1;
  }

  if (test4())
  {
    cerr << "FAILED: test4" << endl;
    return 1;
  }

  if (test5())
  {
    cerr << "FAILED: test5" << endl;
    return 1;
  }

  if (test6())
  {
    cerr << "FAILED: test6" << endl;
    return 1;
  }

  if (test7())
  {
    cerr << "FAILED: test7" << endl;
    return 1;
  }

  if (test8())
  {
    cerr << "FAILED: test8" << endl;
    return 1;
  }

  cout << "SUCCESS: utf8_test complete." << endl;

  return 0;
}
