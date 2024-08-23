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

#include "utf8.h"

namespace utf8 {
#if defined(_WIN32) && defined(UNICODE)
std::string narrow(const std::wstring& wstr, UINT cp)
{
	if (wstr.empty()) return std::string();
	int size_needed = WideCharToMultiByte(cp, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
	std::string strTo(size_needed, 0);
	WideCharToMultiByte(cp, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
	return strTo;
}

std::string narrow(const std::wstring& wstr)
{
	return narrow(wstr, CP_UTF8);
}

std::wstring widen(const std::string& str)
{
	if (str.empty()) return std::wstring();
	int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
	std::wstring wstrTo(size_needed, 0);
	MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
	return wstrTo;
}

std::string console(const std::string& str)
{
	return narrow(widen(str), ::GetConsoleOutputCP());
}

// Translate PAR filename Extended ASCII to UTF8
std::string compatible(const std::string& str)
{
	int size_needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str.c_str(), -1, NULL, 0);
	if (size_needed != 0)
	  return str;

	size_needed = MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, str.c_str(), -1, NULL, 0);
	if (size_needed == 0)
		return str;

	WCHAR* utf8Buffer = new WCHAR[size_needed];
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, utf8Buffer, size_needed);	
	size_needed = WideCharToMultiByte(CP_UTF8, 0, utf8Buffer, -1, NULL, 0, NULL, NULL);
	if (size_needed == 0)
	{
		delete[] utf8Buffer;
		return str;
	}

	char* utf8Char = new char[size_needed];
	WideCharToMultiByte(CP_UTF8, 0, utf8Buffer, -1, utf8Char, size_needed, NULL, NULL);
	std::string strResult(utf8Char);

	delete[] utf8Buffer;
	delete[] utf8Char;

	return strResult;
}
#else
  std::string narrow(std::string wstr, unsigned int cp) { return wstr; }
  std::string narrow(std::string wstr) { return wstr; }
  std::string widen(std::string str) { return str; }
  std::string console(std::string str) { return str; }
  std::string compatible(std::string strData) { return strData; }
#endif
}
