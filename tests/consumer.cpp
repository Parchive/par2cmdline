//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See https://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
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

// Built by tests/consumer_test with only the public include directory on the
// include path and without config.h, the way an embedding application sees
// libpar2.

#include <par2/libpar2.h>

#include <iostream>
#include <string>
#include <vector>

// An embedding application is free to use these names itself.
typedef double u32;
typedef char Result;

int main()
{
  par2::u32 recoveryfilecount = 0;
  if (!par2::ComputeRecoveryFileCount(std::cout, std::cerr, &recoveryfilecount,
                                      par2::scVariable, 4, 1000, 100))
  {
    std::cerr << "FAILED: ComputeRecoveryFileCount" << std::endl;
    return 1;
  }

  std::vector<std::string> extrafiles;
  const par2::Result result = par2::par2repair(std::cout,
                                               std::cerr,
                                               par2::nlSilent,
                                               0,
                                               "",
                                               0,
                                               2,
                                               "nonexistent.par2",
                                               extrafiles,
                                               false,
                                               false,
                                               false,
                                               false,
                                               0);

  if (result != par2::eInsufficientCriticalData)
  {
    std::cerr << "FAILED: expected eInsufficientCriticalData, got "
              << result << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: consumer_test complete." << std::endl;

  return 0;
}
