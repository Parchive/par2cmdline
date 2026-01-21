//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
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


#include <iostream>
#include <stdlib.h>

#include "libpar2internal.h"
#include "galois.h"


// Galois are finite fields.  (A field with a finite number of elements.)
// A field has:
//    addition
//    multiplication
// such that
//    + and * are associative
//    + and * are commutative
//    exists 0 such that a+0 = 0+a = a
//    exists 1 such that a*1 = 1*a = 1
//    for all a, exists addative inverse -a such that a + -a = 0
//    for all a, exists multiplicative inverse a^-1 such that a * a^-1 = 1
//    * is distributive over +


template<typename gtype, typename utype>
int test_field(const gtype a, const gtype b, const gtype c, const gtype zero, const gtype one, int max_value) {

  if ( (a+b)+c != a+(b+c) ) {
    std::cerr << "addative associativity fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }

  if ( (a*b)*c != a*(b*c) ) {
    std::cerr << "multiplicative associativity fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }


  if ( a+b != b+a ) {
    std::cerr << "addative commutivity fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << std::endl;
    return 1;
  }

  if ( a*b != b*a ) {
    std::cerr << "multiplicative commutivity fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << std::endl;
    return 1;
  }

  if ( a*(b+c) != (a*b)+(a*c) ) {
    std::cerr << "addative associativity fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }


  if ( a+zero != a || zero+a != a ) {
    std::cerr << "addative identity fails for "
	 << ((int) (utype) a) << std::endl;
    return 1;
  }

  if ( a*one != a || one*a != a ) {
    std::cerr << "multiplicative identity fails for "
	 << ((int) (utype) a) << std::endl;
    return 1;
  }


  // search for inverses
  int addative_inverse = -1;
  int multiplicative_inverse = -1;
  for (int i = 0; i < max_value; i++) {
    const gtype other(i);

    //std::cout << ((int) (utype) a) << " + " << ((int) (utype) other) << " = " << ((int) (utype) (a + other)) << std::endl;
    //std::cout << ((a + other == zero) ? "true" : "false") << std::endl;

    if (a + other == zero) {
      if (addative_inverse != -1) {
	std::cerr << "value " << ((int) (utype) a) << " has two addative inverses" << std::endl;
	std::cerr << "   at " << addative_inverse << " and " << i << std::endl;
	return 1;
      }
      addative_inverse = i;
    }

    if (a * other == one) {
      if (multiplicative_inverse != -1) {
	std::cerr << "value " << ((int) (utype) a) << " has two multiplicative inverses" << std::endl;
	std::cerr << "   at " << multiplicative_inverse << " and " << i << std::endl;
	return 1;
      }
      multiplicative_inverse = i;
    }
  }
  if (addative_inverse == -1) {
    std::cerr << "value " << ((int) (utype) a) << " has no addative inverse!" << std::endl;
    return 1;
  }
  if (multiplicative_inverse == -1 && a != zero) {
    std::cerr << "value " << ((int) (utype) a) << " has no multiplicative inverse!" << std::endl;
    return 1;
  }

  return 0;
}



template<typename gtype, typename utype>
int test_field_many(const gtype zero, const gtype one, int max_value) {
  srand(345087209);

  for (int i = 0; i < 256*256; i++) {
    if (test_field<gtype, utype>( gtype(i%max_value), gtype(rand() % max_value), gtype(rand() % max_value), zero, one, max_value)) {
      return 1;
    }
  }

  return 0;
}


int test1() {
  return test_field_many<Galois8, u8>(Galois8(0), Galois8(1), 256);
}

int test2() {
  return test_field_many<Galois16, u16>(Galois16(0), Galois16(1), 256*256);
}


template<typename gtype, typename utype>
int test_operators(const gtype a, const gtype b, const gtype zero, const gtype one, int max_value) {
  gtype c;

  c = a;
  c += b;
  if ( c != a+b ) {
    std::cerr << "+= fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }

  if (a-a != zero ) {
    std::cerr << "- fails for "
	 << ((int) (utype) a) << std::endl;
    return 1;
  }

  c = a;
  c -= a;
  if ( c != zero ) {
    std::cerr << "-= fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }

  c = a;
  c *= b;
  if ( c != a*b ) {
    std::cerr << "*= fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) b) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }

  if (a != zero) {
    if ( a/a != one ) {
      std::cerr << "/ fails for "
	   << ((int) (utype) a) << std::endl;
      return 1;
    }

    c = a;
    c /= a;
    if ( c != one ) {
      std::cerr << "/= fails for "
	   << ((int) (utype) a) << ","
	   << ((int) (utype) c) << std::endl;
      return 1;
    }
  }

  unsigned int power = (unsigned int) (utype) b;
  c = one;
  for (unsigned int i = 0; i < power; i++) {
    c *= a;
  }
  if ((a^power) != c) {
    std::cerr << "^ fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }
  if (a.pow(power) != c) {
    std::cerr << "pow() fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }

  c = a;
  c ^= power;
  if (c != (a^power)) {
    std::cerr << "^= fails for "
	 << ((int) (utype) a) << ","
	 << power << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }

  if (a != zero && b != zero) {
    if ( (a.Log() + b.Log()) % (max_value-1) != (a*b).Log() ) {
      std::cerr << "Log = fails for "
	   << ((int) (utype) a) << ","
	   << ((int) (utype) b) << ","
	   << ((int) a.Log()) << ","
	   << ((int) b.Log()) << ","
	   << ((int) (a*b).Log()) << std::endl;
      return 1;
    }
  }

  c = gtype(gtype(a.Log()).ALog());
  if (c != a) {
    std::cerr << "ALog fails for "
	 << ((int) (utype) a) << ","
	 << ((int) (utype) c) << std::endl;
    return 1;
  }


  return 0;
}


template<typename gtype, typename utype>
int test_operators_many(const gtype zero, const gtype one, int max_value) {
  srand(14531119);

  for (int i = 0; i < 256*256; i++) {
    if (test_operators<gtype, utype>( gtype(i%max_value), gtype(rand() % max_value), zero, one, max_value)) {
	return 1;
    }
  }

  return 0;
}


int test3() {
  return test_operators_many<Galois8, u8>(Galois8(0), Galois8(1), 256);
}

int test4() {
  return test_operators_many<Galois16, u16>(Galois16(0), Galois16(1), 256*256);
}


// test powers of Galois
// 2^pow should be unique for pow in range 1 to N-1.
// and 2^N = 2
// and 2^pow != 0 for any pow

template<typename gtype, typename utype>
int test_powers(const gtype two, const gtype zero, int max_value) {
  gtype g = two;

  int used[256*256];
  for (int i = 0; i < max_value; i++)
    used[i] = 0;

  // mark 0 as used, so we get an error if it reaches it.
  used[0] = -1;

  for (int power = 1; power < max_value; power++) {
    int index = (int) (utype) g;
    if (used[index] != 0) {
      std::cerr << "error at power " << power << " was already used by power " << used[index] << std::endl;
      std::cerr << "g = " << ((int) (utype) g) << std::endl;
      return 1;
    }

    used[index] = power;
    g *= two;
  }

  if (g != two) {
    std::cerr << "error after power " << max_value << std::endl;
    std::cerr << "g = " << ((int) (utype) g) << std::endl;
    std::cerr << "two = " << ((int) (utype) two) << std::endl;
    return 1;
  }

  return 0;
}


int test5() {
  return test_powers<Galois8, u8>(Galois8(2), Galois8(0), 256);
}

int test6() {
  return test_powers<Galois16, u16>(Galois16(2), Galois16(0), 256*256);
}



int main() {
  if (test1()) {
    std::cerr << "FAILED: test1" << std::endl;
    return 1;
  }
  if (test2()) {
    std::cerr << "FAILED: test2" << std::endl;
    return 1;
  }
  if (test3()) {
    std::cerr << "FAILED: test3" << std::endl;
    return 1;
  }
  if (test4()) {
    std::cerr << "FAILED: test4" << std::endl;
    return 1;
  }
  if (test5()) {
    std::cerr << "FAILED: test5" << std::endl;
    return 1;
  }
  if (test6()) {
    std::cerr << "FAILED: test6" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: galois_test complete." << std::endl;

  return 0;
}
