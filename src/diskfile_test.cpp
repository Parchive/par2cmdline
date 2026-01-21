//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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
#include <fstream>

#include "libpar2internal.h"


// The file separator
#ifdef _WIN32
std::string fs("\\");
#else
std::string fs("/");
#endif


// test static functions
int test1() {
  // create test file using C++ functions
  std::ofstream input1;
  input1.open("input1.txt", std::ofstream::out | std::ofstream::binary);
  const char *input1_contents = "diskfile_test test1 input1.txt";
  input1 << input1_contents;
  input1.close();

  std::ofstream input2;
  input2.open("input2.txt", std::ofstream::out | std::ofstream::binary);
  const char *input2_contents = "diskfile_test test1 input2.txt";
  input2 << input2_contents;
  input2.close();

  if (DiskFile::FileExists("definitely_not_here")) {
    std::cout << "said file exists when it doesn't" << std::endl;
    return 1;
  }
  if (!DiskFile::FileExists("input1.txt")) {
    std::cout << "said file doesn't exists when it does" << std::endl;
    return 1;
  }

  if (DiskFile::GetFileSize("input1.txt") != strlen(input1_contents)) {
    std::cout << "GetFileSize returned wrong value" << std::endl;
    return 1;
  }

  std::unique_ptr< std::list<std::string> > files = DiskFile::FindFiles(".", "input1.txt", false);
  if (files->size() != 1 || *(files->begin()) != "." + fs + "input1.txt") {
    std::cout << "FindFiles failed on exact name" << std::endl;
    std::cout << "   size=" << files->size() << std::endl;
    for (std::list<std::string>::iterator fn = files->begin();
	 fn != files->end();
	 fn++) {
      std::cout << "   " << *fn << std::endl;
    }

    return 1;
  }
  files = DiskFile::FindFiles(".", "input?.txt", false);
  if (files->size() != 2
      || find(files->begin(), files->end(), std::string("." + fs + "input1.txt")) == files->end()
      || find(files->begin(), files->end(), std::string("." + fs + "input2.txt")) == files->end()) {
    std::cout << "FindFiles failed on ?" << std::endl;
    return 1;
  }
  files = DiskFile::FindFiles(".", "input1?.txt", false);
  if (!files->empty()) {
    std::cout << "FindFiles failed on empty ?" << std::endl;
    return 1;
  }
  files = DiskFile::FindFiles(".", "input*.txt", false);
  if (files->size() != 2
      || find(files->begin(), files->end(), std::string("." + fs + "input1.txt")) == files->end()
      || find(files->begin(), files->end(), std::string("." + fs + "input2.txt")) == files->end()) {
    std::cout << "FindFiles failed on *" << std::endl;
    return 1;
  }
  files = DiskFile::FindFiles(".", "input1*.txt", false);
  if (files->size() != 1 || *(files->begin()) != "." + fs + "input1.txt") {
    std::cout << "FindFiles failed on empty *" << std::endl;
//TODO: Fix bug and uncomment this
//    return 1;
  }
  files = DiskFile::FindFiles(".", "i*p*t*.txt", false);
  if (files->size() != 2
      || find(files->begin(), files->end(), std::string("." + fs + "input1.txt")) == files->end()
      || find(files->begin(), files->end(), std::string("." + fs + "input2.txt")) == files->end()) {
    std::cout << "FindFiles failed on multiple *" << std::endl;
//TODO: Fix bug and uncomment this
//    return 1;
  }


  { // scope to hide variable names
    std::string path, name;
    DiskFile::SplitFilename("input1.txt", path, name);
    if (path != "." + fs
	|| name != "input1.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input1.txt" << std::endl;
      return 1;
    }
    // NB: Keeping value of path, name to see if overwritten
    DiskFile::SplitFilename("." + fs + "input1.txt", path, name);
    if (path != "." + fs
	|| name != "input1.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input1.txt" << std::endl;
      return 1;
    }
    DiskFile::SplitFilename("dir" + fs + "input1.txt", path, name);
    if (path != "dir" + fs
	|| name != "input1.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input1.txt" << std::endl;
      return 1;
    }
    DiskFile::SplitFilename("multiple" + fs + "dirs" + fs + "input1.txt", path, name);
    if (path != "multiple" + fs + "dirs" + fs
	|| name != "input1.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input1.txt" << std::endl;
      return 1;
    }
    DiskFile::SplitFilename(fs + "root_dir" + fs + "input1.txt", path, name);
    if (path != fs + "root_dir" + fs
	|| name != "input1.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input1.txt" << std::endl;
      return 1;
    }
  }


  {
    std::string name;
    DiskFile::SplitRelativeFilename("root" + fs + "dir" + fs + "input1.txt",
			    "root" + fs + "dir" + fs,
			    name);
    if (name != "input1.txt") {
      std::cout << "SplitRelativeFilename failed for full path" << std::endl;
      return 1;
    }
    // intentionally reusing name to see if it is overwritten
    DiskFile::SplitRelativeFilename("root" + fs + "dir" + fs + "input1.txt",
			    "root" + fs,
			    name);
    if (name != "dir" + fs + "input1.txt") {
      std::cout << "SplitRelativeFilename failed for partial path" << std::endl;
      return 1;
    }
  }


  { // scope to hide variable names
    std::string path_and_name1 = DiskFile::GetCanonicalPathname("input1.txt");
    std::string path1, name1;
    DiskFile::SplitFilename(path_and_name1, path1, name1);
    if (path_and_name1 != path1 + name1
#ifdef _WIN32
	|| path1.at(1) != ':' || path1.at(2) != fs.at(0)
#else
	|| path1.at(0) != fs.at(0)
#endif
	|| name1 != "input1.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input1.txt" << std::endl;
      return 1;
    }
    std::string path_and_name2 = DiskFile::GetCanonicalPathname("input2.txt");
    std::string path2, name2;
    DiskFile::SplitFilename(path_and_name2, path2, name2);
    if (path_and_name2 != path2 + name2
	|| path2 != path1
	|| name2 != "input2.txt") {
      std::cout << "CanonicalPathname + SplitFilename failed on input2.txt" << std::endl;
      return 1;
    }

    // check that ././input1.txt is the same as input1.txt
    std::string path_and_name3 = DiskFile::GetCanonicalPathname("." + fs + "." + fs + "input1.txt");
    if (path_and_name3 != path_and_name1) {
      std::cout << "CanonicalPathname + SplitFilename failed on ././input.txt" << std::endl;
      return 1;
    }
  }


  // delete test files using C++ function
  remove("input1.txt");
  remove("input2.txt");

  return 0;
}


// test non-static functions
int test2() {
  // create test file using C++ functions
  std::ofstream input1;
  input1.open("input1.txt", std::ofstream::out | std::ofstream::binary);
  const char *input1_contents = "diskfile_test test1 input1.txt";
  input1 << input1_contents;
  input1.close();

  // read input1.txt
  {
    DiskFile diskfile(std::cout, std::cerr);
    if (diskfile.IsOpen()) {
      std::cout << "IsOpen failed 1" << std::endl;
      return 1;
    }
    if (diskfile.Exists()) {
      std::cout << "Exists failed 1" << std::endl;
      return 1;
    }
    if (!diskfile.Open("input1.txt")) {
      std::cout << "Open failed" << std::endl;
      return 1;
    }
    if (!diskfile.IsOpen()) {
      std::cout << "IsOpen failed 2" << std::endl;
      return 1;
    }
    if (!diskfile.Exists()) {
      std::cout << "Exists failed 2" << std::endl;
      return 1;
    }
    if (diskfile.FileName() != "input1.txt") {
      std::cout << "FileName failed" << std::endl;
      return 1;
    }
    if (diskfile.FileSize() != strlen(input1_contents)) {
      std::cout << "FileSize failed" << std::endl;
      return 1;
    }
    const size_t buffer_len = strlen(input1_contents)+1;  // for end-of-std::string
    u8 *buffer = new u8[buffer_len];
    // put end-of-std::string in buffer.
    buffer[buffer_len-1] = '\0';

    if (!diskfile.Read(0, buffer, buffer_len - 1)) {
      std::cout << "Read whole file returned false" << std::endl;
      return 1;
    }
    if (std::string(input1_contents) != (char *) buffer) {
      std::cout << "Read did not read contents correctly" << std::endl;
      std::cout << "read     \"" << buffer << "\"" << std::endl;
      std::cout << "expected \"" << input1_contents << "\"" << std::endl;
      return 1;
    }

    // random reads
    srand(345087209);
    for (int i = 0; i < 100; i++) {
      // length is always at least 1.
      const u64 offset = rand() % (buffer_len-1-1);
      const size_t length = 1+(rand() % (buffer_len - 1 - offset-1));
      if (!diskfile.Read(offset, buffer + offset, length)) {
	std::cout << "Read partial file returned false" << std::endl;
	std::cout << "   offset=" << offset << std::endl;
	std::cout << "   length=" << length << std::endl;
	std::cout << "   strlen=" << strlen(input1_contents) << std::endl;
	return 1;
      }
      if (std::string(input1_contents) != (char *) buffer) {
	std::cout << "Random Read did not read contents correctly" << std::endl;
	return 1;
      }
    }

    if (!diskfile.IsOpen()) {
      std::cout << "IsOpen failed 3" << std::endl;
      return 1;
    }

    diskfile.Close();
    if (diskfile.IsOpen()) {
      std::cout << "IsOpen failed 4" << std::endl;
      return 1;
    }

    // reopen!
    if (!diskfile.Open()) {
      std::cout << "Open failed 2" << std::endl;
      return 1;
    }
    if (!diskfile.IsOpen()) {
      std::cout << "IsOpen failed 5" << std::endl;
      return 1;
    }

    diskfile.Close();
    if (diskfile.IsOpen()) {
      std::cout << "IsOpen failed 6" << std::endl;
      return 1;
    }

    delete [] buffer;
  }


  {
    std::cout << "create input2.txt, move it to input3.txt, delete it." << std::endl;

    const char *input2_contents = "diskfile_test test3 input2.txt is longer";

    DiskFile diskfile(std::cout, std::cerr);
    if (diskfile.IsOpen()) {
      std::cout << "IsOpen failed 1" << std::endl;
      return 1;
    }
    if (diskfile.Exists()) {
      std::cout << "Exists failed 1" << std::endl;
      return 1;
    }
    if (!diskfile.Create("input2.txt", strlen(input2_contents))) {
      std::cout << "Create failed" << std::endl;
      return 1;
    }
    if (!diskfile.IsOpen()) {
      std::cout << "IsOpen failed 2" << std::endl;
      return 1;
    }
    if (!diskfile.Exists()) {
      std::cout << "Exists failed 2" << std::endl;
      return 1;
    }
    if (diskfile.FileSize() != strlen(input2_contents)) {
      std::cout << "FileSize failed 1" << std::endl;
      return 1;
    }
    if (diskfile.FileName() != "input2.txt") {
      std::cout << "FileName failed 1" << std::endl;
      return 1;
    }
    if (!diskfile.Write(0, input2_contents, strlen(input2_contents))) {
      std::cout << "Write failed 1" << std::endl;
      return 1;
    }

    /*    // confirm write with read

    const size_t buffer_len = strlen(input2_contents)+1;  // for end-of-std::string
    u8 *buffer = new u8[buffer_len];
    // put end-of-std::string in buffer.
    buffer[buffer_len-1] = '\0';
    if (!diskfile.Read(0, buffer, buffer_len - 1)) {
      std::cout << "Read whole file returned false 1" << std::endl;
      return 1;
    }
    if (std::string(input2_contents) != (char *) buffer) {
      std::cout << "Read did not read contents correctly 1" << std::endl;
      return 1;
    }
    */

    diskfile.Close();
    if (diskfile.IsOpen()) {
      std::cout << "IsOpen failed 3" << std::endl;
      return 1;
    }

    // Rename from input2.txt to input3.txt
    if (!diskfile.Rename("input3.txt")) {
      std::cout << "Rename failed 1" << std::endl;
      return 1;
    }
    // C's remove returns 0 on success and non-0 on failure
    if (remove("input2.txt") == 0) {
      std::cout << "input2.txt exists after deletion!" << std::endl;
      return 1;
    }
    if (diskfile.FileName() != "input3.txt") {
      std::cout << "FileName failed 1" << std::endl;
      return 1;
    }
    if (!diskfile.Exists()) {
      std::cout << "Exists failed 3" << std::endl;
      return 1;
    }

    /*
    // read again
    if (!diskfile.Read(0, buffer, buffer_len - 1)) {
      std::cout << "Read whole file returned false 2" << std::endl;
      return 1;
    }
    if (std::string(input2_contents) != (char *) buffer) {
      std::cout << "Read did not read contents correctly 2" << std::endl;
      return 1;
    }
    */

    if (!diskfile.Delete()) {
      std::cout << "Delete failed 1" << std::endl;
      return 1;
    }
    if (diskfile.Exists()) {
      std::cout << "Exists failed 4" << std::endl;
      return 1;
    }

    // C's remove returns 0 on success and non-0 on failure
    if (remove("input3.txt") == 0) {
      std::cout << "input3.txt exists after deletion!" << std::endl;
      return 1;
    }
  }

  // NOTE: C++ does not have a generic function to remove directories.
  // So, this test does not create subdirectories.
  //
  // CreateParentDirectory()

  // random write + read
  {
    std::cout << "create input2.txt, write and read it." << std::endl;

    const char *input2_contents = "diskfile_test test3 input2.txt is longer";
    size_t buffer_len = strlen(input2_contents);

    srand(23461119);
    for (size_t blocksize = 1; blocksize < buffer_len; blocksize *=2) {
      { // scope for variables used in writing
	DiskFile diskfile(std::cout, std::cerr);

	if (!diskfile.Create("input2.txt", strlen(input2_contents))) {
	  std::cout << "Create failed" << std::endl;
	  return 1;
	}

	int blockcount = (buffer_len + (blocksize - 1))/blocksize;
	int *blockorder = new int[blockcount];
	for (int i = 0; i < blockcount; i++)
	  blockorder[i] = i;
	// shuffle
	for (int i = 0; i < blockcount-1; i++) {
	  int other_index = (rand() % (blockcount-(i+1))) + 1;
	  int tmp = blockorder[other_index];
	  blockorder[other_index] = blockorder[i];
	  blockorder[i] = tmp;
	}
	// write blocks
	for (int i = 0; i < blockcount; i++) {
	  const u64 offset = blocksize*blockorder[i];
	  // Calculate actual bytes to write (last block may be smaller)
	  size_t write_len = (offset + blocksize > buffer_len) ? (buffer_len - offset) : blocksize;
	  if (!diskfile.Write(offset, input2_contents + offset, write_len)) {
	    std::cout << "Write failed 1" << std::endl;
	    delete [] blockorder;
	    return 1;
	  }
	}

	diskfile.Close();
	delete [] blockorder;
      }

      { // scope for variables used in reading.
	DiskFile diskfile(std::cout, std::cerr);

	if (!diskfile.Open("input2.txt", strlen(input2_contents))) {
	  std::cout << "Open failed 1" << std::endl;
	  return 1;
	}

	// add one more char, for end-of-std::string
	u8 *buffer = new u8[buffer_len + 1];
	buffer[buffer_len] = '\0';

	if (!diskfile.Read(0, buffer, buffer_len)) {
	  std::cout << "Read whole file returned false 2" << std::endl;
	  return 1;
	}
	if (std::string(input2_contents) != (char *) buffer) {
	  std::cout << "Read did not read contents correctly 2" << std::endl;
	  return 1;
	}

	delete [] buffer;
      }

      if (remove("input2.txt") != 0) {
	std::cout << "input2.txt did not exist" << std::endl;
	return 1;
      }
    }
  }

  // delete test files using C++ function
  remove("input1.txt");

  return 0;
}


// test DiskFileMap
int test3() {
  std::ofstream input1;
  input1.open("input1.txt", std::ofstream::out | std::ofstream::binary);
  const char *input1_contents = "diskfile_test test3 input1.txt";
  input1 << input1_contents;
  input1.close();


  // Hard to screw up.  Except double insert?
  DiskFileMap dfm;

  if (dfm.Find("input1.txt") != NULL) {
    std::cout << "Find succeeded when it shouldn't have" << std::endl;
    return 1;
  }

  DiskFile df1(std::cout, std::cerr);
  df1.Open("input1.txt");
  if (!dfm.Insert(&df1)) {
    std::cout << "Insert failed" << std::endl;
    return 1;
  }
  if (dfm.Find("input1.txt") != &df1) {
    std::cout << "Find failed when it shouldn't have" << std::endl;
    return 1;
  }

  DiskFile df2(std::cout, std::cerr);
  df2.Open("input1.txt");
  if (dfm.Insert(&df2)) {
    std::cout << "Insert succeeded when it shouldn't have" << std::endl;
    return 1;
  }

  if (dfm.Find("input1.txt") != &df1) {
    std::cout << "Find failed when it shouldn't have 2" << std::endl;
    return 1;
  }

  dfm.Remove(&df1);

  if (dfm.Find("input1.txt") != NULL) {
    std::cout << "Find succeeded when it shouldn't have 2" << std::endl;
    return 1;
  }

  // delete test files using C++ function
  remove("input1.txt");

  return 0;
}

// test FileSizeCache
int test4() {
  std::ofstream input1;
  input1.open("input1.txt", std::ofstream::out | std::ofstream::binary);
  const char *input1_contents = "diskfile_test test3 input1.txt";
  input1 << input1_contents;
  input1.close();

  std::ofstream input2;
  input2.open("input2.txt", std::ofstream::out | std::ofstream::binary);
  const char *input2_contents = "diskfile_test test3 input2.txt is longer";
  input2 << input2_contents;
  input2.close();

  // should time this vs. DiskFile::FileSize()

  FileSizeCache cache;
  for (int i = 0; i < 1000; i++) {
    if (cache.get("input1.txt") != strlen(input1_contents)) {
      std::cout << "FileSizeCache failed" << std::endl;
      return 1;
    }
    if (cache.get("input2.txt") != strlen(input2_contents)) {
      std::cout << "FileSizeCache failed 2" << std::endl;
      return 1;
    }
  }

  // delete test files using C++ function
  remove("input1.txt");
  remove("input2.txt");

  return 0;
}


// test that we cannot create a file if one already exists.
int test5() {
  std::ofstream input1;
  input1.open("input1.txt", std::ofstream::out | std::ofstream::binary);
  const char *input1_contents = "diskfile_test test3 input1.txt";
  input1 << input1_contents;
  input1.close();

  DiskFile diskfile(std::cout, std::cerr);
  if (diskfile.Create("input1.txt", strlen(input1_contents))) {
    std::cout << "Create succeeded when file already existed!" << std::endl;
    return 1;
  }

  // delete test files using C++ function
  remove("input1.txt");
  return 0;
}


// Testing Read()/Write() where length > maxlength
// To do this, the functions were modified to take
// maxlength as a parameter.  In production code,
// the default maxlength is like 2GB, which is too
// large for fast tests.
int test6() {
  const char *input1_contents = "diskfile_test test6 input1.txt";

  {
    DiskFile diskfile(std::cout, std::cerr);
    if (!diskfile.Create("input1.txt", strlen(input1_contents))) {
      std::cout << "Create failed!" << std::endl;
      return 1;
    }

    if (!diskfile.Write(0, input1_contents, strlen(input1_contents), 2)) {
      std::cout << "Write failed 1" << std::endl;
      return 1;
    }

    diskfile.Close();
  }

  {
    DiskFile diskfile(std::cout, std::cerr);

    if (!diskfile.Open("input1.txt")) {
      std::cout << "Open failed" << std::endl;
      return 1;
    }

    const size_t buffer_len = strlen(input1_contents)+1;  // for end-of-std::string
    u8 *buffer = new u8[buffer_len];
    // put end-of-std::string in buffer.
    buffer[buffer_len-1] = '\0';

    if (!diskfile.Read(0, buffer, buffer_len - 1, 2)) {
      std::cout << "Read whole file returned false" << std::endl;
      return 1;
    }

    if (std::string(input1_contents) != (char *) buffer) {
      std::cout << "Read did not read contents correctly" << std::endl;
      std::cout << "read     \"" << buffer << "\"" << std::endl;
      std::cout << "expected \"" << input1_contents << "\"" << std::endl;
      return 1;
    }

    diskfile.Close();

    remove("input1.txt");
  }


  const char *input2_contents = "diskfile_test test6 input2.txt is longer";

  // try again, writing mid-file with different maxlength.
  {
    DiskFile diskfile(std::cout, std::cerr);
    if (!diskfile.Create("input2.txt", strlen(input2_contents))) {
      std::cout << "Create 2 failed." << std::endl;
      return 1;
    }

    size_t midpoint = strlen(input2_contents);
    if (!diskfile.Write(midpoint, input2_contents + midpoint, strlen(input2_contents) - midpoint, 3)) {
      std::cout << "Write failed 2" << std::endl;
      return 1;
    }
    if (!diskfile.Write(0, input2_contents, midpoint, 4)) {
      std::cout << "Write failed 3" << std::endl;
      return 1;
    }

    diskfile.Close();
  }


  {
    DiskFile diskfile(std::cout, std::cerr);

    if (!diskfile.Open("input2.txt")) {
      std::cout << "Open failed" << std::endl;
      return 1;
    }

    const size_t buffer_len = strlen(input2_contents)+1;  // for end-of-std::string
    u8 *buffer = new u8[buffer_len];
    // put end-of-std::string in buffer.
    buffer[buffer_len-1] = '\0';


    size_t midpoint = strlen(input2_contents) - 2;
    if (!diskfile.Read(midpoint, buffer + midpoint, strlen(input2_contents) - midpoint, 4)) {
      std::cout << "Read second half of file returned false" << std::endl;
      return 1;
    }
    if (!diskfile.Read(0, buffer, midpoint, 3)) {
      std::cout << "Read first half of file returned false" << std::endl;
      return 1;
    }

    if (std::string(input2_contents) != (char *) buffer) {
      std::cout << "Read did not read contents correctly" << std::endl;
      std::cout << "read     \"" << buffer << "\"" << std::endl;
      std::cout << "expected \"" << input2_contents << "\"" << std::endl;
      return 1;
    }

    diskfile.Close();

    remove("input2.txt");
  }

  return 0;
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

  std::cout << "SUCCESS: diskfile_test complete." << std::endl;

  return 0;
}
