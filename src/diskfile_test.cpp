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
string fs("\\");
#else
string fs("/");
#endif  


// test static functions
int test1() {
  // create test file using C++ functions
  ofstream input1;
  input1.open("input1.txt", ofstream::out | ofstream::binary);
  const char *input1_contents = "diskfile_test test1 input1.txt";
  input1 << input1_contents;
  input1.close();

  ofstream input2;
  input2.open("input2.txt", ofstream::out | ofstream::binary);
  const char *input2_contents = "diskfile_test test1 input2.txt";
  input2 << input2_contents;
  input2.close();

  if (DiskFile::FileExists("definitely_not_here")) {
    cout << "said file exists when it doesn't" << endl;
    return 1;
  }
  if (!DiskFile::FileExists("input1.txt")) {
    cout << "said file doesn't exists when it does" << endl;
    return 1;
  }

  if (DiskFile::GetFileSize("input1.txt") != strlen(input1_contents)) {
    cout << "GetFileSize returned wrong value" << endl;
    return 1;
  }

  std::unique_ptr< list<string> > files = DiskFile::FindFiles(".", "input1.txt", false);
  if (files->size() != 1 || *(files->begin()) != "." + fs + "input1.txt") {
    cout << "FindFiles failed on exact name" << endl;
    cout << "   size=" << files->size() << endl;
    for (list<string>::iterator fn = files->begin();
	 fn != files->end();
	 fn++) {
      cout << "   " << *fn << endl;
    }

    return 1;
  }
  files = DiskFile::FindFiles(".", "input?.txt", false);
  if (files->size() != 2
      || find(files->begin(), files->end(), string("." + fs + "input1.txt")) == files->end()
      || find(files->begin(), files->end(), string("." + fs + "input2.txt")) == files->end()) {
    cout << "FindFiles failed on ?" << endl;
    return 1;
  }
  files = DiskFile::FindFiles(".", "input1?.txt", false);
  if (!files->empty()) {
    cout << "FindFiles failed on empty ?" << endl;
    return 1;
  }
  files = DiskFile::FindFiles(".", "input*.txt", false);
  if (files->size() != 2
      || find(files->begin(), files->end(), string("." + fs + "input1.txt")) == files->end()
      || find(files->begin(), files->end(), string("." + fs + "input2.txt")) == files->end()) {
    cout << "FindFiles failed on *" << endl;
    return 1;
  }
  files = DiskFile::FindFiles(".", "input1*.txt", false);
  if (files->size() != 1 || *(files->begin()) != "." + fs + "input1.txt") {
    cout << "FindFiles failed on empty *" << endl;
//TODO: Fix bug and uncomment this
//    return 1;
  }
  files = DiskFile::FindFiles(".", "i*p*t*.txt", false);
  if (files->size() != 2
      || find(files->begin(), files->end(), string("." + fs + "input1.txt")) == files->end()
      || find(files->begin(), files->end(), string("." + fs + "input2.txt")) == files->end()) {
    cout << "FindFiles failed on multiple *" << endl;
//TODO: Fix bug and uncomment this
//    return 1;
  }


  { // scope to hide variable names
    string path, name;
    DiskFile::SplitFilename("input1.txt", path, name);
    if (path != "." + fs
	|| name != "input1.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input1.txt" << endl;
      return 1;
    }
    // NB: Keeping value of path, name to see if overwritten
    DiskFile::SplitFilename("." + fs + "input1.txt", path, name);
    if (path != "." + fs
	|| name != "input1.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input1.txt" << endl;
      return 1;
    }
    DiskFile::SplitFilename("dir" + fs + "input1.txt", path, name);
    if (path != "dir" + fs
	|| name != "input1.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input1.txt" << endl;
      return 1;
    }
    DiskFile::SplitFilename("multiple" + fs + "dirs" + fs + "input1.txt", path, name);
    if (path != "multiple" + fs + "dirs" + fs
	|| name != "input1.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input1.txt" << endl;
      return 1;
    }
    DiskFile::SplitFilename(fs + "root_dir" + fs + "input1.txt", path, name);
    if (path != fs + "root_dir" + fs
	|| name != "input1.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input1.txt" << endl;
      return 1;
    }
  }


  {
    string name;
    DiskFile::SplitRelativeFilename("root" + fs + "dir" + fs + "input1.txt",
			    "root" + fs + "dir" + fs,
			    name);
    if (name != "input1.txt") {
      cout << "SplitRelativeFilename failed for full path" << endl;
      return 1;
    }
    // intentionally reusing name to see if it is overwritten
    DiskFile::SplitRelativeFilename("root" + fs + "dir" + fs + "input1.txt",
			    "root" + fs,
			    name);
    if (name != "dir" + fs + "input1.txt") {
      cout << "SplitRelativeFilename failed for partial path" << endl;
      return 1;
    }
  }

  
  { // scope to hide variable names
    string path_and_name1 = DiskFile::GetCanonicalPathname("input1.txt");
    string path1, name1;
    DiskFile::SplitFilename(path_and_name1, path1, name1);
    if (path_and_name1 != path1 + name1
#ifdef _WIN32
	|| path1.at(1) != ':' || path1.at(2) != fs.at(0)
#else
	|| path1.at(0) != fs.at(0)
#endif
	|| name1 != "input1.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input1.txt" << endl;
      return 1;
    }
    string path_and_name2 = DiskFile::GetCanonicalPathname("input2.txt");
    string path2, name2;
    DiskFile::SplitFilename(path_and_name2, path2, name2);
    if (path_and_name2 != path2 + name2
	|| path2 != path1
	|| name2 != "input2.txt") {
      cout << "CanonicalPathname + SplitFilename failed on input2.txt" << endl;
      return 1;
    }

    // check that ././input1.txt is the same as input1.txt
    string path_and_name3 = DiskFile::GetCanonicalPathname("." + fs + "." + fs + "input1.txt");
    if (path_and_name3 != path_and_name1) {
      cout << "CanonicalPathname + SplitFilename failed on ././input.txt" << endl;
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
  ofstream input1;
  input1.open("input1.txt", ofstream::out | ofstream::binary);
  const char *input1_contents = "diskfile_test test1 input1.txt";
  input1 << input1_contents;
  input1.close();

  // read input1.txt
  {
    DiskFile diskfile(cout, cerr);
    if (diskfile.IsOpen()) {
      cout << "IsOpen failed 1" << endl;
      return 1;
    }
    if (diskfile.Exists()) {
      cout << "Exists failed 1" << endl;
      return 1;
    }
    if (!diskfile.Open("input1.txt")) {
      cout << "Open failed" << endl;
      return 1;
    }
    if (!diskfile.IsOpen()) {
      cout << "IsOpen failed 2" << endl;
      return 1;
    }
    if (!diskfile.Exists()) {
      cout << "Exists failed 2" << endl;
      return 1;
    }
    if (diskfile.FileName() != "input1.txt") {
      cout << "FileName failed" << endl;
      return 1;
    }
    if (diskfile.FileSize() != strlen(input1_contents)) {
      cout << "FileSize failed" << endl;
      return 1;
    }
    const size_t buffer_len = strlen(input1_contents)+1;  // for end-of-string
    u8 *buffer = new u8[buffer_len];
    // put end-of-string in buffer.
    buffer[buffer_len-1] = '\0';

    if (!diskfile.Read(0, buffer, buffer_len - 1)) {
      cout << "Read whole file returned false" << endl;
      return 1;
    }
    if (string(input1_contents) != (char *) buffer) {
      cout << "Read did not read contents correctly" << endl;
      cout << "read     \"" << buffer << "\"" << endl;
      cout << "expected \"" << input1_contents << "\"" << endl;
      return 1;
    }

    // random reads
    srand(345087209);
    for (int i = 0; i < 100; i++) {
      // length is always at least 1.
      const u64 offset = rand() % (buffer_len-1-1);
      const size_t length = 1+(rand() % (buffer_len - 1 - offset-1));
      if (!diskfile.Read(offset, buffer + offset, length)) {
	cout << "Read partial file returned false" << endl;
	cout << "   offset=" << offset << endl;
	cout << "   length=" << length << endl;
	cout << "   strlen=" << strlen(input1_contents) << endl;
	return 1;
      }
      if (string(input1_contents) != (char *) buffer) {
	cout << "Random Read did not read contents correctly" << endl;
	return 1;
      }
    }
      
    if (!diskfile.IsOpen()) {
      cout << "IsOpen failed 3" << endl;
      return 1;
    }

    diskfile.Close();
    if (diskfile.IsOpen()) {
      cout << "IsOpen failed 4" << endl;
      return 1;
    }

    // reopen!
    if (!diskfile.Open()) {
      cout << "Open failed 2" << endl;
      return 1;
    }
    if (!diskfile.IsOpen()) {
      cout << "IsOpen failed 5" << endl;
      return 1;
    }

    diskfile.Close();
    if (diskfile.IsOpen()) {
      cout << "IsOpen failed 6" << endl;
      return 1;
    }

    delete [] buffer;
  }


  {
    cout << "create input2.txt, move it to input3.txt, delete it." << endl;
    
    const char *input2_contents = "diskfile_test test3 input2.txt is longer";

    DiskFile diskfile(cout, cerr);
    if (diskfile.IsOpen()) {
      cout << "IsOpen failed 1" << endl;
      return 1;
    }
    if (diskfile.Exists()) {
      cout << "Exists failed 1" << endl;
      return 1;
    }
    if (!diskfile.Create("input2.txt", strlen(input2_contents))) {
      cout << "Create failed" << endl;
      return 1;
    }
    if (!diskfile.IsOpen()) {
      cout << "IsOpen failed 2" << endl;
      return 1;
    }
    if (!diskfile.Exists()) {
      cout << "Exists failed 2" << endl;
      return 1;
    }
    if (diskfile.FileSize() != strlen(input2_contents)) {
      cout << "FileSize failed 1" << endl;
      return 1;
    }
    if (diskfile.FileName() != "input2.txt") {
      cout << "FileName failed 1" << endl;
      return 1;
    }
    if (!diskfile.Write(0, input2_contents, strlen(input2_contents))) {
      cout << "Write failed 1" << endl;
      return 1;
    }

    /*    // confirm write with read
    
    const size_t buffer_len = strlen(input2_contents)+1;  // for end-of-string
    u8 *buffer = new u8[buffer_len];
    // put end-of-string in buffer.
    buffer[buffer_len-1] = '\0';
    if (!diskfile.Read(0, buffer, buffer_len - 1)) {
      cout << "Read whole file returned false 1" << endl;
      return 1;
    }
    if (string(input2_contents) != (char *) buffer) {
      cout << "Read did not read contents correctly 1" << endl;
      return 1;
    }
    */

    diskfile.Close();
    if (diskfile.IsOpen()) {
      cout << "IsOpen failed 3" << endl;
      return 1;
    }
    
    // Rename from input2.txt to input3.txt
    if (!diskfile.Rename("input3.txt")) {
      cout << "Rename failed 1" << endl;
      return 1;
    }
    // C's remove returns 0 on success and non-0 on failure
    if (remove("input2.txt") == 0) {
      cout << "input2.txt exists after deletion!" << endl;
      return 1;
    }
    if (diskfile.FileName() != "input3.txt") {
      cout << "FileName failed 1" << endl;
      return 1;
    }
    if (!diskfile.Exists()) {
      cout << "Exists failed 3" << endl;
      return 1;
    }

    /*    
    // read again
    if (!diskfile.Read(0, buffer, buffer_len - 1)) {
      cout << "Read whole file returned false 2" << endl;
      return 1;
    }
    if (string(input2_contents) != (char *) buffer) {
      cout << "Read did not read contents correctly 2" << endl;
      return 1;
    }
    */

    if (!diskfile.Delete()) {
      cout << "Delete failed 1" << endl;
      return 1;
    }
    if (diskfile.Exists()) {
      cout << "Exists failed 4" << endl;
      return 1;
    }
    
    // C's remove returns 0 on success and non-0 on failure
    if (remove("input3.txt") == 0) {
      cout << "input3.txt exists after deletion!" << endl;
      return 1;
    }
  }

  // NOTE: C++ does not have a generic function to remove directories.
  // So, this test does not create subdirectories.
  //
  // CreateParentDirectory()

  // random write + read
  { 
    cout << "create input2.txt, write and read it." << endl;
    
    const char *input2_contents = "diskfile_test test3 input2.txt is longer";
    size_t buffer_len = strlen(input2_contents);
    
    srand(23461119);
    for (size_t blocksize = 1; blocksize < buffer_len; blocksize *=2) {
      { // scope for variables used in writing
	DiskFile diskfile(cout, cerr);
	
	if (!diskfile.Create("input2.txt", strlen(input2_contents))) {
	  cout << "Create failed" << endl;
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
	  if (!diskfile.Write(offset, input2_contents + offset, blocksize)) {
	    cout << "Write failed 1" << endl;
	    delete [] blockorder;
	    return 1;
	  }
	}

	diskfile.Close();
	delete [] blockorder;
      }

      { // scope for variables used in reading.
	DiskFile diskfile(cout, cerr);

	if (!diskfile.Open("input2.txt", strlen(input2_contents))) {
	  cout << "Open failed 1" << endl;
	  return 1;
	}

	// add one more char, for end-of-string
	u8 *buffer = new u8[buffer_len + 1];
	buffer[buffer_len] = '\0';

	if (!diskfile.Read(0, buffer, buffer_len)) {
	  cout << "Read whole file returned false 2" << endl;
	  return 1;
	}
	if (string(input2_contents) != (char *) buffer) {
	  cout << "Read did not read contents correctly 2" << endl;
	  return 1;
	}

	delete [] buffer;
      }

      if (remove("input2.txt") != 0) {
	cout << "input2.txt did not exist" << endl;
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
  ofstream input1;
  input1.open("input1.txt", ofstream::out | ofstream::binary);
  const char *input1_contents = "diskfile_test test3 input1.txt";
  input1 << input1_contents;
  input1.close();


  // Hard to screw up.  Except double insert?
  DiskFileMap dfm;

  if (dfm.Find("input1.txt") != NULL) {
    cout << "Find succeeded when it shouldn't have" << endl;
    return 1;
  }
  
  DiskFile df1(cout, cerr);
  df1.Open("input1.txt");
  if (!dfm.Insert(&df1)) {
    cout << "Insert failed" << endl;
    return 1;
  }
  if (dfm.Find("input1.txt") != &df1) {
    cout << "Find failed when it shouldn't have" << endl;
    return 1;
  }
    
  DiskFile df2(cout, cerr);
  df2.Open("input1.txt");
  if (dfm.Insert(&df2)) {
    cout << "Insert succeeded when it shouldn't have" << endl;
    return 1;
  }
  
  if (dfm.Find("input1.txt") != &df1) {
    cout << "Find failed when it shouldn't have 2" << endl;
    return 1;
  }

  dfm.Remove(&df1);

  if (dfm.Find("input1.txt") != NULL) {
    cout << "Find succeeded when it shouldn't have 2" << endl;
    return 1;
  }
  
  // delete test files using C++ function
  remove("input1.txt");

  return 0;
}

// test FileSizeCache
int test4() {
  ofstream input1;
  input1.open("input1.txt", ofstream::out | ofstream::binary);
  const char *input1_contents = "diskfile_test test3 input1.txt";
  input1 << input1_contents;
  input1.close();

  ofstream input2;
  input2.open("input2.txt", ofstream::out | ofstream::binary);
  const char *input2_contents = "diskfile_test test3 input2.txt is longer";
  input2 << input2_contents;
  input2.close();

  // should time this vs. DiskFile::FileSize()

  FileSizeCache cache;
  for (int i = 0; i < 1000; i++) {
    if (cache.get("input1.txt") != strlen(input1_contents)) {
      cout << "FileSizeCache failed" << endl;
      return 1;
    }
    if (cache.get("input2.txt") != strlen(input2_contents)) {
      cout << "FileSizeCache failed 2" << endl;
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
  ofstream input1;
  input1.open("input1.txt", ofstream::out | ofstream::binary);
  const char *input1_contents = "diskfile_test test3 input1.txt";
  input1 << input1_contents;
  input1.close();

  DiskFile diskfile(cout, cerr);
  if (diskfile.Create("input1.txt", strlen(input1_contents))) {
    cout << "Create succeeded when file already existed!" << endl;
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
    DiskFile diskfile(cout, cerr);
    if (!diskfile.Create("input1.txt", strlen(input1_contents))) {
      cout << "Create failed!" << endl;
      return 1;
    }

    if (!diskfile.Write(0, input1_contents, strlen(input1_contents), 2)) {
      cout << "Write failed 1" << endl;
      return 1;
    }

    diskfile.Close();
  }

  {
    DiskFile diskfile(cout, cerr);
    
    if (!diskfile.Open("input1.txt")) {
      cout << "Open failed" << endl;
      return 1;
    }

    const size_t buffer_len = strlen(input1_contents)+1;  // for end-of-string
    u8 *buffer = new u8[buffer_len];
    // put end-of-string in buffer.
    buffer[buffer_len-1] = '\0';

    if (!diskfile.Read(0, buffer, buffer_len - 1, 2)) {
      cout << "Read whole file returned false" << endl;
      return 1;
    }

    if (string(input1_contents) != (char *) buffer) {
      cout << "Read did not read contents correctly" << endl;
      cout << "read     \"" << buffer << "\"" << endl;
      cout << "expected \"" << input1_contents << "\"" << endl;
      return 1;
    }

    diskfile.Close();

    remove("input1.txt");
  }


  const char *input2_contents = "diskfile_test test6 input2.txt is longer";

  // try again, writing mid-file with different maxlength.
  {
    DiskFile diskfile(cout, cerr);
    if (!diskfile.Create("input2.txt", strlen(input2_contents))) {
      cout << "Create 2 failed." << endl;
      return 1;
    }

    size_t midpoint = strlen(input2_contents);
    if (!diskfile.Write(midpoint, input2_contents + midpoint, strlen(input2_contents) - midpoint, 3)) {
      cout << "Write failed 2" << endl;
      return 1;
    }
    if (!diskfile.Write(0, input2_contents, midpoint, 4)) {
      cout << "Write failed 3" << endl;
      return 1;
    }
      
    diskfile.Close();
  }


  {
    DiskFile diskfile(cout, cerr);
    
    if (!diskfile.Open("input2.txt")) {
      cout << "Open failed" << endl;
      return 1;
    }

    const size_t buffer_len = strlen(input2_contents)+1;  // for end-of-string
    u8 *buffer = new u8[buffer_len];
    // put end-of-string in buffer.
    buffer[buffer_len-1] = '\0';


    size_t midpoint = strlen(input2_contents) - 2;
    if (!diskfile.Read(midpoint, buffer + midpoint, strlen(input2_contents) - midpoint, 4)) {
      cout << "Read second half of file returned false" << endl;
      return 1;
    }
    if (!diskfile.Read(0, buffer, midpoint, 3)) {
      cout << "Read first half of file returned false" << endl;
      return 1;
    }

    if (string(input2_contents) != (char *) buffer) {
      cout << "Read did not read contents correctly" << endl;
      cout << "read     \"" << buffer << "\"" << endl;
      cout << "expected \"" << input2_contents << "\"" << endl;
      return 1;
    }

    diskfile.Close();

    remove("input2.txt");
  }

  return 0;
}


int main() {
  if (test1()) {
    cerr << "FAILED: test1" << endl;
    return 1;
  }
  if (test2()) {
    cerr << "FAILED: test2" << endl;
    return 1;
  }
  if (test3()) {
    cerr << "FAILED: test3" << endl;
    return 1;
  }
  if (test4()) {
    cerr << "FAILED: test4" << endl;
    return 1;
  }
  if (test5()) {
    cerr << "FAILED: test5" << endl;
    return 1;
  }
  if (test6()) {
    cerr << "FAILED: test6" << endl;
    return 1;
  }

  cout << "SUCCESS: diskfile_test complete." << endl;
  
  return 0;
}

