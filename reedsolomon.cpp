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

#include "par2cmdline.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

ReedSolomon::ReedSolomon(void)
{
  inputcount = 0;

  datapresent = 0;
  datamissing = 0;
  datapresentindex = 0;
  datamissingindex = 0;
  database = 0;

  outputrows.empty();

  outputcount = 0;

  parpresent = 0;
  parmissing = 0;
  parpresentindex = 0;
  parmissingindex = 0;

  leftmatrix = 0;

#ifdef LONGMULTIPLY
  glmt = new GaloisLongMultiplyTable;
#endif
}

ReedSolomon::~ReedSolomon(void)
{
  delete [] datapresentindex;
  delete [] datamissingindex;
  delete [] database;
  delete [] parpresentindex;
  delete [] parmissingindex;
  delete [] leftmatrix;

#ifdef LONGMULTIPLY
  delete glmt;
#endif
}

u32 gcd(u32 a, u32 b)
{
  if (a && b)
  {
    while (a && b)
    {
      if (a>b)
      {
        a = a%b;
      }
      else
      {
        b = b%a;
      }
    }

    return a+b;
  }
  else
  {
    return 0;
  }
}

// Set which of the source files are present and which are missing
// and compute the base values to use for the vandermonde matrix.
bool ReedSolomon::SetInput(const vector<bool> &present)
{
  inputcount = (u32)present.size();

  datapresentindex = new u32[inputcount];
  datamissingindex = new u32[inputcount];
  database         = new u16[inputcount];

  unsigned int logbase = 0;

  for (unsigned int index=0; index<inputcount; index++)
  {
    // Record the index of the file in the datapresentindex array 
    // or the datamissingindex array
    if (present[index])
    {
      datapresentindex[datapresent++] = index;
    }
    else
    {
      datamissingindex[datamissing++] = index;
    }

    // Determine the next useable base value.
    // Its log must must be relatively prime to 65535
    while (gcd(Galois::limit, logbase) != 1)
    {
      logbase++;
    }
    if (logbase >= Galois::limit)
    {
      cerr << "Too many input blocks for Reed Solomon matrix." << endl;
      return false;
    }
    unsigned int base = Galois(logbase++).ALog();

    database[index] = base;
  }

  return true;
}

// Record that the specified number of source files are all present
// and compute the base values to use for the vandermonde matrix.
bool ReedSolomon::SetInput(u32 count)
{
  inputcount = count;

  datapresentindex = new u32[inputcount];
  datamissingindex = new u32[inputcount];
  database         = new u16[inputcount];

  unsigned int logbase = 0;

  for (unsigned int index=0; index<count; index++)
  {
    // Record that the file is present
    datapresentindex[datapresent++] = index;

    // Determine the next useable base value.
    // Its log must must be relatively prime to 65535
    while (gcd(Galois::limit, logbase) != 1)
    {
      logbase++;
    }
    if (logbase >= Galois::limit)
    {
      cerr << "Too many input blocks for Reed Solomon matrix." << endl;
      return false;
    }
    unsigned int base = Galois(logbase++).ALog();

    database[index] = base;
  }

  return true;
}

// Record whether the recovery block with the specified
// exponent values is present or missing.
bool ReedSolomon::SetOutput(bool present, u16 exponent)
{
  // Store the exponent and whether or not the recovery block is present or missing
  outputrows.push_back(RSOutputRow(present, exponent));

  outputcount++;

  // Update the counts.
  if (present)
  {
    parpresent++;
  }
  else
  {
    parmissing++;
  }

  return true;
}

// Record whether the recovery blocks with the specified
// range of exponent values are present or missing.
bool ReedSolomon::SetOutput(bool present, u16 lowexponent, u16 highexponent)
{
  for (unsigned int exponent=lowexponent; exponent<=highexponent; exponent++)
  {
    if (!SetOutput(present, exponent))
      return false;
  }

  return true;
}

// Construct the Vandermonde matrix and solve it if necessary
bool ReedSolomon::Compute(void)
{
  u32 outcount = datamissing + parmissing;
  u32 incount = datapresent + datamissing;

  if (datamissing > parpresent)
  {
    cerr << "Not enough recovery blocks." << endl;
    return false;
  }
  else if (outcount == 0)
  {
    cerr << "No output blocks." << endl;
    return false;
  }

  cout << "Computing Reed Solomon matrix." << endl;

  //  Layout of RS Matrix:

  //                                     parpresent
  //                   datapresent       datamissing         datamissing       parmissing
  //             /                     |             \ /                     |           \
  // parpresent  |           (ppi[row])|             | |           (ppi[row])|           |
  // datamissing |          ^          |      I      | |          ^          |     0     |
  //             |(dpi[col])           |             | |(dmi[col])           |           |
  //             +---------------------+-------------+ +---------------------+-----------+
  //             |           (pmi[row])|             | |           (pmi[row])|           |
  // parmissing  |          ^          |      0      | |          ^          |     I     |
  //             |(dpi[col])           |             | |(dmi[col])           |           |
  //             \                     |             / \                     |           /


  // Allocate the left hand matrix

  leftmatrix = new Galois[outcount * incount];
  memset(leftmatrix, 0, outcount * incount * sizeof(Galois));

  // Allocate the right hand matrix only if we are recovering

  Galois *rightmatrix = 0;
  if (datamissing > 0)
  {
    rightmatrix = new Galois[outcount * outcount];
    memset(rightmatrix, 0, outcount *outcount * sizeof(Galois));
  }

  // Fill in the two matrices:

  vector<RSOutputRow>::const_iterator outputrow = outputrows.begin();

  // One row for each present recovery block that will be used for a missing data block
  for (unsigned int row=0; row<datamissing; row++)
  {
    int progress = row * 1000 / (datamissing+parmissing);
    cout << "Constructing: " << progress/10 << '.' << progress%10 << "%\r" << flush;

    // Get the exponent of the next present recovery block
    while (!outputrow->present)
    {
      outputrow++;
    }
    u16 exponent = outputrow->exponent;

    // One column for each present data block
    for (unsigned int col=0; col<datapresent; col++)
    {
      leftmatrix[row * incount + col] = Galois(database[datapresentindex[col]]).pow(exponent);
    }
    // One column for each each present recovery block that will be used for a missing data block
    for (unsigned int col=0; col<datamissing; col++)
    {
      leftmatrix[row * incount + col + datapresent] = (row == col) ? 1 : 0;
    }

    if (datamissing > 0)
    {
      // One column for each missing data block
      for (unsigned int col=0; col<datamissing; col++)
      {
        rightmatrix[row * outcount + col] = Galois(database[datamissingindex[col]]).pow(exponent);
      }
      // One column for each missing recovery block
      for (unsigned int col=0; col<parmissing; col++)
      {
        rightmatrix[row * outcount + col + datamissing] = 0;
      }
    }

    outputrow++;
  }
  // One row for each recovery block being computed
  outputrow = outputrows.begin();
  for (unsigned int row=0; row<parmissing; row++)
  {
    int progress = (row+datamissing) * 1000 / (datamissing+parmissing);
    cout << "Constructing: " << progress/10 << '.' << progress%10 << "%\r" << flush;

    // Get the exponent of the next missing recovery block
    while (outputrow->present)
    {
      outputrow++;
    }
    u16 exponent = outputrow->exponent;

    // One column for each present data block
    for (unsigned int col=0; col<datapresent; col++)
    {
      leftmatrix[(row+datamissing) * incount + col] = Galois(database[datapresentindex[col]]).pow(exponent);
    }
    // One column for each each present recovery block that will be used for a missing data block
    for (unsigned int col=0; col<datamissing; col++)
    {
      leftmatrix[(row+datamissing) * incount + col + datapresent] = 0;
    }

    if (datamissing > 0)
    {
      // One column for each missing data block
      for (unsigned int col=0; col<datamissing; col++)
      {
        rightmatrix[(row+datamissing) * outcount + col] = Galois(database[datamissingindex[col]]).pow(exponent);
      }
      // One column for each missing recovery block
      for (unsigned int col=0; col<parmissing; col++)
      {
        rightmatrix[(row+datamissing) * outcount + col + datamissing] = (row == col) ? 1 : 0;
      }
    }

    outputrow++;
  }
  cout << "Constructing: done." << endl;

  // Solve the matrices only if recovering data
  if (datamissing > 0)
  {
    // Perform Gaussian Elimination and then delete the right matrix (which 
    // will no longer be required).
    bool success = GaussElim(outcount, incount, leftmatrix, rightmatrix, datamissing);
    delete [] rightmatrix;
    return success;
  }

  return true;
}

// Use Gaussian Elimination to solve the matrices
bool ReedSolomon::GaussElim(unsigned int rows, unsigned int leftcols, Galois *leftmatrix, Galois *rightmatrix, unsigned int datamissing)
{
  // Because the matrices being operated on are Vandermonde matrices
  // they are guaranteed not to be singular.

  // Additionally, because Galois arithmetic is being used, all calulations
  // involve exact values with no loss of precision. It is therefore
  // not necessary to carry out any row or column swapping.

  // Solve one row at a time

  int progress = 0;

  // For each row in the matrix
  for (unsigned int row=0; row<datamissing; row++)
  {
    // NB Row and column swapping to find a non zero pivot value or to find the largest value
    // is not necessary due to the nature of the arithmetic and construction of the RS matrix.

    // Get the pivot value.
    Galois pivotvalue = rightmatrix[row * rows + row];
    assert(pivotvalue != 0);
    if (pivotvalue == 0)
    {
      cerr << "RS computation error." << endl;
      return false;
    }

    // If the pivot value is not 1, then the whole row has to be scaled
    if (pivotvalue != 1)
    {
      for (unsigned int col=0; col<leftcols; col++)
      {
        if (leftmatrix[row * leftcols + col] != 0)
        {
          leftmatrix[row * leftcols + col] /= pivotvalue;
        }
      }
      rightmatrix[row * rows + row] = 1;
      for (unsigned int col=row+1; col<rows; col++)
      {
        if (rightmatrix[row * rows + col] != 0)
        {
          rightmatrix[row * rows + col] /= pivotvalue;
        }
      }
    }

    // For every other row in the matrix
    for (unsigned int row2=0; row2<rows; row2++)
    {
      int newprogress = (row*rows+row2) * 1000 / (datamissing*rows);
      if (progress != newprogress)
      {
        progress = newprogress;
        cout << "Solving: " << progress/10 << '.' << progress%10 << "%\r" << flush;
      }

      if (row != row2)
      {
        // Get the scaling factor for this row.
        Galois scalevalue = rightmatrix[row2 * rows + row];

        if (scalevalue == 1)
        {
          // If the scaling factor happens to be 1, just subtract rows
          for (unsigned int col=0; col<leftcols; col++)
          {
            if (leftmatrix[row * leftcols + col] != 0)
            {
              leftmatrix[row2 * leftcols + col] -= leftmatrix[row * leftcols + col];
            }
          }

          for (unsigned int col=row; col<rows; col++)
          {
            if (rightmatrix[row * rows + col] != 0)
            {
              rightmatrix[row2 * rows + col] -= rightmatrix[row * rows + col];
            }
          }
        }
        else if (scalevalue != 0)
        {
          // If the scaling factor is not 0, then compute accordingly.
          for (unsigned int col=0; col<leftcols; col++)
          {
            if (leftmatrix[row * leftcols + col] != 0)
            {
              leftmatrix[row2 * leftcols + col] -= leftmatrix[row * leftcols + col] * scalevalue;
            }
          }

          for (unsigned int col=row; col<rows; col++)
          {
            if (rightmatrix[row * rows + col] != 0)
            {
              rightmatrix[row2 * rows + col] -= rightmatrix[row * rows + col] * scalevalue;
            }
          }
        }
      }
    }
  }
  cout << "Solving: done." << endl;

  return true;
}

bool ReedSolomon::Process(size_t size, u32 inputindex, const void *inputbuffer, u32 outputindex, void *outputbuffer)
{
  // Look up the appropriate element in the RS matrix

  Galois factor = leftmatrix[outputindex * (datapresent + datamissing) + inputindex];

  // Do nothing if the factor happens to be 0
  if (factor == 0)
    return eSuccess;

#ifdef LONGMULTIPLY
  // The 8-bit long multiplication tables
  Galois *table = glmt->tables;

  // Split the factor into Low and High bytes
  unsigned int fl = (factor >> 0) & 0xff;
  unsigned int fh = (factor >> 8) & 0xff;

  // Get the four separate multiplication tables
  Galois *LL = &table[(0*256 + fl) * 256 + 0]; // factor.low  * source.low
  Galois *LH = &table[(1*256 + fl) * 256 + 0]; // factor.low  * source.high
  Galois *HL = &table[(1*256 + 0) * 256 + fh]; // factor.high * source.low
  Galois *HH = &table[(2*256 + fh) * 256 + 0]; // factor.high * source.high

  // Combine the four multiplication tables into two
  unsigned int L[256];
  unsigned int H[256];

  unsigned int *pL = &L[0];
  unsigned int *pH = &H[0];

  for (unsigned int i=0; i<256; i++)
  {
    *pL = *LL + *HL;

    pL++;
    LL++;
    HL+=256;

    *pH = *LH + *HH;

    pH++;
    LH++;
    HH++;
  }

  // Treat the buffers as arrays of 32-bit unsigned ints.
  u32 *src = (u32 *)inputbuffer;
  u32 *end = (u32 *)&((u8*)inputbuffer)[size];
  u32 *dst = (u32 *)outputbuffer;

  // Process the data
  while (src < end)
  {
    u32 s = *src++;

    // Use the two lookup tables computed earlier
    *dst++ ^= (L[(s >> 0) & 0xff]      )
           ^  (H[(s >> 8) & 0xff]      )
           ^  (L[(s >> 16)& 0xff] << 16)
           ^  (H[(s >> 24)& 0xff] << 16);
  }
#else
  // Treat the buffers as arrays of 16-bit Galois values.

  Galois *src = (Galois *)inputbuffer;
  Galois *end = (Galois *)&((u8*)inputbuffer)[size];
  Galois *dst = (Galois *)outputbuffer;

  // Process the data
  while (src < end)
  {
    *dst++ += *src++ * factor;
  }
#endif

  return eSuccess;
}
