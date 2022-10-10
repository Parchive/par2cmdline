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

#ifndef __REEDSOLOMON_H__
#define __REEDSOLOMON_H__

// The ReedSolomon object is used to calculate and store the matrix
// used during recovery block creation or data block reconstruction.
//
// During initialisation, one RSOutputRow object is created for each
// recovery block that either needs to be created or is available for
// use.

class RSOutputRow
{
public:
  RSOutputRow(void) {};
  RSOutputRow(bool _present, u16 _exponent) : present(_present), exponent(_exponent) {}

public:
  bool present;
  u16 exponent;
};

template<class g>
class ReedSolomon
{
public:
  typedef g G;

  ReedSolomon(void);
  ~ReedSolomon(void);

  // Set which input blocks are present or missing
  // Some input blocks are present
  bool SetInput(const vector<bool> &present, std::ostream &sout, std::ostream &serr);
  // All input blocks are present
  bool SetInput(u32 count, std::ostream &sout, std::ostream &serr);

  // Set which output block are available or need to be computed
  bool SetOutput(bool present, u16 exponent);
  bool SetOutput(bool present, u16 lowexponent, u16 highexponent);

  // Compute the RS Matrix
  bool Compute(NoiseLevel noiselevel, std::ostream &sout, std::ostream &serr);

  // Process a block of data
  bool Process(size_t size,             // The size of the block of data
               u32 inputindex,          // The column in the RS matrix
               const void *inputbuffer, // Buffer containing input data
               u32 outputindex,         // The row in the RS matrix
               void *outputbuffer);     // Buffer containing output data

#ifdef ENABLE_CUDA
  // Process a block of data for output blocks [outputIdxStart, outputIdxEnd] on a CUDA device.
  bool ProcessCu( const size_t size,           // The size of the block of data
                  const u32 inputIdxStart,     // The index of the first input block to be processed
                  const u32 inputIdxEnd,       // The index of the last input block to be processed
                  const void *inputBuf,        // Buffer containing input data
                  const u32 outputIdxStart,    // The row for the first output block to process in RS matrix
                  const u32 outputIdxEnd,      // The row for the last output block to process in RS matrix
                  void *outputBuf );           // Buffer containing output data
#endif


private:
		bool InternalProcess(const g &factor, size_t size, const void *inputbuffer, void *outputbuffer);	// Optimization

protected:
  // Perform Gaussian Elimination
  bool GaussElim(NoiseLevel noiselevel,
		 std::ostream &sout,
		 std::ostream &serr,
		 unsigned int rows,
                 unsigned int leftcols,
                 G *leftmatrix,
                 G *rightmatrix,
                 unsigned int datamissing);

protected:
  u32 inputcount;        // Total number of input blocks

  u32 datapresent;       // Number of input blocks that are present
  u32 datamissing;       // Number of input blocks that are missing
  u32 *datapresentindex; // The index numbers of the data blocks that are present
  u32 *datamissingindex; // The index numbers of the data blocks that are missing

  typename G::ValueType *database;// The "base" value to use for each input block

  u32 outputcount;       // Total number of output blocks

  u32 parpresent;        // Number of output blocks that are present
  u32 parmissing;        // Number of output blocks that are missing
  u32 *parpresentindex;  // The index numbers of the output blocks that are present
  u32 *parmissingindex;  // The index numbers of the output blocks that are missing

  vector<RSOutputRow> outputrows; // Details of the output blocks

  G *leftmatrix;    // The main matrix

  // When the matrices are initialised: values of the form base ^ exponent are
  // stored (where the base values are obtained from database[] and the exponent
  // values are obtained from outputrows[]).

#ifdef LONGMULTIPLY
  GaloisLongMultiplyTable<g> *glmt;  // A multiplication table used by Process()
#endif

private:
  // private copy constructor to prevent any misuse.
  ReedSolomon(const ReedSolomon &);
  ReedSolomon& operator=(const ReedSolomon &);
};

template<class g>
inline ReedSolomon<g>::ReedSolomon(void)
: outputrows()
{
  inputcount = 0;

  datapresent = 0;
  datamissing = 0;
  datapresentindex = 0;
  datamissingindex = 0;
  database = 0;

  outputcount = 0;

  parpresent = 0;
  parmissing = 0;
  parpresentindex = 0;
  parmissingindex = 0;

  leftmatrix = 0;

#ifdef LONGMULTIPLY
  glmt = new GaloisLongMultiplyTable<g>;
#endif
}

template<class g>
inline ReedSolomon<g>::~ReedSolomon(void)
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

template<class g>
inline bool ReedSolomon<g>::Process(size_t size, u32 inputindex, const void *inputbuffer, u32 outputindex, void *outputbuffer)
{
	// Optimization: it occurs frequently the function exits early on, so inline the start.
	// This resulted in a speed gain of approx. 8% in repairing.

	// Look up the appropriate element in the RS matrix
	g factor = leftmatrix[outputindex * (datapresent + datamissing) + inputindex];
	// Do nothing if the factor happens to be 0
	if (factor == 0)
		return eSuccess;
	return this->InternalProcess (factor, size, inputbuffer, outputbuffer);
}

u32 gcd(u32 a, u32 b);

// Record whether the recovery block with the specified
// exponent values is present or missing.
template<class g>
inline bool ReedSolomon<g>::SetOutput(bool present, u16 exponent)
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
template<class g>
inline bool ReedSolomon<g>::SetOutput(bool present, u16 lowexponent, u16 highexponent)
{
  for (unsigned int exponent=lowexponent; exponent<=highexponent; exponent++)
  {
    if (!SetOutput(present, exponent))
      return false;
  }

  return true;
}

// Construct the Vandermonde matrix and solve it if necessary
template<class g>
inline bool ReedSolomon<g>::Compute(NoiseLevel noiselevel, std::ostream &sout, std::ostream &serr)
{
  u32 outcount = datamissing + parmissing;
  u32 incount = datapresent + datamissing;

  if (datamissing > parpresent)
  {
    serr << "Not enough recovery blocks." << endl;
    return false;
  }
  else if (outcount == 0)
  {
    serr << "No output blocks." << endl;
    return false;
  }

  if (noiselevel > nlQuiet)
    sout << "Computing Reed Solomon matrix." << endl;

  /*  Layout of RS Matrix:
      NOTE: The second set of columns represents the parity vectors present,
      but this only uses datamissing of them.  Otherwise, it is over constrained.

                                       datamissing
                     datapresent      <=parpresent        datamissing       parmissing
               /                     |             \ /                     |           \
               |           (ppi[row])|             | |           (ppi[row])|           |
   datamissing |          ^          |      I      | |          ^          |     0     |
               |(dpi[col])           |             | |(dmi[col])           |           |
               +---------------------+-------------+ +---------------------+-----------+
               |           (pmi[row])|             | |           (pmi[row])|           |
   parmissing  |          ^          |      0      | |          ^          |     I     |
               |(dpi[col])           |             | |(dmi[col])           |           |
               \                     |             / \                     |           /
  */

  // Allocate the left hand matrix

  leftmatrix = new G[outcount * incount];
  memset(leftmatrix, 0, outcount * incount * sizeof(G));

  // Allocate the right hand matrix only if we are recovering

  G *rightmatrix = 0;
  if (datamissing > 0)
  {
    rightmatrix = new G[outcount * outcount];
    memset(rightmatrix, 0, outcount *outcount * sizeof(G));
  }

  // Fill in the two matrices:

  vector<RSOutputRow>::const_iterator outputrow = outputrows.begin();

  // One row for each present recovery block that will be used for a missing data block
  for (unsigned int row=0; row<datamissing; row++)
  {
    // Define MPDL to skip reporting and speed things up
#ifndef MPDL
    if (noiselevel > nlQuiet)
    {
      int progress = row * 1000 / (datamissing+parmissing);
      sout << "Constructing: " << progress/10 << '.' << progress%10 << "%\r" << flush;
    }
#endif

    // Get the exponent of the next present recovery block
    while (!outputrow->present)
    {
      outputrow++;
    }
    u16 exponent = outputrow->exponent;

    // One column for each present data block
    for (unsigned int col=0; col<datapresent; col++)
    {
      leftmatrix[row * incount + col] = G(database[datapresentindex[col]]).pow(exponent);
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
        rightmatrix[row * outcount + col] = G(database[datamissingindex[col]]).pow(exponent);
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
    // Define MPDL to skip reporting and speed things up
#ifndef MPDL
    if (noiselevel > nlQuiet)
    {
      int progress = (row+datamissing) * 1000 / (datamissing+parmissing);
      sout << "Constructing: " << progress/10 << '.' << progress%10 << "%\r" << flush;
    }
#endif

    // Get the exponent of the next missing recovery block
    while (outputrow->present)
    {
      outputrow++;
    }
    u16 exponent = outputrow->exponent;

    // One column for each present data block
    for (unsigned int col=0; col<datapresent; col++)
    {
      leftmatrix[(row+datamissing) * incount + col] = G(database[datapresentindex[col]]).pow(exponent);
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
        rightmatrix[(row+datamissing) * outcount + col] = G(database[datamissingindex[col]]).pow(exponent);
      }
      // One column for each missing recovery block
      for (unsigned int col=0; col<parmissing; col++)
      {
        rightmatrix[(row+datamissing) * outcount + col + datamissing] = (row == col) ? 1 : 0;
      }
    }

    outputrow++;
  }
  if (noiselevel > nlQuiet)
    sout << "Constructing: done." << endl;

  // Solve the matrices only if recovering data
  if (datamissing > 0)
  {
    // Perform Gaussian Elimination and then delete the right matrix (which
    // will no longer be required).
    bool success = GaussElim(noiselevel, sout, serr, outcount, incount, leftmatrix, rightmatrix, datamissing);
    delete [] rightmatrix;
    return success;
  }

  return true;
}

// Use Gaussian Elimination to solve the matrices
template<class g>
inline bool ReedSolomon<g>::GaussElim(NoiseLevel noiselevel, std::ostream &sout, std::ostream &serr, unsigned int rows, unsigned int leftcols, G *leftmatrix, G *rightmatrix, unsigned int datamissing)
{
  if (noiselevel >= nlDebug)
  {
    for (unsigned int row=0; row<rows; row++)
    {
      sout << ((row==0) ? "/"    : (row==rows-1) ? "\\"    : "|");
      for (unsigned int col=0; col<leftcols; col++)
      {
        sout << " "
             << hex << setw(G::Bits>8?4:2) << setfill('0')
             << (unsigned int)leftmatrix[row*leftcols+col];
      }
      sout << ((row==0) ? " \\ /" : (row==rows-1) ? " / \\" : " | |");
      for (unsigned int col=0; col<rows; col++)
      {
        sout << " "
             << hex << setw(G::Bits>8?4:2) << setfill('0')
             << (unsigned int)rightmatrix[row*rows+col];
      }
      sout << ((row==0) ? " \\"   : (row==rows-1) ? " /"    : " | |");
      sout << endl;

      sout << dec << setw(0) << setfill(' ');
    }
  }

  // Because the matrices being operated on are Vandermonde matrices
  // they are guaranteed not to be singular.

  // Additionally, because Galois arithmetic is being used, all calculations
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
    G pivotvalue = rightmatrix[row * rows + row];
    assert(pivotvalue != 0);
    if (pivotvalue == 0)
    {
      serr << "RS computation error." << endl;
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
      // Define MPDL to skip reporting and speed things up
#ifndef MPDL
      if (noiselevel > nlQuiet)
      {
        int newprogress = (row*rows+row2) * 1000 / (datamissing*rows);
        if (progress != newprogress)
        {
          progress = newprogress;
          sout << "Solving: " << progress/10 << '.' << progress%10 << "%\r" << flush;
        }
      }
#endif

      if (row != row2)
      {
        // Get the scaling factor for this row.
        G scalevalue = rightmatrix[row2 * rows + row];

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
  if (noiselevel > nlQuiet)
    sout << "Solving: done." << endl;
  if (noiselevel >= nlDebug)
  {
    for (unsigned int row=0; row<rows; row++)
    {
      sout << ((row==0) ? "/"    : (row==rows-1) ? "\\"    : "|");
      for (unsigned int col=0; col<leftcols; col++)
      {
        sout << " "
             << hex << setw(G::Bits>8?4:2) << setfill('0')
             << (unsigned int)leftmatrix[row*leftcols+col];
      }
      sout << ((row==0) ? " \\ /" : (row==rows-1) ? " / \\" : " | |");
      for (unsigned int col=0; col<rows; col++)
      {
        sout << " "
             << hex << setw(G::Bits>8?4:2) << setfill('0')
             << (unsigned int)rightmatrix[row*rows+col];
      }
      sout << ((row==0) ? " \\"   : (row==rows-1) ? " /"    : " | |");
      sout << endl;

      sout << dec << setw(0) << setfill(' ');
    }
  }

  return true;
}


#endif // __REEDSOLOMON_H__
