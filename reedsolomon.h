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

class ReedSolomon
{
public:
  ReedSolomon(void);
  ~ReedSolomon(void);

  // Set which input blocks are present or missing
  bool SetInput(const vector<bool> &present); // Some input blocks are present
  bool SetInput(u32 count);                   // All input blocks are present

  // Set which output block are available or need to be computed
  bool SetOutput(bool present, u16 exponent);
  bool SetOutput(bool present, u16 lowexponent, u16 highexponent);

  // Compute the RS Matrix
  bool Compute(void);

  // Process a block of data
  bool Process(size_t size,             // The size of the block of data
               u32 inputindex,          // The column in the RS matrix
               const void *inputbuffer, // Buffer containing input data
               u32 outputindex,         // The row in the RS matrix
               void *outputbuffer);     // Buffer containing output data

protected:
  // Perform Gaussian Elimination
  bool GaussElim(unsigned int rows, 
                 unsigned int leftcols, 
                 Galois *leftmatrix, 
                 Galois *rightmatrix, 
                 unsigned int datamissing);

protected:
  u32 inputcount;        // Total number of input blocks

  u32 datapresent;       // Number of input blocks that are present 
  u32 datamissing;       // Number of input blocks that are missing
  u32 *datapresentindex; // The index numbers of the data blocks that are present
  u32 *datamissingindex; // The index numbers of the data blocks that are missing

  u16 *database;         // The "base" value to use for each input block

  u32 outputcount;       // Total number of output blocks

  u32 parpresent;        // Number of output blocks that are present
  u32 parmissing;        // Number of output blocks that are missing
  u32 *parpresentindex;  // The index numbers of the output blocks that are present
  u32 *parmissingindex;  // The index numbers of the output blocks that are missing

  vector<RSOutputRow> outputrows; // Details of the output blocks

  Galois *leftmatrix;    // The main matrix

  // When the matrices are initialised: values of the form base ^ exponent are
  // stored (where the base values are obtained from database[] and the exponent
  // values are obtained from outputrows[]).

#ifdef LONGMULTIPLY
  GaloisLongMultiplyTable *glmt;  // A multiplication table used by Process()
#endif
};


#endif // __REEDSOLOMON_H__
