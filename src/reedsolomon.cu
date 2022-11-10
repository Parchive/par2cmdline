//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2022 Xiuyan Wu
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

#include <assert.h>

#include "libpar2internal.h"
#include "helper_cuda.cuh"

__global__ void ProcessKer( const int batchSz,                    // size in number of word of each batch
                            const void * __restrict__ inputData,  // size: chunkSz * inputCount
                            const void * __restrict__ bases,      // size: sizeof(G) * inputCount
                            const int inputCount,
                            const int outputCount,                // # of output blocks
                            void * __restrict__ outputBuf,        // size: chunkSz * outputCount
                            const void * __restrict__ exponents   // size: sizeof(G) * outputCount
                          );                   

__global__ void ReduceKer( const u32 * __restrict__ input,       // Countains results from ProcessKer
                           u32 * __restrict__ output,            // Output Location
                           const int outputCount,                // Number of output blocks
                           const int tileCount                   // Number of input tiles
                          );

void CUDART_CB WriteOutputCB( void *data );

typedef struct
{
    size_t batchSz;
    size_t wordPerChunk;
    size_t wordPerBatch;
    size_t batchIdx;
    size_t outCount;
    Galois16 *finalOutput;
    Galois16 *batchOutput;
} workload;

#define TBLOCK_SZ 512
#define MAX_THREAD 1024
#define TILE_WIDTH 16
#define SHARED_MEM_SZ 32768

// Calculate the contribution of words contained in inputBuf to specified output blocks.
template <>
bool ReedSolomon<Galois16>::ProcessCu( const size_t size,          // size of one chunk of data
                                const u32 inputIdxStart,
                                const u32 inputIdxEnd,
                                const void *inputBuf,       // inputCount * size input1-input2-...
                                const u32 outputIdxStart,
                                const u32 outputIdxEnd,
                                void *outputBuf )           // outputCount * size
{
  // BUG: Doesn't really respect memory limit for VRAM.
  // CUDA Device compatible Galois type.
  typedef GaloisCu<G::Bits, G::Generator, G::ValueType> Gd;
  if ( !Gd::uploadTable() ) return false;
  cudaFuncSetCacheConfig(ProcessKer, cudaFuncCachePreferL1);

  const u32 inCount = inputIdxEnd - inputIdxStart + 1;
  const u32 outCount = outputIdxEnd - outputIdxStart + 1;
  const u32 wordPerChunk = size / sizeof(Gd);
  // Batch need to be 4-byte aligned
  const u32 wordPerBatch = (TBLOCK_SZ * SHARED_MEM_SZ / ( MAX_THREAD * TILE_WIDTH * sizeof(Gd) ) - 1) & ~1;
  const u32 tileCount = inCount / TILE_WIDTH + ( inCount % TILE_WIDTH != 0 );
  const u32 batchCount = ceil( (float) wordPerChunk / wordPerBatch );

  /* 
  * size: chunk size
  * VRam footprint: (inCount + outCount) * size + (inputcount + outCount) * sizeof(g) + sizeof(g::GaloisTable)
  *                 <input buffer and output buffer>          <base and exponent>            <Galois log tables>
  * 
  * Assume the total VRam footprint can be fitted into device vram.
  * Align chunkSz to 4 bytes
  * 
  * 
  */

  // Allocate GPU memory buffers
  Gd *d_bases, *d_exponents;

  cudaErrchk( cudaMalloc( (void**) &d_bases, inCount * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_exponents, outCount * sizeof(Gd) ) );

  // Copy bases and exponents to GPU
  u16 *baseOffset = &database[inputIdxStart];
  u16 *exponents = new u16[outCount];
  for ( u32 i = outputIdxStart; i <= outputIdxEnd; ++i ) {
    exponents[i - outputIdxStart] = outputrows[i].exponent;
  }

  cudaErrchk( cudaMemcpyAsync( d_bases, baseOffset, inCount * sizeof(Gd), cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMemcpyAsync( d_exponents, exponents, outCount * sizeof(u16), cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaDeviceSynchronize() );
  delete [] exponents;

  // Set kernel launch parameters
  dim3 dimGrid( tileCount );
  dim3 dimBlock( TBLOCK_SZ );

  // Create stream
  cudaStream_t *stream = new cudaStream_t[batchCount];
  for ( u32 i = 0; i < batchCount; ++i ) {
    cudaErrchk( cudaStreamCreateWithFlags( &stream[i], cudaStreamNonBlocking ) );
  }

  G *batchInput, *batchOutput;
  cudaErrchk( cudaMallocHost( (void**) &batchInput, inCount * wordPerBatch * sizeof(G) ) );
  cudaErrchk( cudaMallocHost( (void**) &batchOutput, outCount * wordPerBatch * sizeof(G) ) );

  Gd **d_input = new Gd*[batchCount];
  Gd **d_intermediate = new Gd*[batchCount];
  Gd **d_output = new Gd*[batchCount];

  cudaEvent_t written, upload;
  cudaErrchk( cudaEventCreate( &written, cudaEventDisableTiming ) );
  cudaErrchk( cudaEventCreate( &upload, cudaEventBlockingSync ) );

  // Concurrent kernel invoking
  for ( u32 batchIdx = 0; batchIdx < batchCount; ++batchIdx ) {
    int batchSz = wordPerBatch;
    if ( batchIdx == batchCount - 1 ) {
      batchSz = wordPerChunk - batchIdx * wordPerBatch;
    }
    int batchSzAligned = batchSz + (batchSz & 1);

    // Allocate memory
    // Gd *d_input, *d_intermediate, *d_output;
    cudaErrchk( cudaMallocAsync( (void**) &d_input[batchIdx], inCount * batchSz * sizeof(Gd), stream[batchIdx] ) );
    cudaErrchk( cudaMallocAsync( (void**) &d_intermediate[batchIdx], tileCount * batchSzAligned * outCount * sizeof(Gd), stream[batchIdx] ) );
    cudaErrchk( cudaMallocAsync( (void**) &d_output[batchIdx], outCount * batchSzAligned * sizeof(Gd), stream[batchIdx] ) );

    // Wait until the last iteration has sent all input data to GPU.
    cudaErrchk( cudaEventSynchronize( upload ) );

    // Copy input data to GPU
    for ( u32 i = 0; i < inCount; ++i ) {
      void *inputBufOffset = (char*) inputBuf + i * size + batchIdx * wordPerBatch * sizeof(G);
      void *batchInputOffset = (char*) batchInput + i * batchSz * sizeof(G);
      memcpy( batchInputOffset, inputBufOffset, batchSz * sizeof(G) );
    }

    cudaErrchk( cudaMemcpyAsync( d_input[batchIdx], batchInput, inCount * batchSz * sizeof(G), cudaMemcpyHostToDevice, stream[batchIdx] ) );
    cudaErrchk( cudaEventRecord( upload, stream[batchIdx] ) );

    // Lauch Compute Kernel
    ProcessKer<<<dimGrid, dimBlock, (batchSzAligned + 1) * TILE_WIDTH * sizeof(G), stream[batchIdx]>>> 
    ( batchSz,
      d_input[batchIdx],
      d_bases,
      inCount,
      outCount,
      d_intermediate[batchIdx],
      d_exponents
    );

    // Lauch Reduce Kernel
    dim3 dimBlockReduce( 32 );
    dim3 dimGridReduce( ceil( outCount / (float) dimBlockReduce.x ), batchSzAligned / 2 );
    ReduceKer<<<dimGridReduce, dimBlockReduce, 0, stream[batchIdx]>>>
    ( (u32*) d_intermediate[batchIdx],
      (u32*) d_output[batchIdx],
      outCount,
      tileCount
    );

    // Wait until output from the last iteration has already
    // been written to actual output buffer.
    cudaErrchk( cudaStreamWaitEvent( stream[batchIdx], written) );

    // Copy Result to batch output buffer
    cudaErrchk( cudaMemcpyAsync( batchOutput,
                                 d_output[batchIdx],
                                 batchSzAligned * outCount * sizeof(Gd),
                                 cudaMemcpyDeviceToHost,
                                 stream[batchIdx]
                                ) );
    
    // Copy result in batch output buffer to actual output buffer
    workload *work = new workload;
    work->batchSz = batchSz;
    work->wordPerChunk = wordPerChunk;
    work->wordPerBatch = wordPerBatch;
    work->batchIdx = batchIdx;
    work->outCount = outCount;
    work->finalOutput = (G*) outputBuf;
    work->batchOutput = batchOutput;
    cudaErrchk( cudaLaunchHostFunc( stream[batchIdx], WriteOutputCB, work ) );
    cudaErrchk( cudaEventRecord( written, stream[batchIdx] ) );

    cudaErrchk( cudaFreeAsync( d_input[batchIdx], stream[batchIdx] ) );
    cudaErrchk( cudaFreeAsync( d_intermediate[batchIdx], stream[batchIdx] ) );
    cudaErrchk( cudaFreeAsync( d_output[batchIdx], stream[batchIdx] ) );

  }
  cudaErrchk( cudaDeviceSynchronize() );

  // Destroy stream
  for ( u32 i = 0; i < batchCount; ++i ) {
    cudaErrchk( cudaStreamDestroy( stream[i] ) );
  }

  cudaFree( d_bases );
  cudaFree( d_exponents );
  cudaFreeHost( batchInput );
  cudaFreeHost( batchOutput );
  delete[] stream;
  delete[] d_input;
  delete[] d_intermediate;
  delete[] d_output;
  
  return true;
}

__global__ void ProcessKer( const int batchSz,
                            const void * __restrict__ inputData,
                            const void * __restrict__ bases,
                            const int inputCount,
                            const int outputCount,
                            void * __restrict__ outputBuf,
                            const void * __restrict__ exponents
                            )
{
  /*
  inputData: I_1,1  I_1,2  ...  ...  I_1,batchSz
             I_2,1  ...    ...  ...  I_2,batchSz
             .
             .
             .
             I_TILE_WIDTH*gridDim.x,1 ... ...  I_TILE_WIDTH*gridDim.x,batchSz

  outputBuf **Transposed**: 
             <<Tile 0>>
             O_1,1  O_1,2  O_1,3  ...  O_1,batchSz
             O_2,1  ...    ...    ...  O_2,batchSz
             .
             .
             .
             O_outputCount,1 ...  ...  O_outputCount,batchSz
             <<Tile 1>>
             O_1,1  O_1,2  O_1,3  ...  O_1,batchSz
             .
             .
             .
             .
             O_outputCount,1 ...  ...  O_outputCount,batchSz
             <<Tile 2>>
             .
             .
             <<Tile gridDim.x>>
  */

  // Need (batchSz * TILE_WIDTH + TILE_WIDTH) * sizeof(G) bytes
  typedef GaloisCu16 G;
  extern __shared__ char sharedMem[];

  const int batchSzAligned = batchSz + (batchSz & 1);
  G *smInput = (G *) sharedMem;  // Shared memory input buffer
  G *smBases = (G *) ( sharedMem + batchSzAligned * TILE_WIDTH * sizeof(G) );
  
  const int wordPerInt = sizeof(u32) / sizeof(G);
  const int intPerBatch = batchSzAligned / wordPerInt;
  const int outBufWidth = outputCount * gridDim.x;
  const int outBufRowPos = outputCount * blockIdx.x;

  // Load input data and bases into shared mem
  for ( int i = 0; i < TILE_WIDTH; ++i )
  {
    int inputIdx = blockIdx.x * TILE_WIDTH + i;
    for ( int j = threadIdx.x; j < batchSzAligned; j += blockDim.x )
    {
      if ( inputIdx < inputCount && j < batchSz ) {
        ((G *) smInput)[i * batchSzAligned + j] = ((G *) inputData)[inputIdx * batchSz + j];
      } else {
        ((G *) smInput)[i * batchSzAligned + j] = 0;
      }
    }
  }

  for ( int i = threadIdx.x; i < TILE_WIDTH; i += blockDim.x )
  {
    int inputIdx = blockIdx.x * TILE_WIDTH + i;
    if ( inputIdx < inputCount ) {
      ((G *) smBases)[i] = ((G *) bases)[inputIdx];
    } else {
      ((G *) smBases)[i] = 0;
    }
  }

  __syncthreads();

  G factors[TILE_WIDTH];
  u16 exponent;
  u32 acc = 0;
  u32 words, res;
  // Each thread compute one output block.
  for ( int i = threadIdx.x; i < outputCount; i += blockDim.x )
  {
    // Calculate factors for this tile for this output block
    exponent = ((G *) exponents)[i].Value();
    // #pragma unroll
    for ( int ii = 0; ii < TILE_WIDTH; ++ii ) {
      factors[ii] = smBases[ii].pow(exponent);
    }

    // For each int in the batch
    for ( int j = 0; j < intPerBatch; ++j )
    {
      acc = 0;
      // For each inputblock in the tile, calculate contribution of corresponding 2 words (a int).
      for ( int k = 0; k < TILE_WIDTH; ++k )
      {
        // factor = smBases[k].pow( exponent );
        words = ((u32 *) smInput)[k * intPerBatch + j];
        res = 0;
        for ( int w = 0; w < wordPerInt; ++w )
        {
          ((G *) &res)[w] = factors[k] * ((G *) &words)[w];
        }
        acc ^= res;
      }

      // Write result to outputBuf
      ((u32 *)outputBuf)[j * outBufWidth + outBufRowPos + i] = acc;
    }
  }
}

__global__ void ReduceKer( const u32 * __restrict__ input,       // Countains results from ProcessKer
                           u32 * __restrict__ output,            // Output Location
                           const int outputCount,                 // Number of output blocks
                           const int tileCount                    // Number of input tiles
                          ) 
{
  /*
  input **Transposed**: 
             <<Tile 0>>
             O_1,1  O_1,2  O_1,3  ...  O_1,batchSz
             O_2,1  ...    ...    ...  O_2,batchSz
             .
             .
             .
             O_outputCount,1 ...  ...  O_outputCount,batchSz
             <<Tile 1>>
             O_1,1  O_1,2  O_1,3  ...  O_1,batchSz
             .
             .
             .
             .
             O_outputCount,1 ...  ...  O_outputCount,batchSz
             <<Tile 2>>
             .
             .
             <<Tile gridDim.x>>
  */
  // 2D Grid: blocks at position x, y process
  // the y^th (two) word of output block x*blockDim to (x+1)*blockDim.
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * outputCount * tileCount;
  
  if (ox >= outputCount)  return;

  u32 acc = 0;
  int inputIdx = ox;
  for ( int i = 0; i < tileCount; ++i ) {
    acc ^= input[row + inputIdx];
    inputIdx += outputCount;
  }

  output[gridDim.y * ox + blockIdx.y] = acc;
}

void CUDART_CB WriteOutputCB( void *data ) {
  // Write output from batch output buffer into actual output buffer.
  workload *work = (workload *) data;
  size_t batchSzAligned = work->batchSz + (work->batchSz & 1);
  for ( u32 i = 0; i < work->outCount; ++i ){
    memcpy( &work->finalOutput[work->wordPerChunk * i + work->wordPerBatch * work->batchIdx],
            &work->batchOutput[batchSzAligned * i],
            work->batchSz * sizeof(Galois16)
          );
  }
  delete work;
}

