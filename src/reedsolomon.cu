#include <assert.h>

#include "libpar2internal.h"
#include "helper_cuda.cuh"

__global__ void ProcessKer( const u32 chunkSz,                    // size in byte of input chunks
                            const void * __restrict__ inputData,  // size: chunkSz * inputCount
                            const void * __restrict__ bases,      // size: sizeof(G) * inputCount
                            const int outputCount,                // # of output blocks
                            void * __restrict__ outputBuf,        // size: chunkSz * outputCount
                            const void * __restrict__ exponents,  // size: sizeof(G) * outputCount
                            const int startIdxInt,
                            const int batchSz                     // # of words each tblock processes
                            );                   


#define TBLOCK_SZ 256
#define MAX_THREAD 1024
#define TILE_WIDTH 32
#define SHARED_MEM_SZ 65536

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
  // CUDA Device compatible Galois type.
  typedef GaloisCu<G::Bits, G::Generator, G::ValueType> Gd;

  const u32 inCount = inputIdxEnd - inputIdxStart + 1;
  const u32 outCount = outputIdxEnd - outputIdxStart + 1;
  // Batch need to be 4-byte aligned
  const u32 wordPerBatch = (TBLOCK_SZ * SHARED_MEM_SZ / ( MAX_THREAD * TILE_WIDTH * sizeof(Gd) ) - 1) & ~1;
  const u32 tileCount = inCount / TILE_WIDTH + ( inCount % TILE_WIDTH != 0 );
  const u32 batchCount = ceil( (float) size / (wordPerBatch * sizeof(Gd)) );

  /* 
  * size: chunk size
  * VRam footprint: (inputcount + outCount) * size + (inputcount + outCount) * sizeof(g) + sizeof(g::GaloisTable)
  *                 <input buffer and output buffer>          <base and exponent>            <Galois log tables>
  * 
  * Assume the total VRam footprint can be fitted into device vram.
  * Align chunkSz to 4 bytes
  * 
  * 
  */

  // TODO :
  // Allocate device memory for input and output (INPUT_W)
  // Launch CUDA Streams
  // Accumulate results using vectorized XOR.
  // Save results to output buffer.

  Gd *d_inputBuf, *d_outputBuf, *d_bases, *d_exponents;
  cudaErrchk( cudaMalloc( (void**) &d_inputBuf, inCount * wordPerBatch * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_outputBuf, tileCount * wordPerBatch * outCount * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_bases, inCount * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_exponents, outCount * sizeof(Gd) ) );  

  dim3 dimGrid( tileCount );
  dim3 dimBlock( TBLOCK_SZ );

  // Sequential kernel invoking
  for ( u32 batchIdx = 0; batchIdx < batchCount; ++batchIdx ) {

    // Copy input data to GPU
    for ( int i = 0; i < inCount; ++i ) {
      void *inputBufOffset = (char*) inputBuf + i * size + batchIdx * wordPerBatch * sizeof(G);
      void *d_inputBufOffset = (char*) d_inputBuf + i * wordPerBatch * sizeof(G);
      cudaErrchk( cudaMemcpy( d_inputBufOffset, inputBufOffset, wordPerBatch * sizeof(G), cudaMemcpyHostToDevice ) );
    }

    // Copy bases and exponents to GPU
    typename G::ValueType *baseOffset = database + inputIdxStart;
    u16 *exponents = new u16[outCount];
    for ( int i = outputIdxStart; i <= outputIdxEnd; ++i ) {
      exponents[i - outputIdxStart] = outputrows[i].exponent;
    }

    cudaErrchk( cudaMemcpy( d_bases, baseOffset, inCount * sizeof(Gd), cudaMemcpyHostToDevice ) );
    cudaErrchk( cudaMemcpy( d_exponents, exponents, outCount * sizeof(u16), cudaMemcpyHostToDevice ) );
    delete [] exponents;

    // Lauch Compute Kernel
    ProcessKer<<<dimGrid, dimBlock, (wordPerBatch + 1) * TILE_WIDTH * sizeof(G)>>> ( wordPerBatch,
                                                                                     d_inputBuf,
                                                                                     d_bases,
                                                                                     outCount,
                                                                                     d_outputBuf,
                                                                                     d_exponents
                                                                                    );
    cudaDeviceSynchronize();
    // Reduce
  }

  
  



 
}

__global__ void ProcessKer( const int batchSz,
                            const void * __restrict__ inputData,
                            const void * __restrict__ bases,
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

  outputBuf Transpose: 
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

  G *smInput = (G *) sharedMem;  // Shared memory input buffer
  G *smBases = (G *) ( sharedMem + batchSz * TILE_WIDTH * sizeof(G) );
  
  const int wordPerInt = sizeof(u32) / sizeof(G);
  const int intPerBatch = batchSz / wordPerInt;
  const int outBufWidth = outputCount * gridDim.x;
  const int outBufRowPos = outputCount * blockIdx.x;

  // Load input data and bases into shared mem
  for ( int i = 0; i < TILE_WIDTH; ++i )
  {
    for ( int j = threadIdx.x; j < intPerBatch; j += blockDim.x )
    {
      ((u32 *) smInput)[i * intPerBatch + j] = ((u32 *) inputData)[(blockIdx.x * TILE_WIDTH + i) * intPerBatch + j];
    }
  }

  for ( int i = threadIdx.x; i < TILE_WIDTH / wordPerInt; i += blockDim.x )
  {
    ((u32 *) smBases)[i] = ((u32 *) bases)[blockIdx.x * TILE_WIDTH / wordPerInt + i];
  }

  // if ( threadIdx.x == 0 ) {
  //   for ( int i = 0; i < batchSz * TILE_WIDTH; ++i ) {
  //     printf( "Block %d: smInput[%d] = %u\n", blockIdx.x, i, smInput[i].Value() );
  //   }
  //   for ( int i = 0; i < TILE_WIDTH; ++i ) {
  //     printf( "Block %d: smBases[%d] = %u\n", blockIdx.x, i, smBases[i].Value() );
  //   }
  // }

  __syncthreads();

  G factor;
  u16 exponent;
  u32 acc = 0;
  u32 words, res;
  // Each thread compute one output block.
  for ( int i = threadIdx.x; i < outputCount; i += blockDim.x )
  {
    exponent = ((u16 *) exponents)[i];
    // For each int in the batch
    for ( int j = 0; j < intPerBatch; ++j )
    {
      acc = 0;
      // For each inputblock in the tile, calculate contribution of corresponding 2 words (a int).
      for ( int k = 0; k < TILE_WIDTH; ++k )
      {
        factor = smBases[k].pow( exponent );
        words = ((u32 *) smInput)[k * intPerBatch + j];
        res = 0;
        for ( int w = 0; w < wordPerInt; ++w )
        {
          ((G *) &res)[w] = factor * ((G *) &words)[w];
        }
        acc ^= res;
      }

      // Write result to outputBuf
      ((u32 *)outputBuf)[j * outBufWidth + outBufRowPos + i] = acc;
    }
  }
}