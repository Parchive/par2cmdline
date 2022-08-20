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
  const u32 wordPerChunk = size / sizeof(Gd);
  // Batch need to be 4-byte aligned
  const u32 wordPerBatch = (TBLOCK_SZ * SHARED_MEM_SZ / ( MAX_THREAD * TILE_WIDTH * sizeof(Gd) ) - 1) & ~1;
  const u32 tileCount = inCount / TILE_WIDTH + ( inCount % TILE_WIDTH != 0 );
  const u32 batchCount = ceil( (float) wordPerChunk / wordPerBatch );

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

  // Allocate GPU memory buffers
  Gd *d_input, *d_intermediate, *d_output, *d_bases, *d_exponents;
  cudaErrchk( cudaMalloc( (void**) &d_input, inCount * wordPerBatch * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_intermediate, tileCount * wordPerBatch * outCount * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_output, outCount * wordPerBatch * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_bases, inCount * sizeof(Gd) ) );
  cudaErrchk( cudaMalloc( (void**) &d_exponents, outCount * sizeof(Gd) ) );

  // Copy bases and exponents to GPU
  u16 *baseOffset = &database[inputIdxStart];
  u16 *exponents = new u16[outCount];
  for ( int i = outputIdxStart; i <= outputIdxEnd; ++i ) {
    exponents[i - outputIdxStart] = outputrows[i].exponent;
  }

  cudaErrchk( cudaMemcpy( d_bases, baseOffset, inCount * sizeof(Gd), cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMemcpy( d_exponents, exponents, outCount * sizeof(u16), cudaMemcpyHostToDevice ) );
  delete [] exponents;

  // Set kernel launch parameters
  dim3 dimGrid( tileCount );
  dim3 dimBlock( TBLOCK_SZ );


  // Sequential kernel invoking
  for ( u32 batchIdx = 0; batchIdx < batchCount; ++batchIdx ) {

    int batchSz = wordPerBatch;
    if ( batchIdx == batchCount - 1 ) {
      batchSz = wordPerChunk - batchIdx * wordPerBatch;
    }
    int batchSzAligned = batchSz + (batchSz & 1);

    // Copy input data to GPU
    for ( int i = 0; i < inCount; ++i ) {
      void *inputBufOffset = (char*) inputBuf + i * size + batchIdx * wordPerBatch * sizeof(G);
      void *d_inputBufOffset = (char*) d_input + i * batchSz * sizeof(G);
      cudaErrchk( cudaMemcpyAsync( d_inputBufOffset, inputBufOffset, batchSz * sizeof(G), cudaMemcpyHostToDevice ) );
    }

    // Lauch Compute Kernel
    ProcessKer<<<dimGrid, dimBlock, (batchSzAligned + 1) * TILE_WIDTH * sizeof(G)>>> ( batchSz,
                                                                                       d_input,
                                                                                       d_bases,
                                                                                       inCount,
                                                                                       outCount,
                                                                                       d_intermediate,
                                                                                       d_exponents
                                                                                      );
    // Lauch Reduce Kernel
    dim3 dimBlockReduce( 32 );
    dim3 dimGridReduce( ceil( outCount / (float) dimBlockReduce.x ), batchSzAligned / 2 );
    ReduceKer<<<dimGridReduce, dimBlockReduce>>>( (u32*) d_intermediate, (u32*) d_output, outCount, tileCount );

    // Copy Result to output buffer
    for ( int i = 0; i < outCount; ++i ){
      cudaErrchk( cudaMemcpyAsync( &((G*) outputBuf)[wordPerChunk * i + wordPerBatch * batchIdx],
                                   &d_output[batchSzAligned * i],
                                   batchSz * sizeof(GaloisCu16),
                                   cudaMemcpyDeviceToHost ) );
    }
    cudaErrchk( cudaDeviceSynchronize() );

  }
 
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
