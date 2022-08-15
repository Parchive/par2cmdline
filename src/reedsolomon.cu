#include <assert.h>

#include "libpar2internal.h"
#include "helper_cuda.cuh"

#define TBLOCK_SZ = 256
#define MAX_THREAD = 1024
#define INPUT_W = 32
#define SHARED_MEM_SZ = 65536

template <typename G>
__global__ void ProcessKer( const int chunkSz,
                            const int inputCount,
                            const void * __restrict__ inputData,
                            const void * __restrict__ bases,
                            const int outputCount,
                            void * __restrict__ outputBuf,
                            const void * __restrict__ exponents,
                            const int batchSz );

// Calculate a chunk of all recovery blocks on CUDA device
template<typename g>
bool ReedSolomon<g>::ProcessCu(size_t size, u32 inputIdxStart, u32 inputIdxEnd, const void *inputBuf,
                               u32 outputIdxStart, u32 outputIdxEnd, void *outputBuf)
{
  // CUDA Device compatible Galois type.
  typedef GaloisCu<G::Bits, G::Generator, G::ValueType> Gd;
  /* 
  * size: chunk size
  * VRam footprint: (inputcount + outCount) * size + (inputcount + outCount) * sizeof(g) + sizeof(g::GaloisTable)
  *                 <input buffer and output buffer>          <base and exponent>            <Galois log tables>
  * 
  * Assume the total VRam footprint can be fitted into device vram.
  * Align chunkSz to 4 bytes
  * 
  */

  u32 outCount = outputIdxEnd - outputIdxStart + 1;
  u32 inCount = inputIdxEnd - inputIdxStart + 1;
  u32 batchSz = TBLOCK_SZ * SHARED_MEM_SZ / ( MAX_THREAD * INPUT_W * sizeof(Gd) ) - 1;
  u32 streamCount = inCount / INPUT_W + ( inCount % INPUT_W != 0 );
  
  // TODO :
  // Allocate device memory for input and output (INPUT_W)
  // Launch CUDA Streams
  // Accumulate results using vectorized XOR.
  // Save results to output buffer.
 
}

template <typename G>
__global__ void ProcessKer( const int chunkSz,
                            const int inputCount,
                            const void * __restrict__ inputData,
                            const void * __restrict__ bases,
                            const int outputCount,
                            void * __restrict__ outputBuf,
                            const void * __restrict__ exponents,
                            const int batchSz )
{
  __shared__ G smInput[inputCount * batchSz];  // Shared memory input buffer
  __shared__ G smBases[inputCount];
  
  const int wordPerInt = 32 / G::Bits;
  const int intPerBatch = batchSz / wordPerInt;
  const int intPerChunk = chunkSz >> 2;

  const int startInputInt = blockIdx.x * intPerBatch;

  // Load input data and bases into shared mem
  for ( int i = 0; i < inputCount; ++i )
  {
    for ( int j = threadIdx.x; j < intPerBatch; j += blockDim.x )
    {
      ((u32 *) smInput)[i * intPerBatch + j] = ((u32 *) inputData)[i * intPerChunk + startInputInt + j];
    }
  }
  for ( int i = threadIdx.x; i < inputCount / wordPerInt; i += blockDim.x )
  {
    ((u32 *) smBases)[i] = ((u32 *) bases)[i];
  }
  __syncthreads();

  // Each thread compute one output block.
  for ( int i = threadIdx.x; i < outputCount; i += blockDim.x )
  {
    const G exponent = (G *) exponents[i];
    // For each int in the batch
    for ( int j = 0; j < intPerBatch; ++j )
    {
      u32 acc;
      // Compute the contribution of words from each inputblock
      for ( int k = 0; k < inputCount; ++k )
      {
        const G left = smBases[k].pow(exponent);
        u32 words = ((u32 *) smInput)[k * intPerBatch + j];
        u32 res;
        for ( int w = 0; w < wordPerInt; ++w )
        {
          ((G *) res)[w] = left * ((G *) words)[w];
        }
        acc ^= res;
      }

      // Write result to outputBuf
      ((u32 *)outputBuf)[i * intPerChunk + startInputInt + j] = acc;
    }
  }
}