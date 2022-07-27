#include <stdio.h>
#include <omp.h>
#include <iostream>
#include "libpar2.h"
#include "galois_cu.cuh"
#include "helper_cuda.cuh"
#include "../profiling/Timer.h"

#define BLOCK_WIDTH 64

void TestMult(void);
void TestMultInpl(void);
void TestDev(void);
void TestDevInpl(void);
void TestAdd(void);
void TestAddInpl(void);
void TestSub(void);
void TestSubInpl(void);
void CompareResults(const Galois16 *resCPU, const GaloisCu16 *resGPU, char op);

__global__
void callDevice( GaloisCu16 *a, GaloisCu16 *b )
{
  printf( "On device\n" );
  printf( "a + b is %u\n", ( *a + *b ).Value() );
  printf( "a - b is %u\n", ( *a - *b ).Value() );
  printf( "a * b is %u\n", ( *a * *b ).Value() );
}

int main()
{
  GaloisCu16::uploadTable();

  TestMult();
  // TestMultInpl();
  // TestDev();
  // TestDevInpl();
  // TestAdd();
  // TestAddInpl();
  // TestSub();
  // TestSubInpl();

  return 0;
}

void CompareResults(const Galois16 *resCPU, const GaloisCu16 *resGPU, char op)
{
  printf("Varifying results.\n");
  bool correct = true;
  size_t resId = 0;
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    for ( size_t j = i; j < Galois16::Count; ++j)
    {
      if ( resCPU[resId].Value() != resGPU[resId].Value() )
      {
        correct = false;
        printf("Result doesn't match! %u %c %u = %u, but got %u.\n",
                i, op, j, resCPU[resId], resGPU[resId]);
      }
      ++resId;
    }
  }

  if (correct)
  {
    printf("Operation %c is correct!\n", op);
  }
}


__global__
void KerMult(__restrict__ GaloisCu16 *vars, __restrict__ GaloisCu16 *res)
{
  unsigned int outRowIdx = blockIdx.x * blockDim.x + threadIdx.x;   // Also the idx of left var.
  unsigned int outColMax = GaloisCu16::Count - outRowIdx;
  unsigned int outStartIdx = ( outColMax + GaloisCu16::Count ) * outRowIdx / 2;
  GaloisCu16 left = vars[ outRowIdx ];

  for ( unsigned int i = 0; i < outColMax; ++i )
  {
    res[ outStartIdx + i ] = left * vars[ outRowIdx + i ];
  }
  
}

void TestMult()
{
  printf("Press any key to start multiplication test...\n");
  std::cin.get();

  Galois16 vals[ Galois16::Count ], *results;
  GaloisCu16 valsCu[ GaloisCu16::Count ], *resultsCu;

  // Fill arrays with every element in GF(2^16)
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    vals[ i ] = Galois16( ( Galois16::ValueType ) i );
    valsCu[ i ] = GaloisCu16( ( Galois16::ValueType ) i );
  }
  printf("Input arrays filled.\n");

  // Allocate space for multiplication result
  size_t resultSz = sizeof( Galois16 ) * ( Galois16::Count + 1 ) * Galois16::Count / 2;
  results = ( Galois16* ) malloc(resultSz);
  resultsCu = ( GaloisCu16* )  malloc(resultSz);
  printf("Output arrays allocated\n");
  printf("Calculating reference results.\n");
  // Calculate multiplication results on CPU.
  {
    Timer timer("CPU");
    size_t resId = 0;
    for ( size_t i = 0; i < Galois16::Count; ++i)
    {
      #pragma omp parallel
      for ( size_t j = i; j < Galois16::Count; ++j)
      {
        results[resId] = vals[i] * vals[j];
        #pragma omp atomic
        ++resId;
      }
    }
  }
  // Upload data to GPU
  GaloisCu16 *d_valsCu, *d_resultsCu;
  cudaErrchk( cudaMalloc( (void**) &d_valsCu, sizeof( valsCu ) ) );
  cudaErrchk( cudaMalloc( (void**) &d_resultsCu, resultSz ) );
  cudaErrchk( cudaMemcpy( d_valsCu, valsCu, sizeof( valsCu ), cudaMemcpyHostToDevice ) );
  printf("Input copied to GPU.\n");

  // Launch kernel
  dim3 dimBlock(BLOCK_WIDTH);
  dim3 dimGrid(GaloisCu16::Count / BLOCK_WIDTH);
  {
    Timer timer("GPU");
    KerMult<<<dimGrid, dimBlock>>> (d_valsCu, d_resultsCu);
    cudaErrchk( cudaGetLastError() );
    printf("Kernel launched.\n");

    cudaDeviceSynchronize();
    printf("Kernel completed.\n");
  }
  // Download results from GPU
  cudaErrchk ( cudaMemcpy(resultsCu, d_resultsCu, resultSz, cudaMemcpyDeviceToHost) );
  CompareResults(results, resultsCu, '*');
  
}