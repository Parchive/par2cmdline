#include "libpar2.h"
#include "galois_cu.cuh"
#include "helper_cuda.cuh"
#include <stdio.h>

void TestMult(void);
void TestMultInpl(void);
void TestDev(void);
void TestDevInpl(void);
void TestAdd(void);
void TestAddInpl(void);
void TestSub(void);
void TestSubInpl(void);

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
  TestMultInpl();
  TestDev();
  TestDevInpl();
  TestAdd();
  TestAddInpl();
  TestSub();
  TestSubInpl();

  return 0;
}

__global__
void KerMult(__restrict__ GaloisCu16 *vars, __restrict__ GaloisCu16 *res)
{
  
}

void TestMult()
{
  Galois16 vals[ Galois16::Count ], *results;
  GaloisCu16 valsCu[ GaloisCu16::Count ], *resultsCu;

  // Fill arrays with every element in GF(2^16)
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    vals[ i ] = Galois16( ( Galois16::ValueType ) i );
    valsCu[ i ] = GaloisCu16( ( GaloisCu16::ValueType ) i );
  }

  // Allocate space for multiplication result
  size_t resultSz = sizeof( Galois16 ) * Galois16::Count * Galois16::Count / 2;
  results = ( Galois16* ) malloc(resultSz);
  resultsCu = ( GaloisCu16* )  malloc(resultSz);
  // Calculate multiplication results on CPU.
  size_t resId = 0;
  for ( size_t i = 0; i < Galois16::Count; ++i)
  {
    #pragma clang loop vectorize(enable) interleave(enable)
    for ( size_t j = i; j < Galois16::Count; ++j)
    {
      results[resId] = vals[i] * vals[j];
      ++resId;
    }
  }
  GaloisCu16 *d_valsCu, *d_resultsCu;
  cudaErrchk( cudaMalloc( (void**) &d_valsCu, sizeof( valsCu ) ) );
  cudaErrchk( cudaMalloc( (void**) &d_resultsCu, resultSz ) );
  cudaErrchk( cudaMemcpy( d_valsCu, valsCu, sizeof( valsCu ), cudaMemcpyHostToDevice ) );


  
}