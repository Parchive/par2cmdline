#ifdef __CUDACC__
#include "galois_cu.cuh"

// Arithmatic Operators

template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype>::Galois(typename Galois<bits,generator,valuetype>::ValueType v)
{
  value = v;
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype> Galois<bits,generator,valuetype>::operator * (const Galois<bits,generator,valuetype> &right) const
{
  if (value == 0 || right.value == 0) return 0;
  unsigned int sum = table.log[value] + table.log[right.value];
  if (sum >= Limit)
  {
    return table.antilog[sum-Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype>& Galois<bits,generator,valuetype>::operator *= (const Galois<bits,generator,valuetype> &right)
{
  if (value == 0 || right.value == 0)
  {
    value = 0;
  }
  else
  {
    unsigned int sum = table.log[value] + table.log[right.value];
    if (sum >= Limit)
    {
      value = table.antilog[sum-Limit];
    }
    else
    {
      value = table.antilog[sum];
    }
  }

  return *this;
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype> Galois<bits,generator,valuetype>::operator / (const Galois<bits,generator,valuetype> &right) const
{
  if (value == 0) return 0;

  assert(right.value != 0);
  if (right.value == 0) {return 0;} // Division by 0!

  int sum = table.log[value] - table.log[right.value];
  if (sum < 0)
  {
    return table.antilog[sum+Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype>& Galois<bits,generator,valuetype>::operator /= (const Galois<bits,generator,valuetype> &right)
{
  if (value == 0) return *this;

  assert(right.value != 0);
  if (right.value == 0) {return *this;} // Division by 0!

  int sum = table.log[value] - table.log[right.value];
  if (sum < 0)
  {
    value = table.antilog[sum+Limit];
  }
  else
  {
    value = table.antilog[sum];
  }

  return *this;
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype> Galois<bits,generator,valuetype>::pow(unsigned int right) const
{
  if (right == 0) return 1;
  if (value == 0) return 0;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> Bits) + (sum & Limit);
  if (sum >= Limit)
  {
    return table.antilog[sum-Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype> Galois<bits,generator,valuetype>::operator ^ (unsigned int right) const
{
  if (right == 0) return 1;
  if (value == 0) return 0;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> Bits) + (sum & Limit);
  if (sum >= Limit)
  {
    return table.antilog[sum-Limit];
  }
  else
  {
    return table.antilog[sum];
  }
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline Galois<bits,generator,valuetype>& Galois<bits,generator,valuetype>::operator ^= (unsigned int right)
{
  if (right == 0) {value = 1; return *this;}
  if (value == 0) return *this;

  unsigned int sum = table.log[value] * right;

  sum = (sum >> Bits) + (sum & Limit);
  if (sum >= Limit)
  {
    value = table.antilog[sum-Limit];
  }
  else
  {
    value = table.antilog[sum];
  }

  return *this;
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline valuetype Galois<bits,generator,valuetype>::Log(void) const
{
  return table.log[value];
}


template <const unsigned int bits, const unsigned int generator, typename valuetype>
__device__ inline valuetype Galois<bits,generator,valuetype>::ALog(void) const
{
  return table.antilog[value];
}

#endif // __CUDACC__