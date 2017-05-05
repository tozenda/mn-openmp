//COMPARAISON PERF cdotc_sub

#include "mnblas.h"

float mncblas_sdot(const int N, const float *X, const int incX,const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register float dot = 0.0 ;

  for (i=0;i<N;i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }

  return dot ;
}

float mncblas_sdot_noomp(const int N, const float *X, const int incX,const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register float dot = 0.0 ;

  for (i=0;i<N;i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }

  return dot ;
}

float mncblas_sdot_omp(const int N, const float *X, const int incX,const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register float dot = 0.0 ;
  #pragma omp for schedule(static)
  for (i=0;i<N;i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }

  return dot ;
}

double mncblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;

  for (i=0;i<N;i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }

  return dot ;
}

double mncblas_ddot_noomp(const int N, const double *X, const int incX, const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;

  for (i=0;i<N;i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }

  return dot ;
}

double mncblas_ddot_omp(const int N, const double *X, const int incX, const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;
  #pragma omp for schedule(static) private(i)
  for (i=0;i<N;i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }

  return dot ;
}
void   mncblas_cdotu_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotu)
{
  /* a completer */

  return ;
}

void   mncblas_cdotc_sub(const int N, const void *X, const int incX,const void *Y, const int incY, void *dotc)
{
  /* a completer */

  return ;
}

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,const void *Y, const int incY, void *dotu)
{
  /* a completer */

  return ;
}

void   mncblas_zdotc_sub(const int N, const void *X, const int incX,const void *Y, const int incY, void *dotc)
{
  /* a completer */

  return ;
}
