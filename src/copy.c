//COMPARAISON PERF Z

#include "mnblas.h"
#include <stdio.h>
#include <x86intrin.h>
#include <nmmintrin.h>

typedef double double2 [2] __attribute__ ((aligned (16))) ;

/*************************** CCOPY **************************/

void mncblas_scopy_noomp(const int N, const float *X, const int incX, float *Y, const int incY){
  register unsigned int i = 0 ;
  for (i=0;i < N; i =incX*(i + 4)){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_scopy_omp(const int N, const float *X, const int incX, float *Y, const int incY){
  register unsigned int i = 0 ;
  #pragma omp for schedule(static)
  for (i=0;i < N; i = i + incX*(4)){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_scopy_vec(const int N, const float *X, const int incX, float *Y, const int incY){
  //TODO
}

/*************************** DCOPY **************************/


void mncblas_dcopy_noomp(const int N, const double *X, const int incX, double *Y, const int incY){
  register unsigned int i = 0 ;

  for (i=0;i < N; i = i + incX*(4)){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_dcopy_omp(const int N, const double *X, const int incX, double *Y, const int incY){
  register unsigned int i = 0 ;
  #pragma omp for schedule(static)
  for (i=0;i < N; i = i + incX*(4)){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }
  return ;
}

void mncblas_dcopy_vec(const int N, const double *X, const int incX, double *Y, const int incY){
  //TODO
}

/*************************** CCOPY **************************/

void mncblas_ccopy_noomp(const int N, const void *X, const int incX, void *Y, const int incY)
{
  float *x = (float *) X;
  float *y = (float *) Y;

  register unsigned int i;

  for(i=0;i<2*N;i = i + incX*(4)){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_ccopy_omp(const int N, const void *X, const int incX, void *Y, const int incY)
{
  float *x = (float *) X;
  float *y = (float *) Y;

  register unsigned int i;
  #pragma omp parallel for schedule(static) private(i)
  for(i=0;i<2*N;i = i + incX*(4)){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_ccopy_vec(const int N, const void *X, const int incX, void *Y, const int incY){
  //TODO
}

/*************************** ZCOPY **************************/

void mncblas_zcopy_noomp(const int N, const void *X, const int incX,void *Y, const int incY)
{
  double *x = (double *) X;
  double *y = (double *) Y;
  register unsigned int i;
  for(i=0;i<2*N;i = i + incX*(4)){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_zcopy_omp(const int N, const void *X, const int incX,void *Y, const int incY){
  double *x = (double *) X;
  double *y = (double *) Y;
  register unsigned int i;
  #pragma omp parallel for schedule(static) private(i)
  for(i=0;i<2*N;i = i + incX*(4)){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_zcopy_vec(const int N, const void *X, const int incX,void *Y, const int incY){
  double *x = (double *) X;
  double *y = (double *) Y;
  register unsigned int i;
  __m128d svg;
  for(i=0;i<2*N;i= i + incX*(4)){
    svg = _mm_load_pd(x+i);
    _mm_store_pd(y+i,svg);
  }
}
