//COMPARAISON PERF cdotc_sub

#include "mnblas.h"
#include <stdio.h>
#include <x86intrin.h>
#include <nmmintrin.h>

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


void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *x = (float *)X;
  float *y = (float *)Y;
  float *dot = (float *)dotu;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){
    if((i+j)%2==1){
      if(i%2==1)
        dot[0] = dot[0] + (x[i] * y[j]);
      else
        dot[0] = dot[0] - (x[i] * y[j]);
      }
    else{
      dot[1] = dot[1] + x[i] * y[j]; 
    }
  }
}


//complexe z !x*y
void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *x = (float *)X;
  float *y = (float *)Y;
  float *dot = (float *)dotc;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){
    if((i+j)%2==1){
      dot[0] = dot[0] + (x[i] * y[j]);
    }
    else{
      if(i%2==0)
        dot[1] = dot[1] + (x[i] * y[j]);
      else
        dot[1] = dot[1] - x[i] * y[j];
    }
  }
}

void   mncblas_cdotc_sub_vec(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *x = (float *)X;
  float *y = (float *)Y;
  float *dot = (float *)dotc;

  __m128 x1, y1, d;

  #pragma omp parallel for schedule (static) private(x1, y1, d)
  for (i=0;i<N;i += 4){
    x1 = _mm_load_ps (x+i) ;
    y1 = _mm_load_ps (y+i) ;
    y1 = _mm_mul_ps (x1, y1) ;
    _mm_store_ps (dot, y1) ;
  }
}

void   mncblas_cdotc_sub_omp(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *x = (float *)X;
  float *y = (float *)Y;
  float *dot = (float *)dotc;

  #pragma omp parallel for schedule (static)
  for (i=0;i<N;i += incX){
    *dot = *dot + x [i] * y [i] ;
  }
}

void   mncblas_cdotc_sub_noomp(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *x = (float *)X;
  float *y = (float *)Y;
  float *dot = (float *)dotc;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){
    if((i+j)%2==1){
      dot[0] = dot[0] + (x[i] * y[j]);
    }
    else{
      if(i%2==0)
        dot[1] = dot[1] + (x[i] * y[j]);
      else
        dot[1] = dot[1] - x[i] * y[j];
    }
  }
}


void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotu;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){
    if((i+j)%2==1){
      if(i%2==1)
        dot[0] = dot[0] + (x[i] * y[j]);
      else
        dot[0] = dot[0] - (x[i] * y[j]);
      }
    else{
      dot[1] = dot[1] + x[i] * y[j];
    }
  }
}

void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotc;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){
    if((i+j)%2==1){
      dot[0] = dot[0] + (x[i] * y[j]);
    }
    else{
      if(i%2==0)
        dot[1] = dot[1] + (x[i] * y[j]);
      else
        dot[1] = dot[1] - x[i] * y[j];
    }
  }
}
