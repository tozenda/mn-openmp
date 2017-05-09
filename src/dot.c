//COMPARAISON PERF cdotc_sub

#include "mnblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <nmmintrin.h>

typedef float float4 [4]  __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;

/*************************** SDOT **************************/

float mncblas_sdot_noomp(const int N, const float *X, const int incX,const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register float dot = 0.0 ;

  for (i=0;i<N;i += 1)
  {

    dot = dot + X [i*incX] * Y [i*incY] ;
  }

  return dot ;
}

float mncblas_sdot_omp(const int N, const float *X, const int incX,const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  float dot = 0.0 ;

  #pragma omp for schedule(static)
  for (i=0;i<N;i += 1)
  {
    dot = dot + X [i*incX] * Y [i*incY] ;
  }

  return dot ;
}

float mncblas_sdot_vec(const int N, const float *X, const int incX,const float *Y, const int incY)
{
  register unsigned int i = 0;
  float4 dot;
  float d = 0.0;
  __m128 x1, y1;

  #pragma omp parallel for reduction(+:d) private(x1, y1)
  for (i=0; i<N ; i += 4){
    x1 = _mm_load_ps (X+i) ;
    y1 = _mm_load_ps (Y+i) ;
    y1 = _mm_dp_ps(y1, x1, 0xFF);
    _mm_store_ps (dot, y1) ;
    d = d + (*dot);
  }
  return(d);
}

/*************************** DDOT **************************/

double mncblas_ddot_noomp(const int N, const double *X, const int incX, const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;

  for (i=0;i<N;i += 1)
  {
    dot = dot + X [i*incX] * Y [i*incY] ;
  }


  return dot ;
}

double mncblas_ddot_omp(const int N, const double *X, const int incX, const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;

  #pragma omp for schedule(static) private(i)
  for (i=0;i<N;i += 1)
  {
    dot = dot + X [i*incX] * Y [i*incY] ;
  }

  return dot ;
}

double mncblas_ddot_vec(const int N, const double *X, const int incX, const double *Y, const int incY)
{
  register unsigned int i = 0;
  double2 dot;
  double d = 0.0;

  __m128d x1, y1;

  #pragma omp parallel for reduction(+:d) schedule (static) private(x1, y1)
  for (i=0;i<N;i += 4){
    x1 = _mm_load_pd (X+i) ;
    y1 = _mm_load_pd (Y+i) ;
    y1 = _mm_dp_pd(y1, x1, 0xFF);
    _mm_store_pd(dot, y1);
    d = d + *dot;
  }
  return(d);
}

/*************************** CDOTU_SUB **************************/

void   mncblas_cdotu_sub_noomp(const int N, const void *X, const int incX,
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

void   mncblas_cdotu_sub_omp(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i;
  float *x = (float *)X;
  float *y = (float *)Y;
  float *dot = (float *)dotu;

  #pragma omp for schedule(static)
  for (i=0; i < N ; i += 1){
    if((i*(incX+incY))%2==1){
      if(i*incX%2==1)
        dot[0] = dot[0] + (x[i*incX] * y[i*incY]);
      else
        dot[0] = dot[0] - (x[i*incX] * y[i*incY]);
      }
    else{
      dot[1] = dot[1] + x[i*incX] * y[incY];
    }
  }
}

void   mncblas_cdotu_sub_vec(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  // register unsigned int i;
  // float *x = (float *)X;
  // float *y = (float *)Y;
  // float *dot = (float *)dotu;

  // __m128 x1, y1, d0, d1;
  // float alpha[1] = {0.0};
  // d0 = _mm_load_ps(dot);
  // d1 = _mm_load_ps(dot);

  // printf("Les valeurs sont initialisés\n");
  // #pragma omp for schedule(static) private(i)
  // for (i=0; i < N ; i += 1){
  //   x1 = _mm_load_ps (x+i*incX) ;
  //   y1 = _mm_load_ps (y+i*incY) ;
  //   y1 = _mm_mul_ps (x1, y1) ;
  //   printf("Initialisation dans le for\n");
  //   if((i*(incX+incY))%2==1){
  //     if(i*incX%2==1)
  //       d0 = _mm_add_ps (d0, y1) ;
  //     else
  //       d0 = _mm_sub_ps (d0, y1) ;
  //     }
  //   else{
  //     d1 = _mm_add_ps(d1, y1);
  //   }
  // }
  // printf("On sort du for\n");
  // float *tmp;
  // _mm_store_ps (tmp, d0) ;
  // dot[0] = *tmp;
  // _mm_store_ps(tmp, d1);
  // dot[1] = *tmp;
}

/*************************** CDOTC_SUB **************************/

//complexe z !x*y


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
  for (i=0;i<N;i += 1){
    if(i*(incX+incY)%2==1){
      dot[0] = dot[0] + (x[i*incX] * y[i*incY]);
    }
    else{
      if(i*incX%2==0)
        dot[1] = dot[1] + (x[i*incX] * y[i*incY]);
      else
        dot[1] = dot[1] - x[i*incX] * y[i*incY];
    }
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

/*************************** ZDOTU_SUB **************************/

void   mncblas_zdotu_sub_noomp(const int N, const void *X, const int incX,
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

void   mncblas_zdotu_sub_omp(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotu;

  for (i ; i < N ; i += 1){
    if((i*(incX+incY))%2==1){
      if(i*incX%2==1)
        dot[0] = dot[0] + (x[i*incX] * y[i*incY]);
      else
        dot[0] = dot[0] - (x[i*incX] * y[i*incY]);
      }
    else{
      dot[1] = dot[1] + x[i*incX] * y[i*incY];
    }
  }
}

  void   mncblas_zdotu_sub_vec(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotu;

  __m128d x1, y1;

  #pragma omp parallel for schedule (static) private(x1, y1)
  for (i=0;i<N;i += 4){
    x1 = _mm_load_pd (x+i*incX) ;
    y1 = _mm_load_pd (y+i*incY) ;
    y1 = _mm_mul_pd (x1, y1) ;
    _mm_store_pd (dot, y1) ;
  }
}

/*************************** ZDOTC_SUB **************************/

void   mncblas_zdotc_sub_noomp(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotc;

  for (i ; i < N ; i += 1){
    if(i*(incX+incY)%2==1){
      dot[0] = dot[0] + (x[i*incX] * y[i*incY]);
    }
    else{
      if(i*incX%2==0)
        dot[1] = dot[1] + (x[i*incX] * y[i*incY]);
      else
        dot[1] = dot[1] - x[i*incX] * y[i*incY];
    }
  }
}

void   mncblas_zdotc_sub_omp(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotc;

  #pragma omp parallel for private(i)
  for (i=0 ; i < N ; i += 1){
    if(i*(incX+incY)%2==1){
      dot[0] = dot[0] + (x[i*incX] * y[i*incY]);
    }
    else{
      if(i*incX%2==0)
        dot[1] = dot[1] + (x[i*incX] * y[i*incY]);
      else
        dot[1] = dot[1] - x[i*incX] * y[i*incY];
    }
  }
}

void   mncblas_zdotc_sub_vec(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  double *x = (double *)X;
  double *y = (double *)Y;
  double *dot = (double *)dotc;

  __m128d x1, y1;

  #pragma omp parallel for schedule (static) private(x1, y1)
  for (i=0;i<N;i += 4){
    x1 = _mm_load_pd (x+i*incX) ;
    y1 = _mm_load_pd (y+i*incY) ;
    y1 = _mm_mul_pd (x1, y1) ;
    _mm_store_pd (dot, y1) ;
  }
}