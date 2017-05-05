//COMPARAISON PERF DOUBLE (d)

#include "mnblas.h"
#include <stdio.h>
#include <x86intrin.h>
#include <nmmintrin.h>

typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;

void mncblas_saxpy_vec (const int N, const float alpha, const float *X, const int incX, float *Y, const int incY){
	register unsigned int i ;
	register unsigned int j ;

	float4 alpha4 ;

	__m128 x1, x2, y1, y2 ;
	__m128 alpha1;

	alpha4 [0] = alpha;
	alpha4 [1] = alpha;
	alpha4 [2] = alpha;
	alpha4 [3] = alpha;

	alpha1 = _mm_load_ps (alpha4) ;

	#pragma omp parallel for schedule (static)
	for (i = 0; i<N; i += 4)
	{
		x1 = _mm_load_ps (X+i) ;
		y1 = _mm_load_ps (Y+i) ;
		x2 = _mm_mul_ps (x1, alpha1) ;
		y2 = _mm_add_ps (y1, x2) ;
		_mm_store_ps (Y+i, y2) ;
	}

	return ;
}


void mncblas_saxpy_noomp (const int N, const float alpha, const float *X, const int incX, float *Y, const int incY){
	/*
	scalar version with unrolled loop
	*/
	register unsigned int i ;
	for ( i = 0; i < N; i += 4) {
		Y [i] = alpha * X[i] + Y[i] ;
		Y [i+1] = alpha * X[i+1] + Y[i+1] ;
		Y [i+2] = alpha * X[i+2] + Y[i+2] ;
		Y [i+3] = alpha * X[i+3] + Y[i+3] ;
	}
	return ;
}

void mncblas_saxpy_omp (const int N, const float alpha, const float *X,const int incX, float *Y, const int incY){
	/*
	scalar version with unrolled loop
	*/
	register unsigned int i ;
	#pragma omp parallel for schedule (static)
	for ( i = 0; i < N; i += 4) {
		Y [i] = alpha * X[i] + Y[i] ;
		Y [i+1] = alpha * X[i+1] + Y[i+1] ;
		Y [i+2] = alpha * X[i+2] + Y[i+2] ;
		Y [i+3] = alpha * X[i+3] + Y[i+3] ;
	}
	return ;
}

void mncblas_daxpy(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY){
	//TODO
}

void mncblas_daxpy_vec(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY){
	register unsigned int i ;
	register unsigned int j ;

	double2 alpha2 ;

	__m128d x1, x2, y1, y2 ;
	__m128d alpha1;

	alpha2 [0] = alpha;
	alpha2 [1] = alpha;
	// alpha4 [2] = alpha;
	// alpha4 [3] = alpha;

	alpha1 = _mm_load_pd (alpha2) ;

	#pragma omp parallel for schedule (static)
	for (i = 0; i<N; i += 4)
	{
		x1 = _mm_load_pd (X+i) ;
		y1 = _mm_load_pd (Y+i) ;
		x2 = _mm_mul_pd (x1, alpha1) ;
		y2 = _mm_add_pd (y1, x2) ;
		_mm_store_pd (Y+i, y2) ;
	}

	return ;
}


void mncblas_daxpy_noomp(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY){
	register unsigned int i ;

	for ( i = 0; i < N; i += 4) {

		Y [i] = alpha * X[i] + Y[i] ;
		Y [i+1] = alpha * X[i+1] + Y[i+1] ;
		Y [i+2] = alpha * X[i+2] + Y[i+2] ;
		Y [i+3] = alpha * X[i+3] + Y[i+3] ;
	}
	return ;
}

void mncblas_daxpy_omp (const int N, const double alpha, const double *X, const int incX, double *Y, const int incY){
	/*
	scalar version with unrolled loop
	*/
	register unsigned int i ;

	#pragma omp parallel for schedule (static)
	for ( i = 0; i < N; i += 4) {
		Y [i] = alpha * X[i] + Y[i] ;
		Y [i+1] = alpha * X[i+1] + Y[i+1] ;
		Y [i+2] = alpha * X[i+2] + Y[i+2] ;
		Y [i+3] = alpha * X[i+3] + Y[i+3] ;
	}
	return ;
}

void mncblas_caxpy (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{
	register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *a = (float *)alpha;
 	float *x = (float *)X;
 	float *y = (float *)Y;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){ 
    y[j] = (a[0]*x[i]) + y[j]; 
  }
}


void mncblas_zaxpy (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{
	register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  double *a = (double *)alpha;
 	double *x = (double *)X;
 	double *y = (double *)Y;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY){ 
    y[j] = (a[0]*x[i]) + y[j]; 
  }
}