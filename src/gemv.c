#include "mnblas.h"
#include <stdio.h>
#include <nmmintrin.h>

typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2[2] __attribute__ ((aligned (16)));

void mncblas_sgemv_vec (const MNCBLAS_LAYOUT layout,const MNCBLAS_TRANSPOSE TransA, const int M, const int N,const float alpha, const float *A, const int lda,const float *X, const int incX, const float beta,float *Y, const int incY)
{
	register unsigned int i ;
	register unsigned int j ;
	register float r ;
	register float x ;
	register unsigned int indice ;

	float4 x4, r4 ;
	__m128 xv4, a1, dot ;
	#pragma omp parallel for schedule(static) private(indice,r,x4,r4,xv4,a1,dot)
	for (i = 0; i < M; i += incX)
	{
		r = 0.0 ;
		indice = i * M ;

		x4 [0] = X [i] ;
		x4 [1] = X [i] ;
		x4 [2] = X [i] ;
		x4 [3] = X [i] ;

		xv4 = _mm_load_ps (x4) ;

		for (j = 0 ; j < M; j += 4)
		{
			a1 = _mm_load_ps (A+indice+j) ;
			dot = _mm_dp_ps (a1, xv4, 0xFF) ;
			_mm_store_ps (r4, dot) ;
			r += r4 [0] ;
		}

		Y [i] = (beta * Y[i])  + (alpha * r) ;

	}

	return ;
}

void mncblas_sgemv_noomp (const MNCBLAS_LAYOUT layout,const MNCBLAS_TRANSPOSE TransA, const int M, const int N,const float alpha, const float *A, const int lda,const float *X, const int incX, const float beta,float *Y, const int incY)
{
	register unsigned int i ;
	register unsigned int j ;
	register float r ;
	register float x ;
	register unsigned int indice ;

	for (i = 0; i < M; i += incX)
	{
		r = 0.0 ;
		x = X [i] ;
		indice = i * M ;

		for (j = 0 ; j < M; j += incY)
		{
			r += A[indice+j] * x ;
		}

		Y [i] = (beta * Y[i])  + (alpha * r) ;

	}
	return ;
}

void mncblas_sgemv_omp (const MNCBLAS_LAYOUT layout,const MNCBLAS_TRANSPOSE TransA, const int M, const int N,const float alpha, const float *A, const int lda,const float *X, const int incX, const float beta,float *Y, const int incY)
{
	register unsigned int i ;
	register unsigned int j ;
	register float r ;
	register float x ;
	register unsigned int indice ;
	r = 0.0 ;
	#pragma omp parallel for schedule(static) private(indice,r)
	for (i = 0; i < M; i += incX)
	{
		x = X [i] ;
		indice = i * M ;
		for (j = 0 ; j < M; j += incY)
		{
			r += A[indice+j] * x ;
		}
		Y [i] = (beta * Y[i])  + (alpha * r) ;
	}
	return ;
}



void mncblas_dgemv_vec (MNCBLAS_LAYOUT layout,MNCBLAS_TRANSPOSE TransA, const int M, const int N,const double alpha, const double *A, const int lda,const double *X, const int incX, const double beta,double *Y, const int incY)
{
	register unsigned int i ;
	register unsigned int j ;
	register float r ;
	register float x ;
	register unsigned int indice ;

	double2 x2, r2 ;
	__m128d xv2, a1, dot ;

	#pragma omp parallel for schedule(static) private(indice,r,x2,r2,xv2,a1,dot)

	for (i = 0; i < M; i += incX)
	{
		r = 0.0 ;
		indice = i * M ;

		x2 [0] = X [i] ;
		x2 [1] = X [i] ;

		xv2 = _mm_load_pd (x2) ;

		for (j = 0 ; j < M; j += 4)
		{
			a1 = _mm_load_pd (A+indice+j) ;
			dot = _mm_dp_pd (a1, xv2, 0xFF) ;
			_mm_store_pd (r2, dot) ;
			r += r2 [0] ;
		}

		Y [i] = (beta * Y[i])  + (alpha * r) ;

	}

	return ;
}

void mncblas_dgemv_noomp (MNCBLAS_LAYOUT layout,MNCBLAS_TRANSPOSE TransA, const int M, const int N,const double alpha, const double *A, const int lda,const double *X, const int incX, const double beta,double *Y, const int incY)
{
	register unsigned int i ;
	register unsigned int j ;
	register float r ;
	register float x ;
	register unsigned int indice ;

	for (i = 0; i < M; i += incX)
	{
		r = 0.0 ;
		x = X [i] ;
		indice = i * M ;

		for (j = 0 ; j < M; j += incY)
		{
			r += A[indice+j] * x ;
		}

		Y [i] = (beta * Y[i])  + (alpha * r) ;

	}
	return ;
}

void mncblas_dgemv_omp (MNCBLAS_LAYOUT layout,MNCBLAS_TRANSPOSE TransA, const int M, const int N,const double alpha, const double *A, const int lda,const double *X, const int incX, const double beta,double *Y, const int incY)
{
	register unsigned int i ;
	register unsigned int j ;
	register float r ;
	register float x ;
	register unsigned int indice ;
	r = 0.0 ;
	#pragma omp parallel for schedule(static) private(indice,r)
	for (i = 0; i < M; i += incX)
	{
		x = X [i] ;
		indice = i * M ;
		for (j = 0 ; j < M; j += incY)
		{
			r += A[indice+j] * x ;
		}
		Y [i] = (beta * Y[i])  + (alpha * r) ;
	}
	return ;
}
void mncblas_cgemv (MNCBLAS_LAYOUT layout,MNCBLAS_TRANSPOSE TransA, const int M, const int N,const void *alpha, const void *A, const int lda,const void *X, const int incX, const void *beta,void *Y, const int incY)
{
	return ;
}


void mncblas_zgemv (MNCBLAS_LAYOUT layout,MNCBLAS_TRANSPOSE TransA, const int M, const int N,const void *alpha, const void *A, const int lda,const void *X, const int incX, const void *beta,void *Y, const int incY)
{

	return ;
}
