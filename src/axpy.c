//COMPARAISON PERF DOUBLE (d)

#include "mnblas.h"
#include <stdio.h>
#include <x86intrin.h>
#include <nmmintrin.h>

typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;


/*************************** SAXPY **************************/

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

	#pragma omp parallel for schedule (static) private(x1,x2,y1,y2,alpha1,i)
	for (i = 0; i<N; i += 4)
	{
		x1 = _mm_load_ps (X+(i*incX)) ;
		y1 = _mm_load_ps (Y+(i*incY)) ;
		x2 = _mm_mul_ps (x1, alpha1) ;
		y2 = _mm_add_ps (y1, x2) ;
		_mm_store_ps (Y+(i*incY), y2) ;
	}
}


void mncblas_saxpy_noomp (const int N, const float alpha, const float *X, const int incX, float *Y, const int incY){
	/*
	scalar version with unrolled loop
	*/
	register unsigned int i ;
	for (i=0 ; i < N; i += 4){
    	Y [i*incY] = alpha * X[i*incX] + Y[i*incY] ;
		Y [(i+1)*incY] = alpha * X[(i+1)*incX] + Y[(i+1)*incY] ;
		Y [(i+2)*incY] = alpha * X[(i+2)*incX] + Y[(i+2)*incY] ;
		Y [(i+3)*incY] = alpha * X[(i+3)*incX] + Y[(i+3)*incY] ;
  	}
}

void mncblas_saxpy_omp (const int N, const float alpha, const float *X,const int incX, float *Y, const int incY){
	/*
	scalar version with unrolled loop
	*/
	register unsigned int i ;
	#pragma omp parallel for schedule (static)
	for (i=0 ; i < N; i += 4){
    	Y [i*incY] = alpha * X[i*incX] + Y[i*incY] ;
		Y [(i+1)*incY] = alpha * X[(i+1)*incX] + Y[(i+1)*incY] ;
		Y [(i+2)*incY] = alpha * X[(i+2)*incX] + Y[(i+2)*incY] ;
		Y [(i+3)*incY] = alpha * X[(i+3)*incX] + Y[(i+3)*incY] ;
  	}
}

/*************************** DAXPY **************************/

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

	#pragma omp parallel for schedule (static) private(x1,x2,y1,y2,alpha1,i)
	for (i = 0; i<N; i += 4)
	{
		x1 = _mm_load_pd (X+(i*incX)) ;
		y1 = _mm_load_pd (Y+(i*incY)) ;
		x2 = _mm_mul_pd (x1, alpha1) ;
		y2 = _mm_add_pd (y1, x2) ;
		_mm_store_pd (Y+(i*incY), y2) ;
	}
}


void mncblas_daxpy_noomp(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY){
	register unsigned int i ;

  	for (i=0 ; i < N; i += 4){
    	Y [i*incY] = alpha * X[i*incX] + Y[i*incY] ;
		Y [(i+1)*incY] = alpha * X[(i+1)*incX] + Y[(i+1)*incY] ;
		Y [(i+2)*incY] = alpha * X[(i+2)*incX] + Y[(i+2)*incY] ;
		Y [(i+3)*incY] = alpha * X[(i+3)*incX] + Y[(i+3)*incY] ;
  	}
}

void mncblas_daxpy_omp (const int N, const double alpha, const double *X, const int incX, double *Y, const int incY){
	/*
	scalar version with unrolled loop
	*/
	register unsigned int i ;

	#pragma omp parallel for schedule (static)
  	for (i=0 ; i < N; i += 4){
    	Y [i*incY] = alpha * X[i*incX] + Y[i*incY] ;
		Y [(i+1)*incY] = alpha * X[(i+1)*incX] + Y[(i+1)*incY] ;
		Y [(i+2)*incY] = alpha * X[(i+2)*incX] + Y[(i+2)*incY] ;
		Y [(i+3)*incY] = alpha * X[(i+3)*incX] + Y[(i+3)*incY] ;
  	}
}

/*************************** CAXPY **************************/

void mncblas_caxpy_noomp (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{
	register unsigned int i = 0 ;
  	register unsigned int j = 0 ;
  	float *alphap = (float *)alpha;
 	float *x = (float *)X;
 	float *y = (float *)Y;

  	for (i=0 ; i < 2*N; i += 4){

			y[i] = (alphap[0]*x[i]-alphap[1]*x[i+1]) + y[i];
	    y[i+1] = (alphap[0]*x[i+1]+alphap[1]*x[i]) + y[i+1];
			y[i+2] = (alphap[0]*x[i+2]-alphap[1]*x[i+3]) + y[i+2];
			y[i+3] = (alphap[0]*x[i+3]+alphap[1]*x[i+2]) + y[i+3];
}
}

void mncblas_caxpy_omp (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{
	register unsigned int i = 0 ;
  	register unsigned int j = 0 ;
  	float *alphap = (float *)alpha;
 	float *x = (float *)X;
 	float *y = (float *)Y;

 	#pragma omp parallel for schedule (static)
  	for (i=0 ; i < 2*N; i += 4){
			y[i] = (alphap[0]*x[i]-alphap[1]*x[i+1]) + y[i];
	    y[i+1] = (alphap[0]*x[i+1]+alphap[1]*x[i]) + y[i+1];
			y[i+2] = (alphap[0]*x[i+2]-alphap[1]*x[i+3]) + y[i+2];
			y[i+3] = (alphap[0]*x[i+3]+alphap[1]*x[i+2]) + y[i+3];
  	}
}

void mncblas_caxpy_vec (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{

	float *a = (float *) alpha;
	float *u = (float *) X;
	float *v = (float *) Y;

	register unsigned int i ;
	register unsigned int j ;

	float4 alphaA ;
	float4 alphaB ;

	__m128 x1, x2, y1, y2 , x3, x4, x5;
	__m128 alpha1;
	__m128 alpha2;

	alphaA [0] = a[0];
	alphaA [1] = a[0];
	alphaA [2] = a[0];
	alphaA [3] = a[0];

	alphaB [0] = a[1];
	alphaB [1] = a[1];
	alphaB [2] = a[1];
	alphaB [3] = a[1];

	float w[2*N];
	for (int i = 0; i < 2*N ; i=i+2){
		w[i] = -u[i+1];
		w[i+1] = u[i];
	}

	alpha1 = _mm_load_ps (alphaA) ;
	alpha2 = _mm_load_ps (alphaB) ;

	#pragma omp parallel for schedule (static)
	for (i = 0; i<2*N; i += 4)
	{
		x1 = _mm_load_ps (w+(i*incX)) ;
		x2 = _mm_load_ps (u+(i*incX)) ;
		y1 = _mm_load_ps (v+(i*incY)) ;
		x3 = _mm_mul_ps (x1, alpha1) ;
		x4  = _mm_mul_ps(x2, alpha2) ;
		x5 = _mm_add_ps(x3, x4);
		y2 = _mm_add_ps (y1, x5) ;
		_mm_store_ps (v+(i*incY), y2) ;
	}
}

/*************************** ZAXPY **************************/

void mncblas_zaxpy_noomp (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{
	register unsigned int i = 0 ;
  	register unsigned int j = 0 ;
  	double *alphap = (double *)alpha;
 	double *x = (double *)X;
 	double *y = (double *)Y;

  	for (i=0 ; i < 2*N; i += 4){
			y[i] = (alphap[0]*x[i]-alphap[1]*x[i+1]) + y[i];
	    y[i+1] = (alphap[0]*x[i+1]+alphap[1]*x[i]) + y[i+1];
			y[i+2] = (alphap[0]*x[i+2]-alphap[1]*x[i+3]) + y[i+2];
			y[i+3] = (alphap[0]*x[i+3]+alphap[1]*x[i+2]) + y[i+3];
  	}
}

void mncblas_zaxpy_omp (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{
	register unsigned int i = 0 ;
  	register unsigned int j = 0 ;
  	double *alphap = (double *)alpha;
 	double *x = (double *)X;
 	double *y = (double *)Y;

	#pragma omp parallel for schedule (static)
  	for (i=0 ; i < 2*N; i += 4){
			y[i] = (alphap[0]*x[i]-alphap[1]*x[i+1]) + y[i];
	    y[i+1] = (alphap[0]*x[i+1]+alphap[1]*x[i]) + y[i+1];
			y[i+2] = (alphap[0]*x[i+2]-alphap[1]*x[i+3]) + y[i+2];
			y[i+3] = (alphap[0]*x[i+3]+alphap[1]*x[i+2]) + y[i+3];
  	}
}

void mncblas_zaxpy_vec (const int N, const void *alpha, const void *X, const int incX,
 void *Y, const int incY)
{

	double *a = (double *) alpha;
	double *u = (double *) X;
	double *v = (double *) Y;

	register unsigned int i ;
	register unsigned int j ;

	double2 alpha2 ;

	__m128d x1, x2, y1, y2 ;
	__m128d alpha1;

	alpha2 [0] = *a;
	alpha2 [1] = *a;
	// alpha4 [2] = alpha;
	// alpha4 [3] = alpha;

	alpha1 = _mm_load_pd (alpha2) ;

	#pragma omp parallel for schedule (static)
	for (i = 0; i<N; i += 4)
	{
		x1 = _mm_load_pd (u+(i*incX)) ;
		y1 = _mm_load_pd (v+(i*incY)) ;
		x2 = _mm_mul_pd (x1, alpha1) ;
		y2 = _mm_add_pd (y1, x2) ;
		_mm_store_pd (v+(i*incY), y2) ;
	}
}
