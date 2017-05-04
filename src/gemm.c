#include <stdlib.h>
#include <stdio.h>

#include "mnblas.h"

#include <nmmintrin.h>

#include <omp.h>

typedef float *floatM;


typedef float float4 [4]  __attribute__ ((aligned (16))) ;
typedef double double4 [4] __attribute__ ((aligned (16))) ;

void mncblas_sgemm_1 (
	MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
	MNCBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const float alpha, const float *A,
	const int lda, const float *B, const int ldb,
	const float beta, float *C, const int ldc
)
{
	/*
	vectorized implementation
	*/

	register unsigned int i ;
	register unsigned int j ;
	register unsigned int k ;
	register unsigned int l ;

	register unsigned int indice_ligne ;

	register float r ;
	floatM   Bcol ;
	float4   R4 ;
	int      err ;

	__m128 av4 ;
	__m128 bv4 ;
	__m128 dot ;

	/*
	Bcol = (floatM) malloc (M * sizeof (float)) ;
	err = posix_memalign ((void **) &Bcol, 16, M*sizeof(float)) ;
	*/

	if (TransB == MNCblasNoTrans)
	{
		Bcol = aligned_alloc (16, M * sizeof (float)) ;
		#pragma omp parallel for schedule(static)
		for (i = 0 ; i < M; i = i + 1)
		{
			#pragma omp parallel for schedule(static) private(av4, bv4, dot)
			for (j = 0 ; j < M; j ++)
			{


				/*
				load a B column (j)
				*/
				// #pragma omp parallel for schedule(static)
				for (l = 0 ; l < M ; l = l + 4)
				{
					Bcol [l]     = B [l        * M + j ] ;
					Bcol [l + 1] = B [(l + 1)  * M + j ] ;
					Bcol [l + 2] = B [(l + 2)  * M + j ] ;
					Bcol [l + 3] = B [(l + 3)  * M + j ] ;
				}

				r = 0.0 ;
				indice_ligne = i * M ;
				// #pragma omp parallel for schedule(static)
				for (k = 0; k < M; k = k + 4)
				{

					av4 = _mm_load_ps (A+indice_ligne + k);
					bv4 = _mm_load_ps (Bcol+ k) ;

					dot = _mm_dp_ps (av4, bv4, 0xFF) ;

					_mm_store_ps (R4, dot) ;

					r = r + R4 [0] ;
				}

				C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
			}
		}
	}
	else
		// on fait 5/4 * M**3 opérations
	{
		#pragma omp parallel for schedule(static)
		for (i = 0 ; i < M; i = i + 1)
		{
			#pragma omp parallel for schedule(static) private(av4, bv4, dot)
			for (j = 0 ; j < M; j ++)
			{

				r = 0.0 ;
				indice_ligne = i * M ;
				// #pragma omp parallel for schedule(static)
				for (k = 0; k < M; k = k + 4)
				{

					av4 = _mm_load_ps (A + indice_ligne + k);
					bv4 = _mm_load_ps (B + indice_ligne + k) ;

					dot = _mm_dp_ps (av4, bv4, 0xFF) ;

					_mm_store_ps (R4, dot) ;

					r = r + R4 [0] ;
				}

				C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
			}
		}
	}
	return ;
}


void mncblas_sgemm_noomp (
	MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
	MNCBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const float alpha, const float *A,
	const int lda, const float *B, const int ldb,
	const float beta, float *C, const int ldc
)
{
	/*
	scalar implementation
	*/

	register unsigned int i ;
	register unsigned int j ;
	register unsigned int k ;
	register float r ;
	for (i = 0 ; i < M; i = i + 4)
	{
		/* i */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k=k+4)
			{
				r = r + A [(i * M) + k    ] * B [(k * M)       + j] ;
				r = r + A [(i * M) + k + 1] * B [(k + 1) * M   + j] ;
				r = r + A [(i * M) + k + 2] * B [(k + 2) * M   + j] ;
				r = r + A [(i * M) + k + 3] * B [(k + 3) * M   + j] ;
			}
			C [(i*M) + j] = (alpha * r) + (beta * C [(i*M) + j]) ;
		}

		/* i + 1 */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k=k+4)
			{
		 		r = r + A [((i + 1) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 1) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 1) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 1) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 1) * M) + j] = (alpha * r) + (beta * C [((i + 1) * M) + j]) ;
		}

		/* i + 2 */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 2) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 2) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 2) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 2) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 2) * M) + j] = (alpha * r) + (beta * C [((i + 2) * M) + j]) ;
		}

		/* i + 3 */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 3) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 3) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 3) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 3) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 3) * M) + j] = (alpha * r) + (beta * C [((i + 3) * M) + j]) ;
		}

	}
	return ;
}

void mncblas_sgemm_omp (
	MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
	MNCBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const float alpha, const float *A,
	const int lda, const float *B, const int ldb,
	const float beta, float *C, const int ldc
)
{
	/*
	scalar implementation
	*/

	register unsigned int i ;
	register unsigned int j ;
	register unsigned int k ;
	register float r ;
	#pragma omp parallel for schedule(static) private(r)
	for (i = 0 ; i < M; i = i + 4)
	{
		/* i */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			//#pragma omp parallel for schedule(static)
			for (k = 0; k < M; k=k+4)
			{
				r = r + A [(i * M) + k    ] * B [(k * M)       + j] ;
				r = r + A [(i * M) + k + 1] * B [(k + 1) * M   + j] ;
				r = r + A [(i * M) + k + 2] * B [(k + 2) * M   + j] ;
				r = r + A [(i * M) + k + 3] * B [(k + 3) * M   + j] ;
			}
			C [(i*M) + j] = (alpha * r) + (beta * C [(i*M) + j]) ;
		}

		/* i + 1 */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k=k+4)
			{
		 		r = r + A [((i + 1) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 1) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 1) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 1) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 1) * M) + j] = (alpha * r) + (beta * C [((i + 1) * M) + j]) ;
		}

		/* i + 2 */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 2) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 2) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 2) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 2) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 2) * M) + j] = (alpha * r) + (beta * C [((i + 2) * M) + j]) ;
		}

		/* i + 3 */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 3) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 3) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 3) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 3) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 3) * M) + j] = (alpha * r) + (beta * C [((i + 3) * M) + j]) ;
		}

	}
	return ;
}

void mncblas_dgemm_1 (
	MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
	MNCBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const double alpha, const double *A,
	const int lda, const double *B, const int ldb,
	const double beta, double *C, const int ldc
)
{
	/*
	vectorized implementation
	*/

	register unsigned int i ;
	register unsigned int j ;
	register unsigned int k ;
	register unsigned int l ;

	register unsigned int indice_ligne ;

	register double r ;
	double *Bcol ;
	double4 *R4 ;
	int      err ;

	__m128d av4 ;
	__m128d bv4 ;
	__m128d dot ;

	/*
	Bcol = (floatM) malloc (M * sizeof (double)) ;
	err = posix_memalign ((void **) &Bcol, 16, M*sizeof(double)) ;
	*/
	printf("variable init ok\n");
	if (TransB == MNCblasNoTrans)
	{
		printf("On rentre dans if1\n");
		Bcol = aligned_alloc (16, M * sizeof (double)) ;

		// #pragma omp parallel for schedule(static)
		for (i = 0 ; i < M; i = i + 1)
		{
			printf("On rentre dans for1 itération %d\n", i);

			// #pragma omp parallel for schedule(static) private(av4, bv4, dot)
			for (j = 0 ; j < M; j ++)
			{
				printf("on rentre dans for 2 itération %d\n", j);


				/*
				load a B column (j)
				*/

				for (l = 0 ; l < M ; l = l + 4){
				Bcol [l]     = B [l        * M + j ] ;
				Bcol [l + 1] = B [(l + 1)  * M + j ] ;
				Bcol [l + 2] = B [(l + 2)  * M + j ] ;
				Bcol [l + 3] = B [(l + 3)  * M + j ] ;
				}
				printf("ok A\n");
				r = 0.0 ;
				indice_ligne = i * M ;

				for (k = 0; k < M; k = k + 2)
				{
					printf("On rentre dans for3 it %d\n", k);

					av4 = _mm_load_pd (A+indice_ligne + k);
					printf("point a-1\n");
					bv4 = _mm_load_pd (Bcol+ k) ;
					printf("point a-2\n");

					dot = _mm_dp_pd (av4, bv4, 0xFF) ;
					printf("point a-3\n");

					_mm_store_pd (R4, dot) ;
					printf("point a-4\n");

					r = r + R4 [0] ;
					printf("point a-5\n");
				}
				printf("ok B\n");
				C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
			}
		}
	}
	else
	{
		printf("On rentre dans else\n");
		#pragma omp parallel for schedule(static)
		for (i = 0 ; i < M; i = i + 1)
		{
			#pragma omp parallel for schedule(static) private(av4, bv4, dot)
			for (j = 0 ; j < M; j ++)
			{

				r = 0.0 ;
				indice_ligne = i * M ;

				for (k = 0; k < M; k = k + 2)
				{

					av4 = _mm_load_pd (A + indice_ligne + k);
					bv4 = _mm_load_pd (B + indice_ligne + k) ;

					dot = _mm_dp_pd (av4, bv4, 0xFF) ;

					_mm_store_pd (R4, dot) ;

					r = r + R4 [0] ;
				}

				C [indice_ligne + j] = (alpha * r) + (beta * C [indice_ligne + j]) ;
			}
		}
	}
	return ;
}

void mncblas_dgemm_noomp(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
	MNCBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const double alpha, const double *A,
	const int lda, const double *B, const int ldb,
	const double beta, double *C, const int ldc)
	{
	/*
	scalar implementation
	*/

	register unsigned int i ;
	register unsigned int j ;
	register unsigned int k ;
	register double r ;
	for (i = 0 ; i < M; i = i + 4)
	{
		/* i */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k=k+4)
			{
				r = r + A [(i * M) + k    ] * B [(k * M)       + j] ;
				r = r + A [(i * M) + k + 1] * B [(k + 1) * M   + j] ;
				r = r + A [(i * M) + k + 2] * B [(k + 2) * M   + j] ;
				r = r + A [(i * M) + k + 3] * B [(k + 3) * M   + j] ;
			}
			C [(i*M) + j] = (alpha * r) + (beta * C [(i*M) + j]) ;
		}

		/* i + 1 */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k=k+4)
			{
		 		r = r + A [((i + 1) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 1) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 1) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 1) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 1) * M) + j] = (alpha * r) + (beta * C [((i + 1) * M) + j]) ;
		}

		/* i + 2 */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 2) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 2) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 2) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 2) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 2) * M) + j] = (alpha * r) + (beta * C [((i + 2) * M) + j]) ;
		}

		/* i + 3 */
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 3) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 3) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 3) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 3) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 3) * M) + j] = (alpha * r) + (beta * C [((i + 3) * M) + j]) ;
		}

	}
	return ;
}

void mncblas_dgemm_omp(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
	MNCBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const double alpha, const double *A,
	const int lda, const double *B, const int ldb,
	const double beta, double *C, const int ldc)
	{
	/*
	scalar implementation
	*/

	register unsigned int i ;
	register unsigned int j ;
	register unsigned int k ;
	register double r ;
	#pragma omp parallel for schedule(static) private(r)
	for (i = 0 ; i < M; i = i + 4)
	{
		/* i */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k=k+4)
			{
				r = r + A [(i * M) + k    ] * B [(k * M)       + j] ;
				r = r + A [(i * M) + k + 1] * B [(k + 1) * M   + j] ;
				r = r + A [(i * M) + k + 2] * B [(k + 2) * M   + j] ;
				r = r + A [(i * M) + k + 3] * B [(k + 3) * M   + j] ;
			}
			C [(i*M) + j] = (alpha * r) + (beta * C [(i*M) + j]) ;
		}

		/* i + 1 */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k=k+4)
			{
		 		r = r + A [((i + 1) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 1) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 1) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 1) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 1) * M) + j] = (alpha * r) + (beta * C [((i + 1) * M) + j]) ;
		}

		/* i + 2 */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 2) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 2) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 2) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 2) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 2) * M) + j] = (alpha * r) + (beta * C [((i + 2) * M) + j]) ;
		}

		/* i + 3 */
		#pragma omp parallel for schedule(static)
		for (j = 0 ; j < M; j ++)
		{
			r = 0.0 ;
			// #pragma omp parallel for schedule(static)
			for (k = 0; k < M; k = k + 4)
			{
				r = r + A [((i + 3) * M) + k    ] * B [(k * M)     + j] ;
				r = r + A [((i + 3) * M) + k + 1] * B [(k + 1) * M + j] ;
				r = r + A [((i + 3) * M) + k + 2] * B [(k + 2) * M + j] ;
				r = r + A [((i + 3) * M) + k + 3] * B [(k + 3) * M + j] ;
			}
			C [((i + 3) * M) + j] = (alpha * r) + (beta * C [((i + 3) * M) + j]) ;
		}

	}
	return ;
}

	void mncblas_cgemm (
		MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		const int K, const void *alpha, const void *A,
		const int lda, const void *B, const int ldb,
		const void *beta, void *C, const int ldc
	)
	{

		return ;
	}

	void mncblas_zgemm (
		MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
		MNCBLAS_TRANSPOSE TransB, const int M, const int N,
		const int K, const void *alpha, const void *A,
		const int lda, const void *B, const int ldb,
		const void *beta, void *C, const int ldc
	)
	{

		return ;
	}
