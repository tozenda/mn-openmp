#include <stdio.h>
#include <string.h>

#include <cblas.h>

/*
  Mesure des cycles
*/

#include <x86intrin.h>

#define NBEXPERIMENTS    256
static long long unsigned int experiments [NBEXPERIMENTS] ;

#define MSIZE    64

typedef float mfloat  [MSIZE] [MSIZE] ;

typedef double mdouble [MSIZE] [MSIZE] ;

mfloat A, B, C ;

long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
    {
      s = s + exps [i] ;
    }

  return s / (NBEXPERIMENTS-2) ;
}

void matrix_float_init (mfloat M, float x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < MSIZE; i++)
      for (j = 0; j < MSIZE; j++)
	if (i > j)
	  M [i][j] = x ;
	else
	  M [i][j] = -x ;

  return ;
}

void matrix_double_init (mdouble M, double x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < MSIZE; i++)
      for (j = 0; j < MSIZE; j++)
	M [i][j] = x ;

  return ;
}

void matrix_float_print (mfloat M)
{
  register unsigned int i ;
  register unsigned int j ;
  

  for (i = 0; i < MSIZE; i++)
    {
      for (j = 0 ; j < MSIZE; j++)
	printf ("%3.2f ", M[i][j]) ;
      printf ("\n") ;
    }
  printf ("\n") ;
  
  return ;
}

void matrix_double_print (mdouble M)
{
  register unsigned int i ;
  register unsigned int j ;
  

  for (i = 0; i < MSIZE; i++)
    {
      for (j = 0 ; j < MSIZE; j++)
	printf ("%3.2f ", M[i][j]) ;
      printf ("\n") ;
    }
  printf ("\n") ;
  
  return ;
}


int main (int argc, char **argv)
{
 unsigned long long start, end ;
 unsigned long long residu ;
 unsigned long long int av ;
 
 int exp ;

 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;  
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      matrix_float_init (A, 1.0) ;
      matrix_float_init (B, 2.0) ;
      matrix_float_init (C, 3.0) ;

      start = _rdtsc () ;
	  
         cblas_sgemm  (
		       CblasRowMajor, CblasNoTrans,  CblasNoTrans,
		       MSIZE, MSIZE, MSIZE, 1.0, (float *) A, MSIZE,
		       (float *) B, MSIZE, 1.0, (float *) C, MSIZE
		      ) ; 
      end = _rdtsc () ;
      experiments [exp] = end - start ;	  
    }
  av = average (experiments) ;

  printf ("cblas_sgemm: nombre de cycles: \t\t\t\t %Ld \n", av-residu) ;
  // matrix_float_print (C) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      matrix_float_init (A, 1.0) ;
      matrix_float_init (B, 2.0) ;
      matrix_float_init (C, 3.0) ;

      start = _rdtsc () ;
	  
         cblas_sgemm  (
		       CblasRowMajor, CblasNoTrans,  CblasNoTrans,
		       MSIZE, MSIZE, MSIZE, 1.0, (float *) A, MSIZE,
		       (float *) B, MSIZE, 1.0, (float *) C, MSIZE
		      ) ; 
      end = _rdtsc () ;
      experiments [exp] = end - start ;	  
    }
  av = average (experiments) ;

  printf ("cblas_sgemm TranB: nombre de cycles: \t\t\t %Ld \n", av-residu) ;
  // matrix_float_print (C) ;
  
  return 0 ;

}
