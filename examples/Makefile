BINARIES= mntest_saxpy mntest_daxpy mntest_scopy mntest_dcopy mntest_sdot mntest_ddot mntest_sgemm mntest_dgemm mntest_sswap mntest_dswap mntest_sgemv clean_o

all: $(BINARIES)

INC_DIR=../include

LIB_DIR=../lib

LIBST=-lmnblas -lgsl -lgslcblas
LIBDYN=-lmnblasdyn -lgsl -lgslcblas

OPTIONS_COMPIL  =-O2 -fPIC -fopenmp -I$(INC_DIR)

OPTIONS_LINK_STATIC  =   -L$(LIB_DIR) $(LIBST) -fopenmp
OPTIONS_LINK_DYN  = -L$(LIB_DIR) $(LIBDYN) -fopenmp
OPTIONS_LINK_CBLAS = -lgsl -lgslcblas -fopenmp

mntest_saxpy: mntest_saxpy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_saxpy mntest_saxpy.o $(OPTIONS_LINK_STATIC)

mntest_daxpy: mntest_daxpy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_daxpy mntest_daxpy.o $(OPTIONS_LINK_STATIC)

mntest_scopy: mntest_scopy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_scopy mntest_scopy.o $(OPTIONS_LINK_STATIC)

mntest_dcopy: mntest_dcopy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dcopy mntest_dcopy.o $(OPTIONS_LINK_STATIC)

mntest_sdot: mntest_sdot.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sdot mntest_sdot.o $(OPTIONS_LINK_STATIC)

mntest_sgemv: mntest_sgemv.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sgemv mntest_sgemv.o $(OPTIONS_LINK_STATIC)

mntest_sswap: mntest_sswap.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sswap mntest_sswap.o $(OPTIONS_LINK_STATIC)

mntest_ddot: mntest_ddot.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_ddot mntest_ddot.o $(OPTIONS_LINK_STATIC)

mntest_sgemm : mntest_sgemm.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sgemm mntest_sgemm.o $(OPTIONS_LINK_STATIC)

mntest_dgemm : mntest_dgemm.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dgemm mntest_dgemm.o $(OPTIONS_LINK_STATIC)

mntest_dswap : mntest_dswap.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dswap mntest_dswap.o $(OPTIONS_LINK_STATIC)

#mntest1dyn: mntest1.o $(LIB_DIR)/libmnblasdyn.so
	#gcc -o mntest1dyn mntest1.o $(OPTIONS_LINK_DYN)

mntest_saxpy.o: mntest_saxpy.c
	gcc $(OPTIONS_COMPIL) -c mntest_saxpy.c

mntest_daxpy.o: mntest_daxpy.c
	gcc $(OPTIONS_COMPIL) -c mntest_daxpy.c

mntest_scopy.o: mntest_scopy.c
	gcc $(OPTIONS_COMPIL) -c mntest_scopy.c

mntest_dcopy.o: mntest_dcopy.c
	gcc $(OPTIONS_COMPIL) -c mntest_dcopy.c

mntest_sdot.o: mntest_sdot.c
	gcc $(OPTIONS_COMPIL) -c mntest_sdot.c

mntest_sswap.o: mntest_sswap.c
	gcc $(OPTIONS_COMPIL) -c mntest_sswap.c

mntest_sgemm.o: mntest_sgemm.c
	gcc $(OPTIONS_COMPIL) -c mntest_sgemm.c

mntest_sgemv.o: mntest_sgemv.c
	gcc $(OPTIONS_COMPIL) -c mntest_sgemv.c

mntest_ddot.o: mntest_ddot.c
	gcc $(OPTIONS_COMPIL) -c mntest_ddot.c

mntest_dgemm.o: mntest_dgemm.c
	gcc $(OPTIONS_COMPIL) -c mntest_dgemm.c

mntest_dswap.o: mntest_dswap.c
	gcc $(OPTIONS_COMPIL) -c mntest_dswap.c

clean_o:
	rm -f *.o

clean:
	rm -f *.o $(BINARIES) *~
