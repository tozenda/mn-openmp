BINARIES=  mntest_saxpy mntest_daxpy mntest_scopy mntest_dcopy mntest_sdot mntest_ddot

all: $(BINARIES)

INC_DIR=../include

LIB_DIR=../lib

LIBST=-lmnblas -lgsl -lgslcblas
LIBDYN=-lmnblasdyn -lgsl -lgslcblas

OPTIONS_COMPIL  =-O2 -fPIC -fopenmp -I$(INC_DIR)

OPTIONS_LINK_STATIC  =   -L$(LIB_DIR) $(LIBST) -fopenmp
OPTIONS_LINK_DYN  = -L$(LIB_DIR) $(LIBDYN) -fopenmp
OPTIONS_LINK_CBLAS = -lgsl -lgslcblas -fopenmp

cblastest3: cblastest3.o
	gcc -o cblastest3 cblastest3.o $(OPTIONS_LINK_CBLAS)

cblastest2: cblastest2.o
	gcc -o cblastest2 cblastest2.o $(OPTIONS_LINK_CBLAS)

cblastest1: cblastest1.o
	gcc -o cblastest1 cblastest1.o $(OPTIONS_LINK_CBLAS)

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

mntest_ddot: mntest_ddot.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_ddot mntest_ddot.o $(OPTIONS_LINK_STATIC)

#mntest1dyn: mntest1.o $(LIB_DIR)/libmnblasdyn.so
	#gcc -o mntest1dyn mntest1.o $(OPTIONS_LINK_DYN)

mntest2dyn: mntest2.o $(LIB_DIR)/libmnblasdyn.so
	gcc -o mntest2dyn mntest2.o $(OPTIONS_LINK_DYN)

mntest2: mntest2.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest2 mntest2.o $(OPTIONS_LINK_STATIC)

mntest3dyn: mntest3.o $(LIB_DIR)/libmnblasdyn.so
	gcc -o mntest3dyn mntest2.o $(OPTIONS_LINK_DYN)

cblastest1.o: cblastest1.c
	gcc $(OPTIONS_COMPIL) -c cblastest1.c

cblastest2.o: cblastest2.c
	gcc $(OPTIONS_COMPIL) -c cblastest2.c

cblastest3.o: cblastest3.c
	gcc $(OPTIONS_COMPIL) -c cblastest3.c

mntest3: mntest3.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest3 mntest3.o $(OPTIONS_LINK_STATIC)

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

mntest_ddot.o: mntest_ddot.c
	gcc $(OPTIONS_COMPIL) -c mntest_ddot.c


mntest2.o: mntest2.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c mntest2.c

mntest3.o: mntest3.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c mntest3.c

clean:
	rm -f *.o $(BINARIES) *~
