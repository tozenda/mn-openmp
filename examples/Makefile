SWAP = mntest_sswap mntest_dswap mntest_cswap mntest_zswap
COPY = mntest_scopy mntest_dcopy mntest_ccopy mntest_zcopy
DOT = mntest_sdot mntest_ddot mntest_cdotc_sub mntest_cdotu_sub mntest_zdotc_sub mntest_zdotu_sub
SWAP = mntest_sswap mntest_dswap mntest_cswap mntest_zswap
AXPY = mntest_saxpy mntest_daxpy mntest_caxpy mntest_zaxpy
GEMV = mntest_sgemv mntest_dgemv mntest_cgemv mntest_zgemv
GEMM = mntest_sgemm mntest_dgemm mntest_cgemm mntest_zgemm

 #BINARIES= $(SWAP) $(COPY) $(DOT) $(AXPY) $(GEMV) $(GEMM)
 BINARIES = $(DOT)

all: $(BINARIES) clean_o

INC_DIR=../include

LIB_DIR=../lib

LIBST=-lmnblas -lgsl -lgslcblas
LIBDYN=-lmnblasdyn -lgsl -lgslcblas

OPTIONS_COMPIL  =-O2 -fPIC -fopenmp -I$(INC_DIR)

OPTIONS_LINK_STATIC  =   -L$(LIB_DIR) $(LIBST) -fopenmp
OPTIONS_LINK_DYN  = -L$(LIB_DIR) $(LIBDYN) -fopenmp

############################## FABRIQUATION DES EXECUTABLES ##############################


#################### COPY ######################

mntest_scopy: mntest_scopy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_scopy mntest_scopy.o $(OPTIONS_LINK_STATIC)

mntest_dcopy: mntest_dcopy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dcopy mntest_dcopy.o $(OPTIONS_LINK_STATIC)

mntest_ccopy : mntest_ccopy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_ccopy mntest_ccopy.o $(OPTIONS_LINK_STATIC)

mntest_zcopy : mntest_zcopy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zcopy mntest_zcopy.o $(OPTIONS_LINK_STATIC)


#################### SWAP ######################

mntest_sswap: mntest_sswap.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sswap mntest_sswap.o $(OPTIONS_LINK_STATIC)

mntest_dswap : mntest_dswap.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dswap mntest_dswap.o $(OPTIONS_LINK_STATIC)

mntest_cswap : mntest_cswap.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_cswap mntest_cswap.o $(OPTIONS_LINK_STATIC)

mntest_zswap : mntest_zswap.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zswap mntest_zswap.o $(OPTIONS_LINK_STATIC)


#################### DOT ######################

mntest_sdot: mntest_sdot.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sdot mntest_sdot.o $(OPTIONS_LINK_STATIC)

mntest_ddot: mntest_ddot.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_ddot mntest_ddot.o $(OPTIONS_LINK_STATIC)

mntest_cdotc_sub: mntest_cdotc_sub.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_cdotc_sub mntest_cdotc_sub.o $(OPTIONS_LINK_STATIC)

mntest_cdotu_sub: mntest_cdotu_sub.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_cdotu_sub mntest_cdotu_sub.o $(OPTIONS_LINK_STATIC)

mntest_zdotc_sub: mntest_zdotc_sub.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zdotc_sub mntest_zdotc_sub.o $(OPTIONS_LINK_STATIC)

mntest_zdotu_sub: mntest_zdotu_sub.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zdotu_sub mntest_zdotu_sub.o $(OPTIONS_LINK_STATIC)


#################### AXPY ######################

mntest_saxpy: mntest_saxpy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_saxpy mntest_saxpy.o $(OPTIONS_LINK_STATIC)

mntest_daxpy: mntest_daxpy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_daxpy mntest_daxpy.o $(OPTIONS_LINK_STATIC)

mntest_caxpy: mntest_caxpy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_caxpy mntest_caxpy.o $(OPTIONS_LINK_STATIC)

mntest_zaxpy: mntest_zaxpy.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zaxpy mntest_zaxpy.o $(OPTIONS_LINK_STATIC)


#################### GEMV ######################

mntest_sgemv: mntest_sgemv.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sgemv mntest_sgemv.o $(OPTIONS_LINK_STATIC)

mntest_dgemv : mntest_dgemv.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dgemv mntest_dgemv.o $(OPTIONS_LINK_STATIC)

mntest_cgemv: mntest_cgemv.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_cgemv mntest_cgemv.o $(OPTIONS_LINK_STATIC)

mntest_zgemv : mntest_zgemv.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zgemv mntest_zgemv.o $(OPTIONS_LINK_STATIC)


#################### GEMM ######################

mntest_sgemm : mntest_sgemm.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_sgemm mntest_sgemm.o $(OPTIONS_LINK_STATIC)

mntest_dgemm : mntest_dgemm.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_dgemm mntest_dgemm.o $(OPTIONS_LINK_STATIC)

mntest_cgemm : mntest_cgemm.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_cgemm mntest_cgemm.o $(OPTIONS_LINK_STATIC)

mntest_zgemm : mntest_zgemm.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest_zgemm mntest_zgemm.o $(OPTIONS_LINK_STATIC)



############################## FABRIQUATION DES FICHIERS OBJETS ##############################


#################### COPY ######################

mntest_scopy.o: mntest_scopy.c
	gcc $(OPTIONS_COMPIL) -c mntest_scopy.c

mntest_dcopy.o: mntest_dcopy.c
	gcc $(OPTIONS_COMPIL) -c mntest_dcopy.c

mntest_ccopy.o: mntest_ccopy.c
	gcc $(OPTIONS_COMPIL) -c mntest_ccopy.c

mntest_zcopy.o: mntest_zcopy.c
	gcc $(OPTIONS_COMPIL) -c mntest_zcopy.c


#################### SWAP ######################

mntest_sswap.o: mntest_sswap.c
	gcc $(OPTIONS_COMPIL) -c mntest_sswap.c

mntest_dswap.o: mntest_dswap.c
	gcc $(OPTIONS_COMPIL) -c mntest_dswap.c

mntest_cswap.o: mntest_cswap.c
	gcc $(OPTIONS_COMPIL) -c mntest_cswap.c

mntest_zswap.o: mntest_zswap.c
	gcc $(OPTIONS_COMPIL) -c mntest_zswap.c


#################### DOT ######################

mntest_sdot.o: mntest_sdot.c
	gcc $(OPTIONS_COMPIL) -c mntest_sdot.c

mntest_ddot.o: mntest_ddot.c
	gcc $(OPTIONS_COMPIL) -c mntest_ddot.c

mntest_cdotc_sub.o: mntest_cdotc_sub.c
	gcc $(OPTIONS_COMPIL) -c mntest_cdotc_sub.c

mntest_cdotu_sub.o: mntest_cdotu_sub.c
	gcc $(OPTIONS_COMPIL) -c mntest_cdotu_sub.c

mntest_zdotc_sub.o: mntest_zdotc_sub.c
	gcc $(OPTIONS_COMPIL) -c mntest_zdotc_sub.c

mntest_zdotu_sub.o: mntest_zdotu_sub.c
	gcc $(OPTIONS_COMPIL) -c mntest_zdotu_sub.c


#################### AXPY ######################

mntest_saxpy.o: mntest_saxpy.c
	gcc $(OPTIONS_COMPIL) -c mntest_saxpy.c

mntest_daxpy.o: mntest_daxpy.c
	gcc $(OPTIONS_COMPIL) -c mntest_daxpy.c

mntest_caxpy.o: mntest_caxpy.c
	gcc $(OPTIONS_COMPIL) -c mntest_caxpy.c

mntest_zaxpy.o: mntest_zaxpy.c
	gcc $(OPTIONS_COMPIL) -c mntest_zaxpy.c


#################### GEMV ######################

mntest_sgemv.o: mntest_sgemv.c
	gcc $(OPTIONS_COMPIL) -c mntest_sgemv.c

mntest_dgemv.o: mntest_dgemv.c
	gcc $(OPTIONS_COMPIL) -c mntest_dgemv.c

mntest_cgemv.o: mntest_cgemv.c
	gcc $(OPTIONS_COMPIL) -c mntest_cgemv.c

mntest_zgemv.o: mntest_zgemv.c
	gcc $(OPTIONS_COMPIL) -c mntest_zgemv.c


#################### GEMM ######################

mntest_sgemm.o: mntest_sgemm.c
	gcc $(OPTIONS_COMPIL) -c mntest_sgemm.c

mntest_dgemm.o: mntest_dgemm.c
	gcc $(OPTIONS_COMPIL) -c mntest_dgemm.c

mntest_cgemm.o: mntest_cgemm.c
	gcc $(OPTIONS_COMPIL) -c mntest_cgemm.c

mntest_zgemm.o: mntest_zgemm.c
	gcc $(OPTIONS_COMPIL) -c mntest_zgemm.c

################### CLEAN ######################

clean_o:
	rm -f *.o

clean:
	rm -f *.o $(BINARIES) *~
