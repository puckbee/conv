


OPT_FLAG= -mkl -qopenmp -D_MKL_ -O3 -march=core-avx2
#OPT_FLAG= -mkl -qopenmp -D_MKL_ -g -O3 -march=core-avx2

# OPT_FLAG= -lopenblas -D_OPENBLAS_ -mavx2 -mfma -qopenmp

all: cnn_test.cpp mat.o cpu.o bench_formats.o basic_operations.o ncnn_wino_3x3.o
#	g++ -O2 cnn_test.cpp mat.cpp -o cnn.test -lopenblas -D_OPENBLAS_ -fopenmp
#	icpc -O3 cnn_test.cpp mat.cpp -o cnn.test -lopenblas -D_OPENBLAS_ -qopenmp
	icpc $(OPT_FLAG) cnn_test.cpp mat.o cpu.o bench_formats.o basic_operations.o ncnn_wino_3x3.o -o conv -qopt-report=5

mat.o: cpu.o
	icpc $(OPT_FLAG) utils/mat.cpp -c

ncnn_wino_3x3.o: mat.o
	icpc $(OPT_FLAG) bench/ncnn_wino_3x3.cpp -c

basic_operations.o: mat.o
	icpc $(OPT_FLAG) utils/basic_operations.cpp -c

bench_formats.o: mat.o
	icpc $(OPT_FLAG) utils/bench_formats.cpp mat.o -c -qopenmp 

cpu.o: 
	icpc $(OPT_FLAG) utils/cpu.cpp -c

wino: winograd.cpp
	icpc winograd.cpp -o wino -O3

debug:
	icpc -O3 cnn_test.cpp mat.cpp -o cnn.test.g -g -lopenblas
clean:
	rm cnn.test *.o conv
