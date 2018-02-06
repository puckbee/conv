all: cnn_test.cpp mat.cpp mat.h
#	icpc -O3 cnn_test.cpp mat.cpp -o cnn.test -mkl -qopenmp
	g++ -O3 cnn_test.cpp mat.cpp -o cnn.test -lopenblas
debug:
	gcc -O3 cnn_test.cpp -o cnn.test.g -g
clean:
	rm cnn.test
