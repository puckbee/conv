all: cnn_test.cpp mat.cpp mat.h
#	icpc -O3 cnn_test.cpp mat.cpp -o cnn.test -mkl -qopenmp
	g++ -O3 cnn_test.cpp mat.cpp -o cnn.test -lopenblas

wino: winograd.cpp
	g++ winograd.cpp -o wino -O3
debug:
	gcc -O3 cnn_test.cpp -o cnn.test.g -g
clean:
	rm cnn.test
