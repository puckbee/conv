all:  
#	icpc -mavx2 -mfma -Wall -L/opt/OpenBLAS/lib  -L/usr/local/include -o conv_winograd  test_sse_ncnn.cpp -lopenblas  -lpthread -qopenmp
	icpc -mavx2 -mfma -Wall -L/opt/OpenBLAS/lib  -L/usr/local/include -o conv_winograd  test_sse_ncnn.cpp -lopenblas  -lpthread -O3 -qopenmp
          
clean:  
	rm -rf *.o conv_winograd  
