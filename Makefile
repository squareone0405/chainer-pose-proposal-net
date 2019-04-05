PROJECT_NAME = ssd

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = g++

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
# note that nvcc defaults to 32-bit architecture. thus, force C/LFLAGS to comply.
# you could also force nvcc to compile 64-bit with -m64 flag. (and remove -m32 instances)

CFLAGS = -std=c++11 -c -fPIC -I$(CUDAPATH)/include
NVCCFLAGS = -c -Xcompiler -fPIC -I$(CUDAPATH)/include

LFLAGS = -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: build clean

build: build_dir gpu
	$(NVCC) $(LFLAGS) -shared -o $(BUILD_DIR)/ssd.so *.o
	$(NVCC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)

