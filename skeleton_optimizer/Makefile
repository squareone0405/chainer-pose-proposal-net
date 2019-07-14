PROJECT_NAME = optimizer

CC = g++

BUILD_DIR = build

CFLAGS = -std=c++11 -c -fPIC -I/usr/include

LDFLAGS= -L</usr/local/lib> -l<libceres.a>

all: build

build: build_dir make
	$(NVCC) $(LFLAGS) -shared -o $(BUILD_DIR)/ssd.so *.o
	$(NVCC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

build_dir:
	mkdir -p $(BUILD_DIR)

make:
	$(CC) $(CFLAGS) *.cpp

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)

