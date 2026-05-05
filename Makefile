# mnist_cuda_cnn_v2/Makefile
# Simple CUDA build. No CMake required.

NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
LDFLAGS ?=

TARGET := build/mnist_cuda_cnn
SRC := src/main.cu

.PHONY: all clean run data debug

all: $(TARGET)

$(TARGET): $(SRC) | build
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

build:
	mkdir -p build

run: $(TARGET)
	./$(TARGET)

data:
	bash scripts/download_mnist.sh

debug: NVCCFLAGS := -O0 -g -G -std=c++17

debug: clean $(TARGET)

clean:
	rm -rf build
