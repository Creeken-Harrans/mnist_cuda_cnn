NVCC ?= nvcc
BUILD_DIR := build
TARGET := $(BUILD_DIR)/mnist_cuda_cnn
SRC := src/main.cu

# You can override CUDAFLAGS, for example:
#   make CUDAFLAGS="-O3 -std=c++17 -arch=sm_86"
CUDAFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
