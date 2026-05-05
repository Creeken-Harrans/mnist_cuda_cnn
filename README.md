# MNIST CUDA CNN from Scratch

这是一个教学型但完整可跑的 C++/CUDA 项目：

- 自动下载 MNIST 数据集；
- 解析原始 IDX 格式；
- 不使用 PyTorch / libtorch / cuDNN；
- 手写 CUDA kernel 完成：
  - convolution forward / backward
  - ReLU forward / backward
  - 2x2 max pooling forward / backward
  - fully-connected forward / backward
  - softmax cross entropy
  - Adam optimizer
- 用一个小型 LeNet 风格 CNN 完成 MNIST 训练和分类。

默认网络：

```text
Input 1x28x28
 -> Conv(1 -> 8, 5x5, padding=2)
 -> ReLU
 -> MaxPool 2x2
 -> Conv(8 -> 16, 5x5, padding=2)
 -> ReLU
 -> MaxPool 2x2
 -> FC(16*7*7 -> 64)
 -> ReLU
 -> FC(64 -> 10)
 -> Softmax Cross Entropy
```

## 依赖

Linux 环境下需要：

```bash
nvcc
make
curl
gzip
```

检查 CUDA 编译器：

```bash
nvcc --version
```

## 编译

```bash
cd mnist_cuda_cnn
make
```

如果你的 GPU 架构比较新，可以手动指定架构，例如 RTX 30 系列常用 `sm_86`：

```bash
make CUDAFLAGS="-O3 -std=c++17 -lineinfo -arch=sm_86"
```

RTX 40 系列常用：

```bash
make CUDAFLAGS="-O3 -std=c++17 -lineinfo -arch=sm_89"
```

## 运行

```bash
./build/mnist_cuda_cnn
```

程序第一次运行会自动下载并解压 MNIST 到：

```text
data/MNIST/raw/
```

也可以手动下载：

```bash
./scripts/download_mnist.sh data/MNIST/raw
```

## 常用参数

```bash
./build/mnist_cuda_cnn --epochs 8 --batch 128 --lr 0.001
```

更多参数：

```bash
./build/mnist_cuda_cnn --help
```

## 准确率说明

默认 8 epochs 通常可以达到 MNIST 测试集约 98%+。由于这是教学型 from-scratch kernel，不像 cuDNN 那样做了高度优化，所以训练速度不会像 PyTorch/cuDNN 那么快。

如果你更重视准确率，可以增加 epoch：

```bash
./build/mnist_cuda_cnn --epochs 12 --batch 128 --lr 0.001
```

## 代码阅读顺序

建议按这个顺序看 `src/main.cu`：

1. `ensure_mnist_downloaded`：下载数据集；
2. `load_mnist_split`：解析 IDX 数据；
3. CUDA kernels：卷积、池化、全连接、softmax、Adam；
4. `forward`：前向传播；
5. `backward`：反向传播；
6. `adam_update`：参数更新；
7. `main`：训练循环和测试评估。
