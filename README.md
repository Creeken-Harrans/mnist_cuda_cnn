# mnist_cuda_cnn_v2

一个尽量简洁、但写法更规范的 **from-scratch C++/CUDA MNIST CNN trainer**。

这个项目的目标不是复刻 PyTorch，也不是追求 cuDNN 级别性能，而是把一个卷积神经网络训练系统的核心链路完整展开：

```text
MNIST 下载
  -> IDX 文件解析
  -> 数据拷贝到 GPU
  -> CNN forward
  -> softmax cross entropy
  -> CNN backward
  -> Adam 参数更新
  -> test set evaluation
```

项目刻意不使用：

```text
PyTorch
libtorch
cuDNN
cuBLAS
Thrust
OpenCV
CMake
```

核心计算全部在 `src/main.cu` 中用 CUDA kernel 手写，方便直接阅读和修改。

---

## 1. 项目结构

```text
mnist_cuda_cnn_v2/
├── Makefile
├── README.md
├── scripts/
│   └── download_mnist.sh
└── src/
    └── main.cu
```

各文件作用：

```text
Makefile
  负责调用 nvcc 编译 CUDA/C++ 程序。

README.md
  项目说明、运行方式、参数说明、训练流程和调试建议。

scripts/download_mnist.sh
  单独下载并解压 MNIST 数据集的脚本。

src/main.cu
  主程序，包含：
  - MNIST 下载逻辑
  - MNIST IDX 解析
  - CUDA DeviceBuffer RAII 封装
  - tqdm-like 终端进度条
  - Conv / ReLU / MaxPool / FC / Softmax / Adam CUDA kernels
  - CNN 模型结构
  - training loop
  - evaluation loop
```

---

## 2. 环境要求

建议环境：

```text
Linux
NVIDIA GPU
CUDA Toolkit
nvcc
C++17-capable host compiler
curl
gzip
make
```

在 Arch Linux 上，通常需要类似这些工具：

```bash
sudo pacman -S cuda make gcc curl gzip
```

检查 CUDA 编译器：

```bash
nvcc --version
```

检查显卡：

```bash
nvidia-smi
```

如果 `nvcc` 不存在，说明 CUDA Toolkit 没有装好，或者 `nvcc` 不在 `PATH` 中。

---

## 3. 编译

在项目根目录执行：

```bash
make
```

编译成功后会生成：

```text
build/mnist_cuda_cnn
```

等价地，你可以直接看 Makefile 中的核心编译命令：

```bash
nvcc -O2 -std=c++17 src/main.cu -o build/mnist_cuda_cnn
```

清理构建产物：

```bash
make clean
```

Debug 编译：

```bash
make debug
```

Debug 模式会使用：

```text
-O0 -g -G -std=c++17
```

其中：

```text
-O0   关闭优化，方便调试
-g    生成 CPU 端 debug 信息
-G    生成 GPU device code debug 信息
```

---

## 4. 运行

默认运行：

```bash
./build/mnist_cuda_cnn
```

第一次运行时，程序会自动下载 MNIST 到：

```text
data/MNIST/raw
```

下载并解压后，目录里应该有这四个 IDX 文件：

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

也可以手动下载数据：

```bash
bash scripts/download_mnist.sh
```

然后再运行训练：

```bash
./build/mnist_cuda_cnn
```

---

## 5. 常用参数

查看帮助：

```bash
./build/mnist_cuda_cnn --help
```

支持参数：

```text
--epochs N
  训练轮数。默认 8。

--batch N
  batch size。默认 128。
  这个实现会在启动时按 batch size 分配 activation buffer。
  不建议超过 512。

--lr FLOAT
  Adam 学习率。默认 0.001。

--weight-decay FLOAT
  Adam 更新时使用的 weight decay。默认 0.0001。

--data PATH
  MNIST raw IDX 文件目录。默认 data/MNIST/raw。

--seed N
  随机种子。默认 42。

--no-progress
  关闭 tqdm-like 训练进度条。
```

示例：

```bash
./build/mnist_cuda_cnn --epochs 8 --batch 128 --lr 0.001
```

更重视准确率可以跑久一点：

```bash
./build/mnist_cuda_cnn --epochs 12 --batch 128 --lr 0.001
```

使用已有 MNIST 数据：

```bash
./build/mnist_cuda_cnn --data /path/to/MNIST/raw
```

关闭进度条，适合重定向日志：

```bash
./build/mnist_cuda_cnn --epochs 8 --no-progress > train.log
```

---

## 6. 训练进度条

训练时会显示类似 `tqdm` 的进度条：

```text
epoch 1/8 [==============>.................]  205/469  43.7% | loss 0.4821 | acc 85.42% | lr 1.0e-03 | 0:12<0:15
```

字段含义：

```text
epoch 1/8
  当前第几个 epoch，以及总 epoch 数。

205/469
  当前 epoch 已完成的 batch 数 / 总 batch 数。

43.7%
  当前 epoch 的进度百分比。

loss
  当前 epoch 到目前为止的平均训练 loss。

acc
  当前 epoch 到目前为止的平均训练 accuracy。

lr
  当前 epoch 使用的 learning rate。

0:12<0:15
  已用时间 < 预计剩余时间。
```

这个进度条是纯 C++ 实现，不依赖 Python 的 `tqdm` 包。

---

## 7. 网络结构

默认 CNN 是一个小型 LeNet 风格网络：

```text
Input: 1 x 28 x 28

  -> Conv2D(1 -> 8, kernel=5x5, padding=2)
  -> ReLU
  -> MaxPool2D(2x2)

  -> Conv2D(8 -> 16, kernel=5x5, padding=2)
  -> ReLU
  -> MaxPool2D(2x2)

  -> Flatten
  -> FC(16*7*7 -> 64)
  -> ReLU
  -> FC(64 -> 10)
```

维度变化：

```text
Input
  B x 1 x 28 x 28

Conv1, pad=2
  B x 8 x 28 x 28

MaxPool1, 2x2
  B x 8 x 14 x 14

Conv2, pad=2
  B x 16 x 14 x 14

MaxPool2, 2x2
  B x 16 x 7 x 7

Flatten
  B x 784

FC1
  B x 64

FC2 / logits
  B x 10
```

其中 `B` 是 batch size。

---

## 8. 代码中的核心模块

`src/main.cu` 虽然是单文件，但内部大致可以分为这些部分：

```text
1. Utility helpers
   - CUDA_CHECK
   - CUDA_LAUNCH_CHECK
   - ceil_div
   - shell_quote
   - run_cmd

2. MNIST loading
   - ensure_mnist_downloaded
   - read_be_u32
   - load_mnist_split

3. Device memory wrapper
   - DeviceBuffer<T>

4. Progress bar
   - ProgressBar
   - format_duration

5. CUDA kernels
   - conv2d_forward_kernel
   - conv2d_backward_input_kernel
   - conv2d_backward_weight_kernel
   - conv2d_backward_bias_kernel
   - relu_forward_kernel
   - relu_backward_kernel
   - maxpool2x2_forward_kernel
   - maxpool2x2_backward_kernel
   - fc_forward_kernel
   - fc_backward_input_kernel
   - fc_backward_weight_kernel
   - fc_backward_bias_kernel
   - softmax_xent_backward_kernel
   - adam_update_kernel

6. Model wrapper
   - Param2D
   - Activations
   - CNN

7. Training utilities
   - init_param
   - init_model
   - forward
   - backward
   - adam_update
   - copy_batch
   - evaluate

8. Program entry
   - parse_args
   - main
```

---

## 9. Forward 过程

`forward(net, x_dev, B)` 对应下面这条路径：

```text
x
  -> conv1.z1
  -> relu1.a1
  -> pool1.p1
  -> conv2.z2
  -> relu2.a2
  -> pool2.p2
  -> fc1.z3
  -> relu3.a3
  -> fc2.logits
```

数学上就是：

```text
z1 = Conv(x, W1, b1)
a1 = ReLU(z1)
p1 = MaxPool(a1)

z2 = Conv(p1, W2, b2)
a2 = ReLU(z2)
p2 = MaxPool(a2)

z3 = p2_flat W3^T + b3
a3 = ReLU(z3)
logits = a3 W4^T + b4
```

`logits` 是未经过 softmax 的原始分数，形状为：

```text
B x 10
```

每一行对应一张图片对数字 `0` 到 `9` 的预测分数。

---

## 10. Loss 和 Softmax

代码使用 softmax cross entropy。

对单个样本：

```text
p_c = exp(z_c) / sum_j exp(z_j)
loss = -log(p_y)
```

为了数值稳定，代码中先减去最大 logit：

```cpp
float maxv = z[0];
for (int c = 1; c < C; ++c) {
    if (z[c] > maxv) maxv = z[c];
}
```

然后计算：

```cpp
expf(z[c] - maxv)
```

这样可以避免 `exp` 溢出。

反向传播中最关键的公式是：

```text
dlogits = (softmax(logits) - one_hot(label)) / B
```

其中 `/ B` 表示使用 batch mean loss。

---

## 11. Backward 过程

`backward(net, x_dev, y_dev, B)` 对应反向链路：

```text
softmax cross entropy backward
  -> fc2 backward
  -> relu3 backward
  -> fc1 backward
  -> pool2 backward
  -> relu2 backward
  -> conv2 backward
  -> pool1 backward
  -> relu1 backward
  -> conv1 backward
```

梯度流大致是：

```text
dlogits
  -> da3, dW_fc2, db_fc2
  -> dz3
  -> dp2, dW_fc1, db_fc1
  -> da2
  -> dz2
  -> dp1, dW_conv2, db_conv2
  -> da1
  -> dz1
  -> dW_conv1, db_conv1
```

注意：

```text
Conv1 不需要计算 dx
```

因为输入图像不是可训练参数，训练时只需要更新权重和偏置。

---

## 12. Adam 更新

每个参数都有：

```text
p   parameter
g   gradient
m   first moment
v   second moment
```

Adam 更新公式：

```text
g_t = grad + weight_decay * param
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

代码中：

```cpp
adam_update_kernel<<<...>>>(...)
```

会对每个参数元素并行执行一次 Adam 更新。

Bias 不使用 weight decay：

```text
weight 参数使用 weight_decay
bias 参数不使用 weight_decay
```

---

## 13. 数据格式

MNIST 原始文件是 IDX 格式。

图像文件：

```text
train-images-idx3-ubyte
t10k-images-idx3-ubyte
```

标签文件：

```text
train-labels-idx1-ubyte
t10k-labels-idx1-ubyte
```

IDX 文件头部使用 big-endian 整数，所以代码中有：

```cpp
read_be_u32
```

图像读取后会归一化到：

```text
[0, 1]
```

即：

```cpp
data.images[i] = float(pixels[i]) / 255.0f;
```

---

## 14. 准确率预期

默认配置：

```bash
./build/mnist_cuda_cnn --epochs 8 --batch 128 --lr 0.001
```

通常可以在 MNIST 上达到比较高的测试准确率，常见范围大约是：

```text
97% ~ 98%+
```

更稳一点可以使用：

```bash
./build/mnist_cuda_cnn --epochs 12 --batch 128 --lr 0.001
```

注意：最终准确率会受到这些因素影响：

```text
GPU 型号
CUDA 版本
编译器版本
随机种子
batch size
学习率
训练轮数
```

这个项目优先保证代码可读和训练链路完整，不追求极致速度。

---

## 15. 性能说明

这个项目中的卷积实现是朴素 CUDA 版本。

例如 `conv2d_backward_weight_kernel` 的思路是：

```text
每个 thread 负责一个 weight 的梯度
然后在 thread 内部遍历 batch 和空间位置累加
```

这很好理解，但不是最快的做法。

高性能框架通常会使用：

```text
im2col + GEMM
shared memory tiling
tensor cores
cuDNN autotune
kernel fusion
mixed precision
```

本项目没有做这些优化，因为重点是从零理解 CNN 训练过程。

---

## 16. 常见问题

### 16.1 `nvcc: command not found`

说明 CUDA Toolkit 没有安装，或者 `nvcc` 不在 `PATH` 中。

先检查：

```bash
which nvcc
nvcc --version
```

如果找不到，需要安装 CUDA Toolkit。

---

### 16.2 `no CUDA-capable device is detected`

说明程序没有找到 NVIDIA GPU，常见原因：

```text
没有 NVIDIA 显卡
NVIDIA 驱动没装好
当前环境是没有 GPU 的容器
CUDA_VISIBLE_DEVICES 被设置为空
```

检查：

```bash
nvidia-smi
```

---

### 16.3 下载 MNIST 失败

程序依赖：

```text
curl
gzip
GitHub raw 访问
```

可以手动执行：

```bash
bash scripts/download_mnist.sh
```

如果网络访问 GitHub raw 不稳定，可以手动下载四个 gzip 文件，解压后放到：

```text
data/MNIST/raw
```

确保最终文件名是：

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

---

### 16.4 `std::filesystem` 编译失败

本项目需要 C++17。

确认 Makefile 中有：

```makefile
-std=c++17
```

如果 CUDA 或 GCC 版本很旧，可能需要升级编译器或 CUDA Toolkit。

---

### 16.5 程序很慢

这是正常的。

原因是本项目没有使用 cuDNN，也没有做复杂 CUDA 优化。它的定位是：

```text
教学型 from-scratch CUDA CNN
```

不是：

```text
工业级高性能深度学习框架
```

如果你想更快，需要逐步优化卷积 kernel，或者接入 cuBLAS/cuDNN。

---

### 16.6 accuracy 很低

可以先尝试：

```bash
./build/mnist_cuda_cnn --epochs 12 --batch 128 --lr 0.001
```

如果仍然明显异常，例如长期低于 90%，建议检查：

```text
MNIST 数据是否完整
label 文件是否对应 image 文件
CUDA kernel 是否报错
是否修改过学习率
是否修改过 batch size
是否修改过网络结构
```

也可以先跑一个小 batch debug：

```bash
./build/mnist_cuda_cnn --epochs 1 --batch 32
```

---

## 17. 调试建议

Debug 编译：

```bash
make debug
```

使用 CUDA memcheck：

```bash
cuda-memcheck ./build/mnist_cuda_cnn --epochs 1 --batch 32 --no-progress
```

如果你的 CUDA 版本较新，也可以使用：

```bash
compute-sanitizer ./build/mnist_cuda_cnn --epochs 1 --batch 32 --no-progress
```

建议调试时先降低 batch size：

```bash
./build/mnist_cuda_cnn --epochs 1 --batch 16 --no-progress
```

这样更容易定位问题。

---

## 18. 适合怎么学习这份代码

推荐阅读顺序：

```text
1. main 函数
   先看整体训练流程。

2. load_mnist_split
   理解 MNIST IDX 是怎么读进来的。

3. DeviceBuffer
   理解 GPU memory 的 RAII 管理。

4. forward
   看 CNN 前向传播的数据流。

5. softmax_xent_backward_kernel
   理解 loss 和 logits 梯度。

6. backward
   看链式法则如何一层层传回去。

7. adam_update_kernel
   看参数如何被 optimizer 更新。

8. conv2d_forward_kernel / conv2d_backward_*_kernel
   最后细看卷积的 forward/backward。
```

不要一开始就钻进卷积反向传播 kernel。更好的顺序是先建立全局图：

```text
数据如何来
参数在哪里
activation 存在哪里
gradient 存在哪里
loss 怎么算
optimizer 怎么更新参数
```

然后再看每个 CUDA kernel 的索引计算。

---

## 19. 设计取舍

这个版本做了几件偏规范的事情：

```text
DeviceBuffer 析构函数 noexcept
CUDA API 调用使用 CUDA_CHECK
CUDA kernel launch 后使用 CUDA_LAUNCH_CHECK
loss 统计按样本数加权
correct 直接读取，不用 round(acc * B)
训练过程有 tqdm-like progress bar
参数解析有基本合法性检查
```

但它没有过度工程化：

```text
没有拆很多 .hpp/.cu 文件
没有引入 CMake
没有引入第三方依赖
没有封装复杂 Tensor 类
没有引入模板化神经网络框架
没有把每个 layer 抽象成继承体系
```

原因是本项目的主要目标是学习：

```text
CNN 训练的完整数学和 CUDA 实现链路
```

而不是制造一个小型 PyTorch。

---

## 20. 典型运行流程

完整流程：

```bash
unzip mnist_cuda_cnn_v2.zip
cd mnist_cuda_cnn_v2
make
./build/mnist_cuda_cnn --epochs 8 --batch 128 --lr 0.001
```

如果想重新开始：

```bash
make clean
make
./build/mnist_cuda_cnn
```

如果想只重新下载数据：

```bash
rm -rf data/MNIST/raw
bash scripts/download_mnist.sh
```

---

## 21. 后续可改进方向

如果你想在这个项目基础上继续学习，可以按下面顺序改：

```text
1. 保存和加载模型参数
2. 加入 validation set
3. 增加 confusion matrix
4. 把 Conv channel 改成 16/32，提高准确率
5. 加入 dropout
6. 加入 learning rate scheduler 参数
7. 把数据拷贝改成 pinned memory
8. 优化 FC kernel
9. 用 shared memory 优化卷积
10. 尝试 im2col + cuBLAS
11. 尝试 half precision / mixed precision
```

推荐先做：

```text
保存模型参数
加载模型参数
打印单张图片预测结果
```

这三个改动最适合继续理解完整机器学习工程链路。

---

## 22. Matplotlib 可视化输出：训练曲线、预测样本、第一层卷积核

这一节是后续追加内容，原 README 上面的说明没有删除。当前版本在保持 CUDA 训练主体尽量少改的前提下，新增了一个 Python/Matplotlib 可视化脚本：

```text
scripts/visualize_results.py
```

CUDA 主程序训练结束后，会默认把少量可视化所需数据导出到：

```text
runs/latest/
├── metrics.csv
├── prediction_samples.csv
└── conv1_weights.csv
```

然后 Python 脚本会把这些 CSV 画成图片：

```text
runs/latest/figures/
├── training_curves.png
├── prediction_grid.png
├── conv1_filters.png
└── confidence_histogram.png
```

### 22.1 先回答一个重要问题：权重现在是怎样保存的？

原始版本里，模型权重并没有被保存成磁盘上的 checkpoint 文件。

训练时的参数都保存在 GPU 显存中的 `DeviceBuffer<float>` 里。模型结构大致是：

```cpp
struct CNN {
    Param2D conv1;
    Param2D conv2;
    Param2D fc1;
    Param2D fc2;
    Activations act;
};
```

其中每个 `Param2D` 里面有：

```cpp
DeviceBuffer<float> w;   // weight 参数
DeviceBuffer<float> b;   // bias 参数
DeviceBuffer<float> gw;  // weight gradient
DeviceBuffer<float> gb;  // bias gradient
DeviceBuffer<float> mw;  // Adam 一阶动量
DeviceBuffer<float> vw;  // Adam 二阶动量
DeviceBuffer<float> mb;
DeviceBuffer<float> vb;
```

也就是说，真正的权重在：

```text
conv1.w, conv1.b
conv2.w, conv2.b
fc1.w,   fc1.b
fc2.w,   fc2.b
```

这些都是 `DeviceBuffer<float>`，也就是 CUDA device memory。它们在程序运行期间存在，程序结束后会通过 RAII 析构释放；原始版本不会把完整模型保存到 `.bin`、`.pt`、`.ckpt` 或 `.npz`。

本次修改为了可视化第一层卷积核，额外导出了：

```text
runs/latest/conv1_weights.csv
```

注意：这个文件只保存 `conv1.w`，用于画第一层卷积核，不是完整模型 checkpoint。如果以后想实现真正的模型保存/恢复，应该再单独保存：

```text
conv1.w, conv1.b
conv2.w, conv2.b
fc1.w, fc1.b
fc2.w, fc2.b
```

并且最好保存成二进制格式，而不是 CSV。

### 22.2 需要安装的 Python 依赖

如果你已经有 Python 环境，直接安装：

```bash
pip install matplotlib numpy
```

项目里也新增了：

```text
requirements.txt
```

所以也可以运行：

```bash
pip install -r requirements.txt
```

### 22.3 训练并导出可视化数据

正常训练：

```bash
make
./build/mnist_cuda_cnn --epochs 8 --batch 128 --lr 0.001
```

训练结束后会看到类似输出：

```text
[export] metrics will be written to runs/latest/metrics.csv
...
[export] wrote runs/latest/prediction_samples.csv
[export] wrote runs/latest/conv1_weights.csv
[export] run: python scripts/visualize_results.py --run-dir runs/latest
```

也可以指定输出目录：

```bash
./build/mnist_cuda_cnn --epochs 8 --batch 128 --out runs/exp01
```

这样 CSV 会写到：

```text
runs/exp01/
```

### 22.4 生成 Matplotlib 图片

默认读取 `runs/latest`：

```bash
python scripts/visualize_results.py --run-dir runs/latest
```

或者使用 Makefile target：

```bash
make visualize
```

如果你训练时使用了自定义输出目录：

```bash
python scripts/visualize_results.py --run-dir runs/exp01
```

### 22.5 每张图是什么意思？

#### training_curves.png

显示训练过程中的：

```text
train loss
train accuracy
test accuracy
```

它回答的问题是：

```text
模型有没有真的在学习？
训练 loss 是否下降？
训练准确率和测试准确率是否同步提高？
是否出现明显 overfitting？
```

#### prediction_grid.png

显示一组测试集图片，以及模型预测结果：

```text
y=<真实标签> pred=<预测标签> conf=<softmax 置信度>
```

标题颜色含义：

```text
绿色：预测正确
红色：预测错误
```

这张图可以帮助你直观看到模型到底在分类哪些数字，而不是只看一个最终准确率。

#### conv1_filters.png

显示第一层卷积核 `conv1.w` 的 8 个 5×5 filter。

第一层卷积核通常会学到一些局部边缘、笔画、方向性响应。这个项目的第一层是：

```text
Conv(1 -> 8, kernel=5x5, padding=2)
```

所以一共有 8 个 filter，每个 filter 是 5×5。

#### confidence_histogram.png

显示导出的测试样本里，模型 softmax confidence 的分布。

它回答的问题是：

```text
模型预测正确时通常有多自信？
模型预测错误时是否也过度自信？
```

### 22.6 新增的命令行参数

本次 CUDA 主程序只做了很小的接口扩展：

```bash
--out DIR
```

指定 CSV 输出目录，默认：

```text
runs/latest
```

例如：

```bash
./build/mnist_cuda_cnn --out runs/exp01
```

另外新增：

```bash
--viz-samples N
```

指定导出多少个测试样本给 Python 画图，默认：

```text
64
```

例如：

```bash
./build/mnist_cuda_cnn --viz-samples 100
```

如果你只想看训练曲线和卷积核，不想导出预测样本，也可以：

```bash
./build/mnist_cuda_cnn --viz-samples 0
```

不过这时 `prediction_grid.png` 和 `confidence_histogram.png` 无法生成。

### 22.7 这次 CUDA/C++ 具体最小改了什么？

核心 CUDA kernel 没有改：

```text
conv2d_forward_kernel
conv2d_backward_input_kernel
conv2d_backward_weight_kernel
maxpool2x2_forward_kernel
maxpool2x2_backward_kernel
fc_forward_kernel
fc_backward_weight_kernel
softmax_xent_backward_kernel
adam_update_kernel
```

也就是说，训练数学逻辑和 GPU kernel 主体没有变。

只是在 host-side C++ 增加了几个导出函数：

```cpp
softmax_host(...)
export_prediction_samples(...)
export_conv1_weights(...)
```

以及训练循环中每个 epoch 追加写入：

```text
metrics.csv
```

这些改动的目的只是把训练过程和最终预测结果保存成 Python 容易读取的 CSV。

### 22.8 一套完整运行流程

从零开始：

```bash
make
./build/mnist_cuda_cnn --epochs 8 --batch 128 --lr 0.001
pip install -r requirements.txt
python scripts/visualize_results.py --run-dir runs/latest
```

然后打开：

```text
runs/latest/figures/training_curves.png
runs/latest/figures/prediction_grid.png
runs/latest/figures/conv1_filters.png
runs/latest/figures/confidence_histogram.png
```

如果你在 VS Code 里，可以直接在 Explorer 里点开这些 PNG。
