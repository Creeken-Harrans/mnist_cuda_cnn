// mnist_cuda_cnn/src/main.cu
// A compact from-scratch C++/CUDA MNIST CNN trainer.
// No PyTorch, no libtorch, no cuDNN. CUDA kernels implement convolution, pooling,
// fully-connected layers, ReLU, softmax cross entropy, and Adam updates.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::ostringstream oss__;                                             \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "    \
                  << cudaGetErrorString(err__);                                  \
            throw std::runtime_error(oss__.str());                               \
        }                                                                        \
    } while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char ch : s) {
        if (ch == '\'') out += "'\\''";
        else out += ch;
    }
    out += "'";
    return out;
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

static void run_cmd(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("Command failed: " + cmd);
    }
}

static void ensure_mnist_downloaded(const std::string& raw_dir) {
    std::filesystem::create_directories(raw_dir);
    const std::string base = "https://raw.githubusercontent.com/fgnt/mnist/master";
    const std::vector<std::string> files = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    };

    for (const auto& f : files) {
        const std::string gz = raw_dir + "/" + f;
        const std::string raw = raw_dir + "/" + f.substr(0, f.size() - 3);
        if (file_exists(raw)) {
            std::cout << "[mnist] found " << raw << "\n";
            continue;
        }
        if (!file_exists(gz)) {
            std::cout << "[mnist] downloading " << f << "\n";
            run_cmd("curl -L --retry 5 --fail -o " + shell_quote(gz) + " " + shell_quote(base + "/" + f));
        }
        std::cout << "[mnist] decompressing " << f << "\n";
        run_cmd("gzip -dkf " + shell_quote(gz));
    }
}

static uint32_t read_be_u32(std::ifstream& in) {
    uint8_t b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Unexpected EOF while reading IDX header");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

struct MnistData {
    std::vector<float> images; // N * 28 * 28, normalized to [0, 1]
    std::vector<int> labels;   // N
    int n = 0;
};

static MnistData load_mnist_split(const std::string& image_path, const std::string& label_path) {
    std::ifstream img(image_path, std::ios::binary);
    std::ifstream lab(label_path, std::ios::binary);
    if (!img) throw std::runtime_error("Cannot open image file: " + image_path);
    if (!lab) throw std::runtime_error("Cannot open label file: " + label_path);

    const uint32_t img_magic = read_be_u32(img);
    const uint32_t n_images = read_be_u32(img);
    const uint32_t rows = read_be_u32(img);
    const uint32_t cols = read_be_u32(img);
    const uint32_t lab_magic = read_be_u32(lab);
    const uint32_t n_labels = read_be_u32(lab);

    if (img_magic != 2051) throw std::runtime_error("Bad image IDX magic: " + std::to_string(img_magic));
    if (lab_magic != 2049) throw std::runtime_error("Bad label IDX magic: " + std::to_string(lab_magic));
    if (n_images != n_labels) throw std::runtime_error("Image/label count mismatch");
    if (rows != 28 || cols != 28) throw std::runtime_error("This code expects 28x28 MNIST images");

    MnistData data;
    data.n = static_cast<int>(n_images);
    data.images.resize(size_t(data.n) * 28 * 28);
    data.labels.resize(data.n);

    std::vector<uint8_t> pixels(size_t(data.n) * 28 * 28);
    std::vector<uint8_t> labels(data.n);
    img.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    lab.read(reinterpret_cast<char*>(labels.data()), labels.size());
    if (!img || !lab) throw std::runtime_error("Unexpected EOF while reading MNIST payload");

    for (size_t i = 0; i < pixels.size(); ++i) {
        data.images[i] = float(pixels[i]) / 255.0f;
    }
    for (int i = 0; i < data.n; ++i) {
        data.labels[i] = int(labels[i]);
    }
    return data;
}

template <typename T>
struct DeviceBuffer {
    T* p = nullptr;
    size_t n = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { alloc(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() { release(); }

    void alloc(size_t count) {
        release();
        n = count;
        if (n > 0) CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    }
    void release() {
        if (p) CUDA_CHECK(cudaFree(p));
        p = nullptr;
        n = 0;
    }
    void zero() { CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T))); }
    void copy_from_host(const T* src, size_t count) {
        if (count > n) throw std::runtime_error("copy_from_host count exceeds buffer size");
        CUDA_CHECK(cudaMemcpy(p, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T* dst, size_t count) const {
        if (count > n) throw std::runtime_error("copy_to_host count exceeds buffer size");
        CUDA_CHECK(cudaMemcpy(dst, p, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// ---------------- CUDA kernels: layer forward/backward ----------------

__global__ void conv2d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int IC, int IH, int IW, int OC, int KH, int KW, int pad, int OH, int OW) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OC * OH * OW;
    if (tid >= total) return;

    int ow = tid % OW;
    int t = tid / OW;
    int oh = t % OH;
    t /= OH;
    int oc = t % OC;
    int n = t / OC;

    float sum = b[oc];
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh - pad + kh;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow - pad + kw;
                if (iw < 0 || iw >= IW) continue;
                int xidx = ((n * IC + ic) * IH + ih) * IW + iw;
                int widx = ((oc * IC + ic) * KH + kh) * KW + kw;
                sum += x[xidx] * w[widx];
            }
        }
    }
    y[tid] = sum;
}

__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ w,
    float* __restrict__ dx,
    int N, int IC, int IH, int IW, int OC, int KH, int KW, int pad, int OH, int OW) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * IC * IH * IW;
    if (tid >= total) return;

    int iw = tid % IW;
    int t = tid / IW;
    int ih = t % IH;
    t /= IH;
    int ic = t % IC;
    int n = t / IC;

    float sum = 0.0f;
    for (int oc = 0; oc < OC; ++oc) {
        for (int kh = 0; kh < KH; ++kh) {
            int oh = ih + pad - kh;
            if (oh < 0 || oh >= OH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int ow = iw + pad - kw;
                if (ow < 0 || ow >= OW) continue;
                int dyidx = ((n * OC + oc) * OH + oh) * OW + ow;
                int widx = ((oc * IC + ic) * KH + kh) * KW + kw;
                sum += dy[dyidx] * w[widx];
            }
        }
    }
    dx[tid] = sum;
}

__global__ void conv2d_backward_weight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float* __restrict__ dw,
    int N, int IC, int IH, int IW, int OC, int KH, int KW, int pad, int OH, int OW) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OC * IC * KH * KW;
    if (tid >= total) return;

    int kw = tid % KW;
    int t = tid / KW;
    int kh = t % KH;
    t /= KH;
    int ic = t % IC;
    int oc = t / IC;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            int ih = oh - pad + kh;
            if (ih < 0 || ih >= IH) continue;
            for (int ow = 0; ow < OW; ++ow) {
                int iw = ow - pad + kw;
                if (iw < 0 || iw >= IW) continue;
                int xidx = ((n * IC + ic) * IH + ih) * IW + iw;
                int dyidx = ((n * OC + oc) * OH + oh) * OW + ow;
                sum += x[xidx] * dy[dyidx];
            }
        }
    }
    dw[tid] = sum;
}

__global__ void conv2d_backward_bias_kernel(
    const float* __restrict__ dy,
    float* __restrict__ db,
    int N, int OC, int OH, int OW) {

    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC) return;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                sum += dy[((n * OC + oc) * OH + h) * OW + w];
            }
        }
    }
    db[oc] = sum;
}

__global__ void relu_forward_kernel(const float* __restrict__ x, float* __restrict__ y, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void relu_backward_kernel(
    const float* __restrict__ y,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dx[i] = (y[i] > 0.0f) ? dy[i] : 0.0f;
}

__global__ void maxpool2x2_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int* __restrict__ mask,
    int N, int C, int IH, int IW) {

    int OH = IH / 2;
    int OW = IW / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (tid >= total) return;

    int ow = tid % OW;
    int t = tid / OW;
    int oh = t % OH;
    t /= OH;
    int c = t % C;
    int n = t / C;

    int ih0 = oh * 2;
    int iw0 = ow * 2;
    int base = ((n * C + c) * IH + ih0) * IW + iw0;
    float best = x[base];
    int best_idx = base;

    int idx1 = base + 1;
    int idx2 = base + IW;
    int idx3 = base + IW + 1;
    if (x[idx1] > best) { best = x[idx1]; best_idx = idx1; }
    if (x[idx2] > best) { best = x[idx2]; best_idx = idx2; }
    if (x[idx3] > best) { best = x[idx3]; best_idx = idx3; }

    y[tid] = best;
    mask[tid] = best_idx;
}

__global__ void maxpool2x2_backward_kernel(
    const float* __restrict__ dy,
    const int* __restrict__ mask,
    float* __restrict__ dx,
    int total_out) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_out) {
        dx[mask[i]] = dy[i]; // no overlap for non-overlapping 2x2 pooling windows
    }
}

__global__ void fc_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int IN, int OUT) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OUT;
    if (tid >= total) return;
    int o = tid % OUT;
    int n = tid / OUT;

    float sum = b[o];
    const float* xn = x + n * IN;
    const float* wo = w + o * IN;
    for (int i = 0; i < IN; ++i) sum += xn[i] * wo[i];
    y[tid] = sum;
}

__global__ void fc_backward_input_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ w,
    float* __restrict__ dx,
    int N, int IN, int OUT) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * IN;
    if (tid >= total) return;
    int i = tid % IN;
    int n = tid / IN;

    float sum = 0.0f;
    for (int o = 0; o < OUT; ++o) {
        sum += dy[n * OUT + o] * w[o * IN + i];
    }
    dx[tid] = sum;
}

__global__ void fc_backward_weight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float* __restrict__ dw,
    int N, int IN, int OUT) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OUT * IN;
    if (tid >= total) return;
    int i = tid % IN;
    int o = tid / IN;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += dy[n * OUT + o] * x[n * IN + i];
    }
    dw[tid] = sum;
}

__global__ void fc_backward_bias_kernel(
    const float* __restrict__ dy,
    float* __restrict__ db,
    int N, int OUT) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) sum += dy[n * OUT + o];
    db[o] = sum;
}

__global__ void softmax_xent_backward_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    float* __restrict__ dlogits,
    float* __restrict__ loss,
    int* __restrict__ correct,
    int N, int C) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* z = logits + n * C;
    float maxv = z[0];
    int pred = 0;
    for (int c = 1; c < C; ++c) {
        if (z[c] > maxv) { maxv = z[c]; pred = c; }
    }

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) sum += expf(z[c] - maxv);
    float inv_sum = 1.0f / sum;
    int y = labels[n];

    for (int c = 0; c < C; ++c) {
        float p = expf(z[c] - maxv) * inv_sum;
        dlogits[n * C + c] = (p - (c == y ? 1.0f : 0.0f)) / float(N);
    }
    float py = expf(z[y] - maxv) * inv_sum;
    atomicAdd(loss, -logf(fmaxf(py, 1e-20f)) / float(N));
    if (pred == y) atomicAdd(correct, 1);
}

__global__ void adam_update_kernel(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float beta1_pow_t,
    float beta2_pow_t,
    float weight_decay) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float grad = g[i] + weight_decay * p[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * grad;
    float vi = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    m[i] = mi;
    v[i] = vi;
    float mhat = mi / (1.0f - beta1_pow_t);
    float vhat = vi / (1.0f - beta2_pow_t);
    p[i] -= lr * mhat / (sqrtf(vhat) + eps);
}

// ---------------- Host-side neural network wrapper ----------------

struct Param2D {
    int rows = 0;
    int cols = 0;
    DeviceBuffer<float> w, b, gw, gb, mw, vw, mb, vb;

    void alloc(int r, int c) {
        rows = r;
        cols = c;
        w.alloc(size_t(r) * c);
        b.alloc(r);
        gw.alloc(size_t(r) * c);
        gb.alloc(r);
        mw.alloc(size_t(r) * c); mw.zero();
        vw.alloc(size_t(r) * c); vw.zero();
        mb.alloc(r); mb.zero();
        vb.alloc(r); vb.zero();
    }
};

struct Activations {
    DeviceBuffer<float> z1, a1, p1;     // conv1/relu/pool
    DeviceBuffer<int>   m1;
    DeviceBuffer<float> z2, a2, p2;     // conv2/relu/pool
    DeviceBuffer<int>   m2;
    DeviceBuffer<float> z3, a3, logits; // fc1/relu/fc2

    DeviceBuffer<float> dz1, da1, dp1;
    DeviceBuffer<float> dz2, da2, dp2;
    DeviceBuffer<float> dz3, da3, dlogits;

    void alloc(int maxB) {
        z1.alloc(size_t(maxB) * 8 * 28 * 28);
        a1.alloc(size_t(maxB) * 8 * 28 * 28);
        p1.alloc(size_t(maxB) * 8 * 14 * 14);
        m1.alloc(size_t(maxB) * 8 * 14 * 14);

        z2.alloc(size_t(maxB) * 16 * 14 * 14);
        a2.alloc(size_t(maxB) * 16 * 14 * 14);
        p2.alloc(size_t(maxB) * 16 * 7 * 7);
        m2.alloc(size_t(maxB) * 16 * 7 * 7);

        z3.alloc(size_t(maxB) * 64);
        a3.alloc(size_t(maxB) * 64);
        logits.alloc(size_t(maxB) * 10);

        dz1.alloc(size_t(maxB) * 8 * 28 * 28);
        da1.alloc(size_t(maxB) * 8 * 28 * 28);
        dp1.alloc(size_t(maxB) * 8 * 14 * 14);

        dz2.alloc(size_t(maxB) * 16 * 14 * 14);
        da2.alloc(size_t(maxB) * 16 * 14 * 14);
        dp2.alloc(size_t(maxB) * 16 * 7 * 7);

        dz3.alloc(size_t(maxB) * 64);
        da3.alloc(size_t(maxB) * 64);
        dlogits.alloc(size_t(maxB) * 10);
    }
};

struct CNN {
    // conv weights are stored as Param2D.w flattened: rows=OC, cols=IC*KH*KW.
    Param2D conv1; // 1 -> 8, 5x5, pad 2
    Param2D conv2; // 8 -> 16, 5x5, pad 2
    Param2D fc1;   // 16*7*7 -> 64
    Param2D fc2;   // 64 -> 10
    Activations act;

    DeviceBuffer<float> loss_dev;
    DeviceBuffer<int> correct_dev;

    int max_batch = 128;

    void alloc(int maxB) {
        max_batch = maxB;
        conv1.alloc(8, 1 * 5 * 5);
        conv2.alloc(16, 8 * 5 * 5);
        fc1.alloc(64, 16 * 7 * 7);
        fc2.alloc(10, 64);
        act.alloc(maxB);
        loss_dev.alloc(1);
        correct_dev.alloc(1);
    }
};

static void init_param(Param2D& p, float stddev, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, stddev);
    std::vector<float> hw(size_t(p.rows) * p.cols);
    std::vector<float> hb(p.rows, 0.0f);
    for (auto& x : hw) x = nd(rng);
    p.w.copy_from_host(hw.data(), hw.size());
    p.b.copy_from_host(hb.data(), hb.size());
}

static void init_model(CNN& net, unsigned seed) {
    std::mt19937 rng(seed);
    init_param(net.conv1, std::sqrt(2.0f / float(1 * 5 * 5)), rng);
    init_param(net.conv2, std::sqrt(2.0f / float(8 * 5 * 5)), rng);
    init_param(net.fc1,   std::sqrt(2.0f / float(16 * 7 * 7)), rng);
    init_param(net.fc2,   std::sqrt(2.0f / float(64)), rng);
}

static void launch_relu_forward(const DeviceBuffer<float>& x, DeviceBuffer<float>& y, int total) {
    relu_forward_kernel<<<ceil_div(total, 256), 256>>>(x.p, y.p, total);
}

static void launch_relu_backward(const DeviceBuffer<float>& y, const DeviceBuffer<float>& dy, DeviceBuffer<float>& dx, int total) {
    relu_backward_kernel<<<ceil_div(total, 256), 256>>>(y.p, dy.p, dx.p, total);
}

static void forward(CNN& net, const float* x_dev, int B) {
    // x: B x 1 x 28 x 28
    conv2d_forward_kernel<<<ceil_div(B * 8 * 28 * 28, 256), 256>>>(
        x_dev, net.conv1.w.p, net.conv1.b.p, net.act.z1.p,
        B, 1, 28, 28, 8, 5, 5, 2, 28, 28);
    launch_relu_forward(net.act.z1, net.act.a1, B * 8 * 28 * 28);
    maxpool2x2_forward_kernel<<<ceil_div(B * 8 * 14 * 14, 256), 256>>>(
        net.act.a1.p, net.act.p1.p, net.act.m1.p, B, 8, 28, 28);

    conv2d_forward_kernel<<<ceil_div(B * 16 * 14 * 14, 256), 256>>>(
        net.act.p1.p, net.conv2.w.p, net.conv2.b.p, net.act.z2.p,
        B, 8, 14, 14, 16, 5, 5, 2, 14, 14);
    launch_relu_forward(net.act.z2, net.act.a2, B * 16 * 14 * 14);
    maxpool2x2_forward_kernel<<<ceil_div(B * 16 * 7 * 7, 256), 256>>>(
        net.act.a2.p, net.act.p2.p, net.act.m2.p, B, 16, 14, 14);

    fc_forward_kernel<<<ceil_div(B * 64, 256), 256>>>(
        net.act.p2.p, net.fc1.w.p, net.fc1.b.p, net.act.z3.p,
        B, 16 * 7 * 7, 64);
    launch_relu_forward(net.act.z3, net.act.a3, B * 64);
    fc_forward_kernel<<<ceil_div(B * 10, 256), 256>>>(
        net.act.a3.p, net.fc2.w.p, net.fc2.b.p, net.act.logits.p,
        B, 64, 10);
}

static void backward(CNN& net, const float* x_dev, const int* y_dev, int B) {
    // loss_dev and correct_dev are filled by softmax_xent_backward_kernel.
    net.loss_dev.zero();
    net.correct_dev.zero();
    softmax_xent_backward_kernel<<<ceil_div(B, 128), 128>>>(
        net.act.logits.p, y_dev, net.act.dlogits.p, net.loss_dev.p, net.correct_dev.p, B, 10);

    // fc2 backward
    fc_backward_input_kernel<<<ceil_div(B * 64, 256), 256>>>(
        net.act.dlogits.p, net.fc2.w.p, net.act.da3.p, B, 64, 10);
    fc_backward_weight_kernel<<<ceil_div(10 * 64, 256), 256>>>(
        net.act.a3.p, net.act.dlogits.p, net.fc2.gw.p, B, 64, 10);
    fc_backward_bias_kernel<<<ceil_div(10, 256), 256>>>(
        net.act.dlogits.p, net.fc2.gb.p, B, 10);

    // ReLU after fc1, then fc1 backward
    launch_relu_backward(net.act.a3, net.act.da3, net.act.dz3, B * 64);
    fc_backward_input_kernel<<<ceil_div(B * 16 * 7 * 7, 256), 256>>>(
        net.act.dz3.p, net.fc1.w.p, net.act.dp2.p, B, 16 * 7 * 7, 64);
    fc_backward_weight_kernel<<<ceil_div(64 * 16 * 7 * 7, 256), 256>>>(
        net.act.p2.p, net.act.dz3.p, net.fc1.gw.p, B, 16 * 7 * 7, 64);
    fc_backward_bias_kernel<<<ceil_div(64, 256), 256>>>(
        net.act.dz3.p, net.fc1.gb.p, B, 64);

    // pool2 backward -> relu2 backward -> conv2 backward
    CUDA_CHECK(cudaMemset(net.act.da2.p, 0, size_t(B) * 16 * 14 * 14 * sizeof(float)));
    maxpool2x2_backward_kernel<<<ceil_div(B * 16 * 7 * 7, 256), 256>>>(
        net.act.dp2.p, net.act.m2.p, net.act.da2.p, B * 16 * 7 * 7);
    launch_relu_backward(net.act.a2, net.act.da2, net.act.dz2, B * 16 * 14 * 14);

    conv2d_backward_input_kernel<<<ceil_div(B * 8 * 14 * 14, 256), 256>>>(
        net.act.dz2.p, net.conv2.w.p, net.act.dp1.p,
        B, 8, 14, 14, 16, 5, 5, 2, 14, 14);
    conv2d_backward_weight_kernel<<<ceil_div(16 * 8 * 5 * 5, 256), 256>>>(
        net.act.p1.p, net.act.dz2.p, net.conv2.gw.p,
        B, 8, 14, 14, 16, 5, 5, 2, 14, 14);
    conv2d_backward_bias_kernel<<<ceil_div(16, 256), 256>>>(
        net.act.dz2.p, net.conv2.gb.p, B, 16, 14, 14);

    // pool1 backward -> relu1 backward -> conv1 weight/bias backward
    CUDA_CHECK(cudaMemset(net.act.da1.p, 0, size_t(B) * 8 * 28 * 28 * sizeof(float)));
    maxpool2x2_backward_kernel<<<ceil_div(B * 8 * 14 * 14, 256), 256>>>(
        net.act.dp1.p, net.act.m1.p, net.act.da1.p, B * 8 * 14 * 14);
    launch_relu_backward(net.act.a1, net.act.da1, net.act.dz1, B * 8 * 28 * 28);

    conv2d_backward_weight_kernel<<<ceil_div(8 * 1 * 5 * 5, 256), 256>>>(
        x_dev, net.act.dz1.p, net.conv1.gw.p,
        B, 1, 28, 28, 8, 5, 5, 2, 28, 28);
    conv2d_backward_bias_kernel<<<ceil_div(8, 256), 256>>>(
        net.act.dz1.p, net.conv1.gb.p, B, 8, 28, 28);
}

static void adam_update_param(Param2D& p, float lr, int step, float weight_decay) {
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    float b1t = std::pow(beta1, float(step));
    float b2t = std::pow(beta2, float(step));

    int nw = p.rows * p.cols;
    adam_update_kernel<<<ceil_div(nw, 256), 256>>>(
        p.w.p, p.gw.p, p.mw.p, p.vw.p, nw, lr, beta1, beta2, eps, b1t, b2t, weight_decay);
    adam_update_kernel<<<ceil_div(p.rows, 256), 256>>>(
        p.b.p, p.gb.p, p.mb.p, p.vb.p, p.rows, lr, beta1, beta2, eps, b1t, b2t, 0.0f);
}

static void adam_update(CNN& net, float lr, int step, float weight_decay) {
    adam_update_param(net.conv1, lr, step, weight_decay);
    adam_update_param(net.conv2, lr, step, weight_decay);
    adam_update_param(net.fc1, lr, step, weight_decay);
    adam_update_param(net.fc2, lr, step, weight_decay);
}

struct BatchDevice {
    DeviceBuffer<float> x;
    DeviceBuffer<int> y;
    std::vector<float> hx;
    std::vector<int> hy;

    void alloc(int maxB) {
        x.alloc(size_t(maxB) * 28 * 28);
        y.alloc(maxB);
        hx.resize(size_t(maxB) * 28 * 28);
        hy.resize(maxB);
    }
};

static void copy_batch(const MnistData& data, const std::vector<int>& order, int start, int B, BatchDevice& batch) {
    for (int i = 0; i < B; ++i) {
        int src = order[start + i];
        std::memcpy(batch.hx.data() + size_t(i) * 28 * 28,
                    data.images.data() + size_t(src) * 28 * 28,
                    28 * 28 * sizeof(float));
        batch.hy[i] = data.labels[src];
    }
    batch.x.copy_from_host(batch.hx.data(), size_t(B) * 28 * 28);
    batch.y.copy_from_host(batch.hy.data(), B);
}

static std::pair<float, float> read_loss_acc(CNN& net, int B) {
    float loss = 0.0f;
    int correct = 0;
    net.loss_dev.copy_to_host(&loss, 1);
    net.correct_dev.copy_to_host(&correct, 1);
    return {loss, float(correct) / float(B)};
}

static float evaluate(CNN& net, const MnistData& test, BatchDevice& batch, int batch_size) {
    std::vector<int> order(test.n);
    std::iota(order.begin(), order.end(), 0);
    int total_correct = 0;
    int total_seen = 0;

    for (int start = 0; start < test.n; start += batch_size) {
        int B = std::min(batch_size, test.n - start);
        copy_batch(test, order, start, B, batch);
        forward(net, batch.x.p, B);
        net.loss_dev.zero();
        net.correct_dev.zero();
        softmax_xent_backward_kernel<<<ceil_div(B, 128), 128>>>(
            net.act.logits.p, batch.y.p, net.act.dlogits.p, net.loss_dev.p, net.correct_dev.p, B, 10);
        CUDA_CHECK(cudaDeviceSynchronize());
        int correct = 0;
        net.correct_dev.copy_to_host(&correct, 1);
        total_correct += correct;
        total_seen += B;
    }
    return float(total_correct) / float(total_seen);
}

struct Options {
    std::string data_dir = "data/MNIST/raw";
    int epochs = 8;
    int batch_size = 128;
    float lr = 1e-3f;
    float weight_decay = 1e-4f;
    unsigned seed = 42;
};

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (a == "--data") opt.data_dir = need_value(a);
        else if (a == "--epochs") opt.epochs = std::stoi(need_value(a));
        else if (a == "--batch") opt.batch_size = std::stoi(need_value(a));
        else if (a == "--lr") opt.lr = std::stof(need_value(a));
        else if (a == "--weight-decay") opt.weight_decay = std::stof(need_value(a));
        else if (a == "--seed") opt.seed = unsigned(std::stoul(need_value(a)));
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./mnist_cuda_cnn [--epochs 8] [--batch 128] [--lr 0.001] "
                         "[--weight-decay 0.0001] [--data data/MNIST/raw] [--seed 42]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    if (opt.batch_size <= 0) throw std::runtime_error("batch size must be positive");
    if (opt.batch_size > 512) throw std::runtime_error("batch size >512 is not recommended for this simple implementation");
    return opt;
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        int dev = 0;
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::cout << "[cuda] device: " << prop.name << "\n";

        ensure_mnist_downloaded(opt.data_dir);

        const std::string train_images = opt.data_dir + "/train-images-idx3-ubyte";
        const std::string train_labels = opt.data_dir + "/train-labels-idx1-ubyte";
        const std::string test_images  = opt.data_dir + "/t10k-images-idx3-ubyte";
        const std::string test_labels  = opt.data_dir + "/t10k-labels-idx1-ubyte";

        std::cout << "[data] loading MNIST\n";
        MnistData train = load_mnist_split(train_images, train_labels);
        MnistData test = load_mnist_split(test_images, test_labels);
        std::cout << "[data] train=" << train.n << " test=" << test.n << "\n";

        CNN net;
        net.alloc(opt.batch_size);
        init_model(net, opt.seed);
        BatchDevice batch;
        batch.alloc(opt.batch_size);

        std::vector<int> order(train.n);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(opt.seed);

        int step = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "[train] architecture: Conv(1->8,5,pad2)-ReLU-MaxPool-"
                     "Conv(8->16,5,pad2)-ReLU-MaxPool-FC(784->64)-ReLU-FC(64->10)\n";
        std::cout << "[train] epochs=" << opt.epochs << " batch=" << opt.batch_size
                  << " lr=" << opt.lr << " weight_decay=" << opt.weight_decay << "\n";

        for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), rng);
            float epoch_loss_sum = 0.0f;
            int epoch_correct = 0;
            int epoch_seen = 0;
            int batch_count = 0;

            // gentle LR decay improves final accuracy while keeping defaults simple
            float lr_epoch = opt.lr;
            if (epoch > opt.epochs * 2 / 3) lr_epoch *= 0.25f;
            else if (epoch > opt.epochs / 2) lr_epoch *= 0.5f;

            for (int start = 0; start < train.n; start += opt.batch_size) {
                int B = std::min(opt.batch_size, train.n - start);
                copy_batch(train, order, start, B, batch);
                forward(net, batch.x.p, B);
                backward(net, batch.x.p, batch.y.p, B);
                ++step;
                adam_update(net, lr_epoch, step, opt.weight_decay);
                CUDA_CHECK(cudaDeviceSynchronize());

                auto [loss, acc] = read_loss_acc(net, B);
                epoch_loss_sum += loss;
                epoch_correct += int(std::round(acc * B));
                epoch_seen += B;
                ++batch_count;

                if (batch_count % 100 == 0) {
                    std::cout << "  epoch " << epoch << " batch " << batch_count
                              << " loss=" << loss << " acc=" << acc * 100.0f << "%\n";
                }
            }

            float train_loss = epoch_loss_sum / float(batch_count);
            float train_acc = float(epoch_correct) / float(epoch_seen);
            float test_acc = evaluate(net, test, batch, opt.batch_size);
            std::cout << "[epoch " << epoch << "] train_loss=" << train_loss
                      << " train_acc=" << train_acc * 100.0f << "%"
                      << " test_acc=" << test_acc * 100.0f << "%"
                      << " lr=" << lr_epoch << "\n";
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        float final_acc = evaluate(net, test, batch, opt.batch_size);
        std::cout << "[done] final_test_acc=" << final_acc * 100.0f << "%"
                  << " elapsed_sec=" << sec << "\n";
        std::cout << "[note] On a normal CUDA GPU, the default 8 epochs should usually reach about 98%+ on MNIST.\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
