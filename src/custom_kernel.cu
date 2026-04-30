/*
 * custom_kernel.cu
 *
 * Implements a fused GELU activation + dropout kernel in CUDA.
 * Registered into PyTorch via torch.utils.cpp_extension.
 *
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Fusing activation + dropout into a single kernel avoids a full
 * round-trip to global memory between the two ops — a classic
 * GPU optimization technique.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <curand_kernel.h>
#include <math.h>

#define BLOCK_SIZE 256

// ─── Device helper: GELU ────────────────────────────────────────────────────

__device__ __forceinline__ float gelu_f(float x) {
    const float k0 = 0.7978845608f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ─── Forward: fused GELU + dropout ──────────────────────────────────────────

__global__ void fused_gelu_dropout_forward_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
          bool*  __restrict__ mask,
    int    N,
    float  p_drop,          // dropout probability
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Per-thread RNG state (cheap: init only the state we need)
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float x    = input[idx];
    float gelu = gelu_f(x);

    // Bernoulli drop
    float u    = curand_uniform(&state);
    bool  keep = (u >= p_drop);
    mask[idx]  = keep;

    // Scale by 1/(1-p) so expected value is preserved (inverted dropout)
    float scale      = keep ? 1.0f / (1.0f - p_drop) : 0.0f;
    output[idx]      = gelu * scale;
}

// ─── Backward: gradient through fused op ────────────────────────────────────

__device__ __forceinline__ float gelu_grad(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float inner  = k0 * (x + k1 * x * x * x);
    float tanh_v = tanhf(inner);
    float sech2  = 1.0f - tanh_v * tanh_v;
    float dg_dx  = 0.5f * (1.0f + tanh_v)
                 + 0.5f * x * sech2 * k0 * (1.0f + 3.0f * k1 * x * x);
    return dg_dx;
}

__global__ void fused_gelu_dropout_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const bool*  __restrict__ mask,
          float* __restrict__ grad_input,
    int   N,
    float p_drop
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float go    = grad_output[idx];
    float x     = input[idx];
    float dg    = gelu_grad(x);
    float scale = mask[idx] ? 1.0f / (1.0f - p_drop) : 0.0f;
    grad_input[idx] = go * dg * scale;
}

// ─── C++ / PyTorch wrapper ───────────────────────────────────────────────────

std::vector<torch::Tensor> fused_gelu_dropout_forward(
    torch::Tensor input,
    float         p_drop,
    unsigned long long seed
) {
    TORCH_CHECK(input.is_cuda(),              "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int N = input.numel();
    auto output = torch::empty_like(input);
    auto mask   = torch::empty({N}, torch::TensorOptions()
                    .dtype(torch::kBool).device(input.device()));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_gelu_dropout_forward_kernel<<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mask.data_ptr<bool>(),
        N, p_drop, seed
    );

    return {output, mask};
}

torch::Tensor fused_gelu_dropout_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor mask,
    float         p_drop
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    int N          = input.numel();
    auto grad_input = torch::empty_like(input);

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_gelu_dropout_backward_kernel<<<grid, BLOCK_SIZE>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        mask.data_ptr<bool>(),
        grad_input.data_ptr<float>(),
        N, p_drop
    );

    return grad_input;
}

// ─── Pybind11 bindings ───────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused GELU + Dropout CUDA kernel";
    m.def("forward",  &fused_gelu_dropout_forward,
          "Fused GELU+Dropout forward (CUDA)");
    m.def("backward", &fused_gelu_dropout_backward,
          "Fused GELU+Dropout backward (CUDA)");
}
