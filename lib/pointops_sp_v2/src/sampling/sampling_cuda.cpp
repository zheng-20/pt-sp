#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "sampling_cuda_kernel.h"

extern THCState *state;

// --------------------------- gathering ----------------------
void gathering_forward_cuda(int c, int n, int m, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();
    gathering_forward_cuda_launcher(c, n, m, points, idx, out);
}

void gathering_backward_cuda(int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
{

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_points = grad_points_tensor.data<float>();
    gathering_backward_cuda_launcher(c, n, m, grad_out, idx, grad_points);
}