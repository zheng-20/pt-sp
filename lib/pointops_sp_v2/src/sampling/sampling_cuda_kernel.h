#ifndef _SAMPLING_CUDA_KERNEL
#define _SAMPLING_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void gathering_forward_cuda(int c, int n, int m, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);
void gathering_backward_cuda(int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void gathering_forward_cuda_launcher(int c, int n, int m, const float *points, const int *idx, float *out);
void gathering_backward_cuda_launcher(int c, int n, int m, const float *grad_out, const int *idx, float *grad_points);

#ifdef __cplusplus
}
#endif
#endif