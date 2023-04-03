#include "../cuda_utils.h"
#include "sampling_cuda_kernel.h"

// ------------------------------ gathering  -----------------------------------
// input: points(c, n) idx(m)
// output: out(c, m)
__global__ void gathering_forward_cuda_kernel(int c, int n, int m, const float *points, const int *idx, float *out)
{
    for (int l = blockIdx.x; l < c; l += gridDim.x)
    {
        for (int j = threadIdx.x; j < m; j += blockDim.x)
        {
            int a = idx[j];
            out[l * m + j] = points[l * n + a];
        }
    }
}

// input: grad_out(c, m) idx(m)
// output: grad_points(c, n)
__global__ void gathering_backward_cuda_kernel(int c, int n, int m, const float *grad_out, const int *idx, float *grad_points)
{
    for (int l = blockIdx.x; l < c; l += gridDim.x)
    {
        for (int j = threadIdx.x; j < m; j += blockDim.x)
        {
            int a = idx[j];
            atomicAdd(grad_points + l * n + a, grad_out[l * m + j]);
        }
    }
}

void gathering_forward_cuda_launcher(int c, int n, int m, const float *points, const int *idx, float *out)
{
    gathering_forward_cuda_kernel<<<c, opt_n_threads(m), 0>>>(c, n, m, points, idx, out);
}

void gathering_backward_cuda_launcher(int c, int n, int m, const float *grad_out, const int *idx, float *grad_points)
{
    gathering_backward_cuda_kernel<<<c, opt_n_threads(m), 0>>>(c, n, m, grad_out, idx, grad_points);
}