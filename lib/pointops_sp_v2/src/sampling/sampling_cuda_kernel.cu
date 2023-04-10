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


// ------------------------------ gathering cluster -----------------------------------
// input: points(c, n) idx(m) idx_3d(m, k)
// output: out(c, m)
__global__ void gathering_cluster_forward_cuda_kernel(int c, int n, int m, int k, const float *points, const int *idx, const int *idx_3d, float *out)
{
    for (int l = blockIdx.x; l < c; l += gridDim.x)
    {
        for (int j = threadIdx.x; j < m; j += blockDim.x)
        {
            int tmp = idx[j];   //add
            int a = idx_3d[j * k + tmp]; // add
            out[l * m + j] = points[l * n + a];
        }
    }
}

// input: grad_out(c, m) idx(m) idx_3d(m, k)
// output: grad_points(c, n)
__global__ void gathering_cluster_backward_cuda_kernel(int c, int n, int m, int k, const float *grad_out, const int *idx, const int *idx_3d, float *grad_points)
{
    for (int l = blockIdx.x; l < c; l += gridDim.x)
    {
        for (int j = threadIdx.x; j < m; j += blockDim.x)
        {
            int tmp = idx[j];   // add
            int a = idx_3d[j * k + tmp];    // add
            atomicAdd(grad_points + l * n + a, grad_out[l * m + j]);
        }
    }
}

void gathering_cluster_forward_cuda_launcher(int c, int n, int m, int k, const float *points, const int *idx, const int *idx_3d, float *out)
{
    gathering_cluster_forward_cuda_kernel<<<c, opt_n_threads(m), 0>>>(c, n, m, k, points, idx, idx_3d, out);
}

void gathering_cluster_backward_cuda_launcher(int c, int n, int m, int k, const float *grad_out, const int *idx, const int *idx_3d, float *grad_points)
{
    gathering_cluster_backward_cuda_kernel<<<c, opt_n_threads(m), 0>>>(c, n, m, k, grad_out, idx, idx_3d, grad_points);
}