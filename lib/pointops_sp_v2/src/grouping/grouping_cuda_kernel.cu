#include "../cuda_utils.h"
#include "grouping_cuda_kernel.h"

// input: points(c, n) idx(m, nsample)
// output: out(c, m, nsample)
__global__ void grouping_forward_cuda_kernel(int c, int n, int m, int nsample, const float *points, const int *idx, float *out)
{
    // int batch_index = blockIdx.x;
    // points += batch_index * n * c;
    // idx += batch_index * m * nsample;
    // out += batch_index * m * nsample * c;
    // const int index = threadIdx.y * blockDim.x + threadIdx.x;
    // const int stride = blockDim.y * blockDim.x;
    // for (int i = index; i < c * m; i += stride)
    // {
    //     const int l = i / m;
    //     const int j = i % m;
    //     for (int k = 0; k < nsample; ++k)
    //     {
    //         int ii = idx[j * nsample + k];
    //         out[(l * m + j) * nsample + k] = points[l * n + ii];
    //     }
    // }

    for (int l = blockIdx.x; l < c; l += gridDim.x)
    {
        for (int j = threadIdx.x; j < m; j += blockDim.x)
        {
            for (int k = 0; k < nsample; ++k)
            {
                int ii = idx[j * nsample + k];
                out[(l * m + j) * nsample + k] = points[l * n + ii];
            }
        }
    }
}

// input: grad_out(c, m, nsample), idx(m, nsample)
// output: grad_points(c, n)
__global__ void grouping_backward_cuda_kernel(int c, int n, int m, int nsample, const float *grad_out, const int *idx, float *grad_points)
{
    // int batch_index = blockIdx.x;
    // grad_out += batch_index * m * nsample * c;
    // idx += batch_index * m * nsample;
    // grad_points += batch_index * n * c;
    // const int index = threadIdx.y * blockDim.x + threadIdx.x;
    // const int stride = blockDim.y * blockDim.x;
    // for (int i = index; i < c * m; i += stride)
    // {
    //     const int l = i / m;
    //     const int j = i % m;
    //     for (int k = 0; k < nsample; ++k)
    //     {
    //         int ii = idx[j * nsample + k];
    //         atomicAdd(grad_points + l * n + ii, grad_out[(l * m + j) * nsample + k]);
    //     }
    // }

    for (int l = blockIdx.x; l < c; l += gridDim.x)
    {
        for (int j = threadIdx.x; j < m; j += blockDim.x)
        {
            for (int k = 0; k < nsample; ++k)
            {
                int ii = idx[j * nsample + k];
                atomicAdd(grad_points + l * n + ii, grad_out[(l * m + j) * nsample + k]);
            }
        }
    }
}

void grouping_forward_cuda_launcher(int c, int n, int m, int nsample, const float *points, const int *idx, float *out)
{
    grouping_forward_cuda_kernel<<<c, opt_n_threads(m), 0>>>(c, n, m, nsample, points, idx, out);
}

void grouping_backward_cuda_launcher(int c, int n, int m, int nsample, const float *grad_out, const int *idx, float *grad_points)
{
    grouping_backward_cuda_kernel<<<c, opt_n_threads(m), 0>>>(c, n, m, nsample, grad_out, idx, grad_points);
}

