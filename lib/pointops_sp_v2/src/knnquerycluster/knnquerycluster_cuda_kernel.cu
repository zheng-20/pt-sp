#include "../cuda_utils.h"
#include "knnquerycluster_cuda_kernel.h"

// __device__ int get_bt_idx(int idx, const int *offset)
// {
//     int i = 0;
//     while (1)
//     {
//         if (idx < offset[i])
//             break;
//         else
//             i++;
//     }
//     return i;
// }

// Note: n is the clusters, m is the points
// Note: m >> n
// Note: nsample <= 200

// input: xyz (n, 3) xyz_idx (n) new_xyz (m, 3)
// output: idx (m, nsample) dist2 (m, nsample)
// offset: (B) sp_offset: (B)
__global__ void knnquerycluster_cuda_kernel(int n, int m, int nsample, const float *__restrict__ xyz, const int *__restrict__ xyz_idx, const float *__restrict__ new_xyz, const int *__restrict__ offset, const int *__restrict__ sp_offset, int *__restrict__ idx, int *__restrict__ idx_abs, float *__restrict__ dist2) {
    // int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    new_xyz += pt_idx * 3;
    // xyz += bs_idx * n * 3;
    // xyz_idx += bs_idx * n * 1;  // add xyz_idx initial position
    idx += pt_idx * nsample;
    idx_abs += pt_idx * nsample;
    dist2 += pt_idx * nsample;
    int bt_idx = get_bt_idx(pt_idx, offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = sp_offset[bt_idx - 1];
    int end = sp_offset[bt_idx];

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    double best[200];
    int besti[200];
    for(int i = 0; i < nsample; i++) {
        best[i] = 1e40;
        besti[i] = start;
    }
    for(int k = start; k < end; k++){
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        for(int j = 0; j < nsample; j++){
            if(d2 < best[j]){
                for(int i = nsample - 1; i > j; i--){
                    best[i] = best[i - 1];
                    besti[i] = besti[i - 1];
                }
                best[j] = d2;
                besti[j] = k;
                break;
            }
        }
    }
    for(int i = 0; i < nsample; i++){
        //idx[i] = besti[i];
        idx[i] = xyz_idx[besti[i]];
        idx_abs[i] = besti[i];
        dist2[i] = best[i];
    }
    //delete []best;
    //delete []besti;
}


void knnquerycluster_cuda_launcher(int n, int m, int nsample, const float *xyz, const int *xyz_idx, const float *new_xyz, const int *offset, const int *sp_offset, int *idx, int *idx_abs, float *dist2, cudaStream_t stream) {  // add xyz_idx
    // param new_xyz: (m, 3)
    // param xyz: (n, 3)
    // param xyz_idx: (n)     // add
    // param offset: (B)
    // param sp_offset: (B)
    // param idx: (m, nsample)
    // param idx_abs: (m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knnquerycluster_cuda_kernel<<<blocks, threads, 0, stream>>>(n, m, nsample, xyz, xyz_idx, new_xyz, offset, sp_offset, idx, idx_abs, dist2);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
