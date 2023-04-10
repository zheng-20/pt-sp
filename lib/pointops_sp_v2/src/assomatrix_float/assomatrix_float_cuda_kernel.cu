#include "../cuda_utils.h"
#include "assomatrix_float_cuda_kernel.h"


// input: val_c (n, ks) idx_c (n, ks) cid (m, 1)
// output: idx (m, n) cnt (m, 1)
__global__ void assomatrix_float_cuda_kernel(int n, int m, int ks, const int *__restrict__ offset, const int *__restrict__ sp_offset, const float *__restrict__ val_c, const int *__restrict__ idx_c, const int *__restrict__ cid, float *__restrict__ idx, int *__restrict__ cnt) {
    // int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    //new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //cid += bs_idx * m * 1 + pt_idx * 1;
    cid += pt_idx * 1;
    //xyz += bs_idx * n * 3;
    // val_c += bs_idx * n * ks;  // add
    // idx_c += bs_idx * n * ks;  // add
    idx += pt_idx * n;
    cnt += pt_idx * 1;     // count number of points located in one superpoint
    int bt_idx = get_bt_idx(pt_idx, sp_offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    //float new_x = new_xyz[0];
    //float new_y = new_xyz[1];
    //float new_z = new_xyz[2];
    int cluster_id = cid[0];

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    //float tmpi[20];
    //for(int i = 0; i < n; i++){
    //    tmpi[i] = 0.0;
    //}
    int num = 0;
    for(int k = start; k < end; k++){
        for (int j = 0; j < ks; j++) {
            int id = idx_c[k * ks + j]; // cluster id of i-th point
            float val = val_c[k * ks + j];
            if (id == cluster_id) {
                //tmpi[k] = val;
                idx[k] = val;
                num++;
            }
        }
    }
    //for(int i = 0; i < n; i++){
    //    idx[i] = tmpi[i];
    //}
    cnt[0] = num;
    //delete []best;
    //delete []besti;
}


void assomatrix_float_cuda_launcher(int n, int m, int ks, const int *offset, const int *sp_offset, const float *val_c, const int *idx_c, const int *cid, float *idx, int *cnt, cudaStream_t stream) {
    // param val_c: (n, ks)
    // param idx_c: (n, ks)
    // param cid: (m, 1)
    // param idx: (m, n)
    // param cnt: (m, 1)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    assomatrix_float_cuda_kernel<<<blocks, threads, 0, stream>>>(n, m, ks, offset, sp_offset, val_c, idx_c, cid, idx, cnt);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
