#include "../cuda_utils.h"
#include "assomatrix_label_cuda_kernel.h"


// __device__ int get_bt_idx_1(int idx, const int *offset)
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

// input: idx_c (n, ks) lab (n, 1) cid (m, 1)
// output: idx (m, n) cnt (m, 1) clab (m, class)
__global__ void assomatrix_label_cuda_kernel(int n, int m, int ks, int category, const int *__restrict__ offset, const int *__restrict__ sp_offset, const int *__restrict__ idx_c, const int *__restrict__ lab, const int *__restrict__ cid, int *__restrict__ idx, int *__restrict__ cnt, int *__restrict__ clab) {
    // int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    //new_xyz += bs_idx * m * 3 + pt_idx * 3;
    //cid += bs_idx * m * 1 + pt_idx * 1;
    cid += pt_idx * 1;
    //xyz += bs_idx * n * 3;
    // idx_c += bs_idx * n * ks;  // add
    // lab += bs_idx * n * 1; // add

    idx += pt_idx * n;
    cnt += pt_idx * 1;     // count number of points located in one superpoint
    clab += pt_idx * category;
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
    //int tmpi[20000];
    //for(int i = 0; i < n; i++){
    //    tmpi[i] = 0;
    //}
    int num = 0;
    for(int k = start; k < end; k++){
        int k_lab = lab[k];
        for (int j = 0; j < ks; j++) {
            int id = idx_c[k * ks + j]; // cluster id of i-th point
            if (id == cluster_id) {
                //tmpi[k] = 1;
                idx[k] = 1;
                clab[k_lab]++;
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


void assomatrix_label_cuda_launcher(int n, int m, int ks, int category, const int *offset, const int *sp_offset, const int *idx_c, const int *lab, const int *cid, int *idx, int *cnt, int *clab, cudaStream_t stream) {
    // param new_xyz: (m, 3)
    // param xyz: (n, 3)
    // param idx: (m, n)
    // param cnt: (m, 1)
    // param clab: (m, class)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    assomatrix_label_cuda_kernel<<<blocks, threads, 0, stream>>>(n, m, ks, category, offset, sp_offset, idx_c, lab, cid, idx, cnt, clab);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
