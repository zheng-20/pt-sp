#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "assomatrix_label_cuda_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void assomatrix_label_cuda(int n, int m, int ks, int category, at::Tensor offset_tensor, at::Tensor sp_offset_tensor, at::Tensor idx_c_tensor, at::Tensor lab_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor, at::Tensor cnt_tensor, at::Tensor clab_tensor) //
{
    CHECK_INPUT(idx_c_tensor);
    CHECK_INPUT(cid_tensor);

    const int *offset = offset_tensor.data_ptr<int>();
    const int *sp_offset = sp_offset_tensor.data_ptr<int>();
    const int *idx_c = idx_c_tensor.data<int>();
    const int *lab = lab_tensor.data<int>();
    const int *cid = cid_tensor.data<int>();
    int *idx = idx_tensor.data<int>();
    int *cnt = cnt_tensor.data<int>();
    int *clab = clab_tensor.data<int>();

    // cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); // Adapt to torch1.5 and above

    assomatrix_label_cuda_launcher(n, m, ks, category, offset, sp_offset, idx_c, lab, cid, idx, cnt, clab, stream);
}
