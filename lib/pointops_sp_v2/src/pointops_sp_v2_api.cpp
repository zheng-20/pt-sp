#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "sampling/sampling_cuda_kernel.h"
#include "knnquerycluster/knnquerycluster_cuda_kernel.h"
#include "assomatrix_label/assomatrix_label_cuda_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gathering_forward_cuda", &gathering_forward_cuda, "gathering_forward_cuda");
    m.def("gathering_backward_cuda", &gathering_backward_cuda, "gathering_backward_cuda");

    m.def("knnquerycluster_cuda", &knnquerycluster_cuda, "knnquerycluster_cuda");
    
    m.def("assomatrix_label_cuda", &assomatrix_label_cuda, "assomatrix_label_cuda");
}