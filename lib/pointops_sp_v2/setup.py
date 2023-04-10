#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    name='pointops_sp_v2',
    ext_modules=[
        CUDAExtension('pointops_sp_v2_cuda', [
            'src/pointops_sp_v2_api.cpp',

            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',

            'src/knnquerycluster/knnquerycluster_cuda.cpp',
            'src/knnquerycluster/knnquerycluster_cuda_kernel.cu',

            'src/assomatrix_label/assomatrix_label_cuda.cpp',
            'src/assomatrix_label/assomatrix_label_cuda_kernel.cu',

            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',

            'src/assomatrix_float/assomatrix_float_cuda.cpp',
            'src/assomatrix_float/assomatrix_float_cuda_kernel.cu'
        ],
                        extra_compile_args={'cxx': ['-g'],
                                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
    