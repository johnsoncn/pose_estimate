# -*- coding: utf-8 -*-
""" 
@Time    : 2022/11/10 17:54
@Author  : Johnson
@FileName: test.py
"""

# 修改当前工作目录
os.chdir("/tmp")

# 查看修改后的工作目录
retval = os.getcwd()
print(retval)

# 检查 Pytorch
import torch, torchvision
print('Pytorch ', torch.__version__)
print('torchvision', torchvision.__version__)
print('CUDA 是否可用',torch.cuda.is_available())


# 检查 mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('CUDA', get_compiling_cuda_version())
print('GCC', get_compiler_version())