# -*- coding: utf-8 -*-
""" 
@Time    : 2022/11/10 17:54
@Author  : Johnson
@FileName: test.py
"""
import os
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


# MS COCO数据集80个类别及其编号ID对应（index从0开始，ID从1开始）
# https://blog.csdn.net/weixin_51697369/article/details/123210202
CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush')


"""
python demo/top_down_img_demo_with_mmdet.py \
        demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
        https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
        --img /home/dingchaofan/pose_estimate/data/multi-person.jpeg \
        --out-img-root outputs/johnson
        
        
# 用标注框（手动框出来）作为`top_down`算法的输入框输入，传入ms coco标注的json文件
!python demo/top_down_img_demo.py \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
        https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
        --img-root tests/data/coco/ \
        --json-file tests/data/coco/test_coco.json \
        --out-img-root outputs/B2/B2_2_gt_img
        
        
        
# 全图输入模型的视频预测：
（1）又好又快。不进行目标检测，直接将全图输入至姿态估计模型中。
（2）仅适用于视频中人体始终在画面中央的场景，仅适用于单人。
（3）扩展阅读：Mediapipe Blaze Pose单人实时人体姿态估计：https://www.bilibili.com/video/BV1dL4y1h7Q6
python demo/top_down_video_demo_full_frame_without_det.py \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py \
         https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth \
        --video-path data/solo_dance.mp4 \
        --out-video-root outputs/B2/B2_5_full_img
"""
