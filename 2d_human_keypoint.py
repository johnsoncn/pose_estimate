# -*- coding: utf-8 -*-
""" 
@Time    : 2022/11/10 17:13
@Author  : Johnson
@FileName: 2d_human_keypoint.py
"""

# git clone https://github.com/open-mmlab/mmpose.git  会下载到 /home/xxx/mmpose


import os, sys
import cv2
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results
from mmdet.apis import inference_detector, init_detector
import matplotlib.pyplot as plt

def show_img_from_path(img_path):
    img = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

def show_img_from_array(img):
    # matplotlib RGB，cv2 BGR
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()



class PoseEstimate:

    def __init__(self):

        # 目标检测模型
        self.det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
        self.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

        # 人体姿态估计模型
        self.pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

        # detection model
        self.det_model = init_detector(self.det_config, self.det_checkpoint)
        # pose estimation model
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint)

    def object_detector(self, img_path):

        # detection：coco格式 xyxy
        mmdet_results = inference_detector(self.det_model, img_path)
        # print(mmdet_results)
        print(len(mmdet_results))
        print(mmdet_results[0].shape)
        print(mmdet_results[1].shape)

if __name__ == "__main__":
    retval = os.getcwd()
    print(retval)
    os.chdir('/home/dingchaofan//mmpose')
    retval = os.getcwd()
    print(retval)

    # wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/images/multi-person.jpeg -O data/multi-person.jpeg
    img_path = '/home/dingchaofan/pose_estimate/data/multi-person.jpeg'

    PE = PoseEstimate()
    PE.object_detector(img_path=img_path)

