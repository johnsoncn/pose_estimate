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

    def pose_detector(self, img_path, save_to):

        # len(mmdet_results)：80  MS COCO目标检测数据集 80 个类别每个预测框的以下信息： detection：coco格式 xyxy
        mmdet_results = inference_detector(self.det_model, img_path)
        person_results = process_mmdet_results(mmdet_results, cat_id=1) # ms coco 行人ID为1，提取所有行人检测结果

        # `top_down` pose estimate
        # pose_results = {'bbox': [x,y,x,y,confidence], 'keypoints':[x,y,confidence]}
        pose_results, returned_outputs = inference_top_down_pose_model(self.pose_model,
                                                                       img_path,
                                                                       person_results,
                                                                       bbox_thr=0.3,
                                                                       format='xyxy',
                                                                       dataset='TopDownCocoDataset')
        # 姿态估计结果可视化处理
        vis_result = vis_pose_result(self.pose_model,
                                     img_path,
                                     pose_results,
                                     radius=8,
                                     thickness=3,
                                     dataset='TopDownCocoDataset',
                                     show=False)
        print(vis_result)
        cv2.imwrite(save_to, vis_result)
        print(f'[INFO] save to {save_to}')


if __name__ == "__main__":
    retval = os.getcwd()
    print(retval)
    os.chdir('/home/dingchaofan/mmpose')
    retval = os.getcwd()
    print(retval)

    # wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/images/multi-person.jpeg -O data/multi-person.jpeg
    img_path = '/home/dingchaofan/pose_estimate/data/multi-person.jpeg'
    save_to = '/home/dingchaofan/pose_estimate/outputs/B1_multi_human.jpg'

    PE = PoseEstimate()
    PE.pose_detector(img_path=img_path, save_to=save_to)

