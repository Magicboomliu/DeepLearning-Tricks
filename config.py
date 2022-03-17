import argparse
import torch
import torch.nn as nn
import numpy as np
from models.hrnet_mrsa import hrnet18
import math
#from model.norm_layer import BaseGroupNorm

def get_parser_for_training():
    parser = argparse.ArgumentParser()
    #-------------------------------------------------------#
    #----------------------  basic  ------------------------#
    #-------------------------------------------------------#
    parser.add_argument("--num_work", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_org_size_ps2", type=int, default=(600,600))   # ps2 dataset is 600
    parser.add_argument("--image_size", type=int, default=(416,416))      # the input size of our model
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--color_list", default=np.array(
        [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0]]
        )
    )
    parser.add_argument("--post_color_list", default=np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0]]
        )
    )
    parser.add_argument("--need_img_norm", default=True)
    parser.add_argument("--image_mean", default=np.array([0.51, 0.51, 0.51]))
    parser.add_argument("--image_std", default=np.array([0.294, 0.297, 0.295]))
    #-------------------------------------------------------#
    #-------------------  model basic  ---------------------#
    #-------------------------------------------------------#
    parser.add_argument("--pretrain", default=True)
    parser.add_argument("--norm_layer", default=nn.BatchNorm2d)
    parser.add_argument("--backbone_type", type=str, default="hrn18")
    parser.add_argument("--hrn_out_channel", type=list, default=[18, 36, 72, 144])
    parser.add_argument("--seg_scale_factor", type=int, default=4)
    parser.add_argument("--slot_channel", type=int, default=6)  # [x1, y1, x2, y2, confidence, slot_use]
    parser.add_argument("--mark_channel", type=int, default=5)  # [x, y, dx, dy, confidence]
    parser.add_argument("--seg_class_num", type=int, default=8)
    parser.add_argument("--feature_map_size", type=int, default=(13,13))
    #-------------------------------------------------------#
    #------------------  transformer basic  ----------------#
    #-------------------------------------------------------#
    parser.add_argument("--swin_out_channel", type=list, default=[64, 128, 256, 512])   # [96, 192, 384, 768]
    parser.add_argument("--swin_downscaling_factors", type=list, default=(4, 2, 2, 2))
    parser.add_argument("--swin_layer_num_list", type=list, default=(2, 2, 6, 2))
    parser.add_argument("--swin_window_size", type=int, default=13)
    parser.add_argument("--swin_dim", type=int, default=64)
    parser.add_argument("--swin_head_dim", type=int, default=32)
    #parser.add_argument("--drop_rate", type=float, default=0.1)
    #-------------------------------------------------------#
    #-----------------  train  basic  ----------------------#
    #-------------------------------------------------------#
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--start_epoch", type=int, default=0)
    #-------------------------------------------------------#
    #--------------  post-processing basic  ----------------#
    #-------------------------------------------------------#
    parser.add_argument("--nms_threshold", type=float, default=0.3)
    parser.add_argument("--mark_nms_bound", type=float, default=0.5)  # 目前图片是13*13的grid，把角点变成一个框，然后nms，1.0 代表一个 feature map grid 的距离
    parser.add_argument("--slot_mark_match_bound", type=float, default=0.5) # 目前图片是13*13的grid，单位长度为一个grid,小于这个距离时，slot-pair 和 marking-point 可以匹配
    #-------------------------------------------------------#
    #------------------  evaluation basic ------------------#
    #-------------------------------------------------------#
    parser.add_argument("--tp_mark_distance_threshold", type=float, default=0.02) # 整张图片的百分比
    parser.add_argument("--tp_mark_angle_threshold", type=float, default=10.0 * math.pi/180)  # 10度
    #-------------------------------------------------------#
    
    parser.add_argument("--random_move",default=True)
    return parser
