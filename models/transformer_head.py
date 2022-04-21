from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F


# [batch_size, 3, h, w]                          [1, 3, 416, 416]
# [batch_size, hidden_dim, h//4, w//4]           [1, 96, 104, 104]
# [batch_size, hidden_dim*2, h//8, w//8]         [1, 192, 52, 52]
# [batch_size, hidden_dim*4, h//16, w//16]       [1, 384, 26, 26]
# [batch_size, hidden_dim*8, h//32, w//32]       [1, 768, 13, 13]

class BaseLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransformerSegHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.relu = nn.GELU()
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ch_list = list(reversed(self.args.swin_out_channel))
        self.layer_list = nn.ModuleList()
        for id in range(len(self.ch_list) - 1):
            self.layer_list.append(
                BaseLayer(
                    dim_in = self.ch_list[id] + self.ch_list[id+1],
                    dim_out = self.ch_list[id+1],
                )
            )
        self.conv_end = nn.Conv2d(self.ch_list[-1], self.args.seg_class_num, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_list):
        '''
        x_list: [b, 768, 13, 13] -- [b, 384, 26, 26] -- [b, 192, 52, 52] -- [b, 96, 104, 104]
        output: [b, seg_class_num, 104, 104]
        '''
        out = x_list[0]
        for id in range(len(self.ch_list) - 1):
            out = self.up_sample(out)
            out = torch.cat([out, x_list[id+1]], dim=1)
            out = self.layer_list[id](out)
        seg_logit = self.conv_end(out)
        return seg_logit


class TransformerPSHead(nn.Module):
    '''
    用于库位检测的 head，角点 pair 检测 && 角点方向、坐标检测
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ch = self.args.swin_out_channel[-1]
        self.conv_slot = nn.Conv2d(self.ch, self.args.slot_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mark = nn.Conv2d(self.ch, self.args.mark_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_list):
        '''
        x_list: [b, 768, 13, 13] -- [b, 384, 26, 26] -- [b, 192, 52, 52] -- [b, 96, 104, 104]
        slot_out: (b, 6, self.args.feature_map_size, self.args.feature_map_size) && channel is [x1, y1, x2, y2, confidence, slot_use] 记录每个 grid 中心点到两个角点的 vector，长度归一化到（0，1）
        mark_out: (b, 5, self.args.feature_map_size, self.args.feature_map_size) && channel is [x, y, dx, dy, confidence] 记录每个角点的坐标和方向，confidence, x, y, from（0，1), dx, dy, from (-1, 1)
        '''
        slot_out = self.sigmoid(self.conv_slot(x_list[0]))
        mark_out = self.sigmoid(self.conv_mark(x_list[0]))
        return slot_out, mark_out



