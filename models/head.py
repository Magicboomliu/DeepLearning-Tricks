import torch
from torch import nn
import torch.nn.functional as F


# torch.Size([1, 18, 104, 104])
# torch.Size([1, 36, 52, 52])
# torch.Size([1, 72, 26, 26])
# torch.Size([1, 144, 13, 13])

class ConcatHead(nn.Module):
    '''
    把 backbone 的多尺度输出合在一起
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x_list):
        upsample_list = []
        for id, x in enumerate(x_list):
            upsample_list.append(F.interpolate(x, scale_factor=int(1<<id), mode="bilinear") )
        upsample_out = torch.cat(upsample_list, dim=1)

        downsample_list = []
        for id, x in enumerate(reversed(x_list)):
            downsample_list.append(F.interpolate(x, scale_factor=1.0/(1<<id), mode="bilinear") )
        downsample_out = torch.cat(downsample_list, dim=1)
        return upsample_out, downsample_out

class ConcatHeadV1(nn.Module):
    '''
    把 backbone 的多尺度输出合在一起
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x_list):
        upsample_list = []
        for id, x in enumerate(x_list):
            upsample_list.append(F.interpolate(x, scale_factor=int(1<<id), mode="bilinear") )
        upsample_out = torch.cat(upsample_list, dim=1)
        downsample_out = F.interpolate(upsample_out, scale_factor=1.0/(1<<(len(x_list) -1)), mode="bilinear")
        return upsample_out, downsample_out

class SegHead(nn.Module):
    '''
    用于分割的 head
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ch = sum(self.args.hrn_out_channel)
        self.conv1 = nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = args.norm_layer(self.ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.ch, self.args.seg_class_num, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.relu(self.norm(self.conv1(x)))
        out = self.conv2(out)
        return out

class PSHead(nn.Module):
    '''
    用于库位检测的 head，角点 pair 检测 && 角点方向、坐标检测
    slot: (6, self.args.feature_map_size, self.args.feature_map_size) && channel is [x1, y1, x2, y2, confidence, slot_use] 记录每个 grid 中心点到两个角点的 vector，长度归一化到（0，1）
    mark: (5, self.args.feature_map_size, self.args.feature_map_size) && channel is [x, y, dx, dy, confidence] 记录每个角点的坐标和方向，confidence, x, y, from（0，1), dx, dy, from (-1, 1)
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ch = sum(self.args.hrn_out_channel)
        self.conv_slot = nn.Conv2d(self.ch, self.args.slot_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mark = nn.Conv2d(self.ch, self.args.mark_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        slot_out = self.activate(self.conv_slot(x))
        mark_out = self.activate(self.conv_mark(x))
        return slot_out, mark_out


class PSHeadV2(nn.Module):
    '''
    用于库位检测的 head，角点 pair 检测 && 角点方向、坐标检测
    slot: (6, self.args.feature_map_size, self.args.feature_map_size) && channel is [x1, y1, x2, y2, confidence, slot_use] 记录每个 grid 中心点到两个角点的 vector，长度归一化到（0，1）
    mark: (5, self.args.feature_map_size, self.args.feature_map_size) && channel is [x, y, dx, dy, confidence] 记录每个角点的坐标和方向，confidence, x, y, from（0，1), dx, dy, from (-1, 1)
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ch = sum(self.args.hrn_out_channel)
        self.conv_slot = nn.Conv2d(self.ch, self.args.slot_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mark = nn.Conv2d(self.ch, self.args.mark_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        slot_logit = self.conv_slot(x)
        mark_logit = self.conv_mark(x)
        slot_out = self.activate(slot_logit)
        mark_out = self.activate(mark_logit)
        return slot_out, mark_out, slot_logit, mark_logit
