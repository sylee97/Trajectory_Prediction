# Tianyang Zhao
# Architecture Details for CVPR 19 paper: Multi-Agent Tensor Fusion for Contextual Trajectory Prediction
# Link: http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Multi-Agent_Tensor_Fusion_for_Contextual_Trajectory_Prediction_CVPR_2019_paper.html
# ArXiv Link: https://arxiv.org/abs/1904.04776
# Feel free to contact: zhaotianyang@pku.edu.cn; Please include 'MATF' in the email title  

## Code for basement helper functions

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),) 

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
