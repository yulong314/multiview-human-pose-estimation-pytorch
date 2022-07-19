# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size[0]*size[1], size[0]*size[1]))
        self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        output = torch.matmul(input_reshape, self.weight)
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, pairs, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        self.CAMNUM = cfg.DATASET.CAMNUM
        self.pairs = pairs
        self.NUM_NEIBORHOOD = len(pairs[0]) - 1
        
        NUM_NETS = self.CAMNUM * self.NUM_NEIBORHOOD
        size = cfg.NETWORK.HEATMAP_SIZE
        
        y = 1  / (2 + self.NUM_NEIBORHOOD)
        x = 2 * y

        self.weights = [x] + [y] * (self.NUM_NEIBORHOOD)
        # self.weights = weights
        # assert sum(self.weights) == 1
        self.aggre = nn.ModuleList()
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size))
        # self.aggre = [[ChannelWiseFC(size) for i in range(self.NUM_NEIBORHOOD )] for j in range(self.CAMNUM)]

    def sort_views(self, target, all_views):
        indicator = [target is item for item in all_views]
        new_views = [target.clone()]
        for i, item in zip(indicator, all_views):
            if not i:
                new_views.append(item.clone())
        return new_views

    def fuse_with_weights(self, views):
        target = torch.zeros_like(views[0])
        for v, w in zip(views, self.weights):
            target += v * w
        return target

    def forward(self, inputs):
        index = 0
        outputs = []
        nviews = len(inputs)

        index = 0
        for i in range (len(self.pairs)):
            warped = [inputs[i]]
            for j in range(self.NUM_NEIBORHOOD):
                fc = self.aggre[index]
                fc_output = fc(inputs[i])
                warped.append(fc_output)
        #     pass
        # for i in range(nviews):
        #     sorted_inputs = self.sort_views(inputs[i], inputs)
        #     warped = [sorted_inputs[0]]
        #     for j in range(1, self.NUM_NEIBORHOOD + 1):
        #         fc = self.aggre[index]
        #         fc_output = fc(sorted_inputs[j])
        #         warped.append(fc_output)
                index += 1
            output = self.fuse_with_weights(warped)
            outputs.append(output)
        return outputs


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, Aggre, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        self.resnet = PoseResNet
        self.aggre_layer = Aggre

    def forward(self, views):
        if isinstance(views, list):
            single_views = []
            for view in views:
                heatmaps = self.resnet(view)
                single_views.append(heatmaps)
            multi_views = []
            if self.config.NETWORK.AGGRE:
                multi_views = self.aggre_layer(single_views)
            return single_views, multi_views
        else:
            return self.resnet(views)


def get_multiview_pose_net(mobilenetv2, CFG,pairs):
    Aggre = Aggregation(CFG,pairs)
    model = MultiViewPose(mobilenetv2, Aggre, CFG)
    return model
