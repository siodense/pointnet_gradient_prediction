import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet2_utils import PointNetSetAbstraction


class PN_arm(nn.Module):
    def __init__(self,num_class,in_channel=3):
        super(PN_arm, self).__init__()
        if in_channel>3:
            self.additional_channels = True
        else:
            self.additional_channels= False

        self.sa1 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=in_channel, mlp=[1024, 1024,2048, 2048,1024,512,64], group_all=True)

        self.lin1 = nn.Linear(64, 256)

        self.gn1=nn.GroupNorm(16,64)
        self.gn2=nn.GroupNorm(16,256)

        self.ss_flin=nn.Linear(256,num_class)

    def forward(self,xyz):
        B, _, _ = xyz.shape
        if self.additional_channels:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz,norm)
        pn_out = F.relu(self.gn2(self.lin1(self.gn1(l1_points.reshape(B, -1)))))

        out=self.ss_flin(pn_out)

        return out, None
