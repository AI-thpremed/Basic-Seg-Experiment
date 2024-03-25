import torch
import torch.nn as nn

class MsGCS(nn.Module):
    def __init__(self,F_g,F_l,F_int,size):
        super(MsGCS, self).__init__()
        print("=============== MsGCS ===============")

        self.Conv = nn.Sequential(
            nn.Conv2d(F_g+F_l,F_int,kernel_size=1),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.rel_h = nn.Parameter(torch.randn([1, 1, size[0], 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, 1, 1, size[1]]), requires_grad=True)
        self.bn = nn.BatchNorm2d(1)
        self.active = nn.Sigmoid()

    def forward(self,g,x):
        x_Multi_Scale = self.Conv(torch.cat([g,x],dim=1))
        content_position = self.rel_h+self.rel_w
        x_att_multi_scale = self.active(self.bn(content_position*x_Multi_Scale))

        return x_att_multi_scale*x
