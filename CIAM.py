import torch
from torch import nn 
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

class CIA(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(CIA, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        # self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        self.scale3 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        # self.process3 = nn.Sequential(
        #                             BatchNorm2d(branch_planes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        #                             )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 4, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]  # 1 1024 16 16
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))  # 1*1卷积 1 256 16 16
        x_list.append(self.process1((F.interpolate(self.scale1(x),  # 3*3卷积(平均池化3 + 1*1卷积 +上采样) 1 256 16 16
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),  # 平均池化 5    1 256 16 16
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))  # 3*3卷积
        x_list.append(self.process3((F.interpolate(self.scale3(x),  # 自适应平均池化 1 256 16 16
                        size=[height, width],
                        mode='bilinear')+x_list[2])))  # 3*3卷积
        # x_list.append(self.process4((F.interpolate(self.scale4(x),
        #                 size=[height, width],
        #                 mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)  # cat之后在经过1*1卷积，shortcut对原始进行1*1卷积
        return out

                    # width_output = x.shape[-1] // 8
                    # height_output = x.shape[-2] // 8

                # x = F.interpolate(
                #         self.spp(self.layer5(self.relu(x))),
                #         size=[height_output, width_output],
                #         mode='bilinear')

                    #self.spp = CIA(planes * 16, spp_planes, planes * 4)