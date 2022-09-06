import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.checkpoint import checkpoint
from thop import profile
from torchstat import stat
from Dynamic_Convolution import Involution2d
from Dynamic_Selfattention import Second_feature_filtering_att
from CIAM import CIA


model_urls = {
    'rednet50': 'file:///C:/Users/Administrator/Desktop/rednet50.pth',
}

class Invo_Plus_Net(nn.Module):

    # Invo_filtering + context++(ASPP+CIA) + connection++

    def __init__(self, num_class=37, pretrained=False):
        super(Invo_Plus_Net, self).__init__()

        layers = [3, 4, 6, 3]
        block = Bottleneck
        block_second = Bottleneck_att
        transblock = TransBasicBlock
        # RGB image branch
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.invo1 = Involution2d(3, 64, kernel_size=7, stride=2, padding=3, reduce_ratio=4,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # use PSPNet extractors
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # depth image branch
        self.inplanes = 64
        #self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.invo1_d = Involution2d(1, 64, kernel_size=7, stride=2, padding=3, reduce_ratio=4,
                               bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        # merge branch
        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # downsamping
        self.maxpool_down4x = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpool_down8x = nn.MaxPool2d(kernel_size=8, stride=8)

        # adjust channel
        self.conv_1024_512 = nn.Conv2d(1024,512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_1024_256 = nn.Conv2d(1024,256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_1024_64 = nn.Conv2d(1024,64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_512_256 = nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_512_64 = nn.Conv2d(512,64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_256_64 = nn.Conv2d(256,64, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_64_256 = nn.Conv2d(64,256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_64_512 = nn.Conv2d(64,512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_64_1024 = nn.Conv2d(64,1024, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_256_512 = nn.Conv2d(256,512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_256_1024 = nn.Conv2d(256,1024, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_512_1024 = nn.Conv2d(512,1024, kernel_size=1, stride=1, padding=0, bias=False)


        self.inplanes = 64
        self.layer1_m = self._make_layer(block_second, 64, layers[0])
        self.layer2_m = self._make_layer(block_second, 128, layers[1], stride=2)
        self.layer3_m = self._make_layer(block_second, 256, layers[2], stride=2)
        self.layer4_m = self._make_layer(block_second, 512, layers[3], stride=2)

        # agant module
        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64*4, 64)
        self.agant2 = self._make_agant_layer(128*4, 128)
        self.agant3 = self._make_agant_layer(256*4, 256)
        # self.agant4 = self._make_agant_layer(512*4, 512)

        # ASPP + CIA
        self.aspp = ASPP(256*4)
        self.cia = CIA(256*4, 256, 256)

        #transpose layer
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)


        # final blcok
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_class, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv = nn.Conv2d(256, num_class, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def encoder(self, rgb, depth):
        rgb = self.invo1(rgb)  # 输入3 输出64 7*7卷积
        rgb = self.bn1(rgb)
        rgb = self.relu(rgb)
        depth = self.invo1_d(depth)  # 输入1 输出64
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)

        m0 = rgb + depth   # 128

        rgb = self.maxpool(rgb)
        depth = self.maxpool_d(depth)
        m = self.maxpool_m(m0)

        # block 1
        rgb = self.layer1(rgb)  # 只有3*3内卷
        depth = self.layer1_d(depth)  # 只有3*3内卷
        m = self.layer1_m(m)  # layer1_m是3*3内卷+自注意力

        m1 = m + rgb + depth  # 64

        # block 2
        rgb = self.layer2(rgb)
        depth = self.layer2_d(depth)
        m = self.layer2_m(m1)

        m2 = m + rgb + depth  # 32

        # block 3
        rgb = self.layer3(rgb)  # 1 1024 16 16
        depth = self.layer3_d(depth)
        m = self.layer3_m(m2)

        m3 = m + rgb + depth  # 16

        # connection++
        m3_up_128 = self.conv_1024_64(F.interpolate(m3, size=m0.size()[2:], mode='bilinear', align_corners=True))  # 1 64 128 128
        m3_up_64 = self.conv_1024_256(F.interpolate(m3, size=m1.size()[2:], mode='bilinear', align_corners=True))  # 1 256 64 64
        m3_up_32 = self.conv_1024_512(F.interpolate(m3, size=m2.size()[2:], mode='bilinear', align_corners=True))  # 1 512 32 32

        m2_up_128 = self.conv_512_64(F.interpolate(m2, size=m0.size()[2:], mode='bilinear', align_corners=True))  # 1 64 128 128
        m2_up_64 = self.conv_512_256(F.interpolate(m2, size=m1.size()[2:], mode='bilinear', align_corners=True))  # 1 256 64 64

        m1_up_128 = self.conv_256_64(F.interpolate(m1, size=m0.size()[2:], mode='bilinear', align_corners=True))  # 1 64 128 128

        m0_down_64 = self.conv_64_256(self.maxpool_m(m0))  # 1 256 64 64
        m0_down_32 = self.conv_64_512(self.maxpool_down4x(m0))  # 1 512 32 32
        m0_down_16 = self.conv_64_1024(self.maxpool_down8x(m0))  # 1 1024 16 16

        m1_down_32 = self.conv_256_512(self.maxpool_m(m1))  # 1 512 32 32
        m1_down_16 = self.conv_256_1024(self.maxpool_down4x(m1))  # 1 1024 16 16

        m2_down_16 = self.conv_512_1024(self.maxpool_m(m2))  # 1 1024 16 16

        m0 = m0 + m3_up_128 + m2_up_128 + m1_up_128  # channel = 64   1 64 128 128
        m1 = m1 + m0_down_64 + m3_up_64 + m2_up_64  # channel = 256    1 256 64 64
        m2 = m2 + m0_down_32 + m3_up_32 + m1_down_32  # channel = 512   1 512 32 32
        m3 = m0_down_16 + m1_down_16 + m2_down_16  # channel = 1024      1 1024 16 16

        # ASPP + CIA
        # m3_aspp = self.aspp(m3)
        m3_cia = self.cia(m3)  # 1 256 16 16
        m3 = m3_cia

        return m0, m1, m2, m3  # channel of m is 1024


    def decoder(self, fuse0, fuse1, fuse2, fuse3):

        # agent3 = self.agant3(fuse3)
        # print(agent3.size()) [1, 256, 16, 16]

        # 0:1 64 128 128   1:1 256 64 64    2:1 512 32 32    3:1 256 16 16
        # upsample 2
        x = self.deconv2(fuse3)  # 1 128 32 32
        x = x + self.agant2(fuse2)  # 1 128 32 32
        # print(x.size()) [1, 128, 32, 32]

        # upsample 3
        x = self.deconv3(x)
        x = x + self.agant1(fuse1)
        # print(x.size()) [1, 64, 64, 64]

        # upsample 4
        x = self.deconv4(x)
        x = x + self.agant0(fuse0)
        # print(x.size()) [1, 64, 128, 128]

        # final
        x = self.final_conv(x)  # 1 64 128 128
        out = self.final_deconv(x)  # 1 6 256 256

        return out

    def forward(self, rgb, depth, phase_checkpoint=False):
        fuses = self.encoder(rgb, depth) # 0:1 64 128 128   1:1 256 64 64    2:1 512 32 32    3:1 256 16 16
        m = self.decoder(*fuses)
        return m

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(model_urls['rednet50'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # print('%%%%% ', k)
            if k in state_dict:
                if k.startswith('invo1'):
                    model_dict[k] = v
                    # print('##### ', k)
                    model_dict[k.replace('invo1', 'invo1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('invo1', 'invo1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6]+'_d'+k[6:]] = v
                    model_dict[k[:6]+'_m'+k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.invo2 = Involution2d(planes, planes, kernel_size=3, stride=stride, groups=16, reduce_ratio=4, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)  # 1*1卷积
        out = self.bn1(out)
        out = self.relu(out)

        out = self.invo2(out)  # 3*3内卷
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 残差
        out = self.relu(out)

        return out

class Bottleneck_att(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck_att, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.invo2 = Involution2d(planes, planes, kernel_size=3, stride=stride, groups=16, reduce_ratio=4, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.second_filtering = Second_feature_filtering_att(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)  # 1*1卷积
        out = self.bn1(out)
        out = self.relu(out)

        out = self.invo2(out)  # 3*3内卷
        out = self.bn2(out)
        out = self.relu(out)

        out = self.second_filtering(out)  # 纯的通道和空间注意力

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 残差
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == "__main__":
    model = Invo_Plus_Net(num_class=6, pretrained=False)
    model.eval()
    img = torch.randn(1,3,256,256)
    dsm = torch.randn(1,1,256,256)
    output = model(img,dsm)
    print(output.size())

    flops, params = profile(model, inputs=(img, dsm))
    print("flops:",flops / 1000000000 , "G")
    print("params",params / 1000000 , "M")

    # flops: 21.109898144 G  Invo_filtering + context++ + connection++
    # params 27.118835 M