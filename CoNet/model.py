import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


# ----------------------------------  ResNet101  ---------------------------------- #
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_1= x

        x = self.layer1(x)
        low_2 = x
        x = self.layer2(x)
        high_1 = x
        x = self.layer3(x)
        high_2 = x
        x = self.layer4(x)
        high_3 = x
        return low_1, low_2, high_1, high_2, high_3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model
# -------------------------------------------  End  ------------------------------------------ #



# ---------------------------------  Collaborative Learning  --------------------------------- #

class Integration(nn.Module):
    def __init__(self):
        super(Integration, self).__init__()

        # ----------> Feature Extract <-----------
        # conv3
        self.conv3_0 = nn.Conv2d(512, 64, 1, padding=0)
        self.bn3_0 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_0 = nn.PReLU()

        # conv4
        self.conv4_0 = nn.Conv2d(1024, 64, 1, padding=0)
        self.bn4_0 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_0 = nn.PReLU()

        # conv5
        self.conv5_0 = nn.Conv2d(2048, 64, 1, padding=0)
        self.bn5_0 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_0 = nn.PReLU()

        # ----------> Feature Extract < -----------
        # ===== Low-level features =====
        self.conv_low = nn.Conv2d(256 + 64, 64, 3, padding=1)
        self.bn_low = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_low = nn.PReLU()

        # ===== High-level features [GPM] =====
        # --- Conv5 ---
        # part0:   1*1*64 Conv
        self.C5_conv1 = nn.Conv2d(64, 64, 1, padding=0)
        self.C5_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C5_relu1 = nn.PReLU()
        # part1:   3*3*64 Conv dilation =1
        self.C5_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.C5_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C5_relu2 = nn.PReLU()
        # part2:   3*3*64 Conv dilation =6
        self.C5_conv3 = nn.Conv2d(64, 64, 3, padding=6, dilation=6)
        self.C5_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C5_relu3 = nn.PReLU()
        # part3:   3*3*64 Conv dilation =12
        self.C5_conv4 = nn.Conv2d(64, 64, 3, padding=12, dilation=12)
        self.C5_bn4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C5_relu4 = nn.PReLU()
        # part4:   3*3*64 Conv dilation =18
        self.C5_conv5 = nn.Conv2d(64, 64, 3, padding=18, dilation=18)
        self.C5_bn5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C5_relu5 = nn.PReLU()
        # part5:   1*1*64 Conv Concatenation
        self.C5_conv = nn.Conv2d(64 * 5, 64, 1, padding=0)
        self.C5_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C5_relu = nn.PReLU()

        # --- Conv4 ---
        # part0:   1*1*64 Conv
        self.C4_conv1 = nn.Conv2d(64, 64, 1, padding=0)
        self.C4_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C4_relu1 = nn.PReLU()
        # part1:   3*3*64 Conv dilation =1
        self.C4_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.C4_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C4_relu2 = nn.PReLU()
        # part2:   3*3*64 Conv dilation =6
        self.C4_conv3 = nn.Conv2d(64, 64, 3, padding=6, dilation=6)
        self.C4_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C4_relu3 = nn.PReLU()
        # part3:   3*3*64 Conv dilation =12
        self.C4_conv4 = nn.Conv2d(64, 64, 3, padding=12, dilation=12)
        self.C4_bn4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C4_relu4 = nn.PReLU()
        # part4:   3*3*64 Conv dilation =18
        self.C4_conv5 = nn.Conv2d(64, 64, 3, padding=18, dilation=18)
        self.C4_bn5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C4_relu5 = nn.PReLU()
        # part5:   1*1*64 Conv Concatenation
        self.C4_conv = nn.Conv2d(64 * 5, 64, 1, padding=0)
        self.C4_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C4_relu = nn.PReLU()

        # --- Conv3 ---
        # part0:   1*1*64 Conv
        self.C3_conv1 = nn.Conv2d(64, 64, 1, padding=0)
        self.C3_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C3_relu1 = nn.PReLU()
        # part1:   3*3*64 Conv dilation =1
        self.C3_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.C3_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C3_relu2 = nn.PReLU()
        # part2:   3*3*64 Conv dilation =6
        self.C3_conv3 = nn.Conv2d(64, 64, 3, padding=6, dilation=6)
        self.C3_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C3_relu3 = nn.PReLU()
        # part3:   3*3*64 Conv dilation =12
        self.C3_conv4 = nn.Conv2d(64, 64, 3, padding=12, dilation=12)
        self.C3_bn4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C3_relu4 = nn.PReLU()
        # part4:   3*3*64 Conv dilation =18
        self.C3_conv5 = nn.Conv2d(64, 64, 3, padding=18, dilation=18)
        self.C3_bn5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C3_relu5 = nn.PReLU()
        # part5:   1*1*64 Conv Concatenation
        self.C3_conv = nn.Conv2d(64 * 5, 64, 1, padding=0)
        self.C3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.C3_relu = nn.PReLU()

        self.conv_high = nn.Conv2d(64 * 3, 64, 3, padding=1)
        self.bn_high = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu_high = nn.PReLU()

        # -------------------- Integration Training ------------------- #
        # Low-level Integration

        self.low_sal = nn.Conv2d(64, 2, 1, padding=0)
        self.pred1_sal = nn.Conv2d(64, 2, 1, padding=0)
        self.pred1_edge = nn.Conv2d(64, 2, 1, padding=0)

        # High-level Integration
        self.high_depth = nn.Conv2d(64, 1, 1, padding=0)
        self.high_sal = nn.Conv2d(64, 2, 1, padding=0)
        self.pred2_sal = nn.Conv2d(64, 2, 1, padding=0)
        self.pred2_depth = nn.Conv2d(64, 1, 1, padding=0)
        # channel attention
        self.conv_ca = nn.Conv2d(1, 64, 3, padding=1)
        self.pool_avg = nn.AvgPool2d(64, stride=2, ceil_mode=True)  # 1/8

        # Depth CNN
        self.D_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.D_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.D_relu1 = nn.PReLU()
        self.D_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.D_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.D_relu2 = nn.PReLU()
        self.D_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.D_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.D_relu3 = nn.PReLU()

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               # m.weight.data.zero_()
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, low_1, low_2, high_1, high_2, high_3):

        # -----------> Feature Extract <----------- #
        # Low-level features
        Low = F.interpolate(torch.cat([low_1, low_2], dim=1), scale_factor=4, mode='bilinear', align_corners=False)
        Low = self.relu_low(self.bn_low(self.conv_low(Low)))

        # High-level features
        h3 = self.relu3_0(self.bn3_0(self.conv3_0(high_1)))
        h3 = F.interpolate(h3, scale_factor=2, mode='bilinear', align_corners=False)
        h4 = self.relu4_0(self.bn4_0(self.conv4_0(high_2)))
        h5 = self.relu5_0(self.bn5_0(self.conv5_0(high_3)))

        c5 = h5
        Conv5_1 = self.C5_relu1(self.C5_bn1(self.C5_conv1(c5)))
        Conv5_2 = self.C5_relu2(self.C5_bn2(self.C5_conv2(c5)))
        Conv5_3 = self.C5_relu3(self.C5_bn3(self.C5_conv3(c5)))
        Conv5_4 = self.C5_relu4(self.C5_bn4(self.C5_conv4(c5)))
        Conv5_5 = self.C5_relu5(self.C5_bn5(self.C5_conv5(c5)))
        Conv5_ori = self.C5_relu(
            self.C5_bn(self.C5_conv(torch.cat([Conv5_1, Conv5_2, Conv5_3, Conv5_4, Conv5_5], dim=1))))
        Conv5 = F.interpolate(Conv5_ori, scale_factor=4, mode='bilinear', align_corners=False)

        c4 = Conv5_ori + h4
        Conv4_1 = self.C4_relu1(self.C4_bn1(self.C4_conv1(c4)))
        Conv4_2 = self.C4_relu2(self.C4_bn2(self.C4_conv2(c4)))
        Conv4_3 = self.C4_relu3(self.C4_bn3(self.C4_conv3(c4)))
        Conv4_4 = self.C4_relu4(self.C4_bn4(self.C4_conv4(c4)))
        Conv4_5 = self.C4_relu5(self.C4_bn5(self.C4_conv5(c4)))
        Conv4_ori = self.C4_relu(
            self.C4_bn(self.C4_conv(torch.cat([Conv4_1, Conv4_2, Conv4_3, Conv4_4, Conv4_5], dim=1))))
        Conv4 = F.interpolate(Conv4_ori, scale_factor=4, mode='bilinear', align_corners=False)

        c3 = Conv5 + Conv4 + h3
        Conv3_1 = self.C3_relu1(self.C3_bn1(self.C3_conv1(c3)))
        Conv3_2 = self.C3_relu2(self.C3_bn2(self.C3_conv2(c3)))
        Conv3_3 = self.C3_relu3(self.C3_bn3(self.C3_conv3(c3)))
        Conv3_4 = self.C3_relu4(self.C3_bn4(self.C3_conv4(c3)))
        Conv3_5 = self.C3_relu5(self.C3_bn5(self.C3_conv5(c3)))
        Conv3 = self.C3_relu(self.C3_bn(self.C3_conv(torch.cat([Conv3_1, Conv3_2, Conv3_3, Conv3_4, Conv3_5], dim=1))))

        High = torch.cat([Conv3, Conv4, Conv5], dim=1)
        High = self.relu_high(self.bn_high(self.conv_high(High)))

        # obtain Low and High
        # -----------> Integration Training <----------- #
        # Low-level Integration
        
        pred_edge1 = self.pred1_edge(Low)

        # Depth CNN
        D1=self.D_relu1(self.D_bn1(self.D_conv1(High)))
        D2=self.D_relu2(self.D_bn2(self.D_conv2(D1)))
        D3=self.D_relu3(self.D_bn3(self.D_conv3(D2)))

        # High-level Integration
        high_depth = self.high_depth(D3)
        # CA
        Att_map_CA = self.pool_avg(self.conv_ca(high_depth))
        Att_map_CA = torch.mul(F.softmax(Att_map_CA, dim=1), 64)
        Att_High = torch.mul(High, Att_map_CA)
        Enhance_High = Att_High + High
        # SA
        high_sal = self.high_sal(Enhance_High)
        Att_map_SA = F.softmax(high_sal,dim=1)[:,1:,:]
        Feature = torch.mul(Enhance_High, Att_map_SA)
        Enhance_Feature = Feature + Enhance_High
        # predict depth attention and sal_map attention
        D_1=self.D_relu1(self.D_bn1(self.D_conv1(Enhance_Feature)))
        D_2=self.D_relu2(self.D_bn2(self.D_conv2(D_1)))
        D_3=self.D_relu3(self.D_bn3(self.D_conv3(D_2)))
        pred_depth = self.pred2_depth(D_3)
        pred_sal2 = self.pred2_sal(Enhance_Feature)

        Enhance_Feature = F.interpolate(Enhance_Feature, scale_factor=4, mode='bilinear', align_corners=False)
        Features = torch.cat([Enhance_Feature,Low],dim=1)

        high_depth = F.interpolate(high_depth, scale_factor=4, mode='bilinear', align_corners=False)
        high_sal = F.interpolate(high_sal, scale_factor=4, mode='bilinear', align_corners=False)
        pred_depth = F.interpolate(pred_depth, scale_factor=4, mode='bilinear', align_corners=False)
        pred_sal2 = F.interpolate(pred_sal2, scale_factor=4, mode='bilinear', align_corners=False)


        return Features, Features, Features, pred_edge1, high_depth, high_sal, pred_depth, pred_sal2



