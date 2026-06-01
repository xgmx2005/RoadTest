import torch
import torchvision
import numpy as np


class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetBackbone(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super().__init__()
        if layers == "18":
            model = torchvision.models.resnet18(weights=None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT)
        elif layers == "34":
            model = torchvision.models.resnet34(weights=None if not pretrained else torchvision.models.ResNet34_Weights.DEFAULT)
        elif layers == "50":
            model = torchvision.models.resnet50(weights=None if not pretrained else torchvision.models.ResNet50_Weights.DEFAULT)
        elif layers == "101":
            model = torchvision.models.resnet101(weights=None if not pretrained else torchvision.models.ResNet101_Weights.DEFAULT)
        elif layers == "152":
            model = torchvision.models.resnet152(weights=None if not pretrained else torchvision.models.ResNet152_Weights.DEFAULT)
        else:
            raise NotImplementedError(layers)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class ParsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=False, backbone="18",
                 cls_dim=(201, 18, 4), use_aux=False):
        super().__init__()
        self.size = size
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = int(np.prod(cls_dim))
        self.model = ResNetBackbone(backbone, pretrained=pretrained)

        if use_aux:
            self.aux_header2 = torch.nn.Sequential(
                ConvBnRelu(128, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"] else ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                ConvBnRelu(256, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"] else ConvBnRelu(1024, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1)
                if backbone in ["34", "18"] else ConvBnRelu(2048, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                ConvBnRelu(384, 256, 3, padding=2, dilation=2),
                ConvBnRelu(256, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1, 1),
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in ["34", "18"] else torch.nn.Conv2d(2048, 8, 1)
        initialize_weights(self.cls)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode="bilinear")
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls


def initialize_weights(*models):
    for model in models:
        _real_init_weights(model)


def _real_init_weights(module):
    if isinstance(module, list):
        for child in module:
            _real_init_weights(child)
        return
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0.0, std=0.01)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Module):
        for child in module.children():
            _real_init_weights(child)
