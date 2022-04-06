from torchvision import models
import torch
from torch import nn
from scipy.special import binom
from efficientnet_pytorch import EfficientNet

class Resnet18AddModule(nn.Module):
    def __init__(self, class_num, drop, nightlight=False, road=False):
        super(Resnet18AddModule, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.resnets = models.resnet18(pretrained=True)
        self.add_linear1 = nn.Linear(1000, 512)
        self.add_batchnorm1 = nn.BatchNorm1d(512)
        self.add_dropout1 = nn.Dropout(drop)
        self.add_relu1 = nn.ReLU(inplace=True)
        self.add_linear2 = nn.Linear(512, 256)
        self.add_batchnorm2 = nn.BatchNorm1d(256)
        self.add_dropout2 = nn.Dropout(drop)
        self.add_relu2 = nn.ReLU(inplace=True)
        self.add_linear3 = nn.Linear(256, 64)
        self.add_batchnorm3 = nn.BatchNorm1d(64)
        self.add_dropout3 = nn.Dropout(drop)
        self.add_relu3 = nn.ReLU(inplace=True)
        self.add_linear4 = nn.Linear(64, class_num)
        self.add_nightlight_linear4 = nn.Linear(66, class_num)
        self.add_road_linear4 = nn.Linear(67, class_num)
        self.add_nightlight_road_linear4 = nn.Linear(69, class_num)
        self.add_softmax = nn.LogSoftmax(dim=1)



    def forward(self, x, nightlight = None, road = None):
        x = self.resnets(x)
        x = self.add_linear1(x)
        x = self.add_batchnorm1(x)
        x = self.add_dropout1(x)
        x = self.add_relu1(x)
        x = self.add_linear2(x)
        x = self.add_batchnorm2(x)
        x = self.add_dropout2(x)
        x = self.add_relu2(x)
        x = self.add_linear3(x)
        x = self.add_batchnorm3(x)
        x = self.add_dropout3(x)
        x = self.add_relu3(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.add_nightlight_road_linear4(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.add_nightlight_linear4(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.add_road_linear4(x)
            else:
                x = self.add_linear4(x)
        out = self.add_softmax(x)
        return out


class Resnet50AddModule(nn.Module):
    def __init__(self, class_num, drop, nightlight=False, road=False):
        super(Resnet50AddModule, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.resnets = models.resnet50(pretrained=True)
        self.add_linear1 = nn.Linear(1000, 512)
        self.add_batchnorm1 = nn.BatchNorm1d(512)
        self.add_dropout1 = nn.Dropout(drop)
        self.add_relu1 = nn.ReLU(inplace=True)
        self.add_linear2 = nn.Linear(512, 256)
        self.add_batchnorm2 = nn.BatchNorm1d(256)
        self.add_dropout2 = nn.Dropout(drop)
        self.add_relu2 = nn.ReLU(inplace=True)
        self.add_linear3 = nn.Linear(256, 64)
        self.add_batchnorm3 = nn.BatchNorm1d(64)
        self.add_dropout3 = nn.Dropout(drop)
        self.add_relu3 = nn.ReLU(inplace=True)
        self.add_linear4 = nn.Linear(64, class_num)
        self.add_nightlight_linear4 = nn.Linear(66, class_num)
        self.add_road_linear4 = nn.Linear(67, class_num)
        self.add_nightlight_road_linear4 = nn.Linear(69, class_num)
        self.add_softmax = nn.LogSoftmax(dim=1)



    def forward(self, x, nightlight = None, road = None):
        x = self.resnets(x)
        x = self.add_linear1(x)
        x = self.add_batchnorm1(x)
        x = self.add_dropout1(x)
        x = self.add_relu1(x)
        x = self.add_linear2(x)
        x = self.add_batchnorm2(x)
        x = self.add_dropout2(x)
        x = self.add_relu2(x)
        x = self.add_linear3(x)
        x = self.add_batchnorm3(x)
        x = self.add_dropout3(x)
        x = self.add_relu3(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.add_nightlight_road_linear4(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.add_nightlight_linear4(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.add_road_linear4(x)
            else:
                x = self.add_linear4(x)
        out = self.add_softmax(x)
        return out


class Resnet18Module(nn.Module):
    def __init__(self, class_num, nightlight=False, road=False):
        super(Resnet18Module, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.resnets = models.resnet18(pretrained=True)
        feature = self.resnets.fc.in_features
        self.resnets.fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=class_num, bias=True)
            else:
                self.fc = nn.Linear(in_features=66, out_features=class_num, bias=True)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=class_num, bias=True)
            else:
                self.resnets.fc = nn.Linear(in_features=feature, out_features=class_num, bias=True)



    def forward(self, x, nightlight = None, road = None):
        x = self.resnets(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.fc(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
        out = self.add_softmax(x)
        return out


class Resnet50Module(nn.Module):
    def __init__(self, class_num, nightlight=False, road=False):
        super(Resnet50Module, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.resnets = models.resnet50(pretrained=True)
        feature = self.resnets.fc.in_features
        self.resnets.fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=class_num, bias=True)
            else:
                self.fc = nn.Linear(in_features=66, out_features=class_num, bias=True)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=class_num, bias=True)
            else:
                self.resnets.fc = nn.Linear(in_features=feature, out_features=class_num, bias=True)

    def forward(self, x, nightlight=None, road=None):
        x = self.resnets(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.fc(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
        out = self.add_softmax(x)
        return out


class Resnet18Modulenewloss(nn.Module):
    def __init__(self, class_num, nightlight=False, road=False):
        super(Resnet18Modulenewloss, self).__init__()
        self.classes = class_num
        self.nightlight = nightlight
        self.road = road
        self.resnets = models.resnet18(pretrained=True)
        feature = self.resnets.fc.in_features
        self.resnets.fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=1, bias=True)
            else:
                self.fc = nn.Linear(in_features=66, out_features=1, bias=True)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=1, bias=True)
            else:
                self.resnets.fc = nn.Linear(in_features=feature, out_features=1, bias=True)



    def forward(self, x, nightlight = None, road = None):
        x = self.resnets(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.fc(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
        out = torch.cat([torch.mul(binom(self.classes - 1, k) * torch.pow(x, k), torch.pow(1 - x, self.classes - 1 - k))
                         for k in range(self.classes)], dim=1)
        return out


class Resnet50Modulenewloss(nn.Module):
    def __init__(self, class_num, nightlight=False, road=False):
        super(Resnet50Modulenewloss, self).__init__()
        self.classes = class_num
        self.nightlight = nightlight
        self.road = road
        self.resnets = models.resnet50(pretrained=True)
        feature = self.resnets.fc.in_features
        self.resnets.fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=1, bias=True)
            else:
                self.fc = nn.Linear(in_features=66, out_features=1, bias=True)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=1, bias=True)
            else:
                self.resnets.fc = nn.Linear(in_features=feature, out_features=1, bias=True)

    def forward(self, x, nightlight=None, road=None):
        x = self.resnets(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.fc(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
        out = torch.cat([torch.mul(binom(self.classes - 1, k) * torch.pow(x, k), torch.pow(1 - x, self.classes - 1 - k))
                         for k in range(self.classes)], dim=1)
        return out


class Efficientnetmodel(nn.Module):
    def __init__(self, class_num, nightlight=False, road=False):
        super(Efficientnetmodel, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.efficient = EfficientNet.from_pretrained('efficientnet-b0')
        feature = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=class_num, bias=True)
                # self.batchnorm = nn.BatchNorm1d(69)
            else:
                self.fc = nn.Linear(in_features=66, out_features=class_num, bias=True)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=class_num, bias=True)
            else:
                self.efficient._fc = nn.Linear(in_features=feature, out_features=class_num, bias=False)

    def forward(self, x, nightlight=None, road=None):
        x = self.efficient(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.fc(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
        out = self.add_softmax(x)
        return out


class Efficientnetmodel1(nn.Module):
    def __init__(self, class_num, drop, nightlight=False, road=False):
        super(Efficientnetmodel1, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.efficient = EfficientNet.from_pretrained('efficientnet-b0')
        feature = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=class_num, bias=True)
                self.batchnorm = nn.BatchNorm1d(69)
            else:
                self.fc = nn.Linear(in_features=66, out_features=class_num, bias=True)
                self.batchnorm = nn.BatchNorm1d(66)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=class_num, bias=True)
                self.batchnorm = nn.BatchNorm1d(67)
            else:
                self.fc = nn.Linear(in_features=64, out_features=class_num, bias=True)
                self.batchnorm = nn.BatchNorm1d(64)

    def forward(self, x, nightlight=None, road=None):
        x = self.efficient(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
            else:
                x = torch.cat([x, nightlight], dim=1)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        out = self.add_softmax(x)
        return out


class Efficientnetmodelnewloss(nn.Module):
    def __init__(self, classes, nightlight=False, road=False):
        super(Efficientnetmodelnewloss, self).__init__()
        self.nightlight = nightlight
        self.road = road
        self.classes = classes
        self.efficient = EfficientNet.from_pretrained('efficientnet-b0')
        feature = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(in_features=feature, out_features=64, bias=True)
        self.add_softmax = nn.LogSoftmax(dim=1)
        if self.nightlight:
            if self.road:
                self.fc = nn.Linear(in_features=69, out_features=1, bias=True)
            else:
                self.fc = nn.Linear(in_features=66, out_features=1, bias=True)
        else:
            if self.road:
                self.fc = nn.Linear(in_features=67, out_features=1, bias=True)
            else:
                self.efficient._fc = nn.Linear(in_features=feature, out_features=1, bias=True)


    def forward(self, x, nightlight = None, road = None):
        x = self.efficient(x)
        if self.nightlight:
            if self.road:
                x = torch.cat([x, nightlight], dim=1)
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
            else:
                x = torch.cat([x, nightlight], dim=1)
                x = self.fc(x)
        else:
            if self.road:
                x = torch.cat([x, road], dim=1)
                x = self.fc(x)
        out = torch.cat([torch.mul(binom(self.classes - 1, k) * torch.pow(x, k), torch.pow(1 - x, self.classes - 1 - k))
                   for k in range(self.classes)], dim=1)
        return out