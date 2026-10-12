import torch
from torch import nn
import timm

from loss import MultiPositiveContrastiveLoss, MultiPositiveTripletLoss


class SEBlock(nn.Module):

    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.squeeze(x).view(b, c)
        s = self.excitation(z).view(b, c, 1, 1)
        return x * s.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels // 4, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels // 4)
        self.conv2 = nn.Conv2d(base_channels // 4, base_channels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels // 4)
        self.conv3 = nn.Conv2d(base_channels // 4, base_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels)
        self.se = SEBlock(base_channels, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out)
        return out


class MS_CAM(nn.Module):

    def __init__(self, inchannel=3, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
        self.conv1 = nn.Conv2d(inchannel, channels, kernel_size=1, stride=1, padding=0)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class CoDeitPhys(nn.Module):

    def __init__(self, deit_init=None, lambda3=0.0, lambda4=1.0):
        super(CoDeitPhys, self).__init__()
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.MultiPositiveContrastiveLoss = MultiPositiveContrastiveLoss(temperature=0.1)
        self.MultiPositiveTripletLoss = MultiPositiveTripletLoss(margin=0.5, distance_metric='cosine')
        self.norm_feature_extractor = SEBottleneck(in_channels=3, base_channels=64)
        self.mscam = MS_CAM(inchannel=3, channels=64, r=4)
        model = timm.create_model('deit_tiny_patch16_224')
        if deit_init is not None:
            checkpoint = torch.load(deit_init, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
        model.head = nn.Linear(model.head.in_features, 192)
        self.deit = model
        self.neck = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, x_1, img_positive, img_negetive, img_mwli):
        x = self.norm_feature_extractor(x)
        x_1 = self.norm_feature_extractor(x_1)
        img_positive = self.norm_feature_extractor(img_positive)
        img_negetive = self.norm_feature_extractor(img_negetive)
        loss_triplet = self.MultiPositiveTripletLoss(x, x_1, img_positive, img_negetive)
        loss_contrastive = self.MultiPositiveContrastiveLoss(x, x_1, img_positive, img_negetive)
        contrast_loss = self.lambda3 * loss_triplet + self.lambda4 * loss_contrastive
        mwli = self.mscam(img_mwli)
        x = torch.cat((x, mwli), dim=1)
        x = self.neck(x)
        out = self.deit(x)
        out = self.fc(out)
        return (out, contrast_loss)

    def predict(self, x, img_mwli):
        physiological = self.norm_feature_extractor(x)
        mwli = self.mscam(img_mwli)
        fused = self.neck(torch.cat((physiological, mwli), dim=1))
        return self.fc(self.deit(fused))
