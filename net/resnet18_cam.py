import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet18


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet18 = resnet18.resnet18(pretrained=True, strides=(1, 2, 2, 2))

        self.stage1 = nn.Sequential(self.resnet18.conv1, self.resnet18.bn1, self.resnet18.relu, self.resnet18.maxpool,
                                    self.resnet18.layer1)
        self.stage2 = nn.Sequential(self.resnet18.layer2)
        self.stage3 = nn.Sequential(self.resnet18.layer3)
        self.stage4 = nn.Sequential(self.resnet18.layer4)

        self.classifier = nn.Conv2d(512, 2, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 2)

        return x

    def train(self, mode=True):
        for p in self.resnet18.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet18.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
