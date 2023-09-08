# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from loss.dynamic_encoding import SwitchEncoding
import torch.nn.functional as F
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
class CIFARNet(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 X 32 X 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 X 8 X 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 X 4 X 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        return self.network(x)

class IMAGENET(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 32 X 32

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 8 X 8

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        output = self.network1(x)
        output = self.network2(output)
        output = self.network3(output)
        output = self.network4(output)
        return output
class KMNISTNet(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=3),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 16 X 16

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        output = self.network1(x)
        output = self.network2(output)
        output = self.network3(output)
        output = self.network4(output)
        return output


class CIFARNet_SelfDirect(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 32 X 32

        self.classifier1 = nn.Sequential(
            nn.Flatten(),
            nn.Softmax()
        )

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.classifier2 = nn.Sequential(
            nn.Flatten(),
            nn.Softmax(),
        )

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.classifier3 = nn.Sequential(
            nn.Flatten(),
            nn.Softmax()
        )

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):

        output = self.network1(x)
        class_res1 = self.classifier1(output)
        output = self.network2(output)
        class_res2 = self.classifier2(output)
        output = self.network3(output)
        class_res3 = self.classifier3(output)
        output = self.network4(output)

        return output, [class_res1, class_res2, class_res3]



class KMNISTNet_Directed(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()
        self.direct = nn.Sequential(
            nn.Flatten(),
            nn.Softmax()
        )

        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=3),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 16 X 16

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.network1(x)
        l1 = self.direct(x)
        x = self.network2(x)
        l2 = self.direct(x)
        x = self.network3(x)
        l3 = self.direct(x)
        x = self.network4(x)
        return x, [l1, l2, l3]


class KMNISTNet_Directed_Dual(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()

        self.direct_plan = nn.Sequential(
            nn.Softmax2d()
        )

        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=3),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 16 X 16

        self.direct_channel1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16),
            nn.Softmax2d()
        )

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.direct_channel2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            nn.Softmax2d()
        )

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.direct_channel3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Softmax2d()
        )

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.network1(x)
        l1_plan = self.direct_plan(x)
        l1_channel = self.direct_channel1(x)
        x = self.network2(x)
        l2_plan = self.direct_plan(x)
        l2_channel = self.direct_channel2(x)
        x = self.network3(x)
        l3_plan = self.direct_plan(x)
        l3_channel = self.direct_channel3(x)
        x = self.network4(x)
        return x, [l1_plan, l2_plan, l3_plan], [l1_channel, l2_channel, l3_channel]


class CIFARNet_SelfDirect_Dual(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()

        self.direct_plan = nn.Sequential(
            nn.Softmax2d()
        )

        self.direct_channel = nn.Sequential(
            torch.max(dim=-1),
            torch.max(dim=-1),
            nn.Softmax()
        )

        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 32 X 32

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):

        output = self.network1(x)
        l1_plan = self.direct_plan(x)
        l1_channel = self.direct_channel(x)

        output = self.network2(output)
        l2_plan = self.direct_plan(x)
        l2_channel = self.direct_channel(x)

        output = self.network3(output)
        l3_plan = self.direct_plan(x)
        l3_channel = self.direct_channel(x)

        output = self.network4(output)

        return output, [l1_plan, l2_plan, l3_plan], [l1_channel, l2_channel, l3_channel]



class KMNISTNet_Directed_Norm(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()

        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=3),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 16 X 16

        self.norm1 = nn.BatchNorm2d(64)

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.norm2 = nn.BatchNorm2d(128)

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.norm3 = nn.BatchNorm2d(256)

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax())

    def forward(self, x):
        x = self.network1(x)
        x = self.norm1(x)
        x = self.network2(x)
        x = self.norm2(x)
        x = self.network3(x)
        x = self.norm3(x)
        x = self.network4(x)
        return x


class CIFARNet_SelfDirect_Norm(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()

        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 32 X 32

        self.norm1 = nn.BatchNorm2d(64)

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.norm2 = nn.BatchNorm2d(128)

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.norm3 = nn.BatchNorm2d(256)

        self.network4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        output = self.network1(x)
        output = self.norm1(output)

        output = self.network2(output)
        output = self.norm2(output)

        output = self.network3(output)
        output = self.norm3(output)

        output = self.network4(output)
        return output


class CIFARNet_Infer(nn.Module):
    def __init__(self, num_class, num_channel=3, device="cpu"):
        super().__init__()

        self.network1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 64 X 32 X 32

        self.norm1 = nn.BatchNorm2d(64)

        self.network2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 128 X 8 X 8

        self.norm2 = nn.BatchNorm2d(128)

        self.network3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # output: 256 X 4 X 4

        self.norm3 = nn.BatchNorm2d(256)

        # decoder
        self.flatten = nn.Flatten()
        self.attention_m1 = torch.randn((1024, 256 * 4 * 4)).to(device)

        self.attention_m2 = torch.randn((512, 1024)).to(device)

        self.attention_m3 = torch.randn((num_class, 512)).to(device)

        self.sf = nn.Softmax()

    def forward(self, x):
        output = self.network1(x)
        output = self.norm1(output)

        output = self.network2(output)
        output = self.norm2(output)

        output = self.network3(output)
        output = self.norm3(output)

        output = self.flatten(output)

        self.attention_m1_sf11 = torch.softmax(self.attention_m1, dim=1)
        self.attention_m1_sf12 = torch.softmax(self.attention_m1, dim=0)
        output11 = torch.matmul(output, torch.transpose(self.attention_m1_sf11, 0, 1))
        output12 = torch.matmul(output, torch.transpose(self.attention_m1_sf12, 0, 1))
        output = torch.mul(output11, output12)
        output = torch.relu(output)

        self.attention_m1_sf21 = torch.softmax(self.attention_m2, dim=1)
        self.attention_m1_sf22 = torch.softmax(self.attention_m2, dim=0)
        output21 = torch.matmul(output, torch.transpose(self.attention_m1_sf21, 0, 1))
        output22 = torch.matmul(output, torch.transpose(self.attention_m1_sf22, 0, 1))
        output = torch.mul(output21, output22)
        output = torch.relu(output)

        self.attention_m1_sf31 = torch.softmax(self.attention_m3, dim=1)
        self.attention_m1_sf32 = torch.softmax(self.attention_m3, dim=0)
        output31 = torch.matmul(output, torch.transpose(self.attention_m1_sf31, 0, 1))
        output32 = torch.matmul(output, torch.transpose(self.attention_m1_sf32, 0, 1))
        output = torch.mul(output31, output32)
        output = torch.relu(output)

        output = self.sf(output)
        return output