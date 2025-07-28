import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import Conv, C3k2, SPPF, C2PSA


class DarkNetV11(nn.Module):
    '''
        This class used to deploy a yolov11's backbone to depth estimation models.
        The model's init only get three elements:
            width        decide the width expandations.
            max_channels decide the maxmim channels.
    '''
    def __init__(self, width=1.5, max_channels=512):
        super().__init__()
        self.width        = width
        self.max_channels = max_channels

        # stride 2
        self.input_conv   = Conv(c1=3, c2=int(64* width), k=3, s=2)
        # stride 4
        self.conv_path1   = Conv(int(64 * width), int(128 * width), 3, 2)
        self.c3k2_block1  = C3k2(int(128 * width), int(256 * width), n=2, c3k=True, e=0.25)

        # stride 8
        self.conv_path2   = Conv(int(256 * width), int(256 * width), n=2, c3k=True, e=0.25)
        self.c3k2_block2  = C3k2(int(256 * width), int(512 * width), 2, True, 0.25)

        # stride 16
        self.conv_path3   = Conv(int(min(512, max_channels) * width), int(min(512, max_channels) * width), 3, 2)
        self.c3k2_block3  = C3k2(int(min(512, max_channels) * width), int(min(1024, max_channels) * width), 2, True)

        # stride 32
        self.conv_path4   = Conv(int(min(1024, max_channels) * width), int(min(1024, max_channels) * width), 3, 2)
        self.c3k2_block4  = C3k2(int(min(1024, max_channels) * width), int(min(1024, max_channels) * width), 2, True)

        self.sppf_block   = SPPF(int(min(1024, max_channels) * width), int(min(1024, max_channels) * width), 5)
        self.c2psa        = C2PSA(int(min(1024, max_channels) * width), int(min(1024, max_channels) * width), 2)


    def forward(self, input_tensor):
        '''
            Forward the backbone and output the stride 4, stride 8, stride 16, stride 32 feature maps
        '''
        stems   = self.input_conv(input_tensor)

        stride4 = self.conv_path1(stems)
        stride4 = self.c3k2_block1(stride4)

        stride8 = self.conv_path2(stride8)
        stride8 = self.c3k2_block2(stride8)

        stride16 = self.conv_path3(stride8)
        stride16 = self.c3k2_block3(stride16)

        stride32 = self.conv_path4(stride16)
        stride32 = self.c3k2_block4(stride32)
        result_list = [stride4, stride8, stride16, stride32]