import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)

    def _upsample(self, x):
        return F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        p1,p2,p3,p4,p5=x[0],x[1],x[2],x[3],x[4]
        # Semantic
        _, _, h, w = p1.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p4))))
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))))
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))))

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))))
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))))

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))))

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        return self._upsample(self.conv3(s2 + s3 + s4 + s5))
