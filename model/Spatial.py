import torch
from torch import nn



class Selective_Spatial_Atttention(nn.Module):
    def __init__(self, dim = 64, M = 4 , reduction = 4):
        super(Selective_Spatial_Atttention, self).__init__()
        self.M = M
        self.dim = dim
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False), #FC
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
            nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, stride=1, padding=1, groups=dim // reduction),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, stride=1, padding=1, groups=dim // reduction),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, 1, kernel_size=1, stride=1, padding=0)
        )
            )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, q1, q2, q3, q4):
        # Get the batch size from one of the input tensors
        batch_size, c, h , w  = q1.shape
        q = torch.cat([q1, q2, q3, q4], dim=1)
        q = q.view(batch_size, self.M, self.dim, q.shape[2], q.shape[3])
        feats_U = torch.sum(q, dim=1)
        # feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_U)
        mask = [fc(feats_Z) for fc in self.fcs]
        mask = torch.cat(mask, dim=1)
        mask = mask.view(batch_size, self.M, 1, h, w)
        mask = self.softmax(mask)
        feats_V = torch.sum(q * mask, dim=1)
        return feats_V