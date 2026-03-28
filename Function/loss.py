#-------------------Contrastive Loss------------------------#
import torch
import torch.nn as nn
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    def forward(self, ouput, target):
        upper = torch.exp(ouput[:,target])
        lower = torch.exp(ouput).sum(1)
        loss = -torch.log(upper / lower)

        return loss