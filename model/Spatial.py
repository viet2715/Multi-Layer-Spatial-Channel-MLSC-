import torch
from torch import nn


class MahalanobisBlock(nn.Module):
    def __init__(self):
        super(MahalanobisBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()
            local_feature_list = []

            for local_feature in support_set_sam:
                local_feature_np = local_feature.detach().cpu().numpy()
                transposed_tensor = np.transpose(local_feature_np, (1, 2, 0))
                reshaped_tensor = np.reshape(transposed_tensor, (h * w, C))

                for line in reshaped_tensor:
                    local_feature_list.append(line)

            local_feature_np = np.array(local_feature_list)
            mean = np.mean(local_feature_np, axis=0)
            local_feature_list = [x - mean for x in local_feature_list]

            covariance_matrix = np.cov(local_feature_np, rowvar=False)
            covariance_matrix = torch.from_numpy(covariance_matrix)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list



    def mahalanobis_similarity(self, input, CovaMatrix_list, regularization=1e-6):
        B, C, h, w = input.size()
        mahalanobis = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            # mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w)
            for j in range(len(CovaMatrix_list)):

                covariance_matrix = CovaMatrix_list[j].float().cuda() + regularization * torch.eye(C).cuda()
                # covariance_matrix = CovaMatrix_list[j] + regularization * torch.eye(C)
                inv_covariance_matrix = torch.linalg.inv(covariance_matrix)
                diff = query_sam - torch.mean(query_sam, dim=1, keepdim=True)
                temp_dis = torch.matmul(torch.matmul(diff.T, inv_covariance_matrix), diff)
                # temp_dis = temp_dis / (torch.norm(temp_dis, p=2, dim=[0, 1 ], keepdim=True))
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            mahalanobis.append(mea_sim.view(1, -1))

        mahalanobis = torch.cat(mahalanobis, 0)

        return mahalanobis


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        maha_sim = self.mahalanobis_similarity(x1, CovaMatrix_list)

        return maha_sim








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