import torch
from torch import nn
import torch.nn.functional as F
from model.Feature_Extraction import CAFE, Mamba_Transformer, Conv_first
from model.Spatial import Selective_Spatial_Attention, MahalanobisBlock
from model.Transformer import  Encoder_Decoder


class Block_1(nn.Module):
  def __init__ (self,dim=64):
    super(Block_1,self).__init__()
    self.dim=dim
    self.conv_first=Conv_first(self.dim)
    self.CAFE=CAFE(self.dim)
    self.Encoder_Decoder=Encoder_Decoder(self.dim)
  def forward (self,q,S):
    q=self.conv_first(q)
    q=self.CAFE(q)
    feature=[]
    for i in S:
      i=self.conv_first(i)
      i=self.CAFE(i)
      feature.append(i)
    S=torch.stack(feature,dim=0)
    output, QK_matrix=self.Encoder_Decoder(q,S)
    return output, q,S, QK_matrix

class Block_2(nn.Module):
  def __init__ (self,dim=64):
    super(Block_2,self).__init__()
    self.dim=dim
    self.CAFE=CAFE(self.dim)
    self.Encoder_Decoder=Encoder_Decoder(self.dim)
  def forward (self, q, S, QK_matrix):
    q=self.CAFE(q)
    feature=[]
    for i in S:
      i=self.CAFE(i)
      feature.append(i)
    S=torch.stack(feature,dim=0)
    output, QK_matrix_out=self.Encoder_Decoder(q, S, QK_matrix)
    return output, q,S, QK_matrix_out
  
class Block_3(nn.Module):
  def __init__ (self,dim=64):
    super(Block_3,self).__init__()
    self.dim=dim
    self.mamba = Mamba_Transformer(self.dim)
    self.Encoder_Decoder=Encoder_Decoder(self.dim)
  def forward (self, q, S, QK_matrix):
    q=self.mamba(q)
    feature=[]
    for i in S:
      i=self.mamba(i)
      feature.append(i)
    S=torch.stack(feature,dim=0)
    output, QK_matrix_out=self.Encoder_Decoder(q, S, QK_matrix)
    return output, q,S, QK_matrix_out
  
class Model(nn.Module):
  def __init__ (self,dim=64,num_class=13):
    super(Model, self).__init__()
    self.dim=dim
    self.Block_1 = Block_1(self.dim) # Conv first
    self.Block_2 = Block_2(self.dim) # SAFE
    self.Block_3 = Block_2(self.dim) # SAFE
    self.Block_4 = Block_3(self.dim) # Mamba
    self.w1 = nn.Parameter(torch.tensor(1.0))
    self.w2 = nn.Parameter(torch.tensor(1.0))
    self.w3 = nn.Parameter(torch.tensor(1.0))
    self.w4 = nn.Parameter(torch.tensor(1.0))
    self.num_class=num_class
    self.conv1d=nn.Conv1d(num_class,num_class,dim)
    self.covariance = MahalanobisBlock()
    self.selective = Selective_Spatial_Attention(self.dim)
    self.Linear_spatial = nn.Conv1d(1,1, 256, 256)
    self.wc = nn.Parameter(torch.tensor(1.0))
    self.ws = nn.Parameter(torch.tensor(1.0))
  def forward (self,q,S):
    output_1, q1, S1, QK_matrix1=self.Block_1(q, S)
    output_2, q2, S2, QK_matrix2=self.Block_2(q1, S1, QK_matrix1)
    output_3, q3, S3, QK_matrix3=self.Block_3(q2, S2, QK_matrix2)
    output_4, q4, S4, QK_matrix4=self.Block_4(q3, S3, QK_matrix3) 
    output=self.w1*output_1+self.w2*output_2+self.w3*output_3+self.w4*output_4  
    output=self.conv1d(output)
    output=output.squeeze(dim=2)

    q_selective = self.selective(q1, q2, q3, q4)
    S_out = []
    for i in range(S1.shape[0]):
        s_selective = self.selective(S1[i], S2[i], S3[i], S4[i])
        S_out.append(s_selective)
    S_out = torch.stack(S_out, dim=0)
    maha_sim = self.covariance(q_selective, S_out)
    output_s = self.Linear_spatial(maha_sim.unsqueeze(dim=1))
    output_s = output_s.squeeze(dim=1)
    output_last = self.ws*output_s+self.wc*output
    return output_last

