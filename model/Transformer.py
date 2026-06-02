import torch
from torch import nn



class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim=64):
        super(ScaledDotProductAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, q, k, v):
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size, dim).
            k (Tensor): Key tensor of shape (batch_size, dim).
            v (Tensor): Value tensor of shape (batch_size, dim).

        Returns:
            output (Tensor): Scaled Dot-Product Attention output tensor of shape (batch_size, dim).
        """
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        output = torch.matmul(attention_weights, v.unsqueeze(2))
        output = output.squeeze(2)
        return output

class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.dim = dim
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.norm = nn.LayerNorm(normalized_shape=self.dim)
        self.FFN = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )
    def forward(self, Support):
      encoded = []
      for index in range(len(Support)):
          s = Support[index]     
          s = self.GAP(s)                                     
          s = s.view(s.size(0), s.size(1))                    
          s = self.ScaledDotProductAttention(s, s, s) + s    
          s = self.norm(s)                                    
          s = self.FFN(s) + s                                
          s = self.norm(s)                           
          s = torch.mean(s, dim=0, keepdim=True)
          encoded.append(s)

      return encoded                                       


class CrossAttention(nn.Module):
    def __init__(self, dim=64):
        super(CrossAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim
    def forward(self, q, k, v):
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size,  dim).
            k (Tensor): Key tensor of shape (batch_size, dim).
            v (Tensor): Value tensor of shape (batch_size, dim).

        Returns:
            output (Tensor): Scaled Dot-Product Attention output tensor of shape (batch_size, dim).
        """
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)

        return attention_weights   

class Encoder_Decoder(nn.Module):
    def __init__(self, dim):
        super(Encoder_Decoder, self).__init__()
        self.dim = dim
        self.encoder_out = Encoder(self.dim)
        self.attention = CrossAttention()
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.norm = nn.LayerNorm(normalized_shape=self.dim)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=dim)
        self.MLP = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )

    def forward(self, q, S, QK_matrix = None):
        # q_spatial = self.CVT(q)
        q = self.GAP(q)                                    
        q = q.view(q.size(0), q.size(1))                    
        q_first = q                                           
        q = self.ScaledDotProductAttention(q, q, q) + q_first
        q = self.norm(q)
        q = self.MLP(q) + q
        q = self.norm(q)                        
        output = []
        encoder_outs = self.encoder_out(S)                    
        QK_matrix_out = []
        for i, encoder_out in enumerate(encoder_outs):
            if QK_matrix is not None:
                out = self.attention(q, encoder_out, encoder_out) + QK_matrix[i]
            out = self.attention(q, encoder_out, encoder_out) 
            QK_matrix_out.append(out)
            out = self.Linear(out)                           
            out = out.squeeze(2)                             
            output.append(out)
        output=torch.stack(output,dim=1)
        QK_matrix_out=torch.stack(QK_matrix_out,dim=0)



        return output, QK_matrix_out                                   