### Self Attention 
import math
import torch 
import torch.nn as nn

###

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # X shape is : (batch_size, seq_len, hidden_dim)
        Q = self.query_proj(X)
        K = self.query_proj(X)
        V = self.query_proj(X)
        # Q K V shape (batch, seq, hidden_dim)

        # attention_value: (batch, seq, seq)
        attention_value = torch.matmul(
            # K -> (batch, hidden_dim, seq)
            Q, K.transpose(-1, -2)
        ) 

        # attention_weight: (batch, seq, seq)
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim),
            dim = -1 
        )

        # (batch, seq, hidden_dim)
        output = torch.matmul(attention_weight, V)
        return output
    
X = torch.rand(3, 2, 4)

self_att_net = SelfAttentionV1(4)
self_att_net.forward(X)

### improve efficiency

class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim*3)

    def forward(self,X):
        # (batch, seq, dim)
        # QKV shape (batch, seq, dim*3)
        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)
        attention_weight = torch.softmax(
            torch.matmul(
                Q, K.transpose(-1,-2)
            ) / math.sqrt(self.hidden_dim),
            dim = -1
        )
        output = attention_weight @ V
        return output

### add some detail

class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim: int = 768, dropout_rate = 0.1, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim*3)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, attention_mask=None):

        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)

        # (batch, seq, seq)
        attention_weight = Q @ K.transpose(-1,-2)/math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, 
                float("-1e20")
            )
        attention_weight = torch.softmax(attention_weight, dim=-1)

        attention_weight = self.attention_dropout(attention_weight)
        attention_result = attention_weight @ V

        #
        output = self.output_proj(attention_result)
        return output

X = torch.rand(3,4,2)
mask = torch.tensor(
    [
        [1,1,1,0],
        [1,1,0,0],
        [1,0,0,0]
    ]
)
mask.unsqueeze(dim=1).repeat(1,4,1)

net = SelfAttentionV3(2)
net(X, mask)

### add some detail

class SelfAttentionInterview(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self,X, attention_mask=None):
        # X shape: (batch, seq, dim)
        Q = self.query(X)
        K = self.query(X)
        V = self.value(X)

        attention_weight = Q @ K.transpose(-1,-2) / math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.attention_dropout(attention_weight)
        output = attention_weight @ V
        return output

class MultiHeadSelfAttentionFormal(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout = 0.1):
        super.__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num # head_num * heed_dim = hidden_dim

        self.query = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)
        self.key = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)
        self.value = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)

        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, X, attention_mask=None):
        # X (b, s, h)

        batch, seq_len, _ = X.size()

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # (b, s, h) => (b, head_num, s, head_dim)
        q_state = Q.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k_state = K.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v_state = V.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)

        # (b, h_n, s, s)
        attention_weight = torch.matmul(
            q_state, k_state.transpose(-1,-2) # (b, h_n, s, h_d) -> (b, head_num, head_dim, s)
        ) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float('-inf')
            )
        print(attention_weight.shape)

        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.attention_dropout(attention_weight)
        output_mid = attention_weight @ v_state # (b, head_num, s, head_dim)

        output_mid = output_mid.transpose(1,2).contiguous()
        output_mid = output_mid.view(batch, seq_len, -1)
        
        output = self.output_proj(output_mid)
        return output
    

class SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout_rate = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num
        self.head_num = head_num

        # layer (mha, ffn)

        # mha
        self.query = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)
        self.key = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)
        self.value = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)  
        self.output_proj = nn.Linear(hidden_dim, hidden_dim) # (hidden_dim, head_dim * head_num)
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.att_ln = nn.LayerNorm(hidden_dim, eps=0.0000001)

        # ffn (升维度 -> 降维度 -> LN)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4) 
        self.down_proj = nn.Linear(hidden_dim *4, hidden_dim)
        self.act_fn = nn.GELU() # ReLU
        self.drop_fnn = nn.Dropout(0.1)
        self.ffn_ln = nn.LayerNorm(hidden_dim, eps=0.0000001)

    def attention_layer(self, query, key, value, attention_mask=None):
        key = key.transpose(2,3)
        attention_weight = torch.matmul(query, key) / math.sqrt(self.head_dim)

        # 自带的下三角矩阵以及 attention_mask

        if attention_mask is not None:
            attention_mask = attention_mask.tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-inf")
            )
        else:
            attention_mask = torch.ones_like(
                attention_weight
            ).tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-inf")
            )
        
        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.drop_att(attention_weight)

        mid_output = attention_weight @ value
        mid_output = mid_output.transpose(1,2).contiguous()

        batch, seq, _, _= mid_output.size() 
        mid_output = mid_output.view(batch, seq, -1)

        output = self.output_proj(mid_output)

        return output

    def mha(self, X, mask=None):

        batch, seq_len, _ = X.size()

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        q_state = Q.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k_state = K.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v_state = V.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)

        output = self.attention_layer(q_state, k_state, v_state,mask)

        # post norm (b, s, h)
        return self.att_ln(X + output)
    
    def ffn(self, X):
        up = self.up_proj(X)
        up = self.act_fn(up)
        down = self.down_proj(up)
        down = self.drop_fnn(down)
        return self.ffn_ln(X + down)

    def forward(self, X, attention_mask=None):
        
        X = self.mha(X, attention_mask)
        X = self.ffn(X)
        return X

import torch.nn.functional as F

class LinearLoRALayer(nn.Module):
    def __int__(self, in_features, out_features, rank, lora_alpha, dropout, merge=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout = dropout
        self.merge = merge

        self.linear = nn.Linear(in_features, out_features)

        if rank>0:
            self.lora_a = nn.Parameter(torch.randn(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

            self.lora_b =  nn.Parameter(torch.randn(rank, in_features))
            self.scale = lora_alpha / rank

        self.dropout == nn.Dropout(dropout) if dropout>0 else nn.Identity()

        if merge:
            self.merge_weight()

    def merge_weight(self, ):
        if self.merge and self.rank>0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)

    def unmerge_weight(self, ):
        if self.merge and self.rank>0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)

    def forward(self, X):
        # X shape (batch, seq, infeature)
        if self.rank>0:
            output_part1 = self.linear(X)
            output_part2 = self.scale * (X @ (self.lora_a @ self.lora_b).T)
            output = output_part1 + output_part2
        else:
            output = self.linear(X)

        output = self.dropout(output)
        return output