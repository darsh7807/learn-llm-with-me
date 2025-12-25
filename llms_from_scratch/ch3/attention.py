import torch

inputs = torch.tensor([
    [0.43,   0.15, 0.89],
    [0.43,   0.15, 0.89],
    [0.43,   0.15, 0.89],
    [0.43,   0.15, 0.89],
    [0.43,   0.15, 0.89],
    [0.43,   0.15, 0.89],
])

query = inputs[1]
atten_score_2=torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
  atten_score_2[i] = torch.dot(x_i, query).item()

print(atten_score_2)

attn_weights_2_tmp = atten_score_2 / atten_score_2.sum()
print(attn_weights_2_tmp.sum()

      )

import torch.nn as nn
class SimpleAttension_v1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.W_query = nn.Parameter(torch.rand(d_in, d_out))
    self.W_key= nn.Parameter(torch.rand(d_in, d_out))
    self.W_value= nn.Parameter(torch.rand(d_in, d_out))

  def forward(self,x):
    keys = x @ self.W_key
    query = x @ self.W_query

    values = x @ self.W_values

    attn_scores = query @ keys.T
    attn_weights = torch.softmax(attn_scores/ keys.shapre[-1] ** 0.5, dim=-1)
    contxt_vec = attn_weights @ values
    return contxt_vec



import torch.nn as nn
class SimpleAttension_v1(nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    self.W_query = nn.Layer(torch.rand(d_in, d_out, bias=qkv_bias))
    self.W_key= nn.Layer(torch.rand(d_in, d_out,bias=qkv_bias))
    self.W_value= nn.Layer(torch.rand(d_in, d_out, bias=qkv_bias))

  def forward(self,x):
    keys = x @ self.W_key
    query = x @ self.W_query

    values = x @ self.W_values

    attn_scores = query @ keys.T
    attn_weights = torch.softmax(attn_scores/ keys.shape[-1] ** 0.5, dim=-1)
    contxt_vec = attn_weights @ values
    return contxt_vec



import torch.nn as nn
class CasualAttension(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
    super().__init__()
    self.W_query = nn.Layer(torch.rand(d_in, d_out, bias=qkv_bias))
    self.W_key= nn.Layer(torch.rand(d_in, d_out,bias=qkv_bias))
    self.W_value= nn.Layer(torch.rand(d_in, d_out, bias=qkv_bias))

    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length),
        diagonal =1)
    )

  def forward(self,x):
    b, num_tokens, d_in = x.shape #batch, no of token in context, dim of each token
    keys = x @ self.W_key
    query = x @ self.W_query
    values = x @ self.W_values

    attn_scores = query @ keys.transport(1,2)
    attn_scores.masked_fill(
        self.mask.bool() [:num_tokens, :num_tokens], -torch.inf
    )
    attn_weights = torch.softmax(attn_scores/ keys.shape[-1] ** 0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    contxt_vec = attn_weights @ values
    return contxt_vec





class MultiHeadAttensionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length,
               dropout, num_heads,qkv_bias=False):
    super()._init__()
    self.heads = nn.ModuleList(
        [
            CasualAttension(d_in, d_out, context_length, dropout, qkv_bias)
        ] for _ in range(num_heads)
    )

def forward(self, x):
  return torch.cat([head(x) for head in self.heads], dim=-1)





class MultiHeadAttension(nn.Module):
  def __init__(self, d_in, d_out,
               context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert(d_out % num_heads ==0)
    self.d_in = d_in
    self.d_out = d_out
    self.context_length = context_length
    self.dropout = nn.Dropout(dropout)
    self.num_heads = num_heads
    self.head_dim = d_out//num_heads
    self.W_query = nn.Linear(d_in, d_out, qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length),
        diagonal =1)
    )

  def forward(self, x):
    b, num_tokens, d_in = x.shape #batch, no of token in context, dim of each token
    keys = x @ self.W_key
    query = x @ self.W_query
    values = x @ self.W_values

    keys= keys.view(b, num_tokens, self.num_heads, self.head_dim)
    query= query.view(b, num_tokens, self.num_heads, self.head_dim)
    values= values.view(b, num_tokens, self.num_heads, self.head_dim)

    keys= keys.transpose(1,2)
    query= query.transpose(1,2)
    values= values.transpose(1,2)

    attn_scores = query @ keys.transpose(2,3)
    attn_scores.masked_fill(
        self.mask.bool() [:num_tokens, :num_tokens], -torch.inf
    )
    attn_weights = torch.softmax(attn_scores/ keys.shape[-1] ** 0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    contxt_vec = (attn_weights @ values).transpose(1,2)
    contxt_vec = contxt_vec.contigous().view(b, num_tokens, self.d_out)
    contxt_vec = self.out_proj(contxt_vec)
    return contxt_vec

