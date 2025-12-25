import torch.nn as nn
import torch
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
    keys = self.W_key(x)
    query =  self.W_query(x)
    values =  self.W_value(x)

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
    contxt_vec = contxt_vec.contiguous().view(b, num_tokens, self.d_out)
    contxt_vec = self.out_proj(contxt_vec)
    return contxt_vec


GPT_CONFIG_124M = {
    "vocab_size":50257,
    "context_length": 1024,
    "emb_dim":768,
    "n_heads": 12,
    "n_layers" : 12,
    "drop_rate": 0.1,
    "qkv_bias" : False,
}

class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps= 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
  def forward(self,x):
    mean=x.mean(dim=-1, keepdim=True)
    var=x.var(dim=-1,keepdim=True, unbiased=False)
    norm_x=(x-mean)/torch.sqrt(var+self.eps)
    return self.scale*norm_x + self.shift


class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return 0.5 * x * (1+torch.tanh(
        torch.sqrt(torch.tensor(2.0/torch.pi))*
                      (x+0.0044715 * torch.pow(x,3))
    ))


class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], cfg["emb_dim"]*4),
        GELU(),
        nn.Linear(cfg["emb_dim"]*4, cfg["emb_dim"])
    )
  def forward(self,x):
    return self.layers(x)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.attn = MultiHeadAttension(
      d_in=cfg["emb_dim"],
      d_out=cfg["emb_dim"],
      context_length=cfg["emb_dim"],
      dropout=cfg["drop_rate"],
      qkv_bias=cfg["qkv_bias"],
      num_heads=cfg["num_heads"]
    )
    self.ff = FeedForward(cfg)
    self.norm1=LayerNorm(cfg["emb_dim"])
    self.norm2=LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


  def forward(self,x):

    shortcut=x
    x=self.norm1(x)
    x=self.attn(x)
    x=self.drop_shortcut(x)
    x=x+shortcut

    shortcut=x
    x=self.norm2(x)
    x=self.ff(x)
    x=self.drop_shortcut(x)
    x=x+shortcut
    return x

class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb=nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb=nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.trf_blocks= nn.Sequential(
        * [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )

    self.final_norm = LayerNorm(cfg["emb_dim"])
    self.out_head=nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch_size, seq_len= in_idx.shape
    tok_embeds=self.tok_emb(in_idx)

    pos_embeds=self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
    )
    x=tok_embeds + pos_embeds
    x=self.drop_emb(x)
    x=self.trf_blocks(x)
    x=self.final_norm(x)
    logits=self.out_head(x)
    return logits



def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits=model(idx_cond)


    logits = logits[:, -1, :]
    probabs = torch.softmax(logits, dim=-1)
    idx_next=torch.argmax(probabs,dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
  return idx
