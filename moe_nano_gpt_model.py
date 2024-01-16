import torch
import torch.nn as nn
import torch.nn.functional as F


# model architecture
class AttentionHead(nn.Module):
  """a single head of self attention"""
  
  def __init__(self, n_embed, head_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B, T, C = x.shape
    K = self.key(x) # (B, T, C)
    Q = self.query(x) # (B, T, C)
    
    wei = Q @ K.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, H, C) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    V = self.value(x) # (B, T, C)
    out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  
class MultiHeadAttention(nn.Module):
  """a multi-head self attention layer"""
  
  def __init__(self, n_embed, n_heads, head_size, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
    self.fc = nn.Linear(head_size * n_heads, n_embed)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads*C)    
    out = self.fc(out) # (B, T, C)
    out = self.dropout(out) 
    return out
  
class Expert(nn.Module):
  def __init__(self, n_embed, n_hidden, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_embed),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
   return self.net(x)
 
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class NoisyTopKGating(nn.Module):
  def __init__(self, n_embed, n_experts, top_k):
    super().__init__()
    self.n_experts = n_experts
    self.top_k = top_k
    self.Wg = nn.Linear(n_embed, n_experts)
    self.Wnoise = nn.Linear(n_embed, n_experts)
    
  def forward(self, x):
    gate = self.Wg(x)
    noise = self.Wnoise(x)
    h = gate + torch.randn_like(noise) * F.softplus(noise)
    
    # print(h.shape)
    
    topk_vals, topk_idxs = torch.topk(h, self.top_k, dim=-1)
    g = torch.full_like(h, float('-inf'))
    g[torch.arange(g.shape[0])[:, None, None], torch.arange(g.shape[1])[:, None], topk_idxs] = topk_vals
    g = F.softmax(g, dim=-1)
    
    return g
  
class MoE(nn.Module):
  def __init__(self, n_embed, n_experts, n_hidden, top_k, dropout):
    super().__init__()
    self.gating = NoisyTopKGating(n_embed, n_experts, top_k)
    self.experts = nn.ModuleList([Expert(n_embed, n_hidden, dropout) for _ in range(n_experts)])
    
  def forward(self, x):
    scores = self.gating(x) # (B, T, n_experts)
    mask = scores > 0
    out = torch.zeros_like(x)
        
    for i, expert in enumerate(self.experts):
      expert_mask = mask[:, :, i]
      inputs_for_expert = x[expert_mask]
      
      if inputs_for_expert.numel() == 0:
        continue
      
      expert_out = expert(inputs_for_expert)
      gating_scores = scores[:, :, i][expert_mask].view(-1, 1)
      out[expert_mask] += expert_out * gating_scores

    return out

class Block(nn.Module):
  def __init__(self, n_embed, n_heads, n_experts, top_k, block_size, dropout):
    super().__init__()
    self.sa_heads = MultiHeadAttention(n_embed, n_heads, n_embed // n_heads, block_size, dropout)
    self.moe = MoE(n_embed, n_embed*4, n_experts, top_k, dropout)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
    
    
  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x)) #  [batch_size, block_size, n_embed]
    x = x + self.moe(self.ln2(x)) # [batch_size, block_size, n_embed]
    return x

class NanoGPTMoE(nn.Module):
  def __init__(self, hyperparameters, device="cpu"):
    super().__init__()
    
    # hyperparameters
    vocab_size = hyperparameters['vocab_size']
    block_size = hyperparameters['block_size']
    n_embed = hyperparameters['n_embed']
    n_heads = hyperparameters['n_heads']
    n_layers = hyperparameters['n_layers']
    dropout = hyperparameters['dropout']
    n_experts = hyperparameters['n_experts']
    top_k = hyperparameters['top_k']
    
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_heads, n_experts, top_k, block_size, dropout) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
    
    self.device = device
    self.block_size = block_size
      
  def forward(self, idx, targets=None):
    # idx and target are both [batch_size, block_size]
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, n_embed]
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # [block_size, n_embed]
    x = tok_emb + pos_emb # [batch_size, block_size, n_embed]
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # [batch_size, block_size, vocab_size]
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
    # return 0, 0
      
  def generate(self, idx, max_new_tokens=100):
    # idx is (B, T)
    for _ in range(max_new_tokens):
      # get the last block_size tokens
      idx_cond = idx[:, -self.block_size:] # (B, T)
      # get the predictions
      logits, _ = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      
    return idx
