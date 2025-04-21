from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3 projections: key, query, value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) Questo serve per trattare il numero di head in parallelo come batch dimension
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Operazione eseguita in parallelo per ogni batch (ovvero per ogni head)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # causal = we consider only past and current tokens attention weights
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Until here we have the attention output for all heads, now we need to concatenate the heads and project the result

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side -> concatenate multihead results (C = nh * hs)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU(approximate="tanh") # GELU is a smoother version of ReLU, no flat region, no dying neurons (no updates)
    def forward(self, x):
        # 2 linear projections with a GELU activation in between
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        # seguiamo lo schema del paper originale "Attention is All You Need"
        x = x + self.attn(self.ln1(x)) #aggregation function, all the tokens communicate -> reduce operation like pooling
        x = x + self.mlp(self.ln2(x)) # every single token individually, no information collected between the tokens
        return x

@dataclass
class GPT2Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2, self).__init__()
        self.config = config
        # definiamo i blocchi che ci servono
        self.transformer = nn.ModuleDict(dict( #il main module è il modulo transformer rappresentato come dizionario, con i sottomoduli accessibili con le chiavi
            wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #position embedding
            #we have 12 layers of transformer blocks, so we have a list accessible by integers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #block still to be defined in a recursive way
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classifier to predict the next word in the sequence

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss