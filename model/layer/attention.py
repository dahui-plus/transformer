from torch import nn, Tensor, add
from typing import Any
import torch


class BaseAttention(nn.Module):
  def __init__(self, **kwargs) -> None:
    super(BaseAttention, self).__init__()
    self.mha = nn.MultiheadAttention(batch_first=True, **kwargs)
    self.layernorm = nn.LayerNorm(normalized_shape=kwargs.get('embed_dim'))
    self.add = add


class CrossAttention(BaseAttention):
  def forward(self, x: Tensor, context: Tensor) -> Tensor:
    atten_output, atten_scores = self.mha(query=x, key=context, value=context)
    self.last_attention_scores = atten_scores
    
    x = self.add(x, atten_output)
    x = self.layernorm(x)
    return x


class GlobalSelfAttention(BaseAttention):
  def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
    atten_output, atten_scores = self.mha(query=x, key=x, value=x, key_padding_mask=key_padding_mask)
    x = self.add(x, atten_output)
    x = self.layernorm(x)
    return x


class CausalSelfAttention(BaseAttention):
  def forward(self, x: Tensor, key_padding_mask: Tensor):
    atten_output, atten_scores = self.mha(query=x, key=x, value=x, key_padding_mask=key_padding_mask, is_causal=True,
                                          attn_mask=nn.Transformer.generate_square_subsequent_mask(sz=x.shape[1], device=x.device))
    x = self.add(x, atten_output)
    x = self.layernorm(x)
    return x


if __name__ == '__main__':
  import torch
  
  sample_ca = CrossAttention(embed_dim=128, num_heads=1)
  sample_gsa = GlobalSelfAttention(embed_dim=128, num_heads=2)
  sample_csa = CausalSelfAttention(embed_dim=128, num_heads=2)
  
  a = torch.rand(size=(64, 128))
  b = torch.rand(size=(64, 128))

  print(a.shape)
  print(b.shape)
  print(sample_ca(a, b).shape)
  
  print(a.shape)
  print(sample_gsa(a).shape)
  
  print(b.shape)
  print(sample_csa(b).shape)
