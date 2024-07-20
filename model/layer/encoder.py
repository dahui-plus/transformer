from torch import nn, Tensor

from .attention import GlobalSelfAttention
from .feed_forward import FeedForward
from .embedding import PositionalEmbedding

class EncoderLayer(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate=.1) -> None:
    super(EncoderLayer, self).__init__()
    self.attention = GlobalSelfAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
    self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)
  
  def forward(self, x: Tensor, source_padding_mask: Tensor) -> Tensor:
    x = self.attention(x, source_padding_mask)
    x = self.ffn(x)
    return x
  
  
class Encoder(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, num_layers, num_heads: int,
               dff: int, padding_idx=0, dropout_rate=.1) -> None:
    super(Encoder, self).__init__()
    
    self.pos_embedding = PositionalEmbedding(num_embedding=vocab_size,
                                             embedding_dim=d_model,
                                             padding_idx=padding_idx)
    
    self.enc_layers = nn.ModuleList([
      EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
      for _ in range(num_layers)
    ])
    
    self.dropout = nn.Dropout(p=dropout_rate)
    
  def forward(self, x: Tensor, source_padding_mask: Tensor) -> Tensor:
    x = self.pos_embedding(x)
    x = self.dropout(x)
    
    for layer in self.enc_layers:
      x = layer(x, source_padding_mask)
    
    return x

if __name__ == '__main__':
  import torch
  
  # Instantiate the encoder.
  sample_encoder = Encoder(num_layers=4,
                          d_model=512,
                          num_heads=8,
                          dff=2048,
                          vocab_size=8500)
  sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
  a = torch.rand(size=(64, 512))

  print(a.shape)
  print(sample_encoder_layer(a).shape)
  

  sample_encoder_output = sample_encoder(a.to(dtype=torch.long))

  # Print the shape.
  print(a.shape)
  print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.

