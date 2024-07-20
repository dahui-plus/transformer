from torch import nn, Tensor


from .embedding import PositionalEmbedding
from .attention import CausalSelfAttention, CrossAttention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate=0.1) -> None:
    super(DecoderLayer, self).__init__()
    self.attention = CausalSelfAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
    self.cross_attention = CrossAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
    self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)
  
  def forward(self, x: Tensor, context: Tensor, key_padding_mask: Tensor) -> Tensor:
    x = self.attention(x, key_padding_mask)
    x = self.cross_attention(x, context)
    
    self.last_atten_scores = self.cross_attention.last_attention_scores
    
    x = self.ffn(x)
    return x


class Decoder(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
               dff: int, padding_idx: int=0, dropout_rate=0.1) -> None:
    super(Decoder, self).__init__()
    self.pos_embedding = PositionalEmbedding(num_embedding=vocab_size,
                                             embedding_dim=d_model,
                                             padding_idx=padding_idx)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.dec_layers = nn.ModuleList([
      DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
      for _ in range(num_layers)
    ])
    
  def forward(self, x: Tensor, context: Tensor, key_padding_mask: Tensor) -> Tensor:
    x = self.pos_embedding(x)
    x = self.dropout(x)
    
    for layer in self.dec_layers:
      x = layer(x, context, key_padding_mask)
    
    self.last_attention_scores = layer.cross_attention.last_attention_scores
    
    return x


if __name__ == '__main__':
  import torch
  
  sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
  a = torch.rand(size=(64, 128, 512))
  b = torch.rand(size=(64, 128, 512))

  sample_decoder_layer_output = sample_decoder_layer(x=a, context=b)

  print(a.shape)
  print(b.shape)
  print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`

  # Instantiate the decoder.
  sample_decoder = Decoder(num_layers=4,
                          d_model=512,
                          num_heads=8,
                          dff=2048,
                          vocab_size=8000)

  a = torch.ones(size=(64, 128), dtype=torch.long)
  output = sample_decoder(x=a, context=sample_decoder_layer_output)

  # Print the shapes.
  print(a.shape)
  print(sample_decoder_layer_output.shape)
  print(output.shape)

