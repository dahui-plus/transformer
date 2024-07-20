from torch import nn, Tensor

from .layer import Encoder, Decoder

class Transformer(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, num_layers: int,
               num_heads: int, dff: int, target_vocab_size,
               padding_idx: int, dropout_rate=0.1) -> None:
    super(Transformer, self).__init__()
    self.encoder = Encoder(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                           num_heads=num_heads, dff=dff, padding_idx=padding_idx, dropout_rate=dropout_rate)
    self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                           num_heads=num_heads, dff=dff, padding_idx=padding_idx, dropout_rate=dropout_rate)
    
    self.linear = nn.Linear(in_features=d_model, out_features=target_vocab_size)
    
  def forward(self, x: Tensor, context: Tensor, source_padding_mask: Tensor, target_padding_mask: Tensor) -> Tensor:
    x = self.encoder(x, source_padding_mask)
    x = self.decoder(context, x, target_padding_mask)
    x = self.linear(x)
    return x

