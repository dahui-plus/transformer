from torch import nn, Tensor, add


class FeedForward(nn.Module):
  def __init__(self, d_model: int, dff: int, dropout_rate=0.1) -> None:
    super().__init__()
    self.seq = nn.ModuleList([
      nn.Linear(in_features=d_model, out_features=dff),
      nn.ReLU(),
      nn.Linear(in_features=dff, out_features=d_model),
      nn.Dropout(dropout_rate)
    ])
    self.add = add
    self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
  def forward(self, x: Tensor) -> Tensor:
    output = x
    for layer in self.seq:
      output = layer(output)

    x = self.add(x, output)
    x = self.layer_norm(x)
    return x

if __name__ == '__main__':
  import torch
  
  sample_ffn = FeedForward(512, 2048)
  a = torch.rand(size=(64, 512))

  print(a.shape)
  print(sample_ffn(a).shape)

