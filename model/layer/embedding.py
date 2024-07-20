from torch import nn, Tensor, arange, concat, sin, cos, sqrt, tensor


def positional_encoding(length: int, depth: int) -> Tensor:
  depth = depth / 2
  positions = arange(0, length).reshape(-1, 1)
  depths = arange(0, depth).reshape(1, -1) / depth
  
  angle_rates = 1 / (10000 ** depths)
  angle_rads = positions * angle_rates
  
  pos_encoding = concat([sin(angle_rads), cos(angle_rads)], dim=-1)
  return pos_encoding


class PositionalEmbedding(nn.Module):
  def __init__(self, num_embedding: int, embedding_dim: int, padding_idx: int=0) -> None:
    super(PositionalEmbedding, self).__init__()
    self.d_model = tensor(embedding_dim)
    self.embedding = nn.Embedding(num_embeddings=num_embedding,
                                  embedding_dim=embedding_dim,
                                  padding_idx=padding_idx)
    self.pos_encoding = positional_encoding(length=1024, depth=embedding_dim)
    
  def forward(self, x: Tensor) -> Tensor:
    self.pos_encoding = self.pos_encoding.to(device=x.device)
    length = x.shape[1]
    x = self.embedding(x)
    x *= sqrt(self.d_model.to(dtype=x.dtype))
    x = x + self.pos_encoding[:length, :]
    return x

if __name__ == '__main__':

  import matplotlib.pyplot as plt
  from torch import norm, einsum
  import torch
  # pos_encoding = positional_encoding(length=2048, depth=512)

  # # Check the shape.
  # print(pos_encoding.shape)

  # # Plot the dimensions.
  # plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
  # plt.ylabel('Depth')
  # plt.xlabel('Position')
  # plt.colorbar()
  # plt.show()

  # pos_encoding/=norm(pos_encoding, dim=1, keepdim=True)
  # p = pos_encoding[1000]
  # dots = einsum('pd,d -> p', pos_encoding, p)
  # plt.subplot(2,1,1)
  # plt.plot(dots)
  # plt.ylim([0,1])
  # plt.plot([950, 950, float('nan'), 1050, 1050],
  #         [0,1,float('nan'),0,1], color='k', label='Zoom')
  # plt.legend()
  # plt.subplot(2,1,2)
  # plt.plot(dots)
  # plt.xlim([950, 1050])
  # plt.ylim([0,1])

  embed_pt = PositionalEmbedding(num_embedding=2048, embedding_dim=512)
  embed_en = PositionalEmbedding(num_embedding=2048, embedding_dim=512)
  pt = torch.ones(size=(64, 512), dtype=torch.long)
  en = torch.ones(size=(64, 512), dtype=torch.long)

  pt_emb = embed_pt(pt)
  print(pt_emb.shape)
  en_emb = embed_en(en)
  print(en_emb.shape)
